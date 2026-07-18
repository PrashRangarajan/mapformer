"""
Aggregate-query task: predict the windowed-MAJORITY obs-type, not the exact
obs at a revisited cell. This flips the readout from a needle question
(retrieval, which favors full-resolution flat attention) to a haystack
question (aggregation, which a pooled/hierarchical representation computes
directly).

Controlled 2x2 with the retrieval task (HIER_ATTN_LONGT.md): same environment,
same architectures (flat Level15 vs HierAttn), only the target changes. If the
winner flips (flat wins retrieval, HierAttn wins aggregation), hierarchy's
value is task-determined, not architectural.

Target at obs-step s = the majority obs-type over the last W_agg steps
(argmax of the windowed obs-type histogram). Predicted from the model's output
at the obs-token position 2s+1 (causal: that position has seen o_1..o_s).
Masked to steps whose window contains at least one real observation.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mapformer.environment import GridWorld
from mapformer.train_variant import VARIANT_MAP

OBS_OFFSET = 4
N_OBS = 16
BLANK = 20


def agg_targets(tokens: torch.Tensor, W: int):
    """tokens (B,2T) -> (target_token (B,T), mask (B,T)) windowed-majority obs-type."""
    B, L = tokens.shape; T = L // 2
    obs = tokens[:, 1::2]
    valid = (obs >= OBS_OFFSET) & (obs < OBS_OFFSET + N_OBS)
    otype = (obs - OBS_OFFSET).clamp(0, N_OBS - 1)
    oh = torch.zeros(B, T, N_OBS, device=tokens.device)
    oh.scatter_(2, otype.unsqueeze(-1), valid.float().unsqueeze(-1))
    pref = oh.cumsum(1)
    prefW = torch.zeros_like(pref); prefW[:, W:] = pref[:, :T - W]
    cnt = pref - prefW
    total = cnt.sum(-1)
    maj = cnt.argmax(-1)
    tgt = torch.where(total > 0, OBS_OFFSET + maj, torch.full_like(maj, BLANK))
    return tgt, (total > 0)


def gen_batch(env, B, T, device):
    toks = [env.generate_trajectory(T)[0] for _ in range(B)]
    return torch.stack(toks).to(device)


def run_eval(model, env, T, n, W, device):
    model.eval(); c = tot = 0
    with torch.no_grad():
        for _ in range(n):
            tokens = gen_batch(env, 1, T, device)
            tgt, mask = agg_targets(tokens, W)
            logits = model(tokens)               # (1, 2T, V)
            obs_logits = logits[:, 1::2, :]      # (1, T, V) at obs positions
            pred = obs_logits.argmax(-1)
            c += (pred[mask] == tgt[mask]).sum().item(); tot += mask.sum().item()
    return c / tot if tot else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list(VARIANT_MAP.keys()))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-steps", type=int, default=256)
    ap.add_argument("--w-agg", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--n-batches", type=int, default=156)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--eval-lens", type=int, nargs="+", default=[256, 512, 1024, 2048])
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = args.device
    env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, n_landmarks=0, seed=0)
    model = VARIANT_MAP[args.variant](vocab_size=env.unified_vocab_size, d_model=128,
                                      n_heads=2, n_layers=1, grid_size=64).to(dev)
    print(f"{args.variant} aggregate task W_agg={args.w_agg} n_steps={args.n_steps} "
          f"params={sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.LinearLR(opt, 1.0, 0.0, total_iters=args.epochs * args.n_batches)
    losses = []
    for ep in range(args.epochs):
        model.train(); el = 0.0
        for _ in range(args.n_batches):
            tokens = gen_batch(env, args.batch_size, args.n_steps, dev)
            tgt, mask = agg_targets(tokens, args.w_agg)
            logits = model(tokens)
            obs_logits = logits[:, 1::2, :]
            loss = F.cross_entropy(obs_logits[mask], tgt[mask])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); el += loss.item()
        losses.append(el / args.n_batches)
        if (ep + 1) % 5 == 0:
            print(f"  Epoch {ep+1:3d}/{args.epochs} | Loss: {losses[-1]:.4f}", flush=True)

    accs = {T: run_eval(model, env, T, 100 if T <= 1024 else 30, args.w_agg, dev)
            for T in args.eval_lens}
    for T, a in accs.items():
        print(f"  aggregate acc T={T}: {a:.3f}")
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "losses": losses,
                "variant": args.variant, "agg_acc": accs, "w_agg": args.w_agg},
               out / f"{args.variant}_agg.pt")
    print(f"Saved: {out}/{args.variant}_agg.pt")


if __name__ == "__main__":
    main()
