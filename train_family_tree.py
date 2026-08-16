"""Trainer for the family-tree task (non-commutative relational structure).

Objective is the paper's own: next-token CE at REVISITED nodes. Getting it right
means recognising you have returned to a specific person, which on a family tree
cannot be done by summing translations.

Effective floor is NOT chance. The gates measure a hub-node baseline of 0.163
against a chance of 0.125, because shallow nodes are revisited more often. Report
against 0.163.

MapEM-NC path integration is a SEQUENTIAL matrix product (paper B.2.2), so it is
slow and sequences are short by necessity -- the paper trained at length 16 and
tested at 32.
"""
import argparse, json, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from mapformer.environment_family_tree import FamilyTreeWorld
from mapformer.train_variant import VARIANT_MAP

_REPO = Path(__file__).resolve().parent

def _loss(model, toks, rev, dev):
    toks = toks.to(dev); rev = rev.to(dev)
    logits = model(toks[:, :-1]); tgt = toks[:, 1:]; m = rev[:, 1:]
    return (F.cross_entropy(logits[m], tgt[m]) if m.any() else logits.sum()*0.0)

@torch.no_grad()
def evaluate(model, env, n_steps, n_b, bs, dev, seed):
    model.eval(); rng = np.random.RandomState(seed); ok = tot = 0
    for _ in range(n_b):
        toks, rev, _ = env.generate_batch(bs, n_steps, rng)
        lg = model(toks[:, :-1].to(dev)); tgt = toks[:, 1:].to(dev); m = rev[:, 1:].to(dev)
        if m.any():
            ok += (lg[m].argmax(-1) == tgt[m]).sum().item(); tot += int(m.sum())
    model.train(); return ok / max(tot, 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=100); ap.add_argument("--n-batches", type=int, default=48)
    ap.add_argument("--batch-size", type=int, default=16); ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--n-obs", type=int, default=8); ap.add_argument("--n-steps", type=int, default=64)
    ap.add_argument("--eval-steps", nargs="+", type=int, default=[64, 128])
    ap.add_argument("--n-layers", type=int, default=2); ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-heads", type=int, default=2); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--output-dir", required=True)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed); dev = torch.device(a.device)
    env = FamilyTreeWorld(depth=a.depth, n_obs_types=a.n_obs, seed=a.seed)
    env_t = FamilyTreeWorld(depth=a.depth, n_obs_types=a.n_obs, seed=10000)
    model = VARIANT_MAP[a.variant](vocab_size=env.unified_vocab_size, d_model=a.d_model,
                                   n_heads=a.n_heads, n_layers=a.n_layers, grid_size=64).to(dev)
    print(f"{a.variant} seed={a.seed} params={sum(p.numel() for p in model.parameters()):,} "
          f"chance={1/a.n_obs:.4f} hub_floor=0.163", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0., 1 - s/(a.epochs*a.n_batches)))
    rng = np.random.RandomState(a.seed); losses = []
    for ep in range(a.epochs):
        t0 = time.time(); acc = 0.
        for _ in range(a.n_batches):
            toks, rev, _ = env.generate_batch(a.batch_size, a.n_steps, rng)
            L = _loss(model, toks, rev, dev)
            opt.zero_grad(); L.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
            acc += L.item()
        losses.append(acc/a.n_batches)
        if (ep+1) % 10 == 0 or ep == 0:
            print(f"  epoch {ep+1}/{a.epochs} loss={losses[-1]:.4f} ({time.time()-t0:.0f}s)", flush=True)
    res = {}
    for T in a.eval_steps:
        acc = evaluate(model, env_t, T, 8, 8, dev, 7000+a.seed)
        res[T] = acc; print(f"  [held-out] n_steps={T}: acc={acc:.4f}", flush=True)
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "losses": losses, "variant": a.variant,
                "seed": a.seed, "results": res, "vocab_size": env.unified_vocab_size,
                "d_model": a.d_model, "n_heads": a.n_heads, "n_layers": a.n_layers},
               out/f"{a.variant}_familytree.pt")
    json.dump(res, open(out/f"{a.variant}_familytree.json","w"), indent=2)
    print(f"DONE {a.variant} final_loss={losses[-1]:.4f}", flush=True)

if __name__ == "__main__":
    main()
