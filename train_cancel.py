"""Trainer for H3, the cancellation knob (CANCEL_PREREG.md). A separate entry point so the running
batches' train_variant.py is not edited (rule 22); it reuses VARIANT_MAP and train() unchanged.

Trains on a 1D ring (environment_cancel.GridWorldCancel, map seed = --seed), then scores revisit
accuracy on a HELD-OUT map (env seed 10000) at --n-steps and 4x --n-steps, writing <variant>.pt
and eval.json in --output-dir."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mapformer.environment_cancel import GridWorldCancel
from mapformer.train import train
from mapformer.train_variant import VARIANT_MAP


@torch.no_grad()
def evaluate(model, env, T, n_trials, dev, seed=0):
    np.random.seed(seed); model.eval()
    ok = tot = 0; nll = 0.0
    for _ in range(n_trials):
        tok, _om, rev = env.generate_trajectory(T)
        tok = tok.unsqueeze(0).to(dev)
        lp = F.log_softmax(model(tok[:, :-1]).float(), dim=-1)
        tgt = tok[0, 1:]; m = rev[1:].to(dev)
        if m.sum() == 0:
            continue
        ok += (lp.argmax(-1)[0][m] == tgt[m]).sum().item(); tot += int(m.sum())
        nll += float(-lp[0, torch.arange(lp.shape[1], device=dev)[m], tgt[m]].sum())
    return ok / tot, nll / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["Vanilla", "RoPE"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--p-plus", type=float, required=True)
    ap.add_argument("--size", type=int, default=32)
    ap.add_argument("--n-layers", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--n-batches", type=int, default=98)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--n-steps", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--data-workers", type=int, default=3)
    ap.add_argument("--n-trials", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output-dir", required=True)
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    env = GridWorldCancel(size=a.size, seed=a.seed, p_plus=a.p_plus)
    model = VARIANT_MAP[a.variant](vocab_size=env.unified_vocab_size, d_model=128, n_heads=2,
                                   n_layers=a.n_layers, grid_size=a.size)
    assert len(model.layers) == a.n_layers, (len(model.layers), a.n_layers)   # rule 17
    print(f"{a.variant} L{a.n_layers} p_plus={a.p_plus} seed={a.seed} params={sum(p.numel() for p in model.parameters()):,}")
    losses = train(model, env, n_epochs=a.epochs, lr=a.lr, batch_size=a.batch_size, n_steps=a.n_steps,
                   n_batches=a.n_batches, device=a.device, schedule="cosine", data_workers=a.data_workers)
    torch.save({"model_state_dict": model.state_dict(), "losses": losses, "variant": a.variant,
                "seed": a.seed, "config": {"args": dict(vars(a)), "vocab_size": env.unified_vocab_size}},
               out / f"{a.variant}.pt")
    te = GridWorldCancel(size=a.size, seed=10000, p_plus=a.p_plus)
    ev = {}
    for T in (a.n_steps, 4 * a.n_steps):
        acc, nll = evaluate(model, te, T, a.n_trials, a.device)
        ev[str(T)] = {"acc": acc, "nll": nll}
        print(f"held-out map T={T}: acc {acc:.4f} nll {nll:.4f}")
    json.dump({"eval": ev, "final_loss": float(np.mean(losses[-max(1, len(losses) // 20):]))},
              open(out / "eval.json", "w"), indent=2)


if __name__ == "__main__":
    main()
