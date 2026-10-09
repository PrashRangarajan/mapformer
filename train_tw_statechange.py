"""Trainer for state-change clauses in the text world (TW_STATECHANGE_PREREG.md). The recipe, eval stream and held-out
map are train_tw_normstep's (T = 1024 words, batch 16, 900 epochs x 98 batches, lr 1e-3, warmup + cosine, 1 layer,
d 128, 2 heads, r = 4 shared, --data-workers 3; eval np seed 10**6, held-out map env seed 10000, T and 2T). The task is
environment_tw_statechange.StateChangeWorld (map seed = --seed). Arms:
  MapWM     Vanilla_r4 (learned raw step: Delta = W_out W_in e)
  NormStep  model_codes.MapWM_NormStep (Delta = W_out W_in LN(e))
  DirOnly   model_textstep.MapWM_DirOnly (steps only on the 12 direction words: every state clause exactly off the map)
  RoPE      index RoPE (canonical base 10000), same layer
MapWM / NormStep / DirOnly share every base weight at a seed (the extra modules draw no random numbers).
With --p-take 0 --p-drop 0 --no-state-vocab the data stream and model are train_tw_normstep's (reproduction check).
Writes <arm>.pt and eval.json in --output-dir."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mapformer.environment_tw_statechange import StateChangeWorld
from mapformer.model_codes import MapWM_NormStep
from mapformer.model_textstep import MapWM_DirOnly
from mapformer.train import train
from mapformer.train_textworld import evaluate
from mapformer.train_variant import VARIANT_MAP

ARMS = {"MapWM": VARIANT_MAP["Vanilla_r4"], "NormStep": MapWM_NormStep, "DirOnly": MapWM_DirOnly,
        "RoPE": VARIANT_MAP["RoPE"]}
PATH_ARMS = ("MapWM", "NormStep", "DirOnly")
EVAL_SEED, HELDOUT = 10**6, 10000


def make_env(seed, size=64, p_take=0.4, p_drop=0.4, state_vocab=True):
    return StateChangeWorld(size=size, seed=seed, p_take=p_take, p_drop=p_drop, state_vocab=state_vocab)


def build(arm, env, n_layers=1, size=64):
    m = ARMS[arm](vocab_size=env.unified_vocab_size, d_model=128, n_heads=2, n_layers=n_layers, grid_size=size)
    if arm == "DirOnly":
        m.set_step_ids([i for a in range(4) for i in env.dir_ids[a]])
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=900)
    ap.add_argument("--n-batches", type=int, default=98)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-steps", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--data-workers", type=int, default=3)
    ap.add_argument("--n-trials", type=int, default=200)
    ap.add_argument("--p-take", type=float, default=0.4)
    ap.add_argument("--p-drop", type=float, default=0.4)
    ap.add_argument("--no-state-vocab", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output-dir", required=True)
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    env = make_env(a.seed, a.size, a.p_take, a.p_drop, not a.no_state_vocab)
    model = build(a.arm, env, a.n_layers, a.size)
    assert len(model.layers) == a.n_layers, (len(model.layers), a.n_layers)   # rule 17
    print(f"{a.arm} L{a.n_layers} seed={a.seed} vocab={env.unified_vocab_size} p_take={a.p_take} p_drop={a.p_drop} "
          f"params={sum(p.numel() for p in model.parameters()):,}")
    losses = train(model, env, n_epochs=a.epochs, lr=a.lr, batch_size=a.batch_size, n_steps=a.n_steps,
                   n_batches=a.n_batches, device=a.device, schedule="cosine", data_workers=a.data_workers)
    torch.save({"model_state_dict": model.state_dict(), "losses": losses, "arm": a.arm, "seed": a.seed,
                "config": {"args": dict(vars(a)), "vocab_size": env.unified_vocab_size}}, out / f"{a.arm}.pt")
    te = make_env(HELDOUT, a.size, a.p_take, a.p_drop, not a.no_state_vocab)
    ev = {}
    for T in (a.n_steps, 2 * a.n_steps):
        acc, nll = evaluate(model, te, T, a.n_trials, a.device, seed=EVAL_SEED)
        ev[str(T)] = {"acc": acc, "nll": nll}
        print(f"held-out map T={T}: acc {acc:.4f} nll {nll:.4f}")
    json.dump({"eval": ev, "final_loss": float(np.mean(losses[-max(1, len(losses) // 20):]))},
              open(out / "eval.json", "w"), indent=2)


if __name__ == "__main__":
    main()
