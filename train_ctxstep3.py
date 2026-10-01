"""Trainer for the context-dependent step, cue DISTANCE (CONTEXT_STEP_DESIGN.md, second revision):
the text world with decoys whose cue is near or far (environment_textworld_ctx3.TextWorldCtx3, --cue --dist; map seed = --seed), held-out-map eval (seed 10000) at
--n-steps and 2x, writing <arm>.pt and eval.json. Builds its own arm map so train_variant.py (md5-guarded
by a running batch) is not edited (rule 22).

Arms: CF context-free (Vanilla_r4), SR Selective-RoPE generator at MapFormer's placement (SRoPEGen),
CG context gate, HS hidden-state step (2 layers only), RoPE index. --n-layers applies to CF, SR, RoPE."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mapformer.environment_textworld_ctx3 import TextWorldCtx3
from mapformer.model_context_step import CtxGateWM, HiddenStepWM, HiddenStepResWM
from mapformer.train import train
from mapformer.train_variant import VARIANT_MAP

ARMS = {"CF": VARIANT_MAP["Vanilla_r4"], "SR": VARIANT_MAP["SRoPEGen"], "CG": CtxGateWM,
        "HS": HiddenStepWM, "HSR": HiddenStepResWM, "RoPE": VARIANT_MAP["RoPE"]}


@torch.no_grad()
def evaluate(model, env, T, n_trials, dev, seed=10**6):   # never a training batch seed (audit m5)
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
    ap.add_argument("--variant", required=True, choices=list(ARMS))
    ap.add_argument("--p-decoy", type=float, default=0.3)
    ap.add_argument("--cue", required=True, choices=["lead", "trail"])
    ap.add_argument("--dist", required=True, choices=["near", "far"])
    ap.add_argument("--warm-cf", default=None,
                    help="HS only: start from a trained 1-layer context-free text-world model (Vanilla_r4.pt from "
                         "runs/textworld): its embeddings (the shared vocabulary prefix), step map, omega, output head "
                         "and its one layer (as HS's path-integrated layer 2) are copied; HS's context layer is fresh")
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
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output-dir", required=True)
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    env = TextWorldCtx3(size=a.size, seed=a.seed, p_decoy=a.p_decoy, cue=a.cue, dist=a.dist)
    model = ARMS[a.variant](vocab_size=env.unified_vocab_size, d_model=128, n_heads=2,
                                   n_layers=a.n_layers, grid_size=a.size)
    assert len(model.layers) == a.n_layers, (len(model.layers), a.n_layers)   # rule 17
    print(f"{a.variant} L{a.n_layers} seed={a.seed} params={sum(p.numel() for p in model.parameters()):,}")
    if a.warm_cf:
        assert a.variant == "HS", "--warm-cf is for the hidden-state step"
        src = torch.load(a.warm_cf, map_location="cpu", weights_only=False)["model_state_dict"]
        dst = model.state_dict(); n = 0
        for k, v in src.items():
            tk = k.replace("layers.0.", "layers.1.") if k.startswith("layers.0.") else k
            if tk not in dst:
                continue
            if tk in ("token_emb.weight", "out_proj.weight", "out_proj.bias"):
                dst[tk][:v.shape[0]] = v
            else:
                assert dst[tk].shape == v.shape, (tk, dst[tk].shape, v.shape); dst[tk] = v
            n += 1
        model.load_state_dict(dst)
        print(f"warm start from {a.warm_cf}: {n} tensors copied (context layer 0 and step_norm fresh)")
    losses = train(model, env, n_epochs=a.epochs, lr=a.lr, batch_size=a.batch_size, n_steps=a.n_steps,
                   n_batches=a.n_batches, device=a.device, schedule="cosine", data_workers=a.data_workers)
    torch.save({"model_state_dict": model.state_dict(), "losses": losses, "variant": a.variant,
                "seed": a.seed, "config": {"args": dict(vars(a)), "vocab_size": env.unified_vocab_size}},
               out / f"{a.variant}.pt")
    te = TextWorldCtx3(size=a.size, seed=10000, p_decoy=a.p_decoy, cue=a.cue, dist=a.dist)
    ev = {}
    for T in (a.n_steps, 2 * a.n_steps):
        acc, nll = evaluate(model, te, T, a.n_trials, a.device)
        ev[str(T)] = {"acc": acc, "nll": nll}
        print(f"held-out map T={T}: acc {acc:.4f} nll {nll:.4f}")
    json.dump({"eval": ev, "final_loss": float(np.mean(losses[-max(1, len(losses) // 20):]))},
              open(out / "eval.json", "w"), indent=2)


if __name__ == "__main__":
    main()
