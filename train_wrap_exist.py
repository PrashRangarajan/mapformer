"""Trainer for WRAP_EXISTENCE_PREREG.md (rule 14 existence for the ring learnability claim). Ring (environment_cancel),
p_plus 0.5, the RING_DIAL recipe (T = 128, batch 128, 98 batches, lr 1e-3, cosine, 1 layer, 2 heads, d 128). Arms:
  SO        SignedAO_r4 from scratch (observation steps 0 by construction, action steps free, signed)
  AO        AbsAO_r4 from scratch (the same with |.|)
  SO_cont   SO warm-started from --init (an SO checkpoint), trained --epochs more on a fresh walk stream
  AO_wrap   AbsAO_r4 CONSTRUCTED from --init (wrap_construct: the signed action angles mod 2 pi), evaluated frozen, then
            trained --epochs on the same fresh walk stream as SO_cont
  AO_wrapg  the same with the per-channel gauge construction (descriptive)
  AO_bal    AbsAO_r4 from the BALANCED construction (Amendment 1 primary: orthogonal action latents, omega' = max(|omega|, 1),
            steps in [0, 2 pi))
  SO_bal    SignedAO_r4 at the balanced gauge with the signed angles (AO_bal's matched control)
Continuations use data_seed_offset 10**6 (fresh walks, same seed). Evaluation on a fresh stream (np seed 10**6, maps redrawn
per trajectory) at T and 4T. Writes <arm>.pt and eval.json (AO_wrap also: frozen accuracy and the construction report).
  python3 -m mapformer.train_wrap_exist --arm SO --seed 600 --device cuda:0 --output-dir runs/wrap_existence/p0/SO_s600
"""
import argparse, json
from pathlib import Path

import numpy as np
import torch

from mapformer.environment_cancel import GridWorldCancel
from mapformer.model_wrap_exist import MapFormerWM_SignedAO_r4, MapFormerWM_AbsAO_r4, wrap_construct, wrap_share
from mapformer.train import train
from mapformer.train_cancel import evaluate

ARMS = ("SO", "AO", "SO_cont", "AO_wrap", "AO_wrapg", "AO_bal", "SO_bal")
BUILT = {"AO_wrap": "min", "AO_wrapg": "gauge", "AO_bal": "balanced", "SO_bal": "signed_balanced"}
EVAL_SEED = 10 ** 6
OFFSET = 10 ** 6


def model_kw(env, size):
    return dict(vocab_size=env.unified_vocab_size, d_model=128, n_heads=2, n_layers=1, grid_size=size,
                action_ids=(env.action_offset + 0, env.action_offset + 1))     # +1, -1 (environment_cancel)


def load_so(path, env, size):
    b = torch.load(path, map_location="cpu", weights_only=False)
    assert b["variant"] == "SO", b["variant"]
    m = MapFormerWM_SignedAO_r4(**model_kw(env, size)); m.load_state_dict(b["model_state_dict"])
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--init", default=None, help="SO checkpoint (SO_cont and the constructed arms)")
    ap.add_argument("--p-plus", type=float, default=0.5)
    ap.add_argument("--size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--n-batches", type=int, default=98)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--n-steps", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--data-workers", type=int, default=3)
    ap.add_argument("--n-trials", type=int, default=400)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output-dir", required=True)
    a = ap.parse_args()
    cont = a.arm in ("SO_cont", *BUILT)
    assert cont == (a.init is not None), "--init is required for exactly SO_cont and the constructed arms"

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    env = GridWorldCancel(size=a.size, seed=a.seed, p_plus=a.p_plus)
    te = GridWorldCancel(size=a.size, seed=10000, p_plus=a.p_plus)
    extra = {}
    if a.arm == "SO":
        model = MapFormerWM_SignedAO_r4(**model_kw(env, a.size))
    elif a.arm == "AO":
        model = MapFormerWM_AbsAO_r4(**model_kw(env, a.size))
    else:
        src = load_so(a.init, env, a.size)
        if a.arm == "SO_cont":
            model = src
        else:
            kw = model_kw(env, a.size)
            model, rep = wrap_construct(src, kw.pop("vocab_size"), kw.pop("action_ids"), mode=BUILT[a.arm], **kw)
            src_acc = evaluate(src, te, a.n_steps, a.n_trials, "cpu", seed=EVAL_SEED)[0]
            frz_acc = evaluate(model, te, a.n_steps, a.n_trials, "cpu", seed=EVAL_SEED)[0]
            extra = {"construction": rep, "source_acc": src_acc, "frozen_acc": frz_acc,
                     "frozen_wrap": wrap_share(model), "source_wrap": wrap_share(src)}
            src.eval(); model.eval(); g = np.random.RandomState(7); ld = {}
            with torch.no_grad():
                for T in (a.n_steps, 4 * a.n_steps):
                    np.random.seed(g.randint(2 ** 31)); tok = torch.stack([te.generate_trajectory(T)[0] for _ in range(8)])
                    ld[str(T)] = float((model(tok) - src(tok)).abs().max())
            extra["logit_diff_max"] = ld
            print(f"constructed: {rep} | source acc {src_acc:.4f} frozen acc {frz_acc:.4f} | max logit diff {ld}", flush=True)
        extra["init"] = a.init
    assert len(model.layers) == 1
    print(f"{a.arm} p_plus={a.p_plus} seed={a.seed} params={sum(p.numel() for p in model.parameters()):,}", flush=True)
    losses = train(model, env, n_epochs=a.epochs, lr=a.lr, batch_size=a.batch_size, n_steps=a.n_steps,
                   n_batches=a.n_batches, device=a.device, schedule="cosine", data_workers=a.data_workers,
                   data_seed_offset=OFFSET if cont else 0)
    torch.save({"model_state_dict": model.state_dict(), "losses": losses, "variant": a.arm, "seed": a.seed,
                "config": {"args": dict(vars(a)), "vocab_size": env.unified_vocab_size,
                           "action_offset": env.action_offset}}, out / f"{a.arm}.pt")
    ev = {}
    for T in (a.n_steps, 4 * a.n_steps):
        acc, nll = evaluate(model, te, T, a.n_trials, a.device, seed=EVAL_SEED)
        ev[str(T)] = {"acc": acc, "nll": nll}
        print(f"fresh maps T={T}: acc {acc:.4f} nll {nll:.4f}", flush=True)
    model.cpu()
    json.dump({"eval": ev, "final_loss": float(np.mean(losses[-max(1, len(losses) // 20):])), "wrap": wrap_share(model),
               **extra}, open(out / "eval.json", "w"), indent=2)


if __name__ == "__main__":
    main()
