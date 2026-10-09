"""Trainer for TW_AMBIG_PREREG.md: the text world where direction words are sometimes actions and sometimes observed
content (environment_tw_ambig.TextWorldAmbig, map seed = --seed; the oracle arms read the tagged stream, whose random
draws are identical). Recipe = the text world's (train_textworld.py / train_tw_normstep.py): T = 1024 words, batch 16,
900 epochs x 98 batches, lr 1e-3, 5% warmup + cosine, AdamW wd 0.05, d 128, 2 heads, --data-workers 3. A separate
entry point; no existing module is edited (rule 22). VARIANT_MAP is not imported (the arms' classes are imported
directly; identity with VARIANT_MAP['Vanilla_r4'] / ['RoPE'] is checked in docs/audits/2026-10-08/tw_ambig_checks.py).

Eval (registered): held-out map (env seed 10000), np seed 10**6 (no training stream uses it), 200 walks, revisit
object accuracy at T = --n-steps (registered) and 2x (no verdict), eval mode -- train_textworld.evaluate's code.
Also (declared, Amendment-2 lesson of TW_NORMSTEP): the same 200 walks at T = --n-steps in TRAIN mode (all dropout on,
mean of 3 dropout seeds). Writes <arm>.pt and eval.json in --output-dir. --n-trials 0 skips the eval (checks)."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mapformer.environment_tw_ambig import TextWorldAmbig
from mapformer.model_tw_ambig import ARMS, build
from mapformer.train import train

EVAL_SEED, HELDOUT = 10**6, 10000


def walks(env, T, n, seed=EVAL_SEED):
    np.random.seed(seed)
    return [env.generate_trajectory(T) for _ in range(n)]


@torch.no_grad()
def score(model, W, dev):
    """Revisit object accuracy and NLL over walks W (train_textworld.evaluate's arithmetic)."""
    ok = tot = 0; nll = 0.0
    for tok, _om, rev in W:
        tok = tok.unsqueeze(0).to(dev)
        lp = F.log_softmax(model(tok[:, :-1]).float(), dim=-1)
        tgt = tok[0, 1:]; m = rev[1:].to(dev)
        if m.sum() == 0:
            continue
        ok += (lp.argmax(-1)[0][m] == tgt[m]).sum().item(); tot += int(m.sum())
        nll += float(-lp[0, torch.arange(lp.shape[1], device=dev)[m], tgt[m]].sum())
    if tot == 0:                     # only reachable in tiny check configs (registered: 200 walks x 1024 words)
        return float("nan"), float("nan")
    return ok / tot, nll / tot


def evaluate_all(model, env_te, T, n_trials, dev, n_drop=3):
    ev = {}
    for TT in (T, 2 * T):
        W = walks(env_te, TT, n_trials); model.eval()
        acc, nll = score(model, W, dev); ev[str(TT)] = {"acc": acc, "nll": nll}
        print(f"held-out map T={TT}: acc {acc:.4f} nll {nll:.4f}", flush=True)
        if TT == T:
            tr = []
            for k in range(n_drop):
                model.train(); torch.manual_seed(k); tr.append(score(model, W, dev)[0])
            model.eval()
            ev[str(TT)]["acc_train_mode"] = float(np.mean(tr)); ev[str(TT)]["acc_train_mode_each"] = tr
            print(f"   train mode (dropout on, {n_drop} seeds): {np.mean(tr):.4f}", flush=True)
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--p-nm", type=float, default=0.3)
    ap.add_argument("--size", type=int, default=64)
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
    tagged = ARMS[a.arm][2]
    env = TextWorldAmbig(size=a.size, seed=a.seed, p_nm=a.p_nm, tag_roles=tagged)
    model = build(a.arm, env, a.size)
    print(f"{a.arm} L{len(model.layers)} seed={a.seed} p_nm={a.p_nm} tagged={tagged} "
          f"params={sum(p.numel() for p in model.parameters()):,}", flush=True)
    losses = train(model, env, n_epochs=a.epochs, lr=a.lr, batch_size=a.batch_size, n_steps=a.n_steps,
                   n_batches=a.n_batches, device=a.device, schedule="cosine", data_workers=a.data_workers)
    torch.save({"model_state_dict": model.state_dict(), "losses": losses, "arm": a.arm, "seed": a.seed,
                "config": {"args": dict(vars(a)), "vocab_size": env.unified_vocab_size, "torch": torch.__version__}},
               out / f"{a.arm}.pt")
    res = {"final_loss": float(np.mean(losses[-max(1, len(losses) // 20):]))}
    if hasattr(model, "ctx_alpha"):
        res["alpha"] = float(model.ctx_alpha.detach().cpu())
    if a.n_trials > 0:
        te = TextWorldAmbig(size=a.size, seed=HELDOUT, p_nm=a.p_nm, tag_roles=tagged)
        res["eval"] = evaluate_all(model, te, a.n_steps, a.n_trials, a.device)
    json.dump(res, open(out / "eval.json", "w"), indent=2)


if __name__ == "__main__":
    main()
