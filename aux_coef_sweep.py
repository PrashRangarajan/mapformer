#!/usr/bin/env python3
"""Test 2: aux_coef dose-response sweep.

Trains Level15PC on lm200 single-seed at aux_coef in {0, 0.01, 0.03, 0.1, 0.3}.
Evaluates revisit accuracy at T=512 OOD on each.

If interference is the mechanism: accuracy monotonically decreases with
aux_coef. The dose-response curve confirms it's specifically the PC
gradient causing the regression.

Outputs:
  AUX_COEF_SWEEP.md — table of (aux_coef → final loss, revisit acc, NLL)
  Checkpoints in runs/Level15PC_aux{coef}/seed0/
"""

import argparse, subprocess, sys, time, os
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mapformer.environment import GridWorld
from mapformer.model_level15_pc import MapFormerWM_Level15PC


REPO = Path("/home/prashr/mapformer")
RUNS = REPO / "runs"
LOGS = REPO / "logs"


def train_one(aux_coef, gpu, seed=0, n_landmarks=200, p_noise=0.10,
              epochs=50, n_batches=156):
    """Launch a single training run with given aux_coef. Returns (proc, ckpt_path)."""
    tag = f"aux{aux_coef}"
    out_dir = RUNS / f"Level15PC_lm200_{tag}" / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "Level15PC.pt"
    log_path = LOGS / f"Level15PC_lm200_{tag}_s{seed}.log"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = [
        "python3", "-u", "-m", "mapformer.train_variant",
        "--variant", "Level15PC", "--seed", str(seed),
        "--n-landmarks", str(n_landmarks),
        "--p-action-noise", str(p_noise),
        "--epochs", str(epochs), "--n-batches", str(n_batches),
        "--aux-coef", str(aux_coef),
        "--device", "cuda", "--output-dir", str(out_dir),
    ]
    log_f = open(log_path, "w")
    proc = subprocess.Popen(cmd, cwd="/home/prashr", env=env,
                            stdout=log_f, stderr=subprocess.STDOUT)
    return proc, log_f, ckpt, log_path


def eval_revisit(model, env, T, n_trials, seed):
    """Return (acc, nll, n_revisits) on OOD trajectories."""
    torch.manual_seed(seed); np.random.seed(seed)
    correct = total = 0; nll_sum = 0.0
    with torch.no_grad():
        for _ in range(n_trials):
            tokens, _, rm = env.generate_trajectory(T)
            tt = tokens.unsqueeze(0).cuda()
            try:
                logits = model(tt[:, :-1])
            except Exception:
                return None, None, 0
            lp = F.log_softmax(logits, dim=-1)
            preds = lp.argmax(-1)[0]
            tgts = tt[0, 1:]
            mask = rm[1:].cuda()
            if mask.sum() == 0: continue
            correct += (preds[mask] == tgts[mask]).sum().item()
            total += mask.sum().item()
            idx = torch.arange(lp.shape[1], device="cuda")[mask]
            nll_sum += -lp[0, idx, tgts[mask]].sum().item()
    return (correct / total if total else None,
            nll_sum / total if total else None, total)


def parse_final_loss(log_path):
    """Extract final-epoch training loss from a log file."""
    if not log_path.exists():
        return None
    text = log_path.read_text()
    for line in reversed(text.splitlines()):
        if "Epoch" in line and "Loss:" in line and "/50" in line and "50/50" in line:
            try:
                loss_str = line.split("Loss:")[1].split("|")[0].strip()
                return float(loss_str)
            except Exception:
                continue
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--coefs", type=float, nargs="+",
                   default=[0.0, 0.01, 0.03, 0.1, 0.3])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--n-batches", type=int, default=156)
    p.add_argument("--T-eval", type=int, default=512)
    p.add_argument("--n-eval-trials", type=int, default=100)
    p.add_argument("--output", default=str(REPO / "AUX_COEF_SWEEP.md"))
    args = p.parse_args()

    LOGS.mkdir(exist_ok=True)
    print(f"[sweep] {len(args.coefs)} runs at aux_coef = {args.coefs}", file=sys.stderr)

    # Train all coefs (2 GPUs in parallel)
    queue = list(args.coefs)
    in_flight = {}  # gpu -> (coef, proc, log_f, ckpt, log_path)
    completed = []  # (coef, ckpt, log_path)
    while queue or in_flight:
        for gpu in [0, 1]:
            if gpu not in in_flight and queue:
                coef = queue.pop(0)
                proc, log_f, ckpt, log_path = train_one(
                    coef, gpu=gpu, seed=args.seed,
                    epochs=args.epochs, n_batches=args.n_batches,
                )
                in_flight[gpu] = (coef, proc, log_f, ckpt, log_path)
                print(f"[sweep] GPU{gpu}: aux_coef={coef} (pid {proc.pid})",
                      file=sys.stderr)
        # poll
        for gpu in list(in_flight.keys()):
            coef, proc, log_f, ckpt, log_path = in_flight[gpu]
            if proc.poll() is not None:
                log_f.close()
                completed.append((coef, ckpt, log_path))
                print(f"[sweep] GPU{gpu}: aux_coef={coef} done "
                      f"(rc={proc.returncode}, ckpt {'exists' if ckpt.exists() else 'MISSING'})",
                      file=sys.stderr)
                del in_flight[gpu]
        time.sleep(5)
    completed.sort(key=lambda x: x[0])

    # Evaluate all
    results = []
    for coef, ckpt, log_path in completed:
        final_loss = parse_final_loss(log_path)
        if not ckpt.exists():
            results.append((coef, final_loss, None, None, "no checkpoint"))
            continue
        c = torch.load(ckpt, map_location="cuda", weights_only=False)
        cfg = c.get("config", {})
        m = MapFormerWM_Level15PC(
            vocab_size=cfg.get("vocab_size"),
            d_model=cfg.get("d_model", 128),
            n_heads=cfg.get("n_heads", 2),
            n_layers=cfg.get("n_layers", 1),
            grid_size=cfg.get("grid_size", 64),
        )
        m.load_state_dict(c["model_state_dict"])
        m = m.cuda().eval()

        # OOD env: same task, fresh obs_map seed
        env = GridWorld(size=64, n_obs_types=16, p_empty=0.5,
                        n_landmarks=200, seed=args.seed + 1000)
        acc, nll, nrev = eval_revisit(m, env, args.T_eval, args.n_eval_trials,
                                       seed=args.seed + 2000)
        del m; torch.cuda.empty_cache()
        results.append((coef, final_loss, acc, nll, "ok"))
        print(f"[eval] aux_coef={coef}: loss={final_loss}, "
              f"OOD T={args.T_eval} acc={acc}, nll={nll}", file=sys.stderr)

    # Markdown
    md = ["# Test 2: aux_coef dose-response sweep on lm200\n"]
    md.append("Train Level15PC at varying `aux_coef`. If PC's aux loss is "
              "interfering with Level 1.5's R_t learning, OOD accuracy on "
              "lm200 should monotonically decrease with `aux_coef`.\n")
    md.append("Reference: `Level15` alone (no PC) gets **0.821 ± 0.025** at "
              "T=512 OOD on lm200.\n")
    md.append("| aux_coef | final train loss | OOD T=512 acc | OOD T=512 NLL |")
    md.append("|---|---|---|---|")
    for coef, loss, acc, nll, status in results:
        l = f"{loss:.4f}" if loss is not None else "—"
        a = f"{acc:.3f}" if acc is not None else f"({status})"
        n = f"{nll:.3f}" if nll is not None else "—"
        md.append(f"| {coef} | {l} | {a} | {n} |")
    md.append("\n**Decision rule:** "
              "If accuracy at coef=0 ≈ Level15 (~0.82) and accuracy at "
              "coef=0.1 ≈ 0.59 (current Level15PC), AND there's a smooth "
              "monotone curve between them: interference confirmed.\n")
    md.append("\n*Auto-generated by `aux_coef_sweep.py`.*\n")
    Path(args.output).write_text("\n".join(md))
    print(f"wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
