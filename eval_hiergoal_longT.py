"""Evaluate EXISTING hier-goal checkpoints at much longer OOD explore lengths.

Motivation: on hier-goal the PoPE variants pin at ~0.948 from T=128 to T=256, so
the task cannot discriminate above ~0.95 -- which confounds "hierarchy adds
nothing to PoPE" with "no headroom exists". Pushing T far out either (a) shows
PoPE stays flat, so the no-headroom reading holds, or (b) finds where it breaks,
restoring a regime where the comparison is meaningful.

Inference only -- no retraining. T_explore is an eval-time parameter.
"""
import argparse
import json
import statistics as st
from pathlib import Path

import torch

from mapformer.environment_hier_goal import HierGoalGridWorld
from mapformer.train_variant import VARIANT_MAP
from mapformer.train_hier_goal import evaluate
from mapformer.agg_hiergoal import DISPLAY

_REPO = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(_REPO / "runs/hiergoal_multiseed"))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--lengths", nargs="+", type=int, default=[256, 512, 1024, 2048])
    ap.add_argument("--n-trials", type=int, default=100)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=str(_REPO / "HIERGOAL_LONGT.md"))
    ap.add_argument("--interleave-path", action="store_true",
                    help="MUST match how the checkpoints were trained. The fixed "
                         "hier-goal task interleaves the BFS path to kill the "
                         "copy-previous-action shortcut (0.969 -> 0.327); scoring "
                         "interleave-trained checkpoints on the un-interleaved env "
                         "is a train/eval mismatch that inflates accuracy.")
    args = ap.parse_args()

    env = HierGoalGridWorld(size=64, room_size=8, seed=10000,
                            interleave_path=args.interleave_path)
    acc = {v: {T: [] for T in args.lengths} for v in args.variants}
    for v in args.variants:
        for s in args.seeds:
            cp = Path(args.runs_dir) / f"seed{s}" / f"{v}_hiergoal.pt"
            if not cp.exists():
                print(f"MISSING {cp}"); continue
            c = torch.load(cp, map_location=args.device, weights_only=False)
            m = VARIANT_MAP[v](vocab_size=c["vocab_size"], d_model=c["d_model"],
                               n_heads=c["n_heads"], n_layers=c["n_layers"],
                               grid_size=64).to(args.device).eval()
            m.load_state_dict(c["model_state"])
            for T in args.lengths:
                a, _ = evaluate(m, env, T, 64, args.n_trials, args.device, seed=2000 + s)
                acc[v][T].append(a)
                print(f"[{DISPLAY.get(v, v)} s{s}] T_explore={T:5d}: acc={a:.3f}", flush=True)
            del m
            torch.cuda.empty_cache()

    def ms(xs):
        return (st.mean(xs), st.pstdev(xs) if len(xs) > 1 else 0.0) if xs else (float("nan"), 0.0)

    lines = ["# Hier-goal at LONG OOD explore length (existing checkpoints, inference only)\n",
             f"Trained at T_explore=64. n_trials={args.n_trials}, seeds={args.seeds}.\n",
             "| variant | " + " | ".join(f"T={T}" for T in args.lengths) + " |",
             "|" + "---|" * (len(args.lengths) + 1)]
    for v in args.variants:
        cells = [f"{ms(acc[v][T])[0]:.3f} ± {ms(acc[v][T])[1]:.3f}" for T in args.lengths]
        lines.append(f"| {DISPLAY.get(v, v)} | " + " | ".join(cells) + " |")
    Path(args.out).write_text("\n".join(lines) + "\n")
    Path(args.out).with_suffix(".json").write_text(json.dumps(
        {DISPLAY.get(v, v): {str(T): ms(acc[v][T]) for T in args.lengths} for v in args.variants}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
