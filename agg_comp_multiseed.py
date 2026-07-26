"""
Aggregate multi-seed compositional checkpoints into mean +/- std tables.

Reuses eval_compositional.eval_ckpt (fresh held-out env, seed=10000) so the
metrics are identical to the single-seed eval, just aggregated across seeds.
Reports the two metrics that carry signal on this task:
  cross_nb_acc : accuracy on non-blank cross-instance cells (compositional target)
  exact_acc    : accuracy on exact-revisit cells (fine recall)
plus cross_nll. Writes markdown + a machine-readable JSON.
"""
import argparse
import json
import statistics as st
from pathlib import Path

from mapformer.eval_compositional import eval_ckpt

METRICS = ["exact_acc", "cross_acc", "cross_nb_acc", "cross_nll"]

# Clean backbone-structure display names. Checkpoints keep their raw filenames
# (Vanilla.pt, ...); this only controls how rows are labelled in the report.
DISPLAY = {
    "Vanilla": "MapWM-Flat",
    "VanillaEM": "MapEM-Flat",
    "Hourglass_k2": "MapWM-Hier",
    "HourglassFlat3": "MapWM-FlatHG",
    "Hourglass_MotifSeg": "MapWM-MotifSeg",
    "PlainHourglass": "Plain-Hier",
    "PlainFlat": "Plain-Flat",
}


def disp(v):
    return DISPLAY.get(v, v)


def mean_std(xs):
    if not xs:
        return (float("nan"), float("nan"), 0)
    m = st.mean(xs)
    s = st.pstdev(xs) if len(xs) > 1 else 0.0
    return (m, s, len(xs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True, help="root holding seed<N>/<Variant>.pt")
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--lengths", nargs="+", type=int, default=[256, 512, 1024, 2048])
    ap.add_argument("--n-traj", type=int, default=200)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="COMPOSITIONAL_MULTISEED.md")
    args = ap.parse_args()

    # raw[variant][T][metric] = list over seeds
    raw = {v: {T: {m: [] for m in METRICS} for T in args.lengths} for v in args.variants}
    seeds_used = {v: [] for v in args.variants}
    for v in args.variants:
        for s in args.seeds:
            cp = Path(args.runs_dir) / f"seed{s}" / f"{v}.pt"
            if not cp.exists():
                print(f"MISSING {cp} -- skipping")
                continue
            _, res = eval_ckpt(str(cp), args.lengths, args.n_traj, args.device,
                               batch_size=args.batch)
            seeds_used[v].append(s)
            for T in args.lengths:
                for m in METRICS:
                    raw[v][T][m].append(res[T][m])
            print(f"[{v} seed{s}] " + " ".join(
                f"T{T}:cnb={res[T]['cross_nb_acc']:.3f}" for T in args.lengths))

    agg = {v: {T: {m: mean_std(raw[v][T][m]) for m in METRICS} for T in args.lengths}
           for v in args.variants}

    def cell(v, T, m):
        mn, sd, n = agg[v][T][m]
        return f"{mn:.3f} ± {sd:.3f} (n={n})"

    lines = [
        "# Compositional-motif results — multi-seed (mean ± std)\n",
        f"Seeds requested: {args.seeds}. Fresh held-out env (seed=10000). "
        "cross_nb_acc = non-blank cross-instance cells (compositional target); "
        "exact_acc = exact-revisit recall.\n",
        f"Seeds actually found per variant: "
        + ", ".join(f"{disp(v)}={seeds_used[v]}" for v in args.variants) + "\n",
    ]
    for metric, label in [("cross_nb_acc", "cross_nb_acc (compositional target)"),
                          ("exact_acc", "exact_acc (fine recall)"),
                          ("cross_nll", "cross_nll (lower better)")]:
        lines.append(f"\n## {label}\n")
        header = "| variant | " + " | ".join(f"T={T}" for T in args.lengths) + " |"
        lines.append(header)
        lines.append("|" + "---|" * (len(args.lengths) + 1))
        for v in args.variants:
            lines.append("| " + disp(v) + " | "
                         + " | ".join(cell(v, T, metric) for T in args.lengths) + " |")

    Path(args.out).write_text("\n".join(lines) + "\n")
    Path(args.out).with_suffix(".json").write_text(json.dumps(
        {v: {str(T): {m: agg[v][T][m] for m in METRICS} for T in args.lengths}
         for v in args.variants}, indent=2))
    print(f"\nwrote {args.out} and {Path(args.out).with_suffix('.json')}")


if __name__ == "__main__":
    main()
