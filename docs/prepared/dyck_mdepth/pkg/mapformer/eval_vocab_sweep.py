"""Multi-seed vocab sweep: does 'VanillaEM crashes at n_obs=256' survive?

VOCAB_SWEEP_RESULTS.md reported, at SINGLE seed, VanillaEM = 0.562 at n_obs=256
(T=512 OOD) -- worse than Vanilla's 0.665, which we read as the EM AND-gate
failing. But separate-q0/k0 EM is seed-unstable on the paper task
(0.898 +/- 0.108, one seed collapsing to 0.778), so a single-seed EM number
cannot distinguish "architecture fails" from "this seed collapsed".

This re-runs the sweep with 3 seeds and adds VanillaEM_P0 (the single-p_0
ablation, App. A.4). All arms are trained in ONE batch under current code, per
the standing rule that a fresh variant is never compared to a stored baseline.

Eval mirrors the original sweep: fresh obs_map (seed=1000), T=128 IID and
T=512 OOD, clean task. The accuracy harness is `revisit_accuracy` (train.py
masking), which differs from the original doc's inline `eval_clean`, so
absolute values may shift slightly -- every arm here uses the same harness, so
the within-table comparisons are what matter.
"""
import argparse
import json
import statistics as st
from pathlib import Path

import torch

from mapformer.environment import GridWorld
from mapformer.train_variant import VARIANT_MAP
from mapformer.eval_paper_task import revisit_accuracy

_REPO = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(_REPO / "runs/vocab_samebatch"))
    ap.add_argument("--variants", nargs="+",
                    default=["Vanilla", "VanillaEM", "VanillaEM_P0"])
    ap.add_argument("--n-obs", nargs="+", type=int, default=[16, 256, 4096])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--lengths", nargs="+", type=int, default=[128, 512])
    ap.add_argument("--n-batches", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--env-seed", type=int, default=1000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=str(_REPO / "VOCAB_SWEEP_MULTISEED.md"))
    args = ap.parse_args()

    dev = torch.device(args.device)
    res = {}
    for nobs in args.n_obs:
        for v in args.variants:
            key = f"{v}|{nobs}"
            res[key] = {T: [] for T in args.lengths}
            cls = VARIANT_MAP[v]
            for s in args.seeds:
                ckpt = Path(args.runs_dir) / f"{v}_vocab{nobs}" / f"seed{s}" / f"{v}.pt"
                if not ckpt.exists():
                    print(f"MISSING {ckpt} -- skipping")
                    continue
                blob = torch.load(ckpt, map_location="cpu", weights_only=False)
                sd, cfg = blob["model_state_dict"], blob["config"]
                env = GridWorld(size=cfg["grid_size"], n_obs_types=nobs,
                                p_empty=cfg["p_empty"], n_landmarks=0,
                                seed=args.env_seed)
                assert env.unified_vocab_size == cfg["vocab_size"], (
                    f"vocab mismatch {env.unified_vocab_size} vs {cfg['vocab_size']}")
                model = cls(vocab_size=cfg["vocab_size"], d_model=cfg["d_model"],
                            n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
                            grid_size=cfg["grid_size"]).to(dev)
                model.load_state_dict(sd)
                for T in args.lengths:
                    acc, nll, n = revisit_accuracy(model, env, args.n_batches,
                                                   args.batch_size, T, dev)
                    res[key][T].append(acc)
                    print(f"[{v} nobs={nobs} s{s}] T={T} acc={acc:.4f} "
                          f"nll={nll:.4f} (n={n})", flush=True)

    def cell(xs):
        if not xs:
            return "n/a"
        return f"{st.mean(xs):.3f}" + (f" ± {st.stdev(xs):.3f}" if len(xs) > 1 else "")

    lines = [
        "# Vocab sweep, MULTI-SEED (n=3), same training batch",
        "",
        "Supersedes the single-seed table in VOCAB_SWEEP_RESULTS.md, whose EM row "
        "could not be distinguished from a collapsed seed.",
        "",
        "`VanillaEM` = paper-faithful separate q0/k0 (App. A.4). "
        "`VanillaEM_P0` = single-p_0 ablation.",
        "",
    ]
    for nobs in args.n_obs:
        lines += [f"## n_obs = {nobs}", "",
                  "| variant | " + " | ".join(f"T={T}" for T in args.lengths) + " |",
                  "|---" * (len(args.lengths) + 1) + "|"]
        for v in args.variants:
            k = f"{v}|{nobs}"
            lines.append(f"| {v} | " + " | ".join(cell(res[k][T]) for T in args.lengths) + " |")
        lines.append("")
        # per-seed spread is the whole point of this rerun
        for v in args.variants:
            k = f"{v}|{nobs}"
            T = args.lengths[-1]
            if res[k][T]:
                pts = ", ".join(f"{x:.3f}" for x in res[k][T])
                lines.append(f"- {v} per-seed @T={T}: {pts}")
        lines.append("")

    Path(args.out).write_text("\n".join(lines) + "\n")
    Path(args.out).with_suffix(".json").write_text(json.dumps(
        {k: {str(t): v for t, v in d.items()} for k, d in res.items()}, indent=2))
    print("\n".join(lines))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
