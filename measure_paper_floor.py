"""Measured floors for the paper-task OOD conditions -- the numbers eval_paper_ood does not report.

`BASELINE_TABLE.md:77-78` says "A 0.80 on OOD-s is the floor, not a score", and `EM_WM_THEORY.md`
2a demotes an EM-vs-WM claim partly on that ground. Nothing in the pipeline actually MEASURES it
per condition, so this does, on exactly the events `revisit_accuracy` scores:

    always-blank : predict the blank token at every scored position
    marginal     : predict the single most frequent target (the best constant predictor)

Conditions and env construction are IMPORTED from `eval_paper_ood`, not retyped, so they cannot
drift from the evaluator (rule 7: a gate must call the task code). No model is involved.

    python3 -m mapformer.measure_paper_floor
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

from mapformer.environment import GridWorld
from mapformer.eval_paper_ood import CONDITIONS, EXTENDED


def floors(L, g, pe, n_obs_types, n_batches, batch_size, env_seed):
    env = GridWorld(size=g, n_obs_types=n_obs_types, p_empty=pe, n_landmarks=0, seed=env_seed)
    c = Counter()
    for _ in range(n_batches):
        tokens, _om, revisit, *_ = env.generate_batch(batch_size, L)
        c.update(tokens[:, 1:][revisit[:, 1:]].tolist())
    n = sum(c.values())
    return dict(n=n, always_blank=c[env.unified_blank] / max(n, 1),
                marginal=max(c.values()) / max(n, 1) if c else float("nan"),
                chance=1.0 / env.unified_vocab_size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-obs-types", type=int, default=16)
    ap.add_argument("--n-batches", type=int, default=8)      # as run_seed_scaleup.sh evaluated
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--env-seed", type=int, default=10000)
    a = ap.parse_args()
    rows, out = [], {}
    for label, L, g, pe in list(CONDITIONS) + list(EXTENDED):
        f = floors(L, g, pe, a.n_obs_types, a.n_batches, a.batch_size, a.env_seed)
        out[label] = f
        rows.append(f"| {label} | {f['always_blank']:.3f} | {f['marginal']:.3f} | "
                    f"{f['chance']:.4f} | {f['n']:,} |")
        print(rows[-1], flush=True)
    txt = "\n".join(["# Measured floors for the paper-task OOD protocol\n",
                     "Constant predictors scored on exactly the events `revisit_accuracy` scores "
                     f"(fresh obs_map, env seed {a.env_seed}; {a.n_batches} x {a.batch_size}). "
                     "No model.\n",
                     "| condition | always-blank | best constant (marginal) | 1/vocab | scored n |",
                     "|---|---|---|---|---|", *rows])
    (__import__("pathlib").Path("/home/prashr/mapformer/PAPER_TASK_FLOORS.md")).write_text(txt + "\n")
    json.dump(out, open("/home/prashr/mapformer/PAPER_TASK_FLOORS.json", "w"), indent=1)
    print("\n" + txt)


if __name__ == "__main__":
    main()
