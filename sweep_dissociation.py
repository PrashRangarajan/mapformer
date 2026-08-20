"""When does hierarchy pay, and when does path integration pay?

The question this answers
-------------------------
The repo has a clean-looking 2x2 at T=256 on the compositional task
(`COMPOSITIONAL_MULTISEED.md`, n=8):

                       cross_nb (transfer)      exact_acc @T=2048 (recall)
  Plain-Flat   (index, flat)     0.216                  0.538
  Plain-Hier   (index, hier)     0.318                  0.592
  MapWM-Flat   (path int, flat)  0.270                  0.646
  MapWM-Hier   (path int, hier)  0.415                  0.710

Hierarchy buys MORE than path integration on transfer (+0.102/+0.145 vs
+0.054/+0.097); path integration buys ~2x what hierarchy does on long-range
recall (+0.108/+0.118 vs +0.054/+0.064). And on PURE retrieval tasks hierarchy
actively hurts (HierAttn 0.769 vs flat 0.861 at T=4096; Match-Query 0.786 vs
0.888).

So the dissociation currently lives ACROSS tasks, which makes it an anecdote:
two architectures, two benchmarks, no controlled axis. This sweep moves it
INSIDE one task.

The axis: `n_templates`, the number of distinct room motifs tiled over the 64
rooms of the grid. Low = highly repetitive, so there is real compositional
structure to exploit. High = each room close to unique, so the task degenerates
toward pure per-cell retrieval with nothing to compose.

Pre-registered predictions, stated before the run:
  P1  the HIERARCHY advantage on cross_nb falls monotonically with n_templates,
      and approaches zero (or goes negative) at the high end.
  P2  the PATH-INTEGRATION advantage on exact_acc is roughly FLAT in
      n_templates -- knowing where you are does not depend on motif repetition.
  P3  therefore the two curves cross, and which ingredient matters is a property
      of the TASK, not of the architecture.

If P1 fails -- hierarchy keeps its advantage with no compositional structure to
exploit -- then "hierarchy buys compositional transfer" is the wrong mechanism
and the claim has to be rewritten, not just re-scoped.

Method notes that decide whether the numbers mean anything:
  - Every arm at every sweep point is trained in ONE batch (standing rule 3).
    Nothing here is compared against a stored checkpoint, including the
    n_templates=4 point that already exists in COMPOSITIONAL_MULTISEED.md.
  - Each model is evaluated on the environment it was TRAINED on.
    `eval_compositional.py`'s CLI hardcodes n_templates=4, so using it here would
    have silently scored every off-default point on the wrong environment.
    That is why this file exists rather than a flag on that one.
  - cross_nb has no analytic floor. The floor is reported per sweep point as the
    blank-excluded marginal, measured on the same scored events.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from mapformer.environment_compositional import CompositionalGridWorld
from mapformer.train_variant import VARIANT_MAP

_REPO = Path(__file__).resolve().parent
DISPLAY = {"Vanilla": "MapWM-Flat", "Hourglass_k2": "MapWM-Hier",
           "PlainFlat": "Plain-Flat", "PlainHourglass": "Plain-Hier"}


@torch.no_grad()
def evaluate(ckpt, n_templates, lengths, n_traj, device, batch=32, env_seed=10000):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    v = ck["variant"]
    model = VARIANT_MAP[v](vocab_size=ck["vocab_size"], d_model=ck["d_model"],
                           n_heads=ck["n_heads"], n_layers=ck["n_layers"],
                           grid_size=64).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    out = {}
    for T in lengths:
        env = CompositionalGridWorld(size=64, room_size=8,
                                     n_templates=n_templates, seed=env_seed)
        blank = env.unified_blank
        agg = {k: [0, 0] for k in ("exact", "cross_nb")}
        marg = Counter()
        done = 0
        while done < n_traj:
            b = min(batch, n_traj - done)
            batch_data = env.generate_batch(b, T)
            tokens = batch_data[0].to(device)
            exact_m = batch_data[2][:, 1:].to(device)
            cross_m = batch_data[4][:, 1:].to(device)
            tgt = tokens[:, 1:]
            pred = model(tokens[:, :-1]).argmax(-1)
            if exact_m.sum() > 0:
                agg["exact"][0] += int((pred[exact_m] == tgt[exact_m]).sum())
                agg["exact"][1] += int(exact_m.sum())
            nb = cross_m & (tgt != blank)
            if nb.sum() > 0:
                agg["cross_nb"][0] += int((pred[nb] == tgt[nb]).sum())
                agg["cross_nb"][1] += int(nb.sum())
                marg.update(tgt[nb].tolist())
            done += b
        n_nb = agg["cross_nb"][1]
        out[T] = {"exact_acc": agg["exact"][0] / max(agg["exact"][1], 1),
                  "cross_nb_acc": agg["cross_nb"][0] / max(n_nb, 1),
                  "cross_nb_floor": (max(marg.values()) / n_nb) if n_nb else float("nan"),
                  "n_exact": agg["exact"][1], "n_cross_nb": n_nb}
    del model
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(_REPO / "runs/dissociation"))
    ap.add_argument("--templates", nargs="+", type=int, default=[2, 4, 8, 16])
    ap.add_argument("--variants", nargs="+",
                    default=["Vanilla", "Hourglass_k2", "PlainFlat", "PlainHourglass"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--lengths", nargs="+", type=int, default=[256, 1024])
    ap.add_argument("--n-traj", type=int, default=128)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--out", default=str(_REPO / "DISSOCIATION_SWEEP.md"))
    args = ap.parse_args()
    dev = torch.device(args.device)

    R = {}
    for nt in args.templates:
        for v in args.variants:
            for s in args.seeds:
                ck = Path(args.runs_dir) / f"nt{nt}" / f"seed{s}" / f"{v}.pt"
                if not ck.exists():
                    print(f"MISSING {ck}", flush=True)
                    continue
                r = evaluate(ck, nt, args.lengths, args.n_traj, dev)
                R.setdefault((nt, v), []).append(r)
                print(f"nt={nt:3d} {v:16s} s{s}  "
                      + "  ".join(f"T={T}: cross_nb={r[T]['cross_nb_acc']:.3f} "
                                  f"exact={r[T]['exact_acc']:.3f}"
                                  for T in args.lengths), flush=True)

    def agg(nt, v, T, key):
        rows = R.get((nt, v), [])
        return np.array([x[T][key] for x in rows]) if rows else np.array([])

    L0 = args.lengths[0]
    LT = args.lengths[-1]
    lines = ["# Dissociation sweep: when does hierarchy pay, and when does path "
             "integration pay?", "",
             "The 2x2 {flat, hierarchical} x {index, path-integrated}, swept over "
             "`n_templates` -- the number of distinct room motifs tiled over the "
             "grid's 64 rooms. **Low = repetitive, real compositional structure "
             "to exploit. High = each room near-unique, so the task degenerates "
             "toward pure retrieval.**", "",
             "Every arm at every sweep point trained in ONE batch (rule 3); each "
             "model evaluated on the environment it was TRAINED on.", "",
             f"## Compositional transfer — `cross_nb_acc` @T={L0}", "",
             "| n_templates | " + " | ".join(DISPLAY[v] for v in args.variants)
             + " | **hierarchy effect** | **path-int effect** | floor |",
             "|---" * (len(args.variants) + 4) + "|"]
    summary = {}
    for nt in args.templates:
        cells, m = [], {}
        for v in args.variants:
            a = agg(nt, v, L0, "cross_nb_acc")
            m[v] = a.mean() if len(a) else float("nan")
            cells.append(f"{a.mean():.3f} ± {a.std(ddof=1):.3f}" if len(a) > 1
                         else (f"{a.mean():.3f}" if len(a) else "—"))
        hier = ((m["Hourglass_k2"] - m["Vanilla"]) +
                (m["PlainHourglass"] - m["PlainFlat"])) / 2
        pint = ((m["Vanilla"] - m["PlainFlat"]) +
                (m["Hourglass_k2"] - m["PlainHourglass"])) / 2
        fl = agg(nt, "Vanilla", L0, "cross_nb_floor")
        summary[nt] = {"cross_nb": m, "hier_effect": float(hier),
                       "pint_effect": float(pint),
                       "floor": float(fl.mean()) if len(fl) else None}
        lines.append(f"| {nt} | " + " | ".join(cells) +
                     f" | **{hier:+.3f}** | **{pint:+.3f}** | "
                     f"{fl.mean():.3f} |" if len(fl) else " — |")
    lines += ["", f"## Precise recall — `exact_acc` @T={LT}", "",
              "| n_templates | " + " | ".join(DISPLAY[v] for v in args.variants)
              + " | **hierarchy effect** | **path-int effect** |",
              "|---" * (len(args.variants) + 3) + "|"]
    for nt in args.templates:
        cells, m = [], {}
        for v in args.variants:
            a = agg(nt, v, LT, "exact_acc")
            m[v] = a.mean() if len(a) else float("nan")
            cells.append(f"{a.mean():.3f} ± {a.std(ddof=1):.3f}" if len(a) > 1
                         else (f"{a.mean():.3f}" if len(a) else "—"))
        hier = ((m["Hourglass_k2"] - m["Vanilla"]) +
                (m["PlainHourglass"] - m["PlainFlat"])) / 2
        pint = ((m["Vanilla"] - m["PlainFlat"]) +
                (m["Hourglass_k2"] - m["PlainHourglass"])) / 2
        summary[nt]["exact"] = m
        summary[nt]["hier_effect_exact"] = float(hier)
        summary[nt]["pint_effect_exact"] = float(pint)
        lines.append(f"| {nt} | " + " | ".join(cells) +
                     f" | **{hier:+.3f}** | **{pint:+.3f}** |")
    lines += ["", "## Pre-registered predictions", "",
              "| # | prediction | outcome |", "|---|---|---|",
              "| P1 | hierarchy's cross_nb advantage FALLS with n_templates, "
              "approaching zero at the high end | _fill in_ |",
              "| P2 | path integration's exact_acc advantage is FLAT in "
              "n_templates | _fill in_ |",
              "| P3 | the two cross: which ingredient matters is a property of "
              "the TASK, not the architecture | _fill in_ |", "",
              "If P1 fails, 'hierarchy buys compositional transfer' is the wrong "
              "mechanism and the claim needs rewriting, not re-scoping."]
    Path(args.out).write_text("\n".join(lines) + "\n")
    json.dump(summary, open(str(args.out).replace(".md", ".json"), "w"),
              indent=2, default=str)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
