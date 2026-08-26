"""Aggregate the MiniWorld factorial: non-blank accuracy per (variant, encoding),
and the POSITION EFFECT (path-integrated - index) under raw vs allocentric.

The headline question: does the position effect flip from <=0 (raw rotation
actions) to positive (allocentric displacement), as on MiniGrid?

Convergence-aware (project rule: accuracy tracks final training loss at
r=-0.996; a stuck arm silently drags the pooled mean). Final train loss is read
from each arm's .pt (the trainer saves `losses`); arms above --loss-thresh are
FLAGGED and the pooled verdict is suppressed if any contributing arm is stuck or
missing. The position effect is reported PER SEED, paired within seed
(path-int - index at the same seed/encoding), with a std -- never a bare pooled
point estimate (Standing Rule 6: three seeds is not a point estimate)."""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

_REPO = os.path.dirname(os.path.abspath(__file__))
PATHINT = ["Vanilla", "MapPoPE-Flat"]          # path-integrated arms
INDEX = ["RoPE", "PoPE-Flat"]                   # sequence-index arms
ENCS = ["raw", "allo", "oracle"]


def _final_loss(runs_dir, seed, variant, enc):
    """Final training loss from the arm's .pt (torch saves `losses`); None if
    absent. Imported lazily so aggregation works even without torch present."""
    pt = os.path.join(runs_dir, f"s{seed}", f"{variant}_{enc}.pt")
    if not os.path.exists(pt):
        return None
    try:
        import torch
        d = torch.load(pt, map_location="cpu")
        ls = d.get("losses")
        return float(ls[-1]) if ls else None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=os.path.join(_REPO, "runs", "miniworld_fixed"))
    ap.add_argument("--variants", nargs="+", default=PATHINT + INDEX)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--length", type=int, default=512)
    ap.add_argument("--loss-thresh", type=float, default=0.6,
                    help="flag arms whose final train loss exceeds this as "
                         "possibly non-converged (the fixed-map seed-0 arms "
                         "converged to 0.01-0.16; >0.6 is suspicious)")
    ap.add_argument("--out", default=os.path.join(_REPO, "MINIWORLD_FIXED_RESULTS.md"))
    args = ap.parse_args()

    key = str(args.length)
    # acc[variant][enc][seed] = nb_acc ; loss[...] = final train loss
    acc = defaultdict(lambda: defaultdict(dict))
    loss = defaultdict(lambda: defaultdict(dict))
    missing, stuck = [], []
    for v in args.variants:
        for e in ENCS:
            for s in args.seeds:
                f = os.path.join(args.runs_dir, f"s{s}", f"{v}_{e}.json")
                if not os.path.exists(f):
                    missing.append(f"{v}_{e}_s{s}")
                    continue
                r = json.load(open(f))
                if key not in r:
                    missing.append(f"{v}_{e}_s{s}(no T={key})")
                    continue
                acc[v][e][s] = r[key]["nb_acc"]
                fl = _final_loss(args.runs_dir, s, v, e)
                loss[v][e][s] = fl
                if fl is not None and fl > args.loss_thresh:
                    stuck.append(f"{v}_{e}_s{s}(loss={fl:.2f})")

    def cell(v, e):
        vals = [acc[v][e][s] for s in args.seeds if s in acc[v][e]]
        a = np.array(vals)
        if len(a) == 0:
            return "—"
        fl = [loss[v][e].get(s) for s in args.seeds if s in acc[v][e]]
        flag = " ⚠" if any(x is not None and x > args.loss_thresh for x in fl) else ""
        return (f"{a.mean():.3f} ± {a.std(ddof=1):.3f}{flag}" if len(a) > 1
                else f"{a.mean():.3f}{flag}")

    # ---- per-seed paired position effect: mean over the 2 path-int and 2 index
    #      arms present at that seed, path-int minus index, one value per seed ----
    def paired_effect(enc):
        per_seed = {}
        for s in args.seeds:
            pi = [acc[v][enc][s] for v in PATHINT if s in acc[v][enc]]
            ix = [acc[v][enc][s] for v in INDEX if s in acc[v][enc]]
            if len(pi) == len(PATHINT) and len(ix) == len(INDEX):   # complete cell
                per_seed[s] = float(np.mean(pi) - np.mean(ix))
        return per_seed

    LABEL = {"raw": "raw (turn/forward)", "allo": "allocentric (24-bin dir)",
             "oracle": "oracle (exact cell Δ)"}
    effects = {e: paired_effect(e) for e in ENCS}
    present = [e for e in ENCS if effects[e]]
    complete = all(len(effects[e]) == len(args.seeds) for e in present) and not stuck and present

    def eff_line(enc, per_seed):
        if not per_seed:
            return f"| {LABEL.get(enc,enc)} | — | no complete seed |"
        vals = np.array([per_seed[s] for s in sorted(per_seed)])
        detail = ", ".join(f"s{s}={per_seed[s]:+.3f}" for s in sorted(per_seed))
        m = vals.mean()
        sd = vals.std(ddof=1) if len(vals) > 1 else float("nan")
        return f"| {LABEL.get(enc,enc)} | **{m:+.3f} ± {sd:.3f}** (n={len(vals)}) | {detail} |"

    lines = [f"# MiniWorld — non-blank accuracy at T={args.length}", "",
             "chance (non-blank) = 1/16 = 0.0625, oracle-acc = 1.0. Path-integrated = "
             f"{{{', '.join(PATHINT)}}}; index = {{{', '.join(INDEX)}}}. ⚠ marks an "
             f"arm with final train loss > {args.loss_thresh}.", "",
             "| variant | position | " + " | ".join(LABEL.get(e,e) for e in present) + " |",
             "|---|---|" + "---|" * len(present)]
    for v in args.variants:
        pos = "path-int" if v in PATHINT else "index"
        lines.append(f"| {v} | {pos} | " + " | ".join(cell(v, e) for e in present) + " |")

    lines += ["", "## Position effect (path-integrated − index), paired within seed", "",
              "| encoding | effect (mean ± std over seeds) | per-seed |",
              "|---|---|---|"] + [eff_line(e, effects[e]) for e in present] + [""]

    if not complete:
        lines += ["> **INCOMPLETE / SUSPECT — verdict withheld.**",
                  f"> missing: {missing or 'none'}",
                  f"> possibly-non-converged (loss>{args.loss_thresh}): {stuck or 'none'}",
                  "> Re-run the flagged arms in the same batch before reading the flip."]
    else:
        # flip test: compare the two richest encodings present (oracle vs allo for
        # the oracle experiment; else allo vs raw). Does the second RAISE the effect?
        pair = ("allo", "oracle") if {"allo", "oracle"} <= set(present) else (
                ("raw", "allo") if {"raw", "allo"} <= set(present) else None)
        if pair:
            lo, hi = pair
            ml, mh = np.mean(list(effects[lo].values())), np.mean(list(effects[hi].values()))
            deltas = [effects[hi][s] - effects[lo][s] for s in args.seeds]
            all_pos = all(x > 0 for x in deltas)
            verdict = (f"CONFIRMS — {LABEL[hi]} raises the position effect at every seed"
                       if (mh > ml + 0.02 and all_pos) else
                       f"no clear flip — {LABEL[hi]} does not reliably raise the gap")
            lines += [f"**Flip ({LABEL[lo]} → {LABEL[hi]}):** {ml:+.3f} → {mh:+.3f} "
                      f"(per-seed Δ = {', '.join(f'{x:+.3f}' for x in deltas)}). {verdict}."]

    open(args.out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
