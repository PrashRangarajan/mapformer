"""Aggregate the aliasing-controlled sweep: does the position effect track
observation ALIASING, or map size?

Written BEFORE the runs finished, with the verdict thresholds hard-coded, so the
conclusion cannot be tuned to the data after the fact. The four outcomes are the
ones pre-registered in run_alias_sweep.sh.

Reads per-seed JSON + checkpoint, pairs Vanilla (path-integrated) against RoPE
(index) WITHIN seed, and reports:
  - the paired effect per aliasing level, with the per-seed spread
  - whether every arm reached a flat loss (rule 10 -- an unconverged arm makes
    the comparison time-to-solve, not capability)
  - each arm's distance above its own non-blank marginal floor (rule 4), since
    accuracy is NOT comparable across n_obs -- the floors differ
  - the minimum detectable effect (rule 11), so a small number is called
    "unmeasured" rather than "null" when the design cannot see it
"""
import argparse
import json
import os

import numpy as np

FLAT_SLOPE = 5e-4
NOISE_FLOOR = 0.150        # measured from two function-identical models
ANCHOR_EFFECT = 0.173      # n_obs=16, grid 32, n=3, converged
ANCHOR_ROPE_S0 = 0.725     # stored value the repro control must reproduce

# occupied cells at grid 32, p_empty 0.5, and the non-blank marginal the gates
# measured at each condition -- the real floor for nb_acc, not 1/n_obs
CELLS_PER_TOKEN = {16: 32, 64: 8, 256: 2}
MARGINAL = {16: 0.077, 64: 0.031, 256: 0.013}


def load(d, variant):
    j = os.path.join(d, f"{variant}_oracle.json")
    p = j[:-5] + ".pt"
    if not (os.path.exists(j) and os.path.exists(p)):
        return None
    try:
        import torch
        r = json.load(open(j))
        L = torch.load(p, map_location="cpu").get("losses") or []
        if not L or "512" not in r:
            return None
        tail = L[int(0.9 * len(L)):]
        return dict(acc=r["512"]["nb_acc"], acc1024=r.get("1024", {}).get("nb_acc"),
                    loss=L[-1], slope=(tail[-1] - tail[0]) / max(1, len(tail) - 1))
    except Exception as e:
        return dict(unreadable=type(e).__name__)


def cell(runs_dir, nobs, seeds, anchor_dir=None):
    """-> (rows, acc effects, loss effects).

    n_obs=16 is split across two directories: seeds 0-2 are the REUSED anchor in
    anchor_dir (flat s{n} layout), seeds 3-4 were topped up in this batch under
    runs_dir/n16/. Look in both rather than silently dropping the top-up.
    """
    rows, eff, leff = [], [], []
    for sd in seeds:
        cands = [os.path.join(runs_dir, f"n{nobs}", f"s{sd}")]
        if anchor_dir:
            cands.append(os.path.join(anchor_dir, f"s{sd}"))
        v = r = None
        for d in cands:
            if v is None:
                v = load(d, "Vanilla")
            if r is None:
                r = load(d, "RoPE")
        rows.append((sd, v, r))
        if v and r and "unreadable" not in v and "unreadable" not in r:
            eff.append(v["acc"] - r["acc"])
            leff.append(v["loss"] - r["loss"])
    return rows, eff, leff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--anchor-dir", required=True,
                    help="runs/rope_converge -- the reused n_obs=16 anchor")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    o = ["# Is the position effect about ALIASING, or about map size?", "",
         "Grid size FIXED at 32; only `n_obs` varies, which RELABELS the obs_map and",
         "changes nothing else -- the gates confirm identical label mass (50.4 scored",
         "per trajectory) and identical revisit lag (median 33) across all three",
         "conditions. Same walks, same scored positions, different amounts of aliasing.",
         "", "400 epochs, warmup+cosine, oracle recode, n=5 seeds, T=512. Verdict",
         "thresholds were hard-coded in this script before the runs finished.", ""]

    # ---- anchor + repro control -------------------------------------------
    arows, aeff, aleff = cell(a.runs_dir, 16, a.seeds, anchor_dir=a.anchor_dir)
    repro = load(os.path.join(a.runs_dir, "n16_repro", "s0"), "RoPE")
    o += ["## Reproducibility of the reused anchor", ""]
    if repro and "unreadable" not in repro:
        drift = repro["acc"] - ANCHOR_ROPE_S0
        o += [f"RoPE n_obs=16 s0 retrained in THIS batch: {repro['acc']:.3f} vs stored "
              f"{ANCHOR_ROPE_S0:.3f} (drift {drift:+.3f}).", ""]
        licensed = abs(drift) < 0.03
        o += ["Cross-batch anchor is LICENSED." if licensed else
              f"**Drift exceeds 0.03. The anchor is NOT licensed; read every "
              f"n_obs=16 number against {abs(drift):.3f} as its error bar.**", ""]
    else:
        licensed = False
        o += ["**Repro control missing -- the reused anchor is unlicensed.**", ""]

    # ---- main table --------------------------------------------------------
    o += ["## Effect by aliasing level", "",
          "| n_obs | cells/token | Vanilla (path-int) | RoPE (index) | effect | per-seed | all flat? |",
          "|---|---|---|---|---|---|---|"]
    curve, allflat_all, table = {}, True, {}
    for nobs in (16, 64, 256):
        if nobs == 16:
            rows, eff, leff = arows, aeff, aleff
        else:
            rows, eff, leff = cell(a.runs_dir, nobs, a.seeds)
        vs = [v["acc"] for _, v, r in rows if v and "unreadable" not in v]
        rs = [r["acc"] for _, v, r in rows if r and "unreadable" not in r]
        flats = [abs(x["slope"]) < FLAT_SLOPE
                 for _, v, r in rows for x in (v, r) if x and "unreadable" not in x]
        allflat = bool(flats) and all(flats)
        allflat_all &= allflat
        if not eff:
            o.append(f"| {nobs} | {CELLS_PER_TOKEN[nobs]} | — | — | — | missing | — |")
            continue
        e = np.array(eff); curve[nobs] = e.mean()
        table[nobs] = dict(v=np.mean(vs), r=np.mean(rs), e=e, l=np.array(leff),
                           vl=[x["loss"] for _, x, _ in rows if x and "unreadable" not in x],
                           rl=[x["loss"] for _, _, x in rows if x and "unreadable" not in x])
        o.append(f"| {nobs} | {CELLS_PER_TOKEN[nobs]} | {np.mean(vs):.3f} | {np.mean(rs):.3f} | "
                 f"**{e.mean():+.3f}** | {', '.join(f'{x:+.3f}' for x in e)} | "
                 f"{'YES' if allflat else '**no**'} |")
    o += ["", f"Measured noise floor (two function-identical models): **{NOISE_FLOOR:.3f}**. "
              f"No effect smaller than this is reportable.", ""]

    # ---- floors: accuracy is NOT comparable across n_obs -------------------
    o += ["## Distance above each condition's own floor", "",
          "Raw accuracy is NOT comparable across `n_obs` -- more classes means a lower",
          "marginal. What is comparable is how far each arm sits above ITS floor.", "",
          "| n_obs | non-blank marginal (floor) | Vanilla above floor | RoPE above floor |",
          "|---|---|---|---|"]
    for nobs, t in table.items():
        m = MARGINAL[nobs]
        o.append(f"| {nobs} | {m:.3f} | {t['v']-m:+.3f} | {t['r']-m:+.3f} |")
    o.append("")

    # ---- the loss gap: what the accuracy effect actually is ----------------
    o += ["## The same comparison in TRAINING LOSS (rule 9)", "",
          "Held-out accuracy here is an affine readout of final training loss",
          "(r = -0.996 over 57 runs). So the accuracy effect above is a loss gap in",
          "disguise, and the loss gap is the more direct measurement of it. If the",
          "aliasing story is right, THIS should shrink with aliasing too.", "",
          "| n_obs | Vanilla loss | RoPE loss | loss gap (negative = path-int fits better) | ranges overlap? |",
          "|---|---|---|---|---|"]
    for nobs, t in table.items():
        vl, rl = t["vl"], t["rl"]
        ov = not (max(vl) < min(rl) or max(rl) < min(vl))
        o.append(f"| {nobs} | {np.mean(vl):.3f} [{min(vl):.3f}-{max(vl):.3f}] | "
                 f"{np.mean(rl):.3f} [{min(rl):.3f}-{max(rl):.3f}] | "
                 f"**{t['l'].mean():+.3f}** | {'yes' if ov else '**no**'} |")
    o += ["", "Where the ranges do NOT overlap, no loss-matched residual can be computed",
          "without extrapolating, so this study cannot separate 'path integration",
          "optimises better' from 'path integration represents better at equal fit'. It",
          "measures the former. That limit is inherent to these runs, not a choice of",
          "analysis.", ""]

    # ---- power -------------------------------------------------------------
    o += ["## Power (rule 11)", "",
          "| n_obs | n | mean effect | sd | MDE | detectable? |", "|---|---|---|---|---|---|"]
    for nobs, t in table.items():
        e = t["e"]
        mde = 2.8 * e.std(ddof=1) / np.sqrt(len(e)) if len(e) > 1 else float("nan")
        o.append(f"| {nobs} | {len(e)} | {e.mean():+.3f} | {e.std(ddof=1):.3f} | {mde:.3f} | "
                 f"{'yes' if abs(e.mean()) > mde else 'NO -> say *unmeasured*, not *null*'} |")
    o.append("")

    # ---- pre-registered verdict -------------------------------------------
    o += ["## Verdict", ""]
    if not allflat_all:
        o += ["**NOT ALL ARMS CONVERGED. Nothing here is interpretable** -- an "
              "unconverged arm makes this a time-to-solve comparison, which is "
              "exactly the confound that inverted the grid-8 sign (rule 10)."]
    elif len(curve) < 3:
        o += ["**INCOMPLETE** -- fewer than three aliasing levels landed. Two points "
              "make a line and a line is not a trend; no claim."]
    else:
        e16, e64, e256 = curve[16], curve[64], curve[256]
        floor_collapse = (256 in table and
                          table[256]["v"] - MARGINAL[256] < 0.10 and
                          table[256]["r"] - MARGINAL[256] < 0.10)
        monotone = e16 >= e64 >= e256
        if floor_collapse:
            o += ["**OUTCOME C -- FLOOR COLLAPSE, uninformative.** At n_obs=256 both arms "
                  "sit within 0.10 of the non-blank marginal, so neither learned the task. "
                  "A shrinking effect here is compression toward a floor, NOT evidence "
                  "about aliasing. The decisive cell did not run; the question is still open."]
        elif monotone and abs(e256) < NOISE_FLOOR:
            o += [f"**OUTCOME A -- ALIASING CONFIRMED, and now MANIPULATED.** The effect "
                  f"falls monotonically with cells/token ({e16:+.3f} -> {e64:+.3f} -> "
                  f"{e256:+.3f}) and lands below the {NOISE_FLOOR:.3f} noise floor at "
                  f"grid-8 aliasing on a grid-32 map. Map size is held fixed throughout, "
                  f"so it cannot explain this. The claim moves from correlational to "
                  f"manipulated -- the strongest form it has had."]
        elif all(abs(v) > NOISE_FLOOR for v in curve.values()):
            o += [f"**OUTCOME B -- ALIASING FALSIFIED.** The effect stays above the noise "
                  f"floor at every aliasing level ({e16:+.3f} / {e64:+.3f} / {e256:+.3f}) "
                  f"even at 2 cells/token, where grid 8 showed nothing. Aliasing is NOT "
                  f"what drives it; map size (or something else co-varying with it) is. "
                  f"**Withdraw the aliasing claim from CLAUDE.md and the memory file.**"]
        else:
            o += [f"**OUTCOME D -- NEITHER.** The curve ({e16:+.3f} / {e64:+.3f} / "
                  f"{e256:+.3f}) is not monotone and does not stay above the floor. "
                  f"Report the curve; claim nothing."]
        if not licensed:
            o += ["", "Note: the n_obs=16 anchor is unlicensed (see above), so the "
                  "leftmost point of this curve carries extra uncertainty."]
    o += ["", "## Scope", "",
          "One environment (MiniWorld OneRoom), one map size, n=5. Aliasing is varied by",
          "relabelling the obs_map, which is the cleanest available manipulation but is",
          "not the only way aliasing could be varied (p_empty and map size both also",
          "change it, and are held fixed here)."]

    open(a.out, "w").write("\n".join(o) + "\n")
    print("\n".join(o))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
