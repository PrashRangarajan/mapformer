"""Preflight/postflight audit for any MiniWorld-style run directory.

WHY THIS EXISTS. Over 2026-08-26..28 roughly 200 training arms produced five
architectural claims, and four were retracted. Every retraction had the same root
cause and each was rediscovered separately, by hand, days apart. This script runs
all five checks at once so the next person gets them in thirty seconds.

The five failure modes, in the order they bit:

1. NON-CONVERGENCE. Arms stopped mid-plateau, so the comparison measured
   time-to-solve, not capability. LinearLR(1.0->0.0) with no warmup was the
   culprit -- it decays from step one, so a run can never escape a plateau late.
   Switching to warmup+cosine moved one arm from 0.448 to 0.990 on the SAME task.
2. ACCURACY IS A READOUT OF LOSS. Measured r = -0.996 over 57 runs
   (acc = 1.039 - 0.555*loss, resid sd 0.021). When that holds, the held-out eval
   carries no information the training loss does not, and every "effect" is a
   loss gap in disguise.
3. NOISE FLOOR UNMEASURED. Two provably function-identical models differed by
   mean |delta| 0.150 (range -0.23..+0.41). Most effects chased were smaller.
4. UNDERPOWERED NULLS. "X is a null" was claimed where the minimum detectable
   effect at n=9 was 0.165 -- the data admitted anything from -0.09 to +0.16.
5. CEILING SELECTION. Conditioning on convergence selected pairs where BOTH arms
   already scored >0.95, so no effect could be expressed either way.

USAGE
    python3 -m mapformer.experiment_audit --runs-dir mapformer/runs/mw_grid_sweep
    python3 -m mapformer.experiment_audit --runs-dir <dir> --control GateDeltaCtl \
        --control-of Vanilla      # measures the noise floor from an inert control
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

_REPO = os.path.dirname(os.path.abspath(__file__))
FLAT_SLOPE = 5e-4          # |loss slope| over the final 10% below this = converged
CONVERGED_LOSS = 0.4       # legacy threshold; ALWAYS report sensitivity, never one value


def _load(runs_dir, length):
    """-> {(variant, cell, seed): dict(acc, loss, slope)}. Cell = the grid/config dir."""
    import torch
    out = {}
    for js in glob.glob(os.path.join(runs_dir, "**", "*.json"), recursive=True):
        pt = js[:-5] + ".pt"
        if not os.path.exists(pt):
            continue
        base = os.path.basename(js)[:-5]
        parts = js.split(os.sep)
        seed = next((int(p[1:]) for p in parts if p.startswith("s") and p[1:].isdigit()), None)
        cell = next((p for p in parts if p.startswith("g") and p[1:].isdigit()), "-")
        if seed is None:
            continue
        try:
            r = json.load(open(js))
            if str(length) not in r:
                continue
            L = torch.load(pt, map_location="cpu").get("losses") or []
            if not L:
                continue
            tail = L[int(0.9 * len(L)):]
            slope = (tail[-1] - tail[0]) / max(1, len(tail) - 1)
            out[(base, cell, seed)] = dict(acc=r[str(length)]["nb_acc"],
                                           loss=L[-1], slope=slope)
        except Exception as e:
            print(f"  [warn] unreadable {js}: {type(e).__name__}")
    return out


def check_convergence(D):
    print("\n[1] CONVERGENCE -- is the comparison measuring capability or time-to-solve?")
    byv = defaultdict(list)
    for (v, c, s), d in D.items():
        byv[v].append(d)
    for v, ds in sorted(byv.items()):
        flat = sum(1 for d in ds if abs(d["slope"]) < FLAT_SLOPE)
        lo = min(d["loss"] for d in ds); hi = max(d["loss"] for d in ds)
        warn = "" if flat == len(ds) else "   <-- NOT ALL FLAT"
        print(f"    {v:18s} n={len(ds):2d}  flat {flat}/{len(ds)}  "
              f"loss {lo:.3f}-{hi:.3f}{warn}")
    allflat = all(abs(d["slope"]) < FLAT_SLOPE for d in D.values())
    print("    => " + ("all arms converged; capability comparison is meaningful."
                       if allflat else
                       "NOT all arms converged. Any effect here may be time-to-solve. "
                       "Try warmup+cosine and a longer budget BEFORE interpreting."))
    return allflat


def check_loss_readout(D):
    print("\n[2] IS ACCURACY JUST THE TRAINING LOSS? (r ~ -0.99 means the eval adds nothing)")
    if len(D) < 6:
        print("    too few runs"); return None
    x = np.array([d["loss"] for d in D.values()]); y = np.array([d["acc"] for d in D.values()])
    r = float(np.corrcoef(x, y)[0, 1])
    A = np.polyfit(x, y, 1); resid = float(np.std(y - np.polyval(A, x), ddof=1))
    print(f"    r = {r:+.4f}   fit acc = {A[1]:.3f} {A[0]:+.3f}*loss   resid sd {resid:.3f}")
    if r < -0.98:
        print("    => held-out accuracy is an AFFINE READOUT of training loss. Any claim "
              "must be argued from the loss, and loss-matched residuals are the real effect.")
    return r


def check_noise_floor(D, control, control_of):
    print("\n[3] MEASURED NOISE FLOOR (two function-identical models)")
    if not control:
        print("    no --control given. STRONGLY RECOMMENDED: train an inert twin of a real "
              "arm (same params, effect multiplied out) and pass it here. Without it you "
              "have no idea which effects are real.")
        return None
    # variant keys carry encoding suffixes (e.g. "GateDeltaCtl_oracle"), so match
    # on prefix rather than equality -- an exact-match bug here silently reports
    # "no pairs found" and the noise floor never gets measured.
    ds = []
    for (v, c, sd_) in list(D):
        if not v.startswith(control):
            continue
        suffix = v[len(control):]
        base = D.get((control_of + suffix, c, sd_))
        if base:
            ds.append(D[(v, c, sd_)]["acc"] - base["acc"])
    if not ds:
        print(f"    no paired {control} / {control_of} runs found"); return None
    a = np.array(ds)
    print(f"    n={len(a)}  mean {a.mean():+.3f}  mean|d| {np.abs(a).mean():.3f}  "
          f"sd {a.std(ddof=1):.3f}  range [{a.min():+.3f}, {a.max():+.3f}]")
    print(f"    => TREAT {np.abs(a).mean():.3f} AS THE NOISE FLOOR. Report no effect "
          f"smaller than this as real.")
    return float(np.abs(a).mean())


def check_power(D, floor):
    print("\n[4] POWER -- what is the smallest effect this design could detect?")
    byv = defaultdict(dict)
    for (v, c, s), d in D.items():
        byv[v][(c, s)] = d["acc"]
    vs = sorted(byv)
    for i, a in enumerate(vs):
        for b in vs[i+1:]:
            keys = set(byv[a]) & set(byv[b])
            if len(keys) < 3:
                continue
            d = np.array([byv[a][k] - byv[b][k] for k in sorted(keys)])
            sd = d.std(ddof=1); n = len(d)
            mde = 2.8 * sd / np.sqrt(n)      # ~80% power, two-sided alpha .05
            verdict = "DETECTABLE" if abs(d.mean()) > mde else "UNDERPOWERED -> report 'unmeasured', NOT 'null'"
            print(f"    {a:16s} - {b:16s} n={n}  mean {d.mean():+.3f}  sd {sd:.3f}  "
                  f"MDE {mde:.3f}  {verdict}")


def check_ceiling(D):
    print("\n[5] CEILING -- can converged arms still differ, or is the task too easy?")
    solved = [k for k, d in D.items() if d["acc"] > 0.95]
    print(f"    {len(solved)}/{len(D)} arm-runs exceed 0.95")
    if len(solved) > 0.4 * max(len(D), 1):
        print("    => a large fraction sit at the ceiling. Conditioning on convergence "
              "will SELECT INTO it and show ~0 by construction. Make the task harder "
              "(more aliasing / bigger map) before comparing converged arms.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--length", type=int, default=512)
    ap.add_argument("--control", default=None,
                    help="an arm PROVABLY function-identical to --control-of; its gap "
                         "measures the noise floor")
    ap.add_argument("--control-of", default="Vanilla")
    args = ap.parse_args()

    print(f"=== experiment audit: {args.runs_dir} (T={args.length}) ===")
    D = _load(args.runs_dir, args.length)
    if not D:
        print("no (json, pt) pairs found"); return
    print(f"loaded {len(D)} arm-runs")
    check_convergence(D)
    check_loss_readout(D)
    floor = check_noise_floor(D, args.control, args.control_of)
    check_power(D, floor)
    check_ceiling(D)
    print("\nRead [1] first: if arms are not converged, nothing below is interpretable.")


if __name__ == "__main__":
    main()
