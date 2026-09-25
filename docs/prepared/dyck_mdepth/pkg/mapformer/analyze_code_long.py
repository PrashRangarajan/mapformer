"""MDEs for the out-of-distribution code readout (CODE_PREREG.md Amendment 1).

bpc: LOWER is better, so a negative contrast is an improvement and the sign
count reports how many seeds the first arm WINS on. Accuracy: higher is better.
Every contrast prints its MDE; one that does not clear is "unmeasured".
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

_REPO = os.path.dirname(os.path.abspath(__file__))
ARMS = ["RoPE", "PoPE-Flat", "Vanilla", "MapPoPE-Flat"]
NICE = {"RoPE": "RoPE (index/RoPE)", "PoPE-Flat": "PoPE (index/PoPE)",
        "Vanilla": "MapWM (path/RoPE)", "MapPoPE-Flat": "MapPoPE (path/PoPE)"}


def mde(d):
    d = np.asarray(d, float)
    return 2.8 * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")


def contrast(name, a, b, lower_better, L):
    common = sorted(set(a) & set(b))
    if len(common) < 1:
        L.append(f"- {name}: no paired seeds"); return
    d = np.array([a[s] - b[s] for s in common])
    m, M = d.mean(), mde(d)
    wins = int((d < 0).sum()) if lower_better else int((d > 0).sum())
    det = len(d) > 1 and abs(m) > M
    line = (f"- **{name}**: {m:+.4f} (MDE {M:.4f}, better {wins}/{len(d)}) "
             + ("**DETECTABLE**" if det else "unmeasured"))
    print(line); L.append(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--long", default=os.path.join(_REPO, "CODE_LONG.json"))
    ap.add_argument("--runs-dir", default=os.path.join(_REPO, "runs", "code"))
    ap.add_argument("--out", default=os.path.join(_REPO, "CODE_RESULTS_OOD.md"))
    args = ap.parse_args()
    D = json.load(open(args.long))
    floors = D.get("_floor_by_distance", {})
    L = []

    def say(s=""):
        print(s); L.append(s)

    bpc, acc, loss = {}, {}, {}
    for key, rec in D.items():
        if key.startswith("_"):
            continue
        arm, _, sd = key.rpartition("_s")
        if arm not in ARMS:
            continue
        bpc.setdefault(arm, {})[int(sd)] = rec["bpc_by_position"]
        acc.setdefault(arm, {})[int(sd)] = {k: v["acc"] for k, v in rec["acc_by_distance"].items()}
        j = Path(args.runs_dir) / f"{arm}_s{sd}.json"
        if j.exists():
            c = json.load(open(j))["curve"]
            loss.setdefault(arm, {})[int(sd)] = float(np.mean([e["train_bpc"] for e in c[-3:]]))

    seeds = sorted({s for a in bpc for s in bpc[a]})
    buckets = list(next(iter(next(iter(bpc.values())).values())).keys())
    dists = sorted({k for a in acc for s in acc[a] for k in acc[a][s]},
                   key=lambda x: int(x.split("-")[0].rstrip("+")))

    say("# Code, beyond the training context: MDEs\n")
    say(f"Trained at seq 512, evaluated at 2048. Seeds {seeds}. "
        f"Pre-registration: `CODE_PREREG.md` Amendment 1.\n")

    say("## O-A  val bpc by position (lower better), seed means\n")
    say("| arm | " + " | ".join(buckets) + " |")
    say("|" + "---|" * (len(buckets) + 1))
    for a in ARMS:
        if a not in bpc:
            continue
        say(f"| {NICE[a]} | " + " | ".join(
            f"{np.mean([bpc[a][s][b] for s in bpc[a]]):.4f}" for b in buckets) + " |")
    say("")
    for b in buckets:
        say(f"**Position {b}**")
        g = {a: {s: bpc[a][s][b] for s in bpc[a]} for a in ARMS if a in bpc}
        contrast("O2  MapPoPE - PoPE   (composition: beats its ENCODING component)",
                 g["MapPoPE-Flat"], g["PoPE-Flat"], True, L)
        contrast("O2  MapPoPE - MapWM  (composition: beats its POSITION component)",
                 g["MapPoPE-Flat"], g["Vanilla"], True, L)
        contrast("O1  MapWM - RoPE     (path integration on the RoPE row)",
                 g["Vanilla"], g["RoPE"], True, L)
        contrast("    PoPE - RoPE      (encoding on the index row)",
                 g["PoPE-Flat"], g["RoPE"], True, L)
        say("")

    say("## O-B  closer accuracy by bracket distance (higher better), seed means\n")
    say("| arm | " + " | ".join(dists) + " |")
    say("|" + "---|" * (len(dists) + 1))
    for a in ARMS:
        if a not in acc:
            continue
        say(f"| {NICE[a]} | " + " | ".join(
            f"{np.mean([acc[a][s][d] for s in acc[a] if d in acc[a][s]]):.3f}"
            if any(d in acc[a][s] for s in acc[a]) else "--" for d in dists) + " |")
    say("| *no-stack floor* | " + " | ".join(
        f"*{floors.get(d, float('nan')):.3f}*" for d in dists) + " |")
    say("")
    for d in dists:
        if floors.get(d, 0) > 0.95:
            say(f"**Distance {d}** -- floor {floors[d]:.3f} > 0.95, registered uninformative\n")
            continue
        say(f"**Distance {d}** (floor {floors.get(d, float('nan')):.3f})")
        g = {a: {s: acc[a][s][d] for s in acc[a] if d in acc[a][s]} for a in ARMS if a in acc}
        contrast("MapPoPE - PoPE", g["MapPoPE-Flat"], g["PoPE-Flat"], False, L)
        contrast("MapPoPE - MapWM", g["MapPoPE-Flat"], g["Vanilla"], False, L)
        contrast("MapWM - RoPE", g["Vanilla"], g["RoPE"], False, L)
        say("")

    say("## Rule 9 and loss overlap\n")
    far = buckets[-1]
    xs = [loss[a][s] for a in ARMS if a in loss for s in loss[a]]
    ys = [bpc[a][s][far] for a in ARMS if a in loss for s in loss[a]]
    if len(xs) > 2:
        r = float(np.corrcoef(xs, ys)[0, 1])
        say(f"- r(final train bpc, OOD bpc at {far}) = **{r:+.3f}** over {len(xs)} runs.")
        say("  - " + ("**|r| > 0.98: the OOD metric is the training loss in disguise.**"
                      if abs(r) > 0.98 else
                      "Below 0.98, so the OOD effect is not simply a convergence gap."))
    rng = {a: (min(loss[a].values()), max(loss[a].values())) for a in ARMS if a in loss}
    say("- final train bpc ranges: " + "; ".join(
        f"{NICE[a]} [{lo:.4f}, {hi:.4f}]" for a, (lo, hi) in rng.items()))
    if "MapPoPE-Flat" in rng and "PoPE-Flat" in rng:
        x, y = rng["MapPoPE-Flat"], rng["PoPE-Flat"]
        ov = not (x[1] < y[0] or y[1] < x[0])
        say(f"- MapPoPE vs PoPE training losses overlap: **{ov}**"
            + ("" if ov else " -- no loss-matched residual can be quoted (this "
                             "project's own rule)."))
    say("")
    Path(args.out).write_text("\n".join(L) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
