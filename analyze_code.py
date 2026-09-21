"""Analysis for the code-modelling batch. Registered in CODE_PREREG.md.

Reports, for every contrast, the paired mean, the MDE (2.8*sd/sqrt(n)), the sign
count, and whether it clears. A contrast that does not clear is printed as
"unmeasured" with its MDE, never as a null.

Also runs the two checks that have overturned readings in this project before:
  rule 9  -- is accuracy just the training loss? r(final train loss, accuracy).
  overlap -- loss-matching requires overlapping losses. If the arms' final
             training losses do not overlap, no residual is quoted.
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
PRIMARY = "d5-8/x33-128"


def mde(d):
    d = np.asarray(d, dtype=float)
    return 2.8 * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")


def contrast(name, a, b, lines):
    """a - b, paired over seeds."""
    common = sorted(set(a) & set(b))
    if not common:
        lines.append(f"- {name}: no paired seeds")
        return
    d = np.array([a[s] - b[s] for s in common])
    m, M = d.mean(), mde(d)
    pos = int((d > 0).sum())
    verdict = "DETECTABLE" if len(d) > 1 and abs(m) > M else "unmeasured"
    lines.append(f"- **{name}**: {m:+.4f} (MDE {M:.4f}, {pos}/{len(d)} positive) "
                 f"{'**' + verdict + '**' if verdict == 'DETECTABLE' else verdict}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delims", default=os.path.join(_REPO, "CODE_DELIMS.json"))
    ap.add_argument("--runs-dir", default=os.path.join(_REPO, "runs", "code"))
    ap.add_argument("--which", default="best")
    ap.add_argument("--out", default=os.path.join(_REPO, "CODE_RESULTS.md"))
    args = ap.parse_args()

    D = json.load(open(args.delims))
    gates = json.load(open(os.path.join(_REPO, "CODE_GATES.json")))
    L = []

    def say(s=""):
        print(s)
        L.append(s)

    # arm -> seed -> record
    by = {a: {} for a in ARMS}
    loss = {a: {} for a in ARMS}
    for key, rec in D.items():
        base, _, which = key.rpartition(".")
        if which != args.which:
            continue
        arm, _, tag = base.rpartition("_s")
        if arm not in by:
            continue
        by[arm][int(tag)] = rec
        j = Path(args.runs_dir) / f"{base}.json"
        if j.exists():
            c = json.load(open(j))["curve"]
            loss[arm][int(tag)] = float(np.mean([e["train_bpc"] for e in c[-3:]]))

    n_seeds = sorted({s for a in ARMS for s in by[a]})
    say(f"# Code modelling: does MapPoPE's Dyck-2 win survive on real Python?\n")
    say(f"Pre-registration: `CODE_PREREG.md`. Gates: `CODE_GATES.md`. "
        f"Checkpoint: **{args.which}-validation**. Seeds present: {n_seeds}.\n")
    say(f"No-stack floor overall **{gates['ngram_overall']:.3f}** "
        f"(order-{gates['ngram_order']} n-gram), majority class "
        f"**{gates['majority']:.3f}**. Overall numbers are floor-dominated and "
        f"are reported only for the registered F3 check.\n")

    # ---- overall table ----
    say("## Overall (registered as uninformative -- P1/F3 only)\n")
    say("| arm | val bpc | closer acc (all) | n seeds |")
    say("|---|---|---|---|")
    for a in ARMS:
        if not by[a]:
            continue
        b = np.mean([r["val_bpc"] for r in by[a].values()])
        c = np.mean([r["overall_closer_acc"] for r in by[a].values()])
        say(f"| {NICE[a]} | {b:.4f} | {c:.3f} | {len(by[a])} |")
    say("")

    # ---- per-cell table ----
    cells = sorted({c for a in ARMS for r in by[a].values() for c in r["cells"]})
    inform = [c for c in cells
              if all(r["cells"][c]["informative"] for a in ARMS for r in by[a].values()
                     if c in r["cells"])]
    say("## Closer-identity accuracy by stratum\n")
    say("Cells whose measured no-stack floor exceeds 0.95 are marked `ceil` and "
        "were registered as uninformative BEFORE any run.\n")
    hdr = "| arm | " + " | ".join(cells) + " |"
    say(hdr)
    say("|" + "---|" * (len(cells) + 1))
    say("| *no-stack floor* | " + " | ".join(
        f"{gates['floors'].get(c, float('nan')):.3f}"
        + ("" if c in inform else " `ceil`") for c in cells) + " |")
    for a in ARMS:
        if not by[a]:
            continue
        row = []
        for c in cells:
            vals = [r["cells"][c]["acc"] for r in by[a].values() if c in r["cells"]]
            row.append(f"{np.mean(vals):.3f}" if vals else "--")
        say(f"| {NICE[a]} | " + " | ".join(row) + " |")
    say("")

    # ---- contrasts ----
    say(f"## Contrasts at the PRIMARY cell `{PRIMARY}` "
        f"(floor {gates['floors'].get(PRIMARY, float('nan')):.3f})\n")

    def cell(a, c):
        return {s: r["cells"][c]["acc"] for s, r in by[a].items() if c in r["cells"]}

    p = {a: cell(a, PRIMARY) for a in ARMS}
    contrast("P2 PRIMARY  MapPoPE - PoPE", p["MapPoPE-Flat"], p["PoPE-Flat"], L)
    contrast("MapPoPE - MapWM (encoding, path row)", p["MapPoPE-Flat"], p["Vanilla"], L)
    contrast("MapWM - RoPE (position, index encoding)", p["Vanilla"], p["RoPE"], L)
    contrast("PoPE - RoPE (encoding, index row)", p["PoPE-Flat"], p["RoPE"], L)
    say("")

    say("## The same contrasts at every informative cell\n")
    for c in cells:
        tag = "" if c in inform else "  *(ceiling cell, registered uninformative)*"
        say(f"**{c}** -- floor {gates['floors'].get(c, float('nan')):.3f}{tag}")
        q = {a: cell(a, c) for a in ARMS}
        contrast("MapPoPE - PoPE", q["MapPoPE-Flat"], q["PoPE-Flat"], L)
        contrast("MapPoPE - MapWM", q["MapPoPE-Flat"], q["Vanilla"], L)
        say("")

    # ---- rule 9 + overlap ----
    say("## Rule 9 and loss overlap\n")
    xs, ys = [], []
    for a in ARMS:
        for s in by[a]:
            if s in loss[a] and PRIMARY in by[a][s]["cells"]:
                xs.append(loss[a][s])
                ys.append(by[a][s]["cells"][PRIMARY]["acc"])
    if len(xs) > 2:
        r = float(np.corrcoef(xs, ys)[0, 1])
        say(f"- r(final train bpc, primary-cell accuracy) = **{r:+.3f}** over {len(xs)} runs.")
        if abs(r) > 0.98:
            say("  - **|r| > 0.98: F4 FIRES.** The held-out metric carries no "
                "information the training loss does not; report this as a "
                "convergence gap, not an effect.")
    rng = {a: (min(loss[a].values()), max(loss[a].values())) for a in ARMS if loss[a]}
    say("- final train bpc ranges: " + ", ".join(
        f"{NICE[a]} [{lo:.4f}, {hi:.4f}]" for a, (lo, hi) in rng.items()))
    if "MapPoPE-Flat" in rng and "PoPE-Flat" in rng:
        a1, a2 = rng["MapPoPE-Flat"], rng["PoPE-Flat"]
        ov = not (a1[1] < a2[0] or a2[1] < a1[0])
        say(f"- MapPoPE vs PoPE losses overlap: **{ov}**. "
            + ("" if ov else "Loss-matching requires overlapping losses, so no "
                             "loss-matched residual is quoted for that pair."))
    say("")
    Path(args.out).write_text("\n".join(L) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
