"""PAPERTASK_PREREG.md analysis. Written before the rerun finished.

Primary readout: FLOOR-NORMALISED accuracy `(acc - floor) / (1 - floor)` at ext-s l=2048,
floors measured per condition in `PAPER_TASK_FLOORS.json`. At pe=0.8 four fifths of the raw
scale is floor, so raw accuracy is reported beside it and is not the primary.

    python3 -m mapformer.analyze_papertask
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from mapformer.ckpt_guard import REPO, load_checkpoint
from mapformer.stats_guard import paired, rule9, table

ARMS = ["Vanilla", "VanillaEM_P0", "MapPoPE-Flat"]
SEEDS = range(8)
RUNS = REPO / "runs/paper_task_rerun"
IID = "IID  l=128 g=64  pe=0.5"


def final_loss(arm, s):
    """Checkpoint first; fall back to the kept training log (the point of the rerun)."""
    pt = RUNS / f"{arm}_s{s}" / f"{arm}.pt"
    try:
        ck = load_checkpoint(pt)
        if ck.losses:
            return float(ck.losses[-1])
    except Exception:
        pass
    log = RUNS / "logs" / f"train_{arm}_s{s}.log"
    if log.exists():
        m = re.findall(r"loss[=: ]+([0-9.]+)", log.read_text())
        if m:
            return float(m[-1])
    return float("nan")


def main():
    acc_path = REPO / "PAPER_OOD_RERUN.json"
    if not acc_path.exists():
        raise FileNotFoundError(f"{acc_path} -- has the rerun's eval finished?")
    acc = json.load(open(acc_path))
    fl = json.load(open(REPO / "PAPER_TASK_FLOORS.json"))
    conds = list(acc[ARMS[0]])

    def norm(arm, c):
        f = fl[c]["always_blank"]
        return {i: (float(v) - f) / (1 - f) for i, v in enumerate(acc[arm][c])}

    L = ["# PAPERTASK rerun report (PAPERTASK_PREREG.md)\n",
         "50 epochs cosine, logs kept. Raw accuracy, then the floor-normalised primary.\n",
         "| condition | floor | " + " | ".join(ARMS) + " | " + " | ".join(a + " (norm)" for a in ARMS) + " |",
         "|---" * (2 + 2 * len(ARMS)) + "|"]
    for c in conds:
        raw = [np.mean(acc[a][c]) for a in ARMS]
        nrm = [np.mean(list(norm(a, c).values())) for a in ARMS]
        L.append(f"| {c} | {fl[c]['always_blank']:.3f} | "
                 + " | ".join(f"{v:.3f}" for v in raw) + " | "
                 + " | ".join(f"**{v:.3f}**" for v in nrm) + " |")

    # convergence check FIRST -- the prereg says no verdict is read until it passes
    iid = {a: np.mean(acc[a][IID]) for a in ARMS}
    ok = all(v >= 0.99 for v in iid.values())
    L += ["", "## Convergence check (before any verdict)\n",
          "| arm | IID | >= 0.99 |", "|---|---|---|"]
    for a in ARMS:
        L.append(f"| {a} | {iid[a]:.3f} | {'yes' if iid[a] >= 0.99 else '**NO**'} |")
    L.append(f"\n-> **{'PASS' if ok else 'FAIL -- P1/P2 not interpreted; this is a budget-curve point only'}** "
             f"(the 16-epoch batch gave WM 0.969)")

    ext = [c for c in conds if "2048" in c or "1024" in c or "l=512" in c]
    cs = []
    for c in ext:
        cs.append(paired(norm("VanillaEM_P0", c), norm("Vanilla", c), f"EM - WM norm {c[:20]}"))
        cs.append(paired(norm("MapPoPE-Flat", c), norm("VanillaEM_P0", c), f"PoPE - EM norm {c[:20]}"))
    L += ["", "## Contrasts, floor-normalised (paired by seed, n=8)\n", table(cs)]

    losses = {(a, s): final_loss(a, s) for a in ARMS for s in SEEDS}
    tgt = [c for c in conds if "2048" in c][0]
    flat_acc = [acc[a][tgt][s] for a in ARMS for s in SEEDS]
    flat_loss = [losses[(a, s)] for a in ARMS for s in SEEDS]
    L += ["", "## Rule 9 -- the check the deleted logs made impossible\n"]
    if np.isfinite(flat_loss).all():
        r9 = rule9(flat_acc, flat_loss)
        L.append(str(r9))
        L.append(f"\nMean final loss: " + ", ".join(f"{a} {np.mean([losses[(a,s)] for s in SEEDS]):.4f}" for a in ARMS))
    else:
        L.append("final losses unavailable -- neither the checkpoints nor the logs carry them")

    c2048 = cs[[i for i, c in enumerate(cs) if "2048" in c.label][0]]
    p1 = c2048.delta >= 0.20 and c2048.verdict == "DETECTABLE"
    cp = cs[[i for i, c in enumerate(cs) if "2048" in c.label and c.label.startswith("PoPE")][0]]
    L += ["", "## Registered verdicts\n",
          f"- **P1 (the effect is real)**: EM - WM norm at l=2048 = {c2048.delta:+.3f} "
          f"(MDE {c2048.mde:.3f}; needs >= +0.20 AND detectable) -> "
          f"**{'CONFIRMED' if p1 and ok else 'NOT CONFIRMED' if ok else 'NOT READ (convergence failed)'}**",
          f"- **P3 (per-pair counterexample)**: PoPE - EM norm at l=2048 = {cp.delta:+.3f} "
          f"(MDE {cp.mde:.3f}) -> **{'now DETECTABLE -- the counterexample is established' if cp.verdict == 'DETECTABLE' else 'still unmeasured, as predicted'}**",
          f"- **P4 (falsifier)**: convergence {'passed' if ok else 'FAILED'} and the l=2048 gap is "
          f"{'below' if abs(c2048.delta) < c2048.mde else 'above'} its MDE -> "
          f"**{'FIRES -- the extended-length advantage was a budget artifact; retract it' if ok and abs(c2048.delta) < c2048.mde else 'does not fire'}**"]
    txt = "\n".join(L)
    print(txt)
    (RUNS / "REPORT.md").write_text(txt + "\n")


if __name__ == "__main__":
    main()
