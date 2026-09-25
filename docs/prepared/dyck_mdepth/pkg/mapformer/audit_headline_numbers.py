"""Re-derive every headline number from raw per-seed data and diff against the docs.

Why this exists: six problems were found in this project's own experimental
apparatus in a single day -- a GPU dispatch bias, an n_layers capacity mismatch,
an evaluator silently dropping seeds, a position/frequency confound, an
out-of-range token id, and `wait` masking total failure -- plus three false
negatives caused by a fixed training budget. At that rate it is not safe to
assume the numbers in BASELINE_TABLE.md are what they claim to be.

This recomputes each claim from the per-seed JSON that the evaluators wrote, and
flags any disagreement with the markdown. It reads only saved artifacts, so it
runs on CPU and cannot disturb anything training.

A PASS here does NOT certify the experiment was well designed -- only that the
reported number matches the data it was computed from. Design errors (the kind
that voided 48 files) are invisible to it.
"""
import json
import re
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent
TOL = 0.0015          # markdown is printed to 3 decimals


def load(name):
    p = _REPO / name
    return json.load(open(p)) if p.exists() else None


def chk(label, claimed, recomputed, out):
    if recomputed is None:
        out.append(("SKIP", label, claimed, None)); return
    ok = abs(claimed - recomputed) <= TOL
    out.append(("PASS" if ok else "**MISMATCH**", label, claimed, recomputed))


def main():
    out = []

    # ---- A: paper task 2x2, n=8 ----
    d = load("INDEX_BASELINE_PAPER_TASK_n8.json")
    if d:
        md = (_REPO / "INDEX_BASELINE_PAPER_TASK_n8.md").read_text()
        for v in ("Vanilla", "MapPoPE-Flat", "RoPE", "PlainFlat", "PoPE-Flat"):
            m = re.search(rf"^\| {re.escape(v)} \| ([\d.]+) ± [\d.]+ \| ([\d.]+)", md, re.M)
            if not m: continue
            rec = d.get(v, {}).get("fresh")
            rec = float(np.mean(rec)) if isinstance(rec, list) else None
            chk(f"A paper-task fresh-map {v}", float(m.group(2)), rec, out)

    # ---- H: MiniGrid 8-cell, n=8 ----
    d = load("MINIGRID_FULL_2X2X2.json")
    if d:
        md = (_REPO / "MINIGRID_FULL_2X2X2.md").read_text()
        for v, disp in [("Vanilla","MapWM-Flat"),("Hourglass_k2","MapWM-Hier"),
                        ("MapPoPE-Flat","MapPoPE-Flat"),("MapPoPE-Hier","MapPoPE-Hier"),
                        ("RoPE","RoPE-Flat"),("PlainHourglass","RoPE-Hier"),
                        ("PoPE-Flat","PoPE-Flat"),("PoPE-Hier","PoPE-Hier")]:
            m = re.search(rf"^\| {re.escape(disp)} [^|]*\|[^|]*\| ([\d.]+) ± [\d.]+ \| ([\d.]+)", md, re.M)
            if not m or v not in d["acc"]: continue
            rec = float(np.mean(d["acc"][v]["1024"]["per_seed"]))
            chk(f"H MiniGrid T=1024 {disp}", float(m.group(2)), rec, out)
            n = len(d["acc"][v]["1024"]["per_seed"])
            if n != 8:
                out.append(("**MISMATCH**", f"H {disp} seed count", 8, n))

    # ---- I: knob sweep, n=8 ----
    d = load("KNOB_SWEEP_n8.json")
    if d:
        md = (_REPO / "KNOB_SWEEP_n8.md").read_text()
        for cond, v in d.items():
            m = re.search(rf"^\| {re.escape(cond)} \|[^|]*\|[^|]*\|[^|]*\| \*\*([+-][\d.]+)\*\*", md, re.M)
            if not m: continue
            rec = float(np.mean(v["vanilla"]) - np.mean(v["rope"]))
            chk(f"I knob {cond} position effect", float(m.group(1)), rec, out)
            if len(v["vanilla"]) != 8:
                out.append(("**MISMATCH**", f"I {cond} seed count", 8, len(v["vanilla"])))

    # ---- BASELINE_TABLE headline block ----
    bt = (_REPO / "BASELINE_TABLE.md").read_text()
    d = load("MINIGRID_FULL_2X2X2.json")
    if d:
        m = lambda k, T: float(np.mean(d["acc"][k][T]["per_seed"]))
        for T in ("512", "1024"):
            enc = np.mean([m("MapPoPE-Flat",T)-m("Vanilla",T), m("MapPoPE-Hier",T)-m("Hourglass_k2",T),
                           m("PoPE-Flat",T)-m("RoPE",T), m("PoPE-Hier",T)-m("PlainHourglass",T)])
            pos = np.mean([m("Vanilla",T)-m("RoPE",T), m("Hourglass_k2",T)-m("PlainHourglass",T),
                           m("MapPoPE-Flat",T)-m("PoPE-Flat",T), m("MapPoPE-Hier",T)-m("PoPE-Hier",T)])
            hie = np.mean([m("Hourglass_k2",T)-m("Vanilla",T), m("MapPoPE-Hier",T)-m("MapPoPE-Flat",T),
                           m("PlainHourglass",T)-m("RoPE",T), m("PoPE-Hier",T)-m("PoPE-Flat",T)])
            row = re.search(rf"^\| \*\*MiniGrid DoorKey-16x16\*\* T={T} \(n=8\) \| \*\*([+−\-][\d.]+)\*\* \| ([+−\-][\d.]+) \| \*?\*?([+−\-][\d.]+)", bt, re.M)
            if row:
                f = lambda s: float(s.replace("−", "-"))
                chk(f"TABLE headline MiniGrid T={T} encoding", f(row.group(1)), float(enc), out)
                chk(f"TABLE headline MiniGrid T={T} hierarchy", f(row.group(2)), float(hie), out)
                chk(f"TABLE headline MiniGrid T={T} position", f(row.group(3)), float(pos), out)

    # ---- report ----
    bad = [r for r in out if r[0].startswith("**")]
    lines = ["# Audit: do the reported numbers match the data?", "",
             "Recomputed from the per-seed JSON each evaluator wrote, and diffed "
             "against the markdown. Tolerance 0.0015 (docs print 3 decimals).", "",
             f"**{len(out)-len(bad)} pass, {len(bad)} mismatch, "
             f"{sum(1 for r in out if r[0]=='SKIP')} skipped.**", "",
             "| check | claimed | recomputed | |", "|---|---|---|---|"]
    for st, lab, c, r in out:
        rr = "—" if r is None else f"{r:.4f}"
        cc = f"{c:.4f}" if isinstance(c, float) else str(c)
        lines.append(f"| {lab} | {cc} | {rr} | {st} |")
    lines += ["", "## What a PASS does and does not mean", "",
              "It means the printed number is what the saved per-seed data "
              "computes to. It does NOT mean the experiment was well designed: "
              "every one of the 48 files in `archive/void/` would have passed "
              "this check on the day it was written. Design errors — a shortcut "
              "in the task, an undertrained arm, a confounded factor — are "
              "invisible here and need the gates and ablations instead."]
    (_REPO / "AUDIT_HEADLINE.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
