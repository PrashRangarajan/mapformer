"""SEARCH_PREREG.md S3 analysis: fixed-k and curriculum recency. Written before any S3 checkpoint existed.

    python3 -m mapformer.analyze_search
"""
from __future__ import annotations

import json
import math

import numpy as np
import torch

from mapformer.ckpt_guard import REPO, load_checkpoint
from mapformer.probe_anatomy import anatomy
from mapformer.probe_rewind import delta_table, load_model, rewind_slope
from mapformer.stats_guard import paired, table

RUNS = REPO / "runs/search"
ARMS = {"P0_fix64": ("VanillaEM_P0_r4", 64), "P0_fix16": ("VanillaEM_P0_r4", 16),
        "WM_fix64": ("Vanilla_r4", 64), "P0_cur": ("VanillaEM_P0_r4", None)}
SEEDS = range(8)


def ckpt(arm, s):
    v = ARMS[arm][0]
    pt = RUNS / f"{arm}_s{s}" / f"{v}_recency.pt"
    js = RUNS / f"{arm}_s{s}" / f"{v}_recency.json"
    if not pt.exists() or not js.exists():
        raise FileNotFoundError(f"missing {pt} / {js}; present: {sorted(p.name for p in RUNS.iterdir())}")
    return pt, js


def epochs_to(losses, thr=0.5):
    for i, l in enumerate(losses):
        if l < thr:
            return i + 1
    return None


@torch.no_grad()
def wrapped_score(m, env, k):
    """sum_i a_i cos(wrap(omega_i (Delta(q_k)_i + (k-1) s_i))) / sum_i a_i, single-p0 only.
    1 = exact rewind modulo each block's period (assumes filler Delta = 0 and symbols share s)."""
    p0 = m.p0_pos.detach().double()                           # (H, dh)
    H, dh = p0.shape
    a = p0.view(H, dh // 2, 2).pow(2).sum(-1)                  # (H, nb)
    D = delta_table(m).view(-1, H, dh // 2)
    s = D[env.sym_offset:env.sym_offset + env.n_symbols].mean(0)
    om = m.path_integrator.omega.detach().double()
    e = om * (D[env.query_offset + k - 1] + (k - 1) * s)
    return float((a * torch.cos(e)).sum() / a.sum())


def linear_ratio(m, env, k):
    D = delta_table(m)
    s = D[env.sym_offset:env.sym_offset + env.n_symbols].mean(0)
    return float(D[env.query_offset + k - 1] @ s / (s @ s))


def main():
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    rows = {}
    for arm, (v, K) in ARMS.items():
        for s in SEEDS:
            pt, js = ckpt(arm, s)
            res = json.load(open(js))
            ck = load_checkpoint(pt)
            r = dict(arm=arm, seed=s, acc=res["1024"]["acc"], acc2048=res["2048"]["acc"],
                     final_loss=float(ck.losses[-1]), ep05=epochs_to(ck.losses))
            if v.startswith("VanillaEM"):
                over = dict(k_fixed=K) if K else {}
                m, _, env = load_model(pt, v, **over)
                # table readouts first, on CPU: anatomy() moves the model to `dev`
                if K:
                    r.update(wrapped=wrapped_score(m, env, K), lin_ratio=linear_ratio(m, env, K),
                             lin_target=-(K - 1))
                    an = anatomy(m, env, s, device=dev)
                    S, S0 = np.array(an["sel"][K - 1]), np.array(an["sel0"][K - 1])
                    r.update(sel_max=float(S.max()), sel0_max=float(S0.max()),
                             diff=float(S.max() - S0.max()))
                else:
                    r["slope"] = rewind_slope(delta_table(m), env)
                    r["wrapped_k"] = [wrapped_score(m, env, k) for k in range(1, 65)]
                    an = anatomy(m, env, s, device=dev)
                    r["acc_k"] = an["acc"]
                    S, S0 = np.array(an["sel"]), np.array(an["sel0"])
                    r["diff_k"] = (S.max(1) - S0.max(1)).tolist()
            rows[(arm, s)] = r
            print(arm, s, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in r.items()
                           if k not in ("wrapped_k", "acc_k", "diff_k")}, flush=True)

    L = ["# S3 report (SEARCH_PREREG.md)\n", "## Per-arm\n",
         "| arm | acc T=1024 mean | >= 0.9 | >= 0.95 | acc T=2048 | final loss | epochs to loss<0.5 (median, reached) |",
         "|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        A = np.array([rows[(arm, s)]["acc"] for s in SEEDS])
        e = [rows[(arm, s)]["ep05"] for s in SEEDS]
        er = [x for x in e if x is not None]
        L.append(f"| {arm} | {A.mean():.3f} +/- {A.std(ddof=1):.3f} | {(A >= 0.9).sum()}/8 | {(A >= 0.95).sum()}/8 | "
                 f"{np.mean([rows[(arm, s)]['acc2048'] for s in SEEDS]):.3f} | "
                 f"{np.mean([rows[(arm, s)]['final_loss'] for s in SEEDS]):.3f} | "
                 f"{np.median(er) if er else '-'} ({len(er)}/8) |")
    L += ["", "## Fixed-k mechanism readouts (single query token)\n",
          "| arm | seed | acc | sel_max | sel0_max | sel - sel0 | wrapped score | linear ratio (target) |",
          "|---|---|---|---|---|---|---|---|"]
    for arm in ("P0_fix16", "P0_fix64"):
        for s in SEEDS:
            r = rows[(arm, s)]
            L.append(f"| {arm} | {s} | {r['acc']:.3f} | {r['sel_max']:.3f} | {r['sel0_max']:.3f} | "
                     f"{r['diff']:+.3f} | {r['wrapped']:+.3f} | {r['lin_ratio']:+.2f} ({r['lin_target']}) |")

    fix = lambda arm, thr: sum(rows[(arm, s)]["acc"] >= thr for s in SEEDS)
    p1 = fix("WM_fix64", 0.95) >= 6
    p2a, p2b = fix("P0_fix16", 0.9) >= 6, fix("P0_fix64", 0.9) <= 2
    L += ["", "## Registered verdicts\n",
          f"- **S3-P1** WM_fix64 >= 0.95 on {fix('WM_fix64', 0.95)}/8 (>= 6 needed): **{'MET' if p1 else 'NOT MET -- fixed-k arms not interpreted'}**",
          f"- **S3-P2** P0_fix16 >= 0.9 on {fix('P0_fix16', 0.9)}/8 (>= 6), P0_fix64 >= 0.9 on {fix('P0_fix64', 0.9)}/8 (<= 2): "
          f"**{'CONFIRMED' if p2a and p2b else 'REFUTED' if not p2b else 'SPLIT'}**"]
    det = (RUNS / "DETERMINISM.txt").read_text() if (RUNS / "DETERMINISM.txt").exists() else "MISSING"
    L += ["", "## Determinism re-check (licenses reuse of stored P0)\n", "```", det.strip(), "```"]
    stored = {}
    for s in SEEDS:
        stored[s] = json.load(open(REPO / f"runs/dof/recency/VanillaEM_P0_r4_s{s}/VanillaEM_P0_r4_recency.json"))["1024"]["acc"]
    cur = {s: rows[("P0_cur", s)]["acc"] for s in SEEDS}
    c = paired(cur, stored, "P0_cur - P0 (stored)")
    sl = [rows[("P0_cur", s)]["slope"] for s in SEEDS]
    refuted = sum(cur[s] >= 0.95 for s in SEEDS) >= 6 and np.median(sl) <= -0.5
    L += ["", table([c]),
          f"\nP0_cur linear slopes: {[round(x, 3) for x in sl]}",
          f"- **S3-P3** (curriculum does NOT close the gap): **{'REFUTED' if refuted else 'MET' if (c.verdict != 'DETECTABLE' or np.mean(list(cur.values())) < 0.9) else 'SPLIT'}**"
          + ("" if "DETERMINISM" in det and "DIFFER" not in det.upper() else "  (reuse NOT licensed -- see determinism block)")]
    txt = "\n".join(L)
    print(txt)
    (RUNS / "S3_report.md").write_text(txt + "\n")
    json.dump({f"{a}_s{s}": r for (a, s), r in rows.items()}, open(REPO / "_SEARCH_S3.json", "w"))


if __name__ == "__main__":
    main()
