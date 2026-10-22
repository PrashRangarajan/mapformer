"""Plan Fig. 11 (docs/paper/expansion/SYNTHESIS_DRAFT2_PLAN.md 3.6): where the content-pair kernel peaks, and target
coherence. ANALYSIS (post hoc, eval-only) of registered batches; stored probe outputs, no model:
  (a) Index arms of PAPER2X2 (torus T = 128, n = 8): for every (head, action token, observation token) pair, the token
      offset D at which the content-pair kernel K_ao(D) peaks, argmax over ODD D in 1..15 (observation keys sit at odd
      offsets from an action query; ANALYSIS_SCRIPT_AUDIT mappope MINOR) -- mappope/kernel_argmax_odd_out.txt. Share of
      the 1,088 pooled pairs per offset.
  (b) Path arms of PAPER2X2 (T = 128, n = 8): target coherence C (the kernel's alignment with the target at the model's
      own phase) against C0 (the same with the content phase beta removed), all revisits, one dot per seed --
      mappope/kernel_probe_paper2x2.json; means as probe_stats_out.txt.
  (c) SCORE_RANK and RANK_MI arms (torus T = 1024; rank 2 n = 12, rank 4 n = 8): C0 by stratum, each stratum read on the
      head with most same-cell attention IN THAT STRATUM (HEAD_RULE=stratum; audit B2) --
      mappope/kernel_probe_score_rank_hstratum.json; means as probe_stats_hstratum_out.txt. Lines: arm means; dots: seeds.
  cd /home/prashr && PYTHONPATH=/home/prashr python3 mapformer/docs/paper/figs/fig_kernel_peak.py"""
import json
import os
import re

import numpy as np
from figstyle import plt, BLUE, ORANGE, AQUA, YELLOW, INK2, MUTED, save, write_out

REPO = "/home/prashr/mapformer"; HERE = os.path.dirname(os.path.abspath(__file__))
MP = f"{REPO}/docs/paper/expansion/scripts/mappope"
COL = ["share", "acc", "massCell", "massRT", "massOff1", "C", "C0", "mAct", "mObsX"]
out = []


def col(J, v, sn, c):
    return np.array([r[COL.index(c)] for r in J[str((v, sn))]], dtype=float)


fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.55), gridspec_kw={"width_ratios": [1.15, 0.85, 1.25]})

# (a) kernel argmax over odd offsets
ax = axs[0]; txt = open(f"{MP}/kernel_argmax_odd_out.txt").read()
ODD = list(range(1, 16, 2)); out.append("(a) argmax over odd D of K_ao(D), PAPER2X2 index arms, pooled (h, a, o) pairs")
for k, (v, lab, c) in enumerate((("PoPE-Flat", "PoPE (polar score)", AQUA), ("RoPE", "RoPE", ORANGE))):
    line = re.search(rf"^{re.escape(v)}: argmax over odd D: (.*?)\s+\|", txt, re.M).group(1)
    cnt = {int(a): int(b) for a, b in re.findall(r"(\d+):(\d+)", line)}; tot = sum(cnt.values())
    sh = np.array([cnt.get(d, 0) / tot for d in ODD])
    ax.bar(np.arange(len(ODD)) + (k - 0.5) * 0.38, sh, width=0.36, color=c, label=lab, zorder=2)
    out.append(f"  {lab:20s} n {tot}: " + "  ".join(f"D={d}: {cnt.get(d, 0)} ({cnt.get(d, 0) / tot:.3f})" for d in ODD if cnt.get(d, 0)))
pp = open(f"{MP}/kernel_probe_paper2x2_out.txt").read()
for v in ("RoPE", "PoPE-Flat"):
    blk = pp.split(f"-- {v} (")[1]
    fr = [float(a) for a in re.search(r"frac\(K\(3\) > K\(1\)\) per run: ([0-9. ]+)", blk).group(1).split()]
    out.append(f"  {v}: share of pairs with K(3) > K(1) per run {min(fr):.2f}-{max(fr):.2f}")
ax.set_xticks(range(len(ODD))); ax.set_xticklabels(ODD); ax.set_xlabel("token offset D of the kernel's peak (odd D)")
ax.set_ylabel("share of content pairs"); ax.set_ylim(0, 1.05); ax.grid(axis="x", visible=False)
ax.set_title("(a) index arms, T = 128: where K(D) peaks", loc="left", fontsize=6.5)
ax.legend(loc="upper right", fontsize=5.6)

# (b) C vs C0, path arms, T = 128
ax = axs[1]; J = json.load(open(f"{MP}/kernel_probe_paper2x2.json"))["paper2x2"]
ps = open(f"{MP}/probe_stats_out.txt").read().split("=== score_rank ===")[0]
out.append("(b) target coherence, PAPER2X2 path arms, all revisits (per seed; mean)")
ax.plot([0.35, 1.025], [0.35, 1.025], color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
ax.text(0.42, 0.45, "C = C0", fontsize=5.4, color=INK2, rotation=40, rotation_mode="anchor")
for v, lab, c, mk in (("Vanilla", "MapWM (RoPE score)", BLUE, "o"), ("MapPoPE-Flat", "MapPoPE (polar score)", AQUA, "s")):
    C, C0 = col(J, v, "all", "C"), col(J, v, "all", "C0")
    for q, a in (("C", C), ("C0", C0)):
        m = re.search(rf"^\s+{q}\s+all\s+MapPoPE-Flat ([0-9.]+) vs Vanilla ([0-9.]+)", ps, re.M)
        assert f"{a.mean():.3f}" == m.group(1 if v != "Vanilla" else 2), (v, q)
    ax.scatter(C0, C, s=12, color=c, marker=mk, alpha=0.8, lw=0, zorder=3, label=lab)
    out.append(f"  {lab:22s} C {C.mean():.3f} ({C.min():.3f}-{C.max():.3f})  C0 {C0.mean():.3f} ({C0.min():.3f}-{C0.max():.3f})"
               f"  C0 - C {np.mean(C0 - C):+.3f}, C < C0 on {int((C < C0).sum())}/{len(C)} seeds")
ax.set_xlim(0.35, 1.025); ax.set_ylim(0.35, 1.025)
ax.set_xlabel("C0 (content phase removed)"); ax.set_ylabel("C (target coherence)")
ax.set_title("(b) path arms, T = 128", loc="left", fontsize=6.5)
ax.legend(loc="upper left", fontsize=5.4, handletextpad=0.2)

# (c) C0 by stratum, T = 1024, per-stratum head
ax = axs[2]; J = json.load(open(f"{MP}/kernel_probe_score_rank_hstratum.json"))["score_rank"]
ph = open(f"{MP}/probe_stats_hstratum_out.txt").read()
ST = [("plain gap<128", "gap < 128"), ("plain gap>=128", "gap >= 128"), ("wrap-only", "wrap-only")]
ARMS = [("Vanilla_r4mi", "rank 4, RoPE score (n 8)", BLUE, "o", "-"),
        ("MapPoPE-Pair_r4mi", "rank 4, polar score (n 8)", AQUA, "s", "-"),
        ("Vanilla", "rank 2, RoPE score (n 12)", ORANGE, "^", "-"),
        ("MapPoPE-Pair", "rank 2, polar score (n 12)", YELLOW, "D", "-")]
rng = np.random.default_rng(0); x = np.arange(3)
out.append("(c) C0 by stratum, SCORE_RANK / RANK_MI, T = 1024, each stratum on its own head (HEAD_RULE=stratum)")
for gi, (v, lab, c, mk, ls) in enumerate(ARMS):
    A = [col(J, v, sn, "C0") for sn, _ in ST]; ms = np.array([a.mean() for a in A]); off = (gi - 1.5) * 0.06
    for j, (sn, _) in enumerate(ST):
        m = re.search(rf"^\s+{re.escape(v)}\s+{re.escape(sn)}\s+C [0-9.]+\s+C0 ([0-9.]+)", ph, re.M)
        assert f"{ms[j]:.3f}" == m.group(1), (v, sn, m.group(1))
        ax.scatter(np.full(len(A[j]), j + off) + rng.uniform(-0.015, 0.015, len(A[j])), A[j], s=4, color=c, alpha=0.45,
                   lw=0, zorder=2)
    ax.plot(x + off, ms, color=c, lw=1.5, ls=ls, marker=mk, ms=3.6, zorder=3, label=lab)
    out.append(f"  {lab:28s} " + "  ".join(f"{s}: {a.mean():.3f} ({a.min():.3f}-{a.max():.3f})" for (s, _), a in zip(ST, A)))
ax.set_xticks(x); ax.set_xticklabels([l for _, l in ST]); ax.set_xlim(-0.4, 2.4); ax.grid(axis="x", visible=False)
ax.set_ylim(-0.25, 1.05); ax.set_ylabel("C0 (code term)")
ax.set_title("(c) path arms, T = 1024: C0 by stratum", loc="left", fontsize=6.5)
ax.legend(loc="lower left", fontsize=5.4, handlelength=1.6)
fig.subplots_adjust(left=0.065, right=0.99, top=0.9, bottom=0.17, wspace=0.42)
save(fig, HERE, "fig_kernel_peak"); write_out(HERE, "fig_kernel_peak", out)
