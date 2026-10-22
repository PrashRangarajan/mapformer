"""Plan Fig. 8 (docs/paper/expansion/SYNTHESIS_DRAFT2_PLAN.md 3.6): revisit accuracy against the number of direction runs
the walk spans between the two visits (maximal runs of one move; bins 2, 3-4, 5-8, 9-16, 17+). ANALYSIS (post hoc,
eval-only, eval mode, no dropout rescale) of registered batches; stored per-revisit rows, no model:
  (a) torus T = 1024, FACTORIAL (n = 4, seeds 500-503): index 1 layer / 4 layers / looped x4, path 1 layer. Solid:
      plain revisits at gap < 128 moves (runs 2 = retrace; 3+ = not retrace) = MAIN_DRAFT2 Table 3. Dashed (index arms):
      plain revisits at gaps 32-127 moves only, every bin (torus_segments_out.txt; the fixed-gap control).
  (b) ring of 32 cells, T = 128, p_plus 0.5, CANCEL fresh seeds 2-7 (n = 6): index 1 / 2 / 3 layers, path 1 layer. Runs 2
      = retrace; 3+ = not retrace, not a one-lap return and NOT wrap-only (plain revisits only; ANALYSIS_SCRIPT_AUDIT depth
      MINOR; = ring_plain_bins.py's 'plain only' column). No ring revisit spans 17+ runs at T = 128.
Values: mean over seeds of per-seed accuracies (seed means, not pooled); dots: seeds. Data: expansion/scripts/depth/out/
segments_bins.json (segments_bins.py from the rows of torus_depth.py / ring_depth.py), asserted equal to the committed
text outputs. Floors on each batch's own stream: retrace is exact on 2-run revisits (= 1 there by construction) and
falls to the best constant elsewhere; best constant = the short-gap stratum's (torus, torus_strata_floors_s500-503.json)
or the 'neither' class's (ring, ring_class_floors_out.txt; that class includes wrap-only revisits).
  cd /home/prashr && PYTHONPATH=/home/prashr python3 mapformer/docs/paper/figs/fig_segments.py"""
import json
import os
import re

import numpy as np
from figstyle import plt, BLUE, ORANGE, AQUA, YELLOW, VIOLET, INK2, FLOOR_LS, save, write_out

REPO = "/home/prashr/mapformer"; HERE = os.path.dirname(os.path.abspath(__file__))
DEP = f"{REPO}/docs/paper/expansion/scripts/depth/out"
J = json.load(open(f"{DEP}/segments_bins.json")); BINS = J["bins"]; x = np.arange(len(BINS))


def mean(r):
    v = [a for a in r["acc"] if a is not None]
    return (float(np.mean(v)), v) if v else (np.nan, [])


# ---- cross-checks against the committed text outputs (3 decimals) ----
tt = open(f"{DEP}/torus_depth_out.txt").read()
for b, lab in zip(BINS, ["short, retrace", "short non-retrace, runs 3-4", "short non-retrace, runs 5-8",
                         "short non-retrace, runs 9-16", "short non-retrace, runs 17-+"]):
    row = re.search(rf"^\| {re.escape(lab)} \|.*$", tt, re.M).group(0).split("|")
    for arm, cell in zip(("L1", "L4", "LOOP", "PATH"), row[4:8]):
        assert f"{mean(J['torus'][arm]['short'][b])[0]:.3f}" == cell.split()[0], (arm, b, cell)
ts = open(f"{DEP}/torus_segments_out.txt").read().split("== plain revisits at gaps 8-31")[0]
for b in BINS:
    row = re.search(rf"^\| {re.escape(b)} \|.*$", ts, re.M).group(0).split("|")
    for arm, cell in zip(("L1", "L4", "LOOP"), row[4:7]):
        assert f"{mean(J['torus'][arm]['gap32'][b])[0]:.3f}" == cell.split()[0], (arm, b, cell)
rp = open(f"{DEP}/ring_plain_bins_out.txt").read().split("== p_plus 0.75")[0]
for b in ("3-4", "5-8", "9-16"):
    row = re.search(rf"^\| {re.escape(b)} \|.*$", rp, re.M).group(0)
    for arm in ("RoPE_L1", "RoPE_L2", "RoPE_L3"):
        plain = re.search(rf"{arm} [0-9.]+ / ([0-9.]+) /", row).group(1)
        assert f"{mean(J['ring']['0.5'][arm][b])[0]:.3f}" == plain, (arm, b, plain)
rd = open(f"{DEP}/ring_depth_out.txt").read().split("== p_plus 0.75")[0]
row = re.search(r"^\| retrace \|.*$", rd, re.M).group(0).split("|")
for arm, cell in zip(("RoPE_L1", "RoPE_L2", "RoPE_L3", "Vanilla_L1"), row[3:7]):
    assert f"{mean(J['ring']['0.5'][arm]['2'])[0]:.3f}" == cell.split()[0], (arm, cell)

# ---- floors ----
FL = json.load(open(f"{HERE}/torus_strata_floors_s500-503.json"))["per_seed"]
t_const = float(np.mean([FL[k]["plain_lag<128"]["const"] for k in FL]))
rcf = open(f"{DEP}/ring_class_floors_out.txt").read()
r_const = float(re.search(r"p_plus 0.5: .*neither: n \d+, best constant ([0-9.]+)", rcf).group(1))

PANELS = [
    ("(a) torus, T = 1024 (FACTORIAL, n 4)", "torus", None, [
        ("PATH", "path, 1 layer (eval mode)", BLUE, "o"),
        ("L1", "index, 1 layer (eval mode)", ORANGE, "s"),
        ("L4", "index, 4 layers", AQUA, "^"),
        ("LOOP", "index, looped x4", YELLOW, "D")], t_const, "short-gap"),
    ("(b) ring, T = 128, p_plus 0.5 (CANCEL, seeds 2-7)", "ring", "0.5", [
        ("Vanilla_L1", "path, 1 layer (eval mode)", BLUE, "o"),
        ("RoPE_L1", "index, 1 layer (eval mode)", ORANGE, "s"),
        ("RoPE_L2", "index, 2 layers", VIOLET, "v"),
        ("RoPE_L3", "index, 3 layers", AQUA, "^")], r_const, "'neither'"),
]
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.75), sharey=True)
rng = np.random.default_rng(0); out = []
for ax, (title, task, p, arms, const, cname) in zip(axs, PANELS, ):
    out.append(title)
    for gi, (arm, lab, col, mk) in enumerate(arms):
        D = J[task][arm] if task == "torus" else J[task][p][arm]
        pops = [("short", "-")] + ([("gap32", "--")] if task == "torus" and arm != "PATH" else []) if task == "torus" \
            else [(None, "-")]
        for pn, ls in pops:
            rows = D[pn] if pn else D
            ms = np.array([mean(rows[b])[0] for b in BINS]); off = (gi - 1.5) * 0.07
            ok = ~np.isnan(ms)
            if ls == "-":
                for j, b in enumerate(BINS):
                    v = mean(rows[b])[1]
                    ax.scatter(np.full(len(v), j + off) + rng.uniform(-0.015, 0.015, len(v)), v, s=4, color=col,
                               alpha=0.45, lw=0, zorder=2)
                ax.plot(x[ok] + off, ms[ok], color=col, lw=1.5, marker=mk, ms=3.6, zorder=3, label=lab)
            else:
                ax.plot(x[ok] + off, ms[ok], color=col, lw=0.9, ls=(0, (3, 1.6)), marker=mk, ms=3.0, mfc="white",
                        mew=0.8, zorder=3, alpha=0.9)
            pname = {"short": "gap < 128", "gap32": "gap 32-127", None: "plain, not lap"}[pn]
            out.append(f"  {lab:28s} [{pname}] " + "  ".join(
                f"{b}: {mean(rows[b])[0]:.3f} ({min(mean(rows[b])[1]):.3f}-{max(mean(rows[b])[1]):.3f}; n {rows[b]['n']})"
                if rows[b]["n"] else f"{b}: --" for b in BINS))
    nb = len(BINS) if task == "torus" else 4
    ax.plot([-0.3, nb - 0.7], [const, const], color=INK2, lw=0.8, ls=FLOOR_LS, zorder=1)
    ax.plot([-0.3, 0.3], [1.0, 1.0], color=INK2, lw=0.8, ls=(0, (1, 1.2)), zorder=1)
    out.append(f"  floors: retrace 1 at 2 runs (exact by construction), best constant ({cname} class) {const:.3f} elsewhere")
    ax.set_xticks(x); ax.set_xticklabels(BINS); ax.set_xlim(-0.4, len(BINS) - 0.6)
    ax.grid(axis="x", visible=False); ax.set_title(title, loc="left", fontsize=6.5)
    ax.set_xlabel("direction runs spanned between the two visits")
axs[0].set_ylabel("revisit accuracy (held-out map)"); axs[0].set_ylim(0.34, 1.04)
axs[1].text(4, 0.86, "no revisit\nspans 17+ runs\nat T = 128", ha="center", va="center", fontsize=5.6, color=INK2)
h, l = axs[0].get_legend_handles_labels()
h += [plt.Line2D([], [], color=INK2, lw=0.9, ls=(0, (3, 1.6)), marker="s", ms=3.0, mfc="white", mew=0.8),
      plt.Line2D([], [], color=INK2, lw=0.8, ls=(0, (1, 1.2))), plt.Line2D([], [], color=INK2, lw=0.8, ls=FLOOR_LS)]
l += ["index arms, gap 32-127 only", "retrace (exact at 2 runs)", "best constant"]
axs[0].legend(h, l, loc="upper right", fontsize=5.4, handlelength=2.2, bbox_to_anchor=(1.0, 0.935))
axs[1].legend(loc="lower right", fontsize=5.4, handlelength=1.8, bbox_to_anchor=(1.0, 0.24))  # floor styles: (a)
fig.subplots_adjust(left=0.07, right=0.99, top=0.91, bottom=0.15, wspace=0.06)
save(fig, HERE, "fig_segments"); write_out(HERE, "fig_segments", out)
