"""F1 of the overall outline: the exchange rate -- accuracy of index-RoPE attention against its number of layers, beside
one layer of path integration, at MATCHED length and depth (trained and tested on the same distribution). Stored
results only (CPU, no model):
  (a) Dyck-2, trained and scored at L32 D12 (DYCK_MDEPTH, registered; T12 cells, 1x budget; metric A2f; seeds 0-7):
      DYCK_MDEPTH_RESULTS.json cells 'T12|RoPE|{1..4}L|L32D12|A2f...' and 'T12|MapWM|1L|...'; floor = best A2f floor at
      L32 D12 (gates, T12 fit) 0.594.
  (b) 1D ring, 32 cells, T = 128, 300 epochs (CANCEL / H3, registered): runs/cancel/p0/*/eval.json acc at 128, FRESH
      SEEDS 2-7 ONLY (seeds 0-1 were the pilot, CANCEL_RESULTS CORRECTED block). Floors: retrace 0.860 / 0.758 / 0.612
      at p_plus 0.5 / 0.75 / 0.9 (docs/audits/2026-10-16/ring_floors_out.txt; another stream of the same task, post hoc).
  (c) 2D torus, T = 1024, 900 epochs, --fast-attn (FACTORIAL, registered; seeds 500-503): FACTORIAL_{L1,L4,LOOP}.json;
      floors on this batch's own eval stream (torus_strata_floors_s500-503.json, post hoc): blank 0.509, retrace 0.844.
  cd /home/prashr && PYTHONPATH=/home/prashr python3 mapformer/docs/paper/figs/fig_exchange_rate.py"""
import json, os
import numpy as np
from figstyle import plt, BLUE, ORANGE, BLUE_RAMP, INK2, MUTED, FLOOR_LS, floor_line, save, write_out

REPO = "/home/prashr/mapformer"; HERE = os.path.dirname(os.path.abspath(__file__))
OR = ["#f4a582", "#eb6834", "#a8431a"]          # ordinal orange (p_plus 0.5 -> 0.9), markers + direct labels carry identity
out = []
fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.35), gridspec_kw={"width_ratios": [1, 1, 1]})
rng = np.random.default_rng(0)


def dots(ax, x, ys, col, mk="o", filled=True):
    ax.scatter(x + rng.uniform(-0.07, 0.07, len(ys)), ys, s=7, marker=mk, color=col if filled else "white",
               edgecolors=col, lw=0.6, zorder=3, alpha=0.8)


# (a) Dyck
D = json.load(open(f"{REPO}/DYCK_MDEPTH_RESULTS.json"))["cells"]
key = lambda arm, L: f"T12|{arm}|{L}L|L32D12|A2f_close_acc_dist_feasible"
ax = axs[0]; L = np.arange(1, 5)
idx = [D[key("RoPE", l)] for l in L]; path1 = D[key("MapWM", 1)]
for l, c in zip(L, idx):
    dots(ax, l, c["per_seed"], ORANGE)
ax.plot(L, [c["mean"] for c in idx], color=ORANGE, lw=1.6, marker="o", ms=3.5, zorder=4)
ax.axhline(path1["mean"], color=BLUE, lw=1.6, zorder=2)
ax.text(0.85, path1["mean"] + 0.006, f"path, 1 layer {path1['mean']:.3f}", ha="left", va="bottom", fontsize=5.8, color=INK2)
floor_line(ax, 0.5935, "best floor 0.594", x=0.98)
ax.set_ylim(0.55, 1.045); ax.set_title("(a) Dyck-2, L32 D12 (DYCK_MDEPTH)", loc="left")
out.append("(a) Dyck-2 L32 D12, A2f, n = 8, T12 1x budget: path MapWM 1L " + f"{path1['mean']:.4f}; index RoPE by layers: "
           + ", ".join(f"{l}L {c['mean']:.4f} ({min(c['per_seed']):.3f}-{max(c['per_seed']):.3f})" for l, c in zip(L, idx))
           + "; MapWM at 2-4L: " + ", ".join(f"{D[key('MapWM', l)]['mean']:.4f}" for l in (2, 3, 4)) + "; floor 0.5935")

# (b) ring
ax = axs[1]; RET = {"0.5": 0.8596, "0.75": 0.7582, "0.9": 0.6118}; MK = {"0.5": "o", "0.75": "s", "0.9": "^"}
acc = lambda arm, p, s: json.load(open(f"{REPO}/runs/cancel/p0/{arm}_p{p}_s{s}/eval.json"))["eval"]["128"]["acc"]
SEEDS = range(2, 8); out.append("(b) ring 32 cells, T = 128, fresh seeds 2-7 (n = 6)")
for (p, col) in zip(("0.5", "0.75", "0.9"), OR):
    ys = [[acc(f"RoPE_L{l}", p, s) for s in SEEDS] for l in (1, 2, 3)]
    for l, y in zip((1, 2, 3), ys):
        dots(ax, l, y, col, MK[p])
    mm = [np.mean(y) for y in ys]
    ax.plot((1, 2, 3), mm, color=col, lw=1.4, marker=MK[p], ms=3.5, zorder=4)
    pa = [acc("Vanilla_L1", p, s) for s in SEEDS]
    out.append(f"  p_plus {p}: path 1L {np.mean(pa):.4f} (min {min(pa):.4f}); index 1/2/3L " + " / ".join(f"{m:.4f}" for m in mm)
               + f"; retrace floor {RET[p]:.3f}")
    ax.axhline(RET[p], color=col, lw=0.7, ls=FLOOR_LS, zorder=1)
    ax.text(3.32, RET[p], f"retrace\np+ {p}", fontsize=5.0, va="center", ha="left", color=INK2, linespacing=0.9)
    ax.text(0.86, mm[0], f"p+ {p}", fontsize=5.6, va="center", ha="right", color=INK2)
pa1 = [acc("Vanilla_L1", "1.0", s) for s in SEEDS]; ia1 = [acc(f"RoPE_L{l}", "1.0", s) for l in (1, 2, 3) for s in SEEDS]
out.append(f"  p_plus 1.0 (no step undone; not drawn): path 1L {np.mean(pa1):.4f}, index 1-3L min {min(ia1):.4f}")
ax.axhline(1.0, color=BLUE, lw=1.6, zorder=2)
ax.text(0.35, 1.006, "path, 1 layer 1.000", ha="left", va="bottom", fontsize=5.8, color=INK2)
ax.set_xlim(0.3, 3.9); ax.set_xticks([1, 2, 3]); ax.set_ylim(0.55, 1.045)
ax.set_title("(b) ring, T 128 (CANCEL, fresh seeds)", loc="left")

# (c) torus
ax = axs[2]
F = {f: json.load(open(f"{REPO}/FACTORIAL_{f}.json")) for f in ("L1", "L4", "LOOP")}
get = lambda f, arm: [r[1] for r in F[f][f"0.0|{arm}|1024"]]
i1, i4, lp, p1 = get("L1", "RoPE"), get("L4", "RoPE"), get("LOOP", "RoPELooped"), get("L1", "Vanilla_r4")
dots(ax, 1, i1, ORANGE); dots(ax, 4, i4, ORANGE); dots(ax, 4.35, lp, ORANGE, "D", filled=False)
ax.plot([1, 4], [np.mean(i1), np.mean(i4)], color=ORANGE, lw=1.6, marker="o", ms=3.5, zorder=4)
ax.scatter([4.35], [np.mean(lp)], marker="D", s=16, color="white", edgecolors=ORANGE, lw=1.2, zorder=5)
ax.text(4.35, np.mean(lp) + 0.03, "looped\nx4", fontsize=5.4, ha="center", va="bottom", color=INK2, linespacing=0.9)
ax.axhline(np.mean(p1), color=BLUE, lw=1.6, zorder=2)
ax.text(0.7, np.mean(p1) + 0.006, f"path, 1 layer {np.mean(p1):.3f}", ha="left", va="bottom", fontsize=5.8, color=INK2)
FL = json.load(open(f"{HERE}/torus_strata_floors_s500-503.json"))["per_seed"]
ret = np.mean([FL[k]["all"]["retrace"] for k in FL]); bl = np.mean([FL[k]["all"]["const"] for k in FL])
floor_line(ax, ret, f"retrace {ret:.3f}", x=0.02, ha="left"); floor_line(ax, bl, f"blank {bl:.3f}", x=0.02, ha="left")
ax.set_xlim(0.6, 4.7); ax.set_xticks([1, 2, 3, 4]); ax.set_ylim(0.45, 1.045)
ax.set_title("(c) torus, T 1024 (FACTORIAL)", loc="left")
out.append(f"(c) torus T = 1024, n = 4: path 1L {np.mean(p1):.4f}; index 1L {np.mean(i1):.4f} ({min(i1):.3f}-{max(i1):.3f}), "
           f"4L {np.mean(i4):.4f} ({min(i4):.3f}-{max(i4):.3f}), looped x4 {np.mean(lp):.4f} ({min(lp):.3f}-{max(lp):.3f}); "
           f"floors (own stream) blank {bl:.4f}, retrace {ret:.4f}")
for a in axs:
    a.set_xlabel("index-attention layers", labelpad=1); a.grid(axis="x", visible=False)
    a.title.set_fontsize(6.5)
axs[0].set_ylabel("accuracy at matched length / depth")
fig.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.17, wspace=0.22)
save(fig, HERE, "fig_exchange_rate"); write_out(HERE, "fig_exchange_rate", out)
