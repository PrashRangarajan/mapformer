"""Diagnostic figures for the verified MapFormer results (2026-08-09 batch).

Run from /home/prashr:

    python3 -m mapformer.make_figures

Everything is read from files already in the repo (JSON where it exists,
per-seed run JSONs under runs/ otherwise). No training, no GPU: the only model
code touched is the two environment builders, which are pure numpy.

Every accuracy panel draws its floor. Where a baseline stronger than chance
exists (family tree's hub baseline), that is drawn as the solid floor and
chance is drawn dotted underneath, because the hub baseline is the number a
result has to beat.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures_2026-08-09")
os.makedirs(OUT, exist_ok=True)

PI_COLOR = "#1b6ca8"      # path integration
IDX_COLOR = "#c8553d"     # index position
FLOOR_COLOR = "#444444"
plt.rcParams.update({"figure.dpi": 140, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.25,
                     "axes.axisbelow": True})


def _p(*parts):
    return os.path.join(HERE, *parts)


def _load(path):
    with open(_p(path)) as fh:
        return json.load(fh)


def _seed_jsons(rundir, fname):
    """Collect {seed: {T: acc}} for one variant across seed* subdirs."""
    out = {}
    root = _p("runs", rundir)
    if not os.path.isdir(root):
        return out
    for sd in sorted(os.listdir(root)):
        f = os.path.join(root, sd, fname)
        if os.path.exists(f):
            with open(f) as fh:
                j = json.load(fh)
            out[sd] = {int(k): (v["match_acc"] if isinstance(v, dict) else v)
                       for k, v in j.items()}
    return out


def _caption(fig, text):
    fig.text(0.5, 0.005, text, ha="center", va="bottom", fontsize=7.5,
             color="#333333", wrap=True)


# ---------------------------------------------------------------- figure 1
def fig_timing():
    d = _load("TIMING_BENCHMARK.json")
    res, params = d["results"], d["params"]
    label = {
        "Vanilla": "MapWM-Flat (parallel scan)",
        "VanillaEM_P0": "MapEM single-p0 (parallel scan)",
        "PlainFlat": "PlainFlat, index pos. (no scan)",
        "MapEM_NC_L": "MapEM-NC-L (sequential matrix scan)",
        "TEMFaithful": "TEMFaithful (sequential RNN)",
    }
    style = {
        "Vanilla": (PI_COLOR, "-", "o"),
        "VanillaEM_P0": ("#4a9ad4", "-", "s"),
        "PlainFlat": ("#7a7a7a", "-", "^"),
        "MapEM_NC_L": ("#e08214", "--", "D"),
        "TEMFaithful": (IDX_COLOR, "--", "v"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for ax, key, name in ((axes[0], "fwdbwd_ms", "forward + backward"),
                          (axes[1], "fwd_ms", "forward only")):
        for v, per_L in res.items():
            Ls = sorted(int(k) for k in per_L)
            ys = [per_L[str(L)][key] for L in Ls]
            c, ls, mk = style[v]
            growth = ys[-1] / ys[0]
            ax.plot(Ls, ys, ls, color=c, marker=mk, ms=4,
                    label=f"{label[v]}  [{growth:.1f}x over 16x L]")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks([128, 256, 512, 1024, 2048])
        ax.set_xticklabels([128, 256, 512, 1024, 2048])
        ax.set_xlabel("sequence length L (tokens)")
        ax.set_ylabel(f"{name} wall-clock (ms per batch)")
        ax.set_title(f"{name}, batch 32, CUDA")
        ax.legend(fontsize=6.5, loc="upper left")

    # O(L) and O(1) reference slopes on the left panel only
    ax = axes[0]
    ref = res["Vanilla"]["128"]["fwdbwd_ms"]
    Ls = np.array([128, 2048])
    ax.plot(Ls, ref * Ls / 128 * 0.9, ":", color="#999999", lw=1)
    ax.text(2048, ref * 16 * 0.9, " O(L)", fontsize=7, color="#777777",
            va="center")
    ax.plot(Ls, [ref * 0.55, ref * 0.55], ":", color="#999999", lw=1)
    ax.text(2048, ref * 0.55, " O(1)", fontsize=7, color="#777777", va="center")

    fig.suptitle("Timing: parallel path integration stays flat, sequential does not "
                 "(single timed run per point, no error bars)", fontsize=10)
    _caption(fig,
             "Source: TIMING_BENCHMARK.json. No chance line applies -- this is wall-clock, "
             "not accuracy. Bracketed factor = time at L=2048 / time at L=128. Params: "
             + ", ".join(f"{label[v].split(' (')[0]} {params[v]/1e3:.0f}K" for v in res)
             + ". TEMFaithful is far smaller yet far slower, so the gap is the scan, not model size.")
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(os.path.join(OUT, "fig1_timing.png"))
    plt.close(fig)


# ---------------------------------------------------------------- figure 2
def fig_match_query():
    configs = [
        ("base: 64$^2$, n_obs=16", "match_scale_base", 0.0625,
         ["runs/match_query", "runs/match_scale_base"]),
        ("big: 128$^2$, n_obs=16", "match_scale_big", 0.0625, ["runs/match_scale_big"]),
        ("alias: 64$^2$, n_obs=4", "match_scale_alias", 0.25, ["runs/match_scale_alias"]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.2), sharey=False)
    axes = axes.ravel()

    for ax, (title, _tag, chance, dirs) in zip(axes, configs):
        for variant, colour, lbl in (("Vanilla", PI_COLOR, "MapWM-Flat (path integration)"),
                                     ("PlainFlat", IDX_COLOR, "PlainFlat (index position)")):
            per_seed = {}
            for d in dirs:
                per_seed.update({f"{d}:{k}": v for k, v in
                                 _seed_jsons(os.path.basename(d),
                                             f"{variant}_matchquery.json").items()})
            Ts = sorted({T for v in per_seed.values() for T in v})
            means, stds, ns = [], [], []
            for T in Ts:
                vals = [v[T] for v in per_seed.values() if T in v]
                means.append(np.mean(vals))
                stds.append(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
                ns.append(len(vals))
                ax.plot([T] * len(vals), vals, "o", color=colour, ms=3.5,
                        alpha=0.45, mec="none", zorder=3)
            ax.errorbar(Ts, means, yerr=stds, color=colour, marker="o", ms=6,
                        capsize=3, lw=1.8, label=lbl, zorder=4)
            for T, m, n in zip(Ts, means, ns):
                ax.annotate(f"n={n}", (T, m), textcoords="offset points",
                            xytext=(0, 9 if colour == PI_COLOR else -14),
                            ha="center", fontsize=6, color=colour)
        ax.axhline(chance, color=FLOOR_COLOR, ls="--", lw=1.2)
        ax.text(256, chance, f" chance = {chance:.4g}", fontsize=7,
                va="bottom", color=FLOOR_COLOR)
        ax.set_xscale("log", base=2)
        ax.set_xticks([256, 512, 1024])
        ax.set_xticklabels([256, 512, 1024])
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("blind query length T_query (tokens)")
        ax.set_ylabel("held-out match accuracy (fraction correct)")
        ax.set_title(title)
        ax.legend(fontsize=6.5, loc="upper right")

    # panel 4: long-query extrapolation, per-seed traces
    ax = axes[3]
    lq = _load("MATCH_QUERY_LONGQ.json")
    for variant, colour, lbl in (("Vanilla", PI_COLOR, "MapWM-Flat (path integration)"),
                                 ("PlainFlat", IDX_COLOR, "PlainFlat (index position)")):
        Ts = sorted(int(k) for k in lq[variant])
        arr = np.array([lq[variant][str(T)] for T in Ts])       # (T, seeds)
        for s in range(arr.shape[1]):
            ax.plot(Ts, arr[:, s], "-", color=colour, lw=0.8, alpha=0.4)
        ax.errorbar(Ts, arr.mean(1), yerr=arr.std(1, ddof=1), color=colour,
                    marker="o", ms=6, capsize=3, lw=1.8, label=lbl, zorder=4)
    ax.axhline(0.0625, color=FLOOR_COLOR, ls="--", lw=1.2)
    ax.text(256, 0.0625, " chance = 0.0625", fontsize=7, va="bottom", color=FLOOR_COLOR)
    ax.axvline(256, color="#999999", ls=":", lw=1)
    ax.text(256, 1.0, " trained T_query", fontsize=6.5, color="#777777", rotation=90,
            va="top", ha="left")
    ax.set_xscale("log", base=2)
    ax.set_xticks([256, 512, 1024, 2048])
    ax.set_xticklabels([256, 512, 1024, 2048])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("blind query length T_query (tokens)")
    ax.set_ylabel("held-out match accuracy (fraction correct)")
    ax.set_title("base 64$^2$: extrapolation to 8x trained T_query (n=3)")
    ax.legend(fontsize=6.5, loc="upper right")

    fig.suptitle("Match-Query: the path-integration / index-position gap never closes "
                 "(faint dots and lines = individual seeds)", fontsize=10)
    _caption(fig,
             "Error bars are standard deviation across seeds (ddof=1), not sem. Panels 1-3 from "
             "per-seed runs/match_scale_*/ and runs/match_query/ JSONs; base has n=5 at "
             "T_query=256/512 but only n=2 at 1024 (seeds 3,4 -- seeds 0-2 were not evaluated at "
             "1024 in that batch), hence the drop; panel 4 re-evaluates seeds 0-2 across all "
             "lengths (MATCH_QUERY_LONGQ.json). In the alias panel the per-seed ranges OVERLAP -- "
             "that is the measured boundary of the result, not a plotting artifact.")
    fig.tight_layout(rect=(0, 0.10, 1, 0.95))
    fig.savefig(os.path.join(OUT, "fig2_match_query_scale.png"))
    plt.close(fig)


# ---------------------------------------------------------------- figure 3
def fig_em_p0():
    arms = [("Vanilla", "MapWM-Flat\n(WM reference)", "#7a7a7a"),
            ("VanillaEM", "MapEM separate q0/k0\n(paper-faithful)", IDX_COLOR),
            ("VanillaEM_P0", "MapEM single p_0\n(ablation)", PI_COLOR)]
    data = {k: _seed_jsons("match_query_em", f"{k}_matchquery.json") for k, _, _ in arms}
    seeds = sorted(data["Vanilla"])

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.4), sharey=True)
    for ax, T, ttl in ((axes[0], 256, "T_query = 256 (trained)"),
                       (axes[1], 512, "T_query = 512 (OOD)")):
        xs = np.arange(len(arms))
        for si, sd in enumerate(seeds):
            ys = [data[k][sd][T] for k, _, _ in arms]
            ax.plot(xs, ys, "-o", ms=5, lw=1.1, alpha=0.8,
                    color=plt.cm.viridis(si / max(1, len(seeds) - 1) * 0.8),
                    label=f"seed {si}")
        means = [np.mean([data[k][sd][T] for sd in seeds]) for k, _, _ in arms]
        ax.plot(xs, means, "_", ms=34, mew=2.5, color="black", zorder=5)
        for x, m in zip(xs, means):
            ax.annotate(f"mean {m:.3f}", (x, m), textcoords="offset points",
                        xytext=(0, 12), ha="center", fontsize=7, fontweight="bold")
        ax.axhline(0.0625, color=FLOOR_COLOR, ls="--", lw=1.2)
        ax.text(-0.45, 0.0625, " chance = 0.0625", fontsize=7, va="bottom",
                color=FLOOR_COLOR)
        ax.set_xticks(xs)
        ax.set_xticklabels([lbl for _, lbl, _ in arms], fontsize=7.5)
        ax.set_xlim(-0.5, len(arms) - 0.5)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("held-out match accuracy (fraction correct)")
        ax.set_title(ttl)
        ax.legend(fontsize=7, loc="lower right")

    # paired deltas, printed on the left panel
    d256 = [data["VanillaEM_P0"][sd][256] - data["VanillaEM"][sd][256] for sd in seeds]
    axes[0].text(0.5, 0.30,
                 "paired single-p_0 - separate q0/k0:\n"
                 + " / ".join(f"{v:+.3f}" for v in d256)
                 + f"\nmean {np.mean(d256):+.3f}, {sum(v > 0 for v in d256)}/{len(d256)} seeds",
                 transform=axes[0].transAxes, fontsize=7, va="center", ha="center",
                 bbox=dict(fc="white", ec="#cccccc", alpha=0.9))

    fig.suptitle("Match-Query, EM parameterisation: paired per seed (n=3, all arms trained "
                 "in one batch)", fontsize=10)
    _caption(fig,
             "Source: runs/match_query_em/seed*/ per-seed JSONs (same numbers as "
             "MATCH_QUERY_EM.md). Lines connect the SAME seed across arms, so the vertical "
             "gap is the paired effect; black dashes are arm means. No error bars are drawn -- "
             "the seed points are shown directly. Seed 0 of the paper-faithful arm falls to "
             "0.107, just above the 0.0625 chance line.")
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    fig.savefig(os.path.join(OUT, "fig3_match_query_em_p0.png"))
    plt.close(fig)


# ---------------------------------------------------------------- figure 4
def fig_family_tree():
    gates = _load("FAMILY_TREE_GATES.json")
    chance, hub, last_obs = gates["chance"], gates["hub"], gates["last_obs"]
    arms = [("MapEM_NC_L", "MapEM-NC-L\nnon-commutative (linear)", "#1b6ca8"),
            ("MapEM_NC_NL", "MapEM-NC-NL\nnon-commutative (MLP)", "#4a9ad4"),
            ("VanillaEM_P0", "MapEM single-p0\nCOMMUTATIVE control", "#e08214"),
            ("PlainFlat", "Plain-Flat\nindex position, no PI", IDX_COLOR)]
    data = {k: _seed_jsons("family_tree", f"{k}_familytree.json") for k, _, _ in arms}
    seeds = sorted(data["PlainFlat"])

    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    width = 0.36
    xs = np.arange(len(arms))
    for off, T, alpha, lbl in ((-width / 2, 64, 1.0, "n_steps = 64 (trained)"),
                               (+width / 2, 128, 0.55, "n_steps = 128 (OOD)")):
        means = [np.mean([data[k][sd][T] for sd in seeds]) for k, _, _ in arms]
        stds = [np.std([data[k][sd][T] for sd in seeds], ddof=1) for k, _, _ in arms]
        ax.bar(xs + off, means, width, yerr=stds, capsize=3, alpha=alpha,
               color=[c for _, _, c in arms], edgecolor="black", lw=0.6, label=lbl)
        for x, (k, _, _) in zip(xs + off, arms):
            vals = [data[k][sd][T] for sd in seeds]
            ax.plot([x] * len(vals), vals, "o", color="black", ms=3, alpha=0.7, zorder=5)
        for x, m in zip(xs + off, means):
            ax.annotate(f"{m:.3f}", (x, m), textcoords="offset points", xytext=(0, 12),
                        ha="center", fontsize=7)

    ax.axhline(hub, color=FLOOR_COLOR, ls="-", lw=1.8)
    ax.axhline(last_obs, color="#888888", ls="-.", lw=1.0)
    ax.axhline(chance, color=FLOOR_COLOR, ls=":", lw=1.4)
    ax.text(0.5, hub + 0.045,
            f"hub baseline = {hub:.3f}  <-- READ EVERY BAR AGAINST THIS FLOOR\n"
            f"last-observation baseline = {last_obs:.3f} (dash-dot)\n"
            f"chance = 1/8 = {chance:.3f} (dotted)",
            fontsize=7.5, va="bottom", ha="center", color=FLOOR_COLOR,
            bbox=dict(fc="white", ec="#cccccc", alpha=0.92))

    ax.set_xticks(xs)
    ax.set_xticklabels([lbl for _, lbl, _ in arms], fontsize=7.5)
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("held-out accuracy at revisited nodes (fraction correct)")
    ax.set_title("Family tree, depth 5 (63 nodes), 8 relational actions, n=3 seeds:\n"
                 "non-commutativity buys +0.005 / +0.014; path integration buys +0.115",
                 fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    _caption(fig,
             "Source: runs/family_tree/seed*/ per-seed JSONs and FAMILY_TREE_GATES.json. Error "
             "bars are standard deviation across the 3 seeds (ddof=1), not sem; black dots are "
             "the individual seeds. Measured non-commutativity of the structure is 1.000. The "
             "hub baseline (always answer the most-visited node) is the floor to read against, "
             "not chance -- shallow nodes are revisited more often.")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(os.path.join(OUT, "fig4_family_tree.png"))
    plt.close(fig)


# ---------------------------------------------------------------- figure 5
def fig_stitch_geometry():
    from mapformer.environment_stitch import StitchWorld
    env = StitchWorld(h=8, w=6, n_obs_types=15, patch=3, seed=0)
    rng = np.random.RandomState(0)
    grid, regA, regB, (oy, ox) = env._build(rng)
    P, h, w = env.patch, env.h, env.w
    H, W = grid.shape

    # geometry checks (reported, not assumed)
    checks = {
        "combined shape == (2h-P, 2w-P)": grid.shape == (2 * h - P, 2 * w - P),
        "|room A| == h*w": len(regA) == h * w,
        "|room B| == h*w": len(regB) == h * w,
        "|A n B| == P*P": len(regA & regB) == P * P,
        "A n B is exactly the P x P patch at (h-P, w-P)":
            (regA & regB) == {(oy + i, ox + j) for i in range(P) for j in range(P)},
        "confounding patch identical to shared patch":
            np.array_equal(grid[:P, :P], grid[oy:oy + P, ox:ox + P]),
        "no walls inside A u B":
            all(grid[c] >= 0 for c in (regA | regB)),
        "walls exist outside A u B":
            (grid < 0).sum() == H * W - len(regA | regB),
    }

    disp = np.ma.masked_where(grid < 0, grid)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4))

    ax = axes[0]
    cmap = plt.get_cmap("tab20", env.n_obs_types)
    cmap.set_bad("#e8e8e8")
    im = ax.imshow(disp, cmap=cmap, vmin=0, vmax=env.n_obs_types - 1,
                   interpolation="nearest")
    for i in range(H):
        for j in range(W):
            if grid[i, j] >= 0:
                ax.text(j, i, str(grid[i, j]), ha="center", va="center", fontsize=6.5,
                        color="black")
    ax.add_patch(Rectangle((-0.5, -0.5), w, h, fill=False, ec="#1b6ca8", lw=2.5,
                           label=f"room A ({h}x{w})"))
    ax.add_patch(Rectangle((ox - 0.5, oy - 0.5), w, h, fill=False, ec="#e08214",
                           lw=2.5, ls="--", label=f"room B ({h}x{w})"))
    ax.add_patch(Rectangle((ox - 0.5, oy - 0.5), P, P, fill=False, ec="#178a3a",
                           lw=3, label=f"SHARED {P}x{P} patch (the true join)"))
    ax.add_patch(Rectangle((-0.5, -0.5), P, P, fill=False, ec="#c8553d", lw=3,
                           label=f"CONFOUNDING {P}x{P} patch (identical, wrong)"))
    ax.set_xticks(range(W)); ax.set_yticks(range(H))
    ax.set_xlabel("column (cell index)"); ax.set_ylabel("row (cell index)")
    ax.set_title("Stitched coordinate space, cell colour/number = observation id")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12),
              framealpha=0.95, ncol=2)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.035, label="observation id (0..14); grey = wall")

    ax = axes[1]
    ax.axis("off")
    lines = ["Geometry checks against the docstring", ""]
    for k, v in checks.items():
        lines.append(("  PASS  " if v else "  FAIL  ") + k)
    lines += ["", f"combined grid: {H} x {W} = {H*W} cells, "
                  f"{len(regA | regB)} floor + {(grid < 0).sum()} wall",
              f"room A origin (0,0); room B origin ({oy},{ox})",
              f"shared patch rows {oy}..{oy+P-1}, cols {ox}..{ox+P-1}",
              f"confounding patch rows 0..{P-1}, cols 0..{P-1}",
              "",
              "Task: walk room A (obs revealed), then room B (obs revealed),",
              "then start inside ONE of the two identical patches with obs",
              "WITHHELD and predict them. From the SHARED patch, walking on",
              "enters room B's cells; from the CONFOUNDING patch, walking on",
              "stays in room A. The two starts are locally indistinguishable",
              "and have different correct answers -- the task's own negative",
              "control (George et al., bioRxiv 10.1101/864421v4).",
              "",
              "Chance for a single held-out observation prediction:",
              f"    1 / n_obs_types = 1/{env.n_obs_types} = {1/env.n_obs_types:.4f}",
              "(drawn on any accuracy plot of this task; this panel is",
              " geometry only, so no accuracy is shown.)"]
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=8,
            family="monospace", transform=ax.transAxes)

    fig.suptitle("environment_stitch.StitchWorld._build() -- what the transitive-inference "
                 "task actually looks like (seed 0)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(OUT, "fig5_stitch_geometry.png"))
    plt.close(fig)
    return checks


# ---------------------------------------------------------------- figure 6
def fig_lap_circuit():
    from mapformer.environment_lap import LapWorld
    env = LapWorld(n_obs_types=40, n_laps=4, size=64, wh_range=(3, 8), seed=0)
    rng = np.random.RandomState(0)
    circuit = env._circuit(rng)
    L = len(circuit)

    # walk the circuit on the torus, K laps
    K = env.n_laps
    x = y = 0
    xs, ys = [x], [y]
    for lap in range(K):
        for a in circuit:
            dx, dy = LapWorld.ACTION_DELTAS[a]
            x, y = x + dx, y + dy
            xs.append(x); ys.append(y)
    xs, ys = np.array(xs), np.array(ys)
    net = (int(xs[L] - xs[0]), int(ys[L] - ys[0]))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    ax = axes[0]
    # plot the lap path first so autoscale sees it, then overlay direction arrows
    ax.plot(ys[:L + 1], xs[:L + 1], "-", color="#cccccc", lw=1)
    for i in range(L):
        ax.annotate("", xy=(ys[i + 1], xs[i + 1]), xytext=(ys[i], xs[i]),
                    arrowprops=dict(arrowstyle="->", color=plt.cm.plasma(i / L),
                                    lw=2.2))
    ax.plot(ys[0], xs[0], "o", ms=11, color="#178a3a", zorder=5)
    ax.text(ys[0], xs[0], "  start = end\n  (net displacement 0)", fontsize=8,
            va="center", color="#178a3a")
    ax.set_xticks(range(int(ys[:L + 1].min()), int(ys[:L + 1].max()) + 1))
    ax.set_yticks(range(int(xs[:L + 1].min()), int(xs[:L + 1].max()) + 1))
    ax.margins(0.18)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_xlabel("torus column offset (cells)")
    ax.set_ylabel("torus row offset (cells)")
    ax.set_title(f"one closed lap: right x{circuit.count(3)}, down x{circuit.count(1)}, "
                 f"left x{circuit.count(2)}, up x{circuit.count(0)}\n"
                 f"loop_len = {L} actions, net displacement = {net}")

    ax = axes[1]
    ax.plot(np.arange(len(xs)), xs, "-", color=PI_COLOR, lw=1.6,
            label="cumulative row displacement")
    ax.plot(np.arange(len(ys)), ys, "-", color="#e08214", lw=1.6,
            label="cumulative column displacement")
    ax.axhline(0, color=FLOOR_COLOR, ls="--", lw=1.4)
    ax.set_ylim(-1.4, max(xs.max(), ys.max()) + 1.6)
    ax.text(len(xs) * 0.5, -1.3,
            "net displacement = 0 at every lap boundary  =>  theta_{t+loop_len} = theta_t",
            fontsize=7.5, va="bottom", ha="center", color=FLOOR_COLOR)
    for lap in range(1, K + 1):
        ax.axvline(lap * L, color="#999999", ls=":", lw=1)
        ax.text(lap * L, ax.get_ylim()[1], f" lap {lap} end", rotation=90, fontsize=6.5,
                va="top", color="#777777")
    ax.legend(fontsize=7.5, loc="upper left")
    ax.set_xlabel("action index within episode (actions, not tokens)")
    ax.set_ylabel("displacement from start (cells)")
    ax.set_title(f"{K} laps: the position code returns to itself every lap,\n"
                 "so laps are indistinguishable by theta alone -- that is the test")

    _caption(fig,
             "Source: environment_lap.LapWorld._circuit()/ACTION_DELTAS, seed 0 (w,h are "
             "redrawn per episode, so loop_len varies and the positional shortcut stays dead). "
             "No accuracy is plotted here. For reference, the lap task's floors from "
             "validate_lap.py are: always-no 0.000, always-yes 0.000, random-boundary "
             "exact = 1/n_laps = 0.250, oracle 1.000.")
    fig.suptitle("environment_lap.LapWorld -- a closed circuit whose net displacement is "
                 "exactly zero", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    fig.savefig(os.path.join(OUT, "fig6_lap_circuit.png"))
    plt.close(fig)
    return net, L


def main():
    fig_timing(); print("wrote fig1_timing.png")
    fig_match_query(); print("wrote fig2_match_query_scale.png")
    fig_em_p0(); print("wrote fig3_match_query_em_p0.png")
    fig_family_tree(); print("wrote fig4_family_tree.png")
    checks = fig_stitch_geometry()
    print("wrote fig5_stitch_geometry.png; geometry checks:",
          {k: bool(v) for k, v in checks.items()})
    net, L = fig_lap_circuit()
    print(f"wrote fig6_lap_circuit.png; loop_len={L} net_displacement={net}")


if __name__ == "__main__":
    main()
