"""Shared style for the paper figures (dataviz reference palette, light mode, print).
Categorical slots in fixed order: 1 blue, 2 orange, 3 aqua (the three slots that validate all-pairs for CVD in the
reference palette); further identity is carried by marker shape and direct labels, never by colour alone.
Within every figure: signed step / path integration = slot 1, |.| step / index RoPE = slot 2, softplus = slot 3."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
YELLOW, MAGENTA, VIOLET = "#eda100", "#e87ba4", "#6250d6"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
# ordinal blue / orange ramps (light surface: start no lighter than step 250)
BLUE_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
ORANGE_RAMP = ["#f4a582", "#ef8a5a", "#eb6834", "#c4501f", "#8f3a16"]
SIGNED, ABS, SOFTPLUS = BLUE, ORANGE, AQUA
FLOOR_LS = (0, (3, 2))

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.3, "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS,
    "axes.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True, "legend.frameon": False,
    "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round", "savefig.dpi": 300,
    "pdf.fonttype": 42, "ps.fonttype": 42, "figure.facecolor": "white", "axes.facecolor": "white",
})


def floor_line(ax, y, label, x=1.0, ha="right", va="bottom", color=MUTED, fs=5.6, **kw):
    """A map-free floor: thin dashed muted line with a direct label (ink, not the line colour)."""
    ax.axhline(y, color=color, lw=0.7, ls=FLOOR_LS, zorder=1, **kw)
    ax.text(x, y, label, transform=ax.get_yaxis_transform(), ha=ha, va=va, fontsize=fs, color=INK2)


def save(fig, here, name):
    for ext in ("pdf", "png"):
        fig.savefig(f"{here}/{name}.{ext}", dpi=300)


def write_out(here, name, lines):
    txt = "\n".join(lines)
    print(txt)
    open(f"{here}/{name}_out.txt", "w").write(txt + "\n")
