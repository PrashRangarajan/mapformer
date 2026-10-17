"""[rank variant: fig_loss_curves_rank.py] Training loss per epoch, signed vs |.| (and softplus), the headline contrast (post hoc re-plot of stored per-epoch
losses; the SOLVED classes are the registered ones: final-5% mean training loss < 0.05, stats_core.classify_run).
Per run a thin line, per arm the median across runs (thick). Training loss is measured with dropout on.
Panels (each a registered batch; all arms of a panel trained in that batch):
  (a) torus T = 1024, 900 epochs, 1 layer d 128: SIGN_MATCHED Signed_r4 / Abs_r4 / Pos_r4 (seeds 0-7)
  (b) ring 32 cells, T = 128, 300 epochs, observation-step-free, from scratch: WRAP_EXISTENCE SO / AO (seeds 600-607)
  (c) torus T = 1024, 900 epochs, observation-step-free: GENERALITY SO4 / AO4 (seeds 700-707)
  (d) torus T = 1024, 900 epochs, observation-step-free, 2 layers d 256: ROBUST SO / AO (seeds 1400-1407)
  cd /home/prashr && CUDA_VISIBLE_DEVICES="" PYTHONPATH=/home/prashr python3 mapformer/docs/paper/figs/fig_loss_curves.py"""
import glob, os
import numpy as np, torch
from figstyle import plt, SIGNED, ABS, SOFTPLUS, INK2, MUTED, FLOOR_LS, save, write_out
from mapformer.stats_core import classify_run

R = "/home/prashr/mapformer/runs"; HERE = os.path.dirname(os.path.abspath(__file__))
PANELS = [
    ("(a) torus, T 1024, 1 layer\nSIGN_MATCHED (registered)", [
        ("signed", SIGNED, f"{R}/sign_matched/p0/Signed_r4_s*/Signed_r4.pt"),
        ("|.|", ABS, f"{R}/sign_matched/p0/Abs_r4_s*/Abs_r4.pt"),
        ("softplus", SOFTPLUS, f"{R}/sign_matched/p0/Pos_r4_s*/Pos_r4.pt")]),
    ("(b) ring, T 128, obs.-step-free\nWRAP_EXISTENCE (registered)", [
        ("signed", SIGNED, f"{R}/wrap_existence/p0/SO_s60[0-7]/SO.pt"),
        ("|.|", ABS, f"{R}/wrap_existence/p0/AO_s60[0-7]/AO.pt")]),
    ("(c) torus, T 1024, obs.-step-free\nGENERALITY (registered)", [
        ("signed", SIGNED, f"{R}/generality/torus/SO4/p0/SignedAO4_r4_s70[0-7]/SignedAO4_r4.pt"),
        ("|.|", ABS, f"{R}/generality/torus/AO4/p0/AbsAO4_r4_s70[0-7]/AbsAO4_r4.pt")]),
    ("(d) torus, 2 layers, d 256\nROBUST (registered)", [
        ("signed", SIGNED, f"{R}/robust/torus/SO/p0/SignedAO4_r4_s14*/SignedAO4_r4.pt"),
        ("|.|", ABS, f"{R}/robust/torus/AO/p0/AbsAO4_r4_s14*/AbsAO4_r4.pt")]),
]

fig, axs = plt.subplots(1, 4, figsize=(7.0, 2.25), sharey=True)
out = ["Training loss curves (dropout on), per run final-5% mean loss and registered class; median curve at checkpoints."]
for ax, (title, arms) in zip(axs, PANELS):
    out.append(f"\n{title.replace(chr(10), ' -- ')}")
    keys = []
    for lab, col, pat in arms:
        cks = sorted(glob.glob(pat)); assert len(cks) == 8, (pat, len(cks))
        L, ns = [], 0
        for ck in cks:
            b = torch.load(ck, map_location="cpu", weights_only=False)
            l = np.asarray(b["losses"], float); L.append(l); c = classify_run(l)
            ns += c["registered"] == "SOLVED"
            out.append(f"  {lab:8s} {os.path.basename(os.path.dirname(ck)):22s} epochs {len(l)}  final-5% {c['tail']:.2e}  "
                       f"min {l.min():.2e}  {c['cls']}")
        L = np.stack(L); E = L.shape[1]; ep = np.arange(1, E + 1)
        for l in L:
            ax.plot(ep, l, color=col, lw=0.45, alpha=0.4)
        med = np.median(L, 0)
        ax.plot(ep, med, color=col, lw=1.6)
        marks = [E // 10, E // 4, E // 2, 3 * E // 4, E]
        cross = np.nonzero(med < 0.05)[0]
        out.append(f"  {lab:8s} median first below 0.05 at epoch {cross[0] + 1 if len(cross) else 'never'}")
        out.append(f"  {lab:8s} SOLVED {ns}/8; median loss at epochs {marks}: " + " ".join(f"{med[m - 1]:.2e}" for m in marks))
        keys.append((lab, col, ns))
    E = L.shape[1]
    ax.axhline(0.05, color=MUTED, lw=0.7, ls=FLOOR_LS, zorder=1)
    ax.set_yscale("log"); ax.set_ylim(1e-6, 5); ax.set_xlim(0, E)
    ax.set_xticks([0, E // 3, 2 * E // 3, E]); ax.set_xlabel("epoch", labelpad=1)
    ax.set_title(title, fontsize=6.5, loc="left", pad=3, color="#0b0b0b")
    ax.grid(axis="x", visible=False)
    for i, (lab, col, ns) in enumerate(keys):
        y = 0.05 + 0.085 * (len(keys) - 1 - i)
        ax.plot([0.04, 0.12], [y + 0.028, y + 0.028], transform=ax.transAxes, color=col, lw=1.6)
        ax.text(0.14, y, f"{lab}  {ns}/8", transform=ax.transAxes, fontsize=5.8, va="bottom", color=INK2)
    ax.minorticks_off(); ax.set_yticks([1e-6, 1e-4, 1e-2, 1])
axs[0].set_ylabel("training loss (log)")
axs[1].text(0.98, 0.058, "SOLVED: loss < 0.05", ha="right", transform=axs[1].get_yaxis_transform(), fontsize=5.4,
            va="bottom", color=INK2)
fig.text(0.995, 0.005, "thin: one run; thick: median of 8 runs; n/8 = runs SOLVED (final-5% mean training loss < 0.05)", ha="right", va="bottom", fontsize=5.4,
         color=MUTED)
fig.subplots_adjust(left=0.065, right=0.985, top=0.83, bottom=0.17, wspace=0.16)
save(fig, HERE, "fig_loss_curves"); write_out(HERE, "fig_loss_curves", out)
