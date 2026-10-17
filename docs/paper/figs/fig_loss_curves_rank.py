"""Training loss per epoch by per-head rank of the content-to-angle map (torus, T = 1024, 900 epochs, seeds 0-7; post hoc
re-plot of stored per-epoch losses; classes are the registered ones: final-5% mean training loss < 0.05). Every arm is
built from our r = 2's initial weights at each seed. Panels: (a) shared latent, r 2 (Vanilla) vs r 4 (Vanilla_r4mi),
runs/rank_mi; (b) per-head latents r 2 (runs/rank_mi Vanilla_r2ph), r 3 (runs/rank3 Vanilla_r3ph), r 4 (runs/rank_sep
Vanilla_r4ph). Training loss is measured with dropout on.
  cd /home/prashr && CUDA_VISIBLE_DEVICES="" PYTHONPATH=/home/prashr python3 mapformer/docs/paper/figs/fig_loss_curves_rank.py"""
import glob, os
import numpy as np, torch
from figstyle import plt, BLUE, ORANGE, AQUA, INK2, MUTED, FLOOR_LS, save, write_out
from mapformer.stats_core import classify_run

R = "/home/prashr/mapformer/runs"; HERE = os.path.dirname(os.path.abspath(__file__))
PANELS = [
    ("(a) torus T 1024, shared latent\nRANK_MI (registered)", [
        ("rank 2", ORANGE, f"{R}/rank_mi/p0/Vanilla_s*/Vanilla.pt"),
        ("rank 4", BLUE, f"{R}/rank_mi/p0/Vanilla_r4mi_s*/Vanilla_r4mi.pt")]),
    ("(b) torus T 1024, per-head latents\nRANK_SEP / RANK3 / RANK_MI (reg.)", [
        ("rank 2", ORANGE, f"{R}/rank_mi/p0/Vanilla_r2ph_s*/Vanilla_r2ph.pt"),
        ("rank 3", AQUA, f"{R}/rank3/p0/Vanilla_r3ph_s*/Vanilla_r3ph.pt"),
        ("rank 4", BLUE, f"{R}/rank_sep/p0/Vanilla_r4ph_s*/Vanilla_r4ph.pt")]),
]

fig, axs = plt.subplots(1, 2, figsize=(4.2, 2.25), sharey=True)
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
    ax.set_yscale("log"); ax.set_ylim(1e-4, 5); ax.set_xlim(0, E)
    ax.set_xticks([0, E // 3, 2 * E // 3, E]); ax.set_xlabel("epoch", labelpad=1)
    ax.set_title(title, fontsize=6.5, loc="left", pad=3, color="#0b0b0b")
    ax.grid(axis="x", visible=False)
    for i, (lab, col, ns) in enumerate(keys):
        y = 0.05 + 0.085 * (len(keys) - 1 - i)
        ax.plot([0.04, 0.12], [y + 0.028, y + 0.028], transform=ax.transAxes, color=col, lw=1.6)
        ax.text(0.14, y, f"{lab}  {ns}/8", transform=ax.transAxes, fontsize=5.8, va="bottom", color=INK2)
    ax.minorticks_off(); ax.set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1])
axs[0].set_ylabel("training loss (log)")
axs[0].text(0.02, 0.044, "SOLVED: loss < 0.05", ha="left", transform=axs[0].get_yaxis_transform(), fontsize=5.4,
            va="top", color=INK2)
fig.text(0.995, 0.005, "thin: one run; thick: median of 8; n/8 = SOLVED (final-5% loss < 0.05)", ha="right", va="bottom", fontsize=5.4,
         color=MUTED)
fig.subplots_adjust(left=0.11, right=0.97, top=0.83, bottom=0.17, wspace=0.12)
save(fig, HERE, "fig_loss_curves_rank"); write_out(HERE, "fig_loss_curves_rank", out)
