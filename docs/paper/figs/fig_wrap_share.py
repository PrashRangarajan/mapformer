"""Figure 2 of the workshop outline: per run, the share of channels in which opposite moves cancel mod 2 pi (|wrap(angle(a)
+ angle(opposite a))| < 0.1 rad, over channels with |wrap(angle(a))| > 0.05 rad -- the wrap-invariant selection of
WRAP_EXISTENCE Amendment 1). Ring: RING_DIAL (registered batch, T = 128, 300 epochs); torus: SIGN_MATCHED (T = 1024, 900
epochs). Filled = SOLVED (final-5% training loss < 0.05). Post hoc re-description of stored checkpoints (CPU)."""
import glob, os
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mapformer.train_variant import VARIANT_MAP
from mapformer.stats_core import classify_run

R = "/home/prashr/mapformer/runs"; HERE = os.path.dirname(os.path.abspath(__file__))


def share(ck, pairs, grid):
    b = torch.load(ck, map_location="cpu", weights_only=False); c = b["config"]; arm = b["variant"]
    m = VARIANT_MAP[arm](vocab_size=c["vocab_size"], d_model=128, n_heads=2, n_layers=1, grid_size=grid)
    m.load_state_dict(b["model_state_dict"])
    with torch.no_grad():
        A = (m.action_to_lie(m.token_emb.weight[None])[0] * m.path_integrator.omega).reshape(c["vocab_size"], -1).numpy()
    ws = []
    for a, o in pairs:
        sel = np.abs(np.angle(np.exp(1j * A[a]))) > 0.05
        ws.append(np.abs(np.angle(np.exp(1j * (A[a] + A[o]))))[sel])
    w = np.concatenate(ws)
    return float((w < 0.1).mean()), classify_run(b["losses"])["registered"] == "SOLVED"


groups = []
for p in ("0.5", "0.75", "0.9"):
    for arm in ("Signed_r4", "Abs_r4"):
        cks = sorted(glob.glob(f"{R}/ring_dial/p0/{arm}_p{p}_s*/{arm}.pt"))
        groups.append((f"ring p+ {p}", arm, [share(ck, [(0, 1)], 32) for ck in cks]))
for arm in ("Signed_r4", "Abs_r4", "Pos_r4"):
    cks = sorted(glob.glob(f"{R}/sign_matched/p0/{arm}_s*/{arm}.pt"))
    groups.append(("torus T 1024", arm, [share(ck, [(0, 1), (2, 3)], 64) for ck in cks]))

COL = {"Signed_r4": "C0", "Abs_r4": "C3", "Pos_r4": "C1"}; LAB = {"Signed_r4": "signed", "Abs_r4": "|.|", "Pos_r4": "softplus"}
fig, ax = plt.subplots(figsize=(6.4, 2.7)); xt, xl, gx = [], [], {}
rng = np.random.default_rng(0)
for i, (task, arm, vals) in enumerate(groups):
    x = i + (i // 2 if i < 6 else 3) * 0.6
    for v, solved in vals:
        ax.scatter(x + rng.uniform(-0.15, 0.15), v, s=14, color=COL[arm], facecolors=COL[arm] if solved else "none", lw=0.9)
    xt.append(x); xl.append(LAB[arm]); gx.setdefault(task, []).append(x)
    print(f"{task:13s} {arm:9s} n={len(vals)} share mean {np.mean([v for v, _ in vals]):.2f} (range {min(v for v, _ in vals):.2f}-"
          f"{max(v for v, _ in vals):.2f}); SOLVED {sum(s for _, s in vals)}/{len(vals)}")
ax.set_xticks(xt); ax.set_xticklabels(xl, fontsize=6.5)
for task, xs in gx.items():
    ax.text(np.mean(xs), -0.2, task.replace("p+", "p_plus"), ha="center", va="top", fontsize=7, transform=ax.get_xaxis_transform())
ax.set_xlim(min(xt) - 0.5, max(xt) + 0.5); ax.set_ylim(-0.03, 1.03); ax.tick_params(labelsize=7)
ax.set_ylabel("channels where opposite\nmoves cancel mod 2 pi", fontsize=7.5)
ax.scatter([], [], s=14, color="k", label="SOLVED"); ax.scatter([], [], s=14, facecolors="none", edgecolors="k", label="not solved")
ax.legend(fontsize=6.5, loc="center right", frameon=False)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"{HERE}/fig_wrap_share.{ext}", dpi=200)
