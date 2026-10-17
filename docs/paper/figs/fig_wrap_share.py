"""Fig. 1 of draft 5, redrawn 2026-10-16 (post hoc, eval-only, CPU): per run, the share of rotary channels in which
opposite moves cancel mod 2 pi, USAGE-WEIGHTED (primary) and unweighted (faint secondary).

Selection (both measures): (move, channel) entries with |wrap(a)| > 0.05 rad, where a = omega_c * step_c(move) -- i.e.
channels whose move angle has collapsed to ~0 mod 2 pi are EXCLUDED from the denominator (WRAP_EXISTENCE Amendment 1).
Cancelling: |wrap(a(move) + a(opposite move))| < 0.1 rad. Torus pairs (0, 1) and (2, 3); ring pair (0, 1).
Usage weight of channel c (ACCOUNTS Amendment 1, model_wrap_exist.usage_weights): mean |q_c| over move-token positions x
mean |k_c| over observation-token positions, first layer, eval mode. Weighted share = sum w over cancelling entries / sum w
over selected entries (model_wrap_exist.wrap_share_weighted's formula; re-implemented here for VARIANT_MAP models, whose
angles come from action_to_lie of the move embeddings rather than a step table).
Streams: ring -- GridWorldCancel(32, seed 10000, p_plus as trained), np seed 10**6, 32 walks of T = 128 (ACCOUNTS T1's
stream; p_plus per cell); torus -- GridWorld(64, 16 obs, p_empty 0.5, seed 10000), np seed 1234, 32 walks of T = 1024.
Runs: RING_DIAL (registered batch, ring 32, T = 128, 300 epochs, seeds 30-37) and SIGN_MATCHED (torus T = 1024, 900
epochs, seeds 0-7). Filled = SOLVED (registered: final-5% mean training loss < 0.05).
  cd /home/prashr && CUDA_VISIBLE_DEVICES="" PYTHONPATH=/home/prashr python3 mapformer/docs/paper/figs/fig_wrap_share.py"""
import glob, os
import numpy as np, torch
from figstyle import plt, SIGNED, ABS, SOFTPLUS, INK2, MUTED, save, write_out
from mapformer.train_variant import VARIANT_MAP
from mapformer.stats_core import classify_run
from mapformer.environment import GridWorld
from mapformer.environment_cancel import GridWorldCancel

R = "/home/prashr/mapformer/runs"; HERE = os.path.dirname(os.path.abspath(__file__))
wrap = lambda x: np.angle(np.exp(1j * x))
torch.set_num_threads(4)


def stream_ring(p):
    env = GridWorldCancel(size=32, seed=10000, p_plus=p); assert getattr(env, "action_offset", 0) == 0
    np.random.seed(10 ** 6)
    return torch.stack([env.generate_trajectory(128)[0] for _ in range(32)])


def stream_torus():
    env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, seed=10000); np.random.seed(1234)
    return torch.stack([env.generate_trajectory(1024)[0] for _ in range(32)])


@torch.no_grad()
def usage(m, tok, n_act):
    """model_wrap_exist.usage_weights, inlined for VARIANT_MAP models (same formula; actions are tokens 0..n_act-1)."""
    m.eval(); layer = m.layers[0]
    h = layer.norm1(m.token_emb(tok)); B, T, _ = h.shape
    Q = layer.q_proj(h).view(B, T, layer.n_heads, layer.d_head); K = layer.k_proj(h).view(B, T, layer.n_heads, layer.d_head)
    qn = torch.sqrt(Q[..., 0::2] ** 2 + Q[..., 1::2] ** 2); kn = torch.sqrt(K[..., 0::2] ** 2 + K[..., 1::2] ** 2)
    is_act = tok < n_act
    return (qn[is_act].mean(0) * kn[~is_act].mean(0)).reshape(-1).double().numpy()


def shares(ck, pairs, grid, tok, n_act):
    b = torch.load(ck, map_location="cpu", weights_only=False); c = b["config"]; arm = b["variant"]
    m = VARIANT_MAP[arm](vocab_size=c["vocab_size"], d_model=128, n_heads=2, n_layers=1, grid_size=grid)
    m.load_state_dict(b["model_state_dict"]); m.eval()
    with torch.no_grad():
        A = (m.action_to_lie(m.token_emb.weight[None])[0] * m.path_integrator.omega).reshape(c["vocab_size"], -1)
        A = A.double().numpy()
    w = usage(m, tok, n_act); assert w.shape[0] == A.shape[1]
    nu = du = 0; nw = dw = 0.0; ncol = 0
    for a, o in pairs:
        sel = np.abs(wrap(A[a])) > 0.05; canc = np.abs(wrap(A[a] + A[o])) < 0.1
        nu += (sel & canc).sum(); du += sel.sum(); nw += (w * (sel & canc)).sum(); dw += (w * sel).sum(); ncol += (~sel).sum()
    return dict(unw=nu / du, wtd=nw / dw, solved=classify_run(b["losses"])["registered"] == "SOLVED",
                excluded=ncol / (len(pairs) * A.shape[1]), name=os.path.basename(os.path.dirname(ck)))


groups = []
for p in ("0.5", "0.75", "0.9"):
    tok = stream_ring(float(p))
    for arm in ("Signed_r4", "Abs_r4"):
        cks = sorted(glob.glob(f"{R}/ring_dial/p0/{arm}_p{p}_s*/{arm}.pt")); assert len(cks) == 8
        groups.append((f"ring, p_plus {p}", arm, [shares(ck, [(0, 1)], 32, tok, 2) for ck in cks]))
tok = stream_torus()
for arm in ("Signed_r4", "Abs_r4", "Pos_r4"):
    cks = sorted(glob.glob(f"{R}/sign_matched/p0/{arm}_s*/{arm}.pt")); assert len(cks) == 8
    groups.append(("torus, T 1024", arm, [shares(ck, [(0, 1), (2, 3)], 64, tok, 4) for ck in cks]))

COL = {"Signed_r4": SIGNED, "Abs_r4": ABS, "Pos_r4": SOFTPLUS}; LAB = {"Signed_r4": "signed", "Abs_r4": "|.|", "Pos_r4": "softplus"}
fig, ax = plt.subplots(figsize=(6.4, 2.7)); xt, xl, gx = [], [], {}
rng = np.random.default_rng(0); out = ["Cancelling share per run: usage-weighted (primary) and unweighted (secondary). "
                                        "Denominator excludes (move, channel) entries with |wrap(a)| <= 0.05."]
for i, (task, arm, vals) in enumerate(groups):
    x = i + (i // 2 if i < 6 else 3) * 0.6; col = COL[arm]
    for v in vals:
        j = rng.uniform(-0.13, 0.13)
        ax.plot([x + j, x + j], [v["unw"], v["wtd"]], color=col, lw=0.4, alpha=0.3, zorder=2)
        ax.scatter(x + j, v["unw"], s=7, marker="_", color=col, alpha=0.45, lw=0.8, zorder=2)
        ax.scatter(x + j, v["wtd"], s=16, color=col, facecolors=col if v["solved"] else "white", lw=0.9, zorder=3)
    mw = np.mean([v["wtd"] for v in vals]); mu = np.mean([v["unw"] for v in vals])
    ax.plot([x - 0.25, x + 0.25], [mw, mw], color=INK2, lw=1.0, zorder=4)
    xt.append(x); xl.append(LAB[arm]); gx.setdefault(task, []).append(x)
    ns = sum(v["solved"] for v in vals)
    out.append(f"{task:16s} {LAB[arm]:8s} n={len(vals)} SOLVED {ns}/8; weighted mean {mw:.2f} (range "
               f"{min(v['wtd'] for v in vals):.2f}-{max(v['wtd'] for v in vals):.2f}); unweighted mean {mu:.2f} (range "
               f"{min(v['unw'] for v in vals):.2f}-{max(v['unw'] for v in vals):.2f}); excluded (collapsed) entries "
               f"{np.mean([v['excluded'] for v in vals]):.2f}")
    for v in vals:
        out.append(f"    {v['name']:22s} weighted {v['wtd']:.3f} unweighted {v['unw']:.3f} {'SOLVED' if v['solved'] else '-'}")
    sv = [v for v in vals if v["solved"]]
    if sv and arm != "Signed_r4":
        out.append(f"    solved {LAB[arm]} runs: weighted " + ", ".join(f"{v['wtd']:.2f}" for v in sv))
ax.set_xticks(xt); ax.set_xticklabels(xl, fontsize=6.5)
for task, xs in gx.items():
    ax.text(np.mean(xs), -0.12, task, ha="center", va="top", fontsize=6.8, color=INK2, transform=ax.get_xaxis_transform())
ax.set_xlim(min(xt) - 0.5, max(xt) + 0.5); ax.set_ylim(-0.03, 1.03); ax.grid(axis="x", visible=False)
ax.set_ylabel("share of channels where opposite\nmoves cancel mod 2 pi")
H = [plt.Line2D([], [], color=MUTED, marker="o", ls="", ms=4, label="usage-weighted, SOLVED run"),
     plt.Line2D([], [], color=MUTED, marker="o", mfc="white", ls="", ms=4, label="usage-weighted, not solved"),
     plt.Line2D([], [], color=MUTED, marker="_", ls="", ms=5, alpha=0.6, label="unweighted (same run)"),
     plt.Line2D([], [], color=INK2, lw=1.0, label="arm mean, weighted")]
ax.legend(handles=H, fontsize=6, loc="center left", bbox_to_anchor=(1.0, 0.5))
fig.subplots_adjust(left=0.1, right=0.74, top=0.96, bottom=0.17)
save(fig, HERE, "fig_wrap_share"); write_out(HERE, "fig_wrap_share", out)
