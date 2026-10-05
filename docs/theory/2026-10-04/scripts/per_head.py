"""Per-head version: for each head h, kappa_h (w-weighted per-move drift / map axis norm, drift WRAPPED to (-pi,pi]
per channel), cos_h (max |cos| between map axes within head), head share of q.k weight. 'best head' = head with
the lowest max(kappa_h, cos_h). Post hoc, CPU."""
import json, numpy as np, torch
torch.set_num_threads(4)
from pathlib import Path
from mapformer.train_variant import VARIANT_MAP
import importlib.util, sys
spec = importlib.util.spec_from_file_location("cm", str(Path(__file__).with_name("common_mode.py")))
R = Path("/home/prashr/mapformer")

@torch.no_grad()
def ph(cp):
    b = torch.load(cp, map_location="cpu", weights_only=False); c = b["config"]; arm = b["variant"]
    m = VARIANT_MAP[arm](vocab_size=c["vocab_size"], d_model=c["d_model"], n_heads=c["n_heads"],
                         n_layers=c["n_layers"], grid_size=c["grid_size"]).eval()
    m.load_state_dict(b["model_state_dict"])
    V = c["vocab_size"]; D = c.get("n_dims", 2); K = c.get("n_obs_types", 16); pe = c.get("p_empty", 0.5); nA = 2 * D
    e = m.token_emb(torch.arange(V)); a = (m.action_to_lie(e[None])[0] * m.path_integrator.omega[None]).numpy()
    L = m.layers[0]; H = c["n_heads"]; dh = c["d_model"] // H; h = L.norm1(e)
    Q = L.q_proj(h).view(V, H, dh); Kk = L.k_proj(h).view(V, H, dh)
    qa = torch.sqrt(Q[..., 0::2] ** 2 + Q[..., 1::2] ** 2); ka = torch.sqrt(Kk[..., 0::2] ** 2 + Kk[..., 1::2] ** 2)
    w = (qa[:nA].mean(0) * ka[nA:].mean(0)).numpy(); share = w.sum(1) / w.sum()
    A = a[:nA]; O = a[nA:]
    U = np.stack([(A[2 * d] - A[2 * d + 1]) / 2 for d in range(D)])
    mm = A.mean(0) + pe * O[K] + (1 - pe) * O[:K].mean(0)
    mmw = np.angle(np.exp(1j * mm))           # wrap per-move drift per channel
    out = []
    for hh in range(H):
        sw = np.sqrt(w[hh] / w[hh].sum())
        Uw = U[:, hh] * sw; un = np.linalg.norm(Uw, axis=1).mean()
        kap = np.linalg.norm(mmw[hh] * sw) / un
        cs = max(abs(Uw[i] @ Uw[j]) / (np.linalg.norm(Uw[i]) * np.linalg.norm(Uw[j]) + 1e-12) for i in range(D) for j in range(i + 1, D))
        out.append((kap, cs, share[hh], un))
    losses = np.asarray(b["losses"], float); fl = losses[-max(1, len(losses) // 20):].mean()
    return out, fl

sets = [("runs/rank_mi/p0", "Vanilla", "2sh"), ("runs/rank_mi/p0", "Vanilla_r2ph", "2ph"), ("runs/rank_sep/p0", "Vanilla_r4mibd", "2bd"),
        ("runs/rank3/p0", "Vanilla_r3ph", "3ph"), ("runs/rank_mi/p0", "Vanilla_r4mi", "4sh"), ("runs/rank_sep/p0", "Vanilla_r4ph", "4ph"),
        ("runs/rank_nd/D2", "Vanilla_r2ph", "D2r2"), ("runs/rank_nd/D2", "Vanilla_r3ph", "D2r3"),
        ("runs/rank_nd/D3", "Vanilla_r3ph", "D3r3"), ("runs/rank_nd/D3", "Vanilla_r4ph", "D3r4")]
res = []
for d, arm, lab in sets:
    for s in range(8):
        o, fl = ph(R / d / f"{arm}_s{s}" / f"{arm}.pt")
        best = min(o, key=lambda t: max(t[0], t[1]))
        res.append((lab, s, fl < 0.05, fl, best, o))
        print(f"{lab:5s} s{s} {'S' if fl<0.05 else '.'} loss {fl:.3f} | " + " | ".join(f"h{i}: kap {k:.3f} cos {c:.3f} share {sh:.2f}" for i, (k, c, sh, un) in enumerate(o))
              + f" || best max(kap,cos) {max(best[0], best[1]):.3f}")
sc = lambda r: max(r[4][0], r[4][1])
S = sorted(sc(r) for r in res if r[2]); U = sorted(sc(r) for r in res if not r[2])
print("best-head score solved:", np.round(S, 3)); print("unsolved:", np.round(U, 3))
from sklearn.metrics import roc_auc_score
y = [r[2] for r in res]; print("AUC (lower score -> solved):", roc_auc_score(y, [-sc(r) for r in res]))
