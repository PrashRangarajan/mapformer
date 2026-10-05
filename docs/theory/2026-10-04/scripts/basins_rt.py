"""Post hoc, CPU. Per head h of each 1-layer path model: a = omega * W_out W_in emb(tok) (H, nb).
drift m = mean_actions a + p_empty a(blank) + (1-p_empty) mean_regular a(obs)  -- per-move common step, wrapped to (-pi,pi].
axes u_d = (a(+d) - a(-d))/2.  Channel weights w = mean_action-queries |q_pair| * mean_obs-keys |k_pair| (exact at layer 1).
kappa_h = ||m||_w / mean_d ||u_d||_w ;  indep_h = s_min/s_max of the w-weighted (D x nb) axis matrix (1 = orthogonal, equal norm).
Basin of a run, from its best head (max over heads of indep among heads with kappa<=0.01):
  CLEAN  some head has kappa<=0.01 and indep>=0.2 ; COLLAPSE some head kappa<=0.01 but every such head indep<0.2 ; CLOCK no head kappa<=0.01."""
import json, numpy as np, torch
torch.set_num_threads(4)
from pathlib import Path
from mapformer.train_variant import VARIANT_MAP
R = Path("/home/prashr/mapformer")

@torch.no_grad()
def heads(cp, weighted=True):
    b = torch.load(cp, map_location="cpu", weights_only=False); c = b["config"]; arm = b["variant"]
    m = VARIANT_MAP[arm](vocab_size=c["vocab_size"], d_model=c["d_model"], n_heads=c["n_heads"],
                         n_layers=c["n_layers"], grid_size=c["grid_size"]).eval()
    m.load_state_dict(b["model_state_dict"])
    V = c["vocab_size"]; D = c.get("n_dims", 2); K = c.get("n_obs_types", 16); pe = c.get("p_empty", 0.5); nA = 2 * D
    e = m.token_emb(torch.arange(V)); a = (m.action_to_lie(e[None])[0] * m.path_integrator.omega[None]).numpy()
    L = m.layers[0]; H = c["n_heads"]; dh = c["d_model"] // H; h = L.norm1(e)
    Q = L.q_proj(h).view(V, H, dh); Kk = L.k_proj(h).view(V, H, dh)
    qa = torch.sqrt(Q[..., 0::2] ** 2 + Q[..., 1::2] ** 2); ka = torch.sqrt(Kk[..., 0::2] ** 2 + Kk[..., 1::2] ** 2)
    w = (qa[:nA].mean(0) * ka[nA:].mean(0)).numpy() if weighted else np.ones_like(a[0])
    A = a[:nA]; O = a[nA:]
    U = np.stack([(A[2 * d] - A[2 * d + 1]) / 2 for d in range(D)])
    mo = pe * O[K] + (1 - pe) * O[:K].mean(0)
    RT = np.stack([np.angle(np.exp(1j * (A[2 * d] + A[2 * d + 1] + 2 * mo))) for d in range(D)])  # retrace drift per axis
    out = []
    for hh in range(H):
        sw = np.sqrt(w[hh] / w[hh].sum()); Uw = U[:, hh] * sw
        sv = np.linalg.svd(Uw, compute_uv=False)
        out.append((max(np.linalg.norm(RT[d, hh] * sw) for d in range(D)) / np.linalg.norm(Uw, axis=1).mean(), sv[-1] / sv[0]))
    fl = np.asarray(b["losses"], float); fl = fl[-max(1, len(fl) // 20):].mean()
    return out, fl

def basin(o):
    ok = [ind for k, ind in o if k <= 0.01]
    if not ok: return "CLOCK"
    return "CLEAN" if max(ok) >= 0.2 else "COLLAPSE"

sets = [("runs/rank_mi/p0", "Vanilla", "torus r2 shared"), ("runs/rank_mi/p0", "Vanilla_r2ph", "torus r2/head"),
        ("runs/rank_sep/p0", "Vanilla_r4mibd", "torus r4 blockdiag (2/head)"), ("runs/rank3/p0", "Vanilla_r3ph", "torus r3/head"),
        ("runs/rank_mi/p0", "Vanilla_r4mi", "torus r4 shared"), ("runs/rank_sep/p0", "Vanilla_r4ph", "torus r4/head"),
        ("runs/rank_nd/D2", "Vanilla_r2ph", "ND D2 r2/head"), ("runs/rank_nd/D2", "Vanilla_r3ph", "ND D2 r3/head"),
        ("runs/rank_nd/D3", "Vanilla_r3ph", "ND D3 r3/head"), ("runs/rank_nd/D3", "Vanilla_r4ph", "ND D3 r4/head"), ("runs/loop_rank_e1800/p0","Vanilla","e1800 r2 shared"), ("runs/loop_rank_e1800/p0","Vanilla_r4mi","e1800 r4 shared"), ("runs/sign_matched/p0","Signed_r4","sign Signed"), ("runs/sign_matched/p0","Abs_r4","sign Abs"), ("runs/sign_matched/p0","Pos_r4","sign Pos")]
tab = {}
for wt in (True, False):
    cnt = {}
    for d, arm, lab in sets:
        row = []
        for s in range(8):
            o, fl = heads(R / d / f"{arm}_s{s}" / f"{arm}.pt", wt)
            bs = basin(o); sv = fl < 0.05
            cnt[(bs, sv)] = cnt.get((bs, sv), 0) + 1
            row.append(f"{bs[:3]}{'+' if sv else '-'}")
        if wt: print(f"{lab:28s} " + " ".join(row))
    print(("weighted" if wt else "unweighted"), "basin x SOLVED:", {f"{k[0]}/{'S' if k[1] else 'U'}": v for k, v in sorted(cnt.items())})
