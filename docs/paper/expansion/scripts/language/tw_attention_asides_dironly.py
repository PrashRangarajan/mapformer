"""tw_attention_asides.py restricted to the 8 DirOnly models (run in parallel to save wall time; same functions, same
stream, same lift vectors' construction; the random generator for the lift vector is re-seeded per model here, so the
off-plane vector differs from the main run's draw for the same model)."""
import sys, json, numpy as np, torch
sys.path.insert(0, "/home/prashr/mapformer/docs/paper/expansion/scripts/language")
import tw_attention_asides as A
from tw_lib import load, ck_ns
torch.set_num_threads(4)
te, W = A.walks(); M = [A.meta(te, g) for g in W]
out = {}
for s in range(10, 18):
    name = f"DirOnly_s{s}"; m, _ = load(ck_ns("DirOnly", s)); rng = np.random.default_rng(100 + s)
    D, oppw = A.raw_moves(m, te, W)
    r = {"opp_wrapaware": oppw, "intact": A.score(m, te, W, M, attn=True), "intact_x1": A.score(m, te, W, M, scale=1.0)["acc"]}
    H, nb = m.path_integrator.omega.shape
    B = np.linalg.qr(D.T)[0]
    for tag, scl in (("lift_off1", 1.0), ("lift_off0.25", 0.25)):
        g0 = rng.standard_normal(D.shape[1]); off = g0 - B @ (B.T @ g0); off = off / np.linalg.norm(off) * np.linalg.norm(D[0]) * scl
        r[tag] = A.score(m, te, W, M, mode="lift", v=torch.tensor(off, dtype=torch.float32).view(H, nb))
    r["lift_in0.5"] = A.score(m, te, W, M, mode="lift", v=torch.tensor(0.5 * D[0], dtype=torch.float32).view(H, nb))
    out[name] = r; a = r["intact"]
    print(f"{name} oppW {oppw:.3f} | acc {a['acc']:.4f} (x1 {r['intact_x1']:.4f}) errs {a['err']} aside {a['aside_err_share']:.2f}"
          f" | attn last {a['attn']['obj_last']:.3f} old {a['attn']['obj_old']:.3f} anoun {a['attn']['anoun']:.3f} same {a['attn']['same_other']:.3f} rest {a['attn']['rest']:.3f}"
          f" | lift-off(1 move) {r['lift_off1']['acc']:.4f} ({r['lift_off1']['aside_err_share']:.2f}) lift-off(0.25) {r['lift_off0.25']['acc']:.4f} ({r['lift_off0.25']['aside_err_share']:.2f})"
          f" lift-in(0.5) {r['lift_in0.5']['acc']:.4f} ({r['lift_in0.5']['aside_err_share']:.2f})", flush=True)
json.dump(out, open("/home/prashr/mapformer/docs/paper/expansion/scripts/language/tw_attention_asides_dironly.json", "w"), indent=1)
