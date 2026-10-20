"""ANALYSIS (post hoc, CPU): where the clock sits at the START of the escape arms (RANK_TESTS experiment 2) and at the end.
E = stalled rank-2 weights lifted to rank 4 (function-preserving), F = same network with a fresh rank-4 step map,
C = rank 2 continued (start = the rank_mi source). Per head: kappa, off-plane share of the latent clock, readgain, indep.
Run: cd /home/prashr && PYTHONPATH=/home/prashr python3 mapformer/docs/paper/expansion/scripts/rank/escape_start.py"""
import glob, os, sys
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
torch.set_num_threads(2)
from rank_geom import head_geometry  # noqa
from mapformer.train_variant import VARIANT_MAP  # noqa
from mapformer.stats_core import classify_run  # noqa
R = "/home/prashr/mapformer/runs"
def geo(ck):
    b = torch.load(ck, map_location="cpu", weights_only=False); c = b["config"]
    m = VARIANT_MAP[b["variant"]](vocab_size=c["vocab_size"], d_model=128, n_heads=2, n_layers=1, grid_size=c["grid_size"]).eval()
    m.load_state_dict(b["model_state_dict"]); return head_geometry(m), b
L = []
for arm, start_pat, end_pat in [("E", f"{R}/rank_tests/escape/lifted/p0/*_s{{s}}/*.pt", f"{R}/rank_tests/escape/E/p0/*_s{{s}}/*.pt"),
                                ("F", f"{R}/rank_tests/escape/fresh/p0/*_s{{s}}/*.pt", f"{R}/rank_tests/escape/F/p0/*_s{{s}}/*.pt"),
                                ("C", f"{R}/rank_mi/p0/Vanilla_s{{s}}/Vanilla.pt", f"{R}/rank_tests/escape/C/p0/*_s{{s}}/*.pt")]:
    for s in range(8):
        st = glob.glob(start_pat.format(s=s)); en = glob.glob(end_pat.format(s=s))
        if not st or not en:
            L.append(f"{arm} s{s}: missing {st} {en}"); continue
        g0, _ = geo(st[0]); g1, b1 = geo(en[0]); cl = classify_run(b1["losses"])["registered"]
        f = lambda g: " ".join(f"[k {h['kappa']:.3f} off {h['lam_perp'] / max(h['lam'], 1e-12):.2f} rg {h['readgain']:.2f} ind {h['indep']:.2f}]" for h in g)
        L.append(f"{arm} s{s} {cl[:4]}  start {f(g0)}  ->  end {f(g1)}")
open(f"{HERE}/escape_start_out.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))

# ---- full-latent clock geometry at the start of E and F (the read subspace of E's heads starts 2-dim, since the new
# W_out columns are 0; here the whole 4-dim latent): off-plane share of c relative to range(U), and |c| / |U|.
L2 = ["", "full 4-dim latent at the START (before any training): off-plane share of c, |c|/|U| (native latent metric)"]
for arm, pat in [("E", f"{R}/rank_tests/escape/lifted/p0/*_s{{s}}/*.pt"), ("F", f"{R}/rank_tests/escape/fresh/p0/*_s{{s}}/*.pt")]:
    row = []
    for s in range(8):
        b = torch.load(glob.glob(pat.format(s=s))[0], map_location="cpu", weights_only=False)
        W = b["model_state_dict"]["action_to_lie.w_in.weight"].numpy(); E_ = b["model_state_dict"]["token_emb.weight"].numpy()
        Z = E_ @ W.T
        U = np.stack([(Z[2 * d] - Z[2 * d + 1]) / 2 for d in range(2)])
        c = Z[:4].mean(0) + 0.5 * Z[4 + 16] + 0.5 * Z[4:20].mean(0)
        coef, *_ = np.linalg.lstsq(U.T, c, rcond=None); cp = c - U.T @ coef
        row.append(f"s{s}: off {np.linalg.norm(cp) / np.linalg.norm(c):.2f} |c|/|U| {np.linalg.norm(c) / np.linalg.norm(U, axis=1).mean():.2f}")
    L2.append(f"{arm}: " + "; ".join(row))
open(f"{HERE}/escape_start_out.txt", "a").write("\n".join(L2) + "\n"); print("\n".join(L2))
