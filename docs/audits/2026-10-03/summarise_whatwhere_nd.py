"""Tables from probe_whatwhere_nd.json -> probe_whatwhere_nd_out.txt. sd is ddof=1 over seeds."""
import json, os, sys
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, "/home/prashr")
from mapformer.stats_core import perm2_p

OUT = os.path.dirname(os.path.abspath(__file__))
J = json.load(open(f"{OUT}/probe_whatwhere_nd.json"))
CELLS = ["2H", "2L", "A2", "3H", "3L"]
LABEL = {"2H": "2H N10 D2 r3ph (memorised)", "2L": "2L N32 D2 r3ph (generalised)",
         "A2": "A2 N32 D2 r2ph (rank_nd)", "3H": "3H N10 D3 r4ph", "3L": "3L N18 D3 r4ph"}


def wh(r, grid="w10_fixed"):
    hs = r[grid]; return max(range(len(hs)), key=lambda h: hs[h]["pos_sd"])


def rh(r):
    hs = r["fidelity"]["heads"]; return max(range(len(hs)), key=lambda h: hs[h]["attn"]["samecell_frac_of_obs"])


def metrics(r):
    w, q = wh(r), rh(r)
    g = r["w10_fixed"][w]; f = r["fidelity"]["heads"]
    out = {
        "pos": g["pos_share"], "content": g["cont_share"], "inter": g["inter_share"],
        "inter/pos": g["inter_of_pos"], "shape1": g["shape1"], "peak0": g["peak0"],
        "inter/pos both": np.mean([h["inter_of_pos"] for h in r["w10_fixed"]]),
        "inter/pos orig": r["w10_orig"][wh(r, "w10_orig")]["inter_of_pos"],
        "peak0 orig": r["w10_orig"][wh(r, "w10_orig")]["peak0"],
        "inter/pos full": r["full_fixed"][wh(r, "full_fixed")]["inter_of_pos"],
        "leak": float(np.mean(r["leak"])), "opp": float(np.mean(r["opp"])),
        "R2 grid (wh)": f[w]["R2_grid_all"], "R2 grid exact (wh)": f[w]["R2_grid_exact"],
        "R2 grid (rh)": f[q]["R2_grid_all"],
        "samecell attn (rh)": f[q]["attn"]["samecell_frac_of_obs"],
        "mean lag (rh)": f[q]["attn"]["mean_lag"],
        "real resid (rh)": f[q]["real_w10"]["residual"],
        "real inter (rh)": f[q]["real_w10"]["interaction"],
        "real inter/pos (rh)": f[q]["real_w10"]["inter_of_pos"],
        "real R2 pos (rh)": f[q]["real_w10"]["R2_position"],
        "real resid both": np.mean([h["real_w10"]["residual"] for h in f]),
        "real inter/pos both": np.mean([h["real_w10"]["inter_of_pos"] for h in f]),
        "samecell attn both": np.mean([h["attn"]["samecell_frac_of_obs"] for h in f]),
    }
    for k in ("held", "own", "final_loss", "map_MI_bits"):
        if k in r: out[k] = r[k]
    return out


def ms(v):
    v = np.asarray(v, float)
    return f"{v.mean():.3f} +/- {v.std(ddof=1):.3f}" if len(v) > 1 else f"{v.mean():.3f}"


L = []
P = lambda s="": L.append(s)
M = {c: [metrics(r) for r in J[c]["runs"]] for c in CELLS}
U = {c: [metrics(r) for r in J[c]["untrained"]] for c in CELLS}
keys = list(M["2L"][0].keys())

P("Verification (rule 9). max |rebuilt - model logits| per cell, and the same with the rotation removed")
for c in CELLS:
    runs = J[c]["runs"] + J[c]["untrained"]
    P(f"  {LABEL[c]:32s} rebuilt {max(r['verify_maxabs'] for r in runs):.1e}   no-rotation "
      f"{min(r['verify_norot'] for r in runs):.1f}..{max(r['verify_norot'] for r in runs):.1f}   "
      f"grid-formula impl check max abs {max(r['fidelity']['impl_check_maxabs'] for r in runs):.1e} "
      f"(rel to score sd {max(r['fidelity']['impl_check_rel'] for r in runs):.1e})   "
      f"pairs/model {J[c]['runs'][0]['fidelity']['n_pairs']}, wrap {np.mean([r['fidelity']['frac_wrap'] for r in J[c]['runs']]):.2f}")
P()
P("Per cell, mean +/- sd over 8 seeds (untrained: 3 inits). wh = where-head (larger position sd on the w10 grid,")
P("as in WHAT_WHERE_ANALYSIS sec. 6); rh = retrieval head (larger share of observation attention on same-cell keys).")
P()
hdr = f"{'readout':22s}" + "".join(f"{c:>22s}" for c in CELLS)
P(hdr)
for k in keys:
    P(f"{k:22s}" + "".join(f"{ms([m[k] for m in M[c]]):>22s}" if k in M[c][0] else f"{'--':>22s}" for c in CELLS))
P()
P("Untrained (3 inits, same grid size)")
P(hdr)
for k in keys:
    if k in U["2L"][0]:
        P(f"{k:22s}" + "".join(f"{ms([m[k] for m in U[c]]):>22s}" for c in CELLS))
P()
P("Contrasts, unpaired exact permutation p (stats_core.perm2_p), n=8 vs 8")
for a, b in (("2H", "2L"), ("3H", "3L"), ("A2", "2L")):
    P(f"  {b} - {a}")
    for k in ("inter/pos", "inter/pos both", "shape1", "peak0", "leak", "opp", "real resid (rh)",
              "real inter/pos (rh)", "samecell attn (rh)", "R2 grid (rh)", "held", "own"):
        x = [m[k] for m in M[a]]; y = [m[k] for m in M[b]]
        P(f"    {k:22s} {np.mean(y) - np.mean(x):+.3f}   p {perm2_p(x, y)['p']:.4f}")
P()
P("Per-seed Spearman rho (p) of separation readouts with accuracy")
for cells in (["2H"], ["2L"], ["A2"], ["3H"], ["3L"], ["2H", "2L", "A2"], ["3H", "3L"]):
    rows = [m for c in cells for m in M[c]]
    P(f"  {'+'.join(cells)} (n={len(rows)})")
    for k in ("inter/pos", "peak0", "leak", "real resid (rh)", "samecell attn (rh)"):
        line = f"    {k:22s}"
        for acc in ("held", "own"):
            x = [m[k] for m in rows]; y = [m[acc] for m in rows]
            rho, p = spearmanr(x, y)
            line += f"  vs {acc}: {rho:+.2f} ({p:.3f})" if np.isfinite(rho) else f"  vs {acc}: --"
        P(line)
P()
P("Per seed, 2H and 2L (where-head grid inter/pos, peak0, leak; rh real residual, same-cell attention, mean lag; held, own)")
for c in ("2H", "2L", "3H"):
    for s, m in enumerate(M[c]):
        P(f"  {c} s{s}  inter/pos {m['inter/pos']:.3f}  peak0 {m['peak0']:.3f}  leak {m['leak']:.3f}  opp {m['opp']:.3f}  "
          f"resid {m['real resid (rh)']:.3f}  same {m['samecell attn (rh)']:.3f}  lag {m['mean lag (rh)']:6.1f}  "
          f"held {m['held']:.3f}  own {m['own']:.3f}  loss {m['final_loss']:.4f}")
P()
P(f"Map entanglement I(o; o' | d) in bits, held-out map (seed 10000): {J['_heldout_map_MI_bits']}")
open(f"{OUT}/probe_whatwhere_nd_out.txt", "w").write("\n".join(L) + "\n")
print("\n".join(L))
