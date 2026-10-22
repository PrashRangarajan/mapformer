"""ANALYSIS (post hoc, CPU, ~1 min): what the leftover latent clock of solved rank > D heads is made of.

Written for docs/paper/review_draft5/ANALYSIS_SCRIPT_AUDIT.md M-RK1 / X2. rank_geom.py's clock is
c = a + o, with a = the mean move latent and o = p_empty * z(blank) + (1 - p_empty) * mean z(objects), the mean
observation latent, in the native latent of the head's read subspace (rank_geom.latent_tables), against the map frame
U (half-differences of opposite moves' latents). Here, per solved head of rank above D (SOLVED by
stats_core.classify_run's registered rule), the parts are reported separately:
  |a| / |U|, |o| / |U|, |c| / |U|     sizes relative to the mean map step |U_d|
  off(v) = |v - P_U v| / |v|          share of v outside the map plane
  cos(a, o)                           alignment of the move and observation parts (-1: they cancel)
  per-pair clocks c_j = z(-e_j) + z(+e_j) (SEARCH's definition, observation step left out) and c_j + 2 o (one
  out-and-back of axis j with its two observation steps): size relative to |U| and off-plane share.
Medians over the solved heads of each set. Sets: sweep_final.SETS with per-head rank > D (the rank_wrap N10 D3 r4 and N32
D2 r3 sets are weight-identical to rank_nd's and are skipped; escape E is reported for its solved runs only).
  cd /home/prashr && PYTHONPATH=/home/prashr python3 mapformer/docs/paper/expansion/scripts/rank/clock_decomp.py \
      > mapformer/docs/paper/expansion/scripts/rank/clock_decomp_out.txt
"""
import glob
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
torch.set_num_threads(2)
from rank_geom import latent_tables  # noqa: E402
from sweep_final import SETS, load  # noqa: E402
from mapformer.stats_core import classify_run  # noqa: E402

SKIP = ("ND D3 G10 r4 (rank_wrap)", "ND D2 G32 r3 (rank_wrap)")


def off(v, Ul):
    coef, *_ = np.linalg.lstsq(Ul.T, v, rcond=None)
    n = np.linalg.norm(v)
    return float(np.linalg.norm(v - Ul.T @ coef) / n) if n > 0 else float("nan")


def cos(u, v):
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))


print(__doc__.split("\n")[0])
cols = ["|a|/|U|", "|o|/|U|", "|c|/|U|", "off(a)", "off(o)", "off(c)", "cos(a,o)", "|c_j|/|U|", "off(c_j)",
        "off(c_j+2o)"]
print(f"| set (solved heads, rank > D) | n heads | " + " | ".join(cols) + " |")
print("|---|---|" + "---|" * len(cols))
for lab, pat, D, r, kind in SETS:
    if r <= D or lab in SKIP:
        continue
    vals = []
    for ck in sorted(glob.glob(pat)):
        m, b = load(ck, kind)
        if classify_run(b["losses"])["registered"] != "SOLVED":
            continue
        pairs = [(0, 1)] if kind == "ring" else [(2 * d, 2 * d + 1) for d in range(D)]
        nA = 2 * len(pairs); obs0 = nA; n_obs = 16
        for Z, F in latent_tables(m):
            if Z.shape[1] <= D:
                continue
            Ul = np.stack([(Z[x] - Z[y]) / 2 for x, y in pairs])
            nU = np.linalg.norm(Ul, axis=1).mean()
            a = Z[:nA].mean(0)
            o = 0.5 * Z[obs0 + n_obs] + 0.5 * Z[obs0:obs0 + n_obs].mean(0)
            c = a + o
            cj = [Z[x] + Z[y] for x, y in pairs]
            vals.append([np.linalg.norm(a) / nU, np.linalg.norm(o) / nU, np.linalg.norm(c) / nU, off(a, Ul), off(o, Ul),
                         off(c, Ul), cos(a, o), np.mean([np.linalg.norm(v) / nU for v in cj]),
                         np.mean([off(v, Ul) for v in cj]), np.mean([off(v + 2 * o, Ul) for v in cj])])
    if not vals:
        print(f"| {lab} | 0 | " + " | ".join("--" for _ in cols) + " |"); continue
    md = np.median(np.array(vals), 0)
    print(f"| {lab} | {len(vals)} | " + " | ".join(f"{x:.4f}" if i == 2 else f"{x:.3f}" for i, x in enumerate(md)) + " |")

# ---- escape arms: size of the latent clock at the start and at the end (rank_geom's lam = |c| / |U| and
# lam_perp = |c_perp| / |U|, native latent of each head's read subspace), per run; does it shrink (removal) or stay?
from rank_geom import head_geometry  # noqa: E402
from mapformer.train_variant import VARIANT_MAP  # noqa: E402
R = "/home/prashr/mapformer/runs"


def geo(ck):
    b = torch.load(ck, map_location="cpu", weights_only=False); c = b["config"]
    m = VARIANT_MAP[b["variant"]](vocab_size=c["vocab_size"], d_model=128, n_heads=2, n_layers=1, grid_size=c["grid_size"]).eval()
    m.load_state_dict(b["model_state_dict"])
    g = head_geometry(m)
    for h, (Z, F) in zip(g, latent_tables(m)):                     # absolute latent sizes (not relative to |U|)
        Ul = np.stack([(Z[2 * d] - Z[2 * d + 1]) / 2 for d in range(2)])
        cl = Z[:4].mean(0) + 0.5 * Z[4 + 16] + 0.5 * Z[4:20].mean(0)
        coef, *_ = np.linalg.lstsq(Ul.T, cl, rcond=None)
        h["c_abs"] = float(np.linalg.norm(cl)); h["cperp_abs"] = float(np.linalg.norm(cl - Ul.T @ coef))
        h["U_abs"] = float(np.linalg.norm(Ul, axis=1).mean())
    return g, b


print("\n== escape arms (RANK_TESTS experiment 2): latent clock lam = |c|/|U| and its off-plane part lam_perp, start -> end, "
      "mean over the two heads; shrink = start / end")
for arm, start_pat, end_pat in [("F fresh r4 step map", f"{R}/rank_tests/escape/fresh/p0/*_s{{s}}/*.pt", f"{R}/rank_tests/escape/F/p0/*_s{{s}}/*.pt"),
                                ("E lifted r2 -> r4", f"{R}/rank_tests/escape/lifted/p0/*_s{{s}}/*.pt", f"{R}/rank_tests/escape/E/p0/*_s{{s}}/*.pt")]:
    rows = []
    for s in range(8):
        g0, _ = geo(glob.glob(start_pat.format(s=s))[0]); g1, b1 = geo(glob.glob(end_pat.format(s=s))[0])
        cl = classify_run(b1["losses"])["registered"]
        l0 = np.mean([h["lam"] for h in g0]); l1 = np.mean([h["lam"] for h in g1])
        p0 = np.mean([h["lam_perp"] for h in g0]); p1 = np.mean([h["lam_perp"] for h in g1])
        o1 = np.mean([h["lam_perp"] / max(h["lam"], 1e-12) for h in g1])
        a0 = np.mean([h["c_abs"] for h in g0]); a1 = np.mean([h["c_abs"] for h in g1])
        q0 = np.mean([h["cperp_abs"] for h in g0]); q1 = np.mean([h["cperp_abs"] for h in g1])
        u0 = np.mean([h["U_abs"] for h in g0]); u1 = np.mean([h["U_abs"] for h in g1])
        rows.append((s, cl, l0, l1, p0, p1, o1, a0 / a1))
        print(f"  {arm} s{s} {cl[:4]}: lam {l0:.4f} -> {l1:.4f} (shrink {l0 / l1:.1f}x); lam_perp {p0:.4f} -> {p1:.4f}; "
              f"end off-plane share {o1:.2f} | absolute |c| {a0:.4f} -> {a1:.4f} ({a0 / a1:.1f}x), |c_perp| {q0:.4f} -> {q1:.4f}"
              f" ({q0 / max(q1, 1e-12):.1f}x), |U| {u0:.3f} -> {u1:.3f}")
    sh = [r[2] / r[3] for r in rows]; sa = [r[7] for r in rows]
    print(f"  {arm}: shrink of lam {min(sh):.1f}-{max(sh):.1f}x (median {np.median(sh):.1f}); of absolute |c| {min(sa):.1f}-"
          f"{max(sa):.1f}x (median {np.median(sa):.1f}); end lam {min(r[3] for r in rows):.4f}-"
          f"{max(r[3] for r in rows):.4f}; end off-plane share {min(r[6] for r in rows):.2f}-{max(r[6] for r in rows):.2f}")
