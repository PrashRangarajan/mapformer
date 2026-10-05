"""Registered readouts for GAIN_GRAIN_PREREG.md: gain granularity between MapEM and MapPoPE on the paper torus.

Paper torus (64x64, 16 objects, p_empty 0.5), trained and tested at T=128, 300 epochs (the paper2x2 / MAPPOPE_PAIR
recipe). Six arms, all rank 2 and 32 angles per head, seeds 26-45 (n=20 each), one batch:
  W  `Vanilla`           MapWM: content rotated by the path angle (content can shift the kernel)
  P  `MapPoPE-Pair`      per-frequency non-negative gain (64 element gains, 2 per angle), learned delta
  S  `GainScalar`        ONE non-negative gain per token per head x fixed learned spectrum A_c >= 0, delta 0
  M  `GainMod4`          one non-negative gain per token per head per scale band (4 x 8 channels), A_c >= 0, delta 0
  E  `VanillaEM`         MapEM: signed scalar content factor A_X x learned position kernel with offsets
  N  `VanillaEM_NonNeg`  MapEM with softplus(A_X): the sign constraint alone (same initial weights as E)

Inputs: GAIN_GRAIN_EVAL.json (eval_gain_grain -> eval_noise_refine; keys '0.0|<arm>|<T>' -> [[seed, acc, nll], ...]),
the checkpoints' per-epoch losses (stats_core.classify_run: SOLVED = final-5% loss < 0.05) and weights (basins).

Primaries (T=128, held-out map):
  (a) GRANULARITY, non-inferiority against P: S vs P and M vs P (granularity_state; verdict granularity_verdict).
  (b) SIGN: N vs E, two-sided (sign_state).
Every branch is exercised on synthetic data by docs/audits/2026-10-05/gain_grain_smoke.py.
`python3 -m mapformer.analyze_gain_grain`
"""
import json
import os

import numpy as np
import torch
from scipy import stats

from mapformer.stats_core import classify_run, perm2_p, perm2_ci, fisher_solved

REPO = "/home/prashr/mapformer"; R = f"{REPO}/runs/gain_grain/p0"
W, P, S, M, E, N = "Vanilla", "MapPoPE-Pair", "GainScalar", "GainMod4", "VanillaEM", "VanillaEM_NonNeg"
ARMS = [W, P, S, M, E, N]
SEEDS = list(range(26, 46))                 # n = 20 per arm (GAIN_GRAIN_PREREG.md, power)
LABEL = {W: "W MapWM r2", P: "P MapPoPE-Pair (per-frequency gain)", S: "S scalar gain", M: "M per-module gain (4)",
         E: "E MapEM (signed scalar)", N: "N MapEM, softplus(A_X)"}
MARGIN = 0.01        # non-inferiority margin on mean T=128 accuracy (granularity)
MIN_D = 0.01         # materiality floor for an accuracy firing (house rule, MAPPOPE_PAIR)
SOLVED_SLACK = 0     # Amendment 1 (audit D1): AS GOOD needs SOLVED(x) >= SOLVED(P); slack 1 gave a realised size ~0.10
CEIL = 0.999
N_MC = 200_000       # Monte Carlo relabellings when C(32, 16) > 250k (the power script lowers it)
FLOORS = "best n-gram 0.598, always-blank 0.507 (docs/WHAT_WHERE_CHECKS.md)"


def mde2(x, y):
    """Two-sample MDE (alpha .05 two-sided, power .8, df = nx + ny - 2, pooled sd)."""
    nx, ny = len(x), len(y); df = nx + ny - 2
    sd = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / df)
    return (stats.t.ppf(0.975, df) + stats.t.ppf(0.8, df)) * sd * np.sqrt(1 / nx + 1 / ny)


def acc_state(x, y):
    """y - x (two-sided permutation): 'POS' / 'NEG' if p < .05 and |d| >= MIN_D; 'CEIL' if both arms >= CEIL on every
    seed (could not have gone the other way, rule 5); else 'NONE'. Returns (state, d, p)."""
    x, y = np.asarray(x, float), np.asarray(y, float); d = y.mean() - x.mean()
    p = perm2_p(x, y, n_mc=N_MC)["p"]
    if x.min() >= CEIL and y.min() >= CEIL:
        return "CEIL", d, p
    if p < 0.05 and abs(d) >= MIN_D:
        return ("POS" if d > 0 else "NEG"), d, p
    return "NONE", d, p


def fisher_state(sx, nx, sy, ny):
    """y vs x on SOLVED, two-sided Fisher: 'POS' / 'NEG' if p < .05 (direction from the rates), else 'NONE'."""
    p = fisher_solved(sx, nx, sy, ny)
    if p < 0.05:
        return ("POS" if sy / ny > sx / nx else "NEG"), p
    return "NONE", p


def ni_p(ref, x, margin=MARGIN):
    """One-sided permutation p for H0: mean(x) - mean(ref) <= -margin (shift test). With equal n the permutation
    distribution of the mean difference is symmetric, so one-sided p = two-sided p / 2 on the favourable side."""
    ref, x = np.asarray(ref, float), np.asarray(x, float); d = x.mean() - ref.mean()
    p2 = perm2_p(ref, x, shift=-margin, n_mc=N_MC)["p"]
    return p2 / 2 if d > -margin else 1 - p2 / 2


def granularity_state(ref_acc, ref_sol, x_acc, x_sol, n_ref, n_x):
    """x (a coarser gain) against ref (MapPoPE-Pair). Order: CONFLICT, WORSE, BETTER, AS GOOD, UNDETERMINED.
    WORSE / BETTER: accuracy (perm p < .05 and |d| >= MIN_D) or SOLVED (Fisher p < .05) fires; which is reported.
    AS GOOD: non-inferior on accuracy (one-sided p < .05 at MARGIN) AND SOLVED(x) >= SOLVED(ref) - SOLVED_SLACK.
    Returns (label, detail)."""
    a, d, pa = acc_state(ref_acc, x_acc); f, pf = fisher_state(ref_sol, n_ref, x_sol, n_x); pn = ni_p(ref_acc, x_acc)
    det = (f"d {d:+.4f} (perm p {pa:.4f}, {a}); SOLVED {x_sol}/{n_x} vs {ref_sol}/{n_ref} (Fisher p {pf:.4f}, {f}); "
           f"non-inferiority at -{MARGIN}: one-sided p {pn:.4f}")
    if (a == "POS" and f == "NEG") or (a == "NEG" and f == "POS"):
        return "CONFLICT", det
    if a == "NEG" or f == "NEG":
        which = " and ".join(w for w, s_ in (("accuracy", a), ("SOLVED", f)) if s_ == "NEG")
        return "WORSE", det + f" -- fires on {which}"
    if a == "POS" or f == "POS":
        which = " and ".join(w for w, s_ in (("accuracy", a), ("SOLVED", f)) if s_ == "POS")
        return "BETTER", det + f" -- fires on {which}"
    if pn < 0.05 and x_sol >= ref_sol - SOLVED_SLACK:
        return "AS GOOD", det + (" -- both arms >= 0.999 on every seed (at ceiling)" if a == "CEIL" else "")
    why = []
    if pn >= 0.05:
        why.append(f"accuracy not shown within {MARGIN}")
    if x_sol < ref_sol - SOLVED_SLACK:
        why.append(f"SOLVED more than {SOLVED_SLACK} below the reference")
    return "UNDETERMINED", det + " -- " + "; ".join(why)


def granularity_verdict(s_lab, m_lab):
    """Headline from (scalar vs P, module vs P). Exhaustive over {AS GOOD, BETTER, WORSE, UNDETERMINED, CONFLICT}^2."""
    ok = ("AS GOOD", "BETTER")
    if "CONFLICT" in (s_lab, m_lab):
        return f"CONFLICT (scalar {s_lab}, per-module {m_lab}): accuracy and SOLVED disagree; reported as it falls"
    if s_lab in ok and m_lab in ok:
        return "SCALAR GAIN SUFFICES (per-module also as good): per-frequency freedom not needed on this task"
    if s_lab in ok and m_lab == "UNDETERMINED":
        return "SCALAR GAIN SUFFICES (per-module undetermined): per-frequency freedom not needed on this task"
    if s_lab in ok and m_lab == "WORSE":
        return "NON-MONOTONE: scalar as good, per-module WORSE (anomalous; the module arm's cost is not granularity)"
    if s_lab == "WORSE" and m_lab in ok:
        return "MODULE GAIN SUFFICES, SCALAR DOES NOT: content must choose among scale bands, not among single frequencies"
    if s_lab == "WORSE" and m_lab == "WORSE":
        return ("PER-FREQUENCY GAIN NEEDED: both coarser non-negative gains cost against MapPoPE-Pair "
                "(confounded: they also have ~32k fewer content-projection parameters, no delta, learned A_c, tied pairs)")
    if s_lab == "WORSE" and m_lab == "UNDETERMINED":
        return ("SCALAR COSTS; per-module undetermined (confounded: fewer content-projection parameters, no delta, "
                "learned A_c, tied pairs)")
    if s_lab == "UNDETERMINED" and m_lab in ok:
        return "MODULE GAIN SUFFICES; scalar undetermined"
    if s_lab == "UNDETERMINED" and m_lab == "WORSE":
        return ("MODULE COSTS; scalar undetermined (non-monotone if the scalar is later shown as good; confounded as "
                "above)")
    return "UNDETERMINED: neither coarser gain shown as good or worse"


def sign_state(e_acc, e_sol, n_acc, n_sol, ne, nn):
    """N (softplus(A_X)) vs E (MapEM, signed), two-sided. Order: CEILING, CONFLICT, NON-NEGATIVE BETTER,
    NON-NEGATIVE WORSE, NO DIFFERENCE DETECTED. Returns (label, detail)."""
    a, d, pa = acc_state(e_acc, n_acc); f, pf = fisher_state(e_sol, ne, n_sol, nn)
    det = f"d {d:+.4f} (perm p {pa:.4f}, {a}); SOLVED {n_sol}/{nn} vs {e_sol}/{ne} (Fisher p {pf:.4f}, {f})"
    if a == "CEIL" and f == "NONE":                                   # Amendment 1 (N1): SOLVED is checked first
        return "CEILING", det + " -- both arms >= 0.999 on every seed: undetermined"
    if (a == "POS" and f == "NEG") or (a == "NEG" and f == "POS"):
        return "CONFLICT", det
    if a == "POS" or f == "POS":
        which = " and ".join(w for w, s_ in (("accuracy", a), ("SOLVED", f)) if s_ == "POS")
        return ("NON-NEGATIVE / POSITIVE-MEAN BETTER (MapEM's signed, zero-mean gain costs; non-negativity and a positive "
                "mean are not separated)"), det + f" -- fires on {which}"
    if a == "NEG" or f == "NEG":
        which = " and ".join(w for w, s_ in (("accuracy", a), ("SOLVED", f)) if s_ == "NEG")
        return "NON-NEGATIVE WORSE (the sign freedom helps)", det + f" -- fires on {which}"
    return "NO DIFFERENCE DETECTED", det + f" -- accuracy unmeasured below MDE {max(mde2(e_acc, n_acc), MIN_D):.4f}"


def headroom_state(w_acc, w_sol, p_acc, p_sol, nw, np_):
    """P vs W (SCORE, MAPPOPE_PAIR's registered contrast, fresh seeds). 'OK' if accuracy or SOLVED fires positive."""
    a, d, pa = acc_state(w_acc, p_acc); f, pf = fisher_state(w_sol, nw, p_sol, np_)
    return ("OK" if "POS" in (a, f) else "NO HEADROOM"), f"P - W: d {d:+.4f} (perm p {pa:.4f}, {a}); SOLVED {p_sol}/{np_} vs {w_sol}/{nw} (Fisher p {pf:.4f}, {f})"


# ----------------------------------------------------------------------------------------------- basins (secondary)
def channel_weights(m, e, nA):
    """Declared per-channel weight w (H, nb) of the score's cosine for each arm (GAIN_GRAIN_PREREG.md, secondaries)."""
    import torch.nn.functional as F
    from mapformer.model_pope import DELTA_MIN, DELTA_MAX
    L = m.layers[0]; H = m.n_heads; dh = m.d_head; V = e.shape[0]; h = L.norm1(e)
    if hasattr(L, "q_gain"):                                             # S / M: mean gains x A_c
        gq, gk = L.gains(h[None]); return (gq[0, :, :nA].mean(1) * gk[0, :, nA:].mean(1) * L.amplitude()).numpy()
    if hasattr(L, "q_content"):                                          # E / N: |q0_c||k0_c| (content is one scalar)
        q0 = m.q0_pos.view(H, -1, 2); k0 = m.k0_pos.view(H, -1, 2); return (q0.norm(dim=-1) * k0.norm(dim=-1)).numpy()
    Q = L.q_proj(h).view(V, H, dh); K = L.k_proj(h).view(V, H, dh)
    if hasattr(L, "pope_delta"):                                         # P: as analyze_score_rank
        mq = F.softplus(Q[:nA]).mean(0); mk = F.softplus(K[nA:]).mean(0)
        z = (mq * mk).numpy() * np.exp(1j * L.pope_delta.clamp(DELTA_MIN, DELTA_MAX).numpy())
        return np.abs(z[:, 0::2] + z[:, 1::2])
    qa = torch.sqrt(Q[..., 0::2] ** 2 + Q[..., 1::2] ** 2); ka = torch.sqrt(K[..., 0::2] ** 2 + K[..., 1::2] ** 2)
    return (qa[:nA].mean(0) * ka[nA:].mean(0)).numpy()                  # W: basins.py


@torch.no_grad()
def head_stats(ck, weighted=True):
    """Per head (kappa, indep) as analyze_score_rank.head_stats, with the channel weights above."""
    from mapformer.train_variant import VARIANT_MAP
    b = torch.load(ck, map_location="cpu", weights_only=False); c = b["config"]; arm = b["variant"]
    m = VARIANT_MAP[arm](vocab_size=c["vocab_size"], d_model=c["d_model"], n_heads=c["n_heads"],
                         n_layers=c["n_layers"], grid_size=c["grid_size"]).eval()
    m.load_state_dict(b["model_state_dict"])
    V = c["vocab_size"]; D = 2; K = c.get("n_obs_types", 16); pe = c.get("p_empty", 0.5); nA = 2 * D
    e = m.token_emb(torch.arange(V)); a = (m.action_to_lie(e[None])[0] * m.path_integrator.omega[None]).numpy()
    w = channel_weights(m, e, nA)
    if not weighted:
        w = np.ones_like(w)
    A = a[:nA]; O = a[nA:]
    U = np.stack([(A[2 * d] - A[2 * d + 1]) / 2 for d in range(D)])
    mm = np.angle(np.exp(1j * (A.mean(0) + pe * O[K] + (1 - pe) * O[:K].mean(0))))
    out = []
    for hh in range(m.n_heads):
        sw = np.sqrt(w[hh] / w[hh].sum()); Uw = U[:, hh] * sw
        sv = np.linalg.svd(Uw, compute_uv=False)
        out.append((float(np.linalg.norm(mm[hh] * sw) / np.linalg.norm(Uw, axis=1).mean()), float(sv[-1] / sv[0])))
    return out


def basin(o):
    ok = [ind for k, ind in o if k <= 0.01]
    if not ok:
        return "CLOCK"
    return "CLEAN" if max(ok) >= 0.2 else "COLLAPSE"


def main():
    from mapformer import train_gain_grain  # noqa: F401  (registers the arms in VARIANT_MAP)
    J = json.load(open(f"{REPO}/GAIN_GRAIN_EVAL.json"))
    acc, nll = {}, {}
    for a in ARMS:
        for T in (128, 512, 1024):
            rows = sorted(J[f"0.0|{a}|{T}"]); assert [r[0] for r in rows] == SEEDS, (a, T, [r[0] for r in rows])
            acc[(a, T)] = np.array([r[1] for r in rows]); nll[(a, T)] = np.array([r[2] for r in rows])
    ck = {(a, s): f"{R}/{a}_s{s}/{a}.pt" for a in ARMS for s in SEEDS}
    los = {k: torch.load(v, map_location="cpu", weights_only=False)["losses"] for k, v in ck.items()}
    assert all(len(l) == 300 for l in los.values()), "a run did not train 300 epochs"
    cl = {k: classify_run(v) for k, v in los.items()}
    sol = {a: sum(cl[(a, s)]["registered"] == "SOLVED" for s in SEEDS) for a in ARMS}
    n = len(SEEDS); A128 = {a: acc[(a, 128)] for a in ARMS}

    print(f"== T=128 held-out accuracy (registered; floors {FLOORS}) | SOLVED | NLL | [T=512, T=1024: no verdict] ==")
    for a in ARMS:
        x = A128[a]
        print(f"  {LABEL[a]:36s} {x.mean():.4f} +/- {x.std(ddof=1):.4f}  SOLVED {sol[a]:2d}/{n}  nll {nll[(a, 128)].mean():.4f}"
              f"  [{acc[(a, 512)].mean():.4f}, {acc[(a, 1024)].mean():.4f}]")
        print("      " + " ".join(f"s{s}:{v:.3f}/{cl[(a, s)]['registered'][:4]}" for s, v in zip(SEEDS, x)))

    hs, hdet = headroom_state(A128[W], sol[W], A128[P], sol[P], n, n)
    sl, sdet = granularity_state(A128[P], sol[P], A128[S], sol[S], n, n)
    ml, mdet = granularity_state(A128[P], sol[P], A128[M], sol[M], n, n)
    gv = granularity_verdict(sl, ml)
    print(f"\n== primary (a) GRANULARITY: coarser non-negative gains against MapPoPE-Pair, T=128, n={n} per arm ==")
    print(f"  headroom control ({hs}): {hdet}")
    print(f"  S scalar  vs P: {sl:12s} {sdet}")
    print(f"  M module  vs P: {ml:12s} {mdet}")
    for nm, x in (("S - P", S), ("M - P", M)):
        ci = perm2_ci(A128[P], A128[x], level=0.90, step=0.0005, n_mc=20000)
        print(f"  {nm} 90% permutation CI (two-sided; its lower end is the one-sided 95% bound): [{ci['lo']:+.4f}, {ci['hi']:+.4f}]")
    q = [] if hs == "OK" else ["NO HEADROOM: MapWM r2 is not detectably below MapPoPE-Pair on these seeds, so an AS GOOD "
                               "verdict does not show that a coarser gain avoids a cost this task can reveal"]
    if gv.startswith(("SCALAR GAIN SUFFICES", "MODULE GAIN SUFFICES")) and "at ceiling" in (sdet + mdet):
        gv += " (at ceiling)"                                           # Amendment 1 (N2)
    print(f"\n  REGISTERED (a): {gv}" + "".join(f"\n    - {x}" for x in q))

    gl, gdet = sign_state(A128[E], sol[E], A128[N], sol[N], n, n)
    print(f"\n== primary (b) SIGN: MapEM with softplus(A_X) vs MapEM, T=128, n={n} per arm (matched initial weights) ==")
    print(f"  N vs E: {gdet}")
    ci = perm2_ci(A128[E], A128[N], level=0.95, step=0.0005, n_mc=20000)
    print(f"  N - E 95% permutation CI: [{ci['lo']:+.4f}, {ci['hi']:+.4f}]")
    print(f"\n  REGISTERED (b): {gl}")

    print("\n== secondaries (no verdict) ==")
    for nm, lo, hi in (("SCORE replication P - W (MAPPOPE_PAIR +0.0243, 16/16 vs 10/16)", W, P),
                       ("bundle S - E (non-neg factorised scalar, delta 0 vs MapEM)", E, S),
                       ("content form S - N (factorised gain, delta 0 vs softplus(q.k), learned offsets)", N, S),
                       ("module vs scalar M - S", S, M), ("MapEM vs MapWM E - W", W, E), ("N - P", P, N)):
        st, d, pa = acc_state(A128[lo], A128[hi]); fs, pf = fisher_state(sol[lo], n, sol[hi], n)
        print(f"  {nm:80s} {d:+.4f} perm p {pa:.4f} ({st}); SOLVED {sol[hi]} vs {sol[lo]} Fisher p {pf:.4f}")
    for T in (512, 1024):
        for nm, lo, hi in (("S - P", P, S), ("M - P", P, M), ("N - E", E, N), ("P - W", W, P)):
            x, y = acc[(lo, T)], acc[(hi, T)]
            print(f"  T={T} (OOD, rule 10) {nm}: {y.mean() - x.mean():+.4f} (perm p {perm2_p(x, y, n_mc=N_MC)['p']:.4f})")
    for nm, lo, hi in (("S - P", P, S), ("M - P", P, M), ("N - E", E, N)):
        x, y = nll[(lo, 128)], nll[(hi, 128)]
        print(f"  T=128 revisit NLL {nm}: {y.mean() - x.mean():+.4f} (perm p {perm2_p(x, y, n_mc=N_MC)['p']:.4f}; lower is better)")
    tails = np.array([cl[(a, s)]["tail"] for a in ARMS for s in SEEDS]); accs = np.concatenate([A128[a] for a in ARMS])
    print(f"  r(final-5% loss, acc@128) over {len(tails)} runs: {np.corrcoef(tails, accs)[0, 1]:+.3f}; within arm: "
          + "  ".join(f"{a[:6]} " + (f"{np.corrcoef([cl[(a, s)]['tail'] for s in SEEDS], A128[a])[0, 1]:+.3f}"
                                     if np.std(A128[a]) > 0 else "n/a") for a in ARMS))
    print("  run classes: " + "  ".join(f"{a}: " + ",".join(f"{c}:{sum(cl[(a, s)]['cls'] == c for s in SEEDS)}"
                                                           for c in ("SOLVED", "STALLED", "DESCENDING", "RISING")) for a in ARMS))

    def t_solve(l):
        r = np.convolve(np.asarray(l, float), np.ones(10) / 10, mode="valid"); i = np.nonzero(r < 0.05)[0]
        return int(i[0]) + 10 if len(i) else None
    ts = {a: [t for t in (t_solve(los[(a, s)]) for s in SEEDS) if t is not None] for a in ARMS}
    print("  epoch at which the 10-epoch running loss first falls below 0.05 (descriptive): " + "  ".join(
        f"{a} " + (f"median {int(np.median(ts[a]))} n={len(ts[a])}" if ts[a] else "none") for a in ARMS))

    print("\n== basins (declared secondary; thresholds of analyze_score_rank, set at T=1024, unchanged). + SOLVED / - not ==")
    for wt in (True, False):
        conc = {}
        for a in ARMS:
            row, c_ = [], 0
            for s in SEEDS:
                bs = basin(head_stats(ck[(a, s)], wt)); sv = cl[(a, s)]["registered"] == "SOLVED"
                c_ += (bs == "CLEAN") == sv; row.append(f"{bs[:3]}{'+' if sv else '-'}")
            conc[a] = c_
            if wt:
                print(f"  {a:17s} " + " ".join(row))
        print(f"  {'weighted' if wt else 'unweighted'}: 'SOLVED iff CLEAN' holds on " + "  ".join(f"{a} {conc[a]}/{n}" for a in ARMS))

    rp = f"{REPO}/runs/gain_grain_pilot/repro/MapPoPE-Pair_s10/MapPoPE-Pair.pt"
    if os.path.exists(rp):
        x = np.array(torch.load(rp, map_location="cpu", weights_only=False)["losses"])
        y = np.array(torch.load(f"{REPO}/runs/mappope_pair/p0/MapPoPE-Pair_s10/MapPoPE-Pair.pt", map_location="cpu",
                                weights_only=False)["losses"])
        print(f"\n== pilot reproduction (train_gain_grain MapPoPE-Pair s10 vs stored mappope_pair): max |per-epoch loss diff| "
              f"{np.abs(x - y).max():.2e} over {len(x)} epochs")


if __name__ == "__main__":
    main()
