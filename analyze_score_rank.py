"""Registered readouts for SCORE_RANK_PREREG.md: does PoPE's score rule rescue per-head rank 2 on the long-walk torus?

Torus, trained and tested at T=1024, 900 epochs (the RANK_MI recipe). Arms (2 x 2, matched initialisation,
model_pope_pair_mi): A2 = MapWM r2 (`Vanilla`), P2 = MapPoPE-Pair r2, A4 = MapWM r4 (`Vanilla_r4mi`), P4 =
MapPoPE-Pair_r4mi. 32 angles per head in every arm; within a rank only the score rule differs.

Inputs: SCORE_RANK_R2.json / _R4.json (eval_score_rank -> eval_noise_refine; keys '0.0|<arm>|<T>' -> [[seed, acc,
nll], ...]), SCORE_RANK_RESCORE_R2.json / _R4.json (same, attention x 1/(1-p); secondary), SCORE_RANK_STRATA_R2.json /
_R4.json (eval_rank_strata; secondary),
the checkpoints' per-epoch losses (stats_core.classify_run: SOLVED = final-5% loss < 0.05) and weights (basins).

Primary (P2 vs A2 at T=1024): SOLVED (two-sided Fisher, fires if p < .05) and accuracy (exact/MC permutation,
stats_core.perm2_p; fires if p < .05 AND |d| >= MIN_D). Branches: see decide(). `python3 -m mapformer.analyze_score_rank`
"""
import json
import os

import numpy as np
import torch
from scipy import stats

from mapformer import train_score_rank  # noqa: F401  (registers MapPoPE-Pair / MapPoPE-Pair_r4mi in VARIANT_MAP)
from mapformer.stats_core import classify_run, perm2_p, fisher_solved

REPO = "/home/prashr/mapformer"; R = f"{REPO}/runs/score_rank/p0"
A2, P2, A4, P4 = "Vanilla", "MapPoPE-Pair", "Vanilla_r4mi", "MapPoPE-Pair_r4mi"
ARMS = [A2, P2, A4, P4]
N_R2 = 12                                               # rank-2 arms: seeds 10..21 (SCORE_RANK_PREREG.md, power)
SEEDS = {A2: list(range(10, 10 + N_R2)), P2: list(range(10, 10 + N_R2)), A4: list(range(10, 18)), P4: list(range(10, 18))}
LABEL = {A2: "A2 MapWM r2", P2: "P2 PoPE-score r2", A4: "A4 MapWM r4", P4: "P4 PoPE-score r4"}
MIN_D = 0.02                                            # materiality floor for an accuracy firing
VOID_A4 = 5                                             # positive control: A4 SOLVED <= 5/8 -> VOID
FLOORS = {1024: "blank 0.507, best n-gram 0.576, retrace 0.843", 512: "blank 0.509, best n-gram 0.588, retrace 0.884",
          2048: "blank 0.508, best n-gram 0.566, retrace 0.790"}   # docs/audits/2026-10-05/score_rank_floors_out.txt


def mde2(x, y):
    """Two-sample MDE (alpha .05 two-sided, power .8, df = nx + ny - 2, pooled sd)."""
    nx, ny = len(x), len(y); df = nx + ny - 2
    sd = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / df)
    return (stats.t.ppf(0.975, df) + stats.t.ppf(0.8, df)) * sd * np.sqrt(1 / nx + 1 / ny)


def acc_state(x, y):
    """y - x: 'POS' / 'NEG' if perm p < .05 and |d| >= MIN_D, 'CEIL' if both >= 0.999 on every seed, else 'NONE'."""
    x, y = np.asarray(x), np.asarray(y); d = y.mean() - x.mean(); p = perm2_p(x, y)["p"]
    if x.min() >= 0.999 and y.min() >= 0.999:
        return "CEIL", d, p
    if p < 0.05 and abs(d) >= MIN_D:
        return ("POS" if d > 0 else "NEG"), d, p
    return "NONE", d, p


def fisher_state(sx, nx, sy, ny):
    """y vs x on SOLVED: 'POS' / 'NEG' if two-sided Fisher p < .05 (direction from the rates), else 'NONE'."""
    p = fisher_solved(sx, nx, sy, ny)
    if p < 0.05:
        return ("POS" if sy / ny > sx / nx else "NEG"), p
    return "NONE", p


def decide(sol, n, acc):
    """The registered branch. sol / n: SOLVED counts and seeds per arm; acc: T=1024 accuracies per arm (A2, P2 used).
    Returns (branch, qualifiers, detail)."""
    q = []
    if sol[A4] <= VOID_A4:
        return "VOID", q, f"positive control failed: MapWM r4 solved {sol[A4]}/{n[A4]} (<= {VOID_A4}/8)"
    prem, p_prem = fisher_state(sol[A2], n[A2], sol[A4], n[A4])
    fs, p_f = fisher_state(sol[A2], n[A2], sol[P2], n[P2])
    as_, d, p_a = acc_state(acc[A2], acc[P2])
    det = (f"P2 - A2: SOLVED {sol[P2]}/{n[P2]} vs {sol[A2]}/{n[A2]} (Fisher p {p_f:.4f}, {fs}); accuracy {d:+.4f} "
           f"(perm p {p_a:.4f}, {as_}); premise A4 vs A2 Fisher p {p_prem:.4f} ({prem})")
    if as_ == "CEIL":
        return "CEILING", q, det + " -- both rank-2 arms >= 0.999 on every seed: undetermined"
    if prem != "POS":
        return "NO DEFICIT TO RESCUE", q, det + " -- MapWM's rank-2 deficit did not replicate at these seeds"
    p4low, p_p4 = fisher_state(sol[P4], n[P4], sol[A4], n[A4])          # A4 vs P4: POS means P4 detectably below A4
    if p4low == "POS":
        q.append(f"PoPE's score fails at rank 4 too (P4 {sol[P4]}/8 vs A4 {sol[A4]}/8, Fisher p {p_p4:.4f}): "
                 f"the rank-2 contrast is uninformative about rescue")
    if fs == "POS" and as_ == "POS":
        r4, p_r4 = fisher_state(sol[P2], n[P2], sol[P4], n[P4])         # POS: P4 detectably above P2
        if r4 == "POS":
            return "PARTIAL RESCUE", q + [f"both readouts fire, but rank 4 still solves detectably more under PoPE's "
                                          f"score (P4 {sol[P4]}/8 vs P2 {sol[P2]}/{n[P2]}, Fisher p {p_r4:.4f})"], det
        return "RESCUE", q + [f"P2 not detectably below P4 on SOLVED (Fisher p {p_r4:.4f})"], det
    if fs == "POS" and as_ == "NEG":
        return "CONFLICT", q, det + " -- SOLVED rises, accuracy falls"
    if fs == "POS" or as_ == "POS":
        which = "SOLVED only (accuracy does not fire)" if fs == "POS" else "accuracy only (SOLVED does not fire)"
        return "PARTIAL RESCUE", q + [which], det
    if as_ == "NEG":
        return "POPE SCORE HURTS AT RANK 2", q, det
    return "NO RESCUE", q, det + f" -- accuracy unmeasured below MDE {max(mde2(acc[A2], acc[P2]), MIN_D):.4f}"


# ----------------------------------------------------------------------------------------------- basins (secondary)
@torch.no_grad()
def head_stats(ck, weighted=True):
    """Per head: (kappa, indep) as docs/theory/2026-10-04/scripts/basins.py, with the declared PoPE channel weight.
    MapWM: w_j = mean_action |q_pair j| * mean_obs |k_pair j| (basins.py). PoPE-Pair: angle j drives elements 2j, 2j+1;
    w_j = | sum_{e in pair j} mean_action softplus(q_e) * mean_obs softplus(k_e) * exp(i delta_e) |, the amplitude of
    channel j's cosine in the score at the mean magnitudes (delta clamped as in the forward)."""
    from mapformer.train_variant import VARIANT_MAP
    from mapformer.model_pope import DELTA_MIN, DELTA_MAX
    b = torch.load(ck, map_location="cpu", weights_only=False); c = b["config"]; arm = b["variant"]
    m = VARIANT_MAP[arm](vocab_size=c["vocab_size"], d_model=c["d_model"], n_heads=c["n_heads"],
                         n_layers=c["n_layers"], grid_size=c["grid_size"]).eval()
    m.load_state_dict(b["model_state_dict"])
    V = c["vocab_size"]; D = 2; K = c.get("n_obs_types", 16); pe = c.get("p_empty", 0.5); nA = 2 * D
    e = m.token_emb(torch.arange(V)); a = (m.action_to_lie(e[None])[0] * m.path_integrator.omega[None]).numpy()
    L = m.layers[0]; H = c["n_heads"]; dh = c["d_model"] // H; h = L.norm1(e)
    Q = L.q_proj(h).view(V, H, dh); Kk = L.k_proj(h).view(V, H, dh)
    if hasattr(L, "pope_delta"):
        mq = torch.nn.functional.softplus(Q[:nA]).mean(0); mk = torch.nn.functional.softplus(Kk[nA:]).mean(0)   # (H, dh)
        dl = L.pope_delta.clamp(DELTA_MIN, DELTA_MAX)
        z = (mq * mk).numpy() * np.exp(1j * dl.numpy())
        w = np.abs(z[:, 0::2] + z[:, 1::2])                                                              # (H, dh/2)
    else:
        qa = torch.sqrt(Q[..., 0::2] ** 2 + Q[..., 1::2] ** 2); ka = torch.sqrt(Kk[..., 0::2] ** 2 + Kk[..., 1::2] ** 2)
        w = (qa[:nA].mean(0) * ka[nA:].mean(0)).numpy()
    if not weighted:
        w = np.ones_like(w)
    A = a[:nA]; O = a[nA:]
    U = np.stack([(A[2 * d] - A[2 * d + 1]) / 2 for d in range(D)])
    mm = np.angle(np.exp(1j * (A.mean(0) + pe * O[K] + (1 - pe) * O[:K].mean(0))))
    out = []
    for hh in range(H):
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
    J = json.load(open(f"{REPO}/SCORE_RANK_R2.json")) | json.load(open(f"{REPO}/SCORE_RANK_R4.json"))
    acc, nll = {}, {}
    for a in ARMS:
        for T in (512, 1024, 2048):
            rows = sorted(J[f"0.0|{a}|{T}"]); assert [r[0] for r in rows] == SEEDS[a], (a, T, [r[0] for r in rows])
            acc[(a, T)] = np.array([r[1] for r in rows]); nll[(a, T)] = np.array([r[2] for r in rows])
    ck = {(a, s): f"{R}/{a}_s{s}/{a}.pt" for a in ARMS for s in SEEDS[a]}
    cl = {k: classify_run(torch.load(v, map_location="cpu", weights_only=False)["losses"]) for k, v in ck.items()}
    sol = {a: sum(cl[(a, s)]["registered"] == "SOLVED" for s in SEEDS[a]) for a in ARMS}
    n = {a: len(SEEDS[a]) for a in ARMS}

    print(f"== T=1024 held-out accuracy (registered; floors on this stream: {FLOORS[1024]}) | SOLVED | [T=512, T=2048] ==")
    for a in ARMS:
        x = acc[(a, 1024)]
        print(f"  {LABEL[a]:18s} n={n[a]:2d} {x.mean():.4f} +/- {x.std(ddof=1):.4f}  SOLVED {sol[a]}/{n[a]}  "
              f"[{acc[(a, 512)].mean():.4f}, {acc[(a, 2048)].mean():.4f}]")
        print("      " + " ".join(f"s{s}:{v:.3f}/{cl[(a, s)]['registered'][:4]}({cl[(a, s)]['tail']:.3f})"
                                for s, v in zip(SEEDS[a], x)))
    acc1024 = {a: acc[(a, 1024)] for a in ARMS}
    br, qual, det = decide(sol, n, acc1024)
    print(f"\n== primary (P2 - A2, T=1024) ==\n  {det}")
    print(f"\n  REGISTERED: {br}" + "".join(f"\n    - {x}" for x in qual))

    print("\n== secondaries (no verdict) ==")
    for name, lo, hi in (("rank effect, MapWM (A4 - A2; replicates RANK_MI on fresh seeds)", A2, A4),
                         ("rank effect, PoPE score (P4 - P2)", P2, P4), ("score rule at rank 4 (P4 - A4)", A4, P4)):
        fs, pf = fisher_state(sol[lo], n[lo], sol[hi], n[hi]); st, d, pa = acc_state(acc1024[lo], acc1024[hi])
        print(f"  {name:62s} SOLVED {sol[hi]}/{n[hi]} vs {sol[lo]}/{n[lo]} Fisher p {pf:.4f} | acc {d:+.4f} perm p {pa:.4f} ({st})")
    for T in (512, 2048):
        st, d, pa = acc_state(acc[(A2, T)], acc[(P2, T)])
        print(f"  P2 - A2 at T={T} (rule 10: {'shorter than' if T < 1024 else 'past'} training length): {d:+.4f} perm p {pa:.4f}"
              + (f"   floors {FLOORS[T]}" if T == 2048 else ""))
    st, d, pa = acc_state(nll[(A2, 1024)], nll[(P2, 1024)])
    print(f"  P2 - A2 revisit NLL at T=1024: {d:+.4f} perm p {pa:.4f} (lower is better)")
    if all(os.path.exists(f"{REPO}/SCORE_RANK_RESCORE_{r}.json") for r in ("R2", "R4")):
        JR = json.load(open(f"{REPO}/SCORE_RANK_RESCORE_R2.json")) | json.load(open(f"{REPO}/SCORE_RANK_RESCORE_R4.json"))
        rs = {a: np.array([r[1] for r in sorted(JR[f"0.0|{a}|1024"])]) for a in ARMS}
        st, d, pa = acc_state(rs[A2], rs[P2])
        print(f"  dropout-scale re-score (attention x 1/(1-p)), T=1024: " + "  ".join(f"{LABEL[a][:2]} {rs[a].mean():.4f}" for a in ARMS)
              + f"; P2 - A2 {d:+.4f} perm p {pa:.4f} ({st})")
    if all(os.path.exists(f"{REPO}/SCORE_RANK_STRATA_{r}.json") for r in ("R2", "R4")):
        JS = json.load(open(f"{REPO}/SCORE_RANK_STRATA_R2.json")) | json.load(open(f"{REPO}/SCORE_RANK_STRATA_R4.json"))
        for k in ("wrap", "plain_lag>=128", "plain_lag<128"):
            v = {a: np.array([JS[f"{a}|{s}|1024"][k]["acc"] for s in SEEDS[a]]) for a in ARMS}
            fl = np.mean([JS[f"{a}|{s}|1024"][k]["floor"] for a in ARMS for s in SEEDS[a]])
            st, d, pa = acc_state(v[A2], v[P2])
            print(f"  stratum {k:15s} (floor {fl:.3f}): " + "  ".join(f"{LABEL[a][:2]} {v[a].mean():.3f}" for a in ARMS)
                  + f"; P2 - A2 {d:+.4f} perm p {pa:.4f}")
    tails = np.array([cl[(a, s)]["tail"] for a in ARMS for s in SEEDS[a]]); accs = np.concatenate([acc1024[a] for a in ARMS])
    print(f"  r(final-5% loss, acc@1024) over {len(tails)} runs: {np.corrcoef(tails, accs)[0, 1]:+.3f}; within arm: "
          + "  ".join(f"{LABEL[a][:2]} {np.corrcoef([cl[(a, s)]['tail'] for s in SEEDS[a]], acc1024[a])[0, 1]:+.3f}" for a in ARMS))
    def t_solve(a, s):
        l = np.asarray(torch.load(ck[(a, s)], map_location="cpu", weights_only=False)["losses"], float)
        r = np.convolve(l, np.ones(10) / 10, mode="valid"); i = np.nonzero(r < 0.05)[0]
        return int(i[0]) + 10 if len(i) else None
    ts = {a: [t for t in (t_solve(a, s) for s in SEEDS[a]) if t is not None] for a in ARMS}
    print("  epoch at which the 10-epoch running loss first falls below 0.05 (runs that get there; speed, descriptive): "
          + "  ".join(f"{LABEL[a][:2]} " + (f"median {int(np.median(ts[a]))} [{min(ts[a])}-{max(ts[a])}] n={len(ts[a])}" if ts[a] else "none")
                      for a in ARMS))
    print("  run classes: " +"  ".join(f"{LABEL[a][:2]} " + ",".join(f"{c}:{sum(cl[(a, s)]['cls'] == c for s in SEEDS[a])}"
                                                                    for c in ("SOLVED", "STALLED", "DESCENDING", "RISING")) for a in ARMS))

    print("\n== basins (T1; declared secondary). Per run: basin of its best head, + SOLVED / - not ==")
    for wt in (True, False):
        cnt = {}
        for a in ARMS:
            row = []
            for s in SEEDS[a]:
                bs = basin(head_stats(ck[(a, s)], wt)); sv = cl[(a, s)]["registered"] == "SOLVED"
                cnt[(a, bs, sv)] = cnt.get((a, bs, sv), 0) + 1
                row.append(f"{bs[:3]}{'+' if sv else '-'}")
            if wt:
                print(f"  {LABEL[a]:18s} " + " ".join(row))
        print(f"  {'weighted' if wt else 'unweighted'}: " + "; ".join(
            f"{LABEL[a][:2]} " + " ".join(f"{b}/{'S' if v else 'U'} {cnt.get((a, b, v), 0)}" for b in ("CLEAN", "COLLAPSE", "CLOCK")
                                          for v in (True, False) if cnt.get((a, b, v), 0)) for a in ARMS))
        conc = {a: cnt.get((a, "CLEAN", True), 0) + sum(cnt.get((a, b, False), 0) for b in ("COLLAPSE", "CLOCK")) for a in ARMS}
        print(f"    'SOLVED iff CLEAN' holds on: " + "  ".join(f"{LABEL[a][:2]} {conc[a]}/{n[a]}" for a in ARMS))
        ck2 = {a: sum(cnt.get((a, "CLOCK", v), 0) for v in (True, False)) for a in (A2, P2)}
        print(f"    CLOCK runs P2 {ck2[P2]}/{n[P2]} vs A2 {ck2[A2]}/{n[A2]} (Fisher p {fisher_solved(ck2[A2], n[A2], ck2[P2], n[P2]):.4f})")

    rp = f"{REPO}/runs/score_rank_pilot/repro/Vanilla_s0/Vanilla.pt"
    if os.path.exists(rp):
        x = np.array(torch.load(rp, map_location="cpu", weights_only=False)["losses"])
        y = np.array(torch.load(f"{REPO}/runs/rank_mi/p0/Vanilla_s0/Vanilla.pt", map_location="cpu", weights_only=False)["losses"])
        print(f"\n== pilot reproduction (train_score_rank Vanilla s0 vs stored rank_mi): max |per-epoch loss diff| "
              f"{np.abs(x - y).max():.2e} over {len(x)} epochs")


if __name__ == "__main__":
    main()
