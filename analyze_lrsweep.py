"""Registered analysis for LRSWEEP_PREREG.md: is the sign learnability cost present when each arm is compared at its OWN best
learning rate? CPU.
  python3 -m mapformer.analyze_lrsweep [--root runs/lrsweep] [--seeds 1500-1507] [--rates 1e-3,3e-3,1e-2,3e-2]
                                       [--json-out PATH | --no-json]
  python3 -m mapformer.analyze_lrsweep --selftest     # every label and reading from synthetic profiles (no run dirs)
  python3 -m mapformer.analyze_lrsweep --power        # selection-aware power table (synthetic, CPU)
Layout: <root>/<tag>/<arm>_s<S>/{<arm>.pt, eval.json}, tag = rate_tag(lr) (e.g. lr1e-02), arm SO / AO (train_gauge,
deltas +1/-1, the GAUGE m-cell recipe at lr). SOLVED = stats_core.classify_run (final 5% of training loss < 0.05).

Primary B1 (best vs best). Each arm's best rate = the rate with the most SOLVED runs; ties broken by the lower mean selection
score (final-5% training loss, capped at the run's first-epoch loss; a non-finite tail scores the first-epoch loss), then by
the lower rate. Selection uses training loss only, never held-out accuracy. AO@best vs SO@best with the ROBUST cell rule
(Amendment 1), but both p-values come from a SELECTION-AWARE permutation null: all C(2n, n) splits of the 2n per-seed rate
PROFILES (a run at every rate) into two arms, the best rate re-selected inside each split by the same rule, the statistic
recomputed at the re-selected rates. The decision p of each route is max(selection-aware p, naive p at the selected rates
(Fisher on SOLVED, unpaired permutation on accuracy)): valid because the selection-aware p is, and never more liberal than
treating the selected rates as fixed in advance (selection inflates both arms under the null, so the selection-aware p can be
the smaller one).
"""
import argparse, json, math

import numpy as np
import torch

from mapformer.analyze_robust import opp_share, cell_label as robust_cell_label, SLB, NSC, UNM, NLB
from mapformer.environment_steps import GridWorldSteps
from mapformer.model_wrap_exist import (MapFormerWM_SignedAO_r4, MapFormerWM_AbsAO_r4, usage_weights,
                                        wrap_share_weighted)
from mapformer.stats_core import perm2_p, fisher_solved, classify_run, _relabellings

REPO = "/home/prashr/mapformer"
ARMS = ("SO", "AO")
DEFAULT_RATES = (1e-3, 3e-3, 1e-2, 3e-2)
UNR = "UNREADABLE"
BIG = 1e6                # selection key = SOLVED * BIG - mean score; scores are capped below 1e5


def rate_tag(lr):
    return f"lr{lr:.0e}"                 # 1e-3 -> lr1e-03, 3e-2 -> lr3e-02 (the driver uses the same python expression)


# ------------------------------------------------------------------------------ registered rule
def label_from(k_ao, k_so, acc_ao, acc_so, p_k, p_acc, n):
    """ROBUST cell rule (analyze_robust.cell_label, Amendment 1), with the two p-values supplied. d = AO - SO."""
    d = float(acc_ao - acc_so)
    if k_so < 4:
        return f"{UNR} (SO SOLVED {k_so} < 4)"
    if (p_k < 0.05 and k_so > k_ao) or (p_acc < 0.05 and d <= -0.05):
        return SLB
    if (p_k < 0.05 and k_ao > k_so) or (p_acc < 0.05 and d >= 0.05):
        return NLB
    if k_so >= 6 and k_ao >= k_so - 1 and acc_ao >= acc_so - 0.01:
        return NSC
    return UNM


def reading(lab, r_ao, r_so, rates):
    """Overall reading, fixed in LRSWEEP_PREREG.md. lab: B1 label; r_ao, r_so: selected rates; rates: the grid."""
    edge = r_ao == max(rates)
    q = " [AO's best rate is the grid's top rate: the AO optimum is not bracketed; 'best' = best TESTED]" if edge else ""
    if lab.startswith(UNR):
        return "UNREADABLE (the signed arm solves < 4/8 at every rate)" + q
    if lab == SLB:
        return "COST AT BEST RATES" + q
    if lab in (NSC, NLB):
        rev = " (reversed: NON-NEGATIVE LEARNS BETTER)" if lab == NLB else ""
        if r_ao > r_so:
            return f"COST CLOSES AT HIGHER LR (AO best {r_ao:g} > SO best {r_so:g}){rev}" + q
        return f"NO COST AT BEST RATES (AO best {r_ao:g}, SO best {r_so:g}){rev}" + q
    return "UNMEASURED" + q


# ------------------------------------------------------------------------------ selection and its permutation null
def select(solved, score, idx):
    """solved, score: (P, R) per profile and rate (rates ascending); idx: (M, m) profile indices of a group.
    Returns the selected rate index per row: most SOLVED, then lowest mean score, then lowest rate (argmax takes the first)."""
    ks = solved[idx].sum(1).astype(float)                  # (M, R)
    ms = score[idx].mean(1)                                # (M, R)
    return np.argmax(ks * BIG - ms, axis=1)


def _at(x, idx, r):
    """x (P, R); idx (M, m); r (M,) -> (M, m) values of each group's profiles at its selected rate."""
    return x[idx, r[:, None]]


def b1_test(solved, score, acc, n, max_exact=250_000, n_mc=200_000, seed=0):
    """Profiles 0..n-1 are SO, n..2n-1 AO. Returns the observed statistics, the selected rates and the selection-aware
    two-sided p-values for T_k = k_AO - k_SO and T_acc = mean acc_AO - mean acc_SO."""
    P = 2 * n
    idx_ao, exact = _relabellings(P, n, max_exact, n_mc, seed)          # rows: the profiles labelled AO
    mask = np.ones((len(idx_ao), P), bool); np.put_along_axis(mask, idx_ao, False, axis=1)
    idx_so = np.nonzero(mask)[1].reshape(len(idx_ao), n)                # complement, ascending
    r_ao, r_so = select(solved, score, idx_ao), select(solved, score, idx_so)
    Tk = _at(solved, idx_ao, r_ao).sum(1) - _at(solved, idx_so, r_so).sum(1)
    Ta = _at(acc, idx_ao, r_ao).mean(1) - _at(acc, idx_so, r_so).mean(1)
    o_ao, o_so = np.arange(n, P)[None], np.arange(n)[None]
    ra, rs = select(solved, score, o_ao)[0], select(solved, score, o_so)[0]
    tk = int(solved[n:, ra].sum() - solved[:n, rs].sum()); ta = float(acc[n:, ra].mean() - acc[:n, rs].mean())
    hk = int((np.abs(Tk) >= abs(tk)).sum()); ha = int((np.abs(Ta) >= abs(ta) - 1e-12).sum())
    M = len(idx_ao)
    pk = hk / M if exact else (hk + 1) / (M + 1); pa = ha / M if exact else (ha + 1) / (M + 1)
    return {"r_ao": int(ra), "r_so": int(rs), "k_ao": int(solved[n:, ra].sum()), "k_so": int(solved[:n, rs].sum()),
            "acc_ao": float(acc[n:, ra].mean()), "acc_so": float(acc[:n, rs].mean()), "T_k": tk, "T_acc": ta,
            "p_k_sel": float(pk), "p_acc_sel": float(pa), "exact": bool(exact), "n_relabel": int(M)}


def b1_decide(t, A, S, n):
    """Decision p-values for a b1_test result: max(selection-aware, naive at the selected rates), per route."""
    nf = fisher_solved(t["k_ao"], n, t["k_so"], n); npp = perm2_p(A[:n, t["r_so"]], A[n:, t["r_ao"]])["p"]
    pk, pa = max(t["p_k_sel"], nf), max(t["p_acc_sel"], npp)
    lab = label_from(t["k_ao"], t["k_so"], t["acc_ao"], t["acc_so"], pk, pa, n)
    return lab, {"p_k": pk, "p_acc": pa, "naive_fisher_p": nf, "naive_perm_p": npp}


def holm(ps):
    """Holm-adjusted p-values (step-down), same order as ps."""
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for i, j in enumerate(o):
        run = max(run, (m - i) * ps[j]); adj[j] = min(1.0, run)
    return adj


# ------------------------------------------------------------------------------ loading
def sel_score(losses):
    """Selection score of a run: final-5% mean training loss, capped at the first-epoch loss (a run that ends worse than it
    started counts as having learned nothing); a non-finite tail scores the first-epoch loss; a non-finite first epoch 1e4."""
    l = np.asarray(losses, float); E = len(l); k5 = max(1, round(0.05 * E))
    l0 = l[0] if np.isfinite(l[0]) else 1e4
    tail = l[-k5:].mean()
    return float(min(tail, l0)) if np.isfinite(tail) else float(l0)


def run_record(d, arm, s, lr, epochs, n_batches, tok=None):
    ev = json.load(open(f"{d}/eval.json")); b = torch.load(f"{d}/{arm}.pt", map_location="cpu", weights_only=False)
    A = b["config"]["args"]
    assert (A["arm"], A["seed"], tuple(b["config"]["deltas"])) == (arm, s, (1, -1)), (d, A)
    assert abs(A["lr"] - lr) <= 1e-9 * lr and abs(A.get("weight_decay", 0.05) - 0.05) < 1e-12, (d, A["lr"], A.get("weight_decay"))
    assert A["epochs"] == epochs and len(b["losses"]) == epochs, (d, A["epochs"], len(b["losses"]))
    want = {"size": 32, "n_steps": 128, "batch_size": 128, "n_batches": n_batches, "data_workers": 3}
    assert all(A.get(k) == v for k, v in want.items()), (d, {k: A.get(k) for k in want})
    L = np.asarray(b["losses"], float); div = not np.all(np.isfinite(L))
    k = classify_run(L); k10 = classify_run(L, solved_loss=0.1)
    fin = lambda x: float(x) if x is not None and np.isfinite(x) else 0.0   # non-finite accuracy counts as 0 (declared)
    rec = {"acc": fin(ev["eval"]["128"]["acc"]), "acc4": fin(ev["eval"]["512"]["acc"]),
           "solved": (k["registered"] == "SOLVED") and not div, "solved10": (k10["registered"] == "SOLVED") and not div,
           "cls": "DIVERGED" if div else k["cls"], "score": sel_score(L), "auc": float(np.nanmean(L)),
           "wrap": (ev.get("wrap") or [None, float("nan")])[1]}
    cls = MapFormerWM_SignedAO_r4 if arm == "SO" else MapFormerWM_AbsAO_r4
    off = b["config"]["action_offset"]
    m = cls(vocab_size=b["config"]["vocab_size"], d_model=128, n_heads=2, n_layers=1, grid_size=32, action_ids=(off, off + 1))
    m.load_state_dict(b["model_state_dict"])
    with torch.no_grad():
        rec["opp"] = opp_share(m, ((0, 1),))
        rec["wrapw"] = wrap_share_weighted(m, usage_weights(m, tok, action_offset=off, n_actions=2)) if tok is not None \
            else float("nan")
    return rec


def load(root, rates, seeds, epochs, n_batches):
    te = GridWorldSteps(size=32, seed=10000, p_first=0.5, deltas=(1, -1)); np.random.seed(10 ** 6)
    tok = torch.stack([te.generate_trajectory(128)[0] for _ in range(32)])
    R = {}
    for lr in rates:
        for arm in ARMS:
            for s in seeds:
                R[lr, arm, s] = run_record(f"{root}/{rate_tag(lr)}/{arm}_s{s}", arm, s, lr, epochs, n_batches, tok)
    return R


def arrays(R, rates, seeds, key):
    """(2n, R) array: rows 0..n-1 SO seeds, n..2n-1 AO seeds; columns rates ascending."""
    return np.array([[R[lr, arm, s][key] for lr in rates] for arm in ARMS for s in seeds], dtype=float)


# ------------------------------------------------------------------------------ analysis
def analyse(R, rates, seeds, out=print):
    n = len(seeds); rates = sorted(rates)
    S, SC, A = arrays(R, rates, seeds, "solved").astype(bool), arrays(R, rates, seeds, "score"), arrays(R, rates, seeds, "acc")
    t = b1_test(S, SC, A, n)
    r_ao, r_so = rates[t["r_ao"]], rates[t["r_so"]]
    lab, P = b1_decide(t, A, S, n); nf, npp = P["naive_fisher_p"], P["naive_perm_p"]
    V = {"B1": {"label": lab, "AO_rate": r_ao, "SO_rate": r_so, "AO_solved": t["k_ao"], "SO_solved": t["k_so"],
                "AO_acc": t["acc_ao"], "SO_acc": t["acc_so"], "d": t["acc_ao"] - t["acc_so"], "p_k_sel": t["p_k_sel"],
                "p_acc_sel": t["p_acc_sel"], "naive_fisher_p": nf, "naive_perm_p": npp, "p_k": P["p_k"],
                "p_acc": P["p_acc"], "exact": t["exact"],
                "n_relabel": t["n_relabel"]}}
    V["reading"] = reading(lab, r_ao, r_so, rates)
    out("== B1 (registered primary): each arm at its own best rate (most SOLVED, then lowest mean final-5% loss) ==")
    out(f"AO best rate {r_ao:g}: SOLVED {t['k_ao']}/{n}, acc {t['acc_ao']:.4f} | SO best rate {r_so:g}: SOLVED "
        f"{t['k_so']}/{n}, acc {t['acc_so']:.4f} | d = AO - SO {t['acc_ao'] - t['acc_so']:+.4f}")
    out(f"selection-aware p ({'exact, ' if t['exact'] else 'MC, '}{t['n_relabel']} relabellings of {2 * n} rate profiles, "
        f"selection redone in each): SOLVED {t['p_k_sel']:.4f}, accuracy {t['p_acc_sel']:.4f}; naive at the selected rates: "
        f"Fisher {nf:.4f}, perm {npp:.4f}; DECISION p = max: SOLVED {P['p_k']:.4f}, accuracy {P['p_acc']:.4f}")
    out(f"B1: {lab}")
    out(f"Reading: {V['reading']}")
    return V, S, SC, A


def secondaries(R, rates, seeds, S, SC, A, out=print):
    n = len(seeds); rates = sorted(rates); W = {}
    out("\n== secondaries (declared; no verdict) ==")
    out("floors: predict blank ~0.50 (p_empty 0.5); +1/-1 ring n-gram 0.554; retrace 0.860 (ROBUST, post hoc)")
    out("per rate, per arm: SOLVED | SOLVED@0.1 | acc T | acc 4T | loss AUC | mean sel. score | wrap-cancel | usage-wtd "
        "wrap-cancel | opposite-sign | classes")
    for lr in rates:
        for arm in ARMS:
            r = [R[lr, arm, s] for s in seeds]
            out(f"{lr:<7g} {arm}: {sum(x['solved'] for x in r)}/{n} | {sum(x['solved10'] for x in r)}/{n} | "
                f"{np.mean([x['acc'] for x in r]):.4f} | {np.mean([x['acc4'] for x in r]):.4f} | "
                f"{np.mean([x['auc'] for x in r]):.3f} | {np.mean([x['score'] for x in r]):.4f} | "
                f"{np.nanmean([x['wrap'] for x in r]):.2f} | {np.nanmean([x['wrapw'] for x in r]):.2f} | "
                f"{np.nanmean([x['opp'] for x in r]):.2f} | {' '.join(x['cls'][:4] for x in r)}")
    out("\nper-rate AO vs SO (Holm over the 4 rates within each route; labels by the ROBUST rule on Holm-adjusted p)")
    pf, pp, rows = [], [], []
    for j, lr in enumerate(rates):
        ka, ks = int(S[n:, j].sum()), int(S[:n, j].sum())
        pf.append(fisher_solved(ka, n, ks, n)); pp.append(perm2_p(A[:n, j], A[n:, j])["p"]); rows.append((ka, ks))
    hf, hp = holm(pf), holm(pp); W["per_rate"] = {}
    for j, lr in enumerate(rates):
        ka, ks = rows[j]; aa, sa = A[n:, j].mean(), A[:n, j].mean()
        lab = label_from(ka, ks, aa, sa, hf[j], hp[j], n)
        W["per_rate"][f"{lr:g}"] = {"label": lab, "AO_solved": ka, "SO_solved": ks, "d": float(aa - sa), "fisher_p": pf[j],
                                    "perm_p": pp[j], "fisher_holm": float(hf[j]), "perm_holm": float(hp[j])}
        out(f"  lr {lr:<7g}: AO {ka}/{n} {aa:.4f} vs SO {ks}/{n} {sa:.4f}; d {aa - sa:+.4f}; Fisher p {pf[j]:.4f} "
            f"(Holm {hf[j]:.4f}); perm p {pp[j]:.4f} (Holm {hp[j]:.4f}) -> {lab}")
    S10 = arrays(R, rates, seeds, "solved10").astype(bool)
    t10 = b1_test(S10, SC, A, n)
    l10, P10 = b1_decide(t10, A, S10, n)
    W["B1_loss0.1"] = {"label": l10, **{k: t10[k] for k in ("k_ao", "k_so", "p_k_sel", "p_acc_sel")}, **P10,
                       "AO_rate": rates[t10["r_ao"]], "SO_rate": rates[t10["r_so"]]}
    out(f"\nthreshold sensitivity, B1 with SOLVED at loss 0.1 (selection redone): AO {rates[t10['r_ao']]:g} "
        f"{t10['k_ao']}/{n} vs SO {rates[t10['r_so']]:g} {t10['k_so']}/{n}; decision p SOLVED {P10['p_k']:.4f}, "
        f"acc {P10['p_acc']:.4f} -> {l10}")
    A4 = arrays(R, rates, seeds, "acc4"); t4 = b1_test(S, SC, A4, n)
    out(f"accuracy at 4T (512) at the B1-selected rates: AO {t4['acc_ao']:.4f} vs SO {t4['acc_so']:.4f}; d "
        f"{t4['acc_ao'] - t4['acc_so']:+.4f}; selection-aware p {t4['p_acc_sel']:.4f}")
    W["acc4_sel"] = {"AO": t4["acc_ao"], "SO": t4["acc_so"], "p_sel": t4["p_acc_sel"]}
    # cross-fit: select on one half of the seeds, compare on the other (n/2 per arm), both directions
    h = n // 2; W["crossfit"] = []
    for sel_half, test_half in ((range(0, h), range(h, n)), (range(h, n), range(0, h))):
        sh, th = list(sel_half), list(test_half)
        ia, iso = np.array(sh) + n, np.array(sh)
        ra, rs = int(select(S, SC, ia[None])[0]), int(select(S, SC, iso[None])[0])
        ka, ks = int(S[np.array(th) + n, ra].sum()), int(S[np.array(th), rs].sum())
        aa, sa = A[np.array(th) + n, ra], A[np.array(th), rs]
        f, p = fisher_solved(ka, len(th), ks, len(th)), perm2_p(sa, aa)["p"]
        W["crossfit"].append({"select_on": [seeds[i] for i in sh], "AO_rate": rates[ra], "SO_rate": rates[rs], "AO_solved": ka,
                              "SO_solved": ks, "d": float(aa.mean() - sa.mean()), "fisher_p": f, "perm_p": p})
        out(f"cross-fit: rates chosen on seeds {seeds[sh[0]]}-{seeds[sh[-1]]} (AO {rates[ra]:g}, SO {rates[rs]:g}), tested on "
            f"{seeds[th[0]]}-{seeds[th[-1]]}: AO {ka}/{len(th)} {aa.mean():.4f} vs SO {ks}/{len(th)} {sa.mean():.4f}; Fisher p "
            f"{f:.4f}, perm p {p:.4f}")
    out("SOLVED-vs-lr curve (descriptive): " + "; ".join(
        f"{arm} " + " ".join(f"{lr:g}:{int(S[(n if arm == 'AO' else 0):(2 * n if arm == 'AO' else n), j].sum())}"
                             for j, lr in enumerate(rates)) for arm in ARMS))
    return W


# ------------------------------------------------------------------------------ selftest and power
def _profiles(so_solved_by_rate, ao_solved_by_rate, n=8, R=4, acc_hi=1.0, acc_lo=0.90, rng=None):
    """Synthetic profiles: per rate the first k seeds of an arm SOLVED (loss 0.001, acc_hi), others STALLED (loss 0.3,
    acc_lo)."""
    S = np.zeros((2 * n, R), bool); SC = np.full((2 * n, R), 0.3); A = np.full((2 * n, R), acc_lo)
    for off, ks in ((0, so_solved_by_rate), (n, ao_solved_by_rate)):
        for j, k in enumerate(ks):
            S[off:off + k, j] = True; SC[off:off + k, j] = 0.001; A[off:off + k, j] = acc_hi
    if rng is not None:
        SC = SC + rng.uniform(0, 1e-3, SC.shape); A = np.clip(A + rng.normal(0, 0.005, A.shape) * ~S, 0, 1)
    return S, SC, A


def selftest():
    n = 8; rates = list(DEFAULT_RATES); rng = np.random.default_rng(0)
    # label_from must equal analyze_robust.cell_label when given Fisher / perm p (the ROBUST rule, unchanged)
    hi, lo, mid = [1.0] * n, [0.9] * n, [0.97, 0.99, 1.0, 0.95, 1.0, 0.98, 0.96, 1.0]
    for ao, so, ka, ks in ((lo, hi, 1, 8), (hi, hi, 8, 8), (mid, hi, 5, 8), (lo, lo, 0, 3), (hi, lo, 8, 4), (hi, hi, 4, 4)):
        a, b = np.array(ao), np.array(so)
        mine = label_from(ka, ks, a.mean(), b.mean(), fisher_solved(ka, n, ks, n), perm2_p(b, a)["p"], n)
        assert mine == robust_cell_label(ao, so, ka, ks, n)[0], (mine, ao, so, ka, ks)
    print("selftest: label_from == analyze_robust.cell_label on the ROBUST selftest cases")
    cases = {   # name: (SO SOLVED by rate, AO SOLVED by rate), rates 1e-3, 3e-3, 1e-2, 3e-2
        "COST AT BEST RATES": ((8, 8, 8, 6), (1, 3, 2, 0)),
        "COST AT BEST RATES, AO at edge": ((8, 8, 8, 2), (0, 1, 1, 2)),
        "COST CLOSES AT HIGHER LR (NSC)": ((8, 8, 7, 3), (1, 4, 7, 8)),
        "COST CLOSES AT HIGHER LR (NLB)": ((4, 4, 2, 0), (0, 2, 8, 8), 0.75, 0.99),
        "NO COST AT BEST RATES": ((8, 6, 5, 2), (8, 7, 6, 5)),
        "UNMEASURED": ((8, 8, 8, 8), (1, 4, 5, 3)),
        "UNREADABLE": ((3, 2, 1, 0), (0, 1, 1, 0)),
    }
    seen = set()
    for name, (so, ao, *lo) in cases.items():
        S, SC, A = _profiles(so, ao, acc_lo=lo[0] if lo else 0.90, rng=rng)
        if len(lo) > 1:                                    # SO's solved runs slightly below AO's (accuracy route to NLB)
            A[:n][S[:n]] = lo[1]
        t = b1_test(S, SC, A, n)
        lab, P = b1_decide(t, A, S, n)
        rd = reading(lab, rates[t["r_ao"]], rates[t["r_so"]], rates)
        print(f"selftest {name:32s}: AO {rates[t['r_ao']]:g} {t['k_ao']}/8 vs SO {rates[t['r_so']]:g} {t['k_so']}/8; p_sel "
              f"{t['p_k_sel']:.4f} / {t['p_acc_sel']:.4f}; decision p {P['p_k']:.4f} / {P['p_acc']:.4f}; {lab} -> {rd}")
        seen.add(lab if not lab.startswith(UNR) else UNR); seen.add(rd.split(" (")[0].split(" [")[0])
    # tie-break: equal SOLVED -> lower mean score wins; equal score -> lower rate
    S = np.zeros((2, 3), bool); SC = np.array([[0.2, 0.1, 0.1], [0.2, 0.1, 0.1]])
    assert select(S, SC, np.array([[0, 1]]))[0] == 1
    S[:, 2] = True; assert select(S, SC, np.array([[0, 1]]))[0] == 2
    # the null reproduces the plain Fisher-like distribution when only one rate exists
    S1, SC1, A1 = _profiles((8,), (1,), R=1)
    t1 = b1_test(S1, SC1, A1, n); assert abs(t1["p_k_sel"] - fisher_solved(1, 8, 8, 8)) < 1e-12, t1["p_k_sel"]
    print(f"selftest: one-rate null = Fisher ({t1['p_k_sel']:.4f}); tie-breaks ok")
    need = {SLB, NSC, NLB, UNM, UNR, "COST AT BEST RATES", "COST CLOSES AT HIGHER LR", "NO COST AT BEST RATES", "UNMEASURED",
            "UNREADABLE"}
    assert need <= seen, need - seen
    assert "not bracketed" in reading(SLB, 3e-2, 1e-3, rates)
    print("selftest passed: every B1 label and every reading reached (plus the edge qualifier)")


def power():
    """Selection-aware p for plausible outcomes (SO at ceiling at the first three rates); AO's best count k at one rate."""
    n = 8; rng = np.random.default_rng(1)
    print("selection-aware SOLVED p (two-sided) vs naive Fisher at the selected rates; AO best count at one rate, others fewer")
    for so in ((8, 8, 8, 8), (8, 8, 8, 0), (7, 8, 6, 0)):
        for k in range(0, 8):
            ao = (min(k, 1), min(k, 3), k, max(k - 2, 0))
            S, SC, A = _profiles(so, ao, rng=rng); t = b1_test(S, SC, A, n)
            print(f"  SO by rate {so}, AO {ao}: SO {t['k_so']}/8 vs AO {t['k_ao']}/8 -> p_sel {t['p_k_sel']:.4f} "
                  f"(naive Fisher {fisher_solved(t['k_ao'], n, t['k_so'], n):.4f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=f"{REPO}/runs/lrsweep")
    ap.add_argument("--seeds", default="1500-1507")
    ap.add_argument("--rates", default=",".join(f"{r:g}" for r in DEFAULT_RATES))
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--n-batches", type=int, default=98, help="smoke only")
    ap.add_argument("--json-out", default=f"{REPO}/LRSWEEP_VERDICTS.json")
    ap.add_argument("--no-json", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--power", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    if a.power:
        power(); return
    lo, _, hi = a.seeds.partition("-"); seeds = list(range(int(lo), int(hi or lo) + 1))
    rates = sorted(float(x) for x in a.rates.split(","))
    R = load(a.root, rates, seeds, a.epochs, a.n_batches)
    print(f"== LRSWEEP: ring +1/-1, 32 cells, T = 128, {a.epochs} epochs, wd 0.05; rates {', '.join(f'{r:g}' for r in rates)};"
          f" seeds {seeds[0]}-{seeds[-1]} ==")
    print("per seed: acc T = 128 and class | " + " | ".join(f"{lr:g}:{arm}" for lr in rates for arm in ARMS))
    for s in seeds:
        print(f"{s} | " + " | ".join(f"{R[lr, arm, s]['acc']:.4f} {R[lr, arm, s]['cls'][:4]}" for lr in rates for arm in ARMS))
    print()
    V, S, SC, A = analyse(R, rates, seeds)
    if not a.no_json:                                       # verdicts written before any secondary can fail
        json.dump(V, open(a.json_out, "w"), indent=2)
    W = secondaries(R, rates, seeds, S, SC, A)
    if not a.no_json:
        json.dump({**V, "secondaries": W}, open(a.json_out, "w"), indent=2)


if __name__ == "__main__":
    main()
