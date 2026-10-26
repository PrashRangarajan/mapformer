"""Registered analysis for EXPTORUS_PREREG.md: do the exp and softplus phase increments pay the sign learnability cost in
ATTENTION on the T = 1024 torus, each arm at its own best of two learning rates? CPU.

  python3 -m mapformer.analyze_exptorus [--root runs/exptorus] [--seeds 3400-3407] [--rates 1e-3,3e-3] [--json-out P|--no-json]
  python3 -m mapformer.analyze_exptorus --selftest       # every label / reading / qualifier, synthetic, no run dirs
  python3 -m mapformer.analyze_exptorus --power          # deterministic and simulated power of the Holm'd primaries
  python3 -m mapformer.analyze_exptorus --init-census [--seeds ...]   # initial pair sums vs pi per arm (no training needed)
  python3 -m mapformer.analyze_exptorus --pilot [--pilot-root runs/exptorus_pilot] [--pilot-seeds 3490-3491]

Layout: <root>/<tag>/<arm>/p0/<V>_s<S>/<V>.pt, tag = rate_tag(lr) (lr1e-03, lr3e-03); arm SO / AO / EXP / SPL;
V = SignedAO4_r4 / AbsAO4_r4 / ExpAO4_r4 / SoftplusAO4_r4 (observation-step-free, only the four moves step). Accuracy from
eval_noise_refine (<eval-dir>/EXPTORUS_<tag>_<arm>.json; held-out map, env seed 10000, 100 walks at T = 1024 and 2048);
strata from eval_rank_strata (..._STRATA.json; a declared secondary, optional).

Registered machinery, imported unchanged:
  analyze_lrsweep    label_from (the ROBUST cell rule), select (most SOLVED, then lowest mean selection score, then lower
                     rate), b1_test (selection-aware permutation null over all C(16, 8) splits of the 16 per-seed rate
                     profiles, the selection redone inside each split), b1_decide (decision p = max(selection-aware, naive at
                     the selected rates) per route), holm, sel_score
  analyze_torus_lrsweep  RECIPE, h_map, runaway (RUNAWAY / COLLAPSED), usage_stream, rate_tag, decay_exponent
What is new: four arms and three contrasts against the signed anchor (EXP and SPL primaries, Holm over the two on each
route's decision p; AO the anchor gate at raw p), the reading, the initial past-pi census and the per-channel cancellation
census (wrap vs through zero, overlap with the past-pi start set).
"""
import argparse, glob, json, math, os, re

import numpy as np
import torch

from mapformer.analyze_lrsweep import label_from, select, b1_test, b1_decide, holm, sel_score, _profiles
from mapformer.analyze_robust import opp_share, SLB, NSC, UNM, NLB
from mapformer.analyze_torus_lrsweep import RECIPE, h_map, runaway, usage_stream, rate_tag, decay_exponent
from mapformer.environment import GridWorld
from mapformer.model_exptorus import MapFormerWM_ExpAO4_r4, MapFormerWM_SoftplusAO4_r4
from mapformer.model_wrap_exist import (MapFormerWM_SignedAO4_r4, MapFormerWM_AbsAO4_r4, action_angles, usage_weights,
                                        wrap_share_weighted)
from mapformer.stats_core import perm2_p, fisher_solved, classify_run

REPO = "/home/prashr/mapformer"
ARMS = ("SO", "AO", "EXP", "SPL")
VAR = {"SO": "SignedAO4_r4", "AO": "AbsAO4_r4", "EXP": "ExpAO4_r4", "SPL": "SoftplusAO4_r4"}
CLS = {"SO": MapFormerWM_SignedAO4_r4, "AO": MapFormerWM_AbsAO4_r4, "EXP": MapFormerWM_ExpAO4_r4,
       "SPL": MapFormerWM_SoftplusAO4_r4}
NAME = {"SO": "signed", "AO": "|.|", "EXP": "exp", "SPL": "softplus"}
PRIMARY = ("EXP", "SPL")            # Holm over these two, per route
ANCHOR = "AO"                       # gate: |.| vs signed must read SIGNED LEARNS BETTER (raw decision p)
DEFAULT_RATES = (1e-3, 3e-3)
PAIRS = ((0, 1), (2, 3))            # environment.GridWorld.ACTION_DELTAS: N/S, W/E
LENGTHS = (1024, 2048)
UNR = "UNREADABLE"
SEL_ONLY = "only at its selected rate"   # Amendment 1.2 wording
MATCHED = 1e-3                          # G-3 A2's rate; signed's best on the torus (G-4)
BIG, TOL = 0.05, 0.1                # non-blind: |wrap(angle of the pair's first move)| > BIG; cancelling: |wrap(sum)| < TOL
PREFIX = "EXPTORUS"
FLOORS = ("floors (T = 1024 eval stream, post hoc, docs/audits/2026-10-20/WHAT_WHERE_T1024.md): blank 0.506, best n-gram "
          "0.575, retrace 0.843; torus always-blank 0.507 (GENERALITY)")
CROSS_BATCH = ("cross-batch references, labelled, never pooled (rule 12): TORUS_LRSWEEP (seeds 2400-2407, same recipe): "
               "|.| best 3e-3 0/8 0.890 vs signed best 1e-3 8/8 1.000; per rate signed 8/8 (1e-3), 0/8 (3e-3, all "
               "DESCENDING). G-3 A2 (SSM, ring, lr 1e-3): exp 6/8 vs signed 7/8 NO SIGN COST; softplus 0/8. SIGNROUTE "
               "(attention, ring): exp 1/8 at 1e-3 and at 3e-3, softplus 1/8 and 3/8, vs signed 8/8")


# ------------------------------------------------------------------------------ init reconstruction and censuses
def init_model(arm, seed):
    """The model train_variant builds at this seed, before any step: torch / numpy seed, then the GridWorld exactly as
    train_variant constructs it (its defaults for the torus), then the model. test_exptorus checks this against a
    checkpoint trained at lr 0 (bitwise)."""
    torch.manual_seed(seed); np.random.seed(seed)
    env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, n_landmarks=0, seed=seed, action_mode="translate",
                    obs_mode="allo", boundary="torus", score_moves_only=False, action_record="commanded", n_headings=4,
                    heading_noise=0.0)
    return CLS[arm](vocab_size=env.unified_vocab_size, d_model=128, n_heads=2, n_layers=1, grid_size=64)


def wrapa(x):
    return torch.angle(torch.exp(1j * x))


@torch.no_grad()
def pair_sums(m):
    """(P, C) raw per-channel pair sums omega_c (a_c(x) + a_c(y)) in radians, and (P, C) the pair's first-move angle."""
    A = action_angles(m).double()
    return torch.stack([A[i] + A[j] for i, j in PAIRS]), torch.stack([A[i] for i, _ in PAIRS])


@torch.no_grad()
def init_census(m0):
    """Initial past-pi record: per pair, channels whose raw pair sum |omega (a(x) + a(y))| exceeds pi (the half-way point to
    the 2 pi wrap: from there the nearest cancelling point is the wrap, not zero); both pairs; and the omega-ladder count
    an exactly-1 / ln 2 step would give (G-3 A2's analytic definition)."""
    S, _ = pair_sums(m0); past = S.abs() > math.pi                       # (P, C)
    om = m0.path_integrator.omega.detach().double().reshape(-1)
    return {"past_by_pair": [int(x) for x in past.sum(1)], "past_both": int(past.all(0).sum()),
            "past_any": int(past.any(0).sum()), "n_channels": int(S.shape[1]),
            "analytic_exp": int((2 * om > math.pi).sum()), "analytic_softplus": int((2 * math.log(2) * om > math.pi).sum()),
            "median_abs_sum": float(S.abs().median()), "max_abs_sum": float(S.abs().max())}


@torch.no_grad()
def cancel_census(m, m0):
    """Per (pair, channel) at the end of training: non-blind (|wrap(first move angle)| > BIG), cancelling (|wrap(pair sum)|
    < TOL), cancelling by a wrap (round(sum / 2 pi) != 0) or through zero (= 0); and the overlap with the initial past-pi set
    (|initial raw pair sum| > pi). Counts over both pairs (2 x 64 (pair, channel) cells per model)."""
    S, A1 = pair_sums(m); S0, _ = pair_sums(m0)
    nb = wrapa(A1).abs() > BIG
    canc = (wrapa(S).abs() < TOL) & nb
    k = torch.round(S / (2 * math.pi))
    wrapc, zeroc = canc & (k != 0), canc & (k == 0)
    past0 = S0.abs() > math.pi
    n_p, n_np = int((past0 & nb).sum()), int((~past0 & nb).sum())
    c_p, c_np = int((canc & past0).sum()), int((canc & ~past0).sum())
    return {"nonblind": int(nb.sum()), "cancel": int(canc.sum()), "cancel_wrap": int(wrapc.sum()),
            "cancel_zero": int(zeroc.sum()), "past0": int(past0.sum()), "past0_nonblind": n_p,
            "cancel_from_past0": c_p, "cancel_not_past0": c_np,
            "p_cancel_given_past0": c_p / n_p if n_p else float("nan"),
            "p_cancel_given_not_past0": c_np / n_np if n_np else float("nan"),
            "turns": sorted({int(x) for x in k[canc].abs().tolist()}),
            "omega_ratio_median": float((m.path_integrator.omega.detach().abs()
                                         / m0.path_integrator.omega.detach().abs()).median()),
            "step_min": float(m.step_table()[m.act].min()), "step_absmax": float(m.step_table()[m.act].abs().max())}


# ------------------------------------------------------------------------------ registered reading
def gate(lab_anchor):
    """Anchor gate: the batch reproduces the torus sign cost (|.| vs signed at best rates reads SIGNED LEARNS BETTER)."""
    return lab_anchor == SLB


def arm_reading(arm, lab, gate_ok, lab_anchor, r_sel, rates, desc=0, unstable=0, lab_matched=None):
    """Registered per-arm reading (EXPTORUS_PREREG.md Sec. 4 + Amendment 1). lab: the arm's B1 label on Holm-adjusted
    decision p; r_sel: the arm's selected rate; desc / unstable: the arm's runs DESCENDING / UNSTABLE-or-DIVERGED and
    unsolved at r_sel (Amendment 1.1); lab_matched: the arm's per-rate label at the matched rate 1e-3 (signed at its best,
    the arm at A2's pre-specified rate, no selection; Amendment 1.2), None when 1e-3 is the only rate."""
    nm = NAME[arm]
    q = ""
    if r_sel == max(rates) and len(rates) > 1:
        q += f" [{nm}'s best rate is the grid's top rate: its optimum is not bracketed; 'best' = best TESTED]"
    if desc:
        q += f" [{desc} {nm} run(s) still DESCENDING at its best rate: budget-scoped]"
    if lab.startswith(UNR):                    # Amendment 1 NOTE: name the right reason when signed < 4/8
        return f"UNREADABLE (the signed arm solves < 4/8 at its best rate)" + q
    if not gate_ok:
        return (f"UNREADABLE (anchor gate: |.| vs signed reads {lab_anchor}, so the batch does not reproduce the torus sign "
                f"cost and no {nm} contrast is read)") + q
    if unstable >= 2:
        q += (f" [{unstable} unsolved {nm} runs UNSTABLE (RUNAWAY / COLLAPSED / DIVERGED) at its best rate: optimiser "
              f"instability; no learnability sentence rests on this cell]")
    if lab in (NSC, NLB):                      # Amendment 1.2: matched-rate check on any non-cost reading
        ok = lab_matched is None or lab_matched in (NSC, NLB)
        where = (f" at the matched rate 1e-3" if ok else
                 f" {SEL_ONLY} {r_sel:g} (selection-inflated; 1e-3 per-rate reads {lab_matched})")
        if lab == NSC:
            return f"NO SIGN COST FOR {nm.upper()} IN ATTENTION ON THE TORUS{where}" + q
        return f"{nm.upper()} LEARNS BETTER THAN SIGNED ON THE TORUS (NON-NEGATIVE LEARNS BETTER){where}" + q
    if lab == SLB:
        return f"{nm.upper()} PAYS THE SIGN COST IN ATTENTION ON THE TORUS (SIGNED LEARNS BETTER at best rates)" + q
    return f"UNMEASURED ({nm} vs signed at best rates)" + q


def batch_reading(r_exp, r_spl):
    """One line for the paper's sign sentence, from the two arm readings (registered map, prereg Sec. 4 + Amendment 1).
    An unstable arm is never read as 'pays' or 'escapes' (Amendment 1.1); an escape seen only at the selected rate carries
    that qualifier into the batch reading (Amendment 1.2)."""
    if r_exp.startswith(UNR) or r_spl.startswith(UNR):
        return "UNREADABLE" if r_exp.startswith(UNR) and r_spl.startswith(UNR) else "PARTLY UNREADABLE"
    pay = lambda r: "PAYS THE SIGN COST" in r and "UNSTABLE" not in r
    free = lambda r: (r.startswith("NO SIGN COST") or "LEARNS BETTER THAN SIGNED" in r) and "UNSTABLE" not in r
    sel = lambda *rs: (" [" + "; ".join(f"{n} escapes {SEL_ONLY} (selection-inflated)" for n, r in rs if SEL_ONLY in r)
                       + "]") if any(SEL_ONLY in r for _, r in rs) else ""
    if pay(r_exp) and pay(r_spl):
        return ("BOTH PAY: in attention on the torus the exp (LRU-style positivity) and softplus (Mamba-Delta-style) "
                "content-dependent increments pay the cost (the SSM's exp no-cost does not transfer)")
    if free(r_exp) and pay(r_spl):
        return ("EXP ESCAPES, SOFTPLUS PAYS: exp's no-cost replicates in attention on the torus; softplus pays"
                + sel(("exp", r_exp)))
    if free(r_exp) and free(r_spl):
        return ("NEITHER PAYS: the cost is specific to |.| among the tested non-negative increments"
                + sel(("exp", r_exp), ("softplus", r_spl)))
    if pay(r_exp) and free(r_spl):
        return "SOFTPLUS ESCAPES, EXP PAYS (not predicted)" + sel(("softplus", r_spl))
    return "MIXED / UNMEASURED: at least one of exp, softplus is UNMEASURED or unstable at its best rate"


# ------------------------------------------------------------------------------ loading
def check_ckpt(ck, arm, s, lr, wd, epochs, n_batches, smoke=False):
    """Load a checkpoint and assert the registered recipe (relaxed to the CPU smoke's sizes with smoke=True)."""
    b = torch.load(ck, map_location="cpu", weights_only=False); c = b["config"]; A = c.get("args", {})
    want = {**RECIPE, "n_batches": n_batches, "epochs": epochs}
    if smoke:
        for k in ("n_steps", "batch_size", "fast_attn"):
            want.pop(k)
    bad = {k: (c.get(k), v) for k, v in want.items() if c.get(k) != v}
    for k, v in (("data_workers", 3), ("variant", VAR[arm]), ("seed", s), ("env", "torus"), ("p_transition_noise", 0.0),
                 ("p_action_noise", 0.0), ("action_mode", "translate"), ("snapshot_every", 0), ("aux_coef", 0.0)):
        if smoke and k == "data_workers":
            continue
        if A.get(k) != v:
            bad[f"args.{k}"] = (A.get(k), v)
    if abs(c.get("lr", -1) - lr) > 1e-9 * lr or abs(A.get("weight_decay", -1) - wd) > 1e-6 * max(wd, 1e-12):
        bad["lr/wd"] = ((c.get("lr"), A.get("weight_decay")), (lr, wd))
    assert not bad and b["variant"] == VAR[arm] and b["seed"] == s and len(b["losses"]) == epochs, (ck, bad, len(b["losses"]))
    return b


def run_record(ck, arm, s, lr, epochs, n_batches, ev, st, tok, h, lengths=LENGTHS, smoke=False, models=True):
    b = check_ckpt(ck, arm, s, lr, 0.05, epochs, n_batches, smoke); c = b["config"]
    L = np.asarray(b["losses"], float); div = not np.all(np.isfinite(L))
    k = classify_run(L); k10 = classify_run(L, solved_loss=0.1)
    fin = lambda x: float(x) if x is not None and np.isfinite(x) else 0.0      # non-finite accuracy counts 0 (declared)
    acc = {T: dict((x[0], x[1]) for x in ev[f"0.0|{VAR[arm]}|{T}"]) for T in lengths}
    rec = {"acc": 0.0 if div else fin(acc[lengths[0]][s]), "acc2": 0.0 if div else fin(acc[lengths[1]][s]),
           "solved": k["registered"] == "SOLVED" and not div, "solved10": k10["registered"] == "SOLVED" and not div,
           "cls": "DIVERGED" if div else k["cls"], "score": sel_score(L),
           "auc": float(np.nanmean(L)) if np.isfinite(L).any() else float("nan"), "tail": float(k["tail"]),
           "final_loss": float(L[-max(1, round(0.05 * len(L))):].mean()),
           "strata": (st or {}).get(f"{VAR[arm]}|{s}|{lengths[0]}")}
    m = CLS[arm](vocab_size=c["vocab_size"], d_model=128, n_heads=2, n_layers=1, grid_size=64)
    m.load_state_dict(b["model_state_dict"]); m.eval()
    with torch.no_grad():
        step_absmax = float(m.step_table()[m.act].abs().max())
    warm = max(1, int(0.05 * epochs))
    rec["runaway"] = (not div) and runaway(L, step_absmax, warm, h)
    rec["step_absmax"] = step_absmax; rec["h_map"] = h
    post = L[warm:][np.isfinite(L[warm:])]
    rec["maxpost_minus_h"] = float(post.max() - h) if post.size else float("nan")
    fin_ = np.where(np.isfinite(L[warm:]), L[warm:], -np.inf)
    rec["maxpost_epoch"] = int(warm + np.argmax(fin_) + 1) if post.size else None
    rec.update(census=None, wrapw=float("nan"), opp=float("nan"))
    if models and not div and all(torch.isfinite(p).all() for p in m.parameters()):
        try:                         # a model-derived secondary can never block the verdict
            m0 = init_model(arm, s)
            assert m0.token_emb.weight.shape == m.token_emb.weight.shape
            rec["census"] = cancel_census(m, m0)
            rec["init"] = init_census(m0)
            with torch.no_grad():
                rec["wrapw"] = wrap_share_weighted(m, usage_weights(m, tok, action_offset=0, n_actions=4), pairs=PAIRS)
                rec["opp"] = opp_share(m, PAIRS)
        except Exception as e:       # noqa: BLE001 -- logged, secondaries become NaN / None
            print(f"WARNING: model secondaries failed for {ck}: {e!r}")
    return rec


def load(root, ev_dir, rates, seeds, epochs, n_batches, lengths=LENGTHS, smoke=False, models=True):
    tok = None
    if models:
        try:
            tok = usage_stream()
        except Exception as e:       # noqa: BLE001
            print(f"WARNING: usage stream unavailable ({e!r}); model secondaries will be NaN"); models = False
    H = {s: h_map(s) for s in seeds}
    R = {}
    for lr in rates:
        for arm in ARMS:
            tag = rate_tag(lr)
            ev = json.load(open(f"{ev_dir}/{PREFIX}_{tag}_{arm}.json"))
            sf = f"{ev_dir}/{PREFIX}_{tag}_{arm}_STRATA.json"
            st = json.load(open(sf)) if os.path.exists(sf) else None
            for s in seeds:
                ck = f"{root}/{tag}/{arm}/p0/{VAR[arm]}_s{s}/{VAR[arm]}.pt"
                R[lr, arm, s] = run_record(ck, arm, s, lr, epochs, n_batches, ev, st, tok, H[s], lengths, smoke, models)
    return R


def arrays(R, rates, seeds, key, arm):
    """(2n, R): rows 0..n-1 the signed seeds, n..2n-1 `arm`'s seeds; columns rates ascending."""
    return np.array([[R[lr, a, s][key] for lr in rates] for a in ("SO", arm) for s in seeds], dtype=float)


# ------------------------------------------------------------------------------ analysis
def contrast(R, rates, seeds, arm):
    """B1 for one arm vs signed: selection-aware test and the decision p per route (unadjusted)."""
    n = len(seeds)
    S, SC, A = (arrays(R, rates, seeds, "solved", arm).astype(bool), arrays(R, rates, seeds, "score", arm),
                arrays(R, rates, seeds, "acc", arm))
    t = b1_test(S, SC, A, n); _, P = b1_decide(t, A, S, n)
    r_a, r_s = rates[t["r_ao"]], rates[t["r_so"]]
    desc = sum(R[r_a, arm, s]["cls"] == "DESCENDING" for s in seeds)
    uns = sum((R[r_a, arm, s]["runaway"] or R[r_a, arm, s]["cls"] == "DIVERGED") and not R[r_a, arm, s]["solved"]
              for s in seeds)                                              # Amendment 1.1: DIVERGED is instability
    return {"arm": arm, "rate": r_a, "SO_rate": r_s, "k": t["k_ao"], "k_SO": t["k_so"], "acc": t["acc_ao"],
            "acc_SO": t["acc_so"], "d": t["acc_ao"] - t["acc_so"], "p_k_sel": t["p_k_sel"], "p_acc_sel": t["p_acc_sel"],
            "naive_fisher_p": P["naive_fisher_p"], "naive_perm_p": P["naive_perm_p"], "p_k": P["p_k"], "p_acc": P["p_acc"],
            "exact": t["exact"], "n_relabel": t["n_relabel"], "desc_at_best": int(desc), "unstable_at_best": int(uns)}, (S, SC, A)


def per_rate_labels(M, rates, n):
    """Per-rate contrast of each arm vs signed at the same rate (Holm over the rates within the arm and route; ROBUST rule
    on Holm p). Returns {arm: {rate: dict}}. Used by the registered matched-rate rule (Amendment 1.2) and the secondary."""
    out = {}
    for arm, (S, SC, A) in M.items():
        pf, pp, rows = [], [], []
        for j, lr in enumerate(rates):
            ka, ks = int(S[n:, j].sum()), int(S[:n, j].sum())
            pf.append(fisher_solved(ka, n, ks, n)); pp.append(perm2_p(A[:n, j], A[n:, j])["p"]); rows.append((ka, ks))
        hf, hp = holm(pf), holm(pp); out[arm] = {}
        for j, lr in enumerate(rates):
            ka, ks = rows[j]; aa, sa = A[n:, j].mean(), A[:n, j].mean()
            out[arm][lr] = {"label": label_from(ka, ks, aa, sa, hf[j], hp[j], n), "k": ka, "k_SO": ks, "acc": float(aa),
                            "acc_SO": float(sa), "d": float(aa - sa), "fisher_p": pf[j], "perm_p": pp[j],
                            "fisher_holm": float(hf[j]), "perm_holm": float(hp[j])}
    return out


def decide(C, n, rates, PR=None):
    """Labels: the anchor at raw decision p; the two primaries on Holm-adjusted decision p (Holm over EXP and SPL, per
    route). Then the gate and the readings. C: {arm: contrast dict}; PR: per_rate_labels (Amendment 1.2)."""
    a = C[ANCHOR]
    a["label"] = label_from(a["k"], a["k_SO"], a["acc"], a["acc_SO"], a["p_k"], a["p_acc"], n)
    hk = holm([C[x]["p_k"] for x in PRIMARY]); ha = holm([C[x]["p_acc"] for x in PRIMARY])
    for i, x in enumerate(PRIMARY):
        c = C[x]; c["p_k_holm"], c["p_acc_holm"] = float(hk[i]), float(ha[i])
        c["label"] = label_from(c["k"], c["k_SO"], c["acc"], c["acc_SO"], c["p_k_holm"], c["p_acc_holm"], n)
    ok = gate(a["label"])
    for x in PRIMARY:
        c = C[x]
        lm = PR[x][MATCHED]["label"] if (PR and len(rates) > 1 and MATCHED in PR[x]) else None
        c["label_matched_1e3"] = lm
        c["reading"] = arm_reading(x, c["label"], ok, a["label"], c["rate"], rates, c["desc_at_best"], c["unstable_at_best"],
                                   lm)
    return {"gate_ok": bool(ok), "anchor_label": a["label"],
            "batch_reading": batch_reading(C["EXP"]["reading"], C["SPL"]["reading"])}


def analyse(R, rates, seeds, out=print):
    n = len(seeds); rates = sorted(rates); C, M = {}, {}
    for arm in ("AO",) + PRIMARY:
        C[arm], M[arm] = contrast(R, rates, seeds, arm)
    PR = per_rate_labels(M, rates, n)
    G = decide(C, n, rates, PR)
    out("== B1 (registered): each arm at its own best rate (most SOLVED, then lowest mean final-5% loss, then lower rate), "
        "vs signed at ITS best rate ==")
    for arm in ("AO",) + PRIMARY:
        c = C[arm]
        hol = (f"; Holm over exp/softplus: SOLVED {c['p_k_holm']:.4f}, acc {c['p_acc_holm']:.4f}" if arm in PRIMARY
               else " (anchor gate: raw decision p)")
        mt = (f" | matched-rate 1e-3 per-rate label (Amendment 1.2): {c['label_matched_1e3']}"
              if arm in PRIMARY and c.get("label_matched_1e3") else "")
        out(f"{NAME[arm]:8s} best {c['rate']:g}: {c['k']}/{n}, acc {c['acc']:.4f} | signed best {c['SO_rate']:g}: "
            f"{c['k_SO']}/{n}, acc {c['acc_SO']:.4f} | d {c['d']:+.4f} | selection-aware p SOLVED {c['p_k_sel']:.4f}, acc "
            f"{c['p_acc_sel']:.4f}; naive Fisher {c['naive_fisher_p']:.4f}, perm {c['naive_perm_p']:.4f}; decision p "
            f"SOLVED {c['p_k']:.4f}, acc {c['p_acc']:.4f}{hol} | DESCENDING at best {c['desc_at_best']}, UNSTABLE+unsolved "
            f"{c['unstable_at_best']} -> {c['label']}{mt}")
    out(f"anchor gate (|.| vs signed must read {SLB}): {'PASS' if G['gate_ok'] else 'FAIL'} ({G['anchor_label']})")
    for arm in PRIMARY:
        out(f"Reading {NAME[arm]}: {C[arm]['reading']}")
    out(f"Batch reading: {G['batch_reading']}")
    flagged = [(lr, arm, s) for (lr, arm, s), x in sorted(R.items()) if x["runaway"] or x["cls"] == "DIVERGED"]
    for lr, arm, s in flagged:
        x = R[lr, arm, s]
        out(f"  {'DIVERGED' if x['cls'] == 'DIVERGED' else 'UNSTABLE'} {NAME[arm]} lr {lr:g} s{s}: max post-warm-up loss - "
            f"H_map {x['maxpost_minus_h']:+.3f} at epoch {x.get('maxpost_epoch', '?')} (H_map {x['h_map']:.3f}), final-5% "
            f"loss {x['tail']:.4f}, step max {x['step_absmax']:.2f}, {x['cls']}")
    V = {"B1": C, **G, "rates": rates, "seeds": seeds,
         "per_rate": {f"{lr:g}|{a}": v for a, d in PR.items() for lr, v in d.items()}}
    return V, M


def secondaries(R, rates, seeds, M, out=print):
    n = len(seeds); rates = sorted(rates); W = {}
    out("\n== secondaries (declared; no verdict) ==")
    out(FLOORS)
    out("decay exponent sum(lr_t) x wd (wd 0.05): " + ", ".join(f"{lr:g}: {decay_exponent(lr):.1f}" for lr in rates))
    out("per rate, per arm: SOLVED | SOLVED@0.1 | acc 1024 | acc 2048 | mean sel. score | usage-wtd cancel | opposite-sign "
        "| UNSTABLE | step max (median) | classes")
    W["cells"] = {}
    for lr in rates:
        for arm in ARMS:
            r = [R[lr, arm, s] for s in seeds]
            cell = {"solved": sum(x["solved"] for x in r), "solved10": sum(x["solved10"] for x in r),
                    "acc": float(np.mean([x["acc"] for x in r])), "acc2": float(np.mean([x["acc2"] for x in r])),
                    "score": float(np.mean([x["score"] for x in r])), "wrapw": float(np.nanmean([x["wrapw"] for x in r])),
                    "opp": float(np.nanmean([x["opp"] for x in r])), "runaway": sum(x["runaway"] for x in r),
                    "step_absmax_median": float(np.median([x["step_absmax"] for x in r])), "cls": [x["cls"] for x in r]}
            W["cells"][f"{lr:g}|{arm}"] = cell
            out(f"{lr:<7g} {NAME[arm]:8s}: {cell['solved']}/{n} | {cell['solved10']}/{n} | {cell['acc']:.4f} | "
                f"{cell['acc2']:.4f} | {cell['score']:.4f} | {cell['wrapw']:.2f} | {cell['opp']:.2f} | {cell['runaway']}/{n} | "
                f"{cell['step_absmax_median']:.2f} | {' '.join(c[:4] for c in cell['cls'])}")
    # per-rate contrasts (A2's rate is 1e-3): Holm over the rates within each arm and route
    out(f"\nper-rate contrasts vs signed at the same rate (Holm over the {len(rates)} rates within each arm and route; ROBUST "
        f"rule on Holm p; 1e-3 = G-3 A2's rate)")
    W["per_rate"] = {}
    for arm, d in per_rate_labels(M, rates, n).items():
        for lr, v in d.items():
            W["per_rate"][f"{lr:g}|{arm}"] = v
            note = "" if lr == MATCHED else "  [not a sign contrast: signed is at its known-bad torus rate here]"
            out(f"  {NAME[arm]:8s} lr {lr:<7g}: {v['k']}/{n} {v['acc']:.4f} vs signed {v['k_SO']}/{n} {v['acc_SO']:.4f}; d "
                f"{v['d']:+.4f}; Fisher {v['fisher_p']:.4f} (Holm {v['fisher_holm']:.4f}); perm {v['perm_p']:.4f} (Holm "
                f"{v['perm_holm']:.4f}) -> {v['label']}{note}")
    # threshold sensitivity, T = 2048, cross-fit
    W["B1_loss0.1"], W["acc2048_sel"], W["crossfit"] = {}, {}, {}
    for arm in ("AO",) + PRIMARY:
        S, SC, A = M[arm]
        S10 = arrays(R, rates, seeds, "solved10", arm).astype(bool)
        t10 = b1_test(S10, SC, A, n); l10, P10 = b1_decide(t10, A, S10, n)
        W["B1_loss0.1"][arm] = {"label_raw_p": l10, "k": t10["k_ao"], "k_SO": t10["k_so"], **P10}
        A2 = arrays(R, rates, seeds, "acc2", arm); t2 = b1_test(S, SC, A2, n)
        W["acc2048_sel"][arm] = {"acc": t2["acc_ao"], "acc_SO": t2["acc_so"], "p_sel": t2["p_acc_sel"]}
        out(f"\n{NAME[arm]}: SOLVED at loss 0.1 (selection redone; raw decision p): {t10['k_ao']}/{n} vs {t10['k_so']}/{n}, p "
            f"SOLVED {P10['p_k']:.4f}, acc {P10['p_acc']:.4f} -> {l10}; T = 2048 at the selected rates (robustness, not "
            f"capability): {t2['acc_ao']:.4f} vs {t2['acc_so']:.4f}, selection-aware p {t2['p_acc_sel']:.4f}")
        h = n // 2; W["crossfit"][arm] = []
        for sh, th in ((list(range(0, h)), list(range(h, n))), (list(range(h, n)), list(range(0, h)))):
            ra, rs = int(select(S, SC, (np.array(sh) + n)[None])[0]), int(select(S, SC, np.array(sh)[None])[0])
            ka, ks = int(S[np.array(th) + n, ra].sum()), int(S[np.array(th), rs].sum())
            aa, sa = A[np.array(th) + n, ra], A[np.array(th), rs]
            f, p = fisher_solved(ka, len(th), ks, len(th)), perm2_p(sa, aa)["p"]
            W["crossfit"][arm].append({"select_on": [seeds[i] for i in sh], "rate": rates[ra], "SO_rate": rates[rs], "k": ka,
                                       "k_SO": ks, "d": float(aa.mean() - sa.mean()), "fisher_p": f, "perm_p": p})
            out(f"  cross-fit: chosen on {seeds[sh[0]]}-{seeds[sh[-1]]} ({NAME[arm]} {rates[ra]:g}, signed {rates[rs]:g}), "
                f"tested on {seeds[th[0]]}-{seeds[th[-1]]}: {ka}/{len(th)} {aa.mean():.4f} vs {ks}/{len(th)} {sa.mean():.4f}; "
                f"Fisher {f:.4f}, perm {p:.4f}")
    # mechanism: initial past-pi record and the end-of-training cancellation census
    out("\nmechanism (declared secondary, descriptive): initial past-pi record per arm (mean over seeds; per pair N/S, W/E, "
        "of 64 channels) and, per rate, the end-of-training census summed over seeds (pair x channel cells): non-blind | "
        "cancelling | by wrap | through zero | cancelling from the past-pi start set | P(cancel | past-pi start) vs "
        "P(cancel | not) | turns | omega end / init (median)")
    W["mechanism"] = {}
    for arm in ARMS:
        ins = [R[rates[0], arm, s].get("init") for s in seeds]
        ins = [x for x in ins if x]
        if ins:
            out(f"  init {NAME[arm]:8s}: past pi N/S {np.mean([x['past_by_pair'][0] for x in ins]):.1f}, W/E "
                f"{np.mean([x['past_by_pair'][1] for x in ins]):.1f}, both {np.mean([x['past_both'] for x in ins]):.1f}; "
                f"analytic (exp step 1: {ins[0]['analytic_exp']}, softplus ln 2: {ins[0]['analytic_softplus']}); median "
                f"|sum| {np.mean([x['median_abs_sum'] for x in ins]):.2f} rad")
        for lr in rates:
            cs = [R[lr, arm, s]["census"] for s in seeds if R[lr, arm, s]["census"]]
            if not cs:
                out(f"  {lr:<7g} {NAME[arm]:8s}: census unavailable"); continue
            tot = {k: sum(c[k] for c in cs) for k in ("nonblind", "cancel", "cancel_wrap", "cancel_zero", "past0_nonblind",
                                                      "cancel_from_past0", "cancel_not_past0")}
            nnp = tot["nonblind"] - tot["past0_nonblind"]
            pc = tot["cancel_from_past0"] / tot["past0_nonblind"] if tot["past0_nonblind"] else float("nan")
            pn = tot["cancel_not_past0"] / nnp if nnp else float("nan")
            turns = sorted(set().union(*[set(c["turns"]) for c in cs]))
            om = float(np.median([c["omega_ratio_median"] for c in cs]))
            for grp, sel in (("solved", True), ("unsolved", False)):
                g = [R[lr, arm, s]["census"] for s in seeds if R[lr, arm, s]["census"] and R[lr, arm, s]["solved"] == sel]
                tot[f"cancel_{grp}"] = sum(c["cancel"] for c in g); tot[f"nonblind_{grp}"] = sum(c["nonblind"] for c in g)
            W["mechanism"][f"{lr:g}|{arm}"] = {**tot, "p_cancel_given_past0": pc, "p_cancel_given_not_past0": pn,
                                               "turns": turns, "omega_ratio_median": om}
            out(f"  {lr:<7g} {NAME[arm]:8s}: {tot['nonblind']} | {tot['cancel']} | {tot['cancel_wrap']} | {tot['cancel_zero']} | "
                f"{tot['cancel_from_past0']} | {pc:.2f} vs {pn:.2f} | {turns} | {om:.2f}")
    out("\nrevisit strata at T = 1024 (eval_rank_strata; mean over seeds): plain lag < 128 | plain lag >= 128 | wrap-only")
    W["strata"] = {}
    for lr in rates:
        for arm in ARMS:
            rows = [R[lr, arm, s]["strata"] for s in seeds]
            if any(x is None for x in rows):
                out(f"  {lr:<7g} {NAME[arm]:8s}: strata unavailable"); continue
            v = {k: float(np.nanmean([x[k]["acc"] if x[k]["acc"] is not None else np.nan for x in rows]))
                 for k in ("plain_lag<128", "plain_lag>=128", "wrap")}
            W["strata"][f"{lr:g}|{arm}"] = v
            out(f"  {lr:<7g} {NAME[arm]:8s}: {v['plain_lag<128']:.3f} | {v['plain_lag>=128']:.3f} | {v['wrap']:.3f}")
    fl = np.array([R[k]["final_loss"] for k in sorted(R)]); ac = np.array([R[k]["acc"] for k in sorted(R)])
    ac2 = np.array([R[k]["acc2"] for k in sorted(R)]); ok = np.isfinite(fl) & (fl > 0)
    if ok.sum() > 2:
        W["rule2"] = {"r_1024": float(np.corrcoef(np.log10(fl[ok]), ac[ok])[0, 1]),
                      "r_2048": float(np.corrcoef(np.log10(fl[ok]), ac2[ok])[0, 1])}
        out(f"\nrule 2: r(log10 final loss, acc) {W['rule2']['r_1024']:+.3f} at T = {LENGTHS[0]}, "
            f"{W['rule2']['r_2048']:+.3f} at T = {LENGTHS[1]} over {int(ok.sum())} runs")
    out(CROSS_BATCH)
    return W


# ------------------------------------------------------------------------------ selftest and power
def _synthetic_R(so, by_arm, rates, seeds, rng, unstable=None, desc=None, so_acc=(1.0, 0.90), lo=None, diverged=None):
    """Synthetic run records: so / by_arm[arm] = SOLVED count per rate (the first k seeds solve, loss 0.001, acc 1.0; others
    STALLED, loss 0.3, acc ~0.90; the signed arm's solved / unsolved accuracy is so_acc). unstable / desc: {arm: count} of
    unsolved runs flagged at EVERY rate. lo: {arm: unsolved accuracy} overrides 0.90 for a non-signed arm."""
    R = {}
    for j, lr in enumerate(rates):
        for arm in ARMS:
            ks = so[j] if arm == "SO" else by_arm[arm][j]
            for i, s in enumerate(seeds):
                sv = i < ks
                hi, lo_ = so_acc if arm == "SO" else (1.0, (lo or {}).get(arm, 0.90))
                R[lr, arm, s] = {"solved": sv, "solved10": sv, "score": (0.001 if sv else 0.3) + rng.uniform(0, 1e-3),
                                 "acc": hi if sv else float(np.clip(lo_ + rng.normal(0, 0.005), 0, 1)),
                                 "acc2": 1.0 if sv else 0.8, "cls": "SOLVED" if sv else "STALLED", "runaway": False,
                                 "step_absmax": 1.0, "wrapw": float("nan"), "opp": float("nan"), "strata": None,
                                 "census": None, "final_loss": 0.001 if sv else 0.3, "tail": 0.001 if sv else 0.3,
                                 "maxpost_minus_h": -1.0, "h_map": 2.06}
            for flag, key, val in ((unstable, "runaway", True), (desc, "cls", "DESCENDING"), (diverged, "cls", "DIVERGED")):
                k_ = (flag or {}).get(arm, 0)
                uns = [s for i, s in enumerate(seeds) if i >= ks][:k_]
                for s in uns:
                    R[lr, arm, s][key] = val
                    if val == "DIVERGED":
                        R[lr, arm, s]["acc"] = R[lr, arm, s]["acc2"] = 0.0     # run_record's rule for a DIVERGED run
                        R[lr, arm, s]["score"] = 2.5                           # sel_score: the first-epoch loss
    return R


def selftest():
    import warnings; warnings.simplefilter("ignore", RuntimeWarning)      # nanmean over the synthetic NaN secondaries
    n = 8; seeds = list(range(3400, 3408)); rates = list(DEFAULT_RATES); rng = np.random.default_rng(0)
    quiet = lambda *_: None
    cases = {   # name: (SO by rate, {arm: by rate}) -> expected (anchor label, EXP label, SPL label)
        "both pay": ((8, 0), {"AO": (0, 1), "EXP": (1, 0), "SPL": (0, 2)}, (SLB, SLB, SLB)),
        "exp escapes": ((8, 0), {"AO": (0, 0), "EXP": (8, 6), "SPL": (0, 1)}, (SLB, NSC, SLB)),
        "exp escapes at 3e-3 (edge)": ((8, 0), {"AO": (0, 0), "EXP": (2, 8), "SPL": (0, 0)}, (SLB, NSC, SLB)),
        "neither pays": ((8, 1), {"AO": (1, 0), "EXP": (8, 4), "SPL": (8, 8)}, (SLB, NSC, NSC)),
        "exp unmeasured": ((8, 0), {"AO": (0, 0), "EXP": (4, 2), "SPL": (0, 0)}, (SLB, UNM, SLB)),
        "softplus better (NLB)": ((4, 0), {"AO": (0, 0), "EXP": (0, 0), "SPL": (8, 8)}, (SLB, UNM, NLB), (0.99, 0.75),
                                  {"AO": 0.60}),
        "gate fails": ((8, 0), {"AO": (6, 3), "EXP": (0, 0), "SPL": (0, 0)}, (UNM, SLB, SLB)),
        "signed < 4": ((3, 2), {"AO": (0, 0), "EXP": (0, 0), "SPL": (0, 0)}, (f"{UNR} (SO SOLVED 3 < 4)",) * 3),
    }
    seen_lab, seen_rd, seen_batch = set(), set(), set()
    for name, (so, by, want, *sa) in cases.items():
        R = _synthetic_R(so, by, rates, seeds, rng, so_acc=sa[0] if sa else (1.0, 0.90), lo=sa[1] if len(sa) > 1 else None)
        V, M = analyse(R, rates, seeds, out=quiet)
        got = (V["B1"]["AO"]["label"], V["B1"]["EXP"]["label"], V["B1"]["SPL"]["label"])
        assert got == want, (name, got, want)
        rd = (V["B1"]["EXP"]["reading"], V["B1"]["SPL"]["reading"])
        print(f"selftest {name:28s}: anchor {got[0]} | exp {V['B1']['EXP']['rate']:g} {V['B1']['EXP']['k']}/8 -> {got[1]} | "
              f"softplus {V['B1']['SPL']['rate']:g} {V['B1']['SPL']['k']}/8 -> {got[2]}\n    exp: {rd[0]}\n    batch: "
              f"{V['batch_reading']}")
        seen_lab |= {g if not g.startswith(UNR) else UNR for g in got}
        seen_rd |= {r.split(" (")[0].split(" [")[0].split(" at the matched")[0].split(f" {SEL_ONLY}")[0] for r in rd}
        seen_batch.add(V["batch_reading"].split(":")[0])
        if name == "gate fails":
            assert not V["gate_ok"] and all(r.startswith("UNREADABLE (anchor gate") for r in rd), rd
            assert V["batch_reading"] == "UNREADABLE"
        if name == "exp escapes at 3e-3 (edge)":       # Amendment 1.2: 1e-3 per-rate reads SLB -> selected-rate only
            assert "not bracketed" in rd[0] and SEL_ONLY in rd[0] and "1e-3 per-rate reads SIGNED" in rd[0], rd[0]
            assert SEL_ONLY in V["batch_reading"] and V["batch_reading"].startswith("EXP ESCAPES"), V["batch_reading"]
        if name == "exp escapes":                      # matched: 1e-3 per-rate also NO SIGN COST
            assert "at the matched rate 1e-3" in rd[0] and SEL_ONLY not in V["batch_reading"], (rd[0], V["batch_reading"])
        if name == "signed < 4":                       # Amendment 1 NOTE: the right reason
            assert all(r.startswith("UNREADABLE (the signed arm solves < 4/8") for r in rd), rd
        secondaries(R, rates, seeds, M, out=quiet)      # must run on synthetic records without the models
    assert {SLB, NSC, NLB, UNM, UNR} <= seen_lab, {SLB, NSC, NLB, UNM, UNR} - seen_lab
    need_rd = {"EXP PAYS THE SIGN COST IN ATTENTION ON THE TORUS", "SOFTPLUS PAYS THE SIGN COST IN ATTENTION ON THE TORUS",
               "NO SIGN COST FOR EXP IN ATTENTION ON THE TORUS", "NO SIGN COST FOR SOFTPLUS IN ATTENTION ON THE TORUS",
               "SOFTPLUS LEARNS BETTER THAN SIGNED ON THE TORUS", "UNMEASURED", "UNREADABLE"}
    assert need_rd <= seen_rd, need_rd - seen_rd
    print("selftest: every label (SLB / NSC / NLB / UNMEASURED / UNREADABLE) and every per-arm reading reached")
    # batch readings not reached above, directly
    seen_batch.add(batch_reading("SOFTPLUS PAYS...", "x").split(":")[0])
    seen_batch.add(batch_reading("EXP PAYS THE SIGN COST", "NO SIGN COST FOR SOFTPLUS").split(":")[0])
    seen_batch.add(batch_reading("UNREADABLE (x)", "SOFTPLUS PAYS THE SIGN COST").split(":")[0])
    need_b = {"BOTH PAY", "EXP ESCAPES, SOFTPLUS PAYS", "NEITHER PAYS", "SOFTPLUS ESCAPES, EXP PAYS (not predicted)",
              "MIXED / UNMEASURED", "UNREADABLE", "PARTLY UNREADABLE"}
    assert need_b <= seen_batch, need_b - seen_batch
    print("selftest: every batch reading reached: " + " | ".join(sorted(need_b)))
    # qualifiers: instability and budget
    R = _synthetic_R((8, 0), {"AO": (0, 0), "EXP": (0, 0), "SPL": (0, 0)}, rates, seeds, rng, unstable={"EXP": 3},
                     desc={"SPL": 5})
    V, _ = analyse(R, rates, seeds, out=quiet)
    assert "UNSTABLE" in V["B1"]["EXP"]["reading"] and V["B1"]["EXP"]["unstable_at_best"] == 3, V["B1"]["EXP"]
    assert "DESCENDING" in V["B1"]["SPL"]["reading"] and V["B1"]["SPL"]["desc_at_best"] == 5
    assert V["batch_reading"].startswith("MIXED"), V["batch_reading"]   # an unstable SLB is not read as 'pays'
    R = _synthetic_R((8, 0), {"AO": (0, 0), "EXP": (0, 0), "SPL": (0, 0)}, rates, seeds, rng, unstable={"EXP": 1})
    V, _ = analyse(R, rates, seeds, out=quiet); assert "UNSTABLE" not in V["B1"]["EXP"]["reading"]
    print("selftest: UNSTABLE (>= 2 unsolved at the selected rate) and DESCENDING qualifiers ok; 1 unstable run does not qualify")
    # Amendment 1.1 (audit MAJOR 1): exp 1/8 at 3e-3 with 5 DIVERGED there, 0/8 at 1e-3, softplus 0/8 -> NOT a clean 'pays'
    R = _synthetic_R((8, 0), {"AO": (0, 0), "EXP": (0, 1), "SPL": (0, 0)}, rates, seeds, rng, diverged={"EXP": 5})
    for s in seeds[1:6]:                                   # DIVERGED only at 3e-3 (the selected rate)
        R[1e-3, "EXP", s].update(cls="STALLED", acc=0.90, acc2=0.8, score=0.3)
    V, _ = analyse(R, rates, seeds, out=quiet); c = V["B1"]["EXP"]
    assert c["rate"] == 3e-3 and c["unstable_at_best"] == 5 and "DIVERGED" in c["reading"], c
    assert V["batch_reading"].startswith("MIXED"), V["batch_reading"]
    print(f"selftest: 5 DIVERGED exp runs at its selected rate -> '{c['reading'][:60]}...' and batch MIXED (Amendment 1.1)")
    # one-rate design (option A, Amendment 1.5): runs, no 'not bracketed', no matched-rate clause needed
    R = _synthetic_R((8,), {"AO": (0,), "EXP": (8,), "SPL": (1,)}, [1e-3], seeds, rng)
    V, M1 = analyse(R, [1e-3], seeds, out=quiet)
    assert V["B1"]["EXP"]["label"] == NSC and "matched rate 1e-3" in V["B1"]["EXP"]["reading"], V["B1"]["EXP"]
    assert V["B1"]["SPL"]["label"] == SLB and "bracketed" not in V["B1"]["EXP"]["reading"]
    secondaries(R, [1e-3], seeds, M1, out=quiet)
    print("selftest: one-rate (option A) analysis runs: exp 8/8 -> NO SIGN COST at the matched rate, softplus 1/8 -> SLB")
    # Holm: with exp at 3/8 and softplus at 0/8 the exp SOLVED route is adjusted x2 past 0.05 and the accuracy route decides
    R = _synthetic_R((8, 0), {"AO": (0, 0), "EXP": (3, 0), "SPL": (0, 0)}, rates, seeds, rng)
    V, _ = analyse(R, rates, seeds, out=quiet); c, c2 = V["B1"]["EXP"], V["B1"]["SPL"]
    want_k, want_a = holm([c["p_k"], c2["p_k"]]), holm([c["p_acc"], c2["p_acc"]])
    assert abs(c["p_k_holm"] - want_k[0]) < 1e-12 and abs(c2["p_acc_holm"] - want_a[1]) < 1e-12
    # step-down: softplus (0/8) has the smaller p and is doubled; exp's (larger) p is kept unless that doubled p exceeds it
    assert abs(c["p_k_holm"] - max(c["p_k"], min(1.0, 2 * c2["p_k"]))) < 1e-12, (c["p_k"], c2["p_k"], c["p_k_holm"])
    print(f"selftest: Holm, exp 3/8 vs 8/8: decision p SOLVED {c['p_k']:.4f} -> Holm {c['p_k_holm']:.4f}; acc "
          f"{c['p_acc']:.4f} -> {c['p_acc_holm']:.4f}; label {c['label']}")
    # censuses on constructed models: an exact signed code cancels through zero, an exp code with sums of 2 pi by a wrap
    torch.manual_seed(0)
    m0 = CLS["SO"](vocab_size=21, d_model=128, n_heads=2, n_layers=1, grid_size=64)
    m = CLS["SO"](vocab_size=21, d_model=128, n_heads=2, n_layers=1, grid_size=64)
    with torch.no_grad():
        e = m.token_emb.weight; e[1] = -e[0]; e[3] = -e[2]              # W_in, W_out bias-free -> exact negatives
        assert m.action_to_lie.w_in.bias is None and m.action_to_lie.w_out.bias is None
    c = cancel_census(m, m0)
    assert c["cancel"] == c["nonblind"] == c["cancel_zero"] and c["cancel_wrap"] == 0, c
    me = CLS["EXP"](vocab_size=21, d_model=128, n_heads=2, n_layers=1, grid_size=64)
    with torch.no_grad():
        me.action_to_lie.w_out.weight.zero_()
        me.path_integrator.omega.fill_(math.pi)                     # steps exp(0) = 1, omega pi: every pair sum 2 pi
    ce = cancel_census(me, me); ie = init_census(me)
    assert ce["cancel"] == ce["cancel_wrap"] == ce["nonblind"] == 128 and ce["cancel_zero"] == 0, ce
    assert ie["past_by_pair"] == [64, 64] and ie["past_both"] == 64, ie
    assert ce["cancel_from_past0"] == 128 and ce["p_cancel_given_past0"] == 1.0, ce
    i0 = init_census(init_model("EXP", 3400))
    assert i0["analytic_exp"] == 22 and i0["analytic_softplus"] == 16, i0          # 11 and 8 per head, 2 heads
    print(f"selftest: census -- exact signed code cancels 100% through zero; constructed exp code (sum 2 pi) 100% by a wrap "
          f"from a past-pi start; analytic counts on the grid-64 omega ladder: exp 11, softplus 8 channels per head (22, 16 per model)")
    # RUNAWAY rule (imported) still behaves as registered in G-4
    L = np.full(100, 0.5); L[50] = 2.2
    assert runaway(L, 1.0, 5, 2.06) and not runaway(np.full(100, 0.5), 1.0, 5, 2.06) and runaway(np.full(100, 0.5), 2e3, 5, 2.06)
    print("selftest passed")


def power(reps=400):
    """(a) Deterministic: label of the exp contrast for exp's best count k (signed 8/8 at 1e-3, 0/8 at 3e-3), with softplus
    at 0/8 or 8/8 (Holm interplay). (b) Simulated binomial seeds, unsolved accuracy ~ N(0.88, 0.07) clipped."""
    n = 8; seeds = list(range(n)); rates = list(DEFAULT_RATES); rng = np.random.default_rng(1)
    print("(a) deterministic: signed (8, 0) by rate; exp (k, k-1); |.| (0, 0)")
    for spl in ((0, 0), (8, 8)):
        for k in range(0, 9):
            R = _synthetic_R((8, 0), {"AO": (0, 0), "EXP": (k, max(k - 1, 0)), "SPL": spl}, rates, seeds, rng)
            V, _ = analyse(R, rates, seeds, out=lambda *_: None); c = V["B1"]["EXP"]
            print(f"  softplus {spl}, exp {k}/8: decision p SOLVED {c['p_k']:.4f} (Holm {c['p_k_holm']:.4f}), acc "
                  f"{c['p_acc']:.4f} (Holm {c['p_acc_holm']:.4f}) -> {c['label']}")
    print(f"(b) simulated ({reps} replicates): share of exp labels; signed true rates (1.0, 0.1); softplus (0.1, 0.1)")
    for p in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        cnt = {}
        for _ in range(reps):
            R = {}
            for j, lr in enumerate(rates):
                for arm, pr in (("SO", (1.0, 0.1)[j]), ("AO", 0.05), ("EXP", p), ("SPL", 0.1)):
                    for s in seeds:
                        sv = bool(rng.random() < pr)
                        R[lr, arm, s] = {"solved": sv, "solved10": sv, "score": rng.uniform(0.002, 0.03) if sv else
                                         rng.uniform(0.06, 0.4), "acc": 1.0 if sv else float(np.clip(rng.normal(0.88, 0.07),
                                                                                                       0.5, 0.995)),
                                         "cls": "SOLVED" if sv else "STALLED", "runaway": False}
            V, _ = analyse(R, rates, seeds, out=lambda *_: None); lab = V["B1"]["EXP"]["label"]
            key = UNR if lab.startswith(UNR) else lab; cnt[key] = cnt.get(key, 0) + 1
        print(f"  exp true solve rate {p:.1f}: " + ", ".join(f"{k} {v / reps:.2f}" for k, v in sorted(cnt.items())))


# ------------------------------------------------------------------------------ init census (no runs) and pilot
def init_census_table(seeds):
    out = {}
    for arm in ARMS:
        rows = [init_census(init_model(arm, s)) for s in seeds]
        out[arm] = rows
        print(f"{NAME[arm]:8s}: past pi per pair N/S " + " ".join(str(r["past_by_pair"][0]) for r in rows) + " | W/E "
              + " ".join(str(r["past_by_pair"][1]) for r in rows) + " | both " + " ".join(str(r["past_both"]) for r in rows)
              + f" | median |sum| {np.mean([r['median_abs_sum'] for r in rows]):.2f} rad, max "
              f"{np.max([r['max_abs_sum'] for r in rows]):.2f} (of {rows[0]['n_channels']} channels; analytic exp "
              f"{rows[0]['analytic_exp']}, softplus {rows[0]['analytic_softplus']})")
    return out


EPOCH_RE = re.compile(r"Epoch\s+(\d+)/(\d+)\s+\|\s+Loss:\s+([0-9.eE+-]+|nan|inf)\s+\|.*\|\s+([0-9.]+)s")


def epoch_seconds(log):
    """Median seconds per epoch from a train_variant log (lines 'Epoch k/E | Loss | LR | t s', every 5th epoch)."""
    t = [float(m.group(4)) for m in EPOCH_RE.finditer(open(log).read())]
    return float(np.median(t)) if t else float("nan")


def pilot(root, seeds, rates, epochs, n_batches, batch_runs, slots):
    """Pilot (seeds outside the batch, never pooled): exp at each batch rate, the driver's command line. Training-side only:
    class, UNSTABLE flag (RUNAWAY / COLLAPSED), final-5% loss, move-step max, s/epoch at the pilot's concurrency, and the
    projected batch wall time = ceil(batch_runs / slots) waves x epochs x s/epoch."""
    secs = []
    for lr in rates:
        for s in seeds:
            d = f"{root}/{rate_tag(lr)}/EXP/p0/{VAR['EXP']}_s{s}"
            b = check_ckpt(f"{d}/{VAR['EXP']}.pt", "EXP", s, lr, 0.05, epochs, n_batches)
            L = np.asarray(b["losses"], float); k = classify_run(L); div = not np.all(np.isfinite(L))
            m = CLS["EXP"](vocab_size=b["config"]["vocab_size"], d_model=128, n_heads=2, n_layers=1, grid_size=64)
            m.load_state_dict(b["model_state_dict"])
            with torch.no_grad():
                step = float(m.step_table()[m.act].abs().max())
            h = h_map(s); uns = (not div) and runaway(L, step, max(1, int(0.05 * epochs)), h)
            log = f"{root}/{rate_tag(lr)}/EXP_s{s}.log"
            sec = epoch_seconds(log) if os.path.exists(log) else float("nan"); secs.append(sec)
            c = cancel_census(m, init_model("EXP", s)) if not div else None
            w = max(1, int(0.05 * epochs)); pw = np.where(np.isfinite(L[w:]), L[w:], -np.inf)
            print(f"pilot exp lr {lr:g} s{s}: {'DIVERGED (counts as UNSTABLE, Amendment 1.1)' if div else k['cls']}"
                  f"{' UNSTABLE' if uns else ''}, final-5% loss "
                  f"{k['tail']:.4f}, max post-warm-up loss - H_map {pw.max() - h:+.3f} at epoch {w + int(np.argmax(pw)) + 1}, "
                  f"step max {step:.2f}, {sec:.2f} s/epoch"
                  + (f"; census: cancel {c['cancel']}/{c['nonblind']} (wrap {c['cancel_wrap']}, zero {c['cancel_zero']}), "
                     f"from past-pi start {c['cancel_from_past0']}" if c else ""))
    occ = f"{root}/launch_occupancy.txt"           # Amendment 1.3: trainers per GPU seen at each launch
    if os.path.exists(occ):
        lines = open(occ).read().split("\n"); lines = [x for x in lines if x.strip()]
        print("launch occupancy (trainers per GPU / other users' GPU processes, at each launch):")
        for x in lines:
            print("  " + x)
        foreign = [x for x in lines if "foreign_trainers=0" not in x or "other_users=0" not in x]
        print("timing readable for the 12 h rule: " + ("NO -- a launch shared the GPUs with a job outside this pilot"
                                                     if foreign else "yes (only this pilot's runs on the GPUs)"))
    else:
        print("launch occupancy: not recorded (timing NOT readable for the 12 h rule)")
    s_ep = float(np.nanmedian(secs))
    waves = math.ceil(batch_runs / slots)
    print(f"timing: median {s_ep:.2f} s/epoch at the pilot's concurrency -> {waves} waves x {epochs} epochs = "
          f"{waves * epochs * s_ep / 3600:.1f} h of training for {batch_runs} runs at {slots} concurrent (+ ~2 min evals)")


def parse_seeds(x):
    lo, _, hi = x.partition("-")
    return list(range(int(lo), int(hi or lo) + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=f"{REPO}/runs/exptorus")
    ap.add_argument("--eval-dir", default=None, help="default <root>/eval")
    ap.add_argument("--seeds", default="3400-3407")
    ap.add_argument("--rates", default=",".join(f"{r:g}" for r in DEFAULT_RATES))
    ap.add_argument("--epochs", type=int, default=900)
    ap.add_argument("--n-batches", type=int, default=98)
    ap.add_argument("--lengths", default="1024,2048", help="smoke only")
    ap.add_argument("--smoke", action="store_true", help="relax the recipe asserts to the CPU smoke's sizes")
    ap.add_argument("--json-out", default=f"{REPO}/EXPTORUS_VERDICTS.json")
    ap.add_argument("--no-json", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--power", action="store_true")
    ap.add_argument("--init-census", action="store_true")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--pilot-root", default=f"{REPO}/runs/exptorus_pilot")
    ap.add_argument("--pilot-seeds", default="3490-3491")
    ap.add_argument("--batch-runs", type=int, default=64)
    ap.add_argument("--slots", type=int, default=4)
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    if a.power:
        power(); return
    seeds = parse_seeds(a.seeds); rates = sorted(float(x) for x in a.rates.split(","))
    if a.init_census:
        init_census_table(seeds); return
    if a.pilot:
        pilot(a.pilot_root, parse_seeds(a.pilot_seeds), rates, a.epochs, a.n_batches, a.batch_runs, a.slots); return
    lengths = tuple(int(x) for x in a.lengths.split(","))
    assert a.smoke or lengths == LENGTHS, "--lengths is for the smoke only"
    ev_dir = a.eval_dir or f"{a.root}/eval"
    R = load(a.root, ev_dir, rates, seeds, a.epochs, a.n_batches, lengths, a.smoke)
    print(f"== EXPTORUS: torus 64 x 64, T = {lengths[0]}, {a.epochs} epochs, wd 0.05, observation-step-free rank 4; arms signed (SO), "
          f"|.| (AO), exp (EXP), softplus (SPL); rates {', '.join(f'{r:g}' for r in rates)}; seeds {seeds[0]}-{seeds[-1]} ==")
    print(f"per seed: acc T = {lengths[0]} and class (! = UNSTABLE) | " + " | ".join(f"{lr:g}:{arm}" for lr in rates for arm in ARMS))
    for s in seeds:
        print(f"{s} | " + " | ".join(f"{R[lr, arm, s]['acc']:.4f} {R[lr, arm, s]['cls'][:4]}"
                                     f"{'!' if R[lr, arm, s]['runaway'] else ''}" for lr in rates for arm in ARMS))
    print()
    V, M = analyse(R, rates, seeds)
    if not a.no_json:                                        # verdicts written before any secondary can fail
        json.dump(V, open(a.json_out, "w"), indent=2)
    W = secondaries(R, rates, seeds, M)
    if not a.no_json:
        json.dump({**V, "secondaries": W}, open(a.json_out, "w"), indent=2, default=str)


if __name__ == "__main__":
    main()
