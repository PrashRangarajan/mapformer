"""Registered analysis for TORUS_LRSWEEP_PREREG.md (G-4): is the sign learnability cost present on the T = 1024 torus when
each step parameterisation is compared at its OWN best learning rate? CPU.
  python3 -m mapformer.analyze_torus_lrsweep [--root runs/torus_lrsweep] [--eval-dir ROOT/eval] [--seeds 2400-2407]
                                             [--rates 1e-3,3e-3,1e-2] [--json-out PATH | --no-json]
  python3 -m mapformer.analyze_torus_lrsweep --pilot  [--pilot-root runs/torus_lrsweep_pilot]   # grid-extension rule
  python3 -m mapformer.analyze_torus_lrsweep --stage2 [--root ...]       # conditional decay-matched stage 2 (Amendment-style)
  python3 -m mapformer.analyze_torus_lrsweep --selftest                  # every label / reading / flag, synthetic, no run dirs
  python3 -m mapformer.analyze_torus_lrsweep --power                     # selection-aware power, synthetic
Layout: <root>/<tag>/<arm>/p0/<V>_s<S>/<V>.pt, tag = rate_tag(lr) (lr1e-03 ...), arm SO / AO, V = SignedAO4_r4 / AbsAO4_r4
(observation-step-free: only the four moves step; AO = |SO| per channel). Accuracy from eval_noise_refine
(<eval-dir>/TORUS_LRSWEEP_<tag>_<arm>.json, held-out map, T = 1024 and 2048, 100 walks) and strata from eval_rank_strata
(..._STRATA.json, optional: a declared secondary).

The registered machinery is LRSWEEP's, imported unchanged from analyze_lrsweep (label_from = the ROBUST cell rule; select =
most SOLVED, then lowest mean selection score, then lowest rate; b1_test = the selection-aware permutation null over all
C(16, 8) splits of the 16 per-seed rate profiles with the selection redone in each split; b1_decide = max(selection-aware p,
naive p at the selected rates) per route). What is new here: the torus loader and its recipe asserts, the torus wording of
the reading, the RUNAWAY flag (G-3 Amendment 1.4, torus form), the stage-2 trigger and reading, the pilot extension rule, and
the torus secondaries (strata, T = 2048, wrap-cancel shares, the audit-fixed clock / map ratio).
"""
import argparse, importlib.util, json, math, os

import numpy as np
import torch

from mapformer.analyze_lrsweep import label_from, select, b1_test, b1_decide, holm, sel_score, _profiles
from mapformer.analyze_robust import opp_share, cell_label as robust_cell_label, SLB, NSC, UNM, NLB
from mapformer.environment import GridWorld
from mapformer.model_wrap_exist import (MapFormerWM_SignedAO4_r4, MapFormerWM_AbsAO4_r4, action_angles, usage_weights,
                                        wrap_share, wrap_share_weighted)
from mapformer.stats_core import perm2_p, fisher_solved, classify_run

REPO = "/home/prashr/mapformer"
ARMS = ("SO", "AO")
VAR = {"SO": "SignedAO4_r4", "AO": "AbsAO4_r4"}
CLS = {"SO": MapFormerWM_SignedAO4_r4, "AO": MapFormerWM_AbsAO4_r4}
DEFAULT_RATES = (1e-3, 3e-3, 1e-2)
PAIRS = ((0, 1), (2, 3))                 # environment.GridWorld.ACTION_DELTAS: N/S, W/E
UNR = "UNREADABLE"
RUNAWAY_MARGIN = 0.05                    # Amendment 1 (MAJOR 2): post-warm-up loss above H_map(seed) + 0.05 = RUNAWAY
STEP_RUNAWAY = 1e3                       # |W_out W_in e(move)| before omega (G-3 Amendment 1.4's move_step_absmax bound)
COLLAPSE = 1.9                           # final-5% mean loss above this = the run ends (nearly) context-free (prereg Sec. 2)
RECIPE = {"n_steps": 1024, "batch_size": 16, "n_batches": 98, "d_model": 128, "n_layers": 1, "n_heads": 2,
          "grid_size": 64, "n_obs_types": 16, "n_landmarks": 0, "schedule": "cosine", "init_from": None,
          "data_seed_offset": 0, "fast_attn": True, "deterministic": False, "p_empty": 0.5}
CLOCK_TABLE = f"{REPO}/docs/paper/expansion/scripts/sign/clock_table.py"
FLOORS = ("floors (T = 1024 eval stream, post hoc, docs/audits/2026-10-20/WHAT_WHERE_T1024.md): blank 0.506, best n-gram "
          "0.575, retrace 0.843; torus always-blank 0.507 (GENERALITY)")
CROSS_BATCH = ("cross-batch reference, labelled, never pooled (rule 12): GENERALITY E3 (seeds 700-707, same recipe, lr 1e-3): "
               "AO4 0/8 0.880 vs SO4 8/8 1.000; ROBUST torus (seeds 1400-1407, 2 layers d 256, lr 1e-3): AO 2/8 0.967 vs "
               "SO 8/8 0.999; LRSWEEP ring (seeds 1500-1507): |.| best 3e-3 2/8 vs signed best 1e-3 8/8")


def rate_tag(lr):
    return f"lr{lr:.0e}"                 # identical to analyze_lrsweep.rate_tag (the driver checks its TAG table against it)


def decay_exponent(lr, wd=0.05, epochs=900, n_batches=98):
    """Total AdamW decoupled decay over the run, sum_t lr_t * wd, for train.py's cosine schedule (5% linear warm-up, cosine
    to 0.1 x lr): exp(-this) is what decay alone does to a weight that receives no gradient."""
    T = epochs * n_batches; warm = max(1, int(0.05 * T)); s = 0.0
    for t in range(T):
        f = (t + 1) / warm if t < warm else 0.1 + 0.45 * (1 + math.cos(math.pi * min((t - warm) / max(1, T - warm), 1.0)))
        s += f
    return lr * wd * s


def clock_table():
    """The audit-fixed clock / map definition (docs/paper/expansion/scripts/sign/clock_table.py), imported from its file."""
    spec = importlib.util.spec_from_file_location("mapformer_clock_table", CLOCK_TABLE)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------------------ registered reading
def reading(lab, r_ao, r_so, rates, ao_desc=0, lab_1e3=None, ao_runaway=0):
    """Overall reading, fixed in TORUS_LRSWEEP_PREREG.md. lab: B1 label; r_ao: AO's selected rate; r_so: the LOWEST rate at
    which SO reaches its best SOLVED count (LRSWEEP Amendment 1 MAJOR 1); ao_desc / ao_runaway: AO runs DESCENDING /
    UNSTABLE-and-unsolved at r_ao (Amendment 1 MAJOR 2); lab_1e3: the per-rate (naive) label at 1e-3, GENERALITY E3's rate.
    Amendment 1 MAJOR 1: every reading that rests on AO's cell at >= 1e-2 (B1 SLB, NSC or NLB) triggers stage 2, and until
    stage 2 reports, a closing or shrinking cost is worded 'with the rate and its coupled decay'."""
    q = ""
    if r_ao == max(rates):
        q += " [AO's best rate is the grid's top rate: the AO optimum is not bracketed; 'best' = best TESTED]"
    if ao_desc:
        q += f" [{ao_desc} AO run(s) still DESCENDING at AO's best rate: budget-scoped]"
    if ao_runaway >= 2:
        q += (f" [{ao_runaway} unsolved AO runs UNSTABLE (RUNAWAY / COLLAPSED) at AO's best rate: optimiser instability at "
              f"that rate; no learnability sentence rests on this rate]")
    if lab.startswith(UNR):
        return "UNREADABLE (the signed arm solves < 4/8 at every rate)" + q
    st2 = " [STAGE 2 TRIGGERED: decay-matched control at AO's rate]" if stage2_trigger(lab, r_ao) else ""
    if lab == SLB:
        return "COST AT BEST RATES" + st2 + q
    if lab in (NSC, NLB):
        rev = " (reversed: NON-NEGATIVE LEARNS BETTER)" if lab == NLB else ""
        sel = " -- cost not detectable at AO's best tested rate (selection-inflated; see the cross-fit)" if lab == NSC else ""
        if r_ao > r_so:
            return (f"COST CLOSES AT HIGHER LR (AO best {r_ao:g} > SO lowest best {r_so:g}; with the rate and its coupled "
                    f"decay){rev}{sel}" + st2 + q)
        con = (" -- contradicts GENERALITY E3 (torus, lr 1e-3, 0/8 vs 8/8)" if lab_1e3 == NSC
               else f" -- does not contradict GENERALITY E3 (1e-3 per-rate contrast reads {lab_1e3})")
        return f"NO COST AT BEST RATES (AO best {r_ao:g}, SO lowest best {r_so:g}){rev}{sel}{con}" + st2 + q
    return "UNMEASURED" + (f" (AO best {r_ao:g} > SO lowest best {r_so:g}: the cost may shrink with the rate and its coupled "
                           f"decay)" if r_ao > r_so else "") + q


def stage2_trigger(lab, r_ao):
    """Amendment 1 MAJOR 1: stage 2 whenever AO's selected rate is >= 1e-2 and B1 reads SLB, NSC or NLB."""
    return lab in (SLB, NSC, NLB) and r_ao >= 1e-2


def stage2_kind(lab, r_ao, r_so_low):
    """Which batch reading stage 2 qualifies: COST (SLB), CLOSES (NSC / NLB with AO's rate above SO's lowest best) or NOCOST."""
    return "COST" if lab == SLB else ("CLOSES" if r_ao > r_so_low else "NOCOST")


def stage2_reading(lab2, kind="COST"):
    """Final reading after the conditional decay-matched stage 2 (SO and AO at AO's selected rate, wd = 5e-5 / lr), labelled
    by the ROBUST rule with naive Fisher / permutation p. kind = stage2_kind of the batch reading."""
    unr = lab2.startswith(UNR)
    if kind == "COST":
        if lab2 == SLB:
            return "COST AT BEST RATES (stands under decay matched to the default's lr x wd)"
        if lab2 in (NSC, NLB):
            return f"COST AT BEST RATES IS COUPLED DECAY (stage 2 reads {lab2})"
        return ("COST AT BEST RATES, scoped to 'AdamW wd 0.05 coupled to lr' (stage 2 "
                + ("UNREADABLE: SO < 4/8 at matched decay)" if unr else "UNMEASURED)"))
    head = "COST CLOSES AT HIGHER LR" if kind == "CLOSES" else "NO COST AT BEST RATES"
    if lab2 in (NSC, NLB):
        return f"{head} (holds at matched decay; stage 2 reads {lab2})"
    if lab2 == SLB:
        return (f"{head.replace('AT HIGHER LR', '').replace('AT BEST RATES', '').strip()} WITH COUPLED DECAY "
                f"(lr alone does not close it; stage 2 reads SIGNED LEARNS BETTER)")
    return f"{head} with its coupled decay; lr and decay not separated (stage 2 {'UNREADABLE' if unr else 'UNMEASURED'})"


def pilot_rule(k_lo, s_lo, k_hi, s_hi, hi_unstable=0):
    """Grid extension (prereg Sec. 8, Amendment 1 MINOR 1): EXTEND iff the pilot AO solves more at 3e-2 than at 1e-2, or as
    many with a mean selection score lower by >= 0.05 and no 3e-2 pilot run UNSTABLE. Training-side only."""
    return k_hi > k_lo or (k_hi == k_lo and s_hi <= s_lo - 0.05 and hi_unstable == 0)


# ------------------------------------------------------------------------------ loading
def h_map(seed, size=64, n_obs=16, p_empty=0.5):
    """Entropy (nats) of the seed's training map (blank + 16 types): the loss of a context-free predictor on that map
    (Amendment 1 MAJOR 2; 2.051-2.102 on seeds 2400-2407). The map is GridWorld's, built exactly as train_variant builds it."""
    om = GridWorld(size=size, n_obs_types=n_obs, p_empty=p_empty, seed=seed).obs_map.numpy().ravel()
    p = np.bincount(om) / om.size; p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def runaway(L, step_absmax, warm_epochs, h):
    """UNSTABLE flag (prereg Sec. 2 + Amendment 1 MAJOR 2). RUNAWAY: any epoch after the 5% warm-up with training loss above
    h + 0.05 (h = H_map(seed), the run's own context-free loss), or a final move step (before omega) above 1e3; or COLLAPSED:
    final-5% mean loss above 1.9. Non-finite losses are DIVERGED, not UNSTABLE."""
    post = L[warm_epochs:]
    post = post[np.isfinite(post)]
    k5 = max(1, round(0.05 * len(L))); tail = L[-k5:]
    collapsed = bool(np.isfinite(tail).all() and tail.mean() > COLLAPSE)
    return bool((post > h + RUNAWAY_MARGIN).any() or (np.isfinite(step_absmax) and step_absmax > STEP_RUNAWAY) or collapsed)


def check_ckpt(ck, arm, s, lr, wd, epochs, n_batches):
    """Load a checkpoint and assert the registered recipe (used by the batch, stage 2 and, Amendment 1 MINOR 5, the pilot)."""
    b = torch.load(ck, map_location="cpu", weights_only=False); c = b["config"]; A = c.get("args", {})
    want = {**RECIPE, "n_batches": n_batches, "epochs": epochs}
    bad = {k: (c.get(k), v) for k, v in want.items() if c.get(k) != v}
    for k, v in (("data_workers", 3), ("variant", VAR[arm]), ("seed", s), ("env", "torus"), ("p_transition_noise", 0.0),
                 ("p_action_noise", 0.0), ("action_mode", "translate"), ("snapshot_every", 0)):
        if A.get(k) != v:
            bad[f"args.{k}"] = (A.get(k), v)
    if abs(c.get("lr", -1) - lr) > 1e-9 * lr or abs(A.get("weight_decay", -1) - wd) > 1e-6 * max(wd, 1e-12):
        bad["lr/wd"] = ((c.get("lr"), A.get("weight_decay")), (lr, wd))
    assert not bad and b["variant"] == VAR[arm] and b["seed"] == s and len(b["losses"]) == epochs, (ck, bad, len(b["losses"]))
    return b


def run_record(ck, arm, s, lr, wd, epochs, n_batches, ev, st, tok, ct, want_model=True, h=None):
    b = check_ckpt(ck, arm, s, lr, wd, epochs, n_batches); c = b["config"]
    h = h_map(s) if h is None else h
    L = np.asarray(b["losses"], float); div = not np.all(np.isfinite(L))
    k = classify_run(L); k10 = classify_run(L, solved_loss=0.1)
    fin = lambda x: float(x) if x is not None and np.isfinite(x) else 0.0   # non-finite accuracy counts 0 (declared)
    acc = {T: dict((x[0], x[1]) for x in ev[f"0.0|{VAR[arm]}|{T}"]) for T in (1024, 2048)}
    # a DIVERGED run scores accuracy 0 whatever the evaluator printed for a non-finite model (declared)
    rec = {"acc": 0.0 if div else fin(acc[1024][s]), "acc2": 0.0 if div else fin(acc[2048][s]),
           "solved": k["registered"] == "SOLVED" and not div,
           "solved10": k10["registered"] == "SOLVED" and not div, "cls": "DIVERGED" if div else k["cls"],
           "score": sel_score(L), "auc": float(np.nanmean(L)) if np.isfinite(L).any() else float("nan"),
           "strata": (st or {}).get(f"{VAR[arm]}|{s}|1024")}
    m = CLS[arm](vocab_size=c["vocab_size"], d_model=128, n_heads=2, n_layers=1, grid_size=64)
    m.load_state_dict(b["model_state_dict"]); m.eval()
    with torch.no_grad():
        step_absmax = float(m.step_table()[m.act].abs().max())
    warm = max(1, int(0.05 * epochs))
    rec["runaway"] = (not div) and runaway(L, step_absmax, warm, h)
    rec["step_absmax"] = step_absmax; rec["h_map"] = h
    post = L[warm:][np.isfinite(L[warm:])]
    rec["maxpost_minus_h"] = float(post.max() - h) if post.size else float("nan")
    rec["tail"] = float(k["tail"])
    nanrec = dict(wrap=float("nan"), wrapw=float("nan"), opp=float("nan"), clock=float("nan"), clock_heads=[],
                  clock_pi=float("nan"))
    rec.update(nanrec)
    if want_model and not div and all(torch.isfinite(p).all() for p in m.parameters()):
        try:                         # Amendment 1 MINOR 2: a model-derived secondary can never block the verdict
            with torch.no_grad():
                rec["wrap"] = wrap_share(m, pairs=PAIRS)[1]
                rec["wrapw"] = wrap_share_weighted(m, usage_weights(m, tok, action_offset=0, n_actions=4), pairs=PAIRS)
                rec["opp"] = opp_share(m, PAIRS)
                run, _ay, heads, pis = ct.ratios(b["model_state_dict"], arm == "SO", PAIRS)
                rec["clock"], rec["clock_heads"], rec["clock_pi"] = run, heads, pis
        except Exception as e:      # noqa: BLE001 -- logged, secondaries become NaN
            print(f"WARNING: model secondaries failed for {ck}: {e!r}")
            rec.update(nanrec)
    return rec


def usage_stream():
    """fig_wrap_share.py's torus stream: GridWorld(64, 16 obs, p_empty 0.5, seed 10000), np seed 1234, 32 walks of T = 1024."""
    env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, seed=10000); np.random.seed(1234)
    return torch.stack([env.generate_trajectory(1024)[0] for _ in range(32)])


def load(root, ev_dir, rates, seeds, epochs, n_batches, wd_of=lambda lr: 0.05, prefix="TORUS_LRSWEEP", models=True):
    tok, ct = None, None
    if models:
        try:
            tok, ct = usage_stream(), clock_table()
        except Exception as e:      # noqa: BLE001 -- Amendment 1 MINOR 2
            print(f"WARNING: usage stream / clock_table unavailable ({e!r}); model secondaries will be NaN")
            models = False
    H = {s: h_map(s) for s in seeds}
    R = {}
    for lr in rates:
        for arm in ARMS:
            tag = rate_tag(lr)
            ev = json.load(open(f"{ev_dir}/{prefix}_{tag}_{arm}.json"))
            sf = f"{ev_dir}/{prefix}_{tag}_{arm}_STRATA.json"
            st = json.load(open(sf)) if os.path.exists(sf) else None
            for s in seeds:
                ck = f"{root}/{tag}/{arm}/p0/{VAR[arm]}_s{s}/{VAR[arm]}.pt"
                R[lr, arm, s] = run_record(ck, arm, s, lr, wd_of(lr), epochs, n_batches, ev, st, tok, ct, models, H[s])
    return R


def arrays(R, rates, seeds, key):
    """(2n, R): rows 0..n-1 SO seeds, n..2n-1 AO seeds; columns rates ascending."""
    return np.array([[R[lr, arm, s][key] for lr in rates] for arm in ARMS for s in seeds], dtype=float)


# ------------------------------------------------------------------------------ analysis
def analyse(R, rates, seeds, out=print):
    n = len(seeds); rates = sorted(rates)
    S, SC, A = arrays(R, rates, seeds, "solved").astype(bool), arrays(R, rates, seeds, "score"), arrays(R, rates, seeds, "acc")
    t = b1_test(S, SC, A, n)
    r_ao, r_so = rates[t["r_ao"]], rates[t["r_so"]]
    lab, P = b1_decide(t, A, S, n)
    ks = S[:n].sum(0); r_so_low = rates[int(np.argmax(ks == ks.max()))]
    lab_1e3 = None
    if 1e-3 in rates:
        j = rates.index(1e-3); ka, kso = int(S[n:, j].sum()), int(S[:n, j].sum())
        lab_1e3 = label_from(ka, kso, A[n:, j].mean(), A[:n, j].mean(), fisher_solved(ka, n, kso, n),
                             perm2_p(A[:n, j], A[n:, j])["p"], n)
    ao_desc = sum(R[r_ao, "AO", s]["cls"] == "DESCENDING" for s in seeds)
    ao_run = sum(R[r_ao, "AO", s]["runaway"] and not R[r_ao, "AO", s]["solved"] for s in seeds)   # Amendment 1 MAJOR 2
    so_run = sum(R[r_so, "SO", s]["runaway"] and not R[r_so, "SO", s]["solved"] for s in seeds)
    rd = reading(lab, r_ao, r_so_low, rates, ao_desc, lab_1e3, ao_run)
    V = {"B1": {"label": lab, "AO_rate": r_ao, "SO_rate": r_so, "SO_lowest_best_rate": r_so_low, "AO_solved": t["k_ao"],
                "SO_solved": t["k_so"], "AO_acc": t["acc_ao"], "SO_acc": t["acc_so"], "d": t["acc_ao"] - t["acc_so"],
                "p_k_sel": t["p_k_sel"], "p_acc_sel": t["p_acc_sel"], "naive_fisher_p": P["naive_fisher_p"],
                "naive_perm_p": P["naive_perm_p"], "p_k": P["p_k"], "p_acc": P["p_acc"], "exact": t["exact"],
                "n_relabel": t["n_relabel"], "AO_desc_at_best": int(ao_desc), "AO_runaway_at_best": int(ao_run),
                "SO_runaway_at_best": int(so_run), "label_1e3": lab_1e3},
         "reading": rd, "stage2_triggered": bool(stage2_trigger(lab, r_ao)),
         "stage2_kind": stage2_kind(lab, r_ao, r_so_low) if stage2_trigger(lab, r_ao) else None,
         "stage2_wd": (5e-5 / r_ao) if stage2_trigger(lab, r_ao) else None}
    out("== B1 (registered primary): each arm at its own best rate (most SOLVED, then lowest mean final-5% loss, then lower "
        "rate) ==")
    out(f"AO best rate {r_ao:g}: SOLVED {t['k_ao']}/{n}, acc {t['acc_ao']:.4f} | SO best rate {r_so:g} (lowest rate with SO's "
        f"best count {r_so_low:g}): SOLVED {t['k_so']}/{n}, acc {t['acc_so']:.4f} | d = AO - SO "
        f"{t['acc_ao'] - t['acc_so']:+.4f}")
    out(f"selection-aware p ({'exact, ' if t['exact'] else 'MC, '}{t['n_relabel']} relabellings of {2 * n} rate profiles, "
        f"selection redone in each): SOLVED {t['p_k_sel']:.4f}, accuracy {t['p_acc_sel']:.4f}; naive at the selected rates: "
        f"Fisher {P['naive_fisher_p']:.4f}, perm {P['naive_perm_p']:.4f}; DECISION p = max: SOLVED {P['p_k']:.4f}, accuracy "
        f"{P['p_acc']:.4f}")
    out(f"UNSTABLE and unsolved (RUNAWAY / COLLAPSED) at the selected rates: AO {ao_run}/{n}, SO {so_run}/{n}; DESCENDING AO "
        f"{ao_desc}/{n}")
    flagged = [(lr, arm, s) for (lr, arm, s), x in sorted(R.items()) if x["runaway"]]
    for lr, arm, s in flagged:
        x = R[lr, arm, s]
        out(f"  UNSTABLE {arm} lr {lr:g} s{s}: max post-warm-up loss - H_map {x['maxpost_minus_h']:+.3f} (H_map "
            f"{x['h_map']:.3f}), final-5% loss {x['tail']:.4f}, step max {x['step_absmax']:.2f}, {x['cls']}")
    out(f"B1: {lab}")
    out(f"Reading: {rd}")
    if V["stage2_triggered"]:
        out(f"STAGE 2 TRIGGERED (registered, Amendment 1): SO and AO at lr {r_ao:g} with wd {5e-5 / r_ao:.6g} (lr x wd = the "
            f"default's 5e-5), seeds {seeds[0]}-{seeds[-1]}, 16 runs, qualifying the {V['stage2_kind']} reading")
    return V, S, SC, A


def secondaries(R, rates, seeds, S, SC, A, out=print):
    n = len(seeds); rates = sorted(rates); W = {}
    out("\n== secondaries (declared; no verdict) ==")
    out(FLOORS)
    out("decay exponent sum(lr_t) x wd over the run (wd 0.05): " + ", ".join(f"{lr:g}: {decay_exponent(lr):.1f}"
                                                                         for lr in rates))
    out("per rate, per arm: SOLVED | SOLVED@0.1 | acc 1024 | acc 2048 | loss AUC | mean sel. score | wrap-cancel | usage-wtd "
        "wrap-cancel | opposite-sign | clock/map (run, median) | UNSTABLE | classes")
    W["cells"] = {}
    for lr in rates:
        for arm in ARMS:
            r = [R[lr, arm, s] for s in seeds]
            cell = {"solved": sum(x["solved"] for x in r), "solved10": sum(x["solved10"] for x in r),
                    "acc": float(np.mean([x["acc"] for x in r])), "acc2": float(np.mean([x["acc2"] for x in r])),
                    "auc": float(np.nanmean([x["auc"] for x in r])), "score": float(np.mean([x["score"] for x in r])),
                    "wrap": float(np.nanmean([x["wrap"] for x in r])), "wrapw": float(np.nanmean([x["wrapw"] for x in r])),
                    "opp": float(np.nanmean([x["opp"] for x in r])), "clock": float(np.nanmedian([x["clock"] for x in r])),
                    "runaway": sum(x["runaway"] for x in r), "cls": [x["cls"] for x in r]}
            W["cells"][f"{lr:g}|{arm}"] = cell
            out(f"{lr:<7g} {arm}: {cell['solved']}/{n} | {cell['solved10']}/{n} | {cell['acc']:.4f} | {cell['acc2']:.4f} | "
                f"{cell['auc']:.3f} | {cell['score']:.4f} | {cell['wrap']:.2f} | {cell['wrapw']:.2f} | {cell['opp']:.2f} | "
                f"{cell['clock']:.3f} | {cell['runaway']}/{n} | {' '.join(c[:4] for c in cell['cls'])}")
    out("\nclock / map (audit-fixed definition, clock_table.py; usage-weighted, symmetric map), SOLVED vs unsolved runs:")
    for lr in rates:
        for arm in ARMS:
            r = [R[lr, arm, s] for s in seeds]
            sv = [x["clock"] for x in r if x["solved"] and np.isfinite(x["clock"])]
            us = [x["clock"] for x in r if not x["solved"] and np.isfinite(x["clock"])]
            f = lambda v: f"median {np.median(v):.3f} ({len(v)})" if v else "--"
            out(f"  {lr:<7g} {arm}: solved {f(sv)}; unsolved {f(us)}; near-pi usage share "
                f"{np.nanmean([x['clock_pi'] for x in r]):.2f}")
    out("\nrevisit strata at T = 1024 (eval_rank_strata; mean over seeds): plain lag < 128 | plain lag >= 128 | wrap-only")
    W["strata"] = {}
    for lr in rates:
        for arm in ARMS:
            rows = [R[lr, arm, s]["strata"] for s in seeds]
            if any(x is None for x in rows):
                out(f"  {lr:<7g} {arm}: strata unavailable"); continue
            v = {k: float(np.nanmean([x[k]["acc"] if x[k]["acc"] is not None else np.nan for x in rows]))
                 for k in ("plain_lag<128", "plain_lag>=128", "wrap")}
            W["strata"][f"{lr:g}|{arm}"] = v
            out(f"  {lr:<7g} {arm}: {v['plain_lag<128']:.3f} | {v['plain_lag>=128']:.3f} | {v['wrap']:.3f}")
    out(f"\nper-rate AO vs SO (Holm over the {len(rates)} rates within each route; labels by the ROBUST rule on Holm p)")
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
        note = " (fresh-seed replication of GENERALITY E3)" if lr == 1e-3 else ""
        out(f"  lr {lr:<7g}: AO {ka}/{n} {aa:.4f} vs SO {ks}/{n} {sa:.4f}; d {aa - sa:+.4f}; Fisher p {pf[j]:.4f} "
            f"(Holm {hf[j]:.4f}); perm p {pp[j]:.4f} (Holm {hp[j]:.4f}) -> {lab}{note}")
    S10 = arrays(R, rates, seeds, "solved10").astype(bool)
    t10 = b1_test(S10, SC, A, n); l10, P10 = b1_decide(t10, A, S10, n)
    W["B1_loss0.1"] = {"label": l10, **{k: t10[k] for k in ("k_ao", "k_so", "p_k_sel", "p_acc_sel")}, **P10,
                       "AO_rate": rates[t10["r_ao"]], "SO_rate": rates[t10["r_so"]]}
    out(f"\nthreshold sensitivity, B1 with SOLVED at loss 0.1 (selection redone): AO {rates[t10['r_ao']]:g} "
        f"{t10['k_ao']}/{n} vs SO {rates[t10['r_so']]:g} {t10['k_so']}/{n}; decision p SOLVED {P10['p_k']:.4f}, "
        f"acc {P10['p_acc']:.4f} -> {l10}")
    A2 = arrays(R, rates, seeds, "acc2"); t2 = b1_test(S, SC, A2, n)
    W["acc2048_sel"] = {"AO": t2["acc_ao"], "SO": t2["acc_so"], "p_sel": t2["p_acc_sel"]}
    out(f"accuracy at T = 2048 (2x the training length) at the B1-selected rates: AO {t2['acc_ao']:.4f} vs SO "
        f"{t2['acc_so']:.4f}; d {t2['acc_ao'] - t2['acc_so']:+.4f}; selection-aware p {t2['p_acc_sel']:.4f}")
    h = n // 2; W["crossfit"] = []
    for sh, th in ((list(range(0, h)), list(range(h, n))), (list(range(h, n)), list(range(0, h)))):
        ra, rs = int(select(S, SC, (np.array(sh) + n)[None])[0]), int(select(S, SC, np.array(sh)[None])[0])
        ka, ks = int(S[np.array(th) + n, ra].sum()), int(S[np.array(th), rs].sum())
        aa, sa = A[np.array(th) + n, ra], A[np.array(th), rs]
        f, p = fisher_solved(ka, len(th), ks, len(th)), perm2_p(sa, aa)["p"]
        W["crossfit"].append({"select_on": [seeds[i] for i in sh], "AO_rate": rates[ra], "SO_rate": rates[rs],
                              "AO_solved": ka, "SO_solved": ks, "d": float(aa.mean() - sa.mean()), "fisher_p": f, "perm_p": p})
        out(f"cross-fit: rates chosen on seeds {seeds[sh[0]]}-{seeds[sh[-1]]} (AO {rates[ra]:g}, SO {rates[rs]:g}), tested on "
            f"{seeds[th[0]]}-{seeds[th[-1]]}: AO {ka}/{len(th)} {aa.mean():.4f} vs SO {ks}/{len(th)} {sa.mean():.4f}; "
            f"Fisher p {f:.4f}, perm p {p:.4f}")
    out("SOLVED-vs-lr (descriptive): " + "; ".join(
        f"{arm} " + " ".join(f"{lr:g}:{int(S[(n if arm == 'AO' else 0):(2 * n if arm == 'AO' else n), j].sum())}"
                             for j, lr in enumerate(rates)) for arm in ARMS))
    out(CROSS_BATCH)
    return W


def analyse_stage2(R, lr, seeds, kind="COST", out=print):
    """Stage 2: SO vs AO at AO's selected rate, wd = 5e-5 / lr; the ROBUST rule with naive Fisher / permutation p."""
    n = len(seeds)
    ao = [R[lr, "AO", s]["acc"] for s in seeds]; so = [R[lr, "SO", s]["acc"] for s in seeds]
    ka = sum(R[lr, "AO", s]["solved"] for s in seeds); ks = sum(R[lr, "SO", s]["solved"] for s in seeds)
    lab, st = robust_cell_label(ao, so, ka, ks, n)
    rd = stage2_reading(lab, kind)
    out(f"STAGE 2 (decay-matched, lr {lr:g}, wd {5e-5 / lr:.6g}): AO {ka}/{n} {np.mean(ao):.4f} vs SO {ks}/{n} "
        f"{np.mean(so):.4f}; d {st['d']:+.4f}; Fisher p {st['fisher_p']:.4f}, perm p {st['perm_p']:.4f} -> {lab}")
    out(f"Final reading: {rd}")
    return {"label": lab, "AO_solved": ka, "SO_solved": ks, **st, "reading": rd}


# ------------------------------------------------------------------------------ selftest and power
def selftest():
    n = 8; rates = list(DEFAULT_RATES); rng = np.random.default_rng(0)
    hi, lo, mid = [1.0] * n, [0.9] * n, [0.97, 0.99, 1.0, 0.95, 1.0, 0.98, 0.96, 1.0]
    for ao, so, ka, ks in ((lo, hi, 1, 8), (hi, hi, 8, 8), (mid, hi, 5, 8), (lo, lo, 0, 3), (hi, lo, 8, 4), (hi, hi, 4, 4)):
        a, b = np.array(ao), np.array(so)
        mine = label_from(ka, ks, a.mean(), b.mean(), fisher_solved(ka, n, ks, n), perm2_p(b, a)["p"], n)
        assert mine == robust_cell_label(ao, so, ka, ks, n)[0], (mine, ao, so, ka, ks)
    print("selftest: label_from == analyze_robust.cell_label on the ROBUST selftest cases")
    cases = {   # name: (SO SOLVED by rate, AO SOLVED by rate[, acc_lo, SO solved acc]); rates 1e-3, 3e-3, 1e-2
        "COST AT BEST RATES": ((8, 8, 5), (0, 2, 1)),
        "COST AT BEST RATES, AO at edge": ((8, 8, 4), (0, 1, 2)),
        "COST CLOSES AT HIGHER LR (NSC)": ((8, 8, 6), (0, 4, 8)),
        "COST CLOSES AT HIGHER LR (NLB)": ((4, 4, 2), (0, 2, 8), 0.75, 0.99),
        "NO COST AT BEST RATES": ((8, 6, 2), (8, 7, 5)),
        "UNMEASURED": ((8, 8, 8), (1, 4, 5)),
        "UNREADABLE": ((3, 2, 0), (0, 1, 0)),
    }
    seen = set()
    for name, (so, ao, *lo_) in cases.items():
        S, SC, A = _profiles(so, ao, R=3, acc_lo=lo_[0] if lo_ else 0.90, rng=rng)
        if len(lo_) > 1:
            A[:n][S[:n]] = lo_[1]
        t = b1_test(S, SC, A, n); lab, P = b1_decide(t, A, S, n)
        ks = S[:n].sum(0); r_low = rates[int(np.argmax(ks == ks.max()))]
        rd = reading(lab, rates[t["r_ao"]], r_low, rates)
        print(f"selftest {name:32s}: AO {rates[t['r_ao']]:g} {t['k_ao']}/8 vs SO {rates[t['r_so']]:g} {t['k_so']}/8; p_sel "
              f"{t['p_k_sel']:.4f} / {t['p_acc_sel']:.4f}; decision p {P['p_k']:.4f} / {P['p_acc']:.4f}; {lab} -> {rd}")
        seen.add(lab if not lab.startswith(UNR) else UNR); seen.add(rd.split(" (")[0].split(" [")[0])
        if name == "COST AT BEST RATES, AO at edge":
            assert stage2_trigger(lab, rates[t["r_ao"]]) and "STAGE 2 TRIGGERED" in rd and "not bracketed" in rd, rd
        if name == "COST AT BEST RATES":
            assert not stage2_trigger(lab, rates[t["r_ao"]]) and "STAGE 2" not in rd, rd
        if name.startswith("COST CLOSES"):                 # Amendment 1 MAJOR 1: AO's cell at 1e-2 -> stage 2 also here
            assert stage2_trigger(lab, rates[t["r_ao"]]) and "STAGE 2 TRIGGERED" in rd and "coupled decay" in rd, rd
            assert stage2_kind(lab, rates[t["r_ao"]], r_low) == "CLOSES"
        if name in ("UNMEASURED", "UNREADABLE", "NO COST AT BEST RATES"):
            assert not stage2_trigger(lab, rates[t["r_ao"]]) and "STAGE 2" not in rd, (name, rd)
    need = {SLB, NSC, NLB, UNM, UNR, "COST AT BEST RATES", "COST CLOSES AT HIGHER LR", "NO COST AT BEST RATES", "UNMEASURED",
            "UNREADABLE"}
    assert need <= seen, need - seen
    # qualifiers
    assert "RUNAWAY" in reading(SLB, 3e-3, 1e-3, rates, ao_runaway=2) and "RUNAWAY" not in reading(SLB, 3e-3, 1e-3, rates,
                                                                                                     ao_runaway=1)
    assert stage2_trigger(NSC, 1e-2) and stage2_trigger(NLB, 1e-2) and stage2_trigger(SLB, 1e-2)
    assert not stage2_trigger(UNM, 1e-2) and not stage2_trigger(NSC, 3e-3) and stage2_kind(NSC, 1e-2, 1e-2) == "NOCOST"
    assert "STAGE 2 TRIGGERED" in reading(NSC, 1e-2, 1e-2, rates, lab_1e3=SLB)
    assert "DESCENDING" in reading(SLB, 3e-3, 1e-3, rates, ao_desc=3)
    assert "contradicts GENERALITY" in reading(NSC, 1e-3, 1e-3, rates, lab_1e3=NSC)
    assert "does not contradict" in reading(NSC, 1e-3, 1e-3, rates, lab_1e3=SLB)
    # stage-2 readings, through the ROBUST rule on synthetic accuracy vectors
    s2 = {SLB: ([0.9] * 8, [1.0] * 8, 1, 8), NSC: ([1.0] * 8, [1.0] * 8, 8, 8), UNM: (mid, [1.0] * 8, 5, 8),
          "UNREADABLE": ([0.9] * 8, [0.9] * 8, 0, 3)}
    got = {}
    for key, (ao, so, ka, ks) in s2.items():
        lab2 = robust_cell_label(ao, so, ka, ks, n)[0]; got[key] = stage2_reading(lab2)
        assert lab2.startswith(key), (key, lab2)
    assert got[SLB].startswith("COST AT BEST RATES (stands") and "COUPLED DECAY" in got[NSC]
    assert "UNMEASURED" in got[UNM] and "UNREADABLE" in got["UNREADABLE"]
    assert "COUPLED DECAY" in stage2_reading(NLB)
    # closing branch (Amendment 1 MAJOR 1)
    assert stage2_reading(NSC, "CLOSES") .startswith("COST CLOSES AT HIGHER LR (holds at matched decay")
    assert stage2_reading(NLB, "CLOSES").startswith("COST CLOSES AT HIGHER LR (holds at matched decay")
    assert stage2_reading(SLB, "CLOSES").startswith("COST CLOSES WITH COUPLED DECAY (lr alone does not close it")
    assert "not separated" in stage2_reading(UNM, "CLOSES") and "not separated" in stage2_reading("UNREADABLE (x)", "CLOSES")
    assert stage2_reading(SLB, "NOCOST").startswith("NO COST WITH COUPLED DECAY")
    for k_ in ("COST", "CLOSES", "NOCOST"):
        print(f"selftest stage 2 [{k_}]: " + " | ".join(stage2_reading(l_, k_) for l_ in (SLB, NSC, UNM, "UNREADABLE (x)")))
    print("selftest: stage-2 readings reached: stands / COUPLED DECAY (NSC, NLB) / scoped (UNMEASURED, UNREADABLE)")
    # pilot rule (Amendment 1 MINOR 1)
    assert pilot_rule(0, 0.5, 1, 0.9) and not pilot_rule(1, 0.5, 0, 0.1) and pilot_rule(0, 0.5, 0, 0.44) \
        and not pilot_rule(0, 0.5, 0, 0.46) and not pilot_rule(0, 0.4, 0, 0.5) and not pilot_rule(0, 0.5, 0, 0.3, 1) \
        and pilot_rule(0, 0.5, 1, 0.9, 2)
    # RUNAWAY rule
    L = np.full(100, 0.5); assert not runaway(L, 10.0, 5, 2.06)
    L2 = L.copy(); L2[3] = 3.0; assert not runaway(L2, 10.0, 5, 2.06)    # inside the warm-up: ignored
    L2[50] = 2.10; assert not runaway(L2, 10.0, 5, 2.06)                 # within H_map + 0.05 (Amendment 1 MAJOR 2)
    L2[50] = 2.12; assert runaway(L2, 10.0, 5, 2.06) and not runaway(L2, 10.0, 5, 2.10)
    assert runaway(L, 2e3, 5, 2.06)
    L3 = np.full(100, 2.03); assert runaway(L3, 10.0, 5, 2.06)              # COLLAPSED: context-free tail
    L3[-5:] = 1.0; assert not runaway(L3, 10.0, 5, 2.06)
    hm = [h_map(s) for s in range(2400, 2408)]
    assert all(abs(a_ - b_) < 5e-4 for a_, b_ in zip(hm, (2.061, 2.073, 2.102, 2.051, 2.054, 2.070, 2.094, 2.086))), hm
    # tie-break and one-rate null
    S = np.zeros((2, 3), bool); SC = np.array([[0.2, 0.1, 0.1], [0.2, 0.1, 0.1]])
    assert select(S, SC, np.array([[0, 1]]))[0] == 1
    S[:, 2] = True; assert select(S, SC, np.array([[0, 1]]))[0] == 2
    S1, SC1, A1 = _profiles((8,), (1,), R=1)
    t1 = b1_test(S1, SC1, A1, n); assert abs(t1["p_k_sel"] - fisher_solved(1, 8, 8, 8)) < 1e-12, t1["p_k_sel"]
    # the clock / map script's move angles equal the model's own angles (|.| and signed, torus action ids 0-3)
    ct = clock_table(); torch.manual_seed(0)
    for arm in ARMS:
        m = CLS[arm](vocab_size=21, d_model=128, n_heads=2, n_layers=1, grid_size=64)
        a_ct = ct.move_angles(m.state_dict(), arm == "SO", 4); a_m = action_angles(m).double().numpy()
        assert np.allclose(a_ct, a_m, atol=1e-6), (arm, np.abs(a_ct - a_m).max())
    # decay exponent: the ring value LRSWEEP quoted (~805 x lr at 300 epochs) and the torus one
    assert abs(decay_exponent(1.0, epochs=300) - 805) < 2, decay_exponent(1.0, epochs=300)
    print(f"selftest: H_map 2400-2407 = audit's values ({' '.join(f'{x:.3f}' for x in hm)})")
    print(f"selftest: one-rate null = Fisher ({t1['p_k_sel']:.4f}); tie-breaks, pilot rule, RUNAWAY rule ok; clock_table "
          f"angles == model angles; decay exponent ring {decay_exponent(1.0, epochs=300):.0f} x lr, torus "
          f"{decay_exponent(1.0):.0f} x lr")
    print("selftest passed: every B1 label, every reading, every qualifier and every stage-2 reading reached")


def power(reps=400):
    """(a) Deterministic profiles: decision label for AO's best count k (SO 8/8 at 1e-3 and 3e-3). (b) Simulated true solve
    rates (binomial seeds), SO true rates (1.0, 1.0, 0.5); AO true rate p at 3e-3 and p/2 at the other rates; accuracy of a
    solved run 1.0, of an unsolved run ~ N(0.88, 0.07) clipped to [0.5, 0.995] (GENERALITY AO4: 0.71-0.96)."""
    n = 8; rng = np.random.default_rng(1)
    print("(a) deterministic profiles, decision p = max(selection-aware, naive)")
    for so in ((8, 8, 8), (8, 8, 4), (8, 8, 0)):
        for k in range(0, 9):
            ao = (min(k, 1), k, max(k - 1, 0))
            S, SC, A = _profiles(so, ao, R=3, rng=rng); t = b1_test(S, SC, A, n); lab, P = b1_decide(t, A, S, n)
            print(f"  SO {so}, AO {ao}: SO {t['k_so']}/8 vs AO {t['k_ao']}/8 -> p_sel {t['p_k_sel']:.4f}, decision p "
                  f"{P['p_k']:.4f} -> {lab}")
    print(f"(b) simulated ({reps} replicates per row): share of B1 labels")
    for p in (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0):
        cnt = {}
        for _ in range(reps):
            tr = np.array([[1.0, 1.0, 0.5]] * n + [[p / 2, p, p / 2]] * n)
            S = rng.random((2 * n, 3)) < tr
            SC = np.where(S, rng.uniform(0.002, 0.03, S.shape), rng.uniform(0.06, 0.4, S.shape))
            A = np.where(S, 1.0, np.clip(rng.normal(0.88, 0.07, S.shape), 0.5, 0.995))
            t = b1_test(S, SC, A, n); lab, _ = b1_decide(t, A, S, n)
            key = UNR if lab.startswith(UNR) else lab; cnt[key] = cnt.get(key, 0) + 1
        print(f"  AO true best rate {p:.2f}: " + ", ".join(f"{k} {v / reps:.2f}" for k, v in sorted(cnt.items())))
    print("(c) size: both arms with true rates (0.75, 0.75, 0.75)")
    cnt = {}
    for _ in range(reps):
        S = rng.random((2 * n, 3)) < 0.75
        SC = np.where(S, rng.uniform(0.002, 0.03, S.shape), rng.uniform(0.06, 0.4, S.shape))
        A = np.where(S, 1.0, np.clip(rng.normal(0.88, 0.07, S.shape), 0.5, 0.995))
        t = b1_test(S, SC, A, n); lab, _ = b1_decide(t, A, S, n)
        key = UNR if lab.startswith(UNR) else lab; cnt[key] = cnt.get(key, 0) + 1
    print("  " + ", ".join(f"{k} {v / reps:.3f}" for k, v in sorted(cnt.items())))


# ------------------------------------------------------------------------------ pilot
def pilot(root, seeds, epochs, n_batches):
    """Grid-extension rule on the pilot (seeds 2492-2493, outside the batch, never pooled; Amendment 1 MINOR 1): AO at 1e-2 vs
    3e-2, full 900 epochs, the driver's command line; training-side only (no evaluation needed). The batch's recipe asserts
    apply (Amendment 1 MINOR 5)."""
    out = {}
    for lr, arm in ((1e-2, "AO"), (3e-2, "AO")):
        for s in seeds:
            ck = f"{root}/{rate_tag(lr)}/{arm}/p0/{VAR[arm]}_s{s}/{VAR[arm]}.pt"
            b = check_ckpt(ck, arm, s, lr, 0.05, epochs, n_batches)
            L = np.asarray(b["losses"], float); k = classify_run(L); div = not np.all(np.isfinite(L))
            with torch.no_grad():
                m = CLS[arm](vocab_size=b["config"]["vocab_size"], d_model=128, n_heads=2, n_layers=1, grid_size=64)
                m.load_state_dict(b["model_state_dict"]); step = float(m.step_table()[m.act].abs().max())
            h = h_map(s); uns = (not div) and runaway(L, step, max(1, int(0.05 * epochs)), h)
            out[lr, arm, s] = {"solved": k["registered"] == "SOLVED" and not div, "score": sel_score(L),
                               "cls": "DIVERGED" if div else k["cls"], "tail": float(k["tail"]), "unstable": uns}
            print(f"pilot {arm} lr {lr:g} s{s}: {out[lr, arm, s]['cls']}{' UNSTABLE' if uns else ''}, final-5% loss "
                  f"{k['tail']:.4f}, sel. score {out[lr, arm, s]['score']:.4f}, H_map {h:.3f}")
    k_lo = sum(out[1e-2, "AO", s]["solved"] for s in seeds); k_hi = sum(out[3e-2, "AO", s]["solved"] for s in seeds)
    s_lo = np.mean([out[1e-2, "AO", s]["score"] for s in seeds]); s_hi = np.mean([out[3e-2, "AO", s]["score"] for s in seeds])
    u_hi = sum(out[3e-2, "AO", s]["unstable"] for s in seeds)
    ext = pilot_rule(k_lo, s_lo, k_hi, s_hi, u_hi)
    print(f"AO 1e-2: {k_lo}/{len(seeds)} SOLVED, mean score {s_lo:.4f}; AO 3e-2: {k_hi}/{len(seeds)}, {s_hi:.4f}, UNSTABLE "
          f"{u_hi} -> " + ("EXTEND: add 3e-2 to the grid by amendment (64 runs)" if ext
                           else "DO NOT EXTEND: grid 1e-3 / 3e-3 / 1e-2"))
    return ext


def parse_seeds(x):
    lo, _, hi = x.partition("-")
    return list(range(int(lo), int(hi or lo) + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=f"{REPO}/runs/torus_lrsweep")
    ap.add_argument("--eval-dir", default=None, help="default <root>/eval")
    ap.add_argument("--seeds", default="2400-2407")
    ap.add_argument("--rates", default=",".join(f"{r:g}" for r in DEFAULT_RATES))
    ap.add_argument("--epochs", type=int, default=900)
    ap.add_argument("--n-batches", type=int, default=98, help="smoke only")
    ap.add_argument("--json-out", default=f"{REPO}/TORUS_LRSWEEP_VERDICTS.json")
    ap.add_argument("--no-json", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--power", action="store_true")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--pilot-root", default=f"{REPO}/runs/torus_lrsweep_pilot")
    ap.add_argument("--pilot-seeds", default="2492-2493")
    ap.add_argument("--stage2", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    if a.power:
        power(); return
    if a.pilot:
        pilot(a.pilot_root, parse_seeds(a.pilot_seeds), a.epochs, a.n_batches); return
    seeds = parse_seeds(a.seeds); ev_dir = a.eval_dir or f"{a.root}/eval"
    rates = sorted(float(x) for x in a.rates.split(","))
    if a.stage2:
        V = json.load(open(a.json_out))
        assert V["stage2_triggered"], "stage 2 is not triggered by the registered B1 result"
        lr = float(V["B1"]["AO_rate"])
        R = load(f"{a.root}/stage2", ev_dir, [lr], seeds, a.epochs, a.n_batches, wd_of=lambda x: 5e-5 / x,
                 prefix="TORUS_LRSWEEP_S2", models=False)
        W = analyse_stage2(R, lr, seeds, V.get("stage2_kind") or "COST")
        if not a.no_json:
            V["stage2"] = W; json.dump(V, open(a.json_out, "w"), indent=2)
        return
    R = load(a.root, ev_dir, rates, seeds, a.epochs, a.n_batches)
    print(f"== TORUS_LRSWEEP (G-4): torus 64 x 64, T = 1024, {a.epochs} epochs, wd 0.05, observation-step-free SignedAO4_r4 "
          f"(SO) vs AbsAO4_r4 (AO); rates {', '.join(f'{r:g}' for r in rates)}; seeds {seeds[0]}-{seeds[-1]} ==")
    print("per seed: acc T = 1024 and class (! = UNSTABLE) | " + " | ".join(f"{lr:g}:{arm}" for lr in rates for arm in ARMS))
    for s in seeds:
        print(f"{s} | " + " | ".join(f"{R[lr, arm, s]['acc']:.4f} {R[lr, arm, s]['cls'][:4]}"
                                     f"{'!' if R[lr, arm, s]['runaway'] else ''}" for lr in rates for arm in ARMS))
    print()
    V, S, SC, A = analyse(R, rates, seeds)
    if not a.no_json:                                        # verdicts written before any secondary can fail
        json.dump(V, open(a.json_out, "w"), indent=2)
    W = secondaries(R, rates, seeds, S, SC, A)
    if not a.no_json:
        json.dump({**V, "secondaries": W}, open(a.json_out, "w"), indent=2)


if __name__ == "__main__":
    main()
