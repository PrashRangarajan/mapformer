"""Registered analysis for RANKRECIPE_PREREG.md: does shared rank 2 still fail the T = 1024 torus at its OWN best recipe?

  python3 -m mapformer.analyze_rankrecipe               # reads runs/rankrecipe/<cell>/p0 + RANKRECIPE_<cell>.json
  python3 -m mapformer.analyze_rankrecipe --selftest    # every label from planted checkpoints on disk

Cells (n = 8 fresh seeds 2600-2607, the RANKWD recipe otherwise, 900 epochs, shared rank, 1 layer, 2 heads, d 128, cosine):
  r2_R0  Vanilla (shared rank 2), lr 1e-3, wd 0.05   -- the default recipe; the in-batch anchor (rule 12)
  r2_R1  Vanilla, lr = RECIPES["R1"] (3e-3, or 2e-3 if the pilot rule fired), wd 0.05
  r2_R2  Vanilla, lr 1e-3, wd 0.2
  r4_R0  Vanilla_r4mi (shared rank 4), lr 1e-3, wd 0.05
Shared machinery (rule 5): analyze_lrsweep.select / b1_test / b1_decide / sel_score (the selection-aware best-vs-best null,
selection on SOLVED then final-5% training loss -- never on held-out accuracy), analyze_torus_lrsweep.h_map / runaway (G-4's
registered UNSTABLE clause), stats_core.classify_run, analyze_rankclock's eval reader and recipe.

G0  r4_R0 SOLVED >= 6/8, else UNREADABLE.
G1  r2_R0 SOLVED <= 2/8, else RANK-2 DIFFICULTY NOT REPLICATED (RANK_MI 0/8, RANKCLOCK 1/8).
B   rank 2 at its best recipe (selected per relabelling) vs rank 4 at R0, b1_test with rank 4's single recipe replicated across
    the three recipe columns (so its "selection" is the identity) -- decision p = max(selection-aware, naive at the selected
    recipe), per route (SOLVED: Fisher; accuracy at T = 1024: permutation):
      k_best <= 2, k_r4 >= 6 and min(decision p_k, p_acc) < .05 -> RANK D+1 NEEDED SURVIVES A RECIPE SEARCH
      k_best >= 6                                               -> RANK D IS HARDER AT THE DEFAULT RECIPE
      otherwise                                                 -> UNMEASURED
Also printed: per cell SOLVED, accuracy at T = 512 / 1024 / 2048, windowed classes (SOLVED / STALLED / DESCENDING / RISING),
UNSTABLE count (G-4's clause on unsolved runs), mean selection score; the recipe selected for rank 2; the gauge-invariant
effective per-move angle growth (|omega * (W_out W_in e_move)| end / init) per cell; r(log10 final loss, acc) per length.
"""
import argparse
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import torch

from mapformer import analyze_rankclock as RC
from mapformer.analyze_lrsweep import b1_decide, b1_test, select, sel_score
from mapformer.analyze_torus_lrsweep import h_map, runaway
from mapformer.stats_core import classify_run, fisher_solved

REPO = RC.REPO
ROOT = f"{REPO}/runs/rankrecipe"
SEEDS = list(range(2600, 2608))
RECIPES = {"R0": (1e-3, 0.05), "R1": (3e-3, 0.05), "R2": (1e-3, 0.2)}   # R1's lr is fixed by the pilot rule (prereg Sec. 5)
CELLS = {"r2_R0": ("Vanilla", "R0"), "r2_R1": ("Vanilla", "R1"), "r2_R2": ("Vanilla", "R2"), "r4_R0": ("Vanilla_r4mi", "R0")}
R2_CELLS = ("r2_R0", "r2_R1", "r2_R2")
WARM_EPOCHS = 45                     # 5% of 900: train()'s cosine warm-up, the window G-4's UNSTABLE clause skips
MOVES = (0, 1, 2, 3)
CODE = RC.CODE


def run_dir(root, cell, seed):
    v = CELLS[cell][0]
    return os.path.join(root, cell, "p0", f"{v}_s{seed}")


def eval_json(cell, repo=REPO):
    return os.path.join(repo, f"RANKRECIPE_{cell}.json")


@torch.no_grad()
def eff_angles(sd):
    """(4, C) effective angle each move adds per channel: omega_c (W_out W_in e_move)_c -- invariant under both step-map gauges."""
    E = sd["token_emb.weight"].double()
    steps = (E[list(MOVES)] @ sd["action_to_lie.w_in.weight"].double().T) @ sd["action_to_lie.w_out.weight"].double().T
    return sd["path_integrator.omega"].reshape(-1).double()[None] * steps, steps


def load(root=ROOT, evals=None, seeds=SEEDS):
    evals = evals or {c: eval_json(c) for c in CELLS}
    R, missing = {}, []
    hm = {s: h_map(s) for s in seeds}
    for cell, (variant, rec) in CELLS.items():
        lr, wd = RECIPES[rec]
        tab = RC._acc_table(evals[cell])
        for s in seeds:
            d = run_dir(root, cell, s)
            p = os.path.join(d, f"{variant}.pt")
            if not os.path.exists(p):
                missing.append(p)
                continue
            b = torch.load(p, map_location="cpu", weights_only=False)
            A = b["config"]["args"]
            assert (A["variant"], A["seed"]) == (variant, s), (p, A["variant"], A["seed"])
            assert abs(A["lr"] - lr) <= 1e-9 * lr and abs(A["weight_decay"] - wd) < 1e-12, (p, A["lr"], A["weight_decay"])
            assert A["epochs"] == RC.EPOCHS and len(b["losses"]) == RC.EPOCHS, (p, A["epochs"], len(b["losses"]))
            want = {k: v for k, v in RC.RECIPE.items() if k != "lr"}
            bad = {k: A.get(k) for k, w in want.items() if A.get(k) != w}
            assert not bad, (p, bad)
            L = np.asarray(b["losses"], float)
            div = not np.all(np.isfinite(L))
            k = classify_run(np.nan_to_num(L, nan=1e9, posinf=1e9))
            try:                                   # geometry is descriptive: a bad snapshot must not block the verdict
                a1, _ = eff_angles(b["model_state_dict"])
                a0, _ = eff_angles(torch.load(os.path.join(d, "snaps", "e0000.pt")))
                growth, amax = float(a1.norm() / a0.norm()), float(a1.abs().max())
                wmax = float(torch.atan2(torch.sin(a1), torch.cos(a1)).abs().max())
            except Exception:
                growth = amax = wmax = float("nan")
            solved = k["registered"] == "SOLVED" and not div
            # UNSTABLE (G-4's clause) on unsolved, non-diverged runs; its step term uses the gauge-invariant effective
            # per-move angle (the pre-omega step is gauge-dependent: Amendment 1 finding 5)
            rec_ = {"solved": solved, "cls": "DIVERGED" if div else k["cls"], "score": sel_score(L),
                    "final_loss": float(k["tail"]) if not div else float("nan"),
                    "unstable": (not solved) and (not div) and runaway(L, amax if np.isfinite(amax) else 0.0,
                                                                         WARM_EPOCHS, hm[s]),
                    "angle_growth": growth, "angle_max": amax, "angle_max_wrapped": wmax, "div": div}
            for T in RC.LENGTHS:
                assert tab is not None and (variant, T, s) in tab, f"no evaluation row for {cell} T={T} s{s}"
                acc = tab[(variant, T, s)][0]
                rec_[f"acc{T}"] = 0.0 if div or acc is None or not np.isfinite(acc) else float(acc)   # declared, as G-4
            R[(cell, s)] = rec_
    return R, missing


def matrices(R, seeds=SEEDS):
    """(2n, 3) solved / score / acc: rows 0..n-1 rank 4 (R0 replicated), rows n..2n-1 rank 2; columns R0, R1, R2."""
    rows_s, rows_c, rows_a = [], [], []
    for s in seeds:
        r = R[("r4_R0", s)]
        rows_s.append([r["solved"]] * 3); rows_c.append([r["score"]] * 3); rows_a.append([r["acc1024"]] * 3)
    for s in seeds:
        rows_s.append([R[(c, s)]["solved"] for c in R2_CELLS])
        rows_c.append([R[(c, s)]["score"] for c in R2_CELLS])
        rows_a.append([R[(c, s)]["acc1024"] for c in R2_CELLS])
    return np.array(rows_s, bool).astype(float), np.array(rows_c, float), np.array(rows_a, float)


SURV = "RANK D+1 NEEDED SURVIVES A THREE-RECIPE SEARCH"
CLOSE = "RANK-D COST CLOSES AT"                  # + " <recipe> (rank 4 not searched)"


def b_label(k_best, k_r4, p_k, p_acc, d_acc, k_r0, best_cell):
    """Amendment 1, findings 1-2. Route test as analyze_lrsweep.label_from: SOLVED (decision p_k < .05 with k_r4 > k_best) or
    accuracy (decision p_acc < .05 AND d = acc_best - acc_r4 <= -0.05). CLOSES needs the selected recipe to beat rank 2's own
    default significantly (Fisher p < .025: Bonferroni x2 for choosing among the two alternatives) and BOTH decision p-values
    against rank 4 >= .05 (no effect-size escape: the liberal side of an unsearched rank 4); it names the recipe."""
    r4_better = (p_k < .05 and k_r4 > k_best) or (p_acc < .05 and d_acc <= -0.05)
    if k_best <= 2 and k_r4 >= 6 and r4_better:
        return SURV
    p_best_r0 = fisher_solved(k_r0, 8, k_best, 8)
    if k_best >= 6 and best_cell != "r2_R0" and p_best_r0 < .025 and p_k >= .05 and p_acc >= .05:
        return f"{CLOSE} {best_cell[3:]} (rank 4 not searched)"
    return "UNMEASURED"


def analyse(R, seeds=SEEDS, out=print):
    n = len(seeds)
    k = {c: sum(R[(c, s)]["solved"] for s in seeds) for c in CELLS}
    v = {"solved": k}
    out("== RANKRECIPE: shared rank 2 at three recipes vs shared rank 4 at the default (SOLVED = final-5% loss < 0.05) ==")
    for c, (variant, rec) in CELLS.items():
        rows = [R[(c, s)] for s in seeds]
        cls = {x: sum(r["cls"] == x for r in rows) for x in ("SOLVED", "STALLED", "DESCENDING", "RISING", "DIVERGED")}
        lr, wd = RECIPES[rec]
        out(f"  {c:6s} (lr {lr:g}, wd {wd:g}) SOLVED {k[c]}/{n} | acc@512/1024/2048 "
            + " / ".join(f"{np.mean([r[f'acc{T}'] for r in rows]):.4f}" for T in RC.LENGTHS)
            + f" | classes S/T/D/R/X {cls['SOLVED']}/{cls['STALLED']}/{cls['DESCENDING']}/{cls['RISING']}/{cls['DIVERGED']}"
            + f" | UNSTABLE {sum(r['unstable'] for r in rows)} | mean sel. score {np.mean([r['score'] for r in rows]):.4f}")
    S, SC, A = matrices(R, seeds)
    t = b1_test(S, SC, A, n)
    _, P = b1_decide(t, A, S, n)
    best = R2_CELLS[t["r_ao"]]
    d_acc = t["acc_ao"] - t["acc_so"]
    lab = b_label(t["k_ao"], t["k_so"], P["p_k"], P["p_acc"], d_acc, k["r2_R0"], best)
    v.update({"best_r2_cell": best, "k_best": t["k_ao"], "k_r4": t["k_so"], "acc_best": t["acc_ao"], "acc_r4": t["acc_so"],
              "d_acc": d_acc, "p_k_sel": t["p_k_sel"], "p_acc_sel": t["p_acc_sel"], "p_k": P["p_k"], "p_acc": P["p_acc"],
              "naive_fisher_p": P["naive_fisher_p"], "naive_perm_p": P["naive_perm_p"],
              "p_best_vs_r2_R0": fisher_solved(k["r2_R0"], n, t["k_ao"], n), "exact": t["exact"],
              "n_relabel": t["n_relabel"]})
    if k["r4_R0"] < 6:
        v["reading"] = f"UNREADABLE (r4_R0 SOLVED {k['r4_R0']}/{n} < 6)"
    elif k["r2_R0"] > 2:
        v["reading"] = f"RANK-2 DIFFICULTY NOT REPLICATED (r2_R0 SOLVED {k['r2_R0']}/{n} > 2)"
    else:
        v["reading"] = lab
    gate = "" if v["reading"] == lab else "  [DESCRIPTIVE ONLY: a gate failed]"
    out(f"\nB (registered primary){gate}: rank 2 at its best recipe = {best} (selected on SOLVED, then final-5% loss): "
        f"SOLVED {t['k_ao']}/{n}, acc@1024 {t['acc_ao']:.4f} | rank 4 R0: SOLVED {t['k_so']}/{n}, acc {t['acc_so']:.4f} "
        f"(d {d_acc:+.4f})")
    out(f"   DECISION p (max of selection-aware and naive; in practice the naive p decides, Amendment 1 finding 3): "
        f"SOLVED {P['p_k']:.4f} (naive Fisher {P['naive_fisher_p']:.4f}, sel. {t['p_k_sel']:.4f}), accuracy {P['p_acc']:.4f} "
        f"(naive perm {P['naive_perm_p']:.4f}, sel. {t['p_acc_sel']:.4f}); best vs r2_R0 Fisher {v['p_best_vs_r2_R0']:.4f}")
    out(f"   rule-label: {lab}")
    out(f"\nREADING (registered): {v['reading']}")
    out("   scope: torus T = 1024, 900 epochs, shared rank, AdamW, cosine; rank 2's recipes limited to the three registered.")
    out("\n== effective per-move angle omega (W_out W_in e_move), end / init (medians; descriptive) ==")
    out("   invariant under both step-map gauges, not under a 2 pi wrap per move (rule 8): a MAGNITUDE, not the function;")
    out("   moves only -- in Vanilla observation tokens add angles too (Amendment 1 finding 6)")
    for c in CELLS:
        rows = [R[(c, s)] for s in seeds]
        out(f"  {c:6s} growth {np.nanmedian([r['angle_growth'] for r in rows]):.3f}  max angle per move "
            f"{np.nanmedian([r['angle_max'] for r in rows]):.2f} rad (wrapped {np.nanmedian([r['angle_max_wrapped'] for r in rows]):.2f})")
    out("\n== rule 2: r(log10 final loss, acc) per length ==")
    fin = [r for r in R.values() if np.isfinite(r["final_loss"])]
    Ls = np.log10(np.array([r["final_loss"] for r in fin]) + 1e-6)
    for T in RC.LENGTHS:
        A_ = np.array([r[f"acc{T}"] for r in fin])
        rr = float(np.corrcoef(Ls, A_)[0, 1]) if len(A_) > 2 and A_.std() > 0 and Ls.std() > 0 else float("nan")
        out(f"   T={T}: r = {rr:+.3f} over {len(A_)} runs")
    out("\nper seed acc@1024 (class; ! = UNSTABLE):")
    for c in CELLS:
        out(f"  {c:6s} " + " ".join(f"{R[(c, s)]['acc1024']:.3f}{CODE[R[(c, s)]['cls']]}{'!' if R[(c, s)]['unstable'] else ''}"
                                    for s in seeds))
    return v


# ------------------------------------------------------------------------------------------- selftest
def _plant(root, cell, seed, solved, final_loss=None, rec_override=None, losses=None):
    from mapformer.environment import GridWorld
    from mapformer.train_variant import VARIANT_MAP
    variant, rec = CELLS[cell]
    lr, wd = RECIPES[rec_override or rec]
    torch.manual_seed(seed)
    env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, n_landmarks=0, seed=0)
    m = VARIANT_MAP[variant](vocab_size=env.unified_vocab_size, d_model=128, n_heads=2, n_layers=1, grid_size=64)
    sd = {k: v.clone() for k, v in m.state_dict().items()}
    d = run_dir(root, cell, seed)
    os.makedirs(os.path.join(d, "snaps"), exist_ok=True)
    torch.save({k: v for k, v in sd.items() if k.startswith(("action_to_lie.", "path_integrator.", "token_emb."))},
               os.path.join(d, "snaps", "e0000.pt"))
    tail = 1e-3 if solved else (final_loss if final_loss is not None else 0.6)
    L = losses if losses is not None else list(np.linspace(2.0, 0.6, 450)) + [tail] * 450
    args = dict(RC.RECIPE, variant=variant, seed=seed, epochs=RC.EPOCHS, lr=lr, weight_decay=wd)
    torch.save({"model_state_dict": sd, "losses": L, "variant": variant, "seed": seed,
                "config": {"vocab_size": env.unified_vocab_size, "d_model": 128, "n_heads": 2, "n_layers": 1,
                           "grid_size": 64, "args": args}}, os.path.join(d, f"{variant}.pt"))


def _evals(root, spec, acc=None, drop=None):
    """acc: {cell: (acc of solved seeds, acc of unsolved seeds)}; default (1.0, 0.9). drop: a cell whose rows are omitted."""
    acc = acc or {}
    paths = {}
    for c, (variant, _) in CELLS.items():
        hi, lo = acc.get(c, (1.0, 0.9))
        p = os.path.join(root, f"RANKRECIPE_{c}.json")
        rows = {} if c == drop else {f"0.0|{variant}|{T}": [[s, hi if i < spec[c] else lo, 0.0] for i, s in enumerate(SEEDS)]
                                     for T in RC.LENGTHS}
        json.dump(rows, open(p, "w"))
        paths[c] = p
    return paths


def _case(tmp, name, spec, acc=None, **plant):
    root = os.path.join(tmp, name)
    for c in CELLS:
        for j, s in enumerate(SEEDS):
            _plant(root, c, s, j < spec[c], **plant.get(c, {}))
    R, missing = load(root, _evals(root, spec, acc), SEEDS)
    assert not missing, missing[:2]
    return R


def selftest():
    silent = lambda *a, **k: None
    ok = True

    def check(good, msg):
        nonlocal ok
        ok &= bool(good)
        print(f"  {'ok ' if good else 'BAD'} {msg}")

    tmp = tempfile.mkdtemp(prefix="rankrecipe_selftest_")
    try:
        cases = [
            ("survives via SOLVED", SURV, {"r2_R0": 0, "r2_R1": 1, "r2_R2": 2, "r4_R0": 8}, None, "r2_R2"),
            ("closes at R1", "RANK-D COST CLOSES AT R1", {"r2_R0": 1, "r2_R1": 7, "r2_R2": 2, "r4_R0": 8}, None, "r2_R1"),
            ("closes at R2", "RANK-D COST CLOSES AT R2", {"r2_R0": 0, "r2_R1": 2, "r2_R2": 6, "r4_R0": 8}, None, "r2_R2"),
            # finding 1(a): best 6/8 vs its own default 2/8 is Fisher p 0.13 -> not a closing
            ("6 vs own default 2: no closing", "UNMEASURED", {"r2_R0": 2, "r2_R1": 6, "r2_R2": 1, "r4_R0": 8}, None, "r2_R1"),
            # finding 1(b): rank 4 significantly better on accuracy blocks CLOSES even below the 0.05 effect floor
            ("closing blocked by rank-4 accuracy edge", "UNMEASURED", {"r2_R0": 0, "r2_R1": 6, "r2_R2": 1, "r4_R0": 8},
             {"r2_R1": (0.97, 0.95), "r2_R0": (0.97, 0.95), "r2_R2": (0.97, 0.95)}, "r2_R1"),
            # finding 2: the accuracy route needs d <= -0.05 -- a significant but small gap does not survive
            ("survives via accuracy (d <= -0.05)", SURV, {"r2_R0": 2, "r2_R1": 1, "r2_R2": 2, "r4_R0": 6},
             {"r2_R0": (1.0, 0.6), "r2_R1": (1.0, 0.6), "r2_R2": (1.0, 0.6)}, None),
            ("small significant acc gap: no survival", "UNMEASURED", {"r2_R0": 2, "r2_R1": 1, "r2_R2": 2, "r4_R0": 6},
             {"r2_R0": (1.0, 0.97), "r2_R1": (1.0, 0.97), "r2_R2": (1.0, 0.97), "r4_R0": (1.0, 0.995)}, None),
            ("3 vs 8 reads UNMEASURED (registered gap)", "UNMEASURED", {"r2_R0": 0, "r2_R1": 3, "r2_R2": 1, "r4_R0": 8}, None, None),
            ("gate G0", "UNREADABLE", {"r2_R0": 0, "r2_R1": 0, "r2_R2": 0, "r4_R0": 5}, None, None),
            ("gate G1", "RANK-2 DIFFICULTY NOT REPLICATED", {"r2_R0": 3, "r2_R1": 0, "r2_R2": 0, "r4_R0": 8}, None, None),
        ]
        for i, (name, want, spec, acc, want_best) in enumerate(cases):
            v = analyse(_case(tmp, f"c{i}", spec, acc), out=silent)
            check(v["reading"].startswith(want) and (want_best is None or v["best_r2_cell"] == want_best),
                  f"{name}: want {want!r} got {v['reading'][:52]!r} best {v['best_r2_cell']}")
        # B is computed (descriptively) even when a gate fails (finding 9)
        v = analyse(_case(tmp, "g", {"r2_R0": 3, "r2_R1": 0, "r2_R2": 0, "r4_R0": 8}), out=silent)
        check("p_k" in v and v["reading"].startswith("RANK-2 DIFFICULTY"), "B still reported when G1 fails")
        # selection on SOLVED then final-5% loss, never accuracy (R1 given the higher accuracy, R2 the lower loss)
        spec = {"r2_R0": 1, "r2_R1": 1, "r2_R2": 1, "r4_R0": 8}
        R = _case(tmp, "tie", spec, {"r2_R1": (1.0, 0.99)},
                  r2_R1={"final_loss": 0.5}, r2_R2={"final_loss": 0.2})
        check(analyse(R, out=silent)["best_r2_cell"] == "r2_R2", "tie on SOLVED broken by final-5% loss, not accuracy")
        # DIVERGED: never SOLVED, accuracy 0, not flagged UNSTABLE
        nanL = list(np.linspace(2.0, 0.6, 450)) + [float("nan")] * 450
        R = _case(tmp, "div", {"r2_R0": 0, "r2_R1": 1, "r2_R2": 1, "r4_R0": 8}, r2_R0={"losses": nanL})
        r = R[("r2_R0", SEEDS[0])]
        check(r["cls"] == "DIVERGED" and not r["solved"] and r["acc1024"] == 0.0 and not r["unstable"],
              f"DIVERGED run: cls {r['cls']}, acc {r['acc1024']}, unstable {r['unstable']}")
        # UNSTABLE: an unsolved run whose post-warm-up loss rises above H_map + 0.05
        spikeL = list(np.linspace(2.0, 0.6, 300)) + [3.0] * 10 + [0.6] * 590
        R = _case(tmp, "uns", {"r2_R0": 0, "r2_R1": 1, "r2_R2": 1, "r4_R0": 8}, r2_R0={"losses": spikeL})
        # (the plant arguments apply to every seed of a cell, so the plain comparison comes from another cell)
        check(R[("r2_R0", SEEDS[0])]["unstable"] and not R[("r2_R1", SEEDS[1])]["unstable"],
              "UNSTABLE fires on a post-warm-up spike above H_map + 0.05, not on a plain unsolved run")
        # a wrong weight decay in a cell is rejected
        try:
            _case(tmp, "badwd", {c: 8 for c in CELLS}, r2_R2={"rec_override": "R0"})
            check(False, "a checkpoint with the wrong weight decay was accepted")
        except AssertionError as e:
            check(True, f"wrong weight decay in a cell raises ({str(e)[:40]}...)")
        # a missing evaluation row raises
        root = os.path.join(tmp, "miss")
        spec = {c: 8 for c in CELLS}
        for c in CELLS:
            for s in SEEDS:
                _plant(root, c, s, True)
        try:
            load(root, _evals(root, spec, drop="r2_R1"), SEEDS)
            check(False, "a missing evaluation row was accepted")
        except AssertionError as e:
            check(True, f"missing evaluation row raises ({str(e)[:40]}...)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("SELFTEST", "PASS" if ok else "FAIL")
    return ok


PILOT_ROOT = f"{REPO}/runs/rankrecipe_pilot"
PILOT_SEEDS = (2692, 2693)


def pilot(root=PILOT_ROOT):
    """The registered R1 rule (prereg Sec. 5): R1 = 2e-3 if EITHER lr-3e-3 pilot run is UNSTABLE by G-4's clause (with the
    gauge-invariant step term), else 3e-3. Also reports the wd-0.2 pilots (descriptive)."""
    unstable = []
    for tag in [f"lr3e-3_s{s}" for s in PILOT_SEEDS] + [f"wd0.2_s{s}" for s in PILOT_SEEDS]:
        b = torch.load(os.path.join(root, tag, "Vanilla.pt"), map_location="cpu", weights_only=False)
        L = np.asarray(b["losses"], float)
        s_ = b["config"]["args"]["seed"]
        k = classify_run(np.nan_to_num(L, nan=1e9, posinf=1e9))
        div = not np.all(np.isfinite(L))
        a, _ = eff_angles(b["model_state_dict"])
        h = h_map(s_)
        un = (not div) and runaway(L, float(a.abs().max()), WARM_EPOCHS, h)
        print(f"pilot {tag}: {'DIVERGED' if div else k['cls']} final-5% loss {k['tail']:.4f} | UNSTABLE {un} "
              f"(H_map {h:.3f}; max post-warm-up loss {np.nanmax(L[WARM_EPOCHS:]):.3f}; max effective angle {float(a.abs().max()):.2f})")
        if tag.startswith("lr3e-3"):
            unstable.append(un or div)
    print("R1 RULE:", "UNSTABLE -> R1 = 2e-3" if any(unstable) else "stable -> R1 = 3e-3 (as registered)")
    return any(unstable)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--pilot", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if a.pilot:
        pilot()
        return
    R, missing = load()
    if missing:
        print(f"MISSING {len(missing)} checkpoints, e.g. {missing[:3]}")
        sys.exit(2)
    json.dump(analyse(R), open(f"{REPO}/RANKRECIPE_VERDICTS.json", "w"), indent=2)


if __name__ == "__main__":
    main()
