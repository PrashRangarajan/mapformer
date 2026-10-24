"""Registered analysis for RANKDECAY_PREREG.md: which parameters must AdamW decay for shared rank 4 to learn the T = 1024 torus?

  python3 -m mapformer.analyze_rankdecay               # reads runs/rankdecay/<arm>/p0 + RANKDECAY_<arm>.json
  python3 -m mapformer.analyze_rankdecay --selftest    # every label from planted checkpoints on disk

RANKWD (registered): decay 0.05 on ALL parameters solves 8/8, no decay 1/8. Descriptively omega ended at ~0.15 of its init
with decay (0.96 without). This batch crosses decay on omega x decay on everything else, all at wd 0.05 where on:

  all      every parameter decayed (train()'s default single-group AdamW; = RANKWD's wd arm)
  none     nothing decayed (weight decay 0; = RANKWD's wd-0 arm)
  noomega  every parameter EXCEPT path_integrator.omega decayed  (--wd-select exclude --wd-names path_integrator.omega)
  omega    ONLY path_integrator.omega decayed                     (--wd-select only    --wd-names path_integrator.omega)

Shared primitives (rule 5): stats_core.classify_run (SOLVED = final-5% loss < 0.05), fisher_solved, perm2_p, signflip_p;
analyze_rankclock's eval-table reader and recipe; analyze_rankwd's norm readout.

G0  all SOLVED >= 6/8, else UNREADABLE.
G1  none SOLVED <= 2/8 and Fisher(all, none) < .05, else NO DECAY EFFECT TO EXPLAIN (RANKWD did not replicate).
P1  (is omega decay NECESSARY?) noomega vs all:
      k_noomega <= 2 and Fisher(all, noomega) < .05            -> OMEGA DECAY NECESSARY
      k_noomega >= k_all - 1 and acc_noomega >= acc_all - 0.01 -> OMEGA DECAY NOT NECESSARY
      otherwise                                                 -> PARTIAL
P2  (is omega decay SUFFICIENT?) omega vs none:
      k_omega >= 6 and Fisher(omega, none) < .05                -> OMEGA DECAY SUFFICIENT
      k_omega <= 2                                              -> OMEGA DECAY NOT SUFFICIENT
      otherwise                                                 -> PARTIAL
Overall (both P1 and P2 read when G0 and G1 pass):
      NECESSARY + SUFFICIENT          -> OMEGA CARRIES THE DECAY EFFECT
      NOT NECESSARY + NOT SUFFICIENT  -> THE DECAY EFFECT IS NOT IN OMEGA
      NECESSARY + NOT SUFFICIENT      -> OMEGA DECAY NECESSARY BUT NOT SUFFICIENT
      NOT NECESSARY + SUFFICIENT      -> EITHER ROUTE SUFFICES
      anything with a PARTIAL         -> PARTIAL
Descriptive: accuracy at T = 512 / 1024 / 2048 with unpaired permutation and paired sign-flip p beside P1 and P2; run classes;
norm growth end / init (omega, token_emb, W_in, W_out from the snapshots; LayerNorm gains from the checkpoint); rule 2.
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
from mapformer.analyze_rankwd import _NORM_NAME, _norms
from mapformer.stats_core import classify_run, fisher_solved, perm2_p, signflip_p

REPO = RC.REPO
ROOT = f"{REPO}/runs/rankdecay"
SEEDS = list(range(2600, 2608))
OMEGA = "path_integrator.omega"
ARMS = {"all": {"weight_decay": 0.05, "wd_select": None, "wd_names": []},
        "none": {"weight_decay": 0.0, "wd_select": None, "wd_names": []},
        "noomega": {"weight_decay": 0.05, "wd_select": "exclude", "wd_names": [OMEGA]},
        "omega": {"weight_decay": 0.05, "wd_select": "only", "wd_names": [OMEGA]}}
VARIANT = "Vanilla_r4mi"


def run_dir(root, arm, seed):
    return os.path.join(root, arm, "p0", f"{VARIANT}_s{seed}")


def eval_json(arm, repo=REPO):
    return os.path.join(repo, f"RANKDECAY_{arm}.json")


def load(root=ROOT, evals=None, seeds=SEEDS):
    evals = evals or {a: eval_json(a) for a in ARMS}
    R, missing = {}, []
    for arm, spec in ARMS.items():
        tab = RC._acc_table(evals[arm])
        for s in seeds:
            d = run_dir(root, arm, s)
            p = os.path.join(d, f"{VARIANT}.pt")
            if not os.path.exists(p):
                missing.append(p)
                continue
            b = torch.load(p, map_location="cpu", weights_only=False)
            A = b["config"]["args"]
            assert (A["variant"], A["seed"]) == (VARIANT, s), (p, A["variant"], A["seed"])
            assert A["epochs"] == RC.EPOCHS and len(b["losses"]) == RC.EPOCHS, (p, A["epochs"], len(b["losses"]))
            got = {"weight_decay": A.get("weight_decay"), "wd_select": A.get("wd_select"),
                   "wd_names": list(A.get("wd_names") or [])}
            assert (abs(got["weight_decay"] - spec["weight_decay"]) < 1e-12 and got["wd_select"] == spec["wd_select"]
                    and got["wd_names"] == spec["wd_names"]), (p, got, spec)
            bad = {k: A.get(k) for k, w in RC.RECIPE.items() if A.get(k) != w}
            assert not bad, (p, bad)
            L = np.asarray(b["losses"], float)
            div = not np.all(np.isfinite(L))
            k = classify_run(np.nan_to_num(L, nan=1e9, posinf=1e9))
            n0 = _norms(torch.load(os.path.join(d, "snaps", "e0000.pt")))
            n1 = _norms(torch.load(os.path.join(d, "snaps", "e0900.pt")))
            growth = {_NORM_NAME[kk]: n1[kk] / max(n0[kk], 1e-12) for kk in n0}
            gains = [v.double() for kk, v in b["model_state_dict"].items() if "norm" in kk and kk.endswith(".weight")]
            growth["ln_gains"] = float(np.median([float(g.norm() / np.sqrt(g.numel())) for g in gains]))
            rec = {"solved": k["registered"] == "SOLVED" and not div, "cls": "DIVERGED" if div else k["cls"],
                   "final_loss": float(k["tail"]) if not div else float("nan"), "growth": growth}
            for T in RC.LENGTHS:
                assert tab is not None and (VARIANT, T, s) in tab, f"no evaluation row for {arm} T={T} s{s}"
                a_ = tab[(VARIANT, T, s)][0]
                rec[f"acc{T}"] = float(a_) if a_ is not None and np.isfinite(a_) else 0.0
            R[(arm, s)] = rec
    return R, missing


def p1_label(k_all, k_no, acc_all, acc_no):
    p = fisher_solved(k_no, 8, k_all, 8)
    if k_no <= 2 and p < .05:
        return "OMEGA DECAY NECESSARY", p
    if k_no >= k_all - 1 and np.mean(acc_no) >= np.mean(acc_all) - 0.01:
        return "OMEGA DECAY NOT NECESSARY", p
    return f"PARTIAL (noomega {k_no}/8 vs all {k_all}/8, Fisher p {p:.4f})", p


def p2_label(k_om, k_none):
    p = fisher_solved(k_none, 8, k_om, 8)
    if k_om >= 6 and p < .05:
        return "OMEGA DECAY SUFFICIENT", p
    if k_om <= 2:
        return "OMEGA DECAY NOT SUFFICIENT", p
    return f"PARTIAL (omega {k_om}/8 vs none {k_none}/8, Fisher p {p:.4f})", p


def overall(p1, p2):
    table = {("OMEGA DECAY NECESSARY", "OMEGA DECAY SUFFICIENT"): "OMEGA CARRIES THE DECAY EFFECT",
             ("OMEGA DECAY NOT NECESSARY", "OMEGA DECAY NOT SUFFICIENT"): "THE DECAY EFFECT IS NOT IN OMEGA",
             ("OMEGA DECAY NECESSARY", "OMEGA DECAY NOT SUFFICIENT"): "OMEGA DECAY NECESSARY BUT NOT SUFFICIENT",
             ("OMEGA DECAY NOT NECESSARY", "OMEGA DECAY SUFFICIENT"): "EITHER ROUTE SUFFICES"}
    return table.get((p1, p2), "PARTIAL")


def analyse(R, seeds=SEEDS, out=print):
    k = {a: sum(R[(a, s)]["solved"] for s in seeds) for a in ARMS}
    acc = {a: [R[(a, s)]["acc1024"] for s in seeds] for a in ARMS}
    v = {"solved": k}
    out("== RANKDECAY: shared rank 4, decay on omega x decay on the rest (SOLVED = final-5% loss < 0.05; S/T/D/R/X) ==")
    for a in ARMS:
        out(f"  {a:7s} SOLVED {k[a]}/8 | acc@512/1024/2048 "
            + " / ".join(f"{np.mean([R[(a, s)][f'acc{T}'] for s in seeds]):.4f}" for T in RC.LENGTHS) + " | "
            + " ".join(f"s{s}:{RC.CODE[R[(a, s)]['cls']]}({R[(a, s)]['final_loss']:.3f})" for s in seeds))
    pg1 = fisher_solved(k["none"], 8, k["all"], 8)
    if k["all"] < 6:
        v["reading"] = f"UNREADABLE (all SOLVED {k['all']}/8 < 6)"
    elif not (k["none"] <= 2 and pg1 < .05):
        v["reading"] = f"NO DECAY EFFECT TO EXPLAIN (none {k['none']}/8 vs all {k['all']}/8, Fisher p {pg1:.4f})"
    else:
        p1, pp1 = p1_label(k["all"], k["noomega"], acc["all"], acc["noomega"])
        p2, pp2 = p2_label(k["omega"], k["none"])
        v["P1"], v["P2"] = p1, p2
        v["reading"] = overall(p1, p2)
        for key, lab, a, b, pf in (("P1", p1, "noomega", "all", pp1), ("P2", p2, "omega", "none", pp2)):
            pa = perm2_p(np.asarray(acc[b]), np.asarray(acc[a]))["p"]
            ps = signflip_p(np.asarray(acc[a]) - np.asarray(acc[b]))["p"]
            out(f"\n{key}: {lab} -- {a} {k[a]}/8 vs {b} {k[b]}/8, Fisher p {pf:.4f}; acc@1024 {np.mean(acc[a]):.4f} vs "
                f"{np.mean(acc[b]):.4f}, perm p {pa:.4f}, paired sign-flip p {ps:.4f}")
    out(f"\nREADING (registered): {v['reading']}")
    out("   scope: torus T = 1024, 900 epochs, shared rank 4, AdamW wd 0.05 where on, lr 1e-3 cosine; one host.")
    out("\n== norm growth end / init (medians; descriptive; ln_gains = RMS gain, init 1) ==")
    for a in ARMS:
        g = [R[(a, s)]["growth"] for s in seeds]
        out(f"  {a:7s} " + "  ".join(f"{kk} {np.median([x[kk] for x in g]):.3f}" for kk in g[0]))
    out("\n== rule 2: r(log10 final loss, acc) per length ==")
    fin = [r for r in R.values() if np.isfinite(r["final_loss"])]
    Ls = np.log10(np.array([r["final_loss"] for r in fin]) + 1e-6)
    for T in RC.LENGTHS:
        A_ = np.array([r[f"acc{T}"] for r in fin])
        rr = float(np.corrcoef(Ls, A_)[0, 1]) if len(A_) > 2 and A_.std() > 0 and Ls.std() > 0 else float("nan")
        out(f"   T={T}: r = {rr:+.3f} over {len(A_)} runs")
    out("\nper seed acc@1024 (class):")
    for a in ARMS:
        out(f"  {a:7s} " + " ".join(f"{R[(a, s)]['acc1024']:.3f}{RC.CODE[R[(a, s)]['cls']]}" for s in seeds))
    return v


# ------------------------------------------------------------------------------------------- selftest
def _plant(root, arm, seed, solved, omega_scale=1.0, override=None):
    from mapformer.environment import GridWorld
    from mapformer.train_variant import VARIANT_MAP
    torch.manual_seed(seed)
    env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, n_landmarks=0, seed=0)
    m = VARIANT_MAP[VARIANT](vocab_size=env.unified_vocab_size, d_model=128, n_heads=2, n_layers=1, grid_size=64)
    sd0 = {k: v.clone() for k, v in m.state_dict().items()}
    sd1 = dict(sd0)
    sd1[OMEGA] = sd0[OMEGA] * omega_scale
    d = run_dir(root, arm, seed)
    os.makedirs(os.path.join(d, "snaps"), exist_ok=True)
    keep = lambda sd: {k: v for k, v in sd.items() if k.startswith(("action_to_lie.", "path_integrator.", "token_emb."))}
    torch.save(keep(sd0), os.path.join(d, "snaps", "e0000.pt"))
    torch.save(keep(sd1), os.path.join(d, "snaps", "e0900.pt"))
    L = list(np.linspace(2.0, 0.6, 450)) + ([1e-3] * 450 if solved else [0.6] * 450)
    args = dict(RC.RECIPE, variant=VARIANT, seed=seed, epochs=RC.EPOCHS, **(override or ARMS[arm]))
    torch.save({"model_state_dict": sd1, "losses": L, "variant": VARIANT, "seed": seed,
                "config": {"vocab_size": env.unified_vocab_size, "d_model": 128, "n_heads": 2, "n_layers": 1,
                           "grid_size": 64, "args": args}}, os.path.join(d, f"{VARIANT}.pt"))


def _evals(tmp, spec, lo=0.9):
    paths = {}
    for a in ARMS:
        p = os.path.join(tmp, f"RANKDECAY_{a}.json")
        json.dump({f"0.0|{VARIANT}|{T}": [[s, 1.0 if i < spec[a] else lo, 0.0] for i, s in enumerate(SEEDS)]
                   for T in RC.LENGTHS}, open(p, "w"))
        paths[a] = p
    return paths


def selftest():
    silent = lambda *a, **k: None
    cases = [
        ("OMEGA CARRIES THE DECAY EFFECT", {"all": 8, "none": 1, "noomega": 1, "omega": 8}),
        ("THE DECAY EFFECT IS NOT IN OMEGA", {"all": 8, "none": 1, "noomega": 8, "omega": 1}),
        ("OMEGA DECAY NECESSARY BUT NOT SUFFICIENT", {"all": 8, "none": 1, "noomega": 1, "omega": 1}),
        ("EITHER ROUTE SUFFICES", {"all": 8, "none": 1, "noomega": 8, "omega": 8}),
        ("PARTIAL", {"all": 8, "none": 1, "noomega": 4, "omega": 8}),
        ("PARTIAL", {"all": 8, "none": 1, "noomega": 1, "omega": 4}),
        ("UNREADABLE", {"all": 5, "none": 1, "noomega": 1, "omega": 8}),
        ("NO DECAY EFFECT TO EXPLAIN", {"all": 8, "none": 5, "noomega": 1, "omega": 8}),
        ("NO DECAY EFFECT TO EXPLAIN", {"all": 6, "none": 2, "noomega": 1, "omega": 8}),   # 6 vs 2: p 0.13
    ]
    ok = True
    tmp = tempfile.mkdtemp(prefix="rankdecay_selftest_")
    try:
        for i, (want, spec) in enumerate(cases):
            root = os.path.join(tmp, f"c{i}")
            for a in ARMS:
                for j, s in enumerate(SEEDS):
                    _plant(root, a, s, j < spec[a])
            R, missing = load(root, _evals(root, spec), SEEDS)
            assert not missing, missing[:2]
            got = analyse(R, out=silent)["reading"]
            good = got.startswith(want)
            ok &= good
            print(f"  {'ok ' if good else 'BAD'} want {want!r:44s} got {got[:70]!r}  {spec}")
        # every P1 / P2 label reached (incl. the closeness clause of NOT NECESSARY with one high-accuracy unsolved seed)
        root = os.path.join(tmp, "close")
        spec = {"all": 8, "none": 1, "noomega": 7, "omega": 1}
        for a in ARMS:
            for j, s in enumerate(SEEDS):
                _plant(root, a, s, j < spec[a])
        R, _ = load(root, _evals(root, spec, lo=0.995), SEEDS)
        p1 = analyse(R, out=silent).get("P1")
        good = p1 == "OMEGA DECAY NOT NECESSARY"
        ok &= good
        print(f"  {'ok ' if good else 'BAD'} P1 closeness clause (noomega 7/8, unsolved @0.995) got {p1!r}")
        # norm growth wiring: omega scaled by 0.15 in the planted final state reads back as 0.150
        root = os.path.join(tmp, "ng")
        spec = {"all": 8, "none": 1, "noomega": 1, "omega": 8}
        for a in ARMS:
            for j, s in enumerate(SEEDS):
                _plant(root, a, s, j < spec[a], omega_scale=0.15 if a == "all" else 1.0)
        R, _ = load(root, _evals(root, spec), SEEDS)
        g = R[("all", SEEDS[0])]["growth"]["omega"]
        good = abs(g - 0.15) < 1e-6 and abs(R[("none", SEEDS[0])]["growth"]["omega"] - 1.0) < 1e-9
        ok &= good
        print(f"  {'ok ' if good else 'BAD'} norm growth: planted omega x0.15 reads {g:.4f}")
        # a checkpoint whose decay setting does not match its arm must be rejected
        root = os.path.join(tmp, "bad")
        for a in ARMS:
            for j, s in enumerate(SEEDS):
                _plant(root, a, s, True, override=ARMS["all"] if a == "omega" else None)
        try:
            load(root, _evals(root, {a: 8 for a in ARMS}), SEEDS)
            print("  BAD a checkpoint with the wrong decay setting was accepted"); ok = False
        except AssertionError as e:
            print(f"  ok  wrong decay setting in an arm's root raises ({str(e)[:50]}...)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("SELFTEST", "PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    R, missing = load()
    if missing:
        print(f"MISSING {len(missing)} checkpoints, e.g. {missing[:3]}")
        sys.exit(2)
    json.dump(analyse(R), open(f"{REPO}/RANKDECAY_VERDICTS.json", "w"), indent=2)


if __name__ == "__main__":
    main()
