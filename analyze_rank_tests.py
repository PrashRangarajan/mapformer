"""Registered analysis for RANK_TESTS_PREREG.md as amended (Amendment 1): experiment 1, per-head rank on the 1D ring;
experiment 2, rank escape.
  python3 -m mapformer.analyze_rank_tests            # from /home/prashr, after the batch
  python3 -m mapformer.analyze_rank_tests --smoke    # branch tests on synthetic counts (no runs needed)
Decisions use SOLVED (final-5% training loss < 0.05, stats_core.classify_run) and two-sided Fisher tests; everything
else is a declared secondary."""
import argparse, json, math, sys
from math import comb
import numpy as np
import torch

from mapformer.stats_core import classify_run, fisher_solved, perm2_p

REPO = "/home/prashr/mapformer"
RING = f"{REPO}/runs/rank_tests/ring/p0"
ESC = f"{REPO}/runs/rank_tests/escape"
RING_ARMS = ("R1", "R2", "R1DF", "R1AO", "R2S")
RING_SEEDS = list(range(20, 28)); ESC_SEEDS = list(range(8)); N = 8
RING_EPOCHS = 300; ESC_EPOCHS = 600                     # Amendment 1: escape budget 600 (was 300 with a pilot rule)
ESC_VARIANT = {"E": "Vanilla_r4mi", "C": "Vanilla", "F": "Vanilla_r4mi"}


# ----------------------------------------------------------------------------------------------- decisions (pure)
def e1(kR1, kR2):
    p = fisher_solved(kR2, N, kR1, N)
    if kR1 <= 2 and kR2 >= 6 and p < .05:
        return "RANK-D DIFFICULTY IN 1D HOLDS (budget-scoped)", p
    if kR1 >= 6:
        return "RANK-D DIFFICULTY IN 1D BROKEN (rank 1 = D solves)", p
    return f"RANK-D DIFFICULTY IN 1D MIXED (R1 {kR1}/8 vs R2 {kR2}/8, Fisher p {p:.4f})", p


def e2(kR1, kR2, kAO, kDF):
    if kR1 > 2:
        return f"E2 NOT READ (rank 1 does not fail: R1 {kR1}/8)"
    if kR2 < 6:
        return f"E2 NOT READ (rank 2 does not solve: R2 {kR2}/8)"
    pAO = fisher_solved(kAO, N, kR1, N); pDF = fisher_solved(kDF, N, kAO, N)
    if kAO >= 6 and pAO < .05:
        return f"OBSERVATION STEPS ARE THE OBSTACLE (R1AO {kAO}/8 vs R1 {kR1}/8, Fisher p {pAO:.4f}; R1DF {kDF}/8 descriptive)"
    if kAO <= 2 and kDF >= 6 and pDF < .05:
        return f"THE ACTION-COMMON COMPONENT (DRIFT) IS THE OBSTACLE (R1DF {kDF}/8 vs R1AO {kAO}/8, Fisher p {pDF:.4f})"
    if kDF <= 2:
        return f"NEITHER IS THE OBSTACLE (R1DF {kDF}/8, R1AO {kAO}/8)"
    return f"PARTIAL (R1 {kR1}/8, R1AO {kAO}/8, R1DF {kDF}/8; Fisher R1AO vs R1 p {pAO:.4f}, R1DF vs R1AO p {pDF:.4f})"


def x1(kE, kC, kF):
    p = fisher_solved(kE, N, kC, N)
    if kE >= 6 and p < .05:
        return "RESCUES", p
    if kE <= 2 and not p < .05:
        if kF >= 6:
            return "NO RESCUE", p
        return f"UNMEASURED AT THIS BUDGET (E {kE}/8, and rank 4 on the trained network solves only {kF}/8: F)", p
    if p < .05 and kC > kE:
        return f"C SOLVES, E DOES NOT ({kE}/8 vs {kC}/8, Fisher p {p:.4f})", p
    return f"PARTIAL ({kE}/8 vs {kC}/8, Fisher p {p:.4f})", p


def x2(kE, kF, epE, epF):
    """epE / epF: epochs to SOLVED per run, unsolved runs censored at budget + 1."""
    p = fisher_solved(kE, N, kF, N)
    ratio = math.exp(np.mean(np.log(epE)) - np.mean(np.log(epF)))
    pp = perm2_p(np.log(epE), np.log(epF))["p"]
    tail = f"E {kE}/8 vs F {kF}/8, Fisher p {p:.4f}; epochs ratio E/F {ratio:.2f}, perm p {pp:.4f}"
    if (p < .05 and kE > kF) or (ratio <= 0.8 and pp < .05):
        return f"ESCAPE (the stalled rank-2 step map is a usable start: {tail})"
    return f"RELEARN (no better than a fresh step map on the trained network: {tail})"


def x3(kE, kF):
    p = fisher_solved(kF, N, kE, N)
    if kF >= 6 and p < .05:
        return f"BASIN COMMITTED BY THE END OF RANK-2 TRAINING (F {kF}/8 vs E {kE}/8, Fisher p {p:.4f})"
    return f"NOT SEPARATED (F {kF}/8 vs E {kE}/8, Fisher p {p:.4f})"


def mcnemar(a, b):
    """Exact two-sided McNemar on paired booleans a, b."""
    n10 = sum(x and not y for x, y in zip(a, b)); n01 = sum(y and not x for x, y in zip(a, b)); n = n10 + n01
    if n == 0:
        return n10, n01, 1.0
    k = min(n10, n01)
    return n10, n01, min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def epochs_to_solved(losses, budget):
    """First epoch whose 10-epoch running mean is < 0.05, counted ONLY for SOLVED runs; others budget + 1 (audit B1)."""
    l = np.asarray(losses, float)
    if len(l) < 10 or classify_run(l)["registered"] != "SOLVED":
        return budget + 1
    rm = np.convolve(l, np.ones(10) / 10, "valid"); hit = np.nonzero(rm < 0.05)[0]
    return int(hit[0]) + 10 if len(hit) else budget + 1


# ----------------------------------------------------------------------------------------------- smoke
def smoke():
    ok = True
    def chk(got, want, tag):
        nonlocal ok
        ok &= want in got; print(f"  {'ok ' if want in got else 'BAD'} {tag} -> {got}")
    for (a, b), want in [((0, 8), "HOLDS"), ((1, 6), "HOLDS"), ((2, 6), "MIXED"), ((6, 8), "BROKEN"), ((3, 8), "MIXED"), ((0, 4), "MIXED")]:
        chk(e1(a, b)[0], want, f"e1{(a, b)}")
    for args, want in [((0, 8, 8, 8), "OBSERVATION STEPS"), ((0, 8, 0, 8), "DRIFT) IS THE OBSTACLE"), ((0, 8, 1, 1), "NEITHER"),
                       ((0, 8, 1, 4), "PARTIAL"), ((3, 8, 8, 8), "rank 1 does not fail"), ((0, 4, 0, 8), "rank 2 does not solve"),
                       ((1, 7, 4, 8), "PARTIAL")]:
        chk(e2(*args), want, f"e2{args}")
    for args, want in [((8, 0, 8), "RESCUES"), ((6, 1, 2), "RESCUES"), ((1, 0, 8), "NO RESCUE"), ((1, 0, 3), "UNMEASURED"),
                       ((4, 1, 8), "PARTIAL"), ((0, 6, 8), "C SOLVES"), ((6, 4, 8), "PARTIAL")]:
        chk(x1(*args)[0], want, f"x1{args}")
    fast = [60] * 8; slow = [400] * 8; cens = [601] * 8
    for args, want in [((8, 1, fast, cens), "ESCAPE"), ((8, 8, fast, slow), "ESCAPE"), ((8, 8, slow, slow), "RELEARN"),
                       ((7, 7, [300] * 7 + [601], [310] * 7 + [601]), "RELEARN")]:
        chk(x2(*args), want, "x2")
    for args, want in [((0, 8), "BASIN COMMITTED"), ((2, 6), "NOT SEPARATED")]:
        chk(x3(*args), want, f"x3{args}")
    chk(str(mcnemar([1] * 6 + [0] * 2, [0] * 8)), "0.03125", "mcnemar 6 vs 0")
    ok &= epochs_to_solved([1.0] * 20 + [0.01] * 30, 300) == 30 and epochs_to_solved([1.0] * 50, 300) == 301
    dip = [1.0] * 20 + [0.01] * 15 + [0.2] * 65                      # dips below 0.05, final tail above: not SOLVED
    ok &= epochs_to_solved(dip, 100) == 101; print(f"  epochs_to_solved transient dip -> {epochs_to_solved(dip, 100)} (want 101)")
    print("SMOKE", "ALL PASS" if ok else "FAILED"); return ok


# ----------------------------------------------------------------------------------------------- batch
def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true"); a = ap.parse_args()
    if a.smoke:
        sys.exit(0 if smoke() else 1)
    out = {}
    print(f"== Experiment 1: per-head rank on the 1D ring (p_plus 0.5, T = 128, {RING_EPOCHS} epochs; floor 0.554) ==")
    k = {}
    for arm in RING_ARMS:
        rows = []
        for s in RING_SEEDS:
            p = f"{RING}/{arm}_s{s}"; b = load(f"{p}/{arm}.pt"); L = b["losses"]; ev = json.load(open(f"{p}/eval.json"))["eval"]
            ar = b["config"]["args"]                                   # provenance (audit B3)
            assert len(L) == RING_EPOCHS and ar["epochs"] == RING_EPOCHS and ar["arm"] == arm and b["seed"] == s, (p, len(L), ar)
            assert ar["p_plus"] == 0.5 and ar["n_steps"] == 128, (p, ar)
            c = classify_run(L); rows.append((s, c["registered"], c["tail"], ev["128"]["acc"], ev["512"]["acc"], epochs_to_solved(L, RING_EPOCHS)))
        k[arm] = sum(r[1] == "SOLVED" for r in rows); out[arm] = rows
        print(f"  {arm:5s} SOLVED {k[arm]}/8 (DESCENDING {sum(r[1] == 'DESCENDING' for r in rows)}) | acc T=128 "
              f"{np.mean([r[3] for r in rows]):.4f} (min {min(r[3] for r in rows):.4f}) T=512 {np.mean([r[4] for r in rows]):.4f} "
              f"| epochs to SOLVED {[r[5] for r in rows]}")
    lab, p = e1(k["R1"], k["R2"])
    print(f"  REGISTERED E1: {lab} (R1 {k['R1']}/8 vs R2 {k['R2']}/8, Fisher p {p:.4f})")
    print(f"  REGISTERED E2: {e2(k['R1'], k['R2'], k['R1AO'], k['R1DF'])}")
    print(f"  declared contrast (M1, matched total latent): R2S {k['R2S']}/8 vs R1 {k['R1']}/8, Fisher p {fisher_solved(k['R2S'], N, k['R1'], N):.4f}")

    print(f"\n== Experiment 2: rank escape (T = 1024 torus; budget {ESC_EPOCHS} epochs) ==")
    src = []
    for s in ESC_SEEDS:
        c = classify_run(load(f"{REPO}/runs/rank_mi/p0/Vanilla_s{s}/Vanilla.pt")["losses"]); src.append(c["registered"])
    print(f"  sources (rank_mi shared rank 2, 900 epochs): {src}")
    kk = {}; ep = {}; solved = {}
    for arm, var in ESC_VARIANT.items():
        rows = []
        for s in ESC_SEEDS:
            b = load(f"{ESC}/{arm}/p0/{var}_s{s}/{var}.pt"); L = b["losses"]; cf = b["config"]
            assert len(L) == ESC_EPOCHS and cf["epochs"] == ESC_EPOCHS and b["seed"] == s and b["variant"] == var, (arm, s, len(L))
            init = cf.get("init_from") or (cf.get("args") or {}).get("init_from")
            want = {"E": f"escape/lifted/p0/Vanilla_r4mi_s{s}/", "C": f"rank_mi/p0/Vanilla_s{s}/", "F": f"escape/fresh/p0/Vanilla_r4mi_s{s}/"}[arm]
            assert init and want in init, (arm, s, init)
            dso = cf.get("data_seed_offset", (cf.get("args") or {}).get("data_seed_offset"))
            assert dso == 1, (arm, s, dso)
            c = classify_run(L); rows.append((s, c["registered"], c["tail"], epochs_to_solved(L, ESC_EPOCHS)))
        kk[arm] = sum(r[1] == "SOLVED" for r in rows); ep[arm] = [r[3] for r in rows]; solved[arm] = [r[1] == "SOLVED" for r in rows]
        out["esc_" + arm] = rows
        print(f"  {arm} ({var:12s}) SOLVED {kk[arm]}/8 (DESCENDING {sum(r[1] == 'DESCENDING' for r in rows)}) | final-5% loss "
              f"{[round(r[2], 3) for r in rows]} | epochs to SOLVED {ep[arm]}")
    l1, p1 = x1(kk["E"], kk["C"], kk["F"])
    print(f"  REGISTERED X1: {l1} (E {kk['E']}/8 vs C {kk['C']}/8, Fisher p {p1:.4f}; F {kk['F']}/8)")
    print(f"  REGISTERED X2: {x2(kk['E'], kk['F'], ep['E'], ep['F']) if l1 == 'RESCUES' else 'NOT READ (X1 is not RESCUES)'}")
    print(f"  REGISTERED X3: {x3(kk['E'], kk['F']) if l1 == 'NO RESCUE' else 'NOT READ (X1 is not NO RESCUE)'}")
    n10, n01, pm = mcnemar(solved["E"], solved["C"])
    print(f"  secondary: paired McNemar E vs C (E only {n10}, C only {n01}) exact p {pm:.4f}; by source class: "
          + ", ".join(f"{cl} E {sum(solved['E'][i] for i in range(8) if src[i] == cl)}/{src.count(cl)} C "
                      f"{sum(solved['C'][i] for i in range(8) if src[i] == cl)}/{src.count(cl)}" for cl in sorted(set(src))))
    for arm in ESC_VARIANT:
        for tag in ("", "_RESCORED"):
            f = f"{REPO}/RANK_TESTS_ESCAPE_{arm}{tag}.json"
            try:
                J = json.load(open(f)); acc = [r[1] for r in J[f"0.0|{ESC_VARIANT[arm]}|1024"]]
                print(f"  secondary: {arm} held-out accuracy T=1024 {'re-scored' if tag else 'eval mode'} mean {np.mean(acc):.4f} min {min(acc):.4f}")
            except (FileNotFoundError, KeyError) as e:
                print(f"  secondary: {arm}{tag} accuracy unavailable ({e})")
    json.dump(out, open(f"{REPO}/RANK_TESTS_VERDICTS.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
