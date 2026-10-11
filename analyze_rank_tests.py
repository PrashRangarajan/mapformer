"""Registered analysis for RANK_TESTS_PREREG.md (experiment 1: ring rank 1 vs 2; experiment 2: rank escape).
  python3 -m mapformer.analyze_rank_tests            # from /home/prashr, after the batch
  python3 -m mapformer.analyze_rank_tests --smoke    # branch tests on synthetic counts (no runs needed)
Decisions use SOLVED (final-5% training loss < 0.05, stats_core.classify_run) and two-sided Fisher tests; everything
else is a declared secondary."""
import argparse, glob, json, math, sys
import numpy as np
import torch

from mapformer.stats_core import classify_run, fisher_solved, perm2_p

REPO = "/home/prashr/mapformer"
RING = f"{REPO}/runs/rank_tests/ring/p0"
ESC = f"{REPO}/runs/rank_tests/escape"
RING_SEEDS = list(range(20, 28)); ESC_SEEDS = list(range(8)); N = 8
ESC_VARIANT = {"E": "Vanilla_r4mi", "C": "Vanilla", "S": "Vanilla_r4mi"}


# ----------------------------------------------------------------------------------------------- decisions (pure)
def e1(kR1, kR2):
    p = fisher_solved(kR2, N, kR1, N)
    if kR1 <= 2 and kR2 >= 6 and p < .05:
        return "T1 IN 1D HOLDS", p
    if kR1 >= 6:
        return "T1 IN 1D BROKEN (rank 1 = D solves)", p
    return f"T1 IN 1D MIXED ({kR1}/8 vs {kR2}/8, Fisher p {p:.4f})", p


def e2(e1_label, kR1, kDF):
    if "BROKEN" in e1_label or kR1 > 2:
        return "E2 NOT READ (rank 1 does not fail)"
    p = fisher_solved(kDF, N, kR1, N)
    if kDF >= 6 and p < .05:
        return f"DRIFT IS THE OBSTACLE (R1DF {kDF}/8 vs R1 {kR1}/8, Fisher p {p:.4f})"
    if kDF <= 2:
        return f"DRIFT IS NOT THE OBSTACLE (R1DF {kDF}/8)"
    return f"PARTIAL (R1DF {kDF}/8, Fisher p vs R1 {p:.4f})"


def x1(kE, kC):
    p = fisher_solved(kE, N, kC, N)
    if kE >= 6 and p < .05:
        return "RESCUES", p
    if kE <= 2 and not p < .05:
        return "NO RESCUE", p
    return f"PARTIAL ({kE}/8 vs {kC}/8, Fisher p {p:.4f})", p


def x2(kE, kS, epE, epS):
    """epE / epS: epochs to SOLVED per run, censored at budget + 1."""
    p = fisher_solved(kE, N, kS, N)
    ratio = math.exp(np.mean(np.log(epE)) - np.mean(np.log(epS)))
    pp = perm2_p(np.log(epE), np.log(epS))["p"]
    if (p < .05 and kE > kS) or (ratio <= 0.8 and pp < .05):
        return f"ESCAPE (faster than from scratch: E {kE}/8 vs S {kS}/8, Fisher p {p:.4f}; epochs ratio {ratio:.2f}, perm p {pp:.4f})"
    return f"RELEARN (no faster than from scratch: E {kE}/8 vs S {kS}/8, Fisher p {p:.4f}; epochs ratio {ratio:.2f}, perm p {pp:.4f})"


def x3(kE, kS):
    p = fisher_solved(kS, N, kE, N)
    if kS >= 6 and p < .05:
        return f"BASIN CHOSEN EARLY (S {kS}/8 vs E {kE}/8, Fisher p {p:.4f})"
    if kS <= 5:
        return f"UNMEASURED AT THIS BUDGET (rank 4 from scratch solves {kS}/8)"
    return f"NOT SEPARATED (S {kS}/8 vs E {kE}/8, Fisher p {p:.4f})"


def epochs_to_solved(losses, budget):
    l = np.asarray(losses, float)
    if len(l) < 10:
        return budget + 1
    rm = np.convolve(l, np.ones(10) / 10, "valid"); hit = np.nonzero(rm < 0.05)[0]
    return int(hit[0]) + 10 if len(hit) else budget + 1


# ----------------------------------------------------------------------------------------------- smoke
def smoke():
    cases = [((0, 8), "HOLDS"), ((1, 6), "HOLDS"), ((2, 6), "MIXED"), ((6, 8), "BROKEN"), ((3, 8), "MIXED"), ((0, 4), "MIXED")]
    ok = True
    for (a, b), want in cases:
        got = e1(a, b)[0]; ok &= want in got; print(f"  e1{(a, b)} -> {got}")
    for args, want in [(("T1 IN 1D HOLDS", 0, 8), "DRIFT IS THE OBSTACLE"), (("T1 IN 1D HOLDS", 0, 1), "NOT THE OBSTACLE"),
                       (("T1 IN 1D HOLDS", 1, 4), "PARTIAL"), (("T1 IN 1D BROKEN", 7, 8), "NOT READ"),
                       (("T1 IN 1D MIXED", 3, 8), "NOT READ"), (("T1 IN 1D MIXED", 2, 8), "OBSTACLE")]:
        got = e2(*args); ok &= want in got; print(f"  e2{args} -> {got}")
    for (a, b), want in [((8, 0), "RESCUES"), ((6, 1), "RESCUES"), ((1, 0), "NO RESCUE"), ((4, 1), "PARTIAL"), ((6, 3), "PARTIAL")]:
        got = x1(a, b)[0]; ok &= want in got; print(f"  x1{(a, b)} -> {got}")
    fast = [60] * 8; slow = [250] * 8; cens = [301] * 8
    for args, want in [((8, 1, fast, cens), "ESCAPE"), ((8, 8, fast, slow), "ESCAPE"), ((8, 8, slow, slow), "RELEARN"),
                       ((7, 7, [200] * 7 + [301], [210] * 7 + [301]), "RELEARN")]:
        got = x2(*args); ok &= want in got; print(f"  x2 -> {got}")
    for args, want in [((0, 8), "BASIN"), ((1, 4), "UNMEASURED"), ((2, 6), "NOT SEPARATED")]:
        got = x3(*args); ok &= want in got; print(f"  x3{args} -> {got}")
    ok &= epochs_to_solved([1.0] * 20 + [0.01] * 30, 300) == 30 and epochs_to_solved([1.0] * 50, 300) == 301
    print("SMOKE", "ALL PASS" if ok else "FAILED"); return ok


# ----------------------------------------------------------------------------------------------- batch
def load_losses(path):
    return torch.load(path, map_location="cpu", weights_only=False)["losses"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true"); a = ap.parse_args()
    if a.smoke:
        sys.exit(0 if smoke() else 1)
    out = {}
    print("== Experiment 1: per-head rank on the 1D ring (p_plus 0.5, T = 128, 300 epochs; floor 0.554) ==")
    k = {}
    for arm in ("R1", "R2", "R1DF"):
        rows = []
        for s in RING_SEEDS:
            p = f"{RING}/{arm}_s{s}"; L = load_losses(f"{p}/{arm}.pt"); ev = json.load(open(f"{p}/eval.json"))["eval"]
            assert len(L) == 300, (p, len(L))
            c = classify_run(L); rows.append((s, c["registered"], c["tail"], ev["128"]["acc"], ev["512"]["acc"], epochs_to_solved(L, 300)))
        k[arm] = sum(r[1] == "SOLVED" for r in rows); out[arm] = rows
        print(f"  {arm:5s} SOLVED {k[arm]}/8 | acc T=128 {np.mean([r[3] for r in rows]):.4f} (min {min(r[3] for r in rows):.4f}) "
              f"T=512 {np.mean([r[4] for r in rows]):.4f} | classes {[r[1][:2] for r in rows]} | epochs to SOLVED {[r[5] for r in rows]}")
    lab, p = e1(k["R1"], k["R2"])
    print(f"  REGISTERED E1: {lab} (R1 {k['R1']}/8 vs R2 {k['R2']}/8, Fisher p {p:.4f})")
    print(f"  REGISTERED E2: {e2(lab, k['R1'], k['R1DF'])}")

    print("\n== Experiment 2: rank escape (T = 1024 torus; sources: rank_mi shared rank 2, 0/8 SOLVED) ==")
    budget = None; kk = {}; ep = {}
    for arm, var in ESC_VARIANT.items():
        rows = []
        for s in ESC_SEEDS:
            L = load_losses(f"{ESC}/{arm}/p0/{var}_s{s}/{var}.pt")
            budget = budget or len(L); assert len(L) == budget, (arm, s, len(L), budget)
            c = classify_run(L); rows.append((s, c["registered"], c["tail"], epochs_to_solved(L, budget)))
        kk[arm] = sum(r[1] == "SOLVED" for r in rows); ep[arm] = [r[3] for r in rows]; out["esc_" + arm] = rows
        print(f"  {arm} ({var:12s}) SOLVED {kk[arm]}/8 | final-5% loss {[round(r[2], 3) for r in rows]} | epochs to SOLVED {ep[arm]}")
    print(f"  budget {budget} epochs")
    l1, p1 = x1(kk["E"], kk["C"])
    print(f"  REGISTERED X1: {l1} (E {kk['E']}/8 vs C {kk['C']}/8, Fisher p {p1:.4f})")
    print(f"  REGISTERED X2: {x2(kk['E'], kk['S'], ep['E'], ep['S']) if l1 == 'RESCUES' else 'NOT READ (X1 is not RESCUES)'}")
    print(f"  REGISTERED X3: {x3(kk['E'], kk['S']) if l1 == 'NO RESCUE' else 'NOT READ (X1 is not NO RESCUE)'}")
    for arm in ESC_VARIANT:
        f = f"{REPO}/RANK_TESTS_ESCAPE_{arm}.json"
        try:
            J = json.load(open(f)); v = ESC_VARIANT[arm]
            acc = [r[1] for r in J[f"0.0|{v}|1024"]]
            print(f"  secondary: {arm} held-out accuracy T=1024 (eval mode) mean {np.mean(acc):.4f} min {min(acc):.4f}")
        except (FileNotFoundError, KeyError) as e:
            print(f"  secondary: {arm} accuracy unavailable ({e})")
    json.dump(out, open(f"{REPO}/RANK_TESTS_VERDICTS.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
