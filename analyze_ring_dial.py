"""Registered analysis for RING_DIAL_PREREG.md as amended (Amendment 1: a LEARNABILITY dial; Signed_r4 vs Abs_r4 at
p_plus 0.5 / 0.75 / 0.9, n = 8).
  python3 -m mapformer.analyze_ring_dial            # from /home/prashr, after the batch (CPU; re-scores every run)
  python3 -m mapformer.analyze_ring_dial --smoke"""
import argparse, json, sys
import numpy as np
import torch

from mapformer.stats_core import perm2_p, perm2_ci, classify_run, fisher_solved

REPO = "/home/prashr/mapformer"; R = f"{REPO}/runs/ring_dial/p0"
ARMS = ("Signed_r4", "Abs_r4"); PS = (0.5, 0.75, 0.9); SEEDS = list(range(30, 38)); EPOCHS = 300; N = 8
FLOORS = {0.5: 0.554, 0.75: 0.544, 0.9: 0.520}                    # CANCEL_RESULTS.md


def label(signed, absv, sol_s, sol_a):
    """d = mean(Abs) - mean(Signed) on accuracy; Fisher on SOLVED counts."""
    d = float(np.mean(absv) - np.mean(signed)); p = perm2_p(signed, absv)["p"]
    ci = perm2_ci(signed, absv, grid=(-0.5, 0.5))                  # audit B2: a fixed grid evaluates s = 0
    pf = fisher_solved(sum(sol_a), N, sum(sol_s), N)
    tail = (f"d {d:+.4f} (perm p {p:.4f}, 95% CI [{ci['lo']:+.4f}, {ci['hi']:+.4f}]); SOLVED Abs {sum(sol_a)}/8 vs "
            f"Signed {sum(sol_s)}/8 (Fisher p {pf:.4f})")
    acc_fires = p < .05 and abs(d) >= 0.05
    if (acc_fires and d < 0) or (pf < .05 and sum(sol_a) < sum(sol_s)):
        return "ABS LEARNS WORSE", tail
    if (acc_fires and d > 0) or (pf < .05 and sum(sol_a) > sum(sol_s)):
        return "ABS LEARNS BETTER", tail
    if ci["lo"] is not None and ci["lo"] >= -0.02 and ci["hi"] <= 0.02:
        return "ABS LEARNS AS WELL", tail
    return "UNMEASURED", tail


def smoke():
    ok = True
    def chk(got, want):
        nonlocal ok
        ok &= got[0] == want; print(f"  {'ok ' if got[0] == want else 'BAD'} {want} -> {got[0]}: {got[1]}")
    one = [1.0] * 8; T8 = [True] * 8; F8 = [False] * 8
    chk(label(one, [0.80, 0.82, 0.79, 0.81, 0.80, 0.83, 0.8, 0.81], T8, F8), "ABS LEARNS WORSE")
    chk(label(one, one, T8, T8), "ABS LEARNS AS WELL")                       # tied at ceiling: the CI must include 0
    chk(label([0.8] * 8, one, F8, T8), "ABS LEARNS BETTER")
    chk(label(one, [1, 1, 1, 1, 0.9, 0.9, 1, 1], T8, [True] * 6 + [False] * 2), "UNMEASURED")
    print("SMOKE", "ALL PASS" if ok else "FAILED"); return ok


@torch.no_grad()
def wrap_opposition(m):
    """Per channel with |step(+1)| > 0.05 rad: |wrap(step(+1) + step(-1))| (0 = the two actions cancel mod 2 pi); median, and
    the share of channels below 0.1 rad. The raw opposition ||D(+1)+D(-1)|| / mean ||D|| is not wrap-aware (audit M1)."""
    A = (m.action_to_lie(m.token_emb.weight[None])[0] * m.path_integrator.omega).reshape(m.token_emb.weight.shape[0], -1).numpy()
    sp, sm = A[0], A[1]; big = np.abs(sp) > 0.05
    w = np.abs(np.angle(np.exp(1j * (sp + sm))))[big]
    raw = float(np.linalg.norm(sp + sm) / ((np.linalg.norm(sp) + np.linalg.norm(sm)) / 2))
    return float(np.median(w)) if w.size else float("nan"), float((w < 0.1).mean()) if w.size else float("nan"), raw


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true"); a = ap.parse_args()
    if a.smoke:
        sys.exit(0 if smoke() else 1)
    from mapformer.train_variant import VARIANT_MAP
    from mapformer.environment_cancel import GridWorldCancel
    from mapformer.train_cancel import evaluate
    from mapformer import rescore_hook
    torch.set_num_threads(8)
    acc, acc4, sol, eps, models = {}, {}, {}, {}, {}
    for p in PS:
        for arm in ARMS:
            acc[p, arm], acc4[p, arm], sol[p, arm], eps[p, arm] = [], [], [], []
            for s in SEEDS:
                d = f"{R}/{arm}_p{p}_s{s}"; b = torch.load(f"{d}/{arm}.pt", map_location="cpu", weights_only=False)
                ar = b["config"]["args"]; ev = json.load(open(f"{d}/eval.json"))["eval"]
                assert ar["arm"] == arm and ar["p_plus"] == p and b["seed"] == s and len(b["losses"]) == EPOCHS, (d, ar)
                assert ar["n_steps"] == 128 and ar["size"] == 32 and ar["epochs"] == EPOCHS, (d, ar)
                acc[p, arm].append(ev["128"]["acc"]); acc4[p, arm].append(ev["512"]["acc"])
                c = classify_run(b["losses"]); sol[p, arm].append(c["registered"] == "SOLVED")
                m = VARIANT_MAP[arm](vocab_size=b["config"]["vocab_size"], d_model=128, n_heads=2, n_layers=1, grid_size=32)
                m.load_state_dict(b["model_state_dict"]); models[p, arm, s] = m
    # dropout-scale re-score (audit B1 / M4): same fresh stream, 400 trajectories, eval mode with the 1/(1-p) scale
    rescore_hook.install("auto"); acc_rs = {}
    for p in PS:
        te = GridWorldCancel(size=32, seed=10000, p_plus=p)
        for arm in ARMS:
            acc_rs[p, arm] = [evaluate(models[p, arm, s], te, 128, 400, "cpu", seed=10 ** 6)[0] for s in SEEDS]
    print("== RING_DIAL (amended): accuracy at T = 128 eval mode [re-scored] | SOLVED | acc T = 512 | wrap-aware opposition ==")
    for p in PS:
        for arm in ARMS:
            wo = [wrap_opposition(models[p, arm, s]) for s in SEEDS]
            print(f"  p_plus {p} {arm:9s} {np.mean(acc[p, arm]):.4f} (min {min(acc[p, arm]):.4f}) [{np.mean(acc_rs[p, arm]):.4f}] | "
                  f"SOLVED {sum(sol[p, arm])}/8 | {np.mean(acc4[p, arm]):.3f} | |wrap(+1 + -1)| median {np.median([x[0] for x in wo]):.3f} rad, "
                  f"share < 0.1 rad {np.median([x[1] for x in wo]):.2f}, raw opposition {np.median([x[2] for x in wo]):.2f} | floor {FLOORS[p]}")
    out = {}
    for p in PS:
        lab, tail = label(acc[p, "Signed_r4"], acc[p, "Abs_r4"], sol[p, "Signed_r4"], sol[p, "Abs_r4"])
        lab_rs, tail_rs = label(acc_rs[p, "Signed_r4"], acc_rs[p, "Abs_r4"], sol[p, "Signed_r4"], sol[p, "Abs_r4"])
        flag = f" [FLAG: re-scored reads {lab_rs}: {tail_rs}]" if lab_rs != lab else ""
        print(f"  REGISTERED p_plus {p}: {lab}: {tail}{flag}"); out[str(p)] = lab
    print("  DIAL (no new test): " + " | ".join(f"p {p}: {out[str(p)]}" for p in PS))
    json.dump({f"{p}|{arm}": {"acc": acc[p, arm], "acc_rs": acc_rs[p, arm], "acc512": acc4[p, arm], "solved": sol[p, arm]}
               for p in PS for arm in ARMS}, open(f"{REPO}/RING_DIAL_VERDICTS.json", "w"), indent=1)


if __name__ == "__main__":
    main()
