"""Registered analysis for RING_BUDGET_PREREG.md. python3 -m mapformer.analyze_ring_budget [--smoke] (from /home/prashr)."""
import argparse, json, sys
import numpy as np
import torch

from mapformer.stats_core import classify_run
from mapformer.analyze_ring_dial import wrap_opposition

REPO = "/home/prashr/mapformer"; R = f"{REPO}/runs/ring_budget/p0"; SEEDS = list(range(40, 48)); EPOCHS = 900; ARM = "Abs_r4"


def decide(classes):
    k = classes.count("SOLVED"); stalled = classes.count("STALLED")
    if k >= 6:
        return f"SLOWER, NOT STUCK ({k}/8 SOLVED)"
    if k <= 2 and stalled >= 4:
        return f"PERSISTENT HANDICAP ({k}/8 SOLVED, {stalled} STALLED)"
    if k <= 2:
        return f"STILL DESCENDING (budget-limited again: {k}/8 SOLVED, {stalled} STALLED)"
    return f"PARTIAL ({k}/8 SOLVED)"


def smoke():
    cases = [(["SOLVED"] * 7 + ["DESCENDING"], "SLOWER"), (["SOLVED"] + ["STALLED"] * 5 + ["DESCENDING"] * 2, "PERSISTENT"),
             (["SOLVED"] * 2 + ["DESCENDING"] * 5 + ["STALLED"], "STILL DESCENDING"), (["SOLVED"] * 4 + ["STALLED"] * 4, "PARTIAL")]
    ok = all(want in decide(c) for c, want in cases)
    for c, want in cases:
        print(f"  {'ok ' if want in decide(c) else 'BAD'} {decide(c)}")
    print("SMOKE", "ALL PASS" if ok else "FAILED"); return ok


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true"); a = ap.parse_args()
    if a.smoke:
        sys.exit(0 if smoke() else 1)
    from mapformer.train_variant import VARIANT_MAP
    from mapformer.environment_cancel import GridWorldCancel
    from mapformer.train_cancel import evaluate
    from mapformer.analyze_rank_tests import epochs_to_solved
    from mapformer import rescore_hook
    torch.set_num_threads(8)
    classes, acc, acc4, ep, models = [], [], [], [], []
    for s in SEEDS:
        d = f"{R}/{ARM}_p0.5_s{s}"; b = torch.load(f"{d}/{ARM}.pt", map_location="cpu", weights_only=False)
        ar = b["config"]["args"]; ev = json.load(open(f"{d}/eval.json"))["eval"]
        assert ar["arm"] == ARM and ar["p_plus"] == 0.5 and ar["epochs"] == EPOCHS and b["seed"] == s and len(b["losses"]) == EPOCHS, (d, ar)
        classes.append(classify_run(b["losses"])["registered"]); acc.append(ev["128"]["acc"]); acc4.append(ev["512"]["acc"])
        ep.append(epochs_to_solved(b["losses"], EPOCHS))
        m = VARIANT_MAP[ARM](vocab_size=b["config"]["vocab_size"], d_model=128, n_heads=2, n_layers=1, grid_size=32)
        m.load_state_dict(b["model_state_dict"]); models.append(m)
    wo = [wrap_opposition(m) for m in models]
    rescore_hook.install("auto"); te = GridWorldCancel(size=32, seed=10000, p_plus=0.5)
    acc_rs = [evaluate(m, te, 128, 400, "cpu", seed=10 ** 6)[0] for m in models]
    print(f"== RING_BUDGET: Abs_r4, ring p_plus 0.5, {EPOCHS} epochs, seeds 40-47 ==")
    for s, c, x, xr, x4, e, w in zip(SEEDS, classes, acc, acc_rs, acc4, ep, wo):
        print(f"  s{s}: {c:10s} acc T=128 {x:.4f} [re-scored {xr:.4f}] T=512 {x4:.4f} | epochs to SOLVED {e} | "
              f"|wrap(+1 + -1)| median {w[0]:.3f} rad, share cancelling {w[1]:.2f}")
    print(f"  REGISTERED: {decide(classes)}")
    print(f"  secondary: mean acc T=128 {np.mean(acc):.4f} (re-scored {np.mean(acc_rs):.4f}), T=512 {np.mean(acc4):.4f}; "
          f"share cancelling median {np.median([w[1] for w in wo]):.2f} (RING_DIAL 300 ep: Abs 0.04, Signed 0.82)")
    json.dump({"classes": classes, "acc": acc, "acc_rs": acc_rs, "acc512": acc4, "epochs_to_solved": ep, "wrap": wo},
              open(f"{REPO}/RING_BUDGET_VERDICTS.json", "w"), indent=1)


if __name__ == "__main__":
    main()
