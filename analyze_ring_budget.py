"""Registered analysis for RING_BUDGET_PREREG.md. python3 -m mapformer.analyze_ring_budget [--smoke] (from /home/prashr)."""
import argparse, json, sys
import numpy as np
import torch

from mapformer.stats_core import classify_run
from mapformer.analyze_ring_dial import wrap_opposition

REPO = "/home/prashr/mapformer"; R = f"{REPO}/runs/ring_budget/p0"; SEEDS = list(range(40, 48)); EPOCHS = 900; ARM = "Abs_r4"


def decide(k900, k300, share_cancel_solved):
    """Amendment 1. k900 / k300: SOLVED counts of 8 at 900 and 300 epochs (same seeds, one batch)."""
    from mapformer.stats_core import fisher_solved
    p = fisher_solved(k900, 8, k300, 8)
    if k900 >= 6:
        how = ("by the wrap solution" if share_cancel_solved is not None and share_cancel_solved >= 0.5 else "by another code")
        return f"SLOWER, NOT STUCK AT THIS BUDGET ({k900}/8 at 900 vs {k300}/8 at 300, Fisher p {p:.4f}; {how})"
    if k900 <= 2:
        return f"NOT CONVERGED BY 900 EPOCHS ({k900}/8 at 900 vs {k300}/8 at 300, Fisher p {p:.4f})"
    if p < .05 and k900 > k300:
        return f"PARTIAL GAIN FROM 3x BUDGET ({k900}/8 vs {k300}/8, Fisher p {p:.4f})"
    return f"PARTIAL ({k900}/8 vs {k300}/8, Fisher p {p:.4f})"


def smoke():
    cases = [((7, 2, 0.8), "SLOWER, NOT STUCK"), ((7, 2, 0.8), "wrap solution"), ((6, 2, 0.1), "another code"),
             ((1, 2, None), "NOT CONVERGED"), ((5, 0, 0.3), "PARTIAL GAIN"), ((4, 2, 0.3), "PARTIAL (")]
    ok = True
    for args, want in cases:
        g = decide(*args); ok &= want in g; print(f"  {'ok ' if want in g else 'BAD'} {g}")
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
    res = {}
    for ep_budget in (900, 300):
        rows = []
        for sd in SEEDS:
            d = f"{R}/{ARM}_p0.5_e{ep_budget}_s{sd}"; b = torch.load(f"{d}/{ARM}.pt", map_location="cpu", weights_only=False)
            ar = b["config"]["args"]; ev = json.load(open(f"{d}/eval.json"))["eval"]; L = b["losses"]
            assert ar["arm"] == ARM and ar["p_plus"] == 0.5 and ar["epochs"] == ep_budget and b["seed"] == sd, (d, ar)
            assert len(L) == ep_budget and ar["n_steps"] == 128 and ar["size"] == 32, (d, ar)
            c = classify_run(L)
            m = VARIANT_MAP[ARM](vocab_size=b["config"]["vocab_size"], d_model=128, n_heads=2, n_layers=1, grid_size=32)
            m.load_state_dict(b["model_state_dict"])
            rows.append(dict(seed=sd, cls=c["cls"], solved=c["registered"] == "SOLVED", tail=float(c["tail"]),
                             min_loss=float(np.min(L)), loss300=float(np.mean(L[290:300])), acc=ev["128"]["acc"],
                             acc512=ev["512"]["acc"], ep=epochs_to_solved(L, ep_budget), wrap=wrap_opposition(m), model=m))
        res[ep_budget] = rows
    rescore_hook.install("auto"); te = GridWorldCancel(size=32, seed=10000, p_plus=0.5)
    for ep_budget in (900, 300):
        for r in res[ep_budget]:
            r["acc_rs"] = evaluate(r["model"], te, 128, 400, "cpu", seed=10 ** 6)[0]
    assert rescore_hook.STATS["hooked"] >= 16, rescore_hook.STATS                  # audit minor: the hook reached every model
    print(f"== RING_BUDGET (amended): Abs_r4, ring p_plus 0.5, seeds 40-47, 900 epochs and a 300-epoch twin ==")
    for ep_budget in (900, 300):
        for r in res[ep_budget]:
            print(f"  e{ep_budget} s{r['seed']}: {r['cls']:10s} tail {r['tail']:.3f} min {r['min_loss']:.3f} loss@300 {r['loss300']:.3f} | "
                  f"acc {r['acc']:.4f} [re-scored {r['acc_rs']:.4f}] T=512 {r['acc512']:.4f} | epochs to SOLVED {r['ep']} | "
                  f"wrap median {r['wrap'][0]:.3f} rad, share cancelling {r['wrap'][1]:.2f}")
    k9 = sum(r["solved"] for r in res[900]); k3 = sum(r["solved"] for r in res[300])
    sol9 = [r["wrap"][1] for r in res[900] if r["solved"]]
    print(f"  REGISTERED: {decide(k9, k3, float(np.median(sol9)) if sol9 else None)}")
    print(f"  secondary: classes at 900 {[r['cls'] for r in res[900]]}; mean acc 900 {np.mean([r['acc'] for r in res[900]]):.4f} vs "
          f"300 {np.mean([r['acc'] for r in res[300]]):.4f}; T=512 {np.mean([r['acc512'] for r in res[900]]):.4f} vs "
          f"{np.mean([r['acc512'] for r in res[300]]):.4f}")
    json.dump({str(e): [{k: v for k, v in r.items() if k != "model"} for r in res[e]] for e in res},
              open(f"{REPO}/RING_BUDGET_VERDICTS.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
