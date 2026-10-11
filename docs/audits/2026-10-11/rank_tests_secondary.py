"""Declared secondaries for RANK_TESTS_PREREG.md (no verdicts; run by the driver after the analysis; CPU).
Ring: per run the T1 drift readout (per-head kappa, CLEAN / CLOCK) and "SOLVED iff CLEAN"; accuracy at T = 128 re-scored
for attention dropout (rescore_hook, one layer) beside eval mode; r(final loss, accuracy).
Escape: the T1 head class at the end of each run (analyze_score_rank.head_stats / basin) and "SOLVED iff CLEAN";
for E, the norm of W_out's two new columns relative to the original two (does the model use the extra rank?)."""
import json, numpy as np, torch
from mapformer.stats_core import classify_run
from mapformer.environment_cancel import GridWorldCancel
from mapformer.train_ring_rank import build, EVAL_SEED
from mapformer.train_cancel import evaluate
from mapformer.ring_rank_readouts import drift_kappa, head_class
from mapformer.analyze_score_rank import head_stats, basin
from mapformer import rescore_hook

REPO = "/home/prashr/mapformer"; RING = f"{REPO}/runs/rank_tests/ring/p0"; ESC = f"{REPO}/runs/rank_tests/escape"
torch.set_num_threads(8)
print("== ring ==")
env = GridWorldCancel(size=32, seed=0, p_plus=0.5)
rows = {}
for arm in ("R1", "R2", "R1DF"):
    for s in range(20, 28):
        b = torch.load(f"{RING}/{arm}_s{s}/{arm}.pt", map_location="cpu", weights_only=False)
        m = build(arm, env, 32); m.load_state_dict(b["model_state_dict"]); m.eval()
        k = drift_kappa(m, env.unified_vocab_size, 16, 0.5); cls = head_class(k)
        solved = classify_run(b["losses"])["registered"] == "SOLVED"
        acc = json.load(open(f"{RING}/{arm}_s{s}/eval.json"))["eval"]["128"]["acc"]
        rows[(arm, s)] = (solved, cls, k, acc, classify_run(b["losses"])["tail"])
    r = [rows[(arm, s)] for s in range(20, 28)]
    iff = sum((x[0] and x[1] == "CLEAN") or (not x[0] and x[1] != "CLEAN") for x in r)
    print(f"  {arm:5s} kappa (min head) {[round(min(x[2]), 4) for x in r]} | classes {[x[1][:2] for x in r]} | "
          f"SOLVED iff CLEAN on {iff}/8")
rescore_hook.install("auto")
for arm in ("R1", "R2", "R1DF"):
    accs = []
    for s in range(20, 28):
        b = torch.load(f"{RING}/{arm}_s{s}/{arm}.pt", map_location="cpu", weights_only=False)
        m = build(arm, env, 32); m.load_state_dict(b["model_state_dict"])
        te = GridWorldCancel(size=32, seed=10000, p_plus=0.5)
        accs.append(evaluate(m, te, 128, 200, "cpu", seed=EVAL_SEED)[0])
    ev = [rows[(arm, s)][3] for s in range(20, 28)]
    print(f"  {arm:5s} acc T=128 eval mode {np.mean(ev):.4f} | re-scored (200 traj) {np.mean(accs):.4f} | per seed re-scored {[round(a, 3) for a in accs]}")
tails = [v[4] for v in rows.values()]; accs = [v[3] for v in rows.values()]
print(f"  r(final-5% loss, acc T=128) over 24 runs: {np.corrcoef(tails, accs)[0, 1]:+.3f}")

print("== escape ==")
for arm, var in (("E", "Vanilla_r4mi"), ("C", "Vanilla"), ("S", "Vanilla_r4mi")):
    out = []
    for s in range(8):
        ck = f"{ESC}/{arm}/p0/{var}_s{s}/{var}.pt"
        b = torch.load(ck, map_location="cpu", weights_only=False)
        o = head_stats(ck); cls = basin(o); solved = classify_run(b["losses"])["registered"] == "SOLVED"
        extra = ""
        if arm == "E":
            W = b["model_state_dict"]["action_to_lie.w_out.weight"]
            extra = f" new/old W_out cols {float(W[:, 2:].norm() / W[:, :2].norm()):.3f}"
        out.append((solved, cls)); print(f"  {arm} s{s}: {'SOLVED' if solved else 'unsolved'} {cls}{extra}")
    iff = sum((x[0] and x[1] == "CLEAN") or (not x[0] and x[1] != "CLEAN") for x in out)
    print(f"  {arm}: SOLVED iff CLEAN on {iff}/8")
