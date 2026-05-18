#!/bin/bash
# Train Level15_SR (successor-representation aux head) on lm200 single-env,
# then re-run the two probes that failed without the SR objective:
#   - probe_goal_distance (head-state version)
#   - probe_active_inference (horizon 1, 4)
#
# If SR pretraining adds usable goal-relative info to the representation,
# these probes should jump above chance.
set -u
cd /home/prashr
REPO=/home/prashr/mapformer
LOGS=$REPO/logs
mkdir -p "$LOGS" "$REPO/paper_figures/sr_probes"

# ---- training: 3 seeds (use seed-outer-loop, but with one variant only) ----
train_one() {
    local seed=$1 gpu=$2 aux_coef=${3:-0.5}
    mkdir -p "$REPO/runs/Level15_SR_lm200/seed${seed}"
    CUDA_VISIBLE_DEVICES=$gpu python3 -u -m mapformer.train_variant \
        --variant Level15_SR --seed $seed \
        --n-landmarks 200 --grid-size 64 \
        --aux-coef $aux_coef \
        --epochs 50 --n-batches 156 \
        --device cuda \
        --output-dir mapformer/runs/Level15_SR_lm200/seed${seed} \
        > "$LOGS/Level15_SR_lm200_s${seed}.log" 2>&1
}

echo "[$(date)] seeds 0 + 1 in parallel"
train_one 0 0 0.5 & P1=$!
train_one 1 1 0.5 & P2=$!
wait $P1 $P2
echo "[$(date)] seed 2"
train_one 2 0 0.5
echo "[$(date)] training done"

# ---- probes on seed 0 (quick signal check before tightening to n=3) ----
PROBE_DIR=$REPO/paper_figures/sr_probes
mkdir -p "$PROBE_DIR/goal_distance" "$PROBE_DIR/active_inference"

CKPT="$REPO/runs/Level15_SR_lm200/seed0/Level15_SR.pt"
echo "[$(date)] goal-distance probe on Level15_SR s0"
CUDA_VISIBLE_DEVICES=0 python3 -u -m mapformer.probe_goal_distance \
    --variant Level15_SR --checkpoint "$CKPT" \
    --n-train-envs 1 --n-test-envs 0 \
    --train-env-seed-base 0 \
    --n-trajectories-per-env 200 \
    --T 128 --n-landmarks 200 \
    --train-goal-frac 0.75 \
    --epochs 20 --batch-size 512 \
    --output-json "$PROBE_DIR/goal_distance/Level15_SR_s0.json" \
    > "$LOGS/probe_gd_Level15_SR_s0.log" 2>&1

echo "[$(date)] active-inference probe on Level15_SR s0 (h=1, 4)"
for h in 1 4; do
    CUDA_VISIBLE_DEVICES=0 python3 -u -m mapformer.probe_active_inference \
        --variant Level15_SR --checkpoint "$CKPT" \
        --n-episodes 50 --T-explore 64 --T-navigate 32 \
        --horizon $h \
        --output-json "$PROBE_DIR/active_inference/Level15_SR_h${h}_s0.json" \
        > "$LOGS/probe_ai_Level15_SR_h${h}_s0.log" 2>&1
done

# ---- aggregate: compare to existing Level15 baselines ----
cd "$REPO"
python3 -u <<'PYEOF' > "$REPO/SR_PRETRAIN_RESULTS.md" 2>"$LOGS/sr_agg.err"
import json, torch, numpy as np
from pathlib import Path

def fmt(arr):
    if not arr: return "—"
    return f"{np.mean(arr):.3f} ± {np.std(arr):.3f} (n={len(arr)})"

print("# Successor-representation pretraining: does goal-relative info emerge?\n")
print("Level15 trained with an auxiliary head predicting `p(token v ∈ next K=8 positions)`")
print("alongside next-token CE. Same architecture otherwise. Single-env lm200.\n")
print("If the SR aux teaches the representation to encode multi-step")
print("reachability, the same probes that failed on Level15 (chance-level)")
print("should improve. No goal conditioning during pretraining.\n")

# Training loss
print("## Training (Level15_SR, n=3 seeds)\n")
losses = []
for s in [0, 1, 2]:
    p = Path(f"runs/Level15_SR_lm200/seed{s}/Level15_SR.pt")
    if not p.exists(): continue
    c = torch.load(p, map_location="cpu", weights_only=False)
    losses.append(c.get("losses", [None])[-1])
print(f"Final training loss (next-token CE only): {fmt([l for l in losses if l is not None])}\n")

# Goal-distance probe
print("## Goal-distance probe (head-state, s0)\n")
print("Comparison to Level15 baseline from PROBE_GOAL_DISTANCE.md.\n")
print("| Variant | train_goals MAE / Spearman | heldout_goals MAE / Spearman | const-baseline |")
print("|---|---|---|---|")
# Existing Level15 baseline (hardcoded from previous probe)
print("| Level15 (no SR) | 12.66 / 0.06 | 14.35 / -0.02 | 10.85 |")
# Level15_SR result
p = Path("paper_figures/sr_probes/goal_distance/Level15_SR_s0.json")
if p.exists():
    j = json.loads(p.read_text())
    r1 = j["results"].get("train_env_train_goals", {})
    r2 = j["results"].get("train_env_heldout_goals", {})
    b = r1.get("mae_const_baseline", float("nan"))
    print(f"| **Level15_SR** | {r1.get('mae', 0):.2f} / {r1.get('spearman', 0):.2f} "
          f"| {r2.get('mae', 0):.2f} / {r2.get('spearman', 0):.2f} | {b:.2f} |")
else:
    print("| **Level15_SR** | — | — | — |")
print()

# Active-inference probe
print("## Active-inference closed-loop (s0)\n")
print("Comparison to Level15 baseline from ACTIVE_INFERENCE_RESULTS.md.\n")
print("| Variant | horizon=1 | horizon=4 |")
print("|---|---|---|")
print("| Level15 (no SR) | 0.02 (1.5× opt) | 0.02 (2.8× opt) |")
row = ["**Level15_SR**"]
for h in [1, 4]:
    p = Path(f"paper_figures/sr_probes/active_inference/Level15_SR_h{h}_s0.json")
    if not p.exists(): row.append("—"); continue
    j = json.loads(p.read_text())
    sr = j["success_rate"]; ratio = j.get("mean_ratio_to_optimal")
    cell = f"{sr:.2f}" if ratio is None else f"{sr:.2f} ({ratio:.1f}× opt)"
    row.append(cell)
print("| " + " | ".join(row) + " |")
print()

print("## Decision\n")
print("- **Goal-distance probe: heldout Spearman > 0.5** → SR pretraining works; tighten error bars + run baselines.")
print("- **Active-inference success > 0.20** → SR-trained world model is usable for goal-directed planning. Workshop-defining.")
print("- **Both ≈ Level15 baseline** → SR aux didn't add usable goal-relative info. Try larger aux_coef, longer horizon, or goal-conditional aux head.")
print()
print("*Auto-generated by run_sr_pretrain_and_probe.sh*")
PYEOF

git pull --rebase 2>&1 | tail -3
git add model_inekf_level15_sr.py train_variant.py run_sr_pretrain_and_probe.sh \
    SR_PRETRAIN_RESULTS.md paper_figures/sr_probes/ 2>/dev/null
for seed in 0 1 2; do
    git add runs/Level15_SR_lm200/seed${seed}/*.pt 2>/dev/null || true
done
git commit -m "Level15_SR: successor-representation aux pretraining + probe sweep

New variant: Level15 + auxiliary head predicting 'token v in next K=8 positions'
trained jointly with next-token CE. Tests whether multi-step reachability,
when supervised during pretraining, lifts the goal-distance probe and
active-inference protocols above chance.

If signal: workshop-defining 'cognitive map → usable for planning' link.
If not: pretraining alone is insufficient; need goal-conditional supervision.
" 2>&1 | tail -3
git push origin main 2>&1 | tail -3
echo "[$(date)] done."
