#!/bin/bash
# Goal-directed task suite: multistop, switching, noisy.
# Compares {Vanilla, Level15, Level15EM, Level15NoDrop} on each.
# Single seed for scoping; multi-seed if any task shows large variance.
set -u
cd /home/prashr
REPO=/home/prashr/mapformer
LOGS=$REPO/logs
mkdir -p "$LOGS"

is_gpu_free() {
    local mem util
    while IFS=', ' read -r mem util; do
        mem=${mem//[^0-9]/}; util=${util//[^0-9]/}
        [ -z "$mem" ] && mem=0; [ -z "$util" ] && util=0
        if [ "$mem" -ge 5000 ] || [ "$util" -ge 50 ]; then return 1; fi
    done < <(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
    return 0
}

echo "[$(date)] Polling for free GPUs..."
while ! is_gpu_free; do sleep 60; done
echo "[$(date)] GPUs free."

train() {
    local variant=$1 task=$2 gpu=$3
    local out="$REPO/runs/${variant}_goal_${task}/seed0"
    mkdir -p "$out"
    # Task-specific args
    local extra=""
    case "$task" in
      multistop) extra="--T-explore 64 --T-navigate 64 --T-per-segment 32 --n-stops 2";;
      switching) extra="--T-explore 64 --T-navigate 64 --T-pre-switch 32 --T-post-switch 32";;
      noisy)     extra="--T-explore 64 --T-navigate 64 --p-transition-noise 0.10";;
    esac
    CUDA_VISIBLE_DEVICES=$gpu python3 -u -m mapformer.train_goal \
        --variant $variant --seed 0 \
        --n-landmarks 200 \
        --task $task \
        $extra \
        --epochs 50 --n-batches 64 --batch-size 128 \
        --device cuda \
        --output-dir mapformer/runs/${variant}_goal_${task}/seed0 \
        > "$LOGS/${variant}_goal_${task}_s0.log" 2>&1
}

run_pair() { train $1 $2 0 & P1=$!; train $3 $4 1 & P2=$!; wait $P1 $P2; }

for task in multistop switching noisy; do
    echo "[$(date)] === Task: $task ==="
    echo "[$(date)] [pair 1] Vanilla + Level15"
    run_pair Vanilla $task Level15 $task
    echo "[$(date)] [pair 2] Level15EM + Level15NoDrop"
    run_pair Level15EM $task Level15NoDrop $task
done

echo "[$(date)] All trainings done. Final losses:"
for task in multistop switching noisy; do
    for v in Vanilla Level15 Level15EM Level15NoDrop; do
        f="$LOGS/${v}_goal_${task}_s0.log"
        [ -f "$f" ] || continue
        echo "  ${task} ${v}: $(grep 'Epoch  50/50' $f | tail -1)"
        echo "       $(grep 'Held-out' $f)"
    done
done

echo "[$(date)] Aggregating results..."
python3 -u <<'PYEOF' > "$REPO/GOAL_TASKS_RESULTS.md" 2>"$LOGS/goal_tasks_eval.err"
import torch, torch.nn.functional as F, numpy as np
from pathlib import Path
from mapformer.environment_goal import GoalDirectedGridWorld
from mapformer.model import MapFormerWM, MapFormerEM
from mapformer.model_inekf_level15 import MapFormerWM_Level15InEKF
from mapformer.model_inekf_level15_em import MapFormerEM_Level15InEKF
from mapformer.model_inekf_level15_nodrop import MapFormerWM_Level15NoDrop
from mapformer.train_goal import evaluate_goal

VARIANT_CLS = {
    "Vanilla":         MapFormerWM,
    "Level15":         MapFormerWM_Level15InEKF,
    "Level15EM":       MapFormerEM_Level15InEKF,
    "Level15NoDrop":   MapFormerWM_Level15NoDrop,
}

def build(variant, ckpt):
    c = torch.load(ckpt, map_location="cuda", weights_only=False)
    cfg = c["config"]
    cls = VARIANT_CLS[variant]
    m = cls(vocab_size=cfg["vocab_size"], d_model=cfg["d_model"],
            n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
            grid_size=cfg["grid_size"])
    m.load_state_dict(c["model_state_dict"])
    return m.cuda().eval()

print("# Goal-directed task suite — sequential, switching, noisy\n")
print("Three behavioural tests of the cognitive map beyond single point-to-point navigation:")
print("- **multistop**: announce two goals up front, navigate A then B (sequential planning).")
print("- **switching**: start toward A, mid-episode the goal switches to B (flexible replan).")
print("- **noisy**: closed-loop BFS oracle, but execution is stochastic (p_transition=0.10).\n")
print("All run on lm200, single seed, 50 epochs, eval on held-out env (obs_map seed=1000).\n")

env_test = GoalDirectedGridWorld(size=64, n_obs_types=16, p_empty=0.5,
                                  n_landmarks=200, seed=1000)

for task in ["multistop", "switching", "noisy"]:
    print(f"## Task: {task}\n")
    if task == "multistop":
        kw = dict(task="multistop", n_stops=2, T_per_segment=32)
        cols = ["T_exp=32, T_seg=16", "T_exp=64, T_seg=32 (train)", "T_exp=128, T_seg=32 (OOD)"]
        eval_args = [
            dict(T_explore=32, T_navigate=32, T_per_segment=16, **{k:v for k,v in kw.items() if k!="T_per_segment"}),
            dict(T_explore=64, T_navigate=64, T_per_segment=32, **{k:v for k,v in kw.items() if k!="T_per_segment"}),
            dict(T_explore=128, T_navigate=64, T_per_segment=32, **{k:v for k,v in kw.items() if k!="T_per_segment"}),
        ]
    elif task == "switching":
        kw = dict(task="switching")
        cols = ["T_exp=32, switch@16", "T_exp=64, switch@32 (train)", "T_exp=128, switch@32 (OOD)"]
        eval_args = [
            dict(T_explore=32, T_navigate=32, T_pre_switch=16, T_post_switch=16, **kw),
            dict(T_explore=64, T_navigate=64, T_pre_switch=32, T_post_switch=32, **kw),
            dict(T_explore=128, T_navigate=64, T_pre_switch=32, T_post_switch=32, **kw),
        ]
    else:  # noisy
        kw = dict(task="noisy")
        cols = ["p_noise=0.10 (train)", "p_noise=0.20 (harder)", "T_exp=128, p_noise=0.10 (OOD)"]
        eval_args = [
            dict(T_explore=64, T_navigate=64, p_transition_noise=0.10, **kw),
            dict(T_explore=64, T_navigate=64, p_transition_noise=0.20, **kw),
            dict(T_explore=128, T_navigate=64, p_transition_noise=0.10, **kw),
        ]
    header = "| Variant | " + " | ".join(cols) + " |"
    print(header)
    print("|" + "|".join(["---"] * (len(cols) + 1)) + "|")
    for variant in ["Vanilla", "Level15", "Level15EM", "Level15NoDrop"]:
        ckpt = Path(f"mapformer/runs/{variant}_goal_{task}/seed0/{variant}_goal.pt")
        if not ckpt.exists():
            print(f"| {variant} | " + " | ".join(["—"] * len(cols)) + " |"); continue
        m = build(variant, ckpt)
        cells = []
        for ea in eval_args:
            acc, _ = evaluate_goal(m, env_test, n_trials=150, device="cuda", seed=2000, **ea)
            cells.append(f"{acc:.3f}" if acc is not None else "—")
        del m; torch.cuda.empty_cache()
        print(f"| **{variant}** | " + " | ".join(cells) + " |")
    print()

print("## Interpretation\n")
print("- **multistop**: longer chained BFS path means more drift between subgoals.")
print("  Predicted: correction's advantage GROWS with #stops vs Vanilla.\n")
print("- **switching**: the switch-token interrupts the navigate stream.")
print("  Predicted: all correction variants similar; Vanilla worst because")
print("  drift on the pre-switch path leaves its map further off.\n")
print("- **noisy**: closed-loop BFS oracle replans after each (stochastic)")
print("  executed action. Predicted: Level15 ≫ Vanilla — correction matters")
print("  most when each commanded action might not execute.\n")
print("*Auto-generated by run_goal_tasks.sh, single seed (seed=0)*\n")
PYEOF

cd "$REPO"
git pull --rebase 2>&1 | tail -3
git add GOAL_TASKS_RESULTS.md environment_goal.py train_goal.py run_goal_tasks.sh
for task in multistop switching noisy; do
    for v in Vanilla Level15 Level15EM Level15NoDrop; do
        git add runs/${v}_goal_${task}/seed0/${v}_goal.pt 2>/dev/null || true
    done
done
git commit -m "Goal-directed task suite: multistop, switching, noisy

Three new behavioural tasks beyond single point-to-point navigation:
- multistop: sequential planning (visit A then B)
- switching: flexible replan when goal changes mid-episode
- noisy: closed-loop control under transition-noise execution

Single seed comparison of Vanilla / Level15 / Level15EM / Level15NoDrop
on each. Result in GOAL_TASKS_RESULTS.md.
" 2>&1 | tail -3
git push origin main 2>&1 | tail -3
echo "[$(date)] Done."
