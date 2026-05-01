#!/bin/bash
# Control experiment: train Level15PC_v4 with aux_coef=0 (no aux loss applied).
#
# This isolates whether v4's +3.4pp lm200 win comes from:
#   (a) the aux loss / forward_model gradient (despite our audit saying it shouldn't)
#   (b) RNG state shift from forward_model's extra ~50K init params
#   (c) clip_grad_norm coupling — joint norm differs when forward_model has grad
#
# With aux_coef=0:
#   - forward_model exists, consumes init RNG (controls for b)
#   - forward_model never gets gradient from aux (since aux loss is multiplied by 0
#     in train.py)
#   - clip_grad_norm includes forward_model params, but their grad is 0
#     so they don't contribute to the joint norm (controls for c)
#
# So Control = same architecture, same RNG, same param count, but
# forward_model is mechanically a no-op.
#
# Predictions:
#   - Control ≈ v4 → win is RNG-shift only (b)
#   - Control ≈ Level15, < v4 → win is from gradient or clip-coupling (a/c)
#
# Wait for the other GPU jobs to complete before launching.
#
# 3 seeds × 2 configs (clean, lm200) = 6 trainings
# At ~10 min each on 2 GPUs in parallel: ~30 min.

set -u
cd /home/prashr
REPO=/home/prashr/mapformer
LOGS=$REPO/logs
mkdir -p "$LOGS"

is_running() {
    ps aux | grep -E "$1" | grep -v grep | grep -v run_v4_control >/dev/null
}

echo "[$(date)] Waiting for tma_standalone jobs to finish (or any other GPU users)..."
while is_running "tma_standalone\b"; do
    sleep 120
done
echo "[$(date)] GPUs free. Starting control experiment."

# Multi-seed: 3 seeds × 2 configs, 2 GPUs at a time
queue=()
for cfg in clean lm200; do
    for seed in 0 1 2; do
        queue+=("$cfg $seed")
    done
done

GPU0_PID=""; GPU1_PID=""
declare -A GPU_TAG

launch_one() {
    local gpu=$1 cfg=$2 seed=$3
    local n_lm=0 p_noise=0.0
    if [ "$cfg" = "lm200" ]; then n_lm=200; p_noise=0.10; fi
    local out_dir="$REPO/runs/Level15PC_v4_control_${cfg}/seed${seed}"
    mkdir -p "$out_dir"
    CUDA_VISIBLE_DEVICES=$gpu python3 -u -m mapformer.train_variant \
        --variant Level15PC_v4 --seed $seed \
        --n-landmarks $n_lm --p-action-noise $p_noise \
        --epochs 50 --n-batches 156 --aux-coef 0.0 \
        --device cuda \
        --output-dir "$out_dir" \
        > "$LOGS/Level15PC_v4_control_${cfg}_s${seed}.log" 2>&1 &
    local pid=$!
    GPU_TAG[$gpu]="${cfg}_s${seed}"
    if [ "$gpu" = "0" ]; then GPU0_PID=$pid; else GPU1_PID=$pid; fi
    echo "[$(date)] GPU$gpu: ${cfg} s${seed} (pid $pid)"
}

# Run 6 jobs on 2 GPUs
while [ ${#queue[@]} -gt 0 ] || [ -n "$GPU0_PID" ] || [ -n "$GPU1_PID" ]; do
    # Launch on free GPUs
    for gpu in 0 1; do
        local_pid_var="GPU${gpu}_PID"
        if [ -z "${!local_pid_var}" ] && [ ${#queue[@]} -gt 0 ]; then
            entry="${queue[0]}"; queue=("${queue[@]:1}")
            cfg=$(echo "$entry" | cut -d' ' -f1)
            seed=$(echo "$entry" | cut -d' ' -f2)
            launch_one $gpu $cfg $seed
        fi
    done
    # Poll for completion
    sleep 30
    for gpu in 0 1; do
        local_pid_var="GPU${gpu}_PID"
        pid="${!local_pid_var}"
        if [ -n "$pid" ] && ! kill -0 $pid 2>/dev/null; then
            echo "[$(date)] GPU$gpu: ${GPU_TAG[$gpu]} done"
            if [ "$gpu" = "0" ]; then GPU0_PID=""; else GPU1_PID=""; fi
        fi
    done
done

echo "[$(date)] All control trainings done."

# Eval and compare
echo "[$(date)] Stage 2: 4-way multi-seed comparison..."
cd /home/prashr
python3 -u <<'PYEOF' > "$REPO/V4_CONTROL_RESULTS.md" 2>"$LOGS/v4_control_eval.err"
import torch, torch.nn.functional as F, numpy as np, statistics as st
from pathlib import Path
from mapformer.environment import GridWorld
from mapformer.model_inekf_level15 import MapFormerWM_Level15InEKF
from mapformer.model_level15_pc_v4 import MapFormerWM_Level15PC_v4

VARIANT_CLS = {
    "Level15": MapFormerWM_Level15InEKF,
    "Level15PC_v4": MapFormerWM_Level15PC_v4,
    "Level15PC_v4_control": MapFormerWM_Level15PC_v4,  # same arch, aux_coef=0 at train
}

def build(variant, ckpt_path):
    c = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    cfg = c.get("config", {})
    cls = VARIANT_CLS[variant]
    m = cls(vocab_size=cfg["vocab_size"], d_model=cfg.get("d_model", 128),
            n_heads=cfg.get("n_heads", 2), n_layers=cfg.get("n_layers", 1),
            grid_size=cfg.get("grid_size", 64))
    m.load_state_dict(c["model_state_dict"])
    return m.cuda().eval()

def eval_revisit(model, env, T, n_trials, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    c = tot = 0; nll_sum = 0.0
    with torch.no_grad():
        for _ in range(n_trials):
            tokens, _, rm = env.generate_trajectory(T)
            tt = tokens.unsqueeze(0).cuda()
            try:
                logits = model(tt[:, :-1])
            except Exception:
                return None, None
            lp = F.log_softmax(logits, dim=-1)
            preds = lp.argmax(-1)[0]; tgts = tt[0, 1:]; mask = rm[1:].cuda()
            if mask.sum() == 0: continue
            c += (preds[mask] == tgts[mask]).sum().item()
            tot += mask.sum().item()
            idx = torch.arange(lp.shape[1], device="cuda")[mask]
            nll_sum += -lp[0, idx, tgts[mask]].sum().item()
    return (c / tot if tot else None, nll_sum / tot if tot else None)


print("# Control experiment: v4 win mechanism\n")
print("Compares Level15, Level15PC_v4 (aux_coef=0.1, our 'win'), and ")
print("Level15PC_v4_control (same architecture, aux_coef=0 — forward_model")
print("exists but never gets gradient).\n")
print("If control ≈ v4 → win is from RNG state shift (forward_model init)")
print("If control ≈ Level15 → win is from the aux loss / clip-norm coupling\n")

variants = [
    ("Level15", "Level15", "Level15"),
    ("Level15PC_v4", "Level15PC_v4", "Level15PC_v4"),
    ("Level15PC_v4_control", "Level15PC_v4_control", "Level15PC_v4"),
]

for cfg_tag, n_lm in [("clean", 0), ("lm200", 200)]:
    print(f"## Config: {cfg_tag}, OOD T=512 (3 seeds)\n")
    print("| Variant | seed 0 | seed 1 | seed 2 | mean ± std |")
    print("|---|---|---|---|---|")
    for label, dirname, ckpt_name in variants:
        per_seed = []
        for seed in [0, 1, 2]:
            ckpt = Path(f"mapformer/runs/{dirname}_{cfg_tag}/seed{seed}/{ckpt_name}.pt")
            if not ckpt.exists():
                per_seed.append(None); continue
            m = build(label, ckpt)
            env_ood = GridWorld(size=64, n_obs_types=16, p_empty=0.5,
                                 n_landmarks=n_lm, seed=seed + 1000)
            a, _ = eval_revisit(m, env_ood, 512, 100, seed=seed + 2000)
            per_seed.append(a)
            del m; torch.cuda.empty_cache()
        valid = [x for x in per_seed if x is not None]
        if valid:
            mean = st.mean(valid)
            std = st.pstdev(valid) if len(valid) > 1 else 0.0
            stats = f"{mean:.3f} ± {std:.3f}"
        else:
            stats = "—"
        cells = [f"{x:.3f}" if x is not None else "—" for x in per_seed]
        print(f"| {label} | {cells[0]} | {cells[1]} | {cells[2]} | **{stats}** |")
    print()

print("\n*Auto-generated by run_v4_control.sh*\n")
PYEOF

echo "[$(date)] Stage 2 done."

cd "$REPO"
git add V4_CONTROL_RESULTS.md run_v4_control.sh
git add runs/Level15PC_v4_control_clean/seed*/Level15PC_v4.pt 2>/dev/null || true
git add runs/Level15PC_v4_control_lm200/seed*/Level15PC_v4.pt 2>/dev/null || true
git commit -m "v4 control: train v4 architecture with aux_coef=0 to isolate mechanism

If v4's +3.4pp lm200 win is from the aux loss or grad-clip coupling,
this control should regress to Level15's performance. If the win is
from RNG state shift (forward_model's extra init params consuming
random numbers and shifting the init for everything that follows),
the control should match v4.

3 seeds × 2 configs (clean, lm200), single-seed evaluation per config
at T=512 OOD. Compares Level15 (no forward_model), Level15PC_v4
(aux_coef=0.1, the 'win'), and Level15PC_v4_control (aux_coef=0).

V4_CONTROL_RESULTS.md table reports mean ± std across seeds for each
variant on each config.
" 2>&1 | tail -3
git push origin main 2>&1 | tail -3
echo "[$(date)] Done."
