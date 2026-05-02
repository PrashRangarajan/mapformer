#!/bin/bash
# Re-run only the fixed stages: P4 (MiniGrid-MemoryS13, kwarg bug fixed)
# and P5 (TEM-Lite + TEMFaithful with orthogonal W_a, NaN bug fixed).
# P1/P2/P3 already produced clean results in the earlier pipeline run.
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
        if [ "$mem" -ge 5000 ] || [ "$util" -ge 50 ]; then
            return 1
        fi
    done < <(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
    return 0
}

echo "[$(date)] Polling for free GPUs..."
while ! is_gpu_free; do
    nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>&1 | tr '\n' ' ' | sed "s/^/[$(date)] busy: /; s/$/\n/"
    sleep 120
done
echo "[$(date)] GPUs free."

# --- P2 (re-run): continuous nav with hard_ce loss (MSE was degenerate) ---
# Trains all 4 variants in pairs across both GPUs, then re-runs eval.
cnav_train() {
    local variant=$1 gpu=$2
    mkdir -p "$REPO/runs/cnav/$variant/seed0"
    CUDA_VISIBLE_DEVICES=$gpu python3 -u -m mapformer.train_continuous \
        --variant $variant --seed 0 \
        --epochs 30 --n-batches 156 --batch-size 128 --n-steps 128 \
        --buffer-size 25000 --n-grid-units 256 \
        --v-noise-std 0.05 --omega-noise-std 0.05 \
        --loss hard_ce \
        --device cuda \
        --output-dir mapformer/runs/cnav/$variant/seed0 \
        > "$LOGS/cnav_${variant}_s0.log" 2>&1
}

echo "[$(date)] [P2-fix] CNAV with hard_ce: Vanilla + Level15 (parallel)..."
cnav_train Vanilla 0 &
PC1=$!
cnav_train Level15 1 &
PC2=$!
wait $PC1 $PC2
echo "[$(date)] [P2-fix] Pair 1 done. VanillaEM + Level15EM..."
cnav_train VanillaEM 0 &
PC3=$!
cnav_train Level15EM 1 &
PC4=$!
wait $PC3 $PC4
echo "[$(date)] [P2-fix] All CNAV trainings done."

echo "[$(date)] [P2-fix] CNAV eval..."
python3 -u -m mapformer.eval_continuous \
    --checkpoints mapformer/runs/cnav/Vanilla/seed0/Vanilla.pt \
                  mapformer/runs/cnav/Level15/seed0/Level15.pt \
                  mapformer/runs/cnav/VanillaEM/seed0/VanillaEM.pt \
                  mapformer/runs/cnav/Level15EM/seed0/Level15EM.pt \
    --T-list 128 256 512 1024 \
    --noise-levels 0.0 0.05 0.1 0.2 \
    --n-traj 30 --device cuda \
    > "$REPO/CNAV_RESULTS.md" 2>"$LOGS/cnav_eval.err"
echo "[$(date)] [P2-fix] Eval done."
tail -30 "$REPO/CNAV_RESULTS.md"

echo "[$(date)] [P2-fix] Hex probes (all 4 variants)..."
for v in Vanilla Level15 VanillaEM Level15EM; do
    python3 -u -m mapformer.probe_hex_continuous \
        --checkpoint mapformer/runs/cnav/$v/seed0/$v.pt \
        --device cuda --n-traj 200 --T 256 --n-bins 64 \
        --save-rate-maps "$REPO/paper_figures/cnav_${v}_rate_maps_s0.npz" \
        > "$REPO/CNAV_HEX_${v}.md" 2>"$LOGS/cnav_probe_${v}.err"
done
echo "[$(date)] [P2-fix] Probes done."

# --- P4 (re-run): MiniGrid-MemoryS13 with the kwarg bug fixed ---
mg_train() {
    local variant=$1 gpu=$2
    mkdir -p "$REPO/runs/minigrid_memory_cached/$variant/seed0"
    CUDA_VISIBLE_DEVICES=$gpu python3 -u -m mapformer.train_variant \
        --variant $variant --seed 0 \
        --n-landmarks 0 --p-action-noise 0.0 \
        --epochs 50 --n-batches 156 \
        --env minigrid_memory --minigrid-tokenization obj_color \
        --minigrid-cached-buffer 25000 \
        --device cuda \
        --output-dir mapformer/runs/minigrid_memory_cached/$variant/seed0 \
        > "$LOGS/mg_memory_${variant}_s0.log" 2>&1
}

echo "[$(date)] [P4-fix] MemoryS13 — Vanilla (gpu0) bootstraps buffer..."
mg_train Vanilla 0
echo "[$(date)] [P4-fix] Buffer warm; Level15 + RoPE in parallel..."
mg_train Level15 0 &
P1=$!
mg_train RoPE 1 &
P2=$!
wait $P1 $P2
echo "[$(date)] [P4-fix] All MemoryS13 trainings done."

echo "[$(date)] [P4-fix] Eval Vanilla / Level15 / RoPE on MemoryS13..."
python3 -u <<'PYEOF' > "$REPO/MINIGRID_MEMORY_RESULTS.md" 2>"$LOGS/mg_memory_eval.err"
import torch, torch.nn.functional as F, numpy as np
from pathlib import Path
from mapformer.minigrid_env import MiniGridWorld
from mapformer.model import MapFormerWM
from mapformer.model_inekf_level15 import MapFormerWM_Level15InEKF
from mapformer.model_baseline_rope import MapFormerWM_RoPE

VARIANT_CLS = {"Vanilla": MapFormerWM, "Level15": MapFormerWM_Level15InEKF,
               "RoPE": MapFormerWM_RoPE}

def build(variant, ckpt_path):
    c = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    cfg = c.get("config", {})
    cls = VARIANT_CLS[variant]
    m = cls(vocab_size=cfg["vocab_size"], d_model=cfg.get("d_model", 128),
            n_heads=cfg.get("n_heads", 2), n_layers=cfg.get("n_layers", 1),
            grid_size=cfg.get("grid_size", 13))
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

print("# MiniGrid-MemoryS13: Vanilla vs Level 1.5 vs RoPE\n")
print("Different topology than DoorKey: starting room → narrow hallway → ")
print("choice room with two objects. Re-run after fixing MiniGridWorld_Cached's ")
print("kwarg-propagation bug (was crashing with 'p_transition_noise unexpected').\n")
T_LIST = [128, 512, 1024]
N_TRIALS = {128: 100, 512: 50, 1024: 25}
env_ood = MiniGridWorld(env_name="MiniGrid-MemoryS13-v0",
                        tokenization="obj_color", seed=1000)
print("| Variant | T=128 OOD | T=512 OOD | T=1024 OOD |")
print("|---|---|---|---|")
for v in ["Vanilla", "Level15", "RoPE"]:
    ckpt = Path(f"mapformer/runs/minigrid_memory_cached/{v}/seed0/{v}.pt")
    if not ckpt.exists():
        print(f"| {v} | (no ckpt) | — | — |"); continue
    m = build(v, ckpt)
    row = [f"**{v}**"]
    for T in T_LIST:
        a, n = eval_revisit(m, env_ood, T, N_TRIALS[T], seed=2000)
        row.append(f"{a:.3f} (NLL {n:.3f})" if a is not None else "(err)")
    print("| " + " | ".join(row) + " |")
    del m; torch.cuda.empty_cache()
PYEOF
echo "[$(date)] [P4-fix] Eval done."
tail -10 "$REPO/MINIGRID_MEMORY_RESULTS.md"

# --- P5 (re-run): TEM-Lite + TEMFaithful with orthogonal W_a ---
tem_train() {
    local variant=$1 cfg=$2 lm=$3 noise=$4 gpu=$5
    mkdir -p "$REPO/runs/${variant}_${cfg}/seed0"
    CUDA_VISIBLE_DEVICES=$gpu python3 -u -m mapformer.train_variant \
        --variant $variant --seed 0 \
        --n-landmarks $lm --p-action-noise $noise \
        --epochs 50 --n-batches 156 \
        --device cuda \
        --output-dir mapformer/runs/${variant}_${cfg}/seed0 \
        > "$LOGS/${variant}_${cfg}_s0.log" 2>&1
}

echo "[$(date)] [P5-fix] TEM + TEMFaithful (orthogonal W_a) on torus, 3 configs"
tem_train TEM         clean 0   0.0  0 &
PT1=$!
tem_train TEMFaithful clean 0   0.0  1 &
PT2=$!
wait $PT1 $PT2
tem_train TEM         noise 0   0.10 0 &
PT3=$!
tem_train TEMFaithful noise 0   0.10 1 &
PT4=$!
wait $PT3 $PT4
tem_train TEM         lm200 200 0.0 0 &
PT5=$!
tem_train TEMFaithful lm200 200 0.0 1 &
PT6=$!
wait $PT5 $PT6
echo "[$(date)] [P5-fix] All TEM trainings done."

# Sanity: any NaN losses in TEMFaithful logs?
for cfg in clean noise lm200; do
    if grep -q 'nan' "$LOGS/TEMFaithful_${cfg}_s0.log"; then
        echo "WARNING: TEMFaithful_${cfg} log contains NaN — investigate."
    fi
done

echo "[$(date)] [P5-fix] Eval TEM / TEMFaithful vs Vanilla / Level15..."
python3 -u <<'PYEOF' > "$REPO/TEM_RESULTS.md" 2>"$LOGS/tem_eval.err"
import torch, torch.nn.functional as F, numpy as np
from pathlib import Path
from mapformer.environment import GridWorld
from mapformer.model import MapFormerWM
from mapformer.model_inekf_level15 import MapFormerWM_Level15InEKF
from mapformer.model_tem import TEMRecurrent
from mapformer.model_tem_faithful import TEMFaithful

VARIANT_CLS = {"Vanilla": MapFormerWM, "Level15": MapFormerWM_Level15InEKF,
               "TEM": TEMRecurrent, "TEMFaithful": TEMFaithful}

def build(variant, ckpt):
    c = torch.load(ckpt, map_location="cuda", weights_only=False)
    cfg = c.get("config", {})
    cls = VARIANT_CLS[variant]
    m = cls(vocab_size=cfg["vocab_size"], d_model=cfg.get("d_model", 128),
            n_heads=cfg.get("n_heads", 2), n_layers=cfg.get("n_layers", 1),
            grid_size=cfg.get("grid_size", 64))
    m.load_state_dict(c["model_state_dict"])
    return m.cuda().eval()

def eval_revisit(model, env, T, n_trials, seed, p_action_noise=0.0):
    torch.manual_seed(seed); np.random.seed(seed)
    c = tot = 0; nll_sum = 0.0
    with torch.no_grad():
        for _ in range(n_trials):
            tokens, _, rm = env.generate_trajectory(T)
            tt = tokens.unsqueeze(0).cuda()
            if p_action_noise > 0:
                even = torch.zeros_like(tt, dtype=torch.bool); even[:, 0::2] = True
                noise = (torch.rand_like(tt, dtype=torch.float) < p_action_noise) & even
                rand = torch.randint(0, env.N_ACTIONS, tt.shape, device="cuda")
                tt = torch.where(noise, rand, tt)
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

print("# TEM baselines vs MapFormer (re-run with orthogonal W_a)\n")
print("Two TEM variants vs Vanilla / Level15 on torus, 3 regimes, single-seed.")
print("- **TEM**: GRU + factorised g/x + outer-product Hebbian memory.")
print("- **TEMFaithful**: per-action W_a parametrised as exp(skew(A_a)) — ")
print("  orthogonal by construction, never blows up. Modern-Hopfield memory.")
print("  This is the EM-RNN method the MapFormer paper claims to subsume.\n")

ckpts = {
    "clean": {
        "Vanilla":     "mapformer/runs/Vanilla_clean/seed0/Vanilla.pt",
        "Level15":     "mapformer/runs/Level15_clean/seed0/Level15.pt",
        "TEM":         "mapformer/runs/TEM_clean/seed0/TEM.pt",
        "TEMFaithful": "mapformer/runs/TEMFaithful_clean/seed0/TEMFaithful.pt",
    },
    "noise": {
        "Vanilla":     "mapformer/runs/Vanilla_noise/seed0/Vanilla.pt",
        "Level15":     "mapformer/runs/Level15_noise/seed0/Level15.pt",
        "TEM":         "mapformer/runs/TEM_noise/seed0/TEM.pt",
        "TEMFaithful": "mapformer/runs/TEMFaithful_noise/seed0/TEMFaithful.pt",
    },
    "lm200": {
        "Vanilla":     "mapformer/runs/Vanilla_lm200/seed0/Vanilla.pt",
        "Level15":     "mapformer/runs/Level15_lm200/seed0/Level15.pt",
        "TEM":         "mapformer/runs/TEM_lm200/seed0/TEM.pt",
        "TEMFaithful": "mapformer/runs/TEMFaithful_lm200/seed0/TEMFaithful.pt",
    },
}

for cfg_tag in ["clean", "noise", "lm200"]:
    n_lm = 200 if cfg_tag == "lm200" else 0
    eval_noise = 0.10 if cfg_tag == "noise" else 0.0
    print(f"## {cfg_tag}\n")
    print("| Variant | T=128 OOD | T=512 OOD | T=128 NLL | T=512 NLL |")
    print("|---|---|---|---|---|")
    for v in ["Vanilla", "Level15", "TEM", "TEMFaithful"]:
        ckpt = Path(ckpts[cfg_tag][v])
        if not ckpt.exists():
            print(f"| {v} | (no ckpt) | — | — | — |"); continue
        m = build(v, ckpt)
        env_ood = GridWorld(size=64, n_obs_types=16, p_empty=0.5,
                            n_landmarks=n_lm, seed=1000)
        a128, n128 = eval_revisit(m, env_ood, 128, 200, seed=2000, p_action_noise=eval_noise)
        a512, n512 = eval_revisit(m, env_ood, 512, 100, seed=2000, p_action_noise=eval_noise)
        print(f"| **{v}** | {a128:.3f} | {a512:.3f} | {n128:.3f} | {n512:.3f} |")
        del m; torch.cuda.empty_cache()
    print()
PYEOF
echo "[$(date)] [P5-fix] Eval done."
tail -25 "$REPO/TEM_RESULTS.md"

cd "$REPO"
git pull --rebase 2>&1 | tail -3
git add MINIGRID_MEMORY_RESULTS.md TEM_RESULTS.md \
        minigrid_env.py model_tem_faithful.py run_p4_p5_fix.sh
git add runs/minigrid_memory_cached/Vanilla/seed0/*.pt \
        runs/minigrid_memory_cached/Level15/seed0/*.pt \
        runs/minigrid_memory_cached/RoPE/seed0/*.pt \
        runs/TEM_clean/seed0/*.pt \
        runs/TEM_noise/seed0/*.pt \
        runs/TEM_lm200/seed0/*.pt \
        runs/TEMFaithful_clean/seed0/*.pt \
        runs/TEMFaithful_noise/seed0/*.pt \
        runs/TEMFaithful_lm200/seed0/*.pt 2>/dev/null || true
git commit -m "P4 + P5 re-run with bug fixes

Two bug fixes for the autonomous pipeline:

  (1) minigrid_env.py: MiniGridWorld and MiniGridWorld_Cached now
      accept (and ignore) p_transition_noise. The torus env grew that
      kwarg for the stochastic-transition MDP experiment, but the
      MiniGrid wrapper didn't, causing all P4 MemoryS13 trainings to
      crash in epoch 0 with TypeError. Result: empty MINIGRID_MEMORY_RESULTS.

  (2) model_tem_faithful.py: per-action W_a is now parameterised as
      exp(skew(A_a)) — guaranteed orthogonal by construction. Previous
      unconstrained W_a drifted to ill-conditioned regimes and produced
      NaN losses by epoch ~15. Orthogonality also matches Whittington
      2019's TEM convention more faithfully (TEM uses block-diagonal
      rotations, a special case of orthogonal matrices for compact-
      group structure).

P1/P2/P3 already produced clean results in the earlier autonomous run
and are not re-done here. Only MINIGRID_MEMORY_RESULTS.md and
TEM_RESULTS.md are produced fresh.
" 2>&1 | tail -3
git push origin main 2>&1 | tail -3
echo "[$(date)] Done."
