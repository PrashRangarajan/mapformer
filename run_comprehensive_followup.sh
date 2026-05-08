#!/bin/bash
# Comprehensive follow-up: GPUs are free, run all pending high-value experiments.
#
# Stage A: TEMFaithful seeds 0, 1, 2 × clean/noise/lm200 — REPLACES the buggy
#   seed-0 checkpoints (the predict-then-update bug from the audit). 9 trainings.
#
# Stage B: TEM-Lite seeds 1, 2 × clean/noise/lm200 — multi-seed verification of
#   the GRU+Hebbian baseline. 6 trainings.
#
# Stage C: MemoryS13 seeds 1, 2 × Vanilla/Level15/RoPE — multi-seed for the
#   cleanest non-noise win we have (+13pp at T=512 OOD). 6 trainings on cached buffer.
#
# Stage D: comprehensive multi-seed eval producing one big results table.

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
    sleep 60
done
echo "[$(date)] GPUs free."

# ============================================================
# Stage A: TEMFaithful (post-bug-fix) multi-seed × all configs
# ============================================================
tem_train() {
    local variant=$1 cfg=$2 lm=$3 noise=$4 seed=$5 gpu=$6
    mkdir -p "$REPO/runs/${variant}_${cfg}/seed${seed}"
    CUDA_VISIBLE_DEVICES=$gpu python3 -u -m mapformer.train_variant \
        --variant $variant --seed $seed \
        --n-landmarks $lm --p-action-noise $noise \
        --epochs 50 --n-batches 156 \
        --device cuda \
        --output-dir mapformer/runs/${variant}_${cfg}/seed${seed} \
        > "$LOGS/${variant}_${cfg}_s${seed}.log" 2>&1
}

run_pair() {
    local v1=$1 c1=$2 lm1=$3 n1=$4 s1=$5
    local v2=$6 c2=$7 lm2=$8 n2=$9 s2=${10}
    tem_train $v1 $c1 $lm1 $n1 $s1 0 &
    P1=$!
    tem_train $v2 $c2 $lm2 $n2 $s2 1 &
    P2=$!
    wait $P1 $P2
}

echo "[$(date)] [A1] TEMFaithful clean s0 (gpu0) + clean s1 (gpu1)"
run_pair TEMFaithful clean 0 0.0 0  TEMFaithful clean 0 0.0 1
echo "[$(date)] [A2] TEMFaithful clean s2 (gpu0) + noise s0 (gpu1)"
run_pair TEMFaithful clean 0 0.0 2  TEMFaithful noise 0 0.10 0
echo "[$(date)] [A3] TEMFaithful noise s1 (gpu0) + noise s2 (gpu1)"
run_pair TEMFaithful noise 0 0.10 1  TEMFaithful noise 0 0.10 2
echo "[$(date)] [A4] TEMFaithful lm200 s0 (gpu0) + lm200 s1 (gpu1)"
run_pair TEMFaithful lm200 200 0.0 0  TEMFaithful lm200 200 0.0 1
echo "[$(date)] [A5] TEMFaithful lm200 s2 (gpu0) + TEM-Lite clean s1 (gpu1)"
run_pair TEMFaithful lm200 200 0.0 2  TEM clean 0 0.0 1
echo "[$(date)] Stage A done."

# ============================================================
# Stage B: TEM-Lite multi-seed (seeds 1, 2 — seed 0 already exists)
# ============================================================
echo "[$(date)] [B1] TEM clean s2 (gpu0) + noise s1 (gpu1)"
run_pair TEM clean 0 0.0 2  TEM noise 0 0.10 1
echo "[$(date)] [B2] TEM noise s2 (gpu0) + lm200 s1 (gpu1)"
run_pair TEM noise 0 0.10 2  TEM lm200 200 0.0 1
echo "[$(date)] [B3] TEM lm200 s2 (gpu0)"
tem_train TEM lm200 200 0.0 2 0
echo "[$(date)] Stage B done."

# ============================================================
# Stage C: MemoryS13 multi-seed (seeds 1, 2)
# ============================================================
mg_train() {
    local variant=$1 seed=$2 gpu=$3
    mkdir -p "$REPO/runs/minigrid_memory_cached/$variant/seed${seed}"
    CUDA_VISIBLE_DEVICES=$gpu python3 -u -m mapformer.train_variant \
        --variant $variant --seed $seed \
        --n-landmarks 0 --p-action-noise 0.0 \
        --epochs 50 --n-batches 156 \
        --env minigrid_memory --minigrid-tokenization obj_color \
        --minigrid-cached-buffer 25000 \
        --device cuda \
        --output-dir mapformer/runs/minigrid_memory_cached/$variant/seed${seed} \
        > "$LOGS/mg_memory_${variant}_s${seed}.log" 2>&1
}

echo "[$(date)] [C1] MemoryS13 Vanilla s1 (gpu0) + Level15 s1 (gpu1)"
mg_train Vanilla 1 0 &
PMG1=$!
mg_train Level15 1 1 &
PMG2=$!
wait $PMG1 $PMG2
echo "[$(date)] [C2] MemoryS13 RoPE s1 (gpu0) + Vanilla s2 (gpu1)"
mg_train RoPE    1 0 &
PMG3=$!
mg_train Vanilla 2 1 &
PMG4=$!
wait $PMG3 $PMG4
echo "[$(date)] [C3] MemoryS13 Level15 s2 (gpu0) + RoPE s2 (gpu1)"
mg_train Level15 2 0 &
PMG5=$!
mg_train RoPE    2 1 &
PMG6=$!
wait $PMG5 $PMG6
echo "[$(date)] Stage C done."

# ============================================================
# Stage D: comprehensive multi-seed eval
# ============================================================
echo "[$(date)] [D] Eval — multi-seed table for all variants on torus..."
python3 -u <<'PYEOF' > "$REPO/MULTISEED_FOLLOWUP.md" 2>"$LOGS/multiseed_followup_eval.err"
import torch, torch.nn.functional as F, numpy as np
from pathlib import Path
from mapformer.environment import GridWorld
from mapformer.minigrid_env import MiniGridWorld
from mapformer.model import MapFormerWM, MapFormerEM
from mapformer.model_inekf_level15 import MapFormerWM_Level15InEKF
from mapformer.model_inekf_level15_em import MapFormerEM_Level15InEKF
from mapformer.model_baseline_rope import MapFormerWM_RoPE
from mapformer.model_tem import TEMRecurrent
from mapformer.model_tem_faithful import TEMFaithful
from mapformer.model_tem_t import TEM_T

VARIANT_CLS = {
    "Vanilla":     MapFormerWM,
    "Level15":     MapFormerWM_Level15InEKF,
    "VanillaEM":   MapFormerEM,
    "Level15EM":   MapFormerEM_Level15InEKF,
    "RoPE":        MapFormerWM_RoPE,
    "TEM":         TEMRecurrent,
    "TEMFaithful": TEMFaithful,
    "TEM_T":       TEM_T,
}

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

print("# Multi-seed follow-up (n=3 where possible) — TEMFaithful post-fix\n")
print("Comprehensive table after the comprehensive follow-up pipeline:")
print("- TEMFaithful re-trained at seeds 0/1/2 with the predict-then-update")
print("  bug fix (audit 2026-05-07).")
print("- TEM (Lite) extended to seeds 0/1/2.")
print("- MemoryS13 extended to seeds 0/1/2 for Vanilla / Level15 / RoPE.\n")

# ----- Torus configs -----
for cfg_tag in ["clean", "noise", "lm200"]:
    n_lm = 200 if cfg_tag == "lm200" else 0
    eval_noise = 0.10 if cfg_tag == "noise" else 0.0
    env_ood = GridWorld(size=64, n_obs_types=16, p_empty=0.5, n_landmarks=n_lm, seed=1000)
    print(f"## torus / {cfg_tag}\n")
    print("| Variant | T=128 OOD | T=512 OOD | T=512 NLL | n |")
    print("|---|---|---|---|---|")
    for variant in ["Vanilla", "Level15", "VanillaEM", "Level15EM", "TEM", "TEMFaithful", "TEM_T"]:
        a128s, a512s, n512s = [], [], []
        for s in [0, 1, 2]:
            ckpt = Path(f"mapformer/runs/{variant}_{cfg_tag}/seed{s}/{variant}.pt")
            if not ckpt.exists(): continue
            m = build(variant, ckpt)
            a128, _ = eval_revisit(m, env_ood, 128, 200, seed=2000+s, p_action_noise=eval_noise)
            a512, n512 = eval_revisit(m, env_ood, 512, 100, seed=2000+s, p_action_noise=eval_noise)
            if a128 is not None: a128s.append(a128)
            if a512 is not None: a512s.append(a512); n512s.append(n512)
            del m; torch.cuda.empty_cache()
        if not a128s:
            print(f"| {variant} | — | — | — | 0 |"); continue
        print(f"| **{variant}** | "
              f"{np.mean(a128s):.3f} ± {np.std(a128s):.3f} | "
              f"{np.mean(a512s):.3f} ± {np.std(a512s):.3f} | "
              f"{np.mean(n512s):.3f} | {len(a128s)} |")
    print()

# ----- MiniGrid-MemoryS13 -----
print("## minigrid_memory (MemoryS13) — clean only\n")
env_mem = MiniGridWorld(env_name="MiniGrid-MemoryS13-v0", tokenization="obj_color", seed=1000)
print("| Variant | T=128 OOD | T=512 OOD | T=1024 OOD | n |")
print("|---|---|---|---|---|")
for variant in ["Vanilla", "Level15", "RoPE"]:
    a128s, a512s, a1024s = [], [], []
    for s in [0, 1, 2]:
        ckpt = Path(f"mapformer/runs/minigrid_memory_cached/{variant}/seed{s}/{variant}.pt")
        if not ckpt.exists(): continue
        m = build(variant, ckpt)
        a128, _ = eval_revisit(m, env_mem, 128, 100, seed=2000+s)
        a512, _ = eval_revisit(m, env_mem, 512, 50, seed=2000+s)
        a1024, _ = eval_revisit(m, env_mem, 1024, 25, seed=2000+s)
        if a128 is not None: a128s.append(a128); a512s.append(a512); a1024s.append(a1024)
        del m; torch.cuda.empty_cache()
    if not a128s:
        print(f"| {variant} | — | — | — | 0 |"); continue
    print(f"| **{variant}** | "
          f"{np.mean(a128s):.3f} ± {np.std(a128s):.3f} | "
          f"{np.mean(a512s):.3f} ± {np.std(a512s):.3f} | "
          f"{np.mean(a1024s):.3f} ± {np.std(a1024s):.3f} | "
          f"{len(a128s)} |")

print("\n*Auto-generated by run_comprehensive_followup.sh*\n")
PYEOF
echo "[$(date)] [D] Eval done."
tail -60 "$REPO/MULTISEED_FOLLOWUP.md"

cd "$REPO"
git pull --rebase 2>&1 | tail -3
git add MULTISEED_FOLLOWUP.md run_comprehensive_followup.sh
git add runs/TEMFaithful_clean/seed0/*.pt runs/TEMFaithful_clean/seed1/*.pt runs/TEMFaithful_clean/seed2/*.pt \
        runs/TEMFaithful_noise/seed0/*.pt runs/TEMFaithful_noise/seed1/*.pt runs/TEMFaithful_noise/seed2/*.pt \
        runs/TEMFaithful_lm200/seed0/*.pt runs/TEMFaithful_lm200/seed1/*.pt runs/TEMFaithful_lm200/seed2/*.pt \
        runs/TEM_clean/seed1/*.pt runs/TEM_clean/seed2/*.pt \
        runs/TEM_noise/seed1/*.pt runs/TEM_noise/seed2/*.pt \
        runs/TEM_lm200/seed1/*.pt runs/TEM_lm200/seed2/*.pt \
        runs/minigrid_memory_cached/Vanilla/seed1/*.pt runs/minigrid_memory_cached/Vanilla/seed2/*.pt \
        runs/minigrid_memory_cached/Level15/seed1/*.pt runs/minigrid_memory_cached/Level15/seed2/*.pt \
        runs/minigrid_memory_cached/RoPE/seed1/*.pt runs/minigrid_memory_cached/RoPE/seed2/*.pt \
        2>/dev/null || true

git commit -m "Comprehensive multi-seed follow-up (TEMFaithful post-fix + extended baselines)

Stage A: TEMFaithful seeds 0/1/2 × clean/noise/lm200 with the audited
  bug fix (predict-then-update -> update-then-predict at action
  positions). REPLACES the buggy seed-0 checkpoints in the previous
  TEM_RESULTS.md (TEMFaithful at 0.42 across all configs). 9 trainings.

Stage B: TEM-Lite extended to n=3 across all configs. The previous
  multi-seed comparison only had seed 0 for TEM-Lite. 6 new trainings.

Stage C: MemoryS13 extended to n=3 for Vanilla / Level15 / RoPE — the
  +13pp Level15 win at T=512 OOD was single-seed; this verifies it
  multi-seed. 6 new trainings on cached buffer.

Stage D: MULTISEED_FOLLOWUP.md aggregates all multi-seed numbers into
  one comprehensive table covering torus (clean/noise/lm200) and
  MemoryS13 (clean only).
" 2>&1 | tail -3
git push origin main 2>&1 | tail -3
echo "[$(date)] Done."
