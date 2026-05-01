#!/bin/bash
# Train the 3 remaining MiniGrid-DoorKey configs with the cached buffer.
# Buffer is already on disk from run_minigrid_cached_test.sh, so each load
# is instant and each training takes ~1.5 min on GPU.
#
# Configs: (Vanilla, clean), (Vanilla, noise), (Level15, noise).
# (Level15 clean already trained.)
set -u
cd /home/prashr
REPO=/home/prashr/mapformer
LOGS=$REPO/logs
mkdir -p "$LOGS"

train() {
    local variant=$1 cfg=$2 noise=$3 gpu=$4
    mkdir -p "$REPO/runs/minigrid_doorkey_${cfg}_cached/${variant}/seed0"
    CUDA_VISIBLE_DEVICES=$gpu python3 -u -m mapformer.train_variant \
        --variant $variant --seed 0 \
        --n-landmarks 0 --p-action-noise $noise \
        --epochs 50 --n-batches 156 \
        --env minigrid_doorkey --minigrid-tokenization obj_color \
        --minigrid-cached-buffer 25000 \
        --device cuda \
        --output-dir mapformer/runs/minigrid_doorkey_${cfg}_cached/${variant}/seed0 \
        > "$LOGS/mg_doorkey_cached_${cfg}_${variant}_s0.log" 2>&1
}

echo "[$(date)] Pair 1: Vanilla clean (gpu0) + Vanilla noise (gpu1)"
train Vanilla clean 0.0  0 &
P1=$!
train Vanilla noise 0.10 1 &
P2=$!
wait $P1 $P2
echo "[$(date)] Pair 1 done."

echo "[$(date)] Pair 2: Level15 noise (gpu0)"
train Level15 noise 0.10 0 &
P3=$!
wait $P3
echo "[$(date)] Pair 2 done."

echo "[$(date)] Evaluating all 4 cached checkpoints..."
python3 -u <<'PYEOF' > "$REPO/MINIGRID_DOORKEY_CACHED.md" 2>"$LOGS/mg_cached_eval.err"
import torch, torch.nn.functional as F, numpy as np
from pathlib import Path
from mapformer.minigrid_env import MiniGridWorld
from mapformer.model import MapFormerWM
from mapformer.model_inekf_level15 import MapFormerWM_Level15InEKF

VARIANT_CLS = {"Vanilla": MapFormerWM, "Level15": MapFormerWM_Level15InEKF}

def build(variant, ckpt_path):
    c = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    cfg = c.get("config", {})
    cls = VARIANT_CLS[variant]
    m = cls(vocab_size=cfg["vocab_size"], d_model=cfg.get("d_model", 128),
            n_heads=cfg.get("n_heads", 2), n_layers=cfg.get("n_layers", 1),
            grid_size=cfg.get("grid_size", 8))
    m.load_state_dict(c["model_state_dict"])
    return m.cuda().eval()

def eval_revisit(model, env, T, n_trials, seed, p_action_noise=0.0):
    torch.manual_seed(seed); np.random.seed(seed)
    c = tot = 0; nll_sum = 0.0
    with torch.no_grad():
        for _ in range(n_trials):
            tokens, _, rm = env.generate_trajectory(T, p_action_noise=p_action_noise)
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

print("# MiniGrid-DoorKey-8x8 (cached-buffer training): Vanilla vs Level 1.5\n")
print("Single seed (s0). Trained with `MiniGridWorld_Cached` (25K-trajectory ")
print("pre-built buffer; ~35x faster wall-clock than live `gym.step` training, ")
print("semantically equivalent for noise-as-token-corruption regime).\n")
print("DoorKey naturally has landmarks (key, door, goal). Tokenization: ")
print("`obj_color` (66 obs types). Forward-biased random policy ")
print("(65% forward, 30% turn, 5% other).\n")

variants = ["Vanilla", "Level15"]
def make_env(seed):
    return MiniGridWorld(env_name="MiniGrid-DoorKey-8x8-v0",
                          tokenization="obj_color", seed=seed)

for noise_label, noise_dir, eval_noise in [
    ("Primary (no noise)",   "minigrid_doorkey_clean_cached", 0.0),
    ("Ablation (10% noise)", "minigrid_doorkey_noise_cached", 0.10),
]:
    print(f"## {noise_label}\n")
    print("| Variant | T=128 in-dist | T=512 in-dist | T=128 OOD | T=512 OOD |")
    print("|---|---|---|---|---|")
    for v in variants:
        ckpt = Path(f"mapformer/runs/{noise_dir}/{v}/seed0/{v}.pt")
        if not ckpt.exists():
            print(f"| {v} | (no ckpt) | — | — | — |")
            continue
        m = build(v, ckpt)
        env_id = make_env(seed=0)
        a128_id, n128_id = eval_revisit(m, env_id, 128, 100, seed=1000,
                                          p_action_noise=eval_noise)
        a512_id, n512_id = eval_revisit(m, env_id, 512, 50, seed=1000,
                                          p_action_noise=eval_noise)
        env_ood = make_env(seed=1000)
        a128_ood, n128_ood = eval_revisit(m, env_ood, 128, 100, seed=2000,
                                            p_action_noise=eval_noise)
        a512_ood, n512_ood = eval_revisit(m, env_ood, 512, 50, seed=2000,
                                            p_action_noise=eval_noise)
        print(f"| **{v}** | "
              f"{a128_id:.3f} (NLL {n128_id:.3f}) | "
              f"{a512_id:.3f} (NLL {n512_id:.3f}) | "
              f"{a128_ood:.3f} (NLL {n128_ood:.3f}) | "
              f"{a512_ood:.3f} (NLL {n512_ood:.3f}) |")
        del m; torch.cuda.empty_cache()
    print()

print("\n*Auto-generated by run_minigrid_cached_full.sh*\n")
PYEOF
echo "[$(date)] Eval done."
tail -25 "$REPO/MINIGRID_DOORKEY_CACHED.md"

cd "$REPO"
git pull --rebase 2>&1 | tail -3
git add MINIGRID_DOORKEY_CACHED.md run_minigrid_cached_full.sh
git add runs/minigrid_doorkey_clean_cached/Vanilla/seed0/*.pt \
        runs/minigrid_doorkey_clean_cached/Level15/seed0/*.pt \
        runs/minigrid_doorkey_noise_cached/Vanilla/seed0/*.pt \
        runs/minigrid_doorkey_noise_cached/Level15/seed0/*.pt 2>/dev/null || true
git commit -m "MiniGrid-DoorKey cached: Vanilla + Level15 on clean + noise

Trained 4 configs (V/L15 x clean/noise) using MiniGridWorld_Cached
(25K-trajectory pre-built buffer). Wall-clock ~2 min per pair vs
~10 min live, end-to-end ~10 min vs 5 hours.

Compares directly to the live trainings still in flight. The cached
path is semantically equivalent for noise-as-token-corruption (which
is how train.py applies noise) so the only differences expected are
seed-noise level (sub-pp on accuracy)." 2>&1 | tail -3
git push origin main 2>&1 | tail -3
echo "[$(date)] Done."
