#!/bin/bash
# RoPE diagnostic: does MapFormer beat RoPE on MiniGrid-DoorKey?
# If RoPE matches MapFormer/Level15 → DoorKey doesn't differentiate
# path-integrating models from generic positional encodings.
set -u
cd /home/prashr
REPO=/home/prashr/mapformer
LOGS=$REPO/logs
mkdir -p "$LOGS"

train() {
    local cfg=$1 noise=$2 gpu=$3
    mkdir -p "$REPO/runs/minigrid_doorkey_${cfg}_cached/RoPE/seed0"
    CUDA_VISIBLE_DEVICES=$gpu python3 -u -m mapformer.train_variant \
        --variant RoPE --seed 0 \
        --n-landmarks 0 --p-action-noise $noise \
        --epochs 50 --n-batches 156 \
        --env minigrid_doorkey --minigrid-tokenization obj_color \
        --minigrid-cached-buffer 25000 \
        --device cuda \
        --output-dir mapformer/runs/minigrid_doorkey_${cfg}_cached/RoPE/seed0 \
        > "$LOGS/mg_doorkey_cached_${cfg}_RoPE_s0.log" 2>&1
}

echo "[$(date)] Training RoPE: clean (gpu0) + noise (gpu1) in parallel"
train clean 0.0  0 &
P1=$!
train noise 0.10 1 &
P2=$!
wait $P1 $P2
echo "[$(date)] RoPE training done."

echo "[$(date)] Evaluating Vanilla, Level15, RoPE side-by-side..."
python3 -u <<'PYEOF' > "$REPO/MINIGRID_DOORKEY_ROPE_DIAG.md" 2>"$LOGS/mg_rope_eval.err"
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

print("# RoPE diagnostic on MiniGrid-DoorKey-8x8\n")
print("Question: does MapFormer/Level15 actually beat RoPE on this env?")
print("If RoPE matches the others, DoorKey-random-policy doesn't differentiate")
print("path-integrating models — suggesting we'd need a richer task (Option 3)")
print("or a different env to test MapFormer's specific contribution.\n")
print("Single seed each variant, cached-buffer training, all numbers from same eval.\n")

variants = ["Vanilla", "Level15", "RoPE"]
def make_env(seed):
    return MiniGridWorld(env_name="MiniGrid-DoorKey-8x8-v0",
                          tokenization="obj_color", seed=seed)

for noise_label, noise_dir, eval_noise in [
    ("Primary (no noise)",   "minigrid_doorkey_clean_cached", 0.0),
    ("Ablation (10% noise)", "minigrid_doorkey_noise_cached", 0.10),
]:
    print(f"## {noise_label}\n")
    print("| Variant | T=128 OOD acc | T=512 OOD acc | T=128 OOD NLL | T=512 OOD NLL |")
    print("|---|---|---|---|---|")
    for v in variants:
        ckpt = Path(f"mapformer/runs/{noise_dir}/{v}/seed0/{v}.pt")
        if not ckpt.exists():
            print(f"| {v} | (no ckpt) | — | — | — |")
            continue
        m = build(v, ckpt)
        env_ood = make_env(seed=1000)
        a128_ood, n128_ood = eval_revisit(m, env_ood, 128, 100, seed=2000,
                                            p_action_noise=eval_noise)
        a512_ood, n512_ood = eval_revisit(m, env_ood, 512, 50, seed=2000,
                                            p_action_noise=eval_noise)
        print(f"| **{v}** | "
              f"{a128_ood:.3f} | "
              f"{a512_ood:.3f} | "
              f"{n128_ood:.3f} | "
              f"{n512_ood:.3f} |")
        del m; torch.cuda.empty_cache()
    print()

print("\n**Decision rule:**")
print("- If RoPE matches Vanilla/Level15 within ~2pp on T=512 OOD: env is not")
print("  differentiating MapFormer's path-integration mechanism. Skip Option 3.")
print("- If RoPE underperforms by >5pp: env exercises path integration; Option 3")
print("  becomes worth pursuing.")
print("\n*Auto-generated by run_minigrid_rope_diag.sh*\n")
PYEOF
echo "[$(date)] Eval done."
tail -20 "$REPO/MINIGRID_DOORKEY_ROPE_DIAG.md"

cd "$REPO"
git pull --rebase 2>&1 | tail -3
git add MINIGRID_DOORKEY_ROPE_DIAG.md run_minigrid_rope_diag.sh
git add runs/minigrid_doorkey_clean_cached/RoPE/seed0/*.pt \
        runs/minigrid_doorkey_noise_cached/RoPE/seed0/*.pt 2>/dev/null || true
git commit -m "RoPE diagnostic on MiniGrid-DoorKey: does the env exercise MapFormer?

Trains RoPE (standard transformer with fixed token-index rotations,
no path integration from action sequence) on the same cached buffer
and compares to Vanilla MapFormer and Level15 on revisit accuracy at
T=128 and T=512 OOD, with and without action noise.

Decision rule: if RoPE matches MapFormer within ~2pp, the env is not
differentiating path-integrating models from generic positional
encodings — meaning the +10pp Level15-noise win we saw is happening
for reasons orthogonal to MapFormer's specific contribution.
" 2>&1 | tail -3
git push origin main 2>&1 | tail -3
echo "[$(date)] Done."
