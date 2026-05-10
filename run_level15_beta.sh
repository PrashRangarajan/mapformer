#!/bin/bash
# Level15Beta — Level 1.5 InEKF + learnable softmax temperature β.
# Tests whether MapFormer's existing attention can recover TEMFaithful's
# landmark-retrieval sharpness just by sharpening softmax. Multi-seed.
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
while ! is_gpu_free; do sleep 60; done
echo "[$(date)] GPUs free."

train() {
    local cfg=$1 lm=$2 noise=$3 seed=$4 gpu=$5
    mkdir -p "$REPO/runs/Level15Beta_${cfg}/seed${seed}"
    CUDA_VISIBLE_DEVICES=$gpu python3 -u -m mapformer.train_variant \
        --variant Level15Beta --seed $seed \
        --n-landmarks $lm --p-action-noise $noise \
        --epochs 50 --n-batches 156 \
        --device cuda \
        --output-dir mapformer/runs/Level15Beta_${cfg}/seed${seed} \
        > "$LOGS/Level15Beta_${cfg}_s${seed}.log" 2>&1
}

run_pair() {
    train $1 $2 $3 $4 0 &
    P1=$!
    train $5 $6 $7 $8 1 &
    P2=$!
    wait $P1 $P2
}

echo "[$(date)] [1] clean s0 + clean s1"
run_pair clean 0 0.0 0  clean 0 0.0 1
echo "[$(date)] [2] clean s2 + noise s0"
run_pair clean 0 0.0 2  noise 0 0.10 0
echo "[$(date)] [3] noise s1 + noise s2"
run_pair noise 0 0.10 1  noise 0 0.10 2
echo "[$(date)] [4] lm200 s0 + lm200 s1"
run_pair lm200 200 0.0 0  lm200 200 0.0 1
echo "[$(date)] [5] lm200 s2"
train lm200 200 0.0 2 0
echo "[$(date)] All Level15Beta trainings done."

# Sanity + show learned β values
for cfg in clean noise lm200; do
    for seed in 0 1 2; do
        f="$LOGS/Level15Beta_${cfg}_s${seed}.log"
        [ -f "$f" ] || continue
        echo "  ${cfg} s${seed}: $(grep 'Epoch  50/50' $f | tail -1)"
    done
done

echo "[$(date)] Eval Level15Beta vs Level15 vs TEMFaithful (4-way comparison)..."
python3 -u <<'PYEOF' > "$REPO/LEVEL15BETA_RESULTS.md" 2>"$LOGS/level15beta_eval.err"
import torch, torch.nn.functional as F, numpy as np
from pathlib import Path
from mapformer.environment import GridWorld
from mapformer.model import MapFormerWM
from mapformer.model_inekf_level15 import MapFormerWM_Level15InEKF
from mapformer.model_inekf_level15_beta import MapFormerWM_Level15Beta
from mapformer.model_tem_faithful import TEMFaithful

VARIANT_CLS = {
    "Vanilla":     MapFormerWM,
    "Level15":     MapFormerWM_Level15InEKF,
    "Level15Beta": MapFormerWM_Level15Beta,
    "TEMFaithful": TEMFaithful,
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

print("# Level15Beta — learnable softmax temperature in attention\n")
print("Tests whether sharpening attention's softmax (one extra learnable scalar)")
print("recovers the TEMFaithful landmark-retrieval gap. Multi-seed (n=3) compared")
print("against Vanilla / Level15 (existing) / TEMFaithful (post-fix).\n")
print("Decision rules (lm200 OOD T=512 mean):")
print("- Level15Beta ≈ Level15 (~0.82): dilution hypothesis wrong, β alone insufficient")
print("- Level15Beta moves toward TEMFaithful (~0.97): β captures most of the gap")
print("- Level15Beta >> Level15: dilution was a major dilution effect\n")

for cfg_tag in ["clean", "noise", "lm200"]:
    n_lm = 200 if cfg_tag == "lm200" else 0
    eval_noise = 0.10 if cfg_tag == "noise" else 0.0
    env_ood = GridWorld(size=64, n_obs_types=16, p_empty=0.5, n_landmarks=n_lm, seed=1000)
    print(f"## {cfg_tag}\n")
    print("| Variant | T=128 OOD | T=512 OOD | T=512 NLL | n |")
    print("|---|---|---|---|---|")
    for variant in ["Vanilla", "Level15", "Level15Beta", "TEMFaithful"]:
        a128s, a512s, n512s = [], [], []
        for s in [0, 1, 2]:
            ckpt = Path(f"mapformer/runs/{variant}_{cfg_tag}/seed{s}/{variant}.pt")
            if not ckpt.exists(): continue
            m = build(variant, ckpt)
            a128, _ = eval_revisit(m, env_ood, 128, 200, seed=2000+s, p_action_noise=eval_noise)
            a512, n512 = eval_revisit(m, env_ood, 512, 100, seed=2000+s, p_action_noise=eval_noise)
            if a128 is not None: a128s.append(a128); a512s.append(a512); n512s.append(n512)
            del m; torch.cuda.empty_cache()
        if not a128s:
            print(f"| {variant} | — | — | — | 0 |"); continue
        print(f"| **{variant}** | "
              f"{np.mean(a128s):.3f} ± {np.std(a128s):.3f} | "
              f"{np.mean(a512s):.3f} ± {np.std(a512s):.3f} | "
              f"{np.mean(n512s):.3f} | {len(a128s)} |")
    print()

# Show learned β values per config (probe what the model decided to do)
print("## Learned β values per config / seed\n")
print("| cfg | seed | β = exp(log_β) |")
print("|---|---|---|")
for cfg_tag in ["clean", "noise", "lm200"]:
    for s in [0, 1, 2]:
        ckpt = Path(f"mapformer/runs/Level15Beta_{cfg_tag}/seed{s}/Level15Beta.pt")
        if not ckpt.exists(): continue
        m = build("Level15Beta", ckpt)
        beta = float(m.layers[0].log_beta.exp().item())
        print(f"| {cfg_tag} | {s} | {beta:.4f} |")
        del m; torch.cuda.empty_cache()
print("\nReference: 1/sqrt(d_head=64) = 0.125 (the standard transformer init).")

print("\n*Auto-generated by run_level15_beta.sh*\n")
PYEOF

cd "$REPO"
git pull --rebase 2>&1 | tail -3
git add LEVEL15BETA_RESULTS.md model_inekf_level15_beta.py train_variant.py run_level15_beta.sh
git add runs/Level15Beta_clean/seed0/*.pt runs/Level15Beta_clean/seed1/*.pt runs/Level15Beta_clean/seed2/*.pt \
        runs/Level15Beta_noise/seed0/*.pt runs/Level15Beta_noise/seed1/*.pt runs/Level15Beta_noise/seed2/*.pt \
        runs/Level15Beta_lm200/seed0/*.pt runs/Level15Beta_lm200/seed1/*.pt runs/Level15Beta_lm200/seed2/*.pt \
        2>/dev/null || true
git commit -m "Level15Beta: learnable softmax temperature in attention (multi-seed)

Tests the dilution hypothesis: does MapFormer's existing attention
recover TEMFaithful's lm200 win just by sharpening the softmax?

One scalar per attention layer (log_β, init at 1/sqrt(d_head)). No
hardcoded action/obs mask. No domain knowledge injected. Just gives
the model a knob to control how sharp its existing Hopfield-equivalent
retrieval is.

Multi-seed comparison (n=3) vs Vanilla / Level15 / TEMFaithful in
LEVEL15BETA_RESULTS.md. Also reports the learned β values per config /
seed as a diagnostic.
" 2>&1 | tail -3
git push origin main 2>&1 | tail -3
echo "[$(date)] Done."
