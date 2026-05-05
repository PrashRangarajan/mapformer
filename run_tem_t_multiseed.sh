#!/bin/bash
# Multi-seed TEM-t to verify the surprising single-seed lm200 win
# (TEM-t 0.853 vs Level15 0.790 at OOD T=512). Trains seeds 1 and 2
# across all 3 configs (clean / noise / lm200) and produces a
# multi-seed comparison against existing Vanilla / Level15 multi-seed
# baselines from RESULTS_PAPER.md.
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

train() {
    local cfg=$1 lm=$2 noise=$3 seed=$4 gpu=$5
    mkdir -p "$REPO/runs/TEM_T_${cfg}/seed${seed}"
    CUDA_VISIBLE_DEVICES=$gpu python3 -u -m mapformer.train_variant \
        --variant TEM_T --seed $seed \
        --n-landmarks $lm --p-action-noise $noise \
        --epochs 50 --n-batches 156 \
        --device cuda \
        --output-dir mapformer/runs/TEM_T_${cfg}/seed${seed} \
        > "$LOGS/TEM_T_${cfg}_s${seed}.log" 2>&1
}

# Pair 1: clean s1 + noise s1
echo "[$(date)] Pair 1: TEM_T clean s1 (gpu0) + noise s1 (gpu1)"
train clean 0   0.0  1 0 &
P1=$!
train noise 0   0.10 1 1 &
P2=$!
wait $P1 $P2
# Pair 2: lm200 s1 + clean s2
echo "[$(date)] Pair 2: lm200 s1 (gpu0) + clean s2 (gpu1)"
train lm200 200 0.0  1 0 &
P3=$!
train clean 0   0.0  2 1 &
P4=$!
wait $P3 $P4
# Pair 3: noise s2 + lm200 s2
echo "[$(date)] Pair 3: noise s2 (gpu0) + lm200 s2 (gpu1)"
train noise 0   0.10 2 0 &
P5=$!
train lm200 200 0.0  2 1 &
P6=$!
wait $P5 $P6
echo "[$(date)] All 6 TEM_T multi-seed trainings done."

# Sanity: any NaN?
for cfg in clean noise lm200; do
    for seed in 1 2; do
        if grep -q 'nan' "$LOGS/TEM_T_${cfg}_s${seed}.log"; then
            echo "WARNING: TEM_T_${cfg} seed${seed} log contains NaN"
        fi
    done
done

echo "[$(date)] Multi-seed eval (TEM_T n=3 vs Level15 n=3 from existing checkpoints)..."
python3 -u <<'PYEOF' > "$REPO/TEM_T_MULTISEED.md" 2>"$LOGS/tem_t_ms_eval.err"
import torch, torch.nn.functional as F, numpy as np
from pathlib import Path
from mapformer.environment import GridWorld
from mapformer.model import MapFormerWM, MapFormerEM
from mapformer.model_inekf_level15 import MapFormerWM_Level15InEKF
from mapformer.model_inekf_level15_em import MapFormerEM_Level15InEKF
from mapformer.model_tem_t import TEM_T

VARIANT_CLS = {
    "Vanilla":   MapFormerWM,
    "Level15":   MapFormerWM_Level15InEKF,
    "VanillaEM": MapFormerEM,
    "Level15EM": MapFormerEM_Level15InEKF,
    "TEM_T":     TEM_T,
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

print("# TEM-t multi-seed verification (n=3) vs MapFormer\n")
print("Verifies the surprising single-seed lm200 result from TEM_T_RESULTS.md")
print("(TEM_T 0.853 vs Level15 0.790 at OOD T=512). If the multi-seed mean")
print("preserves the gap with non-overlapping seed ranges, the finding is real;")
print("if it collapses to within seed noise, single-seed was a fluke.\n")

# Variants: TEM_T at all 3 seeds + the relevant comparators at their available seeds.
# For Vanilla, Level15: we have multi-seed (RESULTS_PAPER.md, n=3) under runs/ as
# Vanilla_clean/seed{0,1,2}/Vanilla.pt etc.
# For VanillaEM, Level15EM: also have multi-seed.

def per_seed(variant, cfg_dir, seeds, env_ood, eval_noise, T, n_trials):
    """Return list of accuracies across the requested seeds."""
    accs = []; nlls = []
    for s in seeds:
        ckpt = Path(f"mapformer/runs/{variant}_{cfg_dir}/seed{s}/{variant}.pt")
        if not ckpt.exists():
            continue
        m = build(variant, ckpt)
        a, n = eval_revisit(m, env_ood, T, n_trials, seed=2000+s, p_action_noise=eval_noise)
        if a is not None:
            accs.append(a); nlls.append(n)
        del m; torch.cuda.empty_cache()
    return accs, nlls

variant_seeds = {
    "Vanilla":   [0, 1, 2],
    "Level15":   [0, 1, 2],
    "VanillaEM": [0, 1, 2],
    "Level15EM": [0, 1, 2],
    "TEM_T":     [0, 1, 2],
}

for cfg_tag in ["clean", "noise", "lm200"]:
    n_lm = 200 if cfg_tag == "lm200" else 0
    eval_noise = 0.10 if cfg_tag == "noise" else 0.0
    env_ood = GridWorld(size=64, n_obs_types=16, p_empty=0.5, n_landmarks=n_lm, seed=1000)
    print(f"## {cfg_tag}\n")
    print("| Variant | T=128 OOD | T=512 OOD | T=512 NLL | n |")
    print("|---|---|---|---|---|")
    for variant in ["Vanilla", "Level15", "VanillaEM", "Level15EM", "TEM_T"]:
        seeds = variant_seeds[variant]
        a128s, _ = per_seed(variant, cfg_tag, seeds, env_ood, eval_noise, 128, 200)
        a512s, n512s = per_seed(variant, cfg_tag, seeds, env_ood, eval_noise, 512, 100)
        if not a128s:
            print(f"| {variant} | (no ckpts) | — | — | 0 |"); continue
        print(f"| **{variant}** | "
              f"{np.mean(a128s):.3f} ± {np.std(a128s):.3f} | "
              f"{np.mean(a512s):.3f} ± {np.std(a512s):.3f} | "
              f"{np.mean(n512s):.3f} | {len(a128s)} |")
    print()

# Per-seed breakdown so the user can read variance directly
print("## Per-seed breakdown (TEM_T only)\n")
for cfg_tag in ["clean", "noise", "lm200"]:
    n_lm = 200 if cfg_tag == "lm200" else 0
    eval_noise = 0.10 if cfg_tag == "noise" else 0.0
    env_ood = GridWorld(size=64, n_obs_types=16, p_empty=0.5, n_landmarks=n_lm, seed=1000)
    print(f"### {cfg_tag}")
    print("| seed | T=128 OOD | T=512 OOD |")
    print("|---|---|---|")
    for s in [0, 1, 2]:
        ckpt = Path(f"mapformer/runs/TEM_T_{cfg_tag}/seed{s}/TEM_T.pt")
        if not ckpt.exists():
            print(f"| {s} | (no ckpt) | — |"); continue
        m = build("TEM_T", ckpt)
        a128, _ = eval_revisit(m, env_ood, 128, 200, seed=2000+s, p_action_noise=eval_noise)
        a512, _ = eval_revisit(m, env_ood, 512, 100, seed=2000+s, p_action_noise=eval_noise)
        print(f"| {s} | {a128:.3f} | {a512:.3f} |")
        del m; torch.cuda.empty_cache()
    print()

print("\n*Auto-generated by run_tem_t_multiseed.sh*\n")
PYEOF
echo "[$(date)] Eval done."
tail -50 "$REPO/TEM_T_MULTISEED.md"

cd "$REPO"
git pull --rebase 2>&1 | tail -3
git add TEM_T_MULTISEED.md run_tem_t_multiseed.sh
git add runs/TEM_T_clean/seed1/*.pt runs/TEM_T_clean/seed2/*.pt \
        runs/TEM_T_noise/seed1/*.pt runs/TEM_T_noise/seed2/*.pt \
        runs/TEM_T_lm200/seed1/*.pt runs/TEM_T_lm200/seed2/*.pt 2>/dev/null || true
git commit -m "TEM-t multi-seed verification (seeds 1, 2 + multi-seed eval)

Tests whether the single-seed TEM-t lm200 win (0.853 vs Level15 0.790
at OOD T=512) holds with multi-seed verification. If the mean across
n=3 seeds preserves the gap with non-overlapping seed ranges, the
'TEM-t beats MapFormer-EM on landmark exploitation' finding is real
and publishable. If it collapses to seed noise, the single-seed was
a fluke and we drop the claim.

TEM_T_MULTISEED.md: full n=3 comparison vs Vanilla / Level15 /
VanillaEM / Level15EM (all of which have existing multi-seed
checkpoints) plus per-seed breakdown for TEM-t.
" 2>&1 | tail -3
git push origin main 2>&1 | tail -3
echo "[$(date)] Done."
