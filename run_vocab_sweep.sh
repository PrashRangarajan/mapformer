#!/bin/bash
# Vocab × correction × backbone sweep.
# Tests whether our `Level15-WM > Level15-EM` ordering on small-vocab tasks
# survives along the paper's vocab-scaling axis (Figure 4c) where EM was
# claimed to dominate. Single seed for scoping; multi-seed if interesting.
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

# Training: clean (no landmarks, no noise), 1 seed, 50 epochs.
train() {
    local variant=$1 nobs=$2 gpu=$3
    local tag="vocab${nobs}"
    mkdir -p "$REPO/runs/${variant}_${tag}/seed0"
    CUDA_VISIBLE_DEVICES=$gpu python3 -u -m mapformer.train_variant \
        --variant $variant --seed 0 \
        --n-landmarks 0 --p-action-noise 0.0 \
        --n-obs-types $nobs \
        --epochs 50 --n-batches 156 \
        --device cuda \
        --output-dir mapformer/runs/${variant}_${tag}/seed0 \
        > "$LOGS/${variant}_${tag}_s0.log" 2>&1
}

run_pair() { train $1 $2 0 & P1=$!; train $3 $4 1 & P2=$!; wait $P1 $P2; }

# n_obs = 256
echo "[$(date)] [n_obs=256] Vanilla + VanillaEM"
run_pair Vanilla 256  VanillaEM 256
echo "[$(date)] [n_obs=256] Level15 + Level15EM"
run_pair Level15 256  Level15EM 256

# n_obs = 4096
echo "[$(date)] [n_obs=4096] Vanilla + VanillaEM"
run_pair Vanilla 4096  VanillaEM 4096
echo "[$(date)] [n_obs=4096] Level15 + Level15EM"
run_pair Level15 4096  Level15EM 4096

echo "[$(date)] All vocab-sweep trainings done. Final losses:"
for tag in vocab256 vocab4096; do
    for v in Vanilla VanillaEM Level15 Level15EM; do
        f="$LOGS/${v}_${tag}_s0.log"
        [ -f "$f" ] || continue
        echo "  ${v} ${tag}: $(grep 'Epoch  50/50' $f | tail -1)"
    done
done

# Eval — including the existing n_obs=16 results for comparison.
echo "[$(date)] Eval vocab sweep (clean OOD at T=128 and T=512)..."
python3 -u <<'PYEOF' > "$REPO/VOCAB_SWEEP_RESULTS.md" 2>"$LOGS/vocab_sweep_eval.err"
import torch, torch.nn.functional as F, numpy as np
from pathlib import Path
from mapformer.environment import GridWorld
from mapformer.model import MapFormerWM, MapFormerEM
from mapformer.model_inekf_level15 import MapFormerWM_Level15InEKF
from mapformer.model_inekf_level15_em import MapFormerEM_Level15InEKF

VARIANT_CLS = {
    "Vanilla":    MapFormerWM,
    "VanillaEM":  MapFormerEM,
    "Level15":    MapFormerWM_Level15InEKF,
    "Level15EM":  MapFormerEM_Level15InEKF,
}

def build(variant, ckpt):
    c = torch.load(ckpt, map_location="cuda", weights_only=False)
    cfg = c.get("config", {})
    cls = VARIANT_CLS[variant]
    m = cls(vocab_size=cfg["vocab_size"], d_model=cfg.get("d_model", 128),
            n_heads=cfg.get("n_heads", 2), n_layers=cfg.get("n_layers", 1),
            grid_size=cfg.get("grid_size", 64))
    m.load_state_dict(c["model_state_dict"])
    return m.cuda().eval(), cfg

def eval_clean(model, env, T, n_trials, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    c = tot = 0; nll_sum = 0.0
    with torch.no_grad():
        for _ in range(n_trials):
            tokens, _, rm = env.generate_trajectory(T)
            tt = tokens.unsqueeze(0).cuda()
            try: logits = model(tt[:, :-1])
            except Exception: return None, None
            lp = F.log_softmax(logits, dim=-1)
            preds = lp.argmax(-1)[0]; tgts = tt[0, 1:]; mask = rm[1:].cuda()
            if mask.sum() == 0: continue
            c += (preds[mask] == tgts[mask]).sum().item()
            tot += mask.sum().item()
            idx = torch.arange(lp.shape[1], device="cuda")[mask]
            nll_sum += -lp[0, idx, tgts[mask]].sum().item()
    return (c / tot if tot else None, nll_sum / tot if tot else None)

print("# Vocab × correction × backbone sweep — does Level15-WM > Level15-EM survive paper's vocab-scaling axis?\n")
print("Tests whether our small-vocab finding (Level15-WM beats Level15-EM on")
print("landmarks/noise/OOD-length) survives along the paper's Figure 4c axis")
print("where EM was claimed to dominate (large vocab + short sequences).\n")
print("Single seed (seed=0). Clean task (no landmarks, no noise). Train at")
print("T=128, eval at T=128 IID and T=512 OOD on a fresh obs_map (seed=1000).\n")
print("Mechanism story (multiplicative AND-gate vs additive):")
print("- short l + large vocab: A_P sharp, A_X bottleneck → EM filters cleanly → EM wins (paper Fig 4c)")
print("- long l + any vocab: A_P degrades from drift → EM AND-gate kills retrieval → WM wins (us)")
print("- our l=128 train + T=512 eval is in the second regime; question is whether vocab pulls it back\n")

for nobs in [16, 256, 4096]:
    tag = "clean" if nobs == 16 else f"vocab{nobs}"
    env_ood = GridWorld(size=64, n_obs_types=nobs, p_empty=0.5, n_landmarks=0, seed=1000)
    print(f"## n_obs = {nobs}\n")
    print("| Variant | T=128 IID | T=512 OOD | T=512 NLL |")
    print("|---|---|---|---|")
    for variant in ["Vanilla", "VanillaEM", "Level15", "Level15EM"]:
        ckpt = Path(f"mapformer/runs/{variant}_{tag}/seed0/{variant}.pt")
        if not ckpt.exists():
            print(f"| {variant} | — | — | — |"); continue
        m, _ = build(variant, ckpt)
        a128, _ = eval_clean(m, env_ood, 128, 200, seed=2000)
        a512, n512 = eval_clean(m, env_ood, 512, 100, seed=2000)
        if a128 is None:
            print(f"| {variant} | crash | crash | crash |")
        else:
            print(f"| **{variant}** | {a128:.3f} | {a512:.3f} | {n512:.3f} |")
        del m; torch.cuda.empty_cache()
    print()

print("## Interpretation\n")
print("Read across rows for the same n_obs to see backbone effect; read down")
print("variant rows across n_obs panels to see scaling effect:")
print("- If Level15 > Level15EM at all three n_obs: drift-dominated regime;")
print("  paper's vocab-scaling claim does not invert under correction at l=128/512.")
print("- If Level15EM > Level15 at n_obs=4096: paper's factorization argument")
print("  reasserts at large vocab, even with correction. Backbone choice is")
print("  vocab-dependent.")
print("- VanillaEM rows show whether the backbone effect requires correction")
print("  to manifest, or is intrinsic to the architecture.\n")
print("*Auto-generated by run_vocab_sweep.sh, single seed (seed=0)*\n")
PYEOF

cd "$REPO"
git pull --rebase 2>&1 | tail -3
git add VOCAB_SWEEP_RESULTS.md run_vocab_sweep.sh train_variant.py
for tag in vocab256 vocab4096; do
    for v in Vanilla VanillaEM Level15 Level15EM; do
        git add runs/${v}_${tag}/seed0/${v}.pt 2>/dev/null || true
    done
done
git commit -m "Vocab × correction × backbone sweep (single seed, clean task)

Tests whether our 'Level15-WM > Level15-EM' ordering on small-vocab
landmark/noise tasks survives along the paper's vocab-scaling axis
(Figure 4c) where MapFormer-EM was claimed to dominate.

Single-seed scoping run at n_obs ∈ {16, 256, 4096}, clean task,
T=128 train / T=512 OOD. Result in VOCAB_SWEEP_RESULTS.md. Multi-seed
follow-up if pattern is interesting.
" 2>&1 | tail -3
git push origin main 2>&1 | tail -3
echo "[$(date)] Done."
