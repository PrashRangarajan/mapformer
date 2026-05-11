#!/bin/bash
# GSF / GSF+NoDrop on clean + noise, multi-seed (n=3).
# Completes the GSF picture: is multi-modal Bayes only useful for landmarks,
# or does it also help on aliased-only / noise regimes?
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

echo "[$(date)] [gsf-cn] Polling for free GPUs..."
while ! is_gpu_free; do sleep 60; done
echo "[$(date)] [gsf-cn] GPUs free."

train() {
    local variant=$1 cfg=$2 noise=$3 seed=$4 gpu=$5
    mkdir -p "$REPO/runs/${variant}_${cfg}/seed${seed}"
    CUDA_VISIBLE_DEVICES=$gpu python3 -u -m mapformer.train_variant \
        --variant $variant --seed $seed \
        --n-landmarks 0 --p-action-noise $noise \
        --epochs 50 --n-batches 156 \
        --device cuda \
        --output-dir mapformer/runs/${variant}_${cfg}/seed${seed} \
        > "$LOGS/${variant}_${cfg}_s${seed}.log" 2>&1
}
run_pair() { train $1 $2 $3 $4 0 & P1=$!; train $5 $6 $7 $8 1 & P2=$!; wait $P1 $P2; }

# Level15GSF clean × 3 seeds + Level15GSF noise × 3 seeds
echo "[$(date)] [gsf-cn] Level15GSF clean s0 + s1"
run_pair Level15GSF clean 0.0 0  Level15GSF clean 0.0 1
echo "[$(date)] [gsf-cn] Level15GSF clean s2 + noise s0"
run_pair Level15GSF clean 0.0 2  Level15GSF noise 0.10 0
echo "[$(date)] [gsf-cn] Level15GSF noise s1 + s2"
run_pair Level15GSF noise 0.10 1  Level15GSF noise 0.10 2

# Level15GSF_NoDrop clean × 3 + noise × 3
echo "[$(date)] [gsf-cn] Level15GSF_NoDrop clean s0 + s1"
run_pair Level15GSF_NoDrop clean 0.0 0  Level15GSF_NoDrop clean 0.0 1
echo "[$(date)] [gsf-cn] Level15GSF_NoDrop clean s2 + noise s0"
run_pair Level15GSF_NoDrop clean 0.0 2  Level15GSF_NoDrop noise 0.10 0
echo "[$(date)] [gsf-cn] Level15GSF_NoDrop noise s1 + s2"
run_pair Level15GSF_NoDrop noise 0.10 1  Level15GSF_NoDrop noise 0.10 2

echo "[$(date)] [gsf-cn] All trainings done."

# Aggregate full table
python3 -u <<'PYEOF' > "$REPO/GSF_FULL_RESULTS.md" 2>"$LOGS/gsf_cn_eval.err"
import torch, torch.nn.functional as F, numpy as np
from pathlib import Path
from mapformer.environment import GridWorld
from mapformer.model import MapFormerWM
from mapformer.model_inekf_level15 import MapFormerWM_Level15InEKF
from mapformer.model_inekf_level15_nodrop import MapFormerWM_Level15NoDrop
from mapformer.model_inekf_gsf import MapFormerWM_Level15GSF
from mapformer.model_inekf_gsf_nodrop import MapFormerWM_Level15GSF_NoDrop
from mapformer.model_tem_faithful import TEMFaithful

VARIANT_CLS = {
    "Vanilla": MapFormerWM,
    "Level15": MapFormerWM_Level15InEKF,
    "Level15NoDrop": MapFormerWM_Level15NoDrop,
    "Level15GSF": MapFormerWM_Level15GSF,
    "Level15GSF_NoDrop": MapFormerWM_Level15GSF_NoDrop,
    "TEMFaithful": TEMFaithful,
}

def build(v, ckpt):
    c = torch.load(ckpt, map_location="cuda", weights_only=False)
    cfg = c["config"]; cls = VARIANT_CLS[v]
    kw = dict(vocab_size=cfg["vocab_size"], d_model=cfg["d_model"],
              n_heads=cfg["n_heads"], n_layers=cfg["n_layers"], grid_size=cfg["grid_size"])
    if v in ("Level15GSF", "Level15GSF_NoDrop"): kw["n_modes"] = 8
    m = cls(**kw); m.load_state_dict(c["model_state_dict"])
    return m.cuda().eval()

def eval_revisit(model, env, T, n_trials, seed, p_noise=0.0):
    torch.manual_seed(seed); np.random.seed(seed)
    c = tot = 0; nll = 0.0
    with torch.no_grad():
        for _ in range(n_trials):
            tokens, _, rm = env.generate_trajectory(T)
            tt = tokens.unsqueeze(0).cuda()
            if p_noise > 0:
                even = torch.zeros_like(tt, dtype=torch.bool); even[:, 0::2] = True
                noise = (torch.rand_like(tt, dtype=torch.float) < p_noise) & even
                rand = torch.randint(0, env.N_ACTIONS, tt.shape, device="cuda")
                tt = torch.where(noise, rand, tt)
            try: logits = model(tt[:, :-1])
            except Exception: return None, None
            lp = F.log_softmax(logits, dim=-1)
            preds = lp.argmax(-1)[0]; tgts = tt[0, 1:]; mask = rm[1:].cuda()
            if mask.sum() == 0: continue
            c += (preds[mask] == tgts[mask]).sum().item(); tot += mask.sum().item()
            idx = torch.arange(lp.shape[1], device="cuda")[mask]
            nll += -lp[0, idx, tgts[mask]].sum().item()
    return (c / tot if tot else None, nll / tot if tot else None)

print("# GSF complete picture — does multi-modal Bayes help beyond landmarks?\n")
print("Multi-seed (n=3) on clean / noise / lm200. Question: is GSF's win a")
print("landmark-specific phenomenon, or does it help in other regimes too?\n")
print("**Prediction (going in):** GSF marginal on clean (Level15 already at ceiling);")
print("modest on noise (drift uncertainty is unimodal-wide, not multi-modal-discrete);")
print("substantial on lm200 (already confirmed). Test the prediction.\n")

for cfg, lm, en in [("clean", 0, 0.0), ("noise", 0, 0.10), ("lm200", 200, 0.0)]:
    env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, n_landmarks=lm, seed=1000)
    print(f"## {cfg}\n| Variant | T=128 OOD | T=512 OOD | T=512 NLL | n |")
    print("|---|---|---|---|---|")
    for v in ["Vanilla", "Level15", "Level15NoDrop", "Level15GSF", "Level15GSF_NoDrop", "TEMFaithful"]:
        a128s, a512s, n512s = [], [], []
        for s in [0, 1, 2]:
            ckpt = Path(f"mapformer/runs/{v}_{cfg}/seed{s}/{v}.pt")
            if not ckpt.exists(): continue
            try: m = build(v, ckpt)
            except Exception: continue
            a128, _ = eval_revisit(m, env, 128, 100, seed=2000+s, p_noise=en)
            a512, n512 = eval_revisit(m, env, 512, 50, seed=2000+s, p_noise=en)
            if a128 is not None: a128s.append(a128); a512s.append(a512); n512s.append(n512)
            del m; torch.cuda.empty_cache()
        if not a128s: print(f"| {v} | — | — | — | 0 |"); continue
        print(f"| **{v}** | {np.mean(a128s):.3f} ± {np.std(a128s):.3f} | "
              f"{np.mean(a512s):.3f} ± {np.std(a512s):.3f} | "
              f"{np.mean(n512s):.3f} | {len(a128s)} |")
    print()
print("*Auto-generated by run_gsf_clean_noise.sh*\n")
PYEOF

cd "$REPO"
git pull --rebase 2>&1 | tail -3
git add GSF_FULL_RESULTS.md run_gsf_clean_noise.sh
for cfg in clean noise; do
    for v in Level15GSF Level15GSF_NoDrop; do
        git add runs/${v}_${cfg}/seed0/${v}.pt runs/${v}_${cfg}/seed1/${v}.pt runs/${v}_${cfg}/seed2/${v}.pt 2>/dev/null || true
    done
done
git commit -m "GSF on clean + noise (multi-seed): complete the regime picture

Tests whether multi-modal Bayesian filtering helps outside the landmark
regime where it was originally motivated. Multi-seed (n=3) Level15GSF and
Level15GSF_NoDrop on clean and noise configs. Result in GSF_FULL_RESULTS.md.

Prediction: GSF marginal on clean (Level15 already at 0.993 ceiling),
modest on noise, big on lm200 (already confirmed). Test the prediction.
" 2>&1 | tail -3
git push origin main 2>&1 | tail -3
echo "[$(date)] [gsf-cn] Done."
