#!/bin/bash
# Disambiguation: was Level15Beta's lm200 win from β or from removed
# post-attention residual dropout?
# Train Level15NoDrop (no β change, only dropout removed) on lm200,
# 3 seeds. Compare to Level15Beta (~0.93) and Level15 (~0.82).
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
    local seed=$1 gpu=$2
    mkdir -p "$REPO/runs/Level15NoDrop_lm200/seed${seed}"
    CUDA_VISIBLE_DEVICES=$gpu python3 -u -m mapformer.train_variant \
        --variant Level15NoDrop --seed $seed \
        --n-landmarks 200 --p-action-noise 0.0 \
        --epochs 50 --n-batches 156 \
        --device cuda \
        --output-dir mapformer/runs/Level15NoDrop_lm200/seed${seed} \
        > "$LOGS/Level15NoDrop_lm200_s${seed}.log" 2>&1
}

echo "[$(date)] [1] s0 + s1"
train 0 0 & P1=$!
train 1 1 & P2=$!
wait $P1 $P2
echo "[$(date)] [2] s2"
train 2 0
echo "[$(date)] All Level15NoDrop trainings done."

for seed in 0 1 2; do
    f="$LOGS/Level15NoDrop_lm200_s${seed}.log"
    [ -f "$f" ] || continue
    echo "  s${seed}: $(grep 'Epoch  50/50' $f | tail -1)"
done

echo "[$(date)] Eval Level15NoDrop vs Level15 vs Level15Beta vs TEMFaithful (lm200)..."
python3 -u <<'PYEOF' > "$REPO/DROPOUT_ABLATION_RESULTS.md" 2>"$LOGS/dropout_ablation_eval.err"
import torch, torch.nn.functional as F, numpy as np
from pathlib import Path
from mapformer.environment import GridWorld
from mapformer.model import MapFormerWM
from mapformer.model_inekf_level15 import MapFormerWM_Level15InEKF
from mapformer.model_inekf_level15_beta import MapFormerWM_Level15Beta
from mapformer.model_inekf_level15_nodrop import MapFormerWM_Level15NoDrop
from mapformer.model_tem_faithful import TEMFaithful

VARIANT_CLS = {
    "Vanilla":        MapFormerWM,
    "Level15":        MapFormerWM_Level15InEKF,
    "Level15Beta":    MapFormerWM_Level15Beta,
    "Level15NoDrop":  MapFormerWM_Level15NoDrop,
    "TEMFaithful":    TEMFaithful,
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

def eval_revisit(model, env, T, n_trials, seed):
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

print("# Dropout ablation — was Level15Beta's lm200 win from β or from removed dropout?\n")
print("Level15Beta differed from Level15 in two ways: (a) learnable β temperature")
print("(but learned values barely moved from init), and (b) post-attention residual")
print("dropout removed. This isolates (b): Level15NoDrop has fixed β = 1/sqrt(d_head)")
print("and only the dropout removed. If it matches Level15Beta on lm200, the win was")
print("dropout. If it matches Level15, β was load-bearing.\n")

env_ood = GridWorld(size=64, n_obs_types=16, p_empty=0.5, n_landmarks=200, seed=1000)
print("## lm200 OOD\n")
print("| Variant | T=128 OOD | T=512 OOD | T=512 NLL | n |")
print("|---|---|---|---|---|")
for variant in ["Vanilla", "Level15", "Level15NoDrop", "Level15Beta", "TEMFaithful"]:
    a128s, a512s, n512s = [], [], []
    for s in [0, 1, 2]:
        ckpt = Path(f"mapformer/runs/{variant}_lm200/seed{s}/{variant}.pt")
        if not ckpt.exists(): continue
        m = build(variant, ckpt)
        a128, _ = eval_revisit(m, env_ood, 128, 200, seed=2000+s)
        a512, n512 = eval_revisit(m, env_ood, 512, 100, seed=2000+s)
        if a128 is not None: a128s.append(a128); a512s.append(a512); n512s.append(n512)
        del m; torch.cuda.empty_cache()
    if not a128s: print(f"| {variant} | — | — | — | 0 |"); continue
    print(f"| **{variant}** | {np.mean(a128s):.3f} ± {np.std(a128s):.3f} | "
          f"{np.mean(a512s):.3f} ± {np.std(a512s):.3f} | "
          f"{np.mean(n512s):.3f} | {len(a128s)} |")
print()

print("## Interpretation\n")
print("- Level15NoDrop ≈ Level15Beta (~0.93): the +12pp win was from dropout removal.")
print("  β was a red herring; the architectural change that mattered was preserving")
print("  attention output features through the residual.")
print("- Level15NoDrop ≈ Level15 (~0.82): β was load-bearing despite barely moving")
print("  from init; small β changes have outsized effects on landmark retrieval.")
print("- Intermediate: both contributed; need a 4th cell (Level15Beta with dropout")
print("  restored) to fully decompose.")
print("\n*Auto-generated by run_dropout_ablation.sh*\n")
PYEOF

cd "$REPO"
git pull --rebase 2>&1 | tail -3
git add DROPOUT_ABLATION_RESULTS.md model_inekf_level15_nodrop.py train_variant.py run_dropout_ablation.sh
git add runs/Level15NoDrop_lm200/seed0/*.pt runs/Level15NoDrop_lm200/seed1/*.pt runs/Level15NoDrop_lm200/seed2/*.pt 2>/dev/null || true
git commit -m "Dropout ablation: isolate β vs post-attn dropout in Level15Beta lm200 win

Level15Beta closed +12pp on lm200 OOD T=512 vs Level15, but learned β
barely moved from init (0.15 vs 0.125). The other architectural diff
between the two layer classes was that Level15Beta dropped self.dropout
on the post-attention residual add. This run isolates that: Level15NoDrop
has fixed β = 1/sqrt(d_head) and only the dropout removed.

Result in DROPOUT_ABLATION_RESULTS.md.
" 2>&1 | tail -3
git push origin main 2>&1 | tail -3
echo "[$(date)] Done."
