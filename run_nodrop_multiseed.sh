#!/bin/bash
# Level15NoDrop multi-seed on clean + noise (only lm200 was multi-seed before).
# Nail down the Pareto trade-off cleanly: does dropout removal hurt clean
# calibration (NLL) by enough to make it net-negative?
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

echo "[$(date)] [NoDrop] Polling for free GPUs..."
while ! is_gpu_free; do sleep 60; done
echo "[$(date)] [NoDrop] GPUs free."

train() {
    local cfg=$1 lm=$2 noise=$3 seed=$4 gpu=$5
    mkdir -p "$REPO/runs/Level15NoDrop_${cfg}/seed${seed}"
    CUDA_VISIBLE_DEVICES=$gpu python3 -u -m mapformer.train_variant \
        --variant Level15NoDrop --seed $seed \
        --n-landmarks $lm --p-action-noise $noise \
        --epochs 50 --n-batches 156 \
        --device cuda \
        --output-dir mapformer/runs/Level15NoDrop_${cfg}/seed${seed} \
        > "$LOGS/Level15NoDrop_${cfg}_s${seed}.log" 2>&1
}
run_pair() { train $1 $2 $3 $4 0 & P1=$!; train $5 $6 $7 $8 1 & P2=$!; wait $P1 $P2; }

# Clean + noise, 3 seeds each
echo "[$(date)] [NoDrop] clean s0 + s1"; run_pair clean 0 0.0 0  clean 0 0.0 1
echo "[$(date)] [NoDrop] clean s2 + noise s0"; run_pair clean 0 0.0 2  noise 0 0.10 0
echo "[$(date)] [NoDrop] noise s1 + s2"; run_pair noise 0 0.10 1  noise 0 0.10 2
echo "[$(date)] [NoDrop] All trainings done."

python3 -u <<'PYEOF' > "$REPO/NODROP_PARETO_RESULTS.md" 2>"$LOGS/nodrop_eval.err"
import torch, torch.nn.functional as F, numpy as np
from pathlib import Path
from mapformer.environment import GridWorld
from mapformer.model import MapFormerWM
from mapformer.model_inekf_level15 import MapFormerWM_Level15InEKF
from mapformer.model_inekf_level15_nodrop import MapFormerWM_Level15NoDrop

VARIANT_CLS = {
    "Vanilla": MapFormerWM,
    "Level15": MapFormerWM_Level15InEKF,
    "Level15NoDrop": MapFormerWM_Level15NoDrop,
}

def build(variant, ckpt):
    c = torch.load(ckpt, map_location="cuda", weights_only=False)
    cfg = c["config"]; cls = VARIANT_CLS[variant]
    m = cls(vocab_size=cfg["vocab_size"], d_model=cfg["d_model"],
            n_heads=cfg["n_heads"], n_layers=cfg["n_layers"], grid_size=cfg["grid_size"])
    m.load_state_dict(c["model_state_dict"])
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

print("# Level15NoDrop — multi-seed Pareto trade-off across regimes\n")
print("Removing the post-attn residual dropout was load-bearing for lm200")
print("(+12pp). On clean it costs NLL (calibration). This run gives n=3 on")
print("clean + noise so the Pareto picture is honest.\n")

for cfg, lm, eval_noise in [("clean", 0, 0.0), ("noise", 0, 0.10), ("lm200", 200, 0.0)]:
    env_ood = GridWorld(size=64, n_obs_types=16, p_empty=0.5, n_landmarks=lm, seed=1000)
    print(f"## {cfg}\n| Variant | T=128 OOD | T=512 OOD | T=512 NLL | n |")
    print("|---|---|---|---|---|")
    for v in ["Vanilla", "Level15", "Level15NoDrop"]:
        a128s, a512s, n512s = [], [], []
        for s in [0,1,2]:
            ckpt = Path(f"mapformer/runs/{v}_{cfg}/seed{s}/{v}.pt")
            if not ckpt.exists(): continue
            m = build(v, ckpt)
            a128, _ = eval_revisit(m, env_ood, 128, 200, seed=2000+s, p_noise=eval_noise)
            a512, n512 = eval_revisit(m, env_ood, 512, 100, seed=2000+s, p_noise=eval_noise)
            if a128 is not None: a128s.append(a128); a512s.append(a512); n512s.append(n512)
            del m; torch.cuda.empty_cache()
        if not a128s: print(f"| {v} | — | — | — | 0 |"); continue
        print(f"| **{v}** | {np.mean(a128s):.3f} ± {np.std(a128s):.3f} | "
              f"{np.mean(a512s):.3f} ± {np.std(a512s):.3f} | "
              f"{np.mean(n512s):.3f} | {len(a128s)} |")
    print()

print("\n*Auto-generated by run_nodrop_multiseed.sh*\n")
PYEOF

cd "$REPO"
git pull --rebase 2>&1 | tail -3
git add NODROP_PARETO_RESULTS.md run_nodrop_multiseed.sh
git add runs/Level15NoDrop_clean/seed*/Level15NoDrop.pt runs/Level15NoDrop_noise/seed*/Level15NoDrop.pt 2>/dev/null || true
git commit -m "Level15NoDrop multi-seed Pareto trade-off (clean + noise)

Multi-seed (n=3) for Level15NoDrop on clean + noise to nail down the
Pareto picture. lm200 already had n=3 from the dropout ablation.
Result in NODROP_PARETO_RESULTS.md.
" 2>&1 | tail -3
git push origin main 2>&1 | tail -3
echo "[$(date)] [NoDrop] Done."
