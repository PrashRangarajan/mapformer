#!/bin/bash
# TEM-t (Whittington 2022) baseline on torus, parameter-matched to MapFormer-EM.
# This is the empirical comparison the MapFormer paper claims structurally
# but never benchmarks. TEM-t and MapFormer-EM differ only in:
#   - sequential per-action W_a (TEM-t) vs parallel cumsum f_Δ(x) (MapFormer-EM)
#   - Q=K from positions only / V from stimuli only (TEM-t) vs
#     Hadamard A_X⊙A_P with separate Q_c,K_c,V (MapFormer-EM)
# Same total parameter scale (~250K).
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
    local cfg=$1 lm=$2 noise=$3 gpu=$4
    mkdir -p "$REPO/runs/TEM_T_${cfg}/seed0"
    CUDA_VISIBLE_DEVICES=$gpu python3 -u -m mapformer.train_variant \
        --variant TEM_T --seed 0 \
        --n-landmarks $lm --p-action-noise $noise \
        --epochs 50 --n-batches 156 \
        --device cuda \
        --output-dir mapformer/runs/TEM_T_${cfg}/seed0 \
        > "$LOGS/TEM_T_${cfg}_s0.log" 2>&1
}

echo "[$(date)] TEM_T on torus, all 3 configs (parallel pairs across both GPUs)..."
train clean 0   0.0  0 &
P1=$!
train noise 0   0.10 1 &
P2=$!
wait $P1 $P2
echo "[$(date)] Pair 1 (clean + noise) done."
train lm200 200 0.0 0
echo "[$(date)] All TEM_T trainings done."

# Sanity: any NaN in logs?
for cfg in clean noise lm200; do
    grep 'Epoch  50/50' "$LOGS/TEM_T_${cfg}_s0.log" | tail -1
    if grep -q 'nan' "$LOGS/TEM_T_${cfg}_s0.log"; then
        echo "WARNING: TEM_T_${cfg} log contains NaN — investigate."
    fi
done

echo "[$(date)] Eval TEM_T vs MapFormer-EM, Level15-EM, TEM, TEMFaithful, Vanilla, Level15..."
python3 -u <<'PYEOF' > "$REPO/TEM_T_RESULTS.md" 2>"$LOGS/tem_t_eval.err"
import torch, torch.nn.functional as F, numpy as np
from pathlib import Path
from mapformer.environment import GridWorld
from mapformer.model import MapFormerWM, MapFormerEM
from mapformer.model_inekf_level15 import MapFormerWM_Level15InEKF
from mapformer.model_inekf_level15_em import MapFormerEM_Level15InEKF
from mapformer.model_tem import TEMRecurrent
from mapformer.model_tem_faithful import TEMFaithful
from mapformer.model_tem_t import TEM_T

VARIANT_CLS = {
    "Vanilla":     MapFormerWM,
    "VanillaEM":   MapFormerEM,
    "Level15":     MapFormerWM_Level15InEKF,
    "Level15EM":   MapFormerEM_Level15InEKF,
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

print("# TEM-t baseline vs MapFormer (full comparison)\n")
print("TEM-t = Whittington 2022 transformer formulation of TEM. Parameter-")
print("matched to MapFormer-EM (~250K total). Differs from MapFormer-EM only")
print("in the position-encoding mechanism: sequential `e_t = ReLU(e_{t-1} W_{a_t})`")
print("vs parallel `θ_t = ω · cumsum(f_Δ(x_t))`. This is the empirical comparison")
print("the MapFormer paper claims structurally but never benchmarks.\n")
print("Single-seed runs across clean / noise / lm200 regimes, evaluated at OOD T=128 and T=512.\n")

ckpts = {
    "clean": {
        "Vanilla":     "mapformer/runs/Vanilla_clean/seed0/Vanilla.pt",
        "Level15":     "mapformer/runs/Level15_clean/seed0/Level15.pt",
        "VanillaEM":   "mapformer/runs/VanillaEM_clean/seed0/VanillaEM.pt",
        "Level15EM":   "mapformer/runs/Level15EM_clean/seed0/Level15EM.pt",
        "TEM":         "mapformer/runs/TEM_clean/seed0/TEM.pt",
        "TEMFaithful": "mapformer/runs/TEMFaithful_clean/seed0/TEMFaithful.pt",
        "TEM_T":       "mapformer/runs/TEM_T_clean/seed0/TEM_T.pt",
    },
    "noise": {
        "Vanilla":     "mapformer/runs/Vanilla_noise/seed0/Vanilla.pt",
        "Level15":     "mapformer/runs/Level15_noise/seed0/Level15.pt",
        "VanillaEM":   "mapformer/runs/VanillaEM_noise/seed0/VanillaEM.pt",
        "Level15EM":   "mapformer/runs/Level15EM_noise/seed0/Level15EM.pt",
        "TEM":         "mapformer/runs/TEM_noise/seed0/TEM.pt",
        "TEMFaithful": "mapformer/runs/TEMFaithful_noise/seed0/TEMFaithful.pt",
        "TEM_T":       "mapformer/runs/TEM_T_noise/seed0/TEM_T.pt",
    },
    "lm200": {
        "Vanilla":     "mapformer/runs/Vanilla_lm200/seed0/Vanilla.pt",
        "Level15":     "mapformer/runs/Level15_lm200/seed0/Level15.pt",
        "VanillaEM":   "mapformer/runs/VanillaEM_lm200/seed0/VanillaEM.pt",
        "Level15EM":   "mapformer/runs/Level15EM_lm200/seed0/Level15EM.pt",
        "TEM":         "mapformer/runs/TEM_lm200/seed0/TEM.pt",
        "TEMFaithful": "mapformer/runs/TEMFaithful_lm200/seed0/TEMFaithful.pt",
        "TEM_T":       "mapformer/runs/TEM_T_lm200/seed0/TEM_T.pt",
    },
}

for cfg_tag in ["clean", "noise", "lm200"]:
    n_lm = 200 if cfg_tag == "lm200" else 0
    eval_noise = 0.10 if cfg_tag == "noise" else 0.0
    print(f"## {cfg_tag}\n")
    print("| Variant | T=128 OOD | T=512 OOD | T=128 NLL | T=512 NLL |")
    print("|---|---|---|---|---|")
    for v in ["Vanilla", "Level15", "VanillaEM", "Level15EM", "TEM", "TEMFaithful", "TEM_T"]:
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

print("\n*Auto-generated by run_tem_t.sh*\n")
PYEOF
echo "[$(date)] Eval done."
tail -40 "$REPO/TEM_T_RESULTS.md"

cd "$REPO"
git pull --rebase 2>&1 | tail -3
git add TEM_T_RESULTS.md model_tem_t.py train_variant.py run_tem_t.sh
git add runs/TEM_T_clean/seed0/*.pt \
        runs/TEM_T_noise/seed0/*.pt \
        runs/TEM_T_lm200/seed0/*.pt 2>/dev/null || true
git commit -m "TEM-t baseline: parameter-matched comparison vs MapFormer-EM

Adds model_tem_t.py with TEM_T — a faithful implementation of
Whittington et al. 2022 (ICLR), the transformer formulation of TEM.
The three modifications in the original paper:
  1. Q = K = E·W_e (position-only) and V = X·W_x (stimulus-only)
  2. Causal attention over past (e_t, x_t) pairs
  3. Recurrent position encoding e_{t+1} = ReLU(e_t · W_a) with
     per-action transition matrix W_a

Single-environment training (TEM-t's headline advantage is multi-env
compositional generalisation; that's a separate experiment).

This is the parameter-matched apples-to-apples comparison the
MapFormer paper claims structurally but never empirically benchmarks.
TEM_T (~250K params) and MapFormer-EM (~200K) differ only in the
position-encoding mechanism: sequential per-action W_a vs parallel
cumsum f_Δ(x). Same transformer scaffolding (FFN + multi-head
attention) on both sides. If MapFormer-EM beats TEM_T, the paper's
structural argument is empirically validated. If they tie, the
contribution is parallelism without expressivity loss.

run_tem_t.sh: polls for free GPUs, then trains TEM_T on clean / noise /
lm200 with parallel pairs across both GPUs, then runs a 7-way eval
table (Vanilla / Level15 / VanillaEM / Level15EM / TEM / TEMFaithful /
TEM_T) at T=128 and T=512 OOD. Result lands in TEM_T_RESULTS.md.
" 2>&1 | tail -3
git push origin main 2>&1 | tail -3
echo "[$(date)] Done."
