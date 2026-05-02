#!/bin/bash
# Auto-launches when GPUs are free. Runs:
#   1. Re-trains discrete Level15_DoG with the fixed DoG kernel + hex probe.
#   2. Trains continuous Vanilla / Level15 + hex probe + eval.
#
# Both pipelines auto-commit results.
set -u
cd /home/prashr
REPO=/home/prashr/mapformer
LOGS=$REPO/logs
mkdir -p "$LOGS"

is_gpu_free() {
    # Both GPUs must be < 50% utilized AND < 5 GB used.
    # nvidia-smi outputs "6463, 100" — strip comma + whitespace.
    local mem util
    while IFS=', ' read -r mem util; do
        mem=${mem//[^0-9]/}
        util=${util//[^0-9]/}
        [ -z "$mem" ] && mem=0
        [ -z "$util" ] && util=0
        if [ "$mem" -ge 5000 ] || [ "$util" -ge 50 ]; then
            return 1
        fi
    done < <(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
    return 0
}

echo "[$(date)] Waiting for GPUs to free up..."
while ! is_gpu_free; do
    nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>&1 | tr '\n' ' ' | sed "s/^/[$(date)] busy: /; s/$/\n/"
    sleep 120
done
echo "[$(date)] GPUs free; launching pipelines."

# --- Pipeline 1: discrete Level15_DoG with fixed kernel (gpu0) ---
echo "[$(date)] [P1] Re-training Level15_DoG with fixed DoG kernel..."
mkdir -p "$REPO/runs/Level15_DoG_fixed_clean/seed0"
CUDA_VISIBLE_DEVICES=0 python3 -u -m mapformer.train_variant \
    --variant Level15_DoG --seed 0 \
    --n-landmarks 0 --p-action-noise 0.0 \
    --epochs 50 --n-batches 156 --aux-coef 0.1 \
    --device cuda \
    --output-dir mapformer/runs/Level15_DoG_fixed_clean/seed0 \
    > "$LOGS/Level15_DoG_fixed_clean_s0.log" 2>&1 &
P1=$!

# --- Pipeline 2: continuous WM + EM, vanilla + Level15 (gpu1, sequential) ---
# Four trainings to test whether EM's position-content decoupling actually
# helps when content is rich (continuous DoG obs) — the regime where the
# Hadamard A_X ⊙ A_P should shine, vs WM's RoPE entanglement.
echo "[$(date)] [P2] Training continuous Vanilla / Level15 / VanillaEM / Level15EM on gpu1..."
mkdir -p "$REPO/runs/cnav/Vanilla/seed0"   "$REPO/runs/cnav/Level15/seed0" \
         "$REPO/runs/cnav/VanillaEM/seed0" "$REPO/runs/cnav/Level15EM/seed0"
(
    for v in Vanilla Level15 VanillaEM Level15EM; do
        CUDA_VISIBLE_DEVICES=1 python3 -u -m mapformer.train_continuous \
            --variant $v --seed 0 \
            --epochs 30 --n-batches 156 --batch-size 128 --n-steps 128 \
            --buffer-size 25000 --n-grid-units 256 \
            --v-noise-std 0.05 --omega-noise-std 0.05 \
            --device cuda \
            --output-dir mapformer/runs/cnav/$v/seed0 \
            > "$LOGS/cnav_${v}_s0.log" 2>&1
        echo "[$(date)] [P2-${v}] done."
    done
) &
P2=$!

wait $P1
echo "[$(date)] [P1] Discrete Level15_DoG training done."
tail -5 "$LOGS/Level15_DoG_fixed_clean_s0.log"

echo "[$(date)] [P1] Probing fixed Level15_DoG for hex..."
python3 -u -m mapformer.probe_hex \
    --checkpoint mapformer/runs/Level15_DoG_fixed_clean/seed0/Level15_DoG.pt \
    --device cuda --n-traj 200 --T 512 \
    --save-rate-maps "$REPO/paper_figures/dog_rate_maps_fixed_s0.npz" \
    > "$REPO/DOG_RESULTS_FIXED.md" 2>"$LOGS/dog_fixed_probe.err"
echo "[$(date)] [P1] Probe done."

wait $P2
echo "[$(date)] [P2] Continuous trainings done."
tail -5 "$LOGS/cnav_Level15_s0.log"

echo "[$(date)] [P2] Probing continuous for hex on all four variants..."
for v in Vanilla Level15 VanillaEM Level15EM; do
    python3 -u -m mapformer.probe_hex_continuous \
        --checkpoint mapformer/runs/cnav/$v/seed0/$v.pt \
        --device cuda --n-traj 200 --T 256 --n-bins 64 \
        --save-rate-maps "$REPO/paper_figures/cnav_${v}_rate_maps_s0.npz" \
        > "$REPO/CNAV_HEX_${v}.md" 2>"$LOGS/cnav_probe_${v}.err"
done
echo "[$(date)] [P2] Probes done."

echo "[$(date)] [P2] Cross-T / cross-noise eval (all four variants)..."
python3 -u -m mapformer.eval_continuous \
    --checkpoints mapformer/runs/cnav/Vanilla/seed0/Vanilla.pt \
                  mapformer/runs/cnav/Level15/seed0/Level15.pt \
                  mapformer/runs/cnav/VanillaEM/seed0/VanillaEM.pt \
                  mapformer/runs/cnav/Level15EM/seed0/Level15EM.pt \
    --T-list 128 256 512 1024 \
    --noise-levels 0.0 0.05 0.1 0.2 \
    --n-traj 30 --device cuda \
    > "$REPO/CNAV_RESULTS.md" 2>"$LOGS/cnav_eval.err"
echo "[$(date)] [P2] Eval done."

# --- Pipeline 3: action-record corruption vs stochastic-transition MDP ---
# Verifies the equivalence we discussed verbally: training MapFormer with
# p_transition_noise=0.10 (env executes random action 10% of steps; records
# the commanded one) should give the same accuracy/NLL as p_action_noise=0.10
# (post-hoc token corruption of clean trajectory) for our uniform policy.
echo "[$(date)] [P3] Stochastic-transition torus runs..."
mkdir -p "$REPO/runs/torus_transnoise/Vanilla/seed0" \
         "$REPO/runs/torus_transnoise/Level15/seed0"
CUDA_VISIBLE_DEVICES=0 python3 -u -m mapformer.train_variant \
    --variant Vanilla --seed 0 \
    --p-transition-noise 0.10 \
    --epochs 50 --n-batches 156 \
    --device cuda \
    --output-dir mapformer/runs/torus_transnoise/Vanilla/seed0 \
    > "$LOGS/torus_transnoise_Vanilla_s0.log" 2>&1 &
P3a=$!
CUDA_VISIBLE_DEVICES=1 python3 -u -m mapformer.train_variant \
    --variant Level15 --seed 0 \
    --p-transition-noise 0.10 \
    --epochs 50 --n-batches 156 \
    --device cuda \
    --output-dir mapformer/runs/torus_transnoise/Level15/seed0 \
    > "$LOGS/torus_transnoise_Level15_s0.log" 2>&1 &
P3b=$!
wait $P3a $P3b
echo "[$(date)] [P3] Stochastic-transition trainings done."

echo "[$(date)] [P3] Eval: stochastic-transition vs action-record (existing checkpoints)"
python3 -u <<'PYEOF' > "$REPO/STOCHASTIC_TRANSITION_RESULTS.md" 2>"$LOGS/stoch_trans_eval.err"
import torch, torch.nn.functional as F, numpy as np
from pathlib import Path
from mapformer.environment import GridWorld
from mapformer.model import MapFormerWM
from mapformer.model_inekf_level15 import MapFormerWM_Level15InEKF

VARIANT_CLS = {"Vanilla": MapFormerWM, "Level15": MapFormerWM_Level15InEKF}

def build(variant, ckpt):
    c = torch.load(ckpt, map_location="cuda", weights_only=False)
    cfg = c.get("config", {})
    cls = VARIANT_CLS[variant]
    m = cls(vocab_size=cfg["vocab_size"], d_model=cfg.get("d_model", 128),
            n_heads=cfg.get("n_heads", 2), n_layers=cfg.get("n_layers", 1),
            grid_size=cfg.get("grid_size", 64))
    m.load_state_dict(c["model_state_dict"])
    return m.cuda().eval()

def eval_revisit(model, env, T, n_trials, seed, p_action_noise=0.0, p_transition_noise=0.0):
    torch.manual_seed(seed); np.random.seed(seed)
    c = tot = 0; nll_sum = 0.0
    with torch.no_grad():
        for _ in range(n_trials):
            tokens, _, rm = env.generate_trajectory(T, p_transition_noise=p_transition_noise)
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

print("# Stochastic-transition MDP vs action-record corruption\n")
print("Tests the equivalence: trained on each noise model, eval on each.")
print("If equivalence holds for our uniform policy, all four diagonal+off-diagonal")
print("cells should show similar accuracy.\n")

# Train type x Eval type matrix
ckpts = {
    "trans-noise":  {
        "Vanilla": "mapformer/runs/torus_transnoise/Vanilla/seed0/Vanilla.pt",
        "Level15": "mapformer/runs/torus_transnoise/Level15/seed0/Level15.pt",
    },
    "action-noise": {
        "Vanilla": "mapformer/runs/Vanilla_noise/seed0/Vanilla.pt",
        "Level15": "mapformer/runs/Level15_noise/seed0/Level15.pt",
    },
}

env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, n_landmarks=0, seed=1000)

for v in ["Vanilla", "Level15"]:
    print(f"## {v}\n")
    print("| train-type \\ eval-type | trans-noise T=128 | trans-noise T=512 | action-noise T=128 | action-noise T=512 |")
    print("|---|---|---|---|---|")
    for train_kind, ckpt_dict in ckpts.items():
        ckpt = ckpt_dict[v]
        if not Path(ckpt).exists():
            print(f"| {train_kind} | (no ckpt) | — | — | — |")
            continue
        m = build(v, ckpt)
        row = [train_kind]
        for eval_p_trans, eval_p_act in [(0.10, 0.0), (0.10, 0.0), (0.0, 0.10), (0.0, 0.10)]:
            T = 128 if len(row) % 2 == 1 else 512
            a, n = eval_revisit(m, env, T, 100 if T == 128 else 50, seed=2000,
                                p_action_noise=eval_p_act, p_transition_noise=eval_p_trans)
            row.append(f"{a:.3f} (NLL {n:.3f})" if a is not None else "(err)")
        print("| " + " | ".join(row) + " |")
        del m; torch.cuda.empty_cache()
    print()

print("\n*Auto-generated by run_dog_fix_and_continuous.sh (P3 stochastic-transition).*\n")
PYEOF

echo "[$(date)] [P3] Eval done."
tail -25 "$REPO/STOCHASTIC_TRANSITION_RESULTS.md"

# --- Pipeline 4: MiniGrid-MemoryS13 — different topology (rooms + hallway)
# Memory env tests retention across distance, has different aliasing
# structure than DoorKey. We just train and eval — no claim about pure
# memory mechanism since revisit-mask doesn't directly capture cross-phase
# memory, but the env is a useful supplementary topology test.
mg_train() {
    local variant=$1 gpu=$2
    mkdir -p "$REPO/runs/minigrid_memory_cached/$variant/seed0"
    CUDA_VISIBLE_DEVICES=$gpu python3 -u -m mapformer.train_variant \
        --variant $variant --seed 0 \
        --n-landmarks 0 --p-action-noise 0.0 \
        --epochs 50 --n-batches 156 \
        --env minigrid_memory --minigrid-tokenization obj_color \
        --minigrid-cached-buffer 25000 \
        --device cuda \
        --output-dir mapformer/runs/minigrid_memory_cached/$variant/seed0 \
        > "$LOGS/mg_memory_${variant}_s0.log" 2>&1
}

echo "[$(date)] [P4] Building MemoryS13 buffer + training Vanilla on gpu0..."
mg_train Vanilla 0

echo "[$(date)] [P4] Buffer warm; training Level15 + RoPE in parallel..."
mg_train Level15 0 &
PV=$!
mg_train RoPE    1 &
PR=$!
wait $PV $PR
echo "[$(date)] [P4] All MemoryEnv trainings done."

echo "[$(date)] [P4] Eval Vanilla / Level15 / RoPE on MemoryS13..."
python3 -u <<'PYEOF' > "$REPO/MINIGRID_MEMORY_RESULTS.md" 2>"$LOGS/mg_memory_eval.err"
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
            grid_size=cfg.get("grid_size", 13))
    m.load_state_dict(c["model_state_dict"])
    return m.cuda().eval()

def eval_revisit(model, env, T, n_trials, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    c = tot = 0; nll_sum = 0.0
    with torch.no_grad():
        for _ in range(n_trials):
            tokens, _, rm = env.generate_trajectory(T)
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

print("# MiniGrid-MemoryS13: Vanilla vs Level 1.5 vs RoPE\n")
print("Different topology than DoorKey: starting room → narrow hallway → ")
print("choice room with two objects. Tests architecture transfer to a ")
print("rooms+hallway layout. NOT a pure-memory test (our self-supervised ")
print("revisit prediction doesn't directly target the cross-phase memory ")
print("the env was designed for); supplementary topology check only.\n")

variants = ["Vanilla", "Level15", "RoPE"]
T_LIST = [128, 512, 1024]
N_TRIALS = {128: 100, 512: 50, 1024: 25}

env_ood = MiniGridWorld(env_name="MiniGrid-MemoryS13-v0",
                        tokenization="obj_color", seed=1000)

print("| Variant | T=128 OOD | T=512 OOD | T=1024 OOD |")
print("|---|---|---|---|")
for v in variants:
    ckpt = Path(f"mapformer/runs/minigrid_memory_cached/{v}/seed0/{v}.pt")
    if not ckpt.exists():
        print(f"| {v} | (no ckpt) | — | — |")
        continue
    m = build(v, ckpt)
    row = [f"**{v}**"]
    for T in T_LIST:
        a, n = eval_revisit(m, env_ood, T, N_TRIALS[T], seed=2000)
        row.append(f"{a:.3f} (NLL {n:.3f})" if a is not None else "(err)")
    print("| " + " | ".join(row) + " |")
    del m; torch.cuda.empty_cache()

print("\n*Auto-generated by run_dog_fix_and_continuous.sh (P4 MemoryEnv).*\n")
PYEOF
echo "[$(date)] [P4] Eval done."
tail -20 "$REPO/MINIGRID_MEMORY_RESULTS.md"

# --- Pipeline 5: TEM-style recurrent baseline on torus -----------------
# RNN with factorised g/x and Hebbian outer-product memory (Whittington
# 2020 simplified; single-env scope). Trained on the same torus task as
# Vanilla and Level15 so we have direct numbers for the comparison the
# user asked for. Single-seed.
tem_train() {
    local cfg=$1 lm=$2 noise=$3 gpu=$4
    mkdir -p "$REPO/runs/TEM_${cfg}/seed0"
    CUDA_VISIBLE_DEVICES=$gpu python3 -u -m mapformer.train_variant \
        --variant TEM --seed 0 \
        --n-landmarks $lm --p-action-noise $noise \
        --epochs 50 --n-batches 156 \
        --device cuda \
        --output-dir mapformer/runs/TEM_${cfg}/seed0 \
        > "$LOGS/TEM_${cfg}_s0.log" 2>&1
}

echo "[$(date)] [P5] TEM-recurrent on torus: clean + noise (parallel pair)"
tem_train clean 0   0.0  0 &
PT1=$!
tem_train noise 0   0.10 1 &
PT2=$!
wait $PT1 $PT2
echo "[$(date)] [P5] Pair 1 done. Running lm200 on gpu0..."
tem_train lm200 200 0.0 0
echo "[$(date)] [P5] All TEM trainings done."

echo "[$(date)] [P5] Eval TEM vs Vanilla vs Level15 on torus..."
python3 -u <<'PYEOF' > "$REPO/TEM_RESULTS.md" 2>"$LOGS/tem_eval.err"
import torch, torch.nn.functional as F, numpy as np
from pathlib import Path
from mapformer.environment import GridWorld
from mapformer.model import MapFormerWM
from mapformer.model_inekf_level15 import MapFormerWM_Level15InEKF
from mapformer.model_tem import TEMRecurrent

VARIANT_CLS = {"Vanilla": MapFormerWM, "Level15": MapFormerWM_Level15InEKF,
               "TEM": TEMRecurrent}

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

print("# TEM-style recurrent baseline vs MapFormer\n")
print("Single-seed comparison on the torus task across three regimes")
print("(clean, action-noise 10%, landmarks 200). TEM is a simplified")
print("Whittington 2020-style RNN with factorised g/x state and Hebbian")
print("outer-product memory — trained on a single environment, so it")
print("loses TEM's primary lever (compositional generalisation).\n")

ckpts = {
    "clean": {
        "Vanilla": "mapformer/runs/Vanilla_clean/seed0/Vanilla.pt",
        "Level15": "mapformer/runs/Level15_clean/seed0/Level15.pt",
        "TEM":     "mapformer/runs/TEM_clean/seed0/TEM.pt",
    },
    "noise": {
        "Vanilla": "mapformer/runs/Vanilla_noise/seed0/Vanilla.pt",
        "Level15": "mapformer/runs/Level15_noise/seed0/Level15.pt",
        "TEM":     "mapformer/runs/TEM_noise/seed0/TEM.pt",
    },
    "lm200": {
        "Vanilla": "mapformer/runs/Vanilla_lm200/seed0/Vanilla.pt",
        "Level15": "mapformer/runs/Level15_lm200/seed0/Level15.pt",
        "TEM":     "mapformer/runs/TEM_lm200/seed0/TEM.pt",
    },
}

for cfg_tag in ["clean", "noise", "lm200"]:
    n_lm = 200 if cfg_tag == "lm200" else 0
    eval_noise = 0.10 if cfg_tag == "noise" else 0.0
    print(f"## {cfg_tag}\n")
    print("| Variant | T=128 OOD | T=512 OOD | T=128 NLL | T=512 NLL |")
    print("|---|---|---|---|---|")
    for v in ["Vanilla", "Level15", "TEM"]:
        ckpt = Path(ckpts[cfg_tag][v])
        if not ckpt.exists():
            print(f"| {v} | (no ckpt at {ckpt}) | — | — | — |")
            continue
        m = build(v, ckpt)
        env_ood = GridWorld(size=64, n_obs_types=16, p_empty=0.5,
                            n_landmarks=n_lm, seed=1000)
        a128, n128 = eval_revisit(m, env_ood, 128, 200, seed=2000, p_action_noise=eval_noise)
        a512, n512 = eval_revisit(m, env_ood, 512, 100, seed=2000, p_action_noise=eval_noise)
        print(f"| **{v}** | {a128:.3f} | {a512:.3f} | {n128:.3f} | {n512:.3f} |")
        del m; torch.cuda.empty_cache()
    print()

print("\n*Auto-generated by run_dog_fix_and_continuous.sh (P5 TEM).*\n")
PYEOF
echo "[$(date)] [P5] Eval done."
tail -25 "$REPO/TEM_RESULTS.md"

cd "$REPO"
git pull --rebase 2>&1 | tail -3
git add DOG_RESULTS_FIXED.md CNAV_HEX_Vanilla.md CNAV_HEX_Level15.md CNAV_RESULTS.md \
        STOCHASTIC_TRANSITION_RESULTS.md MINIGRID_MEMORY_RESULTS.md \
        TEM_RESULTS.md model_tem.py \
        run_dog_fix_and_continuous.sh \
        environment.py train.py train_variant.py \
        paper_figures/dog_rate_maps_fixed_s0.npz \
        paper_figures/cnav_Vanilla_rate_maps_s0.npz \
        paper_figures/cnav_Level15_rate_maps_s0.npz
git add runs/Level15_DoG_fixed_clean/seed0/*.pt \
        runs/cnav/Vanilla/seed0/*.pt \
        runs/cnav/Level15/seed0/*.pt \
        runs/torus_transnoise/Vanilla/seed0/*.pt \
        runs/torus_transnoise/Level15/seed0/*.pt \
        runs/minigrid_memory_cached/Vanilla/seed0/*.pt \
        runs/minigrid_memory_cached/Level15/seed0/*.pt \
        runs/minigrid_memory_cached/RoPE/seed0/*.pt \
        runs/TEM_clean/seed0/*.pt \
        runs/TEM_noise/seed0/*.pt \
        runs/TEM_lm200/seed0/*.pt 2>/dev/null || true
git commit -m "Fixed DoG kernel + continuous nav + stochastic-transition equivalence

Three pipelines, all auto-launched after GPUs freed up:

P1: Re-runs discrete Level15_DoG with the corrected (normalized) DoG
target kernel. Earlier DOG_RESULTS.md was on silently zero targets;
DOG_RESULTS_FIXED.md is the actual hex-emergence test.

P2: First results on continuous 2D nav (Cueva/Wei/Sorscher setup):
Vanilla / Level15 with v_noise=omega_noise=0.05, both with a 256-unit
ReLU bottleneck. Cross-T + cross-noise eval in CNAV_RESULTS.md;
per-variant hex probes in CNAV_HEX_*.md.

P3: Stochastic-transition vs action-record equivalence test. Trains
Vanilla and Level15 with --p-transition-noise 0.10 (env executes a
random action 10% of the time but records the commanded one). Compares
to existing --p-action-noise 0.10 checkpoints under each eval-time
noise model. Validates the verbal equivalence argument with empirical
numbers in STOCHASTIC_TRANSITION_RESULTS.md.

environment.py / train.py / train_variant.py: --p-transition-noise
flag and a generate_trajectory(p_transition_noise=...) parameter that
applies the noise at execution time, distinct from train.py's
existing post-hoc record corruption.
" 2>&1 | tail -3
git push origin main 2>&1 | tail -3
echo "[$(date)] Done."
