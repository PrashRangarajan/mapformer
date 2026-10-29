#!/bin/bash
# Mode-conditioned cognitive maps (GSF (2)-variation) on multi-env.
# Tests whether modes spontaneously specialize when each has its own
# per-block frequencies omega_k.
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

echo "[$(date)] [modeomega] Polling for free GPUs..."
while ! is_gpu_free; do sleep 60; done
echo "[$(date)] [modeomega] GPUs free."

train() {
    local variant=$1 seed=$2 gpu=$3
    mkdir -p "$REPO/runs/${variant}_multienv/seed${seed}"
    CUDA_VISIBLE_DEVICES=$gpu python3 -u -m mapformer.train_multienv \
        --variant $variant --seed $seed \
        --n-landmarks 200 --n-train-envs 50 --n-test-envs 50 \
        --epochs 50 --n-batches 156 \
        --device cuda \
        --output-dir mapformer/runs/${variant}_multienv/seed${seed} \
        > "$LOGS/${variant}_multienv_s${seed}.log" 2>&1
}

# Train both modeomega variants in parallel (with and without dropout)
echo "[$(date)] [modeomega] ModeOmega + NoDrop_ModeOmega"
train Level15GSF_ModeOmega 0 0 & P1=$!
train Level15GSF_NoDrop_ModeOmega 0 1 & P2=$!
wait $P1 $P2

echo "[$(date)] [modeomega] Done. Final losses:"
for v in Level15GSF_ModeOmega Level15GSF_NoDrop_ModeOmega; do
    f="$LOGS/${v}_multienv_s0.log"
    [ -f "$f" ] || continue
    echo "  ${v}: $(grep 'Epoch  50/50' $f | tail -1)"
    echo "       $(grep -E 'Train|Held-out' $f | head -3)"
done

# Diagnostic: omega_modes spread + per-env mode winner
python3 -u <<'PYEOF' > "$REPO/MODEOMEGA_RESULTS.md" 2>"$LOGS/modeomega_eval.err"
import torch, torch.nn.functional as F, numpy as np
from pathlib import Path
from mapformer.environment_multienv import MultiEnvGridWorld
from mapformer.model_inekf_gsf_modeomega import (
    MapFormerWM_Level15GSF_ModeOmega,
    MapFormerWM_Level15GSF_NoDrop_ModeOmega,
)
from mapformer.train_multienv import evaluate

print("# Mode-conditioned cognitive maps — GSF (2)-variation results\n")
print("K=8 modes, each with mode-specific omega_k (per-block frequencies).")
print("Trained on 50 random torus envs (lm200), evaluated on 50 held-out envs.")
print("Question: do modes spontaneously specialize when given mode-specific")
print("scale parameters? Or do they collapse to identical values?\n")

print("## Held-out accuracy\n")
print("| Variant | Train-env acc | Held-out T=128 | Held-out T=512 |")
print("|---|---|---|---|")

VARIANT_CLS = {
    "Level15GSF_ModeOmega": MapFormerWM_Level15GSF_ModeOmega,
    "Level15GSF_NoDrop_ModeOmega": MapFormerWM_Level15GSF_NoDrop_ModeOmega,
}

for v in ["Level15GSF_ModeOmega", "Level15GSF_NoDrop_ModeOmega"]:
    ckpt = Path(f"mapformer/runs/{v}_multienv/seed0/{v}_multienv.pt")
    if not ckpt.exists(): print(f"| {v} | — | — | — |"); continue
    c = torch.load(ckpt, map_location="cpu", weights_only=False)
    tr = c.get("train_acc", float("nan"))
    te = c.get("test_acc", float("nan"))
    te2 = c.get("test_acc_T2", float("nan"))
    print(f"| **{v}** | {tr:.3f} | {te:.3f} | {te2:.3f} |")
print()

# Compare to existing Level15GSF_NoDrop (no mode-omega) on multi-env
print("**For comparison (from MULTIENV_RESULTS.md):** Level15GSF_NoDrop held-out T=128 0.999, T=512 0.980; TEMFaithful 1.000 / 0.965.\n")

# Diagnostic: omega_modes spread post-training
print("## Diagnostic: omega_modes spread post-training\n")
print("If modes specialized, omega values should differ across K. If collapsed, they should all be similar.\n")
for v in ["Level15GSF_ModeOmega", "Level15GSF_NoDrop_ModeOmega"]:
    ckpt = Path(f"mapformer/runs/{v}_multienv/seed0/{v}_multienv.pt")
    if not ckpt.exists(): continue
    c = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = c["config"]
    cls = VARIANT_CLS[v]
    m = cls(vocab_size=cfg["vocab_size"], d_model=cfg["d_model"],
            n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
            grid_size=cfg["grid_size"], n_modes=8)
    m.load_state_dict(c["model_state_dict"])
    omega = m.omega_modes.detach()  # (K, H, NB)
    K, H, NB = omega.shape
    # Spread metric: std across K, averaged over (H, NB)
    spread = omega.std(dim=0).mean().item()
    mean_omega = omega.mean().item()
    range_ = (omega.max() - omega.min()).item()
    print(f"### {v}")
    print(f"- omega_modes mean: {mean_omega:.4f}, range across all (k, h, nb): {range_:.4f}")
    print(f"- std across K (averaged over h, nb): {spread:.4f}")
    print(f"- relative spread: {spread / abs(mean_omega):.3f} (compare to 0.10 at init)")
    # Show first head's omega values per mode
    print(f"- omega per mode (head 0, first 4 blocks):")
    for k in range(K):
        print(f"  mode {k}: {omega[k, 0, :4].tolist()}")
    print()

# Per-env mode winner — sample held-out envs and see which mode wins
print("## Per-env mode winner diagnostic\n")
print("On each of 20 held-out envs, run a T=128 trajectory and check which")
print("mode has highest cumulative log-likelihood at the end. If modes specialized,")
print("different modes should win on different envs.\n")

for v in ["Level15GSF_NoDrop_ModeOmega"]:
    ckpt = Path(f"mapformer/runs/{v}_multienv/seed0/{v}_multienv.pt")
    if not ckpt.exists(): continue
    c = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = c["config"]
    cls = VARIANT_CLS[v]
    m = cls(vocab_size=cfg["vocab_size"], d_model=cfg["d_model"],
            n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
            grid_size=cfg["grid_size"], n_modes=8).cuda()
    m.load_state_dict(c["model_state_dict"])
    m.eval()
    world = MultiEnvGridWorld(size=64, n_obs_types=16, p_empty=0.5,
                              n_landmarks=200, n_train_envs=50, n_test_envs=50, seed=0)
    rng = np.random.RandomState(3000)
    winners = []
    with torch.no_grad():
        for env_idx in range(20):
            # Sample multiple trajectories per env to get statistics
            env = world.test_envs[env_idx]
            for _ in range(5):
                tokens, _, _ = env.generate_trajectory(128)
                _ = m(tokens.unsqueeze(0).cuda())
                # Final-step mixture weights
                log_w = m.last_log_w[0, :, -1]  # (K,)
                winner = int(log_w.argmax().item())
                winners.append((env_idx, winner))
    # Count: for each test env, what's the distribution of winners?
    from collections import Counter
    by_env = {}
    for env_idx, w in winners:
        by_env.setdefault(env_idx, []).append(w)
    print(f"### {v}\n")
    print("| Env | Mode winners (5 trajectories) |")
    print("|---|---|")
    for env_idx in sorted(by_env.keys())[:20]:
        wins = Counter(by_env[env_idx])
        win_str = ", ".join(f"m{m}:{c}" for m, c in wins.most_common())
        print(f"| {env_idx} | {win_str} |")
    # Overall mode usage
    all_wins = Counter(w for _, w in winners)
    print(f"\n**Overall mode usage (100 trajectories total):** {dict(all_wins.most_common())}\n")
    print(f"**Interpretation:** if modes specialized, you should see different envs producing different winning modes.")
    print(f"If all envs produce the same winner, modes have collapsed to functionally equivalent.\n")
    del m

print("*Auto-generated by run_modeomega.sh, single seed (seed=0)*\n")
PYEOF

cd "$REPO"
git pull --rebase 2>&1 | tail -3
git add MODEOMEGA_RESULTS.md model_inekf_gsf_modeomega.py train_variant.py run_modeomega.sh
for v in Level15GSF_ModeOmega Level15GSF_NoDrop_ModeOmega; do
    git add runs/${v}_multienv/seed0/${v}_multienv.pt 2>/dev/null || true
done
git commit -m "Mode-conditioned cognitive maps (GSF (2)-variation) on multi-env

K=8 modes, each with mode-specific per-block frequencies omega_k. Tests
whether modes spontaneously specialize when given the architectural capacity
to do so. Single seed, multi-env (50 train / 50 held-out), lm200.

Diagnostics: omega_modes spread post-training (collapsed vs distinct), and
per-env mode-winner distribution (do different envs activate different modes?).
" 2>&1 | tail -3
git push origin main 2>&1 | tail -3
echo "[$(date)] [modeomega] Done."
