#!/bin/bash
# Two TEM experiments:
#   (A) TEM in the NEW regime — action-noise. Post-fix TEMFaithful on the
#       noise task (n_landmarks=0, p_action_noise=0.10). Fills the data gap:
#       no current-TEM noise number exists (TEM_T_RESULTS noise row was stale).
#   (B) Direct transformer-machinery test — TEMFaithful_FFN (TEM + a
#       per-position FFN on the retrieved content) on clean + lm200. Tests
#       whether adding one piece of transformer machinery closes TEM's
#       ~1-3pp clean-regime lag.
#
# GPU 0 only, sequential (GPU 1 is another user's). Waits for the TEM
# scaling sweep to finish first.
set -u
cd /home/prashr
REPO=/home/prashr/mapformer
LOGS=$REPO/logs
mkdir -p "$LOGS" "$REPO/paper_figures/tem_noise_ffn"

GPU=0
SCALING_PID=1881864
echo "[$(date)] waiting for TEM scaling sweep (PID $SCALING_PID)..."
while kill -0 "$SCALING_PID" 2>/dev/null; do sleep 60; done
echo "[$(date)] TEM scaling done; starting noise + FFN runs on GPU $GPU"

train_one() {
    local variant=$1 cfg=$2 seed=$3 lm=$4 noise=$5
    mkdir -p "$REPO/runs/${variant}_${cfg}/seed${seed}"
    CUDA_VISIBLE_DEVICES=$GPU python3 -u -m mapformer.train_variant \
        --variant "$variant" --seed $seed \
        --n-landmarks $lm --grid-size 64 --p-action-noise $noise \
        --epochs 50 --n-batches 156 \
        --device cuda \
        --output-dir mapformer/runs/${variant}_${cfg}/seed${seed} \
        > "$LOGS/${variant}_${cfg}_s${seed}.log" 2>&1
}

# (A) TEMFaithful on noise
for seed in 0 1 2; do
    echo "[$(date)] TEMFaithful noise s$seed"
    train_one TEMFaithful noise $seed 0 0.10
done
# (B) TEMFaithful_FFN on clean and lm200
for seed in 0 1 2; do
    echo "[$(date)] TEMFaithful_FFN clean s$seed"
    train_one TEMFaithful_FFN clean $seed 0 0.0
done
for seed in 0 1 2; do
    echo "[$(date)] TEMFaithful_FFN lm200 s$seed"
    train_one TEMFaithful_FFN lm200 $seed 200 0.0
done
echo "[$(date)] training done"

# --- eval ---
ED=$REPO/paper_figures/tem_noise_ffn
eval_one() {
    local variant=$1 cfg=$2 seed=$3 pan=$4
    local ckpt="$REPO/runs/${variant}_${cfg}/seed${seed}/${variant}.pt"
    [ -f "$ckpt" ] || { echo "  miss ${variant}_${cfg} s$seed"; return; }
    CUDA_VISIBLE_DEVICES=$GPU python3 -m mapformer.eval_single_env \
        --variant "$variant" --checkpoint "$ckpt" --p-action-noise "$pan" \
        > "$ED/${variant}_${cfg}_s${seed}.json" \
        2>"$LOGS/temnf_eval_${variant}_${cfg}_s${seed}.err"
}
echo "[$(date)] evaluating"
for s in 0 1 2; do
    eval_one TEMFaithful      noise $s 0.10   # noise regime: eval under noise
    eval_one TEMFaithful_FFN  clean $s 0.0
    eval_one TEMFaithful_FFN  lm200 $s 0.0
done

# --- aggregate ---
cd "$REPO"
python3 -u <<'PYEOF' > "$REPO/TEM_NOISE_FFN_RESULTS.md" 2>"$LOGS/tem_nf_agg.err"
import json, numpy as np
from pathlib import Path

def fmt(arr):
    if not arr: return "—"
    return f"{np.mean(arr):.3f} ± {np.std(arr):.3f} (n={len(arr)})"

def collect(variant, cfg):
    a128, a512, n512 = [], [], []
    for s in [0, 1, 2]:
        p = Path(f"paper_figures/tem_noise_ffn/{variant}_{cfg}_s{s}.json")
        if not p.exists(): continue
        j = json.loads(p.read_text())
        if j.get("acc_T128") is not None: a128.append(j["acc_T128"])
        if j.get("acc_T512") is not None: a512.append(j["acc_T512"])
        if j.get("nll_T512") is not None: n512.append(j["nll_T512"])
    return a128, a512, n512

print("# TEM in the noise regime + the transformer-machinery direct test\n")

print("## (A) TEM in the new regime: action-noise (p=0.10)\n")
print("Post-fix TEMFaithful on the noise task — fills the data gap (the old")
print("TEM_T_RESULTS noise row was a stale pre-bug-fix number). Reference")
print("numbers: Vanilla 0.638, Level15 0.702, Level15NoDrop 0.699 (n=3,")
print("NODROP_PARETO_RESULTS.md).\n")
print("| Variant | T=128 noise | T=512 noise OOD | T=512 NLL |")
print("|---|---|---|---|")
a128, a512, n512 = collect("TEMFaithful", "noise")
print(f"| **TEMFaithful** | {fmt(a128)} | {fmt(a512)} | {fmt(n512)} |")
print("| Vanilla (ref) | — | 0.638 | — |")
print("| Level15 (ref) | — | 0.702 | — |")
print("| Level15NoDrop (ref) | — | 0.699 | — |")
print()

print("## (B) Direct test: does a per-position FFN close TEM's clean-regime lag?\n")
print("TEMFaithful_FFN = TEMFaithful + a per-position FFN on the retrieved")
print("content (the fixed Hopfield bank is unchanged). If this closes the")
print("~3pp clean-regime gap to Level15, the missing-FFN hypothesis holds.\n")
print("### clean (n_landmarks=0)\n")
print("| Variant | T=128 acc | T=512 OOD acc | T=512 NLL |")
print("|---|---|---|---|")
a,b,c = collect("TEMFaithful_FFN", "clean")
print(f"| **TEMFaithful_FFN** | {fmt(a)} | {fmt(b)} | {fmt(c)} |")
print("| TEMFaithful (no FFN, ref) | 1.000 | 0.966 ± 0.008 | 0.182 |")
print("| Level15 (ref) | 1.000 | 0.993 | 0.039 |")
print()
print("### lm200\n")
print("| Variant | T=128 acc | T=512 OOD acc | T=512 NLL |")
print("|---|---|---|---|")
a,b,c = collect("TEMFaithful_FFN", "lm200")
print(f"| **TEMFaithful_FFN** | {fmt(a)} | {fmt(b)} | {fmt(c)} |")
print("| TEMFaithful (no FFN, ref) | 1.000 | 0.969 ± 0.010 | 0.171 |")
print()
print("## Decision\n")
print("- **TEMFaithful_FFN clean ≈ Level15 (~0.99)**: the clean-regime lag WAS the missing per-position FFN. Confirms the transformer-machinery hypothesis — TEM needs only the FFN, not the whole transformer, to close the gap.")
print("- **TEMFaithful_FFN clean ≈ TEMFaithful (~0.97)**: the FFN does not help; the clean lag is from something else (e.g. learned content attention, not per-position processing).")
print("- **TEMFaithful_FFN lm200 < TEMFaithful**: the FFN hurts where TEM already wins — added machinery is not free.\n")
print("*Auto-generated by run_tem_noise_and_ffn.sh*")
PYEOF

git pull --rebase 2>&1 | tail -3
git add model_tem_ffn.py train_variant.py eval_single_env.py \
    run_tem_noise_and_ffn.sh TEM_NOISE_FFN_RESULTS.md \
    TEM_RESULTS.md TEM_T_RESULTS.md paper_figures/tem_noise_ffn/ 2>/dev/null
for cfg_pair in "TEMFaithful noise" "TEMFaithful_FFN clean" "TEMFaithful_FFN lm200"; do
    set -- $cfg_pair
    for s in 0 1 2; do
        git add runs/${1}_${2}/seed${s}/*.pt 2>/dev/null || true
    done
done
git commit -m "TEM noise regime + transformer-machinery direct test (TEMFaithful_FFN)

(A) Post-fix TEMFaithful on the action-noise task — fills the data gap
    (the TEM_T_RESULTS noise row was a stale pre-bug-fix number).
(B) TEMFaithful_FFN: TEM + a per-position FFN on the retrieved content,
    fixed Hopfield bank unchanged. Tests whether one piece of transformer
    machinery closes TEM's ~3pp clean-regime lag.

Also: removed stale pre-bug-fix TEMFaithful rows from TEM_RESULTS.md and
TEM_T_RESULTS.md (predict-then-update bug, ~0.42 chance numbers).
" 2>&1 | tail -3
git push origin main 2>&1 | tail -2
echo "[$(date)] done."
