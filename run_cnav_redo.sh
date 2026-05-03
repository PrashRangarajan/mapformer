#!/bin/bash
# Re-run CNAV with the hard_ce loss fix. Old MSE-trained CNAV checkpoints
# collapsed to predicting zero (mean position error ~20 cells = chance);
# hard_ce gives ~2 cells in 500 step smoke test. This re-runs all 4
# variants (Vanilla / Level15 / VanillaEM / Level15EM) with hard_ce, then
# re-evaluates and re-probes for hex.
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

cnav_train() {
    local variant=$1 gpu=$2
    mkdir -p "$REPO/runs/cnav/$variant/seed0"
    CUDA_VISIBLE_DEVICES=$gpu python3 -u -m mapformer.train_continuous \
        --variant $variant --seed 0 \
        --epochs 30 --n-batches 156 --batch-size 128 --n-steps 128 \
        --buffer-size 25000 --n-grid-units 256 \
        --v-noise-std 0.05 --omega-noise-std 0.05 \
        --loss hard_ce \
        --device cuda \
        --output-dir mapformer/runs/cnav/$variant/seed0 \
        > "$LOGS/cnav_${variant}_s0.log" 2>&1
}

echo "[$(date)] Pair 1: Vanilla (gpu0) + Level15 (gpu1) ..."
cnav_train Vanilla 0 &
P1=$!
cnav_train Level15 1 &
P2=$!
wait $P1 $P2
echo "[$(date)] Pair 1 done. Pair 2: VanillaEM (gpu0) + Level15EM (gpu1) ..."
cnav_train VanillaEM 0 &
P3=$!
cnav_train Level15EM 1 &
P4=$!
wait $P3 $P4
echo "[$(date)] All 4 CNAV trainings done."
for v in Vanilla Level15 VanillaEM Level15EM; do
    echo "  $v: $(grep 'Epoch  30/30' $LOGS/cnav_${v}_s0.log | tail -1)"
done

echo "[$(date)] Cross-T / cross-noise eval (all 4 variants)..."
python3 -u -m mapformer.eval_continuous \
    --checkpoints mapformer/runs/cnav/Vanilla/seed0/Vanilla.pt \
                  mapformer/runs/cnav/Level15/seed0/Level15.pt \
                  mapformer/runs/cnav/VanillaEM/seed0/VanillaEM.pt \
                  mapformer/runs/cnav/Level15EM/seed0/Level15EM.pt \
    --T-list 128 256 512 1024 \
    --noise-levels 0.0 0.05 0.1 0.2 \
    --n-traj 30 --device cuda \
    > "$REPO/CNAV_RESULTS.md" 2>"$LOGS/cnav_eval.err"
echo "[$(date)] Eval done."
tail -30 "$REPO/CNAV_RESULTS.md"

echo "[$(date)] Hex probes (all 4 variants, T=256)..."
for v in Vanilla Level15 VanillaEM Level15EM; do
    python3 -u -m mapformer.probe_hex_continuous \
        --checkpoint mapformer/runs/cnav/$v/seed0/$v.pt \
        --device cuda --n-traj 200 --T 256 --n-bins 64 \
        --save-rate-maps "$REPO/paper_figures/cnav_${v}_rate_maps_s0.npz" \
        > "$REPO/CNAV_HEX_${v}.md" 2>"$LOGS/cnav_probe_${v}.err"
done
echo "[$(date)] Hex probes done."

cd "$REPO"
git pull --rebase 2>&1 | tail -3
git add CNAV_RESULTS.md CNAV_HEX_Vanilla.md CNAV_HEX_Level15.md \
        CNAV_HEX_VanillaEM.md CNAV_HEX_Level15EM.md \
        train_continuous.py run_cnav_redo.sh \
        paper_figures/cnav_Vanilla_rate_maps_s0.npz \
        paper_figures/cnav_Level15_rate_maps_s0.npz \
        paper_figures/cnav_VanillaEM_rate_maps_s0.npz \
        paper_figures/cnav_Level15EM_rate_maps_s0.npz
git add runs/cnav/Vanilla/seed0/*.pt \
        runs/cnav/Level15/seed0/*.pt \
        runs/cnav/VanillaEM/seed0/*.pt \
        runs/cnav/Level15EM/seed0/*.pt 2>/dev/null || true
git commit -m "CNAV re-run with hard_ce loss (replaces broken-MSE results)

The original CNAV pipeline used MSE on sparse DoG targets, which has
a degenerate near-zero minimum (target is ~0% sparse, all-zero
prediction gives MSE close to mean of squared targets). All 4 variants
collapsed to ~20-cell mean position error on a 64-cell torus, i.e.
chance.

train_continuous.py: replaces MSE default with hard_ce — closest
place-cell argmax as classification target. Smoke test: position
error drops from ~20 cells (chance) to ~2 cells (well within a
single place-cell spacing of 4 cells) in 500 training steps.

This re-run produces CNAV_RESULTS.md and CNAV_HEX_*.md with actually
informative numbers. The position-decoding column is the one to read
(MSE column is meaningless when training with hard_ce since the
model output is now logits, not DoG values).
" 2>&1 | tail -3
git push origin main 2>&1 | tail -3
echo "[$(date)] Done."
