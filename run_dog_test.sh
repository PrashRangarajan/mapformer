#!/bin/bash
# Train Level15_DoG (Sorscher Option A) seed 0 on clean, then probe for hex.
set -u
cd /home/prashr
REPO=/home/prashr/mapformer
LOGS=$REPO/logs
mkdir -p "$LOGS" "$REPO/runs/Level15_DoG_clean/seed0"

echo "[$(date)] Training Level15_DoG clean s0 (aux_coef=0.1, 50 epochs)..."
CUDA_VISIBLE_DEVICES=0 python3 -u -m mapformer.train_variant \
    --variant Level15_DoG --seed 0 \
    --n-landmarks 0 --p-action-noise 0.0 \
    --epochs 50 --n-batches 156 --aux-coef 0.1 \
    --device cuda \
    --output-dir mapformer/runs/Level15_DoG_clean/seed0 \
    > "$LOGS/Level15_DoG_clean_s0.log" 2>&1
echo "[$(date)] Training done."
tail -8 "$LOGS/Level15_DoG_clean_s0.log"

echo "[$(date)] Probing for hex (200 trajs, T=512)..."
mkdir -p "$REPO/paper_figures"
python3 -u -m mapformer.probe_hex \
    --checkpoint mapformer/runs/Level15_DoG_clean/seed0/Level15_DoG.pt \
    --device cuda --n-traj 200 --T 512 \
    --save-rate-maps "$REPO/paper_figures/dog_rate_maps_s0.npz" \
    > "$REPO/DOG_RESULTS.md" 2> "$LOGS/dog_probe.err"

echo "[$(date)] Probe done. Results:"
cat "$REPO/DOG_RESULTS.md"

cd "$REPO"
git add model_level15_dog.py probe_hex.py run_dog_test.sh DOG_RESULTS.md \
        train.py train_variant.py
git add runs/Level15_DoG_clean/seed0/*.pt 2>/dev/null || true
git commit -m "Level15_DoG: Sorscher Option A — DoG aux head + ReLU bottleneck

Adds MapFormerWM_Level15_DoG variant testing whether hex emerges in a
non-negative bottleneck layer when supervised with DoG-of-position
place-cell targets, alongside the standard categorical CE loss.

Architecture: hidden -> Linear -> ReLU (probe layer) -> Linear -> 256
place cells. Targets are DoG(d(pos, c_j)) with sigma_E=1.5, sigma_I=3.0,
non-negative-clipped, on a 16x16 grid of place-cell centers over the
64x64 torus.

probe_hex.py: build per-unit rate maps from trajectories, compute
Sargolini-style grid scores via SAC + rotational correlations.

DOG_RESULTS.md: probe output for the seed-0 run.
" 2>&1 | tail -3
git push origin main 2>&1 | tail -3
echo "[$(date)] Done."
