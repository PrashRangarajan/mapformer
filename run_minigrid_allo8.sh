#!/usr/bin/env bash
# The FULL 8-cell allocentric DoorKey-16 factorial, n=3, trained in ONE batch.
#
# Why all eight and not just the missing one: MINIGRID_ALLOCENTRIC_2X2X2.md ran
# only 7 arms, and its headline ("all four path-integrated arms outrank all
# index arms, min path-int 0.809 > max index 0.807") turns on the index maximum
# -- computed without PoPE-Hier, which was one of the two best arms in the RAW
# factorial (0.955). The untested cell is exactly the one that can overturn the
# claim, and 0.002 is the margin.
#
# Those 7 checkpoints were trained on the other machine (commit cd34232) and do
# not exist here, so adding one arm locally would be a cross-machine,
# cross-batch comparison -- rule 3 in its worst form. All 24 runs go in one
# batch instead; the other machine's 7 arms then become an independent
# replication rather than the basis of the claim.
#
# No `local` anywhere (under `set -u` it expands every word before assigning).
set -uo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"; cd "$REPO/.."
R="$REPO/runs/minigrid_allo8"; mkdir -p "$R"
LOG="$REPO/minigrid_allo8.log"; echo "start $(date)" > "$LOG"
VARS="Vanilla Hourglass_k2 MapPoPE-Flat MapPoPE-Hier RoPE PlainHourglass PoPE-Flat PoPE-Hier"

# 1. Pre-build the allocentric buffers sequentially so 8 trainers do not race to
#    build the same 25K-trajectory pickle at once.
echo "$(date +%H:%M) building allocentric buffers" >> "$LOG"
for SEED in 0 1 2; do
  python3 -u -c "from mapformer.minigrid_env import MiniGridWorld_Cached as C; \
e=C(env_name='MiniGrid-DoorKey-16x16-v0',tokenization='obj_color',seed=$SEED,\
buffer_size=25000,allocentric=True); e.generate_batch(2,128)" >> "$LOG" 2>&1
done
echo "$(date +%H:%M) buffers ready" >> "$LOG"

# 2. 24 runs. GPUS/CAP are env-overridable because the right dispatch is a
#    property of the MACHINE, not the experiment: on a box where both GPUs are
#    ours use `GPUS="0 1" CAP=8`; where someone else holds GPU 0 use the
#    default. Single-GPU costs wall-clock, not correctness -- all 24 arms still
#    train in one batch, which is the property that matters.
#    The counter persists across the whole nest; resetting it per round is the
#    bug that once put 2-of-3 jobs on the contended GPU.
GPUS="${GPUS:-1}"; CAP="${CAP:-5}"
set -- $GPUS; NG=$#
echo "$(date +%H:%M) dispatch: GPUS='$GPUS' CAP=$CAP" >> "$LOG"
i=0
for SEED in 0 1 2; do
  for V in $VARS; do
    D="$R/${V}_s${SEED}"
    if [ -f "$D/${V}.pt" ]; then echo "skip $V s$SEED" >> "$LOG"; i=$((i+1)); continue; fi
    while [ "$(jobs -rp | wc -l)" -ge "$CAP" ]; do sleep 15; done
    set -- $GPUS; shift $(( i % NG )); GPU=$1
    echo "$(date +%H:%M) $V s$SEED -> cuda:$GPU" >> "$LOG"
    python3 -u -m mapformer.train_variant --variant "$V" --seed "$SEED" \
      --n-landmarks 0 --epochs 50 --n-batches 156 --n-layers 3 \
      --env minigrid_doorkey16 --minigrid-tokenization obj_color \
      --minigrid-allocentric --minigrid-cached-buffer 25000 \
      --device "cuda:$GPU" --output-dir "$D" > "$R/train_${V}_s${SEED}.log" 2>&1 &
    i=$((i+1)); sleep 3
  done
done
wait

# 3. `wait` returns regardless of child success -- verify the artifacts exist
#    rather than inferring success from the absence of a crash. Also print each
#    arm's final training loss: on the H=12 sweep accuracy tracked final loss at
#    r = -0.996, so an arm that merely converged worse is indistinguishable from
#    an arm that is worse unless the losses are on the record.
N=$(ls "$R"/*/*.pt 2>/dev/null | wc -l)
echo "$(date +%H:%M) trained $N/24 checkpoints" >> "$LOG"
if [ "$N" -lt 24 ]; then
  echo "INCOMPLETE: $N/24 -- not evaluating" >> "$LOG"; exit 1
fi
for F in "$R"/*/*.pt; do
  python3 -c "import sys,torch; d=torch.load(sys.argv[1],map_location='cpu',weights_only=False); l=d.get('losses') or [float('nan')]; print(f\"  final_loss {sys.argv[1].split('/')[-2]:22s} {l[-1]:.4f}\")" "$F" >> "$LOG" 2>&1
done

# The eval writes MINIGRID_ALLOCENTRIC_8CELL.json (per-seed) beside the .md.
# COMMIT BOTH: the 7-arm run committed only the .md, so its headline
# ("min path-int 0.809 > max index 0.807") cannot be re-derived per-seed the way
# AUDIT_HEADLINE.md requires -- 0.809/0.807 are arm MEANS with 1-sd bands that
# overlap at every length, and at T=128 the ordering is already reversed.
python3 -u -m mapformer.eval_minigrid_2x2 --runs-dir "$R" \
  --variants $VARS --seeds 0 1 2 --lengths 128 512 1024 \
  --allocentric --device "cuda:${GPUS%% *}" \
  --out "$REPO/MINIGRID_ALLOCENTRIC_8CELL.md" >> "$LOG" 2>&1
touch "$R/.done"
echo "$(date) DONE" >> "$LOG"
