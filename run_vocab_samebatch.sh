#!/usr/bin/env bash
# Multi-seed, same-batch vocab sweep: Vanilla / VanillaEM / VanillaEM_P0
# x n_obs {16,256,4096} x seeds {0,1,2} = 27 runs.
# Tests whether "VanillaEM crashes at n_obs=256" (single seed, 0.562) is an
# architectural fact or a collapsed seed -- separate-q0/k0 EM is seed-unstable
# on the paper task (0.898 +/- 0.108, worst seed 0.778).
# Waits for the compositional same-batch job so the GPUs are not shared.
# NB: inside `local`, reference $1/$2 -- never $s -- see run_em_comp_samebatch.sh.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$REPO/.."
LOG="$REPO/vocab_samebatch.log"; : > "$LOG"; echo "start $(date)" >> "$LOG"
for _ in $(seq 1 480); do [ -f "$REPO/.em_comp_samebatch_done" ] && break; sleep 30; done
echo "$(date +%H:%M) compositional done; starting vocab sweep" >> "$LOG"
OUT="$REPO/runs/vocab_samebatch"
JOBS=()
for n in 16 256 4096; do for s in 0 1 2; do for v in Vanilla VanillaEM VanillaEM_P0; do
  JOBS+=("$v:$n:$s"); done; done; done
run(){ local g=$1 j=$2; IFS=: read -r v n s <<<"$j"
  local o="$OUT/${v}_vocab${n}/seed${s}"
  [ -f "$o/${v}.pt" ] && return
  echo "$(date +%H:%M) [gpu$g] $v nobs=$n s$s" >> "$LOG"
  python3 -u -m mapformer.train_variant --variant "$v" --seed "$s" \
    --n-landmarks 0 --p-action-noise 0.0 --n-obs-types "$n" \
    --epochs 50 --n-batches 156 --device "cuda:$g" --output-dir "$o" >> "$LOG" 2>&1; }
( for i in "${!JOBS[@]}"; do [ $((i%2)) -eq 0 ] && run 0 "${JOBS[$i]}"; done ) & P0=$!
( for i in "${!JOBS[@]}"; do [ $((i%2)) -eq 1 ] && run 1 "${JOBS[$i]}"; done ) & P1=$!
wait $P0 $P1
python3 -u -m mapformer.eval_vocab_sweep --runs-dir "$OUT" --seeds 0 1 2 \
  --device cuda:0 --out "$REPO/VOCAB_SWEEP_MULTISEED.md" >> "$LOG" 2>&1
cd "$REPO"; git add VOCAB_SWEEP_MULTISEED.md VOCAB_SWEEP_MULTISEED.json \
  eval_vocab_sweep.py run_vocab_samebatch.sh 2>/dev/null
git diff --cached --quiet || { git commit -q -m "Multi-seed same-batch vocab sweep (supersedes single-seed EM row)

Vanilla / VanillaEM / VanillaEM_P0 x n_obs {16,256,4096} x 3 seeds, all trained
in one batch. The single-seed 'VanillaEM crashes at n_obs=256' (0.562) could not
be distinguished from a collapsed seed given EM's 0.898 +/- 0.108 spread on the
paper task. Per-seed values reported. Auto-committed; interpretation pending."
  git push origin main >> "$LOG" 2>&1; }
echo "$(date +%H:%M) DONE" >> "$LOG"; touch "$REPO/.vocab_samebatch_done"
