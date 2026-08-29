#!/usr/bin/env bash
# FINAL WAVE of the aliasing sweep -- scoped down from the original 25-run plan.
#
# WHY THE CUT. The original queue would have run ~17 more hours. Most of that was
# seeds the design does not need, because the conditions turned out to have very
# different variance:
#   n_obs=256  effect +0.389, sd 0.022 over 5 seeds  -> already over-powered
#   n_obs=64   in the same tight regime; seed 4 buys nothing
#   n_obs=16   effect +0.173, sd 0.125 -- the ONLY high-variance cell, because of
#              bimodal basin selection in the path-integrated arm
#
# The headline claim is "the effect DIFFERS between n_obs=16 and n_obs=256"
# (+0.216). The sd of that difference is 0.127, dominated entirely by the anchor.
# At n=3 the MDE is 0.205 -- detectable by a margin of only 0.011, i.e. a coin
# flip. At n=5 it is 0.159, a clean call. So the four anchor runs are the only
# remaining seeds worth spending, and n64 seed 4 is dropped.
#
# 5 runs instead of 13. ~5.5 h instead of ~17 h.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPO="$(cd "$(dirname "$0")" && pwd)"; cd "$REPO/.."
R="$REPO/runs/alias_sweep"
LOG="$REPO/alias_finish.log"; echo "alias finish start $(date)" > "$LOG"
G=32; T=512; NBUF=24000; EP=400; NB=180; BS=24; DM=256; NL=4; NH=4; NW=24; ETRIALS=128
MAXPG=3
A="train_mini""world"      # split so pgrep cannot match this script's own cmdline

# count MY training arms on a given GPU, whoever launched them (the previous
# wave's arms are still running and must be counted against the slot budget)
on_gpu(){ pgrep -u "$USER" -af "$A" 2>/dev/null | grep -c -- "--device cuda:$1" || true; }

launch(){
  local V="$1" SEED="$2" NOBS="$3" OUT="$4"
  mkdir -p "$OUT"
  [ -f "$OUT/${V}_oracle.pt" ] && { echo "skip ${V} n${NOBS} s${SEED}" >> "$LOG"; return; }
  local GPU=""
  while :; do
    if [ "$(on_gpu 0)" -lt "$MAXPG" ]; then GPU=0; break; fi
    if [ "$(on_gpu 1)" -lt "$MAXPG" ]; then GPU=1; break; fi
    sleep 60
  done
  echo "$(date +%H:%M) ${V} n_obs=${NOBS} s${SEED} -> cuda:$GPU" >> "$LOG"
  python3 -u -m mapformer.train_miniworld --variant "$V" --seed "$SEED" --oracle \
    --grid-size $G --n-obs "$NOBS" --n-steps $T --buffer-size $NBUF --epochs $EP \
    --n-batches $NB --batch-size $BS --d-model $DM --n-layers $NL --n-heads $NH \
    --n-workers $NW --schedule cosine --eval-trials $ETRIALS --eval-lengths 512 1024 \
    --device "cuda:$GPU" --output-dir "$OUT" > "$R/${V}_n${NOBS}_s${SEED}_fin.log" 2>&1 &
  sleep 45      # let it allocate before the next slot check, so we do not overfill
}

# reproducibility control first: it licenses (or voids) the whole reused anchor,
# so it is the single most informative run left
launch RoPE 0 16 "$R/n16_repro/s0"
for SEED in 3 4; do
  for V in Vanilla RoPE; do launch "$V" "$SEED" 16 "$R/n16/s$SEED"; done
done
wait

# `wait` returns regardless of child success -- verify artifacts rather than
# inferring success from the absence of a crash
N_PT=$(find "$R" -name "*_oracle.pt" | wc -l)
echo "$(date +%H:%M) finished; $N_PT checkpoints present" >> "$LOG"

python3 -u -m mapformer.agg_alias --runs-dir "$R" \
    --anchor-dir "$REPO/runs/rope_converge" --seeds 0 1 2 3 4 \
    --out "$REPO/ALIASING_CONTROLLED.md" >> "$LOG" 2>&1

touch "$REPO/.alias_finish_done"
echo "$(date) DONE" >> "$LOG"
