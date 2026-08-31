#!/usr/bin/env bash
# TOP-UP: seeds 3-7 for the two arms that decide Q2.
#
# WHY. The loop-on-path-integration comparison came out +0.297 with sd 0.296 and
# MDE 0.479 at n=3 -- UNDERPOWERED, so it is not reportable either way. The
# variance is in the BASELINE: one layer of path integration on Match-Query is
# unstable (0.398 / 0.449 / 0.800, sd 0.219), and the single seed where it trained
# well is the one where the loop added nothing (-0.029). The loop arm is tighter
# (sd 0.133). At n=8 the MDE falls to 0.293, just under the observed effect.
#
# Only PI_flat and PI_loop are topped up -- they are the pair Q2 compares, and the
# other three arms do not enter it. Same settings as the main batch, so seeds 0-2
# there and 3-7 here pool (the pipeline is bit-reproducible; the aliasing sweep's
# repro control drifted 0.000).
#
# NOTE the baseline mis-specification this exposed, recorded so it is not repeated:
# the PUBLISHED Match-Query 128^2 number (0.823) is a THREE-LAYER config. Our L3
# arm reproduces it exactly (0.824). The 1-layer arm was chosen for parameter parity
# with the loop, so the honest framing of Q2 is "does a loop recover what DEPTH
# provides", not "does a loop help where path integration is insufficient".
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPO="$(cd "$(dirname "$0")" && pwd)"; cd "$REPO/.."
R="$REPO/runs/loop_headroom"
LOG="$REPO/loop_topup.log"; echo "top-up queued $(date)" > "$LOG"
EP=300; NB=48; BS=16; SZ=128; NOBS=16; TE=512; TQ=256; DM=128; NH=2
A="train_match_""query"
until [ -f "$REPO/.loop_headroom_done" ]; do sleep 120; done
while [ "$(pgrep -u "$USER" -f "$A" | wc -l)" -gt 0 ]; do sleep 60; done
echo "$(date +%H:%M) main batch clear" >> "$LOG"
MAXPG=3
on_gpu(){ pgrep -u "$USER" -af "$A" 2>/dev/null | grep -c -- "--device cuda:$1" || true; }
launch(){ V="$1"; SEED="$2"; LBL="$3"
  OUT="$R/$LBL/s$SEED"; mkdir -p "$OUT"
  [ -f "$OUT/${V}.pt" ] && return
  GPU=""
  while :; do
    N0=$(on_gpu 0); N1=$(on_gpu 1)
    if [ "$N0" -le "$N1" ] && [ "$N0" -lt "$MAXPG" ]; then GPU=0; break; fi
    if [ "$N1" -lt "$N0" ] && [ "$N1" -lt "$MAXPG" ]; then GPU=1; break; fi
    if [ "$N0" -lt "$MAXPG" ]; then GPU=0; break; fi
    if [ "$N1" -lt "$MAXPG" ]; then GPU=1; break; fi
    sleep 45
  done
  echo "$(date +%H:%M) $LBL s$SEED -> cuda:$GPU (load $N0/$N1)" >> "$LOG"
  python3 -u -m mapformer.train_match_query --variant "$V" --seed "$SEED" \
    --epochs $EP --n-batches $NB --batch-size $BS --size $SZ --n-obs $NOBS \
    --T-explore $TE --T-query $TQ --eval-query 256 512 --n-layers 1 \
    --d-model $DM --n-heads $NH --schedule cosine --fast-attn \
    --device "cuda:$GPU" --output-dir "$OUT" > "$R/${LBL}_s${SEED}.log" 2>&1 &
  sleep 15
}
for SEED in 3 4 5 6 7; do
  launch Vanilla "$SEED" PI_flat
  launch Looped  "$SEED" PI_loop
done
wait
echo "$(date +%H:%M) $(find "$R" -name '*.pt' | wc -l) checkpoints" >> "$LOG"
python3 -u -m mapformer.agg_loop_headroom --runs-dir "$R" \
    --out "$REPO/LOOP_HEADROOM.md" >> "$LOG" 2>&1
touch "$REPO/.loop_topup_done"; echo "$(date) DONE" >> "$LOG"
