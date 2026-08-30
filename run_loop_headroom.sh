#!/usr/bin/env bash
# DOES A LOOP HELP WHERE PATH INTEGRATION IS NOT ENOUGH?
#
# WHY THE TORUS COULD NOT ANSWER THIS. The looped pilot found recursion buys
# depth's horizon at a quarter of the parameters for the INDEX arm (+0.363,
# sd 0.018, 3/3 seeds) but nothing measurable on top of PATH INTEGRATION
# (+0.046, sd 0.074, MDE 0.120, one seed negative). That null is uninterpretable:
# MapFormer-WM already scores 0.948 there with 0.052 of headroom, so there was
# essentially nothing for a loop to win. A ceiling cannot distinguish "the loop
# adds nothing" from "there was nothing to add".
#
# THE TESTBED. Match-Query is the project's most trustworthy task -- it is the one
# result that survived the 2026-08-09 audit, and it passed the context-destruction
# ablation (0.918 -> 0.074 with explore observations shuffled). On the 128^2
# config, path integration is NECESSARY but far from SUFFICIENT:
#     path-integrated  0.823 +/- 0.043      index  0.192 +/- 0.022   chance 0.0625
# 0.177 of headroom to a perfect 1.0, and a seed sd small enough to see a loop
# effect (MDE 0.070 at n=3). That is exactly the regime the torus lacked.
#
# THE 2x2, all at ONE BLOCK of parameters (204,630 path-int / 204,182 index):
#     Vanilla L1     path integration, no loop
#     Looped x4      path integration + loop      <- the question
#     RoPE L1        index, no loop
#     RoPELooped x4  index + loop                 <- the reverse direction
# plus Vanilla L3 (601,174, three real blocks) as the real-depth reference.
#
# PRE-REGISTERED, before any GPU is spent:
#   Q1 sanity -- Vanilla L1 >> RoPE L1. If this fails the task is not behaving as
#      published and nothing else here is readable.
#   Q2 THE QUESTION -- Looped vs Vanilla L1, with 0.177 of headroom available:
#        gain > MDE -> the loop COMPLEMENTS path integration once there is room.
#                      The torus null was a ceiling artifact and the recursive
#                      MapFormer is worth building.
#        gain ~ 0   -> they are SUBSTITUTES even with headroom. That generalises
#                      the torus negative instead of explaining it away, and is
#                      the stronger result of the two.
#   Q3 the reverse -- RoPELooped vs RoPE L1, where the index arm sits near its
#      floor (0.192 against chance 0.0625):
#        gain      -> a loop helps whichever arm has headroom, position code
#                     irrelevant.
#        no gain   -> a loop needs a working position code to build on; on the
#                     torus it only ever moved an arm that was already partly
#                     succeeding.
#   Q4 -- Looped vs Vanilla L3: does looping match REAL depth at a third of the
#      parameters on a task harder than the torus? The pilot said yes there.
#
# DEVIATIONS from the published Match-Query runs, so the 0.823 is a REFERENCE and
# not a baseline: warmup+cosine instead of LinearLR-from-step-one (rule 10), and
# fast-attn. Every arm here is retrained in this batch under identical settings
# (rule 3), so internal validity does not depend on either.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPO="$(cd "$(dirname "$0")" && pwd)"; cd "$REPO/.."
R="$REPO/runs/loop_headroom"; mkdir -p "$R"
LOG="$REPO/loop_headroom.log"; echo "loop-headroom queued $(date)" > "$LOG"
EP=300; NB=48; BS=16; SZ=128; NOBS=16; TE=512; TQ=256; DM=128; NH=2
A="train_match_""query"

echo "$(date +%H:%M) waiting for the visits test" >> "$LOG"
until [ -f "$REPO/.visits_test_done" ]; do sleep 180; done
for P in "train_mini""world" "train_var""iant" "$A"; do
  while [ "$(pgrep -u "$USER" -f "$P" | wc -l)" -gt 0 ]; do sleep 120; done
done
echo "$(date +%H:%M) GPUs clear" >> "$LOG"

MAXPG=3
on_gpu(){ pgrep -u "$USER" -af "$A" 2>/dev/null | grep -c -- "--device cuda:$1" || true; }
launch(){   # variant n_layers seed label
  V="$1"; NL="$2"; SEED="$3"; LBL="$4"
  OUT="$R/$LBL/s$SEED"; mkdir -p "$OUT"
  [ -f "$OUT/${V}.pt" ] && { echo "skip $LBL s$SEED" >> "$LOG"; return; }
  GPU=""
  while :; do
    N0=$(on_gpu 0); N1=$(on_gpu 1)
    if [ "$N0" -le "$N1" ] && [ "$N0" -lt "$MAXPG" ]; then GPU=0; break; fi
    if [ "$N1" -lt "$N0" ] && [ "$N1" -lt "$MAXPG" ]; then GPU=1; break; fi
    if [ "$N0" -lt "$MAXPG" ]; then GPU=0; break; fi
    if [ "$N1" -lt "$MAXPG" ]; then GPU=1; break; fi
    sleep 60
  done
  echo "$(date +%H:%M) $LBL ($V L$NL) s$SEED -> cuda:$GPU (load $N0/$N1)" >> "$LOG"
  python3 -u -m mapformer.train_match_query --variant "$V" --seed "$SEED" \
    --epochs $EP --n-batches $NB --batch-size $BS --size $SZ --n-obs $NOBS \
    --T-explore $TE --T-query $TQ --eval-query 256 512 --n-layers "$NL" \
    --d-model $DM --n-heads $NH --schedule cosine --fast-attn \
    --device "cuda:$GPU" --output-dir "$OUT" > "$R/${LBL}_s${SEED}.log" 2>&1 &
  sleep 20
}

for SEED in 0 1 2; do
  launch Vanilla    1 "$SEED" PI_flat
  launch Looped     1 "$SEED" PI_loop
  launch RoPE       1 "$SEED" IX_flat
  launch RoPELooped 1 "$SEED" IX_loop
  launch Vanilla    3 "$SEED" PI_L3
done
wait
echo "$(date +%H:%M) $(find "$R" -name '*.pt' | wc -l)/15 checkpoints" >> "$LOG"
python3 -u -m mapformer.agg_loop_headroom --runs-dir "$R" \
    --out "$REPO/LOOP_HEADROOM.md" >> "$LOG" 2>&1
touch "$REPO/.loop_headroom_done"; echo "$(date) DONE" >> "$LOG"
