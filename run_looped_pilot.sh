#!/usr/bin/env bash
# LOOPED-TRANSFORMER PILOT: does RECURSION buy attention horizon cheaply?
#
# BACKGROUND (HORIZON_RESULTS.md). An index model's ability to path-integrate
# scales with DEPTH -- horizon ~2 at 1 layer, ~16 at 2, ~32 at 4 -- but then
# stops: 3.17M params at 4 layers x 256 wide still collapses past interval ~32,
# while a 204K 1-layer PATH-INTEGRATED model reaches 65+ at 0.880. A looped
# transformer buys effective depth at CONSTANT parameters, which separates two
# things that grid could not.
#
# Q1 (index arm). Is the horizon about effective depth, or about having DISTINCT
#     layers that can specialise? RoPELooped is 204K params at effective depth 4.
#       ~ RoPE-L4  -> recursion buys depth's horizon at 1/4 the parameters.
#       ~ RoPE-L1  -> weight sharing does NOT buy horizon; specialisation is what
#                     depth was providing, and looping is not a substitute.
#
# Q2 (path-integrated arm). Real depth HURT MapFormer at long range in that grid
#     (Vanilla L2 d256 0.976 -> L4 d256 0.782 at interval 65+). Does recursion
#     inherit that penalty?
#       Looped < Vanilla-L1 at 65+ -> stacking recursion on MapFormer is
#                     counterproductive, and the "recursive MapFormer" idea is
#                     dead before it is built.
#       Looped ~ Vanilla-L1        -> recursion is neutral for path integration
#                     and the two mechanisms compose.
#
# Both answers are decision-relevant and neither is obvious, which is the point
# of running it before building anything on top.
#
# DESIGN NOTES
#   * ALL THREE configs are retrained here. The published horizon table used 16
#     epochs of LinearLR and this uses 300 of warmup+cosine, so the stored numbers
#     are NOT a valid baseline for these (rule 3: never compare a fresh variant to
#     a stored checkpoint).
#   * LinearLR(1.0->0.0) decays from step one and can trap a run on a plateau --
#     it is what inverted the grid-8 sign (rule 10). --schedule cosine throughout.
#   * Plain ALBERT-style sharing: no per-iteration depth embedding, theta computed
#     once. Both are deliberate; see model_looped.py for why.
#   * Param parity verified: Looped 207,457 == L1 207,457, vs L4 802,273.
#
# 18 runs, ~9 s/epoch for the depth-4 configs -> ~1.5 h total.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPO="$(cd "$(dirname "$0")" && pwd)"; cd "$REPO/.."
R="$REPO/runs/looped_pilot"; mkdir -p "$R"
LOG="$REPO/looped_pilot.log"; echo "looped pilot queued $(date)" > "$LOG"
EP=300; NB=98; BS=128; T=128; DM=128; NH=2
A="train_var""iant"

echo "$(date +%H:%M) waiting for the aliasing follow-up to finish" >> "$LOG"
until [ -f "$REPO/.alias_followup_done" ]; do sleep 180; done
# and for its arms to actually exit
B="train_mini""world"
while [ "$(pgrep -u "$USER" -f "$B" | wc -l)" -gt 0 ]; do sleep 120; done
echo "$(date +%H:%M) GPUs clear; starting" >> "$LOG"

MAXPG=3
on_gpu(){ pgrep -u "$USER" -af "$A" 2>/dev/null | grep -c -- "--device cuda:$1" || true; }
launch(){   # variant seed n_layers label
  V="$1"; SEED="$2"; NL="$3"; LBL="$4"
  D="$R/$LBL/${V}_s${SEED}"; mkdir -p "$D"
  [ -f "$D/${V}.pt" ] && { echo "skip $LBL $V s$SEED" >> "$LOG"; return; }
  GPU=""
  while :; do
    if [ "$(on_gpu 0)" -lt "$MAXPG" ]; then GPU=0; break; fi
    if [ "$(on_gpu 1)" -lt "$MAXPG" ]; then GPU=1; break; fi
    sleep 30
  done
  echo "$(date +%H:%M) $LBL $V s$SEED -> cuda:$GPU" >> "$LOG"
  python3 -u -m mapformer.train_variant --variant "$V" --seed "$SEED" \
    --epochs $EP --n-batches $NB --batch-size $BS --n-steps $T \
    --n-layers "$NL" --n-heads $NH --d-model $DM --n-landmarks 0 \
    --schedule cosine --device "cuda:$GPU" --output-dir "$D" \
    > "$R/${LBL}_${V}_s${SEED}.log" 2>&1 &
  sleep 5
}

for SEED in 0 1 2; do
  launch Vanilla    "$SEED" 1 L1
  launch RoPE       "$SEED" 1 L1
  launch Vanilla    "$SEED" 4 L4
  launch RoPE       "$SEED" 4 L4
  launch Looped     "$SEED" 1 Loop4
  launch RoPELooped "$SEED" 1 Loop4
done
wait

N_PT=$(find "$R" -name "*.pt" | wc -l)
echo "$(date +%H:%M) training done; $N_PT/18 checkpoints" >> "$LOG"

echo "$(date +%H:%M) probing horizon per config" >> "$LOG"
python3 -u -m mapformer.probe_revisit_distance --runs-dir "$R/L1" \
    --variants RoPE Vanilla --seeds 0 1 2 --device cuda:0 \
    --out "$REPO/LOOPED_L1.md" >> "$LOG" 2>&1
python3 -u -m mapformer.probe_revisit_distance --runs-dir "$R/L4" \
    --variants RoPE Vanilla --seeds 0 1 2 --device cuda:0 \
    --out "$REPO/LOOPED_L4.md" >> "$LOG" 2>&1
python3 -u -m mapformer.probe_revisit_distance --runs-dir "$R/Loop4" \
    --variants RoPELooped Looped --seeds 0 1 2 --device cuda:0 \
    --out "$REPO/LOOPED_Loop4.md" >> "$LOG" 2>&1

python3 -u -m mapformer.agg_looped --repo "$REPO" --runs-dir "$R" \
    --out "$REPO/LOOPED_PILOT.md" >> "$LOG" 2>&1

touch "$REPO/.looped_pilot_done"
echo "$(date) DONE" >> "$LOG"
