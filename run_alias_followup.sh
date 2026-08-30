#!/usr/bin/env bash
# FOLLOW-UP to the aliasing sweep, restructured around --fast-attn (2.56x faster,
# 37% of the memory, mathematically identical: logits 1.4e-06, grads 2.4e-08,
# grad cosine 1.0000000000).
#
# WHAT IS LEFT AND WHY
#
# The 400-epoch curve is complete and inverts the pre-registered aliasing
# prediction: effect +0.173 / +0.310 / +0.374 as aliasing FALLS (32 / 8 / 2 cells
# per token at fixed grid 32). Two things stop that being publishable.
#
#   THREAT  -- the index arm is flat in only 1/5 seeds at n_obs=256, so +0.374 is
#              an upper bound. The convergence-sensitivity check says the bias runs
#              the other way (both-flat only: +0.173 / +0.316 / +0.408), but that
#              rests on ONE converged pair. Measure the endpoint instead.
#   GAP     -- the map-size axis at matched aliasing (2.0 cells/token) is two
#              points: grid 8 -0.010, grid 32 +0.374. Two points are a line, not a
#              trend -- the exact error the H12 budget curve taught. grid 16 at
#              n_obs=64 is the missing middle, and it converts the two-factor
#              account (map size = can you localise; aliasing = what imprecision
#              costs) from post-hoc story into a pre-registered prediction.
#
# PRE-REGISTERED for grid 16 (2.0 cells/token, 128 occupied cells), before it runs:
#   effect lands BETWEEN -0.010 and +0.374 -> two-factor account SUPPORTED; the
#       position effect is graded in map size at fixed aliasing.
#   effect near -0.010 -> the jump is a threshold, not a gradient; grid 8 and 16
#       are both inside attention's reach and only grid 32 is not.
#   effect near or above +0.374 -> map size saturates early; something other than
#       map size separates grid 8 from the rest, and the account is wrong.
#
# WHY WAVE A EXISTS. --fast-attn is mathematically identical but the
# attention-dropout RNG draws differ, so a fast-attn run is not bit-identical to
# one without it. Every number above was produced WITHOUT it. Wave A retrains the
# n_obs=256 cell at the SAME 400 epochs with fast-attn: if the effect reproduces,
# the optimisation is licensed for everything after it. If it does not, the
# speedup is abandoned rather than silently mixed in. This costs 6 runs and buys
# ~20 hours (18 runs in ~9 h, against ~29 h for 12 runs unoptimised).
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPO="$(cd "$(dirname "$0")" && pwd)"; cd "$REPO/.."
R="$REPO/runs/alias_follow"; mkdir -p "$R"
LOG="$REPO/alias_followup.log"; echo "followup start $(date)" > "$LOG"
T=512; NBUF=24000; NB=180; BS=24; DM=256; NL=4; NH=4; NW=24; ETRIALS=128
MAXPG=6        # ~2.1 GiB/job with fast-attn -> 12.6 GiB of 24 GiB
A="train_mini""world"

on_gpu(){ pgrep -u "$USER" -af "$A" 2>/dev/null | grep -c -- "--device cuda:$1" || true; }
free_slots(){ echo $(( (MAXPG - $(on_gpu 0)) + (MAXPG - $(on_gpu 1)) )); }

launch(){    # variant seed n_obs grid epochs outdir
  local V="$1" SEED="$2" NOBS="$3" G="$4" EP="$5" OUT="$6"
  mkdir -p "$OUT"
  [ -f "$OUT/${V}_oracle.pt" ] && { echo "skip $V g$G n$NOBS s$SEED" >> "$LOG"; return; }
  local GPU=""
  while :; do
    if [ "$(on_gpu 0)" -lt "$MAXPG" ]; then GPU=0; break; fi
    if [ "$(on_gpu 1)" -lt "$MAXPG" ]; then GPU=1; break; fi
    sleep 60
  done
  echo "$(date +%H:%M) $V g$G n_obs=$NOBS s$SEED ${EP}ep -> cuda:$GPU" >> "$LOG"
  python3 -u -m mapformer.train_miniworld --variant "$V" --seed "$SEED" --oracle \
    --fast-attn --grid-size "$G" --n-obs "$NOBS" --n-steps $T --buffer-size $NBUF \
    --epochs "$EP" --n-batches $NB --batch-size $BS --d-model $DM --n-layers $NL \
    --n-heads $NH --n-workers $NW --schedule cosine --eval-trials $ETRIALS \
    --eval-lengths 512 1024 --device "cuda:$GPU" --output-dir "$OUT" \
    > "$R/${V}_g${G}_n${NOBS}_s${SEED}_e${EP}.log" 2>&1 &
  sleep 40
}

# ---- wait for the previous wave to clear -----------------------------------
echo "$(date +%H:%M) waiting for the n16 wave to finish" >> "$LOG"
while [ "$(pgrep -u "$USER" -f "$A" | wc -l)" -gt 0 ]; do sleep 120; done
echo "$(date +%H:%M) GPUs clear" >> "$LOG"

# ---- buffers (serial; EGL contexts saturate a GPU otherwise) ---------------
python3 -u -m mapformer.prebuild_buffers --grid-size 16 --n-obs 64 --seeds 0 1 2 \
    --n-steps $T --buffer-size $NBUF --eval-trials $ETRIALS --n-workers $NW \
    --oracle >> "$LOG" 2>&1

# ---- WAVE A: code control ---------------------------------------------------
echo "$(date +%H:%M) === WAVE A: fast-attn control at n_obs=256, 400ep ===" >> "$LOG"
for SEED in 0 1 2; do
  for V in Vanilla RoPE; do launch "$V" "$SEED" 256 32 400 "$R/ctl_n256/s$SEED"; done
done
wait
echo "$(date +%H:%M) wave A done" >> "$LOG"

python3 -u - "$R" "$REPO" >> "$LOG" 2>&1 <<'PYEOF'
import json, os, sys
import numpy as np
R, REPO = sys.argv[1], sys.argv[2]
E = []
for s in (0, 1, 2):
    try:
        v = json.load(open(f"{R}/ctl_n256/s{s}/Vanilla_oracle.json"))["512"]["nb_acc"]
        r = json.load(open(f"{R}/ctl_n256/s{s}/RoPE_oracle.json"))["512"]["nb_acc"]
        E.append(v - r)
    except Exception:
        pass
REF, TOL = 0.374, 0.074      # n=5 reference and its MDE at n=3
if not E:
    print("WAVE A PRODUCED NOTHING -- refusing to continue"); sys.exit(1)
m = float(np.mean(E))
print(f"wave A: fast-attn effect {m:+.3f} (n={len(E)}, per-seed "
      f"{['%+.3f' % x for x in E]}) vs reference {REF:+.3f}")
if abs(m - REF) <= TOL:
    open(f"{REPO}/.fastattn_licensed", "w").write(f"{m:.4f}")
    print(f"|delta| {abs(m-REF):.3f} <= {TOL:.3f} -> fast-attn LICENSED, wave B proceeds")
else:
    print(f"|delta| {abs(m-REF):.3f} > {TOL:.3f} -> fast-attn CHANGES THE RESULT. "
          f"Wave B is NOT run. Every fast-attn number must be treated as a "
          f"different code path, and the speedup is abandoned.")
PYEOF

[ -f "$REPO/.fastattn_licensed" ] || {
  echo "$(date +%H:%M) wave A failed the gate; stopping" >> "$LOG"
  touch "$REPO/.alias_followup_done"; exit 1; }

# ---- WAVE B: the two things that were actually missing ----------------------
echo "$(date +%H:%M) === WAVE B: grid16 (400ep) + n256 budget extension (800ep) ===" >> "$LOG"
for SEED in 0 1 2; do
  for V in Vanilla RoPE; do launch "$V" "$SEED" 64 16 400 "$R/g16/s$SEED"; done
done
for SEED in 0 1 2; do
  for V in Vanilla RoPE; do launch "$V" "$SEED" 256 32 800 "$R/n256_800/s$SEED"; done
done
wait

N_PT=$(find "$R" -name "*_oracle.pt" | wc -l)
echo "$(date +%H:%M) finished; $N_PT/18 checkpoints present" >> "$LOG"
touch "$REPO/.alias_followup_done"
echo "$(date) DONE" >> "$LOG"
