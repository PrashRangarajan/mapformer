#!/usr/bin/env bash
# WAVE B of the aliasing follow-up, split out of run_alias_followup.sh to fix a
# GPU-BALANCE BUG in that script's slot picker.
#
# THE BUG. The picker was fill-first:
#     if on_gpu(0) < MAXPG -> GPU=0 ; elif on_gpu(1) < MAXPG -> GPU=1
# With MAXPG raised from 3 to 6 (fast-attn cut memory to ~2.1 GiB/job), GPU 0
# never fills at 6 jobs, so ALL of wave A landed on cuda:0 and cuda:1 sat at 0%
# for three hours. Wave B would have been worse, not better: its first 6 jobs are
# the SHORT grid-16 runs (400 ep) and its next 6 are the LONG budget-extension
# runs (800 ep), so GPU 0 would finish early and idle for hours while GPU 1 ground
# through all the long ones. ~9 h instead of ~5.5 h.
#
# THE FIX, two parts:
#   1. pick the LESS LOADED GPU rather than the first one with a free slot;
#   2. INTERLEAVE long and short jobs so neither device gets all the long ones.
#
# Wave A was left running untouched -- it is ~80% done and rebalancing it would
# throw away three hours to save one.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPO="$(cd "$(dirname "$0")" && pwd)"; cd "$REPO/.."
R="$REPO/runs/alias_follow"
LOG="$REPO/alias_waveb.log"; echo "wave B start $(date)" > "$LOG"
T=512; NBUF=24000; NB=180; BS=24; DM=256; NL=4; NH=4; NW=24; ETRIALS=128
MAXPG=6
A="train_mini""world"
on_gpu(){ pgrep -u "$USER" -af "$A" 2>/dev/null | grep -c -- "--device cuda:$1" || true; }

# wait for wave A to finish
echo "$(date +%H:%M) waiting for wave A" >> "$LOG"
while [ "$(pgrep -u "$USER" -f "$A" | wc -l)" -gt 0 ]; do sleep 120; done
echo "$(date +%H:%M) wave A clear" >> "$LOG"

# ---- the gate: did fast-attn reproduce the reference effect? ----------------
python3 -u - "$R" "$REPO" >> "$LOG" 2>&1 <<'PYEOF'
import json, sys
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
REF, TOL = 0.374, 0.074
if not E:
    print("WAVE A PRODUCED NOTHING -- refusing to continue"); sys.exit(1)
m = float(np.mean(E))
print(f"wave A: fast-attn effect {m:+.3f} (n={len(E)}, per-seed "
      f"{['%+.3f' % x for x in E]}) vs reference {REF:+.3f}")
if abs(m - REF) <= TOL:
    open(f"{REPO}/.fastattn_licensed", "w").write(f"{m:.4f}")
    print(f"|delta| {abs(m-REF):.3f} <= {TOL:.3f} -> LICENSED, wave B proceeds")
else:
    print(f"|delta| {abs(m-REF):.3f} > {TOL:.3f} -> fast-attn CHANGES THE RESULT; "
          f"wave B NOT run and the speedup is abandoned.")
PYEOF
[ -f "$REPO/.fastattn_licensed" ] || {
  echo "$(date +%H:%M) gate failed; stopping" >> "$LOG"
  touch "$REPO/.alias_followup_done"; exit 1; }

launch(){   # variant seed n_obs grid epochs outdir
  V="$1"; SEED="$2"; NOBS="$3"; G="$4"; EP="$5"; OUT="$6"
  mkdir -p "$OUT"
  [ -f "$OUT/${V}_oracle.pt" ] && { echo "skip $V g$G n$NOBS s$SEED" >> "$LOG"; return; }
  GPU=""
  while :; do
    N0=$(on_gpu 0); N1=$(on_gpu 1)
    # BALANCE, not fill-first: send the job to whichever device holds fewer.
    if [ "$N0" -le "$N1" ] && [ "$N0" -lt "$MAXPG" ]; then GPU=0; break; fi
    if [ "$N1" -lt "$N0" ] && [ "$N1" -lt "$MAXPG" ]; then GPU=1; break; fi
    if [ "$N0" -lt "$MAXPG" ]; then GPU=0; break; fi
    if [ "$N1" -lt "$MAXPG" ]; then GPU=1; break; fi
    sleep 60
  done
  echo "$(date +%H:%M) $V g$G n_obs=$NOBS s$SEED ${EP}ep -> cuda:$GPU (load $N0/$N1)" >> "$LOG"
  python3 -u -m mapformer.train_miniworld --variant "$V" --seed "$SEED" --oracle \
    --fast-attn --grid-size "$G" --n-obs "$NOBS" --n-steps $T --buffer-size $NBUF \
    --epochs "$EP" --n-batches $NB --batch-size $BS --d-model $DM --n-layers $NL \
    --n-heads $NH --n-workers $NW --schedule cosine --eval-trials $ETRIALS \
    --eval-lengths 512 1024 --device "cuda:$GPU" --output-dir "$OUT" \
    > "$R/${V}_g${G}_n${NOBS}_s${SEED}_e${EP}.log" 2>&1 &
  sleep 40
}

# INTERLEAVED: one long (800ep n256) and one short (400ep grid16) per step, so
# the long jobs are spread across both devices instead of piling onto one.
echo "$(date +%H:%M) === WAVE B (balanced, interleaved) ===" >> "$LOG"
for SEED in 0 1 2; do
  launch Vanilla "$SEED" 256 32 800 "$R/n256_800/s$SEED"
  launch Vanilla "$SEED"  64 16 400 "$R/g16/s$SEED"
  launch RoPE    "$SEED" 256 32 800 "$R/n256_800/s$SEED"
  launch RoPE    "$SEED"  64 16 400 "$R/g16/s$SEED"
done
wait

N_PT=$(find "$R" -name "*_oracle.pt" | wc -l)
echo "$(date +%H:%M) finished; $N_PT/18 checkpoints present" >> "$LOG"
touch "$REPO/.alias_followup_done"
echo "$(date) DONE" >> "$LOG"
