#!/usr/bin/env bash
# Re-run the 7 factorial arms that died of CUDA OOM. Root cause: the launcher
# capped TOTAL concurrent jobs (jobs -rp >= 6), not PER-GPU, so as arms finished
# unevenly up to 4 landed on one GPU and OOM'd (CLAUDE.md: "per-GPU concurrency
# cap, not total"). This scheduler enforces a STRICT <=2 arms per GPU (d=256 =>
# ~7 GB each => ~14 GB/GPU, safe under 24 GB). Buffers are already cached, so
# arms start immediately (no rebuild).
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPO="$(cd "$(dirname "$0")" && pwd)"; cd "$REPO/.."
R="$REPO/runs/miniworld_fixed"
LOG="$REPO/miniworld_missing.log"; echo "missing-rerun start $(date)" > "$LOG"
G=8; T=512; NBUF=3000; EP=100; NB=180; BS=24; DMODEL=256; NLAYERS=4; NHEADS=4
MAXPG=2                                    # <= this many arms per GPU

# each entry: "variant seed alloflag"  (alloflag = "" or "--allocentric")
ARMS=(
  "PoPE-Flat 0 --allocentric"
  "Vanilla 1 --allocentric"
  "MapPoPE-Flat 1 "
  "RoPE 1 --allocentric"
  "MapPoPE-Flat 2 "
  "RoPE 2 --allocentric"
  "PoPE-Flat 2 "
)
declare -a PIDG0=() PIDG1=()             # live PIDs per GPU

alive() { local out=(); for p in "$@"; do kill -0 "$p" 2>/dev/null && out+=("$p"); done; echo "${out[@]:-}"; }

for entry in "${ARMS[@]}"; do
  read -r V S ALLO <<<"$entry"
  TAG=$([ -n "${ALLO:-}" ] && echo allo || echo raw)
  OUT="$R/s${S}"
  if [ -f "$OUT/${V}_${TAG}.pt" ]; then echo "skip ${V}_s${S}_${TAG} (exists)" >> "$LOG"; continue; fi
  # wait for a GPU slot (<=MAXPG alive on that GPU)
  while :; do
    PIDG0=($(alive "${PIDG0[@]:-}")); PIDG1=($(alive "${PIDG1[@]:-}"))
    if [ "${#PIDG0[@]}" -lt "$MAXPG" ]; then GPU=0; break; fi
    if [ "${#PIDG1[@]}" -lt "$MAXPG" ]; then GPU=1; break; fi
    sleep 15
  done
  echo "$(date +%H:%M) ${V}_s${S}_${TAG} -> cuda:$GPU" >> "$LOG"
  python3 -u -m mapformer.train_miniworld --variant "$V" --seed "$S" ${ALLO:-} --fixed-map \
    --grid-size $G --n-steps $T --buffer-size $NBUF --epochs $EP --n-batches $NB \
    --batch-size $BS --d-model $DMODEL --n-layers $NLAYERS --n-heads $NHEADS \
    --eval-lengths 512 1024 --device "cuda:$GPU" --output-dir "$OUT" \
    > "$R/${V}_s${S}_${TAG}.log" 2>&1 &
  PID=$!
  [ "$GPU" -eq 0 ] && PIDG0+=("$PID") || PIDG1+=("$PID")
  sleep 3
done
wait
echo "$(date +%H:%M) all rerun arms exited; verifying" >> "$LOG"

# VERIFY every expected json exists (do NOT trust `wait`'s exit code -- CLAUDE.md)
MISS=0
for entry in "${ARMS[@]}"; do
  read -r V S ALLO <<<"$entry"; TAG=$([ -n "${ALLO:-}" ] && echo allo || echo raw)
  [ -f "$R/s${S}/${V}_${TAG}.json" ] || { echo "STILL MISSING ${V}_s${S}_${TAG}" >> "$LOG"; MISS=$((MISS+1)); }
done
echo "missing after rerun: $MISS" >> "$LOG"
if [ "$MISS" -eq 0 ]; then
  python3 -u -m mapformer.agg_miniworld --runs-dir "$R" --length 512 \
    --out "$REPO/MINIWORLD_FIXED_RESULTS.md" >> "$LOG" 2>&1
  touch "$REPO/.mw_missing_done"
fi
echo "$(date) DONE (miss=$MISS)" >> "$LOG"
