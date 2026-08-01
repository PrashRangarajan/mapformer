#!/usr/bin/env bash
# Content-axis top-up to n=8 (seeds 3-7) for the 6 variants whose n=3 rankings
# are unresolved, plus the clock task to n=8. Auto-commits + pushes at the end.
#   contested trio : Hourglass_k2, Hourglass_CoarsePI, MapPoPE_Hier  (0.423/0.451/0.455 +-0.12)
#   hierarchy claim: PlainHourglass, PlainFlat, HourglassFlat3
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$REPO/.."
LOG="$REPO/topup_n8.log"; : > "$LOG"; echo "start $(date)" >> "$LOG"
CV=(Hourglass_k2 Hourglass_CoarsePI MapPoPE_Hier PlainHourglass PlainFlat HourglassFlat3)
KV=(Vanilla Hourglass_k2 Hourglass_CoarsePI PoPE MapPoPE_Hier MapPoPE_CoarseIdx PlainFlat)

cpj(){ local g=$1 v=$2 s=$3 o="$REPO/runs/comp_multiseed/seed$3"
  [ -f "$o/${v}.pt" ] && return
  echo "$(date +%H:%M) [gpu$g] comp $v s$s" >> "$LOG"
  python3 -u -m mapformer.train_compositional --variant "$v" --target motif --n-steps 256 \
    --epochs 50 --n-batches 156 --n-layers 3 --seed "$s" --device cuda:$g --output-dir "$o" >> "$LOG" 2>&1; }
ckj(){ local g=$1 v=$2 s=$3 o="$REPO/runs/clock_scan/seed$3"
  [ -f "$o/${v}_clock.pt" ] && return
  echo "$(date +%H:%M) [gpu$g] clock $v s$s" >> "$LOG"
  python3 -u -m mapformer.train_clock --variant "$v" --seed "$s" --epochs 25 --n-batches 64 \
    --T-explore 64 --T-navigate 64 --eval-explore 64 128 192 256 --n-layers 3 \
    --device cuda:$g --output-dir "$o" >> "$LOG" 2>&1; }

JOBS=()
for s in 3 4 5 6 7; do for v in "${CV[@]}"; do JOBS+=("c:$v:$s"); done; done
for s in 3 4 5 6 7; do for v in "${KV[@]}"; do JOBS+=("k:$v:$s"); done; done
rj(){ local g=$1 j=$2; IFS=: read -r t v s <<<"$j"; [ "$t" = c ] && cpj "$g" "$v" "$s" || ckj "$g" "$v" "$s"; }
( for i in "${!JOBS[@]}"; do [ $((i%2)) -eq 0 ] && rj 0 "${JOBS[$i]}"; done; echo "$(date +%H:%M) GPU0 DONE">>"$LOG" ) & P0=$!
( for i in "${!JOBS[@]}"; do [ $((i%2)) -eq 1 ] && rj 1 "${JOBS[$i]}"; done; echo "$(date +%H:%M) GPU1 DONE">>"$LOG" ) & P1=$!
wait $P0 $P1
echo "$(date +%H:%M) aggregating" >> "$LOG"
python3 -u -m mapformer.agg_comp_multiseed --runs-dir "$REPO/runs/comp_multiseed" --seeds 0 1 2 3 4 5 6 7 \
  --variants Vanilla VanillaEM Hourglass_k2 Hourglass_CoarseIdx Hourglass_CoarsePI PoPE MapPoPE \
             MapPoPE_Hier MapPoPE_CoarseIdx HourglassFlat3 PlainHourglass PlainFlat \
  --lengths 256 512 1024 2048 --n-traj 200 --batch 16 --device cuda:0 \
  --out "$REPO/COMPOSITIONAL_MULTISEED.md" >> "$LOG" 2>&1
python3 -u -m mapformer.agg_clock --runs-dir "$REPO/runs/clock_scan" --seeds 0 1 2 3 4 5 6 7 \
  --variants "${KV[@]}" --lengths 64 128 192 256 --out "$REPO/CLOCK_SCAN.md" >> "$LOG" 2>&1
echo "$(date +%H:%M) aggregated -> committing" >> "$LOG"
cd "$REPO"
git add COMPOSITIONAL_MULTISEED.md COMPOSITIONAL_MULTISEED.json CLOCK_SCAN.md CLOCK_SCAN.json 2>/dev/null
if ! git diff --cached --quiet; then
  git commit -q -m "Content axis + clock to n=8 (seeds 3-7)

Tops up the 6 compositional variants whose n=3 rankings were unresolved
(Hourglass_k2, CoarsePI, MapPoPE_Hier were within 0.03 at +-0.12) plus the
clock task, after n=8 showed CoarseIdx's apparent stability was 3-seed luck.
Auto-committed by run_topup_n8.sh; interpretation pending review."
  git push origin main >> "$LOG" 2>&1
fi
echo "$(date +%H:%M) DONE" >> "$LOG"; touch "$REPO/.topup_n8_done"
