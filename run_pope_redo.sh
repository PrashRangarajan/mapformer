#!/usr/bin/env bash
# Re-run the PoPE arms of the fixed-hier-goal comparison with the CORRECTED
# d-frequency PoPE. The in-flight hiergoal_fixed run imported the old d/2 code,
# so its PoPE / MapPoPE_Hier checkpoints are stale and are deleted here.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$REPO/.."
LOG="$REPO/pope_redo.log"; : > "$LOG"
for _ in $(seq 1 240); do [ -f "$REPO/.hiergoal_fixed_done" ] && break; sleep 30; done
echo "$(date +%H:%M) hiergoal_fixed finished; removing stale d/2 PoPE checkpoints" >> "$LOG"
rm -f "$REPO"/runs/hiergoal_fixed/seed*/{PoPE,MapPoPE_Hier}_hiergoal.pt
VS=(PoPE MapPoPE_Hier)
JOBS=(); for s in 0 1 2; do for v in "${VS[@]}"; do JOBS+=("$v:$s"); done; done
run(){ local g=$1 j=$2; IFS=: read -r v s <<<"$j"
  local o="$REPO/runs/hiergoal_fixed/seed$s"
  [ -f "$o/${v}_hiergoal.pt" ] && return
  echo "$(date +%H:%M) [gpu$g] $v s$s" >> "$LOG"
  python3 -u -m mapformer.train_hier_goal --variant "$v" --seed "$s" --epochs 25 --n-batches 64 \
    --T-explore 64 --T-navigate 64 --eval-explore 64 128 192 256 --n-layers 3 \
    --interleave-path --device "cuda:$g" --output-dir "$o" >> "$LOG" 2>&1; }
( for i in "${!JOBS[@]}"; do [ $((i%2)) -eq 0 ] && run 0 "${JOBS[$i]}"; done ) & P0=$!
( for i in "${!JOBS[@]}"; do [ $((i%2)) -eq 1 ] && run 1 "${JOBS[$i]}"; done ) & P1=$!
wait $P0 $P1
python3 -u -m mapformer.agg_hiergoal --runs-dir "$REPO/runs/hiergoal_fixed" --seeds 0 1 2 \
  --variants Vanilla Hourglass_k2 PlainFlat PlainHourglass PoPE MapPoPE_Hier \
  --lengths 64 128 192 256 --out "$REPO/HIERGOAL_FIXED.md" >> "$LOG" 2>&1
cd "$REPO"; git add HIERGOAL_FIXED.md HIERGOAL_FIXED.json run_pope_redo.sh 2>/dev/null
git diff --cached --quiet || { git commit -q -m "Fixed hier-goal 2x2 with corrected d-frequency PoPE

Re-runs the PoPE arms after correcting PoPE to d bands (was d/2). Non-PoPE
variants were unaffected and are reused. Task uses the interleaved BFS path
(copy-previous shortcut 0.969 -> 0.327). Auto-committed; interpretation pending."
  git push origin main >> "$LOG" 2>&1; }
touch "$REPO/.pope_redo_done"
