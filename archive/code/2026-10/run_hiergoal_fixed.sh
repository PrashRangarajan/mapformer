#!/usr/bin/env bash
# CORE COMPARISON re-run on the FIXED hier-goal task (interleaved BFS path).
# The original task was ~94% solvable by copying the previous action; the
# interleave drops that shortcut to 0.327 while keeping the oracle ceiling at
# 1.00, so accuracy above ~0.33 now genuinely requires knowing position.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$REPO/.."
LOG="$REPO/hiergoal_fixed.log"; : > "$LOG"; echo "start $(date)" >> "$LOG"
VS=(Vanilla Hourglass_k2 PlainFlat PlainHourglass PoPE MapPoPE_Hier)
JOBS=(); for s in 0 1 2; do for v in "${VS[@]}"; do JOBS+=("$v:$s"); done; done
run(){ local g=$1 j=$2; IFS=: read -r v s <<<"$j"
  local o="$REPO/runs/hiergoal_fixed/seed$s"
  [ -f "$o/${v}_hiergoal.pt" ] && return
  echo "$(date +%H:%M) [gpu$g] $v s$s" >> "$LOG"
  python3 -u -m mapformer.train_hier_goal --variant "$v" --seed "$s" --epochs 25 --n-batches 64 \
    --T-explore 64 --T-navigate 64 --eval-explore 64 128 192 256 --n-layers 3 \
    --interleave-path --device "cuda:$g" --output-dir "$o" >> "$LOG" 2>&1; }
( for i in "${!JOBS[@]}"; do [ $((i%2)) -eq 0 ] && run 0 "${JOBS[$i]}"; done; echo "$(date +%H:%M) GPU0 DONE">>"$LOG" ) & P0=$!
( for i in "${!JOBS[@]}"; do [ $((i%2)) -eq 1 ] && run 1 "${JOBS[$i]}"; done; echo "$(date +%H:%M) GPU1 DONE">>"$LOG" ) & P1=$!
wait $P0 $P1
python3 -u -m mapformer.agg_hiergoal --runs-dir "$REPO/runs/hiergoal_fixed" --seeds 0 1 2 \
  --variants "${VS[@]}" --lengths 64 128 192 256 --out "$REPO/HIERGOAL_FIXED.md" >> "$LOG" 2>&1
cd "$REPO"; git add environment_hier_goal.py train_hier_goal.py run_hiergoal_fixed.sh HIERGOAL_FIXED.md HIERGOAL_FIXED.json 2>/dev/null
git diff --cached --quiet || { git commit -q -m "Re-run core comparison on FIXED hier-goal (interleaved BFS)

The original task was ~94% solvable by copying the previous action (BFS returns
runs). Deterministic balanced interleave drops that shortcut to 0.327 while
keeping shortest-path length and a 1.00 oracle ceiling, so accuracy now requires
knowing the remaining displacement. Re-runs the 2x2 plus PoPE/MapPoPE-Hier.
Auto-committed; interpretation pending review."; git push origin main >> "$LOG" 2>&1; }
echo "$(date +%H:%M) DONE" >> "$LOG"; touch "$REPO/.hiergoal_fixed_done"
