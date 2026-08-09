#!/usr/bin/env bash
# Same-batch compositional rerun: Vanilla, VanillaEM, VanillaEM_P0 trained
# TOGETHER under current code, 3 seeds. Required because EM_P0_COMP.md compared
# a fresh VanillaEM_P0 against 07-23 baselines (17 days apart) -- the same
# stale-baseline pattern that invalidated the lm200 leaderboard.
# NB: reference positional args as $1/$2 inside `local`, never as $s -- bash
# expands all words of a `local` command BEFORE assigning, so `local s=$2 o=${s}`
# dies under `set -u` (silently, if stderr is discarded).
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$REPO/.."
LOG="$REPO/em_comp_samebatch.log"; : > "$LOG"; echo "start $(date)" >> "$LOG"
OUT="$REPO/runs/comp_em_samebatch"
JOBS=(); for s in 0 1 2; do for v in Vanilla VanillaEM VanillaEM_P0; do JOBS+=("$v:$s"); done; done
run(){ local g=$1 j=$2; IFS=: read -r v s <<<"$j"
  local o="$OUT/seed$s"
  [ -f "$o/${v}.pt" ] && return
  echo "$(date +%H:%M) [gpu$g] $v s$s" >> "$LOG"
  python3 -u -m mapformer.train_compositional --variant "$v" --target motif \
    --n-steps 256 --epochs 50 --n-batches 156 --n-layers 3 --seed "$s" \
    --device "cuda:$g" --output-dir "$o" >> "$LOG" 2>&1; }
( for i in "${!JOBS[@]}"; do [ $((i%2)) -eq 0 ] && run 0 "${JOBS[$i]}"; done ) & P0=$!
( for i in "${!JOBS[@]}"; do [ $((i%2)) -eq 1 ] && run 1 "${JOBS[$i]}"; done ) & P1=$!
wait $P0 $P1
python3 -u -m mapformer.agg_comp_multiseed --runs-dir "$OUT" --seeds 0 1 2 \
  --variants Vanilla VanillaEM VanillaEM_P0 --lengths 256 512 1024 2048 \
  --n-traj 200 --batch 16 --device cuda:0 --out "$REPO/EM_COMP_SAMEBATCH.md" >> "$LOG" 2>&1
cd "$REPO"; git add EM_COMP_SAMEBATCH.md EM_COMP_SAMEBATCH.json run_em_comp_samebatch.sh 2>/dev/null
git diff --cached --quiet || { git commit -q -m "Same-batch compositional: Vanilla vs EM vs EM-single-p0

Supersedes EM_P0_COMP.md, which compared a fresh VanillaEM_P0 against 07-23
baselines. All three arms trained together under current code, 3 seeds.
Auto-committed; interpretation pending."; git push origin main >> "$LOG" 2>&1; }
echo "$(date +%H:%M) DONE" >> "$LOG"; touch "$REPO/.em_comp_samebatch_done"
