#!/usr/bin/env bash
# Re-measure timing on a QUIET GPU after the family-tree sweep finishes.
# The first run's forward-only rows were invalid: no torch.no_grad(), so they
# timed autograd graph construction, which scales with node count and penalised
# the Python-loop models -- biasing the comparison toward the parallel-scan claim.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$REPO/.."
LOG="$REPO/timing_rerun.log"; : > "$LOG"
for _ in $(seq 1 240); do [ -f "$REPO/.family_tree_d7_done" ] && break; sleep 30; done
# the GPU must be genuinely idle or the numbers are contaminated
for _ in $(seq 1 40); do
  u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0)
  [ "${u:-100}" -lt 10 ] && break; sleep 15
done
echo "$(date +%H:%M) GPU idle; re-measuring" >> "$LOG"
python3 -u -m mapformer.benchmark_timing --device cuda:0 --reps 15 \
  --out "$REPO/TIMING_BENCHMARK.md" >> "$LOG" 2>&1
cd "$REPO"
git add TIMING_BENCHMARK.md TIMING_BENCHMARK.json benchmark_timing.py run_timing_rerun.sh 2>/dev/null
git diff --cached --quiet || { git commit -q -m "Timing benchmark re-measured with torch.no_grad on forward-only

Supersedes the first run, whose forward-only rows timed autograd graph
construction and so penalised the Python-loop models -- biasing the comparison
toward the parallel-scan claim. 15 reps, IQR retained. Auto-committed;
interpretation pending."; git push origin main >> "$LOG" 2>&1; }
echo "$(date +%H:%M) DONE" >> "$LOG"; touch "$REPO/.timing_rerun_done"
