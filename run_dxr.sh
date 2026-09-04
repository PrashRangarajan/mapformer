#!/usr/bin/env bash
# The D x r rank-threshold batch. Pre-registered in mapformer_math.tex sec 2.5 /
# 6.2 and in paper_rank/sections/06_experiments_nd.tex.
#
# WHAT IT TESTS. Rank binds twice and the two halves predict effects in DIFFERENT
# cells of this table, which is why all three ranks are run rather than two:
#
#   GEOMETRY  the phase is confined to an r-dimensional subtorus, so packing N^D
#             positions into it forces  min separation ~ N^(-max(D/r, 1)).
#             Predicts an r=2 deficit ordered D=5 >> D=3 > D=2, tracking the
#             measured separation ratios 11x / 2.0x / 1.0x, and predicts NOTHING
#             for r=D vs r=D+2 -- both are above the threshold.
#   OPTIMISN  r=D is expressible but not reachable: r=2 in 2D costs 0.085 at 4x
#             training length with a skewed basis. Predicts a further r=D -> r=D+2
#             gap of roughly that size at EVERY D, not growing with D.
#
# FALSIFIERS. r=2 unimpaired at D=5 kills the geometric account and the
# explanation of the paper's Table 6. A flat r=D vs r=D+2 contrast everywhere
# makes the optimisation half a 2D artifact. An r=2 deficit that does not grow
# with D refutes the packing bound as the mechanism, since the exponent is the
# only thing D changes.
#
# GATES ALREADY PASSED (ND_GATES.md, run before this script existed): action-stream
# ngram at orders 1-5 reaches at most 0.536 against a measured 0.526 majority
# class; revisit rates 0.23 / 0.28 / 0.62; 30-79 scored positions per trajectory.
#
# State-space size is held near-constant (1024 / 1000 / 1024 cells) so only
# dimensionality moves. The cross-D revisit-rate difference is a real confound,
# so the PRIMARY contrast is WITHIN D; cross-D is descriptive.
set -u
# The package is importable from the PARENT of the repo, not the repo itself --
# `python3 -m mapformer.x` needs /home/prashr on the path. Every working script
# here does this; omitting it made all 8 first-wave jobs die instantly with
# ModuleNotFoundError, and because they died before the next poll the GPU
# balancer saw zero load and sent everything to cuda:0.
REPO="$(cd "$(dirname "$0")" && pwd)"; cd "$REPO/.."
R=$REPO/runs/dxr
LOG=$REPO/dxr.log
mkdir -p "$R"
echo "dxr start $(date)" >> "$LOG"

# D:grid:arms -- arms[0] is always the r=2 reference the contrasts are paired to
CONFIGS=("2:32:Vanilla,Vanilla_r4" "3:10:Vanilla,Vanilla_r3,Vanilla_r5" \
         "5:4:Vanilla,Vanilla_r5,Vanilla_r7")
SEEDS="0 1 2 3 4 5 6 7"
MAXPG=3

# Count only REAL interpreters. `pgrep -f` matches any shell whose command line
# mentions the pattern -- including the tool call that launched this script and
# every later one that discusses it -- which parked run_rope_canonical.sh in its
# wait loop for two hours against zero running jobs. `comm` is the executable
# name, so a shell cannot match however it quotes things.
busy() { ps -u "$USER" -o comm=,args= \
         | awk -v d="$1" '$1=="python3" && /mapformer\.train_variant/ && index($0,d)' \
         | wc -l; }

for SEED in $SEEDS; do
  for CFG in "${CONFIGS[@]}"; do
    D="${CFG%%:*}"; REST="${CFG#*:}"; N="${REST%%:*}"; ARMS="${REST#*:}"
    for V in ${ARMS//,/ }; do
      OUT="$R/D${D}/${V}_s${SEED}"; mkdir -p "$OUT"
      [ -f "$OUT/${V}.pt" ] && continue
      while :; do
        N0=$(busy "cuda:0"); N1=$(busy "cuda:1")
        # balance to the LESS loaded device -- a fill-first picker silently
        # becomes a single-GPU scheduler whenever the job count is <= MAXPG
        if [ "$N0" -le "$N1" ] && [ "$N0" -lt "$MAXPG" ]; then G=0; break; fi
        if [ "$N1" -lt "$MAXPG" ]; then G=1; break; fi
        if [ "$N0" -lt "$MAXPG" ]; then G=0; break; fi
        sleep 15
      done
      echo "$(date +%H:%M:%S) D=$D $V s$SEED -> cuda:$G" >> "$LOG"
      python3 -u -m mapformer.train_variant --variant "$V" --env nd \
        --n-dims "$D" --grid-size "$N" --seed "$SEED" \
        --epochs 300 --lr 1e-3 --n-batches 98 --batch-size 128 --n-steps 128 \
        --n-layers 1 --n-heads 2 --d-model 128 --n-landmarks 0 --schedule cosine \
        --data-workers 3 --device "cuda:$G" --output-dir "$OUT" \
        > "$R/D${D}_${V}_s${SEED}.log" 2>&1 &
      sleep 6
    done
  done
done
wait

# `wait` returns regardless of child success, so verify the artifacts rather than
# inferring them from the absence of a crash.
NCK=$(find "$R" -name '*.pt' | wc -l)
echo "$(date +%H:%M) $NCK/64 checkpoints" >> "$LOG"
if [ "$NCK" -lt 64 ]; then
  echo "INCOMPLETE -- evaluating what exists anyway" >> "$LOG"
fi

python3 -u -m mapformer.eval_nd --runs-dir "$R" \
  --configs "${CONFIGS[@]}" --seeds $SEEDS --lengths 128 512 \
  --n-trials 100 --device cuda:0 --out "$REPO/DXR_RANK_THRESHOLD.md" >> "$LOG" 2>&1

if [ -f "$REPO/DXR_RANK_THRESHOLD.md" ]; then
  touch "$REPO/.dxr_done"; echo "$(date) DONE" >> "$LOG"
else
  echo "$(date) EVAL FAILED -- no marker written" >> "$LOG"
fi
