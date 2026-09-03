#!/usr/bin/env bash
# DOES HIERARCHY HELP PARITY? A test of this project's own stated principle.
#
# The standing principle (project memory): hierarchy helps only when a lossy
# summary is a SUFFICIENT STATISTIC, which is why it loses on exact-recall tasks --
# our navigation benchmark is exact recall, and hierarchy lost there (MotifSeg 0.254
# vs flat 0.281; MotifSeg-FR 0.157; enwik8 a null on bpc).
#
# Parity is the case the principle says should WIN, and it says so sharply: parity
# is a TREE REDUCTION, parity(a,b,c,d) = parity(parity(a,b), parity(c,d)). The
# partial parity of a pooled pair is EXACTLY a sufficient statistic, with no loss at
# all, and a log-depth tree should extrapolate in length far better than a linear
# scan. If hierarchy does not help here it does not help anywhere, and the principle
# is wrong rather than merely narrow.
#
# 2x2 at EXACT parameter parity within each row:
#   index    RoPE n_layers=3  595,586   vs  PlainHourglass  595,586
#   path-int HourglassFlat3   596,034   vs  Hourglass_k2    596,034
# The flat twins are the correct controls: Hourglass variants IGNORE --n-layers and
# are always the 3-block scaffold, so pairing them against a 1-layer model would be
# a 3x capacity confound (a mistake this repo has already made once).
#
# The index-row flat arm is REUSED from the frontier batch -- same trainer, same
# recipe, same seeds, same code, launched hours apart. That is a rule-3 risk taken
# knowingly: it is the identical command line, and re-running it is cheap, so it IS
# re-run here rather than read from the old directory.
#
# PRE-REGISTERED, before the numbers:
#   hierarchy helps parity in BOTH rows -> the sufficient-statistic principle is
#       predictive, and the navigation losses were about the TASK not the mechanism.
#   hierarchy helps in NEITHER -> the principle fails on the one task where the
#       summary is provably lossless, and should be retired rather than narrowed.
#   hierarchy helps only the path-integrated row -> it interacts with the position
#       code, which nothing so far predicts.
# Length extrapolation is the sharper half: the tree argument predicts a FLATTER
# decay from L=16 to L=256, not merely a higher score at L=16.
#
# 4 arms x 16 seeds = 64 runs at ~90 s -- about 10 minutes.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPO="$(cd "$(dirname "$0")" && pwd)"; cd "$REPO/.."
R="$REPO/runs/hier_parity"; mkdir -p "$R"
LOG="$REPO/hier_parity.log"; echo "hier-parity start $(date)" > "$LOG"
A="train_algo""rithmic"; MAXPG=5
on_gpu(){ pgrep -u "$USER" -af "$A" 2>/dev/null | grep -c -- "--device cuda:$1" || true; }
for SEED in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
  for SPEC in "RoPE 3" "PlainHourglass 1" "HourglassFlat3 1" "Hourglass_k2 1"; do
    set -- $SPEC; V="$1"; NL="$2"
    OUT="$R/${V}_s${SEED}"; mkdir -p "$OUT"
    [ -f "$OUT/${V}_parity.json" ] && continue
    GPU=""
    while :; do
      N0=$(on_gpu 0); N1=$(on_gpu 1)
      if [ "$N0" -le "$N1" ] && [ "$N0" -lt "$MAXPG" ]; then GPU=0; break; fi
      if [ "$N1" -lt "$N0" ] && [ "$N1" -lt "$MAXPG" ]; then GPU=1; break; fi
      if [ "$N0" -lt "$MAXPG" ]; then GPU=0; break; fi
      if [ "$N1" -lt "$MAXPG" ]; then GPU=1; break; fi
      sleep 10
    done
    echo "$(date +%H:%M:%S) $V L$NL s$SEED -> cuda:$GPU" >> "$LOG"
    python3 -u -m mapformer.train_algorithmic --variant "$V" --task parity \
      --seed "$SEED" --epochs 300 --n-batches 50 --batch-size 128 \
      --train-length 16 --eval-lengths 16 32 64 128 256 --lr 1e-3 \
      --d-model 128 --n-heads 2 --n-layers "$NL" --schedule cosine \
      --device "cuda:$GPU" --output-dir "$OUT" \
      > "$R/${V}_s${SEED}.log" 2>&1 &
    sleep 3
  done
done
wait
N=$(find "$R" -name '*.json' | wc -l); echo "$(date +%H:%M) $N/64 results" >> "$LOG"
python3 -u -m mapformer.agg_hier_parity --runs-dir "$R" \
  --out "$REPO/HIER_PARITY.md" >> "$LOG" 2>&1
touch "$REPO/.hier_parity_done"; echo "$(date) DONE" >> "$LOG"
