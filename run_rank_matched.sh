#!/usr/bin/env bash
# Rank at matched length. Pre-registration: RANK_MATCHED_PREREG.md (Amendments 1-2).
#   300-epoch batch:        bash run_rank_matched.sh
#   900-epoch pilot:        EPOCHS=900 TAG=_e900 SEEDS="0 1" PILOT=1 bash run_rank_matched.sh
#   900-epoch full batch:   EPOCHS=900 TAG=_e900 bash run_rank_matched.sh   (all 8 seeds; the
#                           pilot's seeds 0-1 are reused, verified by epochs and code md5)
#   continuation (Amdt 3):  EPOCHS=900 TAG=_e900c INIT_TAG=_e900 bash run_rank_matched.sh
#                           (each run starts from runs/rank_matched_e900's weights: a warm
#                           restart of the same recipe on a fresh data stream)
set -uo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
EPOCHS="${EPOCHS:-300}"; TAG="${TAG:-}"; PILOT="${PILOT:-0}"; INIT_TAG="${INIT_TAG:-}"
R="$REPO/runs/rank_matched${TAG}"; mkdir -p "$R/p0"
LOG="$REPO/rank_matched${TAG}.log"
exec 9>"$REPO/.run_rank_matched.lock"
# a refused launch must be visible: nohup sends stdout to /dev/null
flock -n 9 || { echo "$(date) REFUSED: another run_rank_matched.sh holds the lock" >> "$LOG"; exit 1; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO/.."
echo "start $(date) epochs=$EPOCHS pilot=$PILOT" >> "$LOG"
# markers from an earlier invocation (e.g. the pilot) must not read as this run being done
rm -f "$R/.train_done" "$REPO/.rank_matched${TAG}_done"
# the training code must be the code any reused checkpoint was trained with
MODS="model.py model_rank.py train.py train_variant.py environment.py data_parallel.py"
( cd "$REPO" && md5sum $MODS ) > "$R/.code_md5.now"
if [ -f "$R/code_md5.txt" ]; then
  cmp -s "$R/code_md5.txt" "$R/.code_md5.now" || { echo "ABORT: training code changed since $R/code_md5.txt" >> "$LOG"; exit 1; }
else
  mv "$R/.code_md5.now" "$R/code_md5.txt"
fi
ARMS="Vanilla Vanilla_r4"; SEEDS="${SEEDS:-0 1 2 3 4 5 6 7}"
MAXPG=5
# count REAL trainers on a device (ps comm=python3; never pgrep -f, which matches shells)
ntrain(){ ps -u "$USER" -o comm=,args= | awk -v d="--device cuda:$1" '$1=="python3" && /mapformer\.train_variant/ && index($0,d)' | wc -l; }
freemem(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1"; }
# pick the less loaded device that has a slot AND ~4 GiB free (explicit attention at
# 16 x 2 x 2047^2 fp32 is ~3 GB per job)
pick(){ local best="" bn=99; for g in 0 1; do n=$(ntrain $g)
          if [ "$n" -lt "$MAXPG" ] && [ "$(freemem $g)" -gt 4500 ] && [ "$n" -lt "$bn" ]; then best=$g; bn=$n; fi
        done; echo "$best"; }
# a checkpoint is reusable only if it was trained at the requested budget and length
matches(){ python3 - "$1" "$EPOCHS" "$2" <<'PY'
import sys, torch
c = torch.load(sys.argv[1], map_location="cpu", weights_only=False)["config"]
ok = c.get("epochs") == int(sys.argv[2]) and c.get("n_steps") == 1024 and c.get("batch_size") == 16
ok = ok and (c.get("init_from") or "") == sys.argv[3]
sys.exit(0 if ok else 1)
PY
}
init_for(){ [ -n "$INIT_TAG" ] && echo "$REPO/runs/rank_matched${INIT_TAG}/p0/$1_s$2/$1.pt" || echo ""; }
for SEED in $SEEDS; do
  for V in $ARMS; do
    OUT="$R/p0/${V}_s${SEED}"; mkdir -p "$OUT"
    if [ -f "$OUT/${V}.pt" ]; then
      matches "$OUT/${V}.pt" "$(init_for $V $SEED)" || { echo "ABORT: $OUT/${V}.pt exists but not at epochs=$EPOCHS n_steps=1024 bs=16 init=$(init_for $V $SEED)" >> "$LOG"; exit 1; }
      echo "skip $V s$SEED (done, config matches)" >> "$LOG"; continue
    fi
    INIT=$(init_for $V $SEED); EXTRA=""
    if [ -n "$INIT" ]; then
      [ -f "$INIT" ] || { echo "ABORT: init checkpoint missing $INIT" >> "$LOG"; exit 1; }
      EXTRA="--init-from $INIT --data-seed-offset 1 --save-full-state"
    fi
    G=""; while [ -z "$G" ]; do G=$(pick); [ -z "$G" ] && sleep 30; done
    echo "$(date +%H:%M:%S) $V s$SEED -> cuda:$G" >> "$LOG"
    OMP_NUM_THREADS=4 setsid nohup python3 -u -m mapformer.train_variant --variant "$V" --seed "$SEED" \
      --epochs "$EPOCHS" --lr 1e-3 --n-batches 98 --batch-size 16 --n-steps 1024 \
      --n-layers 1 --n-heads 2 --d-model 128 --n-landmarks 0 --schedule cosine \
      --data-workers 3 --device "cuda:$G" --output-dir "$OUT" $EXTRA \
      > "$R/${V}_s${SEED}.log" 2>&1 &
    sleep 45     # let it claim its memory before the next pick
  done
done
while [ "$(ps -u "$USER" -o comm=,args= | awk -v r="$R/" '$1=="python3" && /mapformer\.train_variant/ && index($0,r)' | wc -l)" -gt 0 ]; do sleep 60; done
missing=0
for SEED in $SEEDS; do for V in $ARMS; do
  [ -f "$R/p0/${V}_s${SEED}/${V}.pt" ] || { echo "MISSING $V s$SEED" >> "$LOG"; missing=$((missing+1)); }
done; done
echo "$(date +%H:%M) missing=$missing" >> "$LOG"
[ "$missing" -eq 0 ] || { echo "not all checkpoints present -- evaluation NOT run" >> "$LOG"; exit 1; }
touch "$R/.train_done"
if [ "$PILOT" = 1 ]; then
  # a pilot reads TRAINING LOSS ONLY (Amendment 1-2)
  python3 -u -m mapformer.analyze_rank_matched --tag "$TAG" --seeds $SEEDS --classify-only >> "$LOG" 2>&1 \
    || { echo "classification FAILED" >> "$LOG"; exit 1; }
  touch "$REPO/.rank_matched${TAG}_done"; echo "$(date) PILOT DONE (no eval)" >> "$LOG"; exit 0
fi
fail(){ echo "$(date) FAILED: $1 -- done marker NOT set" >> "$LOG"; exit 1; }
python3 -u -m mapformer.eval_noise_refine --runs-dir "$R" --variants $ARMS --noises 0.0 \
  --seeds $SEEDS --lengths 512 1024 2048 --n-trials 100 --device cuda:0 \
  --title "Rank at matched length: trained AND tested at T=1024 (epochs=$EPOCHS)" \
  --out "$REPO/RANK_MATCHED${TAG}.md" >> "$LOG" 2>&1 || fail eval_noise_refine
python3 -u -m mapformer.eval_rank_strata --runs-dir "$R" --variants $ARMS --seeds $SEEDS \
  --lengths 1024 2048 --n-trials 100 --device cuda:0 \
  --check-json "$REPO/RANK_MATCHED${TAG}.json" --out "$REPO/RANK_MATCHED${TAG}_STRATA.json" >> "$LOG" 2>&1 || fail eval_rank_strata
[ -f "$REPO/RANK_SWEEP_STRATA.json" ] || python3 -u -m mapformer.eval_rank_strata --runs-dir "$REPO/runs/rank_sweep" --variants $ARMS --seeds $SEEDS \
  --lengths 1024 2048 --n-trials 100 --device cuda:1 \
  --check-json "$REPO/RANK_SWEEP.json" --out "$REPO/RANK_SWEEP_STRATA.json" >> "$LOG" 2>&1 || fail "old strata"
python3 -u -m mapformer.probe_action_geometry --runs-dir "$R/p0" --arms $ARMS --seeds $SEEDS --device cuda:0 \
  --out "$REPO/RANK_MATCHED${TAG}_GEOMETRY.md" >> "$LOG" 2>&1 || fail probe_action_geometry
python3 -u -m mapformer.analyze_rank_matched --tag "$TAG" --seeds $SEEDS > "$REPO/RANK_MATCHED${TAG}_ANALYSIS.txt" 2>&1 || fail analyze
touch "$REPO/.rank_matched${TAG}_done"; echo "$(date) DONE" >> "$LOG"
