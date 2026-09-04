#!/usr/bin/env bash
# SELECTIVE ROPE'S ANGLE GENERATOR vs MAPFORMER'S, on two tasks.
#
# Both papers put a content-dependent cumulative sum in the rotation PHASE, three
# days apart, neither citing the other. Selective RoPE (arXiv:2511.17388, ICLR 2026)
# computes  theta = temp * cumsum(conv1d(W_omega q));  MapFormer computes
# theta = omega * cumsum(W_out W_in x). This swaps the generator and changes
# nothing else.
#
# THREE OF THE FOUR DIFFERENCES ARE KNOBS THIS PROJECT ALREADY CARES ABOUT, so each
# gets its own arm and any difference can be ATTRIBUTED rather than just observed:
#
#   arm            bottleneck  conv  gate   params      vs Vanilla
#   RoPE           index position, no content angle at all   199,042
#   Vanilla        r=2         no    no     199,490     --      (MapFormer)
#   ConvAngle      r=2         YES   no     199,683     +193
#   NoBottleneck   none        no    no     207,363     +7,873
#   GateAngle      r=2         no    YES    207,683     +8,193
#   SRoPEGen       none        YES   YES    215,875     +16,385  (Selective RoPE)
#
# NOT PARAMETER-MATCHED, and it cannot be: removing the rank bottleneck IS the
# design difference, and it costs 8k parameters on a 199k model. Read the +8.2%
# alongside any win. ConvAngle is the one arm that is essentially free (+193), so it
# is the cleanest single-knob test.
#
# SCOPE, stated plainly. This swaps the GENERATOR while keeping MapFormer's
# PLACEMENT -- the angle is computed once from token embeddings before the blocks.
# At 1 layer that is close to their design, since q = W_Q LayerNorm(x) is itself a
# learned linear map of the token. It is NOT a faithful Selective RoPE: their
# per-head, per-layer, query-sourced angle is not reproduced, and at depth > 1 the
# difference is real. A negative result here does not refute their method.
#
# Two tasks, because they stress different things:
#   PARITY  the iterative task where MapFormer's cumsum is provably the right
#           primitive (path-int - index = +0.078 at L=128, 8/8 seeds). If SRoPE's
#           generator is strictly better engineering, it should also win here.
#   TORUS   the navigation task where the cumsum is a literal path integral. This
#           is where the conv should HURT if the mechanism matters: smoothing the
#           increment over a local window blurs a displacement that is exact.
#
# 6 arms x 16 seeds on parity (~15 min), then 6 x 8 on the torus (~1.1 h).
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPO="$(cd "$(dirname "$0")" && pwd)"; cd "$REPO/.."
R="$REPO/runs/selective"; mkdir -p "$R"
LOG="$REPO/selective.log"; echo "selective start $(date)" > "$LOG"
ARMS="RoPE Vanilla ConvAngle NoBottleneck GateAngle SRoPEGen"
MAXPG=5
slot(){ P="$1"
  while :; do
    N0=$(pgrep -u "$USER" -af "$P" 2>/dev/null | grep -c -- "--device cuda:0" || true)
    N1=$(pgrep -u "$USER" -af "$P" 2>/dev/null | grep -c -- "--device cuda:1" || true)
    if [ "$N0" -le "$N1" ] && [ "$N0" -lt "$MAXPG" ]; then echo 0; return; fi
    if [ "$N1" -lt "$N0" ] && [ "$N1" -lt "$MAXPG" ]; then echo 1; return; fi
    if [ "$N0" -lt "$MAXPG" ]; then echo 0; return; fi
    if [ "$N1" -lt "$MAXPG" ]; then echo 1; return; fi
    sleep 10
  done
}

# ---------- parity ----------
A="train_algo""rithmic"
for SEED in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
  for V in $ARMS; do
    OUT="$R/parity/${V}_s${SEED}"; mkdir -p "$OUT"
    [ -f "$OUT/${V}_parity.json" ] && continue
    G=$(slot "$A")
    echo "$(date +%H:%M:%S) parity $V s$SEED -> cuda:$G" >> "$LOG"
    python3 -u -m mapformer.train_algorithmic --variant "$V" --task parity \
      --seed "$SEED" --epochs 300 --n-batches 50 --batch-size 128 \
      --train-length 16 --eval-lengths 16 32 64 128 256 --lr 1e-3 \
      --d-model 128 --n-heads 2 --n-layers 1 --schedule cosine \
      --device "cuda:$G" --output-dir "$OUT" \
      > "$R/parity_${V}_s${SEED}.log" 2>&1 &
    sleep 3
  done
done
wait
echo "$(date +%H:%M) parity done: $(find "$R/parity" -name '*.json' | wc -l)/96" >> "$LOG"

# ---------- torus ----------
B="train_var""iant"
for SEED in 0 1 2 3 4 5 6 7; do
  for V in $ARMS; do
    OUT="$R/torus/p0/${V}_s${SEED}"; mkdir -p "$OUT"
    [ -f "$OUT/${V}.pt" ] && continue
    G=$(slot "$B")
    echo "$(date +%H:%M:%S) torus $V s$SEED -> cuda:$G" >> "$LOG"
    python3 -u -m mapformer.train_variant --variant "$V" --seed "$SEED" \
      --epochs 300 --lr 1e-3 --n-batches 98 --batch-size 128 --n-steps 128 \
      --n-layers 1 --n-heads 2 --d-model 128 --n-landmarks 0 --schedule cosine \
      --data-workers 3 --device "cuda:$G" --output-dir "$OUT" \
      > "$R/torus_${V}_s${SEED}.log" 2>&1 &
    sleep 6
  done
done
wait
echo "$(date +%H:%M) torus done: $(find "$R/torus" -name '*.pt' | wc -l)/48" >> "$LOG"

python3 -u -m mapformer.eval_noise_refine --runs-dir "$R/torus" \
  --variants $ARMS --noises 0.0 --seeds 0 1 2 3 4 5 6 7 \
  --lengths 128 512 1024 --n-trials 100 --device cuda:0 \
  --out "$REPO/_SELECTIVE_TORUS.md" >> "$LOG" 2>&1
python3 -u -m mapformer.agg_selective --repo "$REPO" --runs-dir "$R" \
  --out "$REPO/SELECTIVE_ROPE.md" >> "$LOG" 2>&1
touch "$REPO/.selective_done"; echo "$(date) DONE" >> "$LOG"
