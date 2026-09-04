#!/usr/bin/env bash
# IS THE "BETWEEN" JUST A WIDER BOTTLENECK?
#
# Selective RoPE's two parameter-adding knobs each help on the torus and are
# INDISTINGUISHABLE FROM EACH OTHER (GateAngle - NoBottleneck = +0.018 / +0.028,
# both inside their MDEs), so the likely explanation is capacity. The
# gate-as-token-suppressor hypothesis was tested and falsified directly
# (GATE_PROBE.md: 1.35x on the torus where it helps, 1.54x on parity where it
# hurts, and nowhere near the ~0 suppression the story needed).
#
# If it is capacity, MapFormer already owns the knob, far cheaper:
#
#   r=2   204,373      --        the paper's "for instance r = 2"
#   r=4   204,757    +384
#   r=8   205,525  +1,152
#   r=16  207,061  +2,688
#   r=32  210,133  +5,760
#   ---- for comparison: the sigmoid gate cost +8,193 and bought +0.086 ----
#
# WHAT THE PAPER ACTUALLY SAYS, and it is not "r=2 was a guess". App. A.7:
#   "the internal projection W_in in R^{d x r} (r << d) maps the high-dimensional
#    input X to a low-dimensional Delta_in (for instance, in a 2D environment,
#    Delta_in could be the 2D movement vector Delta_tk, where r = 2)"
# and App. A.1: "a in R^r (e.g. in 2D, r = 2 and 'move right')". So r is meant to
# be the DIMENSIONALITY OF THE ACTION SPACE. The bottleneck is an inductive bias,
# not a capacity budget: it forces the angle to depend on exactly a 2-vector,
# which is what a 2D displacement is. It checks out arithmetically -- the torus's
# four actions are +/-x and +/-y, so two dimensions span them exactly, and
# observations need Delta = 0, which lies in any subspace. r=2 is SUFFICIENT BY
# CONSTRUCTION here.
#
# PRE-REGISTERED, with the paper's prediction named first because it is the
# hypothesis this run is most likely to confirm:
#   accuracy FLAT in r -> the paper's inductive-bias story is right, two dimensions
#       suffice for a 2D environment, and CAPACITY IS NOT what Selective RoPE's
#       ~8k-parameter knobs bought on the torus. The gate's +0.086 then needs
#       another explanation, and the leading candidate is OPTIMISATION: a full-rank
#       map is better conditioned than a rank-2 one, which is a different claim
#       from "more capacity". That prediction is testable the way this project has
#       twice before -- an optimisation artifact shrinks when the baseline trains
#       better (the loop went +0.373 -> +0.146 across recipes, and its
#       training-length win was +0.052 raw against +0.006 loss-matched).
#   accuracy RISES with r -> the bottleneck is under-provisioned even for a 2D
#       environment, contradicting the paper's own justification, and the cheapest
#       r that reaches the gate's +0.086 is the answer to "something between
#       MapFormer and Selective RoPE". Report that r and the parameter ratio.
#   accuracy FALLS with r -> the bias is actively load-bearing and over-provisioning
#       costs. This would be the first evidence anywhere in this repository for the
#       paper's choice, which is currently asserted and never measured.
#
# NOTE ON A RETRACTION THIS REPLACES. An earlier version of this header framed
# "flat in r" as the boring outcome and r=2 as an unmotivated default. That was
# wrong: it read the paper's "for instance" as hedging when it is illustrating a
# principle (r = dim of the action space). Flat is the paper-confirming result.
#
# Torus, T=128 train, eval to 1024, 8 seeds, one batch, same recipe as the
# selective run so the gate's numbers are directly comparable.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPO="$(cd "$(dirname "$0")" && pwd)"; cd "$REPO/.."
R="$REPO/runs/rank_sweep"; mkdir -p "$R"
LOG="$REPO/rank_sweep.log"; echo "queued $(date)" > "$LOG"
ARMS="Vanilla Vanilla_r4 Vanilla_r8 Vanilla_r16 Vanilla_r32"

# Count real python trainers only -- pgrep -f also matches the shell that typed
# the pattern, which stalled run_rope_canonical.sh for two hours against zero jobs.
busy(){ ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /mapformer\.train_/' | wc -l; }
echo "$(date +%H:%M) waiting for the canonical-RoPE control" >> "$LOG"
until [ -f "$REPO/.rope_canonical_done" ]; do sleep 60; done
while [ "$(busy)" -gt 0 ]; do sleep 30; done
echo "$(date +%H:%M) clear; registering" >> "$LOG"

python3 - "$REPO" >> "$LOG" 2>&1 <<'PYEOF'
import sys, pathlib, subprocess
p = pathlib.Path(sys.argv[1]) / "train_variant.py"
t = p.read_text()
if "Vanilla_r4" not in t:
    a = "from mapformer.model_rope_canonical import MapFormerWM_RoPE_Canonical"
    assert t.count(a) == 1, "import anchor missing"
    t = t.replace(a, a + "\nfrom mapformer.model_rank import (MapFormerWM_r4, MapFormerWM_r8,\n"
                        "                                  MapFormerWM_r16, MapFormerWM_r32)", 1)
    b = '    "RoPE_Canonical": MapFormerWM_RoPE_Canonical,'
    assert t.count(b) == 1, "dict anchor missing"
    t = t.replace(b, b + '\n    "Vanilla_r4": MapFormerWM_r4,'
                        '\n    "Vanilla_r8": MapFormerWM_r8,'
                        '\n    "Vanilla_r16": MapFormerWM_r16,'
                        '\n    "Vanilla_r32": MapFormerWM_r32,', 1)
    p.write_text(t); print("registered rank arms")
r = subprocess.run([sys.executable, "-c",
    "from mapformer.train_variant import VARIANT_MAP as V;"
    "print({k: sum(p.numel() for p in V[k](vocab_size=21,d_model=128,n_heads=2,"
    "n_layers=1,grid_size=64).parameters()) "
    "for k in ('Vanilla','Vanilla_r4','Vanilla_r8','Vanilla_r16','Vanilla_r32')})"],
    capture_output=True, text=True, cwd="/home/prashr")
print(r.stdout or r.stderr)
PYEOF

MAXPG=5; A="train_var""iant"
for SEED in 0 1 2 3 4 5 6 7; do
  for V in $ARMS; do
    OUT="$R/p0/${V}_s${SEED}"; mkdir -p "$OUT"
    [ -f "$OUT/${V}.pt" ] && continue
    while :; do
      N0=$(pgrep -u "$USER" -af "$A" 2>/dev/null | grep -c -- "--device cuda:0" || true)
      N1=$(pgrep -u "$USER" -af "$A" 2>/dev/null | grep -c -- "--device cuda:1" || true)
      if [ "$N0" -le "$N1" ] && [ "$N0" -lt "$MAXPG" ]; then G=0; break; fi
      if [ "$N1" -lt "$MAXPG" ]; then G=1; break; fi
      if [ "$N0" -lt "$MAXPG" ]; then G=0; break; fi
      sleep 15
    done
    echo "$(date +%H:%M:%S) $V s$SEED -> cuda:$G" >> "$LOG"
    python3 -u -m mapformer.train_variant --variant "$V" --seed "$SEED" \
      --epochs 300 --lr 1e-3 --n-batches 98 --batch-size 128 --n-steps 128 \
      --n-layers 1 --n-heads 2 --d-model 128 --n-landmarks 0 --schedule cosine \
      --data-workers 3 --device "cuda:$G" --output-dir "$OUT" \
      > "$R/${V}_s${SEED}.log" 2>&1 &
    sleep 6
  done
done
wait
echo "$(date +%H:%M) $(find "$R" -name '*.pt' | wc -l)/40 checkpoints" >> "$LOG"
python3 -u -m mapformer.eval_noise_refine --runs-dir "$R" --variants $ARMS \
  --noises 0.0 --seeds 0 1 2 3 4 5 6 7 --lengths 128 512 1024 --n-trials 100 \
  --device cuda:0 --out "$REPO/RANK_SWEEP.md" >> "$LOG" 2>&1
touch "$REPO/.rank_sweep_done"; echo "$(date) DONE" >> "$LOG"
