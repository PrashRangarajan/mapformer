#!/usr/bin/env bash
# IS THE ROPE BASELINE'S FREQUENCY SCHEDULE LOAD-BEARING?
#
# The repo computes inv_freq_c = base^(-c/(n_b-1)); canonical RoPE is
# base^(-c/n_b). The comment above the line claims the canonical form, so the file
# contradicts itself. The schedules agree to within 1% over the high-frequency
# blocks that resolve position at these sequence lengths and differ by up to 25% at
# frequencies whose wavelength (47k-63k tokens) is DC over anything we run --
# so the expectation is no difference. That is a REASONED expectation, and this
# session has punished several of those, so it gets measured.
#
# PRE-REGISTERED DECISION, fixed before the run:
#   |canonical - repo| INSIDE its MDE at every length
#       -> the choice does not matter. Switch the code to canonical so the file
#          stops contradicting itself, delete the discussion from the LaTeX, and
#          record in CLAUDE.md that RoPE runs before this date used the n_b-1
#          schedule.
#   |canonical - repo| OUTSIDE its MDE anywhere
#       -> the index baseline has been mildly mis-specified throughout. Do NOT
#          silently switch: report which margins move, and by how much, first.
#
# Reproducibility note: inv_freq is a registered buffer and lives in the state
# dict, so stored RoPE checkpoints keep their own schedule when loaded whatever
# the code later says. Switching the default affects future runs only.
#
# 2 arms x 16 seeds on parity, ~3 minutes.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPO="$(cd "$(dirname "$0")" && pwd)"; cd "$REPO/.."
LOG="$REPO/rope_canonical.log"; echo "queued $(date)" > "$LOG"
R="$REPO/runs/rope_canon"; mkdir -p "$R"

echo "$(date +%H:%M) waiting for the selective batch to clear" >> "$LOG"
until [ -f "$REPO/.selective_done" ]; do sleep 60; done
for P in "train_algo""rithmic" "train_var""iant"; do
  while [ "$(pgrep -u "$USER" -f "$P" | wc -l)" -gt 0 ]; do sleep 30; done
done

# register the control now that nothing is spawning from train_variant
python3 - "$REPO" >> "$LOG" 2>&1 <<'PYEOF'
import sys, pathlib
p = pathlib.Path(sys.argv[1]) / "train_variant.py"
t = p.read_text()
if "RoPE_Canonical" not in t:
    a = "from mapformer.model_selective import (MapFormerWM_SRoPEGen, MapFormerWM_NoBottleneck,"
    assert t.count(a) == 1, "import anchor missing"
    t = t.replace(a, "from mapformer.model_rope_canonical import MapFormerWM_RoPE_Canonical\n" + a, 1)
    b = '    "SRoPEGen": MapFormerWM_SRoPEGen,'
    assert t.count(b) == 1, "dict anchor missing"
    t = t.replace(b, b + '\n    "RoPE_Canonical": MapFormerWM_RoPE_Canonical,', 1)
    p.write_text(t)
    print("registered RoPE_Canonical")
import subprocess
r = subprocess.run([sys.executable, "-c",
    "from mapformer.train_variant import VARIANT_MAP; "
    "import torch; m=VARIANT_MAP['RoPE'](vocab_size=2,d_model=128,n_heads=2,n_layers=1,grid_size=64); "
    "c=VARIANT_MAP['RoPE_Canonical'](vocab_size=2,d_model=128,n_heads=2,n_layers=1,grid_size=64); "
    "print('repo last-block  ', float(m.inv_freq[-1])); "
    "print('canon last-block ', float(c.inv_freq[-1])); "
    "print('differ:', not torch.allclose(m.inv_freq, c.inv_freq))"],
    capture_output=True, text=True, cwd="/home/prashr")
print(r.stdout or r.stderr)
PYEOF

MAXPG=5; A="train_algo""rithmic"
for SEED in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
  for V in RoPE RoPE_Canonical; do
    OUT="$R/${V}_s${SEED}"; mkdir -p "$OUT"
    [ -f "$OUT/${V}_parity.json" ] && continue
    while :; do
      N0=$(pgrep -u "$USER" -af "$A" 2>/dev/null | grep -c -- "--device cuda:0" || true)
      N1=$(pgrep -u "$USER" -af "$A" 2>/dev/null | grep -c -- "--device cuda:1" || true)
      if [ "$N0" -le "$N1" ] && [ "$N0" -lt "$MAXPG" ]; then G=0; break; fi
      if [ "$N1" -lt "$MAXPG" ]; then G=1; break; fi
      if [ "$N0" -lt "$MAXPG" ]; then G=0; break; fi
      sleep 10
    done
    python3 -u -m mapformer.train_algorithmic --variant "$V" --task parity \
      --seed "$SEED" --epochs 300 --n-batches 50 --batch-size 128 \
      --train-length 16 --eval-lengths 16 32 64 128 256 --lr 1e-3 \
      --d-model 128 --n-heads 2 --n-layers 1 --schedule cosine \
      --device "cuda:$G" --output-dir "$OUT" > "$R/${V}_s${SEED}.log" 2>&1 &
    sleep 2
  done
done
wait
python3 -u -m mapformer.agg_rope_canonical --runs-dir "$R" \
  --out "$REPO/ROPE_CANONICAL.md" >> "$LOG" 2>&1
touch "$REPO/.rope_canonical_done"; echo "$(date) DONE" >> "$LOG"
