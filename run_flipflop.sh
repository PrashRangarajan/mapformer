#!/usr/bin/env bash
# Flip-Flop LM (Liu et al. 2023): the PUBLISHED external check. 6 arms x 8
# seeds, ONE batch. Architecture per CoPE sec 5.1: 4 layers, d=256, 4 heads.
# Gates: FLIPFLOP_GATES.md. Chance 0.500; eval scores the FINAL read only.
set -u
REPO=/home/prashr/mapformer
cd /home/prashr

OUT=$REPO/runs/flipflop
mkdir -p "$OUT"
LOG=$REPO/runs/flipflop/logs; mkdir -p "$LOG"

ARMS=(Signed_r4 Abs_r4 Pos_r4 CARoPE_r4 RoPE PlainFlat)
SEEDS=(0 1 2 3 4 5 6 7)
MAXPG=3            # per GPU
T=512; EPOCHS=200

# Count real interpreters, NOT shells. `pgrep -f` matches the shell that typed
# the pattern AND every later tool call that mentions it; a waiter built on it
# once sat two hours against zero running jobs. `comm` is the executable name,
# so a shell cannot match however it quotes things.
busy(){ ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /mapformer\.train_flipflop/' | wc -l; }
on_gpu(){ ps -u "$USER" -o comm=,args= | awk -v g="cuda:$1" '$1=="python3" && /mapformer\.train_flipflop/ && index($0,g)' | wc -l; }

# Balance to the LESS loaded device. A fill-first picker silently becomes a
# single-GPU scheduler whenever the job count is <= MAXPG (rule 13).
pick(){ local a b; a=$(on_gpu 0); b=$(on_gpu 1); if [ "$a" -le "$b" ]; then echo 0; else echo 1; fi; }

for s in "${SEEDS[@]}"; do          # seed OUTER, variant INNER: a full
  for v in "${ARMS[@]}"; do         # low-confidence table lands first
    ck="$OUT/${v}_s${s}/${v}_flipflop.pt"
    [ -f "$ck" ] && { echo "skip $v s$s (done)"; continue; }
    while [ "$(busy)" -ge $((2*MAXPG)) ]; do sleep 20; done
    g=$(pick)
    echo "launch $v s$s -> cuda:$g"
    setsid nohup python3 -u -m mapformer.train_flipflop \
      --variant "$v" --seed "$s" --T $T \
      --epochs $EPOCHS --n-batches 48 --batch-size 16 \
      --schedule cosine --lr 1e-3 --n-layers 4 --d-model 256 --n-heads 4 \
      --fast-attn --device "cuda:$g" \
      --output-dir "$OUT/${v}_s${s}" \
      > "$LOG/${v}_s${s}.log" 2>&1 &
    sleep 8
  done
done

while [ "$(busy)" -gt 0 ]; do sleep 30; done

# `wait` returns regardless of child success, so verify the ARTIFACT exists
# rather than inferring completion from the absence of a crash.
missing=0
for s in "${SEEDS[@]}"; do for v in "${ARMS[@]}"; do
  [ -f "$OUT/${v}_s${s}/${v}_flipflop.json" ] || { echo "MISSING $v s$s"; missing=$((missing+1)); }
done; done
echo "missing=$missing"
[ "$missing" -eq 0 ] && touch "$REPO/.flipflop_done"
echo "batch finished $(date)"
