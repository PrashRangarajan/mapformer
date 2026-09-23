#!/bin/bash
# Re-run the C1 runs lost to CUDA OOM, with a GPU picker that looks at ACTUAL
# occupancy instead of a launch counter.
#
# The bug: run_code2048_seeds.sh assigned cuda:$((i%2)) -- a blind round-robin.
# busy() counted TOTAL processes, not per-card, so when runs finished out of
# order the counter desynchronised from which card was free and two
# 2048-context runs (13.5-14.7 GiB each) landed on the same 24 GiB card.
# Vanilla_s1 and PoPE-Flat_s2 died of OOM. Rule 13, reintroduced by me.
set -u
exec 9>"/tmp/.mapformer_$(basename "$0" .sh).lock"
flock -n 9 || { echo "another instance is already running -- exiting"; exit 0; }

REPO=/home/prashr/mapformer
OUT=$REPO/runs/code2048
cd /home/prashr

# a card is FREE if no training process of ours is on it
gpu_busy(){ ps -u "$USER" -o comm=,args= | awk -v d="cuda:$1" '$1=="python3" && /train_hourglass_enwik8/ && index($0,d)' | wc -l; }
pick_free(){ for g in 0 1; do [ "$(gpu_busy $g)" -eq 0 ] && { echo "$g"; return; }; done; echo ""; }

run(){
  local arm=$1 seed=$2
  [ -f "$OUT/${arm}_s${seed}.json" ] && { echo "skip ${arm}_s${seed} (done)"; return; }
  local g=""
  while [ -z "$g" ]; do g=$(pick_free); [ -z "$g" ] && sleep 60; done
  echo "launch $arm seed=$seed on cuda:$g  $(date +%H:%M:%S)"
  OMP_NUM_THREADS=4 setsid nohup python3 -u -m mapformer.train_hourglass_enwik8 \
    --model "$arm" --device "cuda:$g" --seed "$seed" \
    --iters 36000 --eval-every 1000 --save-ckpt \
    --bottleneck-r 4 --seq-len 2048 --batch-size 4 --dim 512 --n-layers 9 --lr 2e-4 \
    --data $REPO/data/code_train.bin --data-val $REPO/data/code_val.bin \
    --out "$OUT" --tag "_s${seed}" >> "$OUT/${arm}_s${seed}.log" 2>&1 &
  sleep 90     # let it claim its memory before the next pick_free
}

run Vanilla 1
run PoPE-Flat 2
while [ "$(ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /train_hourglass_enwik8/ && /runs\/code2048/' | wc -l)" -gt 0 ]; do sleep 60; done
missing=0
for s in 0 1 2; do for a in RoPE PoPE-Flat Vanilla MapPoPE-Flat; do
  [ -f "$OUT/${a}_s${s}.json" ] || { echo "MISSING ${a}_s${s}"; missing=$((missing+1)); }
done; done
echo "missing=$missing"; [ "$missing" -eq 0 ] && touch "$OUT/.seeds_done"
echo "c1 fill finished $(date)"
