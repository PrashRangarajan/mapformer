#!/usr/bin/env bash
# PAIRSPLIT_PREREG.md: EMPair vs EMPairConst at n=48 (seeds 8-47), to resolve the accuracy split.
set -u
REPO=/home/prashr/mapformer
cd /home/prashr
OUT=$REPO/runs/pairsplit
mkdir -p "$OUT/logs"
MAXPG=4
mine(){ ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /mapformer\.train_recency/ && /runs\/pairsplit/' | wc -l; }
on_gpu(){ ps -u "$USER" -o comm=,args= | awk -v g="cuda:$1" '$1=="python3" && /runs\/pairsplit/ && index($0,g)' | wc -l; }
pick(){ local a b; a=$(on_gpu 0); b=$(on_gpu 1); if [ "$a" -le "$b" ]; then echo 0; else echo 1; fi; }
go(){ local v=$1 s=$2
  [ -f "$OUT/${v}_s${s}/${v}_recency.json" ] && { echo "skip $v s$s"; return; }
  while [ "$(mine)" -ge $((2*MAXPG)) ]; do sleep 20; done
  g=$(pick); echo "launch $v s$s -> cuda:$g"
  setsid nohup python3 -u -m mapformer.train_recency \
    --variant "$v" --seed "$s" --k-max 64 --T 1024 --eval-T 1024 2048 \
    --epochs 300 --n-batches 48 --batch-size 16 \
    --schedule cosine --lr 1e-3 --n-layers 1 --d-model 128 --n-heads 2 \
    --device "cuda:$g" --output-dir "$OUT/${v}_s${s}" \
    > "$OUT/logs/${v}_s${s}.log" 2>&1 &
  sleep 6; }
go EMPair_r4 0                      # determinism re-check first
for s in $(seq 8 47); do go EMPair_r4 "$s"; go EMPairConst_r4 "$s"; done
while [ "$(mine)" -gt 0 ]; do sleep 30; done
python3 - <<'PY' > "$OUT/DETERMINISM.txt" 2>&1
from mapformer.ckpt_guard import compare_checkpoints
print(compare_checkpoints("runs/pairsplit/EMPair_r4_s0/EMPair_r4_recency.pt",
                          "runs/pairorigin/EMPair_r4_s0/EMPair_r4_recency.pt").report())
PY
missing=0
for v in EMPair_r4 EMPairConst_r4; do for s in $(seq 8 47); do
  [ -f "$OUT/${v}_s${s}/${v}_recency.json" ] || { echo "MISSING $v s$s"; missing=$((missing+1)); }
done; done
echo "missing=$missing"; cat "$OUT/DETERMINISM.txt"
[ "$missing" -eq 0 ] && touch "$OUT/.done"
echo "batch finished $(date)"
