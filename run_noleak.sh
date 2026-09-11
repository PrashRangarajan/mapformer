#!/usr/bin/env bash
# Leakage test (NOLEAK_PREREG.md): 2 new arms x 8 seeds + 1 determinism re-check.
set -u
REPO=/home/prashr/mapformer
cd /home/prashr
OUT=$REPO/runs/noleak; REP=$REPO/runs/noleak_repro; mkdir -p "$OUT/logs" "$REP/logs"
MAXPG=3
busy(){ ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /mapformer\.train_recency/' | wc -l; }
on_gpu(){ ps -u "$USER" -o comm=,args= | awk -v g="cuda:$1" '$1=="python3" && /mapformer\.train_recency/ && index($0,g)' | wc -l; }
pick(){ local a b; a=$(on_gpu 0); b=$(on_gpu 1); if [ "$a" -le "$b" ]; then echo 0; else echo 1; fi; }
go(){ local v=$1 s=$2 dir=$3
  while [ "$(busy)" -ge $((2*MAXPG)) ]; do sleep 20; done
  g=$(pick); echo "launch $v s$s -> cuda:$g ($dir)"
  setsid nohup python3 -u -m mapformer.train_recency \
    --variant "$v" --seed "$s" --k-max 64 --T 1024 --eval-T 1024 2048 \
    --epochs 300 --n-batches 48 --batch-size 16 \
    --schedule cosine --lr 1e-3 --n-layers 1 --d-model 128 --n-heads 2 \
    --device "cuda:$g" --output-dir "$dir/${v}_s${s}" \
    > "$dir/logs/${v}_s${s}.log" 2>&1 &
  sleep 8; }
go EMUnf_0_e8 0 "$REP"
for s in 0 1 2 3 4 5 6 7; do for v in EMNoLeak_e8 EMNoLeak_e64; do
  [ -f "$OUT/${v}_s${s}/${v}_recency.pt" ] && { echo "skip $v s$s"; continue; }
  go "$v" "$s" "$OUT"
done; done
while [ "$(busy)" -gt 0 ]; do sleep 30; done
python3 - <<'PY' > "$REP/DETERMINISM.txt" 2>&1
import torch
a=torch.load('/home/prashr/mapformer/runs/noleak_repro/EMUnf_0_e8_s0/EMUnf_0_e8_recency.pt',map_location='cpu',weights_only=False)
b=torch.load('/home/prashr/mapformer/runs/unfreeze/EMUnf_0_e8_s0/EMUnf_0_e8_recency.pt',map_location='cpu',weights_only=False)
eq=all(torch.equal(a['model_state'][k],b['model_state'][k]) for k in b['model_state']) and a['losses']==b['losses']
print(f"EMUnf_0_e8 s0 bitwise identical to stored: {eq}"); print("REUSE LICENSED" if eq else "REUSE NOT LICENSED")
PY
missing=0
for s in 0 1 2 3 4 5 6 7; do for v in EMNoLeak_e8 EMNoLeak_e64; do
  [ -f "$OUT/${v}_s${s}/${v}_recency.json" ] || { echo "MISSING $v s$s"; missing=$((missing+1)); }
done; done
echo "missing=$missing"; cat "$REP/DETERMINISM.txt"
[ "$missing" -eq 0 ] && touch "$OUT/.done"
echo "batch finished $(date)"
