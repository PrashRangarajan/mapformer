#!/usr/bin/env bash
# MagOnly control (MAGONLY_PREREG.md): EMDoF_magonly seeds 0-23 into runs/dof/recency,
# plus two determinism controls that must be bitwise identical to stored checkpoints.
set -u
REPO=/home/prashr/mapformer
cd /home/prashr
OUT=$REPO/runs/dof/recency; REP=$REPO/runs/magonly_repro
mkdir -p "$OUT/logs" "$REP/logs"
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
# determinism controls first, so a failure is known early
go EMDoF_alignfree 0 "$REP"
go VanillaEM_P0_r4 0 "$REP"
for s in $(seq 0 23); do
  [ -f "$OUT/EMDoF_magonly_s${s}/EMDoF_magonly_recency.pt" ] && { echo "skip magonly s$s"; continue; }
  go EMDoF_magonly "$s" "$OUT"
done
while [ "$(busy)" -gt 0 ]; do sleep 30; done
python3 - <<'PY' > "$REP/DETERMINISM.txt" 2>&1
import torch
ok=True
for v in ['EMDoF_alignfree','VanillaEM_P0_r4']:
    a=torch.load(f'/home/prashr/mapformer/runs/magonly_repro/{v}_s0/{v}_recency.pt',map_location='cpu',weights_only=False)
    b=torch.load(f'/home/prashr/mapformer/runs/dof/recency/{v}_s0/{v}_recency.pt',map_location='cpu',weights_only=False)
    eq=all(torch.equal(a['model_state'][k],b['model_state'][k]) for k in b['model_state']) and a['losses']==b['losses']
    ok&=eq; print(f"{v} s0 bitwise identical to stored: {eq}")
print("REUSE LICENSED" if ok else "REUSE NOT LICENSED -- retrain comparators in-batch")
PY
missing=0
for s in $(seq 0 23); do [ -f "$OUT/EMDoF_magonly_s${s}/EMDoF_magonly_recency.json" ] || { echo "MISSING s$s"; missing=$((missing+1)); }; done
echo "missing=$missing"; cat "$REP/DETERMINISM.txt"
[ "$missing" -eq 0 ] && touch "$REPO/runs/dof/.magonly_done"
echo "batch finished $(date)"
