#!/usr/bin/env bash
# SEARCH_PREREG.md S3: fixed-k and curriculum recency, 4 arms x 8 seeds + a determinism re-check.
set -u
REPO=/home/prashr/mapformer
cd /home/prashr
OUT=$REPO/runs/search
mkdir -p "$OUT/logs"
MAXPG=4
busy(){ ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /mapformer\.train_recency/' | wc -l; }
on_gpu(){ ps -u "$USER" -o comm=,args= | awk -v g="cuda:$1" '$1=="python3" && /mapformer\.train_recency/ && index($0,g)' | wc -l; }
pick(){ local a b; a=$(on_gpu 0); b=$(on_gpu 1); if [ "$a" -le "$b" ]; then echo 0; else echo 1; fi; }
go(){ local arm=$1 v=$2 s=$3; shift 3
  [ -f "$OUT/${arm}_s${s}/${v}_recency.json" ] && { echo "skip $arm s$s"; return; }
  while [ "$(busy)" -ge $((2*MAXPG)) ]; do sleep 20; done
  g=$(pick); echo "launch $arm s$s -> cuda:$g"
  setsid nohup python3 -u -m mapformer.train_recency \
    --variant "$v" --seed "$s" --k-max 64 --T 1024 --eval-T 1024 2048 \
    --epochs 300 --n-batches 48 --batch-size 16 \
    --schedule cosine --lr 1e-3 --n-layers 1 --d-model 128 --n-heads 2 \
    --device "cuda:$g" --output-dir "$OUT/${arm}_s${s}" "$@" \
    > "$OUT/logs/${arm}_s${s}.log" 2>&1 &
  sleep 8; }
go P0_repro VanillaEM_P0_r4 0
# seed outer, arm inner: a full one-seed table lands first
for s in $(seq 0 7); do
  go P0_fix64 VanillaEM_P0_r4 "$s" --k-fixed 64
  go WM_fix64 Vanilla_r4      "$s" --k-fixed 64
  go P0_fix16 VanillaEM_P0_r4 "$s" --k-fixed 16
  go P0_cur   VanillaEM_P0_r4 "$s" --k-curriculum 2,30
done
while [ "$(busy)" -gt 0 ]; do sleep 30; done
python3 - <<'PY' > "$OUT/DETERMINISM.txt" 2>&1
from mapformer.ckpt_guard import compare_checkpoints
r = compare_checkpoints("runs/search/P0_repro_s0/VanillaEM_P0_r4_recency.pt",
                        "runs/dof/recency/VanillaEM_P0_r4_s0/VanillaEM_P0_r4_recency.pt")
print(r.report())
PY
missing=0
for arm in P0_fix64 WM_fix64 P0_fix16 P0_cur; do for s in $(seq 0 7); do
  ls "$OUT/${arm}_s${s}/"*_recency.json >/dev/null 2>&1 || { echo "MISSING $arm s$s"; missing=$((missing+1)); }
done; done
echo "missing=$missing"; cat "$OUT/DETERMINISM.txt"
[ "$missing" -eq 0 ] && touch "$OUT/.done"
echo "batch finished $(date)"
