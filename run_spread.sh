#!/usr/bin/env bash
# SPREAD_PREREG.md: queries-per-token vs number-of-query-tokens. 4 arms x 8 seeds + repro.
set -u
REPO=/home/prashr/mapformer
cd /home/prashr
OUT=$REPO/runs/spread
mkdir -p "$OUT/logs"
K4="1,4,16,64"
K16="1,2,3,4,6,8,11,16,22,26,32,38,45,52,58,64"
MAXPG=4
busy(){ ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /mapformer\.train_recency/' | wc -l; }
on_gpu(){ ps -u "$USER" -o comm=,args= | awk -v g="cuda:$1" '$1=="python3" && /mapformer\.train_recency/ && index($0,g)' | wc -l; }
pick(){ local a b; a=$(on_gpu 0); b=$(on_gpu 1); if [ "$a" -le "$b" ]; then echo 0; else echo 1; fi; }
go(){ local arm=$1 ep=$2 s=$3; shift 3
  [ -f "$OUT/${arm}_s${s}/VanillaEM_P0_r4_recency.json" ] && { echo "skip $arm s$s"; return; }
  while [ "$(busy)" -ge $((2*MAXPG)) ]; do sleep 20; done
  g=$(pick); echo "launch $arm s$s (${ep} ep) -> cuda:$g"
  setsid nohup python3 -u -m mapformer.train_recency \
    --variant VanillaEM_P0_r4 --seed "$s" --k-max 64 --T 1024 --eval-T 1024 2048 \
    --epochs "$ep" --n-batches 48 --batch-size 16 \
    --schedule cosine --lr 1e-3 --n-layers 1 --d-model 128 --n-heads 2 \
    --device "cuda:$g" --output-dir "$OUT/${arm}_s${s}" "$@" \
    > "$OUT/logs/${arm}_s${s}.log" 2>&1 &
  sleep 8; }
go repro 300 0
# long arms first so they are not the tail; seed outer within each block
for s in $(seq 0 7); do
  go m16_e1200 1200 "$s" --k-set "$K16"
  go m64_e1200 1200 "$s"
done
for s in $(seq 0 7); do
  go m4_e300 300 "$s" --k-set "$K4"
  go m16_e300 300 "$s" --k-set "$K16"
done
while [ "$(busy)" -gt 0 ]; do sleep 30; done
python3 - <<'PY' > "$OUT/DETERMINISM.txt" 2>&1
from mapformer.ckpt_guard import compare_checkpoints
print(compare_checkpoints("runs/spread/repro_s0/VanillaEM_P0_r4_recency.pt",
                          "runs/dof/recency/VanillaEM_P0_r4_s0/VanillaEM_P0_r4_recency.pt").report())
PY
missing=0
for arm in m4_e300 m16_e300 m16_e1200 m64_e1200; do for s in $(seq 0 7); do
  [ -f "$OUT/${arm}_s${s}/VanillaEM_P0_r4_recency.json" ] || { echo "MISSING $arm s$s"; missing=$((missing+1)); }
done; done
echo "missing=$missing"; cat "$OUT/DETERMINISM.txt"
[ "$missing" -eq 0 ] && touch "$OUT/.done"
echo "batch finished $(date)"
