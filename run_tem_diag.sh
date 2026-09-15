#!/usr/bin/env bash
set -u
REPO=/home/prashr/mapformer; OUT=$REPO/runs/tem_recency_diag; mkdir -p "$OUT/logs"; cd /home/prashr
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
g=0
run(){ tag=$1; v=$2; s=$3; shift 3
  echo "$(date +%H:%M) $tag s$s -> cuda:$g"
  setsid nohup python3 -u -m mapformer.train_recency --variant "$v" --seed "$s" --k-max 64 --T 1024 \
    --eval-T 1024 2048 --epochs 300 --n-batches 48 --batch-size 16 --schedule cosine --lr 1e-3 \
    --n-layers 1 --d-model 128 --n-heads 2 --device "cuda:$g" --output-dir "$OUT/${tag}_s${s}" "$@" \
    > "$OUT/logs/${tag}_s${s}.log" 2>&1 &
  g=$((1-g)); sleep 5; }
for s in 0 1; do
  run D1_k1 TEMRecency_Query $s --k-fixed 1
  run D2_counter TEMRecency_Query_CounterInstalled $s
  run D3_init1 TEMRecency_Query_Init1 $s
done
while ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /mapformer\.train_recency/ && /tem_recency_diag/' | grep -q .; do sleep 60; done
n=$(ls $OUT/*/*_recency.json 2>/dev/null | wc -l); echo "jsons=$n/6"; [ "$n" -eq 6 ] && touch "$OUT/.done"
echo "tem diag finished $(date)"
