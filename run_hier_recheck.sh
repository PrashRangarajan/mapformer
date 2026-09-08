#!/usr/bin/env bash
# Re-measure "hierarchy buys compositional transfer" at the recipe that turned out
# to matter. The published +0.130 (MapWM-Hier 0.415 - MapWM-FlatHG 0.285) was
# measured entirely inside the old recipe, whose own effect is +0.160.
# BOTH arms are retrained here in ONE batch (rule 3) rather than reusing arm C.
set -u
REPO=/home/prashr/mapformer; cd /home/prashr
OUT=$REPO/runs/hier_recheck; mkdir -p "$OUT" "$OUT/logs"
busy(){ ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /mapformer\.train_/' | wc -l; }
on_gpu(){ ps -u "$USER" -o comm=,args= | awk -v g="cuda:$1" '$1=="python3" && /mapformer\.train_/ && index($0,g)' | wc -l; }
pick(){ local a b; a=$(on_gpu 0); b=$(on_gpu 1); [ "$a" -le "$b" ] && echo 0 || echo 1; }
for s in 0 1 2 3 4 5 6 7; do for v in Hourglass_k2 HourglassFlat3; do
  O="$OUT/${v}_s${s}"; [ -f "$O/${v}.pt" ] && continue
  mkdir -p "$O"; while [ "$(busy)" -ge 6 ]; do sleep 20; done
  g=$(pick); echo "$v s$s -> cuda:$g"
  setsid nohup python3 -u -m mapformer.train_compositional --variant "$v" \
    --target motif --n-steps 256 --epochs 150 --n-batches 156 --n-layers 3 \
    --schedule cosine --lr 1e-3 --seed "$s" --device "cuda:$g" \
    --output-dir "$O" > "$OUT/logs/${v}_s${s}.log" 2>&1 &
  sleep 5
done; done
while [ "$(busy)" -gt 0 ]; do sleep 30; done
miss=0
for s in 0 1 2 3 4 5 6 7; do for v in Hourglass_k2 HourglassFlat3; do
  [ -f "$OUT/${v}_s${s}/${v}.pt" ] || { echo "MISSING $v s$s"; miss=$((miss+1)); }
done; done
echo "missing=$miss"; [ "$miss" -eq 0 ] && touch "$REPO/.hier_recheck_done"
echo "finished $(date)"
