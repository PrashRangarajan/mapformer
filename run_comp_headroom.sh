#!/usr/bin/env bash
# Compositional headroom batch. See COMP_HEADROOM_PREREG.md.
set -u
REPO=/home/prashr/mapformer; cd /home/prashr
OUT=$REPO/runs/comp_headroom; mkdir -p "$OUT" "$OUT/logs"
busy(){ ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /mapformer\.train_/' | wc -l; }
on_gpu(){ ps -u "$USER" -o comm=,args= | awk -v g="cuda:$1" '$1=="python3" && /mapformer\.train_/ && index($0,g)' | wc -l; }
pick(){ local a b; a=$(on_gpu 0); b=$(on_gpu 1); [ "$a" -le "$b" ] && echo 0 || echo 1; }

launch(){ # tag variant schedule lr epochs seed
  local tag=$1 v=$2 sch=$3 lr=$4 ep=$5 s=$6
  local O="$OUT/${tag}_s${s}"
  [ -f "$O/$v.pt" ] && return
  mkdir -p "$O"; while [ "$(busy)" -ge 6 ]; do sleep 20; done
  local g; g=$(pick); echo "$tag s$s -> cuda:$g"
  setsid nohup python3 -u -m mapformer.train_compositional --variant "$v" \
    --target motif --n-steps 256 --epochs "$ep" --n-batches 156 --n-layers 3 \
    --schedule "$sch" --lr "$lr" --seed "$s" --device "cuda:$g" \
    --output-dir "$O" > "$OUT/logs/${tag}_s${s}.log" 2>&1 &
  sleep 5
}
for s in 0 1 2 3 4 5 6 7; do
  launch A Hourglass_k2    linear 3e-4 50  "$s"
  launch B Hourglass_k2    cosine 1e-3 50  "$s"
  launch C Hourglass_k2    cosine 1e-3 150 "$s"
  launch D LoopedHourglass cosine 1e-3 150 "$s"
done
while [ "$(busy)" -gt 0 ]; do sleep 30; done
miss=0
for s in 0 1 2 3 4 5 6 7; do
  for pair in "A:Hourglass_k2" "B:Hourglass_k2" "C:Hourglass_k2" "D:LoopedHourglass"; do
    t="${pair%%:*}"; v="${pair##*:}"
    [ -f "$OUT/${t}_s${s}/$v.pt" ] || { echo "MISSING $t s$s"; miss=$((miss+1)); }
  done
done
echo "missing=$miss"; [ "$miss" -eq 0 ] && touch "$REPO/.comp_headroom_done"
echo "finished $(date)"
