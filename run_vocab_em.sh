#!/usr/bin/env bash
# Vocab sweep with a HEALTHY EM arm. See VOCAB_EM_PREREG.md.
# Better recipe throughout (cosine, lr 1e-3) so recipe and init are separable.
set -u
REPO=/home/prashr/mapformer; cd /home/prashr
OUT=$REPO/runs/vocab_em; mkdir -p "$OUT" "$OUT/logs"
ARMS="Vanilla Vanilla_r4 VanillaEM VanillaEM_P0 VanillaEM_P0_r4"
busy(){ ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /mapformer\.train_variant/' | wc -l; }
on_gpu(){ ps -u "$USER" -o comm=,args= | awk -v g="cuda:$1" '$1=="python3" && /mapformer\.train_variant/ && index($0,g)' | wc -l; }
pick(){ local a b; a=$(on_gpu 0); b=$(on_gpu 1); [ "$a" -le "$b" ] && echo 0 || echo 1; }
for s in 0 1 2 3 4 5 6 7; do for n in 16 64 256; do for v in $ARMS; do
  D="$OUT/${v}_n${n}_s${s}"; [ -f "$D/${v}.pt" ] && continue
  mkdir -p "$D"; while [ "$(busy)" -ge 6 ]; do sleep 15; done
  g=$(pick); echo "$v n=$n s$s -> cuda:$g"
  setsid nohup python3 -u -m mapformer.train_variant --variant "$v" --seed "$s" \
    --n-landmarks 0 --p-action-noise 0.0 --n-obs-types "$n" \
    --epochs 50 --n-batches 156 --schedule cosine --lr 1e-3 \
    --device "cuda:$g" --output-dir "$D" > "$OUT/logs/${v}_n${n}_s${s}.log" 2>&1 &
  sleep 4
done; done; done
while [ "$(busy)" -gt 0 ]; do sleep 20; done
miss=0
for s in 0 1 2 3 4 5 6 7; do for n in 16 64 256; do for v in $ARMS; do
  [ -f "$OUT/${v}_n${n}_s${s}/${v}.pt" ] || { echo "MISSING $v n$n s$s"; miss=$((miss+1)); }
done; done; done
echo "missing=$miss"; [ "$miss" -eq 0 ] && touch "$REPO/.vocab_em_done"
echo "finished $(date)"
