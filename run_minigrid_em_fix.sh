#!/usr/bin/env bash
# EM init 2x2 on allocentric DoorKey-16x16. See MINIGRID_EM_FIX_PREREG.md.
# No --fast-attn (invalid for MapEM); buffer already built by run_minigrid_em.sh.
set -u
REPO=/home/prashr/mapformer; cd /home/prashr
OUT=$REPO/runs/minigrid_em_fix; mkdir -p "$OUT" "$OUT/logs"
ARMS="Vanilla_r4 VanillaEM VanillaEM_r4 VanillaEM_P0 VanillaEM_P0_r4"
busy(){ ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /mapformer\.train_variant/' | wc -l; }
on_gpu(){ ps -u "$USER" -o comm=,args= | awk -v g="cuda:$1" '$1=="python3" && /mapformer\.train_variant/ && index($0,g)' | wc -l; }
pick(){ local a b; a=$(on_gpu 0); b=$(on_gpu 1); [ "$a" -le "$b" ] && echo 0 || echo 1; }
for s in 0 1 2 3 4 5 6 7; do for v in $ARMS; do
  D="$OUT/${v}_s${s}"; [ -f "$D/${v}.pt" ] && continue
  mkdir -p "$D"; while [ "$(busy)" -ge 6 ]; do sleep 15; done
  g=$(pick); echo "$v s$s -> cuda:$g"
  setsid nohup python3 -u -m mapformer.train_variant --variant "$v" --seed "$s" \
    --n-landmarks 0 --epochs 50 --n-batches 156 --n-layers 3 \
    --env minigrid_doorkey16 --minigrid-tokenization obj_color \
    --minigrid-allocentric --minigrid-cached-buffer 25000 \
    --device "cuda:$g" --output-dir "$D" > "$OUT/logs/${v}_s${s}.log" 2>&1 &
  sleep 4
done; done
while [ "$(busy)" -gt 0 ]; do sleep 20; done
miss=0
for s in 0 1 2 3 4 5 6 7; do for v in $ARMS; do
  [ -f "$OUT/${v}_s${s}/${v}.pt" ] || { echo "MISSING $v s$s"; miss=$((miss+1)); }
done; done
echo "missing=$miss"; [ "$miss" -eq 0 ] && touch "$REPO/.minigrid_em_fix_done"
echo "finished $(date)"
