#!/usr/bin/env bash
# T3_PREREG.md part A: MapPoPE_T3 and its inert twin on Bach Chorales, context 512, 5 seeds.
set -u
REPO=/home/prashr/mapformer
cd /home/prashr
OUT=$REPO/runs/jsb_t3; mkdir -p "$OUT/logs"
MAXPG=2
busy(){ ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /mapformer\.train_jsb/' | wc -l; }
on_gpu(){ ps -u "$USER" -o comm=,args= | awk -v g="cuda:$1" '$1=="python3" && /mapformer\.train_jsb/ && index($0,g)' | wc -l; }
pick(){ local a b; a=$(on_gpu 0); b=$(on_gpu 1); if [ "$a" -le "$b" ]; then echo 0; else echo 1; fi; }
for s in 0 1 2 3 4; do
  for A in MapPoPE_T3 MapPoPE_T3inert; do
    n="${A}_r2"; d="$OUT/${n}_s$s"
    [ -f "$d/$n.json" ] && { echo "skip $n s$s"; continue; }
    while [ "$(busy)" -ge $((2*MAXPG)) ]; do sleep 15; done
    g=$(pick); echo "$(date +%H:%M) launch $n s$s -> cuda:$g"
    OMP_NUM_THREADS=2 setsid nohup python3 -u -m mapformer.train_jsb --arch "$A" --seed "$s" \
      --train-len 512 --eval-every 250 --device "cuda:$g" --output-dir "$d" \
      > "$OUT/logs/${n}_s$s.log" 2>&1 &
    sleep 5
  done
done
while [ "$(busy)" -gt 0 ]; do sleep 20; done
echo "part A done $(date)"
# Part B: Indirect Indexing at the 100k budget, 8 seeds each.
OUTB=$REPO/runs/indirect_t3; mkdir -p "$OUTB/logs"
busy2(){ ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /mapformer\.train_indirect/' | wc -l; }
on_gpu2(){ ps -u "$USER" -o comm=,args= | awk -v g="cuda:$1" '$1=="python3" && /mapformer\.train_indirect/ && index($0,g)' | wc -l; }
pick2(){ local a b; a=$(on_gpu2 0); b=$(on_gpu2 1); if [ "$a" -le "$b" ]; then echo 0; else echo 1; fi; }
for s in 0 1 2 3 4 5 6 7; do
  for A in MapPoPE_T3 MapPoPE_T3inert; do
    n="${A}_r2"; d="$OUTB/${n}_s$s"
    [ -f "$d/$n.json" ] && { echo "skip $n s$s"; continue; }
    while [ "$(busy2)" -ge $((2*MAXPG)) ]; do sleep 20; done
    g=$(pick2); echo "$(date +%H:%M) launch $n s$s -> cuda:$g"
    OMP_NUM_THREADS=2 setsid nohup python3 -u -m mapformer.train_indirect --arch "$A" --seed "$s" \
      --device "cuda:$g" --output-dir "$d" > "$OUTB/logs/${n}_s$s.log" 2>&1 &
    sleep 10
  done
done
while [ "$(busy2)" -gt 0 ]; do sleep 30; done
echo "batch finished $(date)"
