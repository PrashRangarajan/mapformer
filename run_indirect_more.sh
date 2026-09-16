#!/usr/bin/env bash
# INDIRECT_PREREG amendment 1: seeds 3-7 at the same budget, to measure the SOLVE RATE.
# Waits for the JSB batch so the two do not contend.
set -u
REPO=/home/prashr/mapformer
cd /home/prashr
OUT=$REPO/runs/indirect
ARMS="RoPE PoPE MapWM MapPoPE"
MAXPG=2
busy(){ ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /mapformer\.train_indirect/' | wc -l; }
jsb_busy(){ ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /mapformer\.train_jsb/' | wc -l; }
on_gpu(){ ps -u "$USER" -o comm=,args= | awk -v g="cuda:$1" '$1=="python3" && /mapformer\.train_indirect/ && index($0,g)' | wc -l; }
pick(){ local a b; a=$(on_gpu 0); b=$(on_gpu 1); if [ "$a" -le "$b" ]; then echo 0; else echo 1; fi; }
nm(){ case "$1" in MapWM|MapPoPE) echo "$1_r2";; *) echo "$1";; esac; }
echo "$(date +%H:%M) waiting for the JSB batch"
until [ -f "$REPO/runs/jsb/.done" ] || { [ "$(jsb_busy)" -eq 0 ] && [ -s "$REPO/runs/jsb_driver.log" ] && grep -q "batch finished" "$REPO/runs/jsb_driver.log"; }; do sleep 60; done
echo "$(date +%H:%M) starting seeds 3-7"
for s in 3 4 5 6 7; do
  for A in $ARMS; do
    n=$(nm "$A"); d="$OUT/${n}_s$s"
    [ -f "$d/$n.json" ] && { echo "skip $n s$s"; continue; }
    while [ "$(busy)" -ge $((2*MAXPG)) ]; do sleep 20; done
    g=$(pick); echo "$(date +%H:%M) launch $n s$s -> cuda:$g"
    OMP_NUM_THREADS=2 setsid nohup python3 -u -m mapformer.train_indirect --arch "$A" --seed "$s" \
      --device "cuda:$g" --output-dir "$d" > "$OUT/logs/${n}_s$s.log" 2>&1 &
    sleep 10
  done
done
while [ "$(busy)" -gt 0 ]; do sleep 30; done
python3 -u -m mapformer.analyze_indirect --runs-dir "$OUT" --out "$REPO/INDIRECT_RESULTS.md" > "$OUT/analyze8.log" 2>&1
echo "analysis exit $?"
echo "batch finished $(date)"
