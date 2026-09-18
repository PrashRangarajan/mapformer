#!/usr/bin/env bash
# JSBLEN_PREREG amendment 1: MapPoPE and MapWM at r=1 and r=4, training context 512.
set -u
REPO=/home/prashr/mapformer
cd /home/prashr
OUT=$REPO/runs/jsb_len512
mkdir -p "$OUT/logs"
MAXPG=2
busy(){ ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /mapformer\.train_jsb/' | wc -l; }
on_gpu(){ ps -u "$USER" -o comm=,args= | awk -v g="cuda:$1" '$1=="python3" && /mapformer\.train_jsb/ && index($0,g)' | wc -l; }
pick(){ local a b; a=$(on_gpu 0); b=$(on_gpu 1); if [ "$a" -le "$b" ]; then echo 0; else echo 1; fi; }
for s in 0 1 2 3 4; do
  for spec in MapPoPE:1 MapPoPE:4 MapWM:1 MapWM:4; do
    IFS=: read -r A R <<< "$spec"
    n="${A}_r${R}"; d="$OUT/${n}_s$s"
    [ -f "$d/$n.json" ] && { echo "skip $n s$s"; continue; }
    while [ "$(busy)" -ge $((2*MAXPG)) ]; do sleep 15; done
    g=$(pick); echo "$(date +%H:%M) launch $n s$s -> cuda:$g"
    OMP_NUM_THREADS=2 setsid nohup python3 -u -m mapformer.train_jsb --arch "$A" --rank "$R" --seed "$s" \
      --train-len 512 --eval-every 250 --device "cuda:$g" --output-dir "$d" \
      > "$OUT/logs/${n}_s$s.log" 2>&1 &
    sleep 5
  done
done
while [ "$(busy)" -gt 0 ]; do sleep 20; done
python3 -u -m mapformer.analyze_jsb_len --runs-dir "$OUT" --out "$REPO/JSB_LENGTH_RESULTS_RANK.md" > "$OUT/analyze_rank.log" 2>&1
echo "analysis exit $?"
echo "batch finished $(date)"
