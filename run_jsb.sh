#!/usr/bin/env bash
# JSB_PREREG.md: 4 arms x 5 seeds. Waits for the Indirect Indexing batch to clear.
set -u
REPO=/home/prashr/mapformer
cd /home/prashr
OUT=$REPO/runs/jsb
mkdir -p "$OUT/logs"
ARMS="RoPE PoPE MapWM MapPoPE"
MAXPG=2
busy(){ ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /mapformer\.train_jsb/' | wc -l; }
ind_busy(){ ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /mapformer\.train_indirect/' | wc -l; }
on_gpu(){ ps -u "$USER" -o comm=,args= | awk -v g="cuda:$1" '$1=="python3" && /mapformer\.train_jsb/ && index($0,g)' | wc -l; }
pick(){ local a b; a=$(on_gpu 0); b=$(on_gpu 1); if [ "$a" -le "$b" ]; then echo 0; else echo 1; fi; }
nm(){ case "$1" in MapWM|MapPoPE) echo "$1_r2";; *) echo "$1";; esac; }
echo "$(date +%H:%M) waiting for the Indirect Indexing batch"
until [ -f "$REPO/runs/indirect/.done" ] || [ "$(ind_busy)" -eq 0 ]; do sleep 60; done
echo "$(date +%H:%M) GPUs free; starting"
for s in 0 1 2 3 4; do
  for A in $ARMS; do
    n=$(nm "$A"); d="$OUT/${n}_s$s"
    [ -f "$d/$n.json" ] && { echo "skip $n s$s"; continue; }
    while [ "$(busy)" -ge $((2*MAXPG)) ]; do sleep 20; done
    g=$(pick); echo "$(date +%H:%M) launch $n s$s -> cuda:$g"
    OMP_NUM_THREADS=2 setsid nohup python3 -u -m mapformer.train_jsb --arch "$A" --seed "$s" \
      --eval-every 250 --device "cuda:$g" --output-dir "$d" > "$OUT/logs/${n}_s$s.log" 2>&1 &
    sleep 10
  done
done
while [ "$(busy)" -gt 0 ]; do sleep 30; done
missing=0
for s in 0 1 2 3 4; do for A in $ARMS; do n=$(nm "$A")
  [ -f "$OUT/${n}_s$s/$n.json" ] || { echo "MISSING $n s$s"; missing=$((missing+1)); }
done; done
echo "missing=$missing"
[ "$missing" -gt 0 ] && exit 1
python3 -u -m mapformer.analyze_jsb --runs-dir "$OUT" --out "$REPO/JSB_RESULTS.md" > "$OUT/analyze.log" 2>&1
echo "analysis exit $?"
[ -s "$REPO/JSB_RESULTS.md" ] && touch "$OUT/.done"
echo "batch finished $(date)"
