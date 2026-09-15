#!/usr/bin/env bash
# DYCK_PREREG.md: 8 arms x 8 seeds, one batch. Usage: run_dyck.sh [batch_size] (default 128)
set -u
REPO=/home/prashr/mapformer
cd /home/prashr
BS=${1:-128}
OUT=$REPO/runs/dyck_bs$BS
mkdir -p "$OUT/logs"
MAXPG=6
ARMS="MapWM:1:1:2 MapEM:1:1:2 RoPE:1:1:2 RoPE:2:2:2 CoPE:1:1:2 CoPE:2:2:2 MapWM:1:1:4 MapEM:1:1:4"

busy(){ ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /mapformer\.train_dyck/' | wc -l; }
on_gpu(){ ps -u "$USER" -o comm=,args= | awk -v g="cuda:$1" '$1=="python3" && /mapformer\.train_dyck/ && index($0,g)' | wc -l; }
pick(){ local a b; a=$(on_gpu 0); b=$(on_gpu 1); if [ "$a" -le "$b" ]; then echo 0; else echo 1; fi; }
nm(){ if [ "$1" = MapWM ] || [ "$1" = MapEM ]; then echo "$1-$2L_r$4"; else echo "$1-$2L"; fi; }

for s in $(seq 0 7); do
  for arm in $ARMS; do
    IFS=: read -r A NL NH R <<< "$arm"
    name=$(nm "$A" "$NL" "$NH" "$R"); dir="$OUT/${name}_s$s"
    [ -f "$dir/$name.json" ] && { echo "skip $name s$s"; continue; }
    while [ "$(busy)" -ge $((2*MAXPG)) ]; do sleep 10; done
    g=$(pick); echo "$(date +%H:%M) launch $name s$s -> cuda:$g"
    OMP_NUM_THREADS=2 setsid nohup python3 -u -m mapformer.train_dyck --arch "$A" \
      --n-layers "$NL" --n-heads "$NH" --rank "$R" --seed "$s" --batch-size "$BS" \
      --device "cuda:$g" --output-dir "$dir" > "$OUT/logs/${name}_s$s.log" 2>&1 &
    sleep 3
  done
done
while [ "$(busy)" -gt 0 ]; do sleep 20; done
missing=0
for s in $(seq 0 7); do for arm in $ARMS; do
  IFS=: read -r A NL NH R <<< "$arm"; name=$(nm "$A" "$NL" "$NH" "$R")
  [ -f "$OUT/${name}_s$s/$name.json" ] || { echo "MISSING $name s$s"; missing=$((missing+1)); }
done; done
echo "missing=$missing"
[ "$missing" -gt 0 ] && exit 1
python3 -u -m mapformer.analyze_dyck --runs-dir "$OUT" --out "$REPO/DYCK_RESULTS_bs$BS.md" > "$OUT/analyze.log" 2>&1
echo "analysis exit $?"
[ -s "$REPO/DYCK_RESULTS_bs$BS.md" ] && touch "$OUT/.done"
echo "batch finished $(date)"
