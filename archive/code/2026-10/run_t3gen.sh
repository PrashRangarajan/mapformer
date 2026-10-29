#!/usr/bin/env bash
# T3GEN_PREREG.md: G1 Dyck-2, G3 forced phase (both cheap), then G2 torus.
set -u
REPO=/home/prashr/mapformer
cd /home/prashr
MAXPG=2
bz(){ ps -u "$USER" -o comm=,args= | awk -v m="$1" '$1=="python3" && index($0,m)' | wc -l; }
pk(){ local m=$1 a b; a=$(ps -u "$USER" -o args= | awk -v m="$m" -v g="cuda:0" 'index($0,m)&&index($0,g)' | wc -l)
      b=$(ps -u "$USER" -o args= | awk -v m="$m" -v g="cuda:1" 'index($0,m)&&index($0,g)' | wc -l)
      if [ "$a" -le "$b" ]; then echo 0; else echo 1; fi; }

echo "=== G1 Dyck-2"
OUT=$REPO/runs/dyck_t3; mkdir -p "$OUT/logs"
for s in $(seq 0 7); do for A in MapPoPE_T3 MapPoPE_T3inert; do
  n="${A}-1L_r2"; d="$OUT/${n}_s$s"
  [ -f "$d/$n.json" ] && continue
  while [ "$(bz mapformer.train_dyck)" -ge $((2*MAXPG)) ]; do sleep 10; done
  g=$(pk mapformer.train_dyck); echo "$(date +%H:%M) dyck $n s$s -> cuda:$g"
  OMP_NUM_THREADS=2 setsid nohup python3 -u -m mapformer.train_dyck --arch "$A" --n-layers 1 --n-heads 1 \
    --rank 2 --seed "$s" --device "cuda:$g" --output-dir "$d" > "$OUT/logs/${n}_s$s.log" 2>&1 &
  sleep 3
done; done
while [ "$(bz mapformer.train_dyck)" -gt 0 ]; do sleep 15; done
python3 -u -m mapformer.analyze_dyck --runs-dir "$OUT" --out "$REPO/DYCK_T3_RESULTS.md" > "$OUT/analyze.log" 2>&1

echo "=== G3 forced phase"
OUT=$REPO/runs/jsb_forced; mkdir -p "$OUT/logs"
for s in 0 1 2 3 4; do for PI in 0.02 0.1; do
  n="MapPoPE_T3_r2_pi$PI"; d="$OUT/${n}_s$s"
  [ -f "$d/$n.json" ] && continue
  while [ "$(bz mapformer.train_jsb)" -ge $((2*MAXPG)) ]; do sleep 10; done
  g=$(pk mapformer.train_jsb); echo "$(date +%H:%M) forced $n s$s -> cuda:$g"
  OMP_NUM_THREADS=2 setsid nohup python3 -u -m mapformer.train_jsb --arch MapPoPE_T3 --phase-init "$PI" \
    --seed "$s" --train-len 512 --eval-every 250 --device "cuda:$g" --output-dir "$d" \
    > "$OUT/logs/${n}_s$s.log" 2>&1 &
  sleep 5
done; done
while [ "$(bz mapformer.train_jsb)" -gt 0 ]; do sleep 15; done

echo "=== G2 torus"
OUT=$REPO/runs/torus_t3; mkdir -p "$OUT/logs"
for s in $(seq 0 7); do for V in MapPoPE_T3 MapPoPE_T3inert MapPoPE-Flat; do
  d="$OUT/${V}_s$s"
  [ -f "$d/$V.pt" ] && continue
  while [ "$(bz mapformer.train_variant)" -ge $((2*MAXPG)) ]; do sleep 20; done
  g=$(pk mapformer.train_variant); echo "$(date +%H:%M) torus $V s$s -> cuda:$g"
  OMP_NUM_THREADS=2 setsid nohup python3 -u -m mapformer.train_variant --variant "$V" --seed "$s" \
    --epochs 50 --schedule cosine --n-batches 98 --batch-size 128 --n-steps 128 \
    --n-layers 1 --n-heads 2 --d-model 128 --n-landmarks 0 \
    --device "cuda:$g" --output-dir "$d" > "$OUT/logs/${V}_s$s.log" 2>&1 &
  sleep 8
done; done
while [ "$(bz mapformer.train_variant)" -gt 0 ]; do sleep 30; done
echo "batch finished $(date)"
