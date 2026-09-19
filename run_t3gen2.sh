#!/usr/bin/env bash
# T3GEN_PREREG amendment 1: Dyck-2 and torus at phase init 0.1.
set -u
REPO=/home/prashr/mapformer
cd /home/prashr
MAXPG=2
bz(){ ps -u "$USER" -o comm=,args= | awk -v m="$1" '$1=="python3" && index($0,m)' | wc -l; }
pk(){ local m=$1 a b; a=$(ps -u "$USER" -o args= | awk -v m="$m" -v g="cuda:0" 'index($0,m)&&index($0,g)' | wc -l)
      b=$(ps -u "$USER" -o args= | awk -v m="$m" -v g="cuda:1" 'index($0,m)&&index($0,g)' | wc -l)
      if [ "$a" -le "$b" ]; then echo 0; else echo 1; fi; }
echo "=== G1b Dyck-2 at init 0.1"
OUT=$REPO/runs/dyck_t3; mkdir -p "$OUT/logs"
for s in $(seq 0 7); do
  n="MapPoPE_T3pi01-1L_r2"; d="$OUT/${n}_s$s"
  [ -f "$d/$n.json" ] && continue
  while [ "$(bz mapformer.train_dyck)" -ge $((2*MAXPG)) ]; do sleep 10; done
  g=$(pk mapformer.train_dyck); echo "$(date +%H:%M) dyck $n s$s -> cuda:$g"
  OMP_NUM_THREADS=2 setsid nohup python3 -u -m mapformer.train_dyck --arch MapPoPE_T3pi01 --n-layers 1 \
    --n-heads 1 --rank 2 --seed "$s" --device "cuda:$g" --output-dir "$d" > "$OUT/logs/${n}_s$s.log" 2>&1 &
  sleep 3
done
while [ "$(bz mapformer.train_dyck)" -gt 0 ]; do sleep 15; done
echo "=== G2b torus at init 0.1"
OUT=$REPO/runs/torus_t3; mkdir -p "$OUT/logs"
for s in $(seq 0 7); do
  V=MapPoPE_T3pi01; d="$OUT/${V}_s$s"
  [ -f "$d/$V.pt" ] && continue
  while [ "$(bz mapformer.train_variant)" -ge $((2*MAXPG)) ]; do sleep 20; done
  g=$(pk mapformer.train_variant); echo "$(date +%H:%M) torus $V s$s -> cuda:$g"
  OMP_NUM_THREADS=2 setsid nohup python3 -u -m mapformer.train_variant --variant "$V" --seed "$s" \
    --epochs 50 --schedule cosine --n-batches 98 --batch-size 128 --n-steps 128 \
    --n-layers 1 --n-heads 2 --d-model 128 --n-landmarks 0 \
    --device "cuda:$g" --output-dir "$d" > "$OUT/logs/${V}_s$s.log" 2>&1 &
  sleep 8
done
while [ "$(bz mapformer.train_variant)" -gt 0 ]; do sleep 30; done
python3 -u -m mapformer.eval_paper_ood --runs-dir "$OUT" \
  --variants MapPoPE-Flat MapPoPE_T3 MapPoPE_T3inert MapPoPE_T3pi01 --seeds 0 1 2 3 4 5 6 7 \
  --extended --n-batches 8 --batch-size 32 --device cuda:0 --out "$REPO/TORUS_T3_RESULTS.md" \
  > "$OUT/eval_pi01.log" 2>&1
echo "eval exit $?"
echo "batch finished $(date)"
