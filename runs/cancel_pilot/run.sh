#!/usr/bin/env bash
# H3 pilot: recipe and noise floor before pre-registration. 16 runs, 4 serial streams.
cd /home/prashr
R=/home/prashr/mapformer/runs/cancel_pilot
stream(){ local G=$1; shift
  for spec in "$@"; do set -- $spec
    OUT="$R/${1}_L${2}_p${3}_s${4}"; [ -f "$OUT/eval.json" ] && continue
    OMP_NUM_THREADS=2 python3 -u -m mapformer.train_cancel --variant $1 --n-layers $2 --p-plus $3 --seed $4 \
      --epochs 300 --device cuda:$G --output-dir "$OUT" > "$OUT.log" 2>&1
  done; }
stream 0 "Vanilla 1 0.5 0" "RoPE 1 0.5 0" "Vanilla 1 1.0 0" "RoPE 1 1.0 0" &
stream 1 "Vanilla 4 0.5 0" "RoPE 4 0.5 0" "Vanilla 4 1.0 0" "RoPE 4 1.0 0" &
stream 0 "Vanilla 1 0.5 1" "RoPE 1 0.5 1" "Vanilla 1 1.0 1" "RoPE 1 1.0 1" &
stream 1 "Vanilla 4 0.5 1" "RoPE 4 0.5 1" "Vanilla 4 1.0 1" "RoPE 4 1.0 1" &
wait
touch $R/.done
