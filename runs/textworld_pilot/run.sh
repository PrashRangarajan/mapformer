#!/usr/bin/env bash
# Text-world pilot: speed and convergence before TEXTWORLD_PREREG.md is written. Not reused.
cd /home/prashr
R=/home/prashr/mapformer/runs/textworld_pilot
go(){ OUT="$R/${1}_L${2}_s${3}"; mkdir -p "$OUT"
  OMP_NUM_THREADS=4 python3 -u -m mapformer.train_textworld --variant $1 --n-layers $2 --seed $3 \
    --device cuda:$4 --output-dir "$OUT" > "$OUT.log" 2>&1; }
go Vanilla_r4 1 0 0 & go RoPE 1 0 0 & go Vanilla_r4 1 1 1 & go RoPE 1 1 1 &
wait; touch $R/.done
