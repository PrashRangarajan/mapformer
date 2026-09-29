#!/usr/bin/env bash
# Context-step pilot (CONTEXT_STEP_DESIGN.md): CF, SR, CG, HS with decoys (p 0.3), 2 seeds. Not reused.
cd /home/prashr
R=/home/prashr/mapformer/runs/ctxstep_pilot
go(){ OUT="$R/${1}_L${2}_s${3}"; mkdir -p "$OUT"
  OMP_NUM_THREADS=3 python3 -u -m mapformer.train_ctxstep --variant $1 --n-layers $2 --seed $3 \
    --device cuda:$4 --output-dir "$OUT" > "$OUT.log" 2>&1; }
( go CF 1 0 0; go CF 1 1 0 ) & ( go SR 1 0 1; go SR 1 1 1 ) & ( go CG 1 0 0; go CG 1 1 0 ) & ( go HS 2 0 1; go HS 2 1 1 ) &
wait; touch $R/.done
