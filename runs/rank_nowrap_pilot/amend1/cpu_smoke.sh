# RANK_NOWRAP Amendment 1 end-to-end smoke, CPU ONLY (no GPU while another user's jobs are on the GPUs).
# M32 pilot (redraw arm) trained on CPU for 3 short epochs (10 batches) at seeds 100-101; A32/B32/AL/BL are the earlier
# 40-epoch GPU pilot checkpoints (symlinked). Eval, re-score and analysis all on CPU (exercises the N9 CPU fallback).
set -u
export CUDA_VISIBLE_DEVICES=""
cd /home/prashr
for S in 100 101; do
  OMP_NUM_THREADS=2 python3 -u -m mapformer.train_rank_nowrap --variant Vanilla_r2ph_om32_redraw --env nd --n-dims 2 --grid-size 32     --seed $S --epochs 3 --lr 1e-3 --n-batches 10 --batch-size 16 --n-steps 1024 --n-layers 1 --n-heads 2 --d-model 128     --n-landmarks 0 --schedule cosine --data-workers 3 --device cpu --output-dir /home/prashr/mapformer/runs/rank_nowrap_pilot/amend1/N32/D2/Vanilla_r2ph_om32_redraw_s$S --save-full-state || exit 1
done
for N in 32 256; do
  if [ $N = 32 ]; then ARMS=Vanilla_r2ph_om32,Vanilla_r3ph_om32,Vanilla_r2ph_om32_redraw; else ARMS=Vanilla_r2ph_om32,Vanilla_r3ph_om32; fi
  python3 -u -m mapformer.eval_rank_nowrap --runs-dir /home/prashr/mapformer/runs/rank_nowrap_pilot/amend1/N$N --configs "2:$N:$ARMS" --seeds 100 101 --lengths 1024 2048 --n-trials 100 --device cpu --out /home/prashr/mapformer/runs/rank_nowrap_pilot/amend1/N$N/EVAL_D2.md || exit 1
  python3 -u -m mapformer.rescore_hook --scale auto -- mapformer.eval_rank_nowrap --runs-dir /home/prashr/mapformer/runs/rank_nowrap_pilot/amend1/N$N --configs "2:$N:$ARMS" --seeds 100 101 --lengths 1024 --n-trials 100 --device cpu --out /home/prashr/mapformer/runs/rank_nowrap_pilot/amend1/RESCORE_N$N.md || exit 1
done
python3 -u -m mapformer.analyze_rank_nowrap --runs-dir /home/prashr/mapformer/runs/rank_nowrap_pilot/amend1 --seeds 100 101 --rescore-fmt "/home/prashr/mapformer/runs/rank_nowrap_pilot/amend1/RESCORE_N{N}.json" --json-out /home/prashr/mapformer/runs/rank_nowrap_pilot/amend1/SMOKE_ANALYSIS.json > /home/prashr/mapformer/runs/rank_nowrap_pilot/amend1/SMOKE_ANALYSIS.txt 2>&1 || exit 1
touch /home/prashr/mapformer/runs/rank_nowrap_pilot/amend1/.smoke_done
