set -u
cd /home/prashr
for N in 32 256; do
  python3 -u -m mapformer.eval_rank_nowrap --runs-dir /home/prashr/mapformer/runs/rank_nowrap_pilot/N$N --configs "2:$N:Vanilla_r2ph_om32,Vanilla_r3ph_om32" --seeds 100 101 --lengths 1024 2048 --n-trials 100 --device cuda:1 --out /home/prashr/mapformer/runs/rank_nowrap_pilot/N$N/EVAL_D2.md || exit 1
  python3 -u -m mapformer.rescore_hook --scale auto -- mapformer.eval_rank_nowrap --runs-dir /home/prashr/mapformer/runs/rank_nowrap_pilot/N$N --configs "2:$N:Vanilla_r2ph_om32,Vanilla_r3ph_om32" --seeds 100 101 --lengths 1024 --n-trials 100 --device cuda:1 --out /home/prashr/mapformer/runs/rank_nowrap_pilot/RESCORE_N$N.md || exit 1
done
CUDA_VISIBLE_DEVICES=1 python3 -u -m mapformer.analyze_rank_nowrap --runs-dir /home/prashr/mapformer/runs/rank_nowrap_pilot --seeds 100 101 --rescore-fmt "/home/prashr/mapformer/runs/rank_nowrap_pilot/RESCORE_N{N}.json" --json-out /home/prashr/mapformer/runs/rank_nowrap_pilot/SMOKE_ANALYSIS.json > /home/prashr/mapformer/runs/rank_nowrap_pilot/SMOKE_ANALYSIS.txt 2>&1 || exit 1
touch /home/prashr/mapformer/runs/rank_nowrap_pilot/.smoke_done
