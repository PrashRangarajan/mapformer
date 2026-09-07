#!/usr/bin/env bash
# Pre-registered in LOCALISATION_PREREG.md. Eval-only, no training.
set -u
cd /home/prashr
python3 -u -m mapformer.probe_localisation \
  --runs-dir /home/prashr/mapformer/runs/sign/p0 \
  --variants Signed_r4 Abs_r4 --seeds 0 1 2 3 4 5 \
  --lengths 128 1024 --ks 4 8 16 32 --n-trials 60 --device cuda:0 \
  --out /home/prashr/mapformer/LOCALISATION.md \
  > /home/prashr/mapformer/localisation.log 2>&1
python3 -u -m mapformer.probe_localisation \
  --runs-dir /home/prashr/mapformer/runs/rank_sweep/p0 \
  --variants Vanilla Vanilla_r4 --seeds 0 1 2 3 4 5 \
  --lengths 128 1024 --ks 4 8 16 32 --n-trials 60 --device cuda:1 \
  --out /home/prashr/mapformer/LOCALISATION_RANK.md \
  >> /home/prashr/mapformer/localisation.log 2>&1
touch /home/prashr/mapformer/.localisation_done
