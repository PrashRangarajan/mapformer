#!/usr/bin/env bash
# Every swap-test number in CTXSTEP_PILOT1/2/3.md and CTXSTEP_HS_RECIPE.md, from swap_test.py (audit B1).
set -u
cd /home/prashr; export PYTHONPATH=/home/prashr
D=/home/prashr/mapformer/docs/audits/2026-09-27; R=/home/prashr/mapformer/runs; OUT=$D/swap_results.jsonl; : > $OUT
st(){ python3 $D/swap_test.py --json $OUT "$@" 2>&1 | grep -v Warn; }
for s in 0 1; do for arm in CF CG SR; do
  st --task ctx --ckpt $R/ctxstep_pilot/${arm}_L1_s$s/$arm.pt --arm $arm --offset 5
  st --task ctx --ckpt $R/ctxstep_pilot/${arm}_L1_s$s/$arm.pt --arm $arm --offset 5 --replace-at 1 --replace-with and
done; done
for cue in lead trail; do for s in 0 1; do for al in "CF 1" "CG 1" "SR 1" "HS 2"; do set -- $al
  st --task ctx2 --cue $cue --ckpt $R/ctxstep2_pilot/${cue}_$1_L$2_s$s/$1.pt --arm $1 --layers $2 --offset 7
done; done; done
for s in 0 1; do
  st --task ctx2 --cue lead --ckpt $R/ctxstep2_pilot/lead_SR_L1_s$s/SR.pt --arm SR --offset 7 --replace-at -1 --replace-with walked
  st --task ctx2 --cue trail --ckpt $R/ctxstep2_pilot/trail_CG_L1_s$s/CG.pt --arm CG --offset 7 --replace-at 1 --replace-with and
done
for cue in lead trail; do
  for al in "far CF 1" "far CG 1" "far SR 1" "far HS 2" "near CG 1" "near SR 1"; do set -- $al
    st --task ctx3 --cue $cue --dist $1 --ckpt $R/ctxstep3_pilot/${cue}_$1_$2_L$3_s0/$2.pt --arm $2 --layers $3 --offset 15
  done
  for rec in cold warm; do for s in 1 5; do
    st --task ctx3 --cue $cue --dist far --ckpt $R/hs_recipe_pilot/${cue}_far_${rec}_s$s/HS.pt --arm HS --layers 2 --offset 15
  done; done
done
touch $D/.swap_all_done
