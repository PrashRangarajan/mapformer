#!/usr/bin/env bash
set -u; cd /home/prashr
R=/home/prashr/mapformer/runs
python3 -u -m mapformer.probe_accumulator \
  --specs "Vanilla (forget batch)=$R/forget/p0:Vanilla" \
          "Forget=$R/forget/p0:Forget" \
          "Vanilla (L15 batch)=$R/l15_ablation/p0:Vanilla" \
          "Level15=$R/l15_ablation/p0:Level15" \
          "Vanilla (PoPE batch)=$R/popewrap/g64/p0:Vanilla" \
          "MapPoPE=$R/popewrap/g64/p0:MapPoPE" \
  --seeds 0 1 2 3 4 5 --lengths 128 1024 --n-trials 20 --device cuda:0 \
  --out /home/prashr/mapformer/ACCUMULATOR.md > /home/prashr/mapformer/accumulator.log 2>&1
touch /home/prashr/mapformer/.accumulator_done
