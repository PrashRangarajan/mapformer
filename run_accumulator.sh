#!/usr/bin/env bash
# GUARD (2026-10-29 disk cleanup): the checkpoints of runs this driver writes or reads were
# deleted; it would retrain those runs or overwrite tracked results. Losses/configs survive in
# runs/<dir>/CKPT_META.jsonl. See docs/cleanup/DISK_TIERS_ANALYSIS.md section 4, D2_MANIFEST.md.
if [ "${MAPFORMER_ALLOW_DELETED_RERUN:-0}" != "1" ]; then
  echo "REFUSED: $(basename "$0"): checkpoints of its runs were deleted on 2026-10-29 (docs/cleanup/DISK_TIERS_ANALYSIS.md, section 4); relaunching would retrain them or overwrite tracked results. Set MAPFORMER_ALLOW_DELETED_RERUN=1 to run anyway (rule 30: reproduce archived batches from their pre-registration commit)." >&2
  exit 3
fi
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
