#!/usr/bin/env bash
# PoPE under the paper's OWN OOD-d / OOD-s protocol.
#
# Closes a caveat raised in INDEX_BASELINE_PAPER_TASK.md rather than leaving it
# standing: MapPoPE-Flat reaches 1.000 +/- 0.001 on the IID metric, but that is
# T=128 in-distribution against a 0.506 floor -- a ceiling effect as much as a
# result -- and PoPE has never been run under the paper's OOD protocol, where
# the ordering could differ. Inference only, on the checkpoints trained today.
set -euo pipefail
cd "$(dirname "$0")/.."
until [ -f mapformer/runs/level15_meets/.done50 ]; do sleep 60; done
echo "training pipelines finished; running PoPE OOD eval"
python3 -u -m mapformer.eval_paper_ood --runs-dir mapformer/runs/paper_task \
  --variants Vanilla VanillaEM_P0 MapPoPE-Flat RoPE PlainFlat PoPE-Flat \
  --seeds 0 1 2 --device cuda:1 \
  --out mapformer/PAPER_OOD_WITH_POPE.md
echo DONE; touch mapformer/runs/.pope_ood_done
