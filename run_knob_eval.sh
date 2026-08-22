#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -u -m mapformer.eval_knob_sweep --device cuda:1 \
  > mapformer/runs/knob_sweep/eval.log 2>&1
touch mapformer/runs/knob_sweep/.done
