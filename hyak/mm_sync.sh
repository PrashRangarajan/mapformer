#!/bin/bash
# Run on the WORKSTATION (rsync copies from the machine it runs on). Pushes the Memory Maze batch code to the cluster
# as a minimal package: <MM_PROJ>/mapformer/ holds only the files below plus an EMPTY __init__.py (the repo's package
# __init__ imports plotting modules the cluster environment does not need). Prints md5s on both sides.
#   MM_REMOTE=<ssh alias> MM_PROJ=<cluster project dir> bash hyak/mm_sync.sh
set -euo pipefail
: "${MM_REMOTE:?}" "${MM_PROJ:?}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
FILES="model.py lie_groups.py model_mm.py train_mm.py mm_data.py mm_gate.py mm_gate_v2.py mm_gate_v3.py mm_encoder.py test_mm.py mm_floors.py analyze_mm.py stats_core.py"
SLURM="hyak/mm_prep.slurm hyak/mm_runs.slurm hyak/mm_floors.slurm hyak/mm_batch_tasks.txt"
ssh -o BatchMode=yes "$MM_REMOTE" "mkdir -p '$MM_PROJ/mapformer' '$MM_PROJ/logs' && : > '$MM_PROJ/mapformer/__init__.py'"
(cd "$REPO" && rsync -a $FILES "$MM_REMOTE:$MM_PROJ/mapformer/" && rsync -a $SLURM "$MM_REMOTE:$MM_PROJ/")
(cd "$REPO" && md5sum $FILES) > /tmp/mm_sync_local.md5
ssh -o BatchMode=yes "$MM_REMOTE" "cd '$MM_PROJ/mapformer' && md5sum $FILES" > /tmp/mm_sync_remote.md5
if diff -q /tmp/mm_sync_local.md5 /tmp/mm_sync_remote.md5 >/dev/null; then echo "sync OK (md5 match)"; else echo "MD5 MISMATCH"; exit 1; fi
