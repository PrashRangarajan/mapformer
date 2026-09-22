#!/bin/bash
# Chain, revised 2026-09-22: enwik8 composition is PAUSED at the user's request.
# Its script and pre-registration are intact; re-add the final line to resume.
#   C2  runs/code_decay    (finishing)
#   AB  runs/code_ablate   PoPE sigma/ReLU/delta ablation
#   -- paused:  MAXPG=6 bash "$REPO/run_enwik8_comp.sh"
set -u
REPO=/home/prashr/mapformer
log(){ echo "[$(date +%H:%M:%S)] $*"; }
while [ ! -f "$REPO/runs/code_decay/.done" ]; do sleep 120; done
log "C2 complete -- starting AB (PoPE ablation)"
MAXPG=4 bash "$REPO/run_ablate.sh"
log "AB complete. enwik8 composition is PAUSED -- nothing further queued."
