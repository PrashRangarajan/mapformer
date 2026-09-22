#!/bin/bash
# Chain 2026-09-22:  AB (PoPE ablation) -> C1 seeds.  enwik8 remains PAUSED.
set -u
REPO=/home/prashr/mapformer
log(){ echo "[$(date +%H:%M:%S)] $*"; }
while [ ! -f "$REPO/runs/code_decay/.done" ]; do sleep 60; done
log "starting AB (PoPE ablation)"
MAXPG=4 bash "$REPO/run_ablate.sh"
log "AB complete -- starting C1 seeds (2048, seeds 1-2)"
MAXPG=2 bash "$REPO/run_code2048_seeds.sh"
log "done. enwik8 composition remains PAUSED."
