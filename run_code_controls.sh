#!/bin/bash
# Chain, revised 2026-09-22: the ablation is inserted AHEAD of the enwik8
# composition batch -- it is cheaper (~8h vs ~18h) and it tests a live account
# (the non-negativity bound) rather than closing a small dead claim.
#   C2  runs/code_decay    12 runs @ 512   (finishing)
#   AB  runs/code_ablate   12 runs @ 512   PoPE sigma/ReLU/delta ablation
#   E   runs/enwik8_comp   35 runs @ 512   composition at n=12
set -u
REPO=/home/prashr/mapformer
log(){ echo "[$(date +%H:%M:%S)] $*"; }
while [ ! -f "$REPO/runs/code_decay/.done" ]; do sleep 120; done
log "C2 complete -- starting AB (PoPE ablation)"
MAXPG=4 bash "$REPO/run_ablate.sh"
log "AB complete -- starting E (enwik8 composition, n=12)"
MAXPG=6 bash "$REPO/run_enwik8_comp.sh"
log "all batches complete"
