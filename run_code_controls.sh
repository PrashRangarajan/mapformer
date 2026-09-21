#!/bin/bash
# Chain the three queued batches so the cards are never oversubscribed.
#   C1  runs/code2048     4 runs @ 2048 ctx, 13.5-14.7 GiB each -> 1 per card
#   C2  runs/code_decay   12 runs @ 512, the repaired-baseline control
#   E   runs/enwik8_comp  39 runs @ 512, the composition claim at n=12
# Adding a 512 job alongside a 2048 job put GPU1 at 94% memory, and an OOM
# mid-batch voids it (CODE_PREREG Amendment 2), so C2 waits for C1.
set -u
REPO=/home/prashr/mapformer
log(){ echo "[$(date +%H:%M:%S)] $*"; }

while [ ! -f "$REPO/runs/code2048/.done" ]; do sleep 120; done
log "C1 (2048) complete -- starting C2 (decay)"
MAXPG=4 bash "$REPO/run_code_decay.sh"

log "C2 (decay) complete -- starting E (enwik8 composition, n=12)"
MAXPG=6 bash "$REPO/run_enwik8_comp.sh"
log "all three batches complete"
