#!/bin/bash
# Supervisor: run C2 (decay) only after C1 (2048) frees the cards.
# The 2048 arms need 13.5-14.7 GiB each, so one per 24 GiB card; adding a
# 512-context job alongside pushed GPU1 to 94% and risked OOMing an expensive
# run mid-batch, which CODE_PREREG Amendment 2 says voids it.
set -u
REPO=/home/prashr/mapformer
while [ ! -f "$REPO/runs/code2048/.done" ]; do sleep 120; done
echo "C1 (2048) complete $(date); starting C2 (decay) at full concurrency"
MAXPG=4 bash "$REPO/run_code_decay.sh"
