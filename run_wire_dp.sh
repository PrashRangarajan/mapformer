#!/usr/bin/env bash
# Waits for the 2x2 batch to fully clear, then wires parallel data generation
# into train.py and PROVES the serial path is byte-identical to the reference
# fingerprint captured before the edit. Reverts and commits nothing on mismatch.
set -uo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"; cd "$REPO/.."
LOG="$REPO/wire_dp.log"; echo "waiting for the 2x2 batch $(date)" > "$LOG"
A="train_var""iant"
until [ -f "$REPO/.l15_loop_supp_done" ]; do sleep 120; done
while [ "$(pgrep -u "$USER" -f "$A" | wc -l)" -gt 0 ]; do sleep 60; done
echo "$(date +%H:%M) batch clear; wiring" >> "$LOG"
python3 -u -m mapformer.wire_data_parallel \
  --reference "/tmp/claude-1002/-home-prashr-mapformer/11c678ec-9c7c-4954-8b14-36979f03e955/scratchpad/dp_ref/serial_reference.json" --device cuda:0 >> "$LOG" 2>&1
RC=$?
if [ $RC -eq 0 ]; then
  cd "$REPO"
  git add train.py train_variant.py
  git commit -q -F - <<'MSG'
Wire parallel data generation into the trainer, serial path verified

--data-workers N, default 0. At 0 the serial path is taken unchanged;
verified by training an 8-epoch run before and after the edit and
comparing a SHA-256 of the final weights, which catches any change to
the data stream, the RNG draw order or the optimisation where comparing
final loss would not.

At N>0 generation runs in N worker processes: ~2.15x at 3 workers and
~3.4x at 6, against an epoch that was 79-95% trajectory generation. The
parallel path seeds each batch by its index, so it is reproducible for
any worker count, but it draws a different sample from the same
generator than the serial path -- runs are comparable among themselves
and not against stored serial checkpoints (rule 3).

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
MSG
  git push -q origin main && echo "$(date +%H:%M) committed and pushed" >> "$LOG"
else
  echo "$(date +%H:%M) wiring FAILED rc=$RC; nothing committed" >> "$LOG"
fi
touch "$REPO/.wire_dp_done"
