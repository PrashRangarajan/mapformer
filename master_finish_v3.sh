#!/bin/bash
# master_finish_v3.sh — waits for existing MambaLike orchestrator, then
# runs zero-shot eval, finalize, CoPE resume, final finalize.
#
# Different from v2: does NOT relaunch MambaLike. Attaches to the
# already-running orchestrator_mambalike.py (was PID 3356687 when this
# was written) via pgrep polling, so we don't double-queue work.
#
# Launch with:
#   nohup bash /home/prashr/mapformer/master_finish_v3.sh \
#       > /home/prashr/mapformer/master_finish_v3.log 2>&1 &

set -u
cd /home/prashr
REPO=/home/prashr/mapformer

is_running () {
    ps aux | grep -E "$1" | grep -v grep | grep -v master_finish >/dev/null
}

echo "[$(date)] master_finish_v3: waiting for existing MambaLike orchestrator..."
while is_running "mapformer\.orchestrator_mambalike\b"; do
    sleep 60
done
echo "[$(date)] MambaLike orchestrator done."

# ---- Zero-shot transfer eval ----
ZS_VARIANTS="Vanilla VanillaEM RoPE Level1 Level15 Level15EM PC LSTM MambaLike"

echo "[$(date)] Zero-shot eval: config=clean (Axis 1 + Axis 2)..."
cd /home/prashr
python3 -u -m mapformer.zero_shot_eval \
    --runs-dir mapformer/runs --config clean \
    --variants $ZS_VARIANTS \
    --model-seeds 0 1 2 \
    --n-test-seeds 3 \
    --lengths 128 512 \
    --n-trials-128 50 \
    --n-trials-long 20 \
    --include-bias \
    --output "$REPO/ZERO_SHOT_TRANSFER_clean.md" 2>&1 | tail -5
echo "[$(date)] ZERO_SHOT_TRANSFER_clean.md written."

echo "[$(date)] Zero-shot eval: config=lm200 (all 3 axes)..."
python3 -u -m mapformer.zero_shot_eval \
    --runs-dir mapformer/runs --config lm200 \
    --variants $ZS_VARIANTS \
    --model-seeds 0 1 2 \
    --n-test-seeds 3 \
    --lengths 128 512 \
    --n-trials-128 50 \
    --n-trials-long 20 \
    --include-bias \
    --include-lm-sweep \
    --output "$REPO/ZERO_SHOT_TRANSFER_lm200.md" 2>&1 | tail -5
echo "[$(date)] ZERO_SHOT_TRANSFER_lm200.md written."

# ---- First finalize: includes RESULTS_PAPER.md + ZERO_SHOT_TRANSFER_*.md ----
echo "[$(date)] Running first finalize..."
cd "$REPO"
bash orchestrator_finalize.sh 2>&1 | tail -15
echo "[$(date)] First finalize done."

# ---- CoPE resume in background-foreground (so we wait for it) ----
echo "[$(date)] Launching CoPE resume orchestrator (~20h)..."
cd /home/prashr
nohup python3 -u -m mapformer.orchestrator_cope_resume \
    > "$REPO/orchestrator_cope_resume.log" 2>&1 &
CR_PID=$!
echo "  CoPE resume PID: $CR_PID"
wait $CR_PID
echo "[$(date)] CoPE resume done."

# ---- Second finalize: full CoPE rows ----
echo "[$(date)] Running second finalize..."
cd "$REPO"
bash orchestrator_finalize.sh 2>&1 | tail -15

echo "[$(date)] master_finish_v3: all done."
