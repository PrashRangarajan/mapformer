#!/bin/bash
# retrain_level15em.sh — waits for master_finish_v3.sh to fully exit
# (so its second finalize has committed CoPE numbers), then retrains
# Level15EM with the safe init (log_R_init_bias=3.0) and runs a third
# finalize to fold the new Level15EM rows into RESULTS_PAPER.md.
#
# Why we need this: the original Level15EM training had a bad init
# (log_R_init_bias=0.0) that gave Kalman gain K≈0.5 at start, which
# in EM's Hadamard-product attention destroyed gradient signal in
# 3 of 9 seeds. The safe init starts K≈0.05, letting the model
# behave like vanilla MapFormer at init and gradually learn correction.
#
# Launch with:
#   nohup bash /home/prashr/mapformer/retrain_level15em.sh \
#       > /home/prashr/mapformer/retrain_level15em.log 2>&1 &

set -u
cd /home/prashr
REPO=/home/prashr/mapformer

is_running () {
    ps aux | grep -E "$1" | grep -v grep | grep -v retrain_level15em >/dev/null
}

echo "[$(date)] retrain_level15em: waiting for master_finish_v3.sh to fully exit..."
while is_running "master_finish_v3\.sh"; do
    sleep 60
done
echo "[$(date)] master_finish_v3.sh done."

# Sanity: confirm the broken runs really were moved aside
if [ -d "$REPO/runs/Level15EM_clean" ] && [ -f "$REPO/runs/Level15EM_clean/seed0/Level15EM.pt" ]; then
    echo "[$(date)] FATAL: Level15EM_clean/seed0/Level15EM.pt still exists. Aborting."
    echo "         (orchestrator would skip it because checkpoint exists.)"
    exit 1
fi

echo "[$(date)] Launching orchestrator_level15_em (9 fresh runs with safe init)..."
nohup python3 -u -m mapformer.orchestrator_level15_em \
    > "$REPO/orchestrator_level15_em_safe.log" 2>&1 &
LE_PID=$!
echo "  Level15EM-safe PID: $LE_PID"
wait $LE_PID
echo "[$(date)] Level15EM retraining done."

echo "[$(date)] Running third finalize to fold safe-init Level15EM rows in..."
cd "$REPO"
bash orchestrator_finalize.sh 2>&1 | tail -15

echo "[$(date)] retrain_level15em: all done."
