#!/bin/bash
# Master completion script — runs the remaining autonomous steps
# after the three current orchestrators (em, multilayer, level15_em)
# finish. This fills the gaps left when followup.sh exited early.
#
# Steps:
#   1. Wait for orchestrator_em, orchestrator_multilayer,
#      orchestrator_level15_em to exit.
#   2. Launch orchestrator_baselines (LSTM / CoPE / MambaLike) and wait.
#   3. Run orchestrator_finalize.sh (includes VanillaEM + Level15EM
#      rows now) and let it commit + push.
#
# Launch with:
#   nohup bash /home/prashr/mapformer/master_finish.sh \
#       > /home/prashr/mapformer/master_finish.log 2>&1 &

set -u
cd /home/prashr

REPO=/home/prashr/mapformer

is_running () {
    # Use grep -E so alternation works; avoid pgrep's regex pitfalls.
    ps aux | grep -E "$1" | grep -v grep | grep -v master_finish >/dev/null
}

echo "[$(date)] master_finish: waiting for em + multilayer + level15_em orchestrators..."
while is_running "mapformer\.orchestrator_(em|multilayer|level15_em)\b"; do
    sleep 60
done
echo "[$(date)] em + multilayer + level15_em all done."

# 2. Extra baselines (LSTM / CoPE / MambaLike)
if ! compgen -G "$REPO/runs/MambaLike_*/seed*/MambaLike.pt" > /dev/null; then
    echo "[$(date)] Launching orchestrator_baselines..."
    nohup python3 -u -m mapformer.orchestrator_baselines \
        > "$REPO/orchestrator_baselines.log" 2>&1 &
    BL_PID=$!
    echo "  orchestrator_baselines PID: $BL_PID"
    wait $BL_PID
    echo "[$(date)] orchestrator_baselines done."
else
    echo "[$(date)] orchestrator_baselines already has checkpoints, skipping."
fi

# 3. Final evaluation — regenerate RESULTS_PAPER.md with all variants
#    (including VanillaEM + Level15EM + LSTM + CoPE + MambaLike rows)
echo "[$(date)] Running orchestrator_finalize.sh..."
cd "$REPO"
bash orchestrator_finalize.sh 2>&1 | tail -20

echo "[$(date)] master_finish: all done."
