#!/bin/bash
# Waits for orchestrator_em and orchestrator_multilayer to finish,
# then launches orchestrator_level15_em.
# Launch with:
#   nohup bash chain_level15_em.sh > chain_level15_em.log 2>&1 &

cd /home/prashr

echo "[$(date)] chain_level15_em waiting for EM + multilayer orchestrators..."
while pgrep -f "mapformer.orchestrator_em\|mapformer.orchestrator_multilayer" > /dev/null; do
    sleep 30
done

echo "[$(date)] EM + multilayer done. Launching Level15EM orchestrator."
nohup python3 -u -m mapformer.orchestrator_level15_em \
    > /home/prashr/mapformer/orchestrator_level15_em.log 2>&1 &
PID=$!
echo "  PID: $PID"

# Wait for it to finish
wait $PID
echo "[$(date)] Level15EM training complete."

# Re-run finalize to include Level15EM + VanillaEM rows in the tables.
# The finalize script will need variants_main updated to include these,
# but that's a separate edit; for now just log completion.
echo "[$(date)] chain_level15_em done."
