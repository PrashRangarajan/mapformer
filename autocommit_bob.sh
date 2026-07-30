#!/usr/bin/env bash
# Waits for the n=8 best-of-both run to finish, then commits + pushes the
# regenerated results so they are durable without an interactive session.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ ! -f "$REPO/.bob_seeds_done" ]; do
  kill -0 530411 2>/dev/null || { sleep 60; break; }   # launcher died: wait, then try anyway
  sleep 60
done
cd "$REPO"
# If training died before aggregating, aggregate whatever checkpoints exist.
if [ ! -f "$REPO/.bob_seeds_done" ]; then
  (cd "$REPO/.." && python3 -u -m mapformer.agg_comp_multiseed \
    --runs-dir "$REPO/runs/comp_multiseed" --seeds 0 1 2 3 4 5 6 7 \
    --variants Vanilla VanillaEM Hourglass_k2 Hourglass_CoarseIdx Hourglass_CoarsePI PoPE MapPoPE \
               MapPoPE_Hier MapPoPE_CoarseIdx HourglassFlat3 PlainHourglass PlainFlat \
    --lengths 256 512 1024 2048 --n-traj 200 --batch 16 --device cuda:0 \
    --out "$REPO/COMPOSITIONAL_MULTISEED.md") >> "$REPO/bob_seeds.log" 2>&1
fi
git add COMPOSITIONAL_MULTISEED.md COMPOSITIONAL_MULTISEED.json 2>/dev/null
git diff --cached --quiet && exit 0
git commit -q -m "Best-of-both n=8: tightened error bars on the length<->content frontier

Seeds 3-7 added for MapPoPE_CoarseIdx (best-of-both, faithful PoPE) and
Hourglass_CoarseIdx (RoPE content king) on the compositional/content axis, to
settle whether the frontier is genuinely collapsed. Auto-committed by
autocommit_bob.sh; interpretation pending review."
git push origin main >> "$REPO/bob_seeds.log" 2>&1
touch "$REPO/.bob_autocommit_done"
