#!/usr/bin/env bash
# Waits for the long-T eval (and the EM-fix run) to finish, then commits+pushes
# whatever results exist, so nothing is lost if the session closes.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$REPO"
for _ in $(seq 1 240); do [ -f "$REPO/.longt_done" ] && break; sleep 30; done
for _ in $(seq 1 120); do [ -f "$REPO/.em_fix_done" ] && break; sleep 30; done
git add eval_hiergoal_longT.py HIERGOAL_LONGT.md HIERGOAL_LONGT.json \
        model_em_fixed.py train_variant.py run_em_fix.sh \
        EM_FIX_COMP.md EM_FIX_COMP.json EM_FIX_HIERGOAL.md EM_FIX_HIERGOAL.json 2>/dev/null
git diff --cached --quiet || {
  git commit -q -m "EM eq.13 fix + long-T hier-goal eval (results, interpretation pending)

EM fix: Hadamard on probabilities per paper eq.13 rather than on raw signed
logits (the original made the gate an XNOR: 69.9% of positive scores came from
double-mismatches). Long-T: existing checkpoints evaluated at T_explore up to
2048 (32x the training horizon) to test whether hier-goal has any headroom
above ~0.95. Auto-committed by autocommit_longt.sh."
  git push origin main; }
touch "$REPO/.longt_autocommit_done"
