#!/usr/bin/env bash
# GATES FOR THE ALIASING-CONTROLLED SWEEP. Run BEFORE any GPU is spent.
#
# THE QUESTION. The surviving MiniWorld claim is "the position effect scales with
# observation ALIASING": grid 8 (2 cells/token) -0.010, grid 32 (32 cells/token)
# +0.173, torus (128 cells/token) +0.461. But aliasing CO-VARIES with map size in
# that series, so it is correlational -- "bigger map" explains it equally well.
#
# THE MANIPULATION. Hold grid size FIXED at 32 and vary n_obs so aliasing sweeps
# down to grid 8's level on a grid-32 map (p_empty 0.5 -> ~512 occupied cells):
#     n_obs=16  -> 32 cells/token   (the existing +0.173 anchor)
#     n_obs=64  ->  8 cells/token   (new intermediate)
#     n_obs=256 ->  2 cells/token   (grid 8's aliasing, at grid 32's map size)
#
# PRE-REGISTERED PREDICTIONS, so this cannot be read after the fact:
#   ALIASING drives it -> effect collapses toward 0 at n_obs=256, monotone in
#                         cells/token. The claim becomes manipulated, not
#                         correlational.
#   MAP SIZE drives it -> effect stays near +0.173 at all three n_obs. The
#                         aliasing story is dead and must be withdrawn.
#   Non-monotone       -> neither; report the curve and claim nothing.
#
# Three points, not two -- this session already learned that two points make a
# line and a line is not a trend (rule 5 corollary).
#
# WHY GATE FIRST. The knob sweep trained 42 models and gated afterwards; one
# condition turned out void (0.932 order-1 action shortcut). Raising n_obs to 256
# also moves the vocab from 27 to 267, which is exactly the setting where an
# out-of-range token id crashes as a FAKE CUDA OOM -- hence the new G8 gate.
set -uo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"; cd "$REPO/.."
OUT="$REPO/ALIASING_GATES.md"; : > "$OUT"
LOG="$REPO/alias_gates.log"; echo "alias gates start $(date)" > "$LOG"

for NOBS in 16 64 256; do
  echo "$(date +%H:%M) gating grid32 n_obs=$NOBS" >> "$LOG"
  python3 -u -m mapformer.validate_miniworld \
      --grids 32 --T 512 --n-episodes 40 --n-obs "$NOBS" --oracle \
      --out "$REPO/_gate_n${NOBS}.md" >> "$LOG" 2>&1
  {
    echo "## grid 32, n_obs=${NOBS}"
    echo
    cat "$REPO/_gate_n${NOBS}.md" 2>/dev/null || echo "(gate produced no output)"
    echo
  } >> "$OUT"
  rm -f "$REPO/_gate_n${NOBS}.md"
done

# Refuse to greenlight unless every gate in every condition passed.
python3 -u - "$OUT" >> "$LOG" <<'PYEOF'
import re, sys
s = open(sys.argv[1]).read()
fails = re.findall(r'^.*\bFAIL\b.*$', s, re.M)
print(f"\n{'GATES CLEAN -- safe to train' if not fails else 'GATE FAILURES:'}")
for f in fails:
    print("   ", f.strip())
open("/home/prashr/mapformer/.alias_gates_clean" if not fails
     else "/home/prashr/mapformer/.alias_gates_FAILED", "w").write("\n".join(fails))
PYEOF
touch "$REPO/.alias_gates_done"
echo "$(date) DONE" >> "$LOG"
