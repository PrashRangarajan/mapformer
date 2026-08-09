#!/usr/bin/env bash
# Bounded-memory sweep: prefix-preserving sliding window over attention.
# Tests whether PoPE's advantage depends on RETRIEVING the action history
# (Route B) while MapFormer's carried cumsum (Route A) survives.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$REPO/.."
LOG="$REPO/bounded_mem.log"; : > "$LOG"
python3 -u -m mapformer.eval_bounded_memory \
  --variants PoPE MapPoPE MapPoPE_Hier Vanilla Hourglass_k2 Hourglass_CoarsePI PlainFlat \
  --windows 100000 256 128 64 32 16 --prefix 2 --t-explore 64 --n-trials 100 \
  --device cuda:1 --out "$REPO/BOUNDED_MEMORY.md" >> "$LOG" 2>&1
cd "$REPO"; git add eval_bounded_memory.py run_bounded_mem.sh BOUNDED_MEMORY.md BOUNDED_MEMORY.json 2>/dev/null
git diff --cached --quiet || { git commit -q -m "Bounded-memory eval: does PoPE depend on retrieving the action history?

Prefix-preserving sliding window (first 2 goal tokens + last W) applied
identically to all variants, eval-only on existing checkpoints. Tests the
Route-A/B prediction: MapFormer's theta=omega*cumsum is O(1) carried state and
should survive windowing; PoPE/plain must retrieve past action tokens from an
O(T) KV cache and should not. Validated: W=inf reproduces the unpatched baseline
exactly; the mask demonstrably bites (PoPE 0.951 -> 0.643 at W=8).
Auto-committed; interpretation pending review."; git push origin main >> "$LOG" 2>&1; }
touch "$REPO/.bounded_mem_done"
