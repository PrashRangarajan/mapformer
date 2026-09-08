#!/usr/bin/env bash
# Delete a run directory ONLY if it has no completion marker.
# Written 2026-09-08 after I rm -rf'd runs/forget_clock -- a COMPLETED batch,
# 48/48 with missing=0 -- because I read "no processes running" as "early in the
# batch" instead of "finished". The marker said otherwise and I did not look.
set -u
d="${1:?usage: safe_clear.sh <run-dir> [marker]}"
m="${2:-.$(basename "$d")_done}"
[ -f "$m" ] && { echo "REFUSING: $m exists -- $d is a COMPLETED batch"; exit 1; }
n=$(find "$d" -name '*.pt' 2>/dev/null | wc -l)
[ "$n" -gt 0 ] && echo "WARNING: $d holds $n checkpoints and no marker (partial run)"
read -r -p "delete $d ? [y/N] " a; [ "$a" = y ] && rm -rf "$d" && echo deleted || echo kept
