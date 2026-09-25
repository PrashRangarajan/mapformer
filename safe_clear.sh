#!/usr/bin/env bash
# Delete a run directory ONLY if it has no completion marker.
# Written 2026-09-08 after I rm -rf'd runs/forget_clock -- a COMPLETED batch,
# 48/48 with missing=0 -- because I read "no processes running" as "early in the
# batch" instead of "finished". The marker said otherwise and I did not look.
#
# Audit 2026-09-24: the marker was a RELATIVE path, so run from /home/prashr (where every
# `python3 -m mapformer.X` runs) it looked for /home/prashr/.<name>_done and never found
# one -- the guard failed OPEN. And only 46 of 351 run dirs have a marker named
# .<basename>_done; drivers also write <run-dir>/.train_done and .<prefix><TAG>_done.
# Now: markers resolve against the repo; any repo marker mentioning the basename, or naming a
# prefix of it, refuses; any .*done* file inside the directory refuses; and a live python3
# process whose command line names the directory refuses. Fails CLOSED: it may refuse a
# directory that is safe to delete (then delete it by hand, knowingly).
set -u
REPO="$(cd "$(dirname "$0")" && pwd)"
d="${1:?usage: safe_clear.sh <run-dir> [marker]}"
d="$(cd "$d" 2>/dev/null && pwd)" || { echo "no such dir: $1"; exit 1; }
b="$(basename "$d")"
m="${2:-$REPO/.${b}_done}"
case "$m" in /*) ;; *) m="$REPO/$m" ;; esac
[ -f "$m" ] && { echo "REFUSING: $m exists -- $d is a COMPLETED batch"; exit 1; }
[ -f "$d/.train_done" ] && { echo "REFUSING: $d/.train_done exists"; exit 1; }
inner=$(find "$d" -maxdepth 2 -name '.*done*' 2>/dev/null | head -3)
[ -n "$inner" ] && { echo "REFUSING: completion marker(s) inside $d: $inner"; exit 1; }
other=$(ls -a "$REPO" | grep -E "^\..*${b}.*_done$" | head -3)
[ -n "$other" ] && { echo "REFUSING: marker(s) naming $b exist: $other"; exit 1; }
# a driver's marker may name a PREFIX of the run dir (.rank_proj_done for runs/rank_proj_train)
pre=$(ls -a "$REPO" | sed -n 's/^\.\(.*\)_done$/\1/p' | while read -r x; do
        [ -n "$x" ] && case "$b" in "$x"*) echo ".${x}_done" ;; esac; done | head -3)
[ -n "$pre" ] && { echo "REFUSING: marker(s) naming a prefix of $b exist: $pre"; exit 1; }
live=$(ps -u "$USER" -o comm=,args= | awk -v r="$d" '$1=="python3" && index($0, r)' | wc -l)
[ "$live" -gt 0 ] && { echo "REFUSING: $live python3 process(es) reference $d"; exit 1; }
n=$(find "$d" -name '*.pt' 2>/dev/null | wc -l)
[ "$n" -gt 0 ] && echo "WARNING: $d holds $n checkpoints and no marker (partial run)"
read -r -p "delete $d ? [y/N] " a; [ "$a" = y ] && rm -rf "$d" && echo deleted || echo kept
