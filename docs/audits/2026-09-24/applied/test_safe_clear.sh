#!/usr/bin/env bash
# Exercises safe_clear.sh on SCRATCH directories only (every "y" answer targets a scratch
# dir); the two real run dirs are given the answer "n" and must be refused before the prompt.
# usage: bash test_safe_clear.sh <scratch-dir> [old safe_clear.sh for comparison]
set -u
T="$1/sc_test"; rm -rf "$T"; mkdir -p "$T"; SC=/home/prashr/mapformer/safe_clear.sh; OLD="${2:-}"
cd /home/prashr
t(){ echo "--- $1"; shift; "$@" 2>&1 | sed "s#$T#<scratch>#g"; echo "   (exit ${PIPESTATUS[0]})"; }
mkdir -p $T/a_trained/p0 && touch $T/a_trained/.train_done $T/a_trained/p0/x.pt
t "1 .train_done inside" bash -c "echo y | bash $SC $T/a_trained"
mkdir -p $T/b_tagged && touch $T/b_tagged/.b_tagged_e900_done
t "2 other marker inside" bash -c "echo y | bash $SC $T/b_tagged"
mkdir -p $T/c_explicit && touch $T/c_marker_done
t "3 explicit absolute marker" bash -c "echo y | bash $SC $T/c_explicit $T/c_marker_done"
mkdir -p $T/rank_matched_e900
t "4 basename has a repo marker (.rank_matched_e900_done)" bash -c "echo y | bash $SC $T/rank_matched_e900"
mkdir -p $T/rank_proj_train
t "5 a repo marker names a prefix (.rank_proj_done)" bash -c "echo y | bash $SC $T/rank_proj_train"
mkdir -p $T/d_live; (setsid nohup python3 -c "import time; time.sleep(15)" $T/d_live > /dev/null 2>&1 &); sleep 1
t "6 a live python3 process names the dir" bash -c "echo y | bash $SC $T/d_live"
mkdir -p $T/e_partial && touch $T/e_partial/y.pt
t "7 unmarked partial run, answer n" bash -c "echo n | bash $SC $T/e_partial"; [ -d $T/e_partial ] && echo "   still there"
t "8 unmarked partial run, answer y (scratch)" bash -c "echo y | bash $SC $T/e_partial"; [ -d $T/e_partial ] || echo "   gone"
t "9 no such dir" bash -c "echo y | bash $SC $T/nonexistent"
t "10 REAL runs/rank_matched_e900 from /home/prashr, answer n" bash -c "echo n | bash mapformer/safe_clear.sh mapformer/runs/rank_matched_e900"
t "11 REAL runs/rank_proj_train from /home/prashr, answer n" bash -c "echo n | bash mapformer/safe_clear.sh mapformer/runs/rank_proj_train"
if [ -n "$OLD" ]; then
  t "12 HEAD's safe_clear.sh on runs/rank_matched_e900, answer n (the fail-open)" bash -c "echo n | bash $OLD mapformer/runs/rank_matched_e900"
fi
rm -rf "$T"
