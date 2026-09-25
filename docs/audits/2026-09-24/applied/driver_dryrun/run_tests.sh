#!/usr/bin/env bash
# Dry-run the lib_driver.sh drivers in a FAKE repo with a stub python3 (no training).
# usage: mkdir -p D/bin && cp stub_python3 D/bin/python3 && chmod +x D/bin/python3 && bash run_tests.sh D
set -u
D="$1"; M=/home/prashr/mapformer
F="$D/fake/mapformer"; rm -rf "$D/fake"; mkdir -p "$F/runs"
cp "$M/lib_driver.sh" "$M/run_rank_matched.sh" "$M/run_rank_proj.sh" "$M/run_rank_perhead_pilot.sh" "$F/"
for f in model.py model_rank.py model_rank_perhead.py train.py train_variant.py environment.py data_parallel.py; do cp "$M/$f" "$F/"; done
export PATH="$D/bin:$PATH" STUB_CALLS="$D/calls.log" DRV_SPACING=1 DRV_POLL=1 DRV_WAIT_POLL=1 STUB_TRAIN_S=4
: > "$D/calls.log"
say(){ echo; echo "=== $*"; }
say "1. run_rank_matched.sh TAG=_dry EPOCHS=3 SEEDS='0 1 2' (6 jobs, default MAXPG=2)"
( EPOCHS=3 TAG=_dry SEEDS="0 1 2" bash "$F/run_rank_matched.sh" ); echo "exit $?"
grep -E "START|eval|probe|analyze" "$D/calls.log" | sed "s#$F#<fake>#g"
echo "max trainers on one device at any launch: $(grep -o 'incl. me: [0-9]*' "$D/calls.log" | awk '{print $3}' | sort -n | tail -1)"
ls -a "$F" | grep -E "_done$"; ls "$F" | grep "RANK_MATCHED_dry"
say "2. same again: every checkpoint reused (matches() reads the config), evals rerun, marker reset then set"
: > "$D/calls.log"; ( EPOCHS=3 TAG=_dry SEEDS="0 1 2" bash "$F/run_rank_matched.sh" ); echo "exit $?"
grep -c START "$D/calls.log" | sed 's/^/launches: /'; grep -c "^skip" "$F/rank_matched_dry.log" | sed 's/^/skip lines in log (cumulative): /'; ls -a "$F" | grep -E "_dry_done$"
say "3. a reused checkpoint at another budget (EPOCHS=4) must ABORT, no marker"
( EPOCHS=4 TAG=_dry SEEDS="0 1 2" bash "$F/run_rank_matched.sh" ); echo "exit $?"; tail -n 1 "$F/rank_matched_dry.log" | sed "s#$F#<fake>#g"; ls -a "$F" | grep -cE "_dry_done$" | sed 's/^/markers: /'
say "4. training code changed since code_md5.txt must ABORT"
echo "# edit" >> "$F/train.py"; ( EPOCHS=3 TAG=_dry SEEDS="0 1 2" bash "$F/run_rank_matched.sh" ); echo "exit $?"; tail -n 1 "$F/rank_matched_dry.log" | sed "s#$F#<fake>#g"
cp "$M/train.py" "$F/train.py"
say "5. a failing evaluator: drv_fail, no marker"
( STUB_FAIL=eval_rank_strata EPOCHS=3 TAG=_dry SEEDS="0 1 2" bash "$F/run_rank_matched.sh" ); echo "exit $?"; tail -n 1 "$F/rank_matched_dry.log"; ls -a "$F" | grep -cE "_dry_done$" | sed 's/^/markers: /'
say "6. lock held by another driver: REFUSED"
( exec 9>"$F/.run_rank_matched.lock"; flock -n 9; EPOCHS=3 TAG=_dry2 SEEDS="0" bash "$F/run_rank_matched.sh"; echo "exit $?" ); tail -n 1 "$F/rank_matched_dry2.log"
say "7. run_rank_perhead_pilot.sh"
: > "$D/calls.log"; ( bash "$F/run_rank_perhead_pilot.sh" ); echo "exit $?"; grep -E "START|eval|probe|analyze" "$D/calls.log" | sed "s#$F#<fake>#g"; ls -a "$F" | grep perhead_pilot_done
say "8. run_rank_proj.sh with the e900c control FAILED in its latest invocation: must fail, not wait forever"
for s in 0 1 2 3 4 5 6 7; do mkdir -p "$F/runs/rank_proj/p0/Vanilla_s$s"; touch "$F/runs/rank_proj/p0/Vanilla_s$s/Vanilla.pt"; done
printf 'start old\nFAILED: something -- done marker NOT set\nstart new\nFAILED: eval -- done marker NOT set\n' > "$F/rank_matched_e900c.log"
: > "$D/calls.log"; ( timeout 120 bash "$F/run_rank_proj.sh" ); echo "exit $?"; grep -c START "$D/calls.log" | sed 's/^/launches: /'; tail -n 1 "$F/rank_proj.log" | sed "s#$F#<fake>#g"
say "9. an OLD failure followed by a clean new invocation: keeps waiting (timeout 8 s, exit 124 expected), then finishes once the marker appears"
printf 'start old\nFAILED: something\nstart new\nrunning\n' > "$F/rank_matched_e900c.log"
( timeout 8 bash "$F/run_rank_proj.sh" ); echo "exit $? (124 = still waiting, as it should)"
sleep 3   # the killed driver's last `sleep` child still holds the inherited lock fd for <= 1 s
touch "$F/.rank_matched_e900c_done"; ( bash "$F/run_rank_proj.sh" ); echo "exit $?"; tail -n 1 "$F/rank_proj.log" | sed "s#$F#<fake>#g"
say "10. a reused proj checkpoint trained from ANOTHER init must fail (patch 05)"
/usr/bin/python3 -c "
import torch; p='$F/runs/rank_proj_train/p0/Vanilla_s3/Vanilla.pt'; b=torch.load(p, weights_only=False); b['config']['init_from']='/elsewhere.pt'; torch.save(b, p)"
rm -f "$F/.rank_proj_done"; ( bash "$F/run_rank_proj.sh" ); echo "exit $?"; tail -n 1 "$F/rank_proj.log" | sed "s#$F#<fake>#g"; ls -a "$F" | grep -c rank_proj_done | sed 's/^/rank_proj markers: /'
