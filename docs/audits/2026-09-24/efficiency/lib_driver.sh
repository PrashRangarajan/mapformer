# lib_driver.sh -- shared batch-driver helpers (PROPOSAL, audit 2026-09-24).
#
# Replaces logic copy-pasted across the run_*.sh drivers: 81 define on_gpu(), 65 busy(),
# 51 pick(), 37 is_gpu_free(); 34 still call `pgrep`, the self-matching trap CLAUDE.md
# documents twice. Every helper here encodes a rule that was bought by a failure:
#   rule 13  balance to the LESS loaded device (a fill-first picker idles a GPU)
#   pgrep    count interpreters by `comm`, never by pattern (shells match themselves)
#   md5      refuse to reuse checkpoints trained by different code
#   wait     `wait` returns regardless of child success: verify artifacts, not exit codes
#   done     a completion marker is set only after every artifact exists
#
# Usage in a driver:
#   source "$REPO/lib_driver.sh"
#   drv_lock "$REPO/.run_x.lock" "$LOG" || exit 1
#   drv_md5_guard "$R" model.py train.py environment.py || exit 1
#   G=$(drv_wait_slot 2 4500)                    # MAXPG=2, 4.5 GiB free
#   drv_launch "$R/${V}_s${S}.log" --device "cuda:$G" -- python3 -u -m mapformer.train_variant ...
#   drv_wait_dir "$R/"                           # until no trainer has $R/ in its argv
#   drv_require "$R/p0/A_s0/A.pt" ... || exit 1  # then, and only then:
#   drv_done "$REPO/.x_done"
set -uo pipefail

DRV_MODULE_RE="${DRV_MODULE_RE:-mapformer\\.train_}"   # which python modules count as trainers
DRV_POLL="${DRV_POLL:-30}"

# real trainers on device $1 (python3 by comm, so no shell ever matches)
drv_ntrain() {
  ps -u "$USER" -o comm=,args= |
    awk -v d="--device cuda:$1" -v re="$DRV_MODULE_RE" '$1=="python3" && $0 ~ re && index($0,d)' | wc -l
}
drv_freemem() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1"; }
drv_ngpu()    { nvidia-smi --query-gpu=index --format=csv,noheader | wc -l; }

# least-loaded device with a free slot and enough memory; empty if none
drv_pick() {  # $1 = max jobs per GPU, $2 = min free MiB
  local best="" bn=999 g n
  for ((g = 0; g < $(drv_ngpu); g++)); do
    n=$(drv_ntrain "$g")
    if [ "$n" -lt "$1" ] && [ "$(drv_freemem "$g")" -gt "$2" ] && [ "$n" -lt "$bn" ]; then best=$g; bn=$n; fi
  done
  echo "$best"
}
drv_wait_slot() { local g=""; while [ -z "$g" ]; do g=$(drv_pick "$1" "$2"); [ -z "$g" ] && sleep "$DRV_POLL"; done; echo "$g"; }

# single-instance lock; a refusal is logged because nohup discards stdout
drv_lock() {  # $1 = lock file, $2 = log
  exec 9>"$1"
  flock -n 9 || { echo "$(date) REFUSED: $1 is held" >> "$2"; return 1; }
}

# record the training code's md5 on first use; refuse to continue if it changed
drv_md5_guard() {  # $1 = run dir, rest = module files relative to $REPO
  local r="$1"; shift
  ( cd "$REPO" && md5sum "$@" ) > "$r/.code_md5.now"
  if [ -f "$r/code_md5.txt" ]; then
    cmp -s "$r/code_md5.txt" "$r/.code_md5.now" || { echo "ABORT: code changed since $r/code_md5.txt"; return 1; }
  else mv "$r/.code_md5.now" "$r/code_md5.txt"; fi
}

# detached launch that survives session teardown (harness trap: >2 min must be setsid)
drv_launch() {  # $1 = log file, then `--`, then the command
  local log="$1"; shift; [ "$1" = "--" ] && shift
  OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" setsid nohup "$@" > "$log" 2>&1 &
  sleep "${DRV_SPACING:-20}"   # let it allocate before the next pick reads free memory
}

# block until no trainer carries $1 in its argv
drv_wait_dir() {
  while [ "$(ps -u "$USER" -o comm=,args= | awk -v r="$1" '$1=="python3" && index($0,r)' | wc -l)" -gt 0 ]; do
    sleep 60; done
}
drv_require() { local m=0 f; for f in "$@"; do [ -f "$f" ] || { echo "MISSING $f"; m=$((m + 1)); }; done; [ "$m" -eq 0 ]; }
drv_done()    { touch "$1"; echo "$(date) DONE $1"; }
