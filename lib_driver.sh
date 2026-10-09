# lib_driver.sh -- shared batch-driver helpers (audit 2026-09-24, efficiency #8; applied 2026-09-24).
#
# Replaces logic copy-pasted across the run_*.sh drivers (81 define on_gpu(), 65 busy(), 51
# pick(); 34 used `pgrep`, whose -f form matches the shell that typed the pattern). Every
# helper encodes a rule that was bought by a failure:
#   rule 13  balance to the LESS loaded device; a fill-first picker idles a GPU
#   pgrep    count interpreters by `comm` (python3), never by pattern: shells match themselves
#   MAXPG    jobs per GPU default 2: at the rank config 5 jobs/GPU cost 5-10% vs one alone and
#            ~3.4 GB each (efficiency audit #4); raise it only after re-measuring
#   md5      refuse to reuse checkpoints trained by different code
#   setsid   anything over ~2 min must survive the launching session
#   wait     `wait` returns regardless of child success: verify artifacts, not exit codes
#   done     a completion marker is set only after every artifact exists
#
# Usage (the driver sets REPO and LOG first):
#   source "$REPO/lib_driver.sh"
#   drv_lock "$REPO/.run_x.lock" || exit 1
#   drv_md5_guard "$R" model.py train.py environment.py || exit 1
#   G=$(drv_wait_slot)                                   # least-loaded GPU with a free slot
#   drv_launch "$R/${V}_s${S}.log" python3 -u -m mapformer.train_variant ... --device "cuda:$G"
#   drv_wait_dir "$R/"                                   # until no python3 has $R/ in its argv
#   drv_require "$R/p0/A_s0/A.pt" ... || drv_fail "missing checkpoints"
#   drv_done "$REPO/.x_done"                             # only after drv_require passed
#
# Knobs (environment): DRV_MAXPG (2), DRV_MINFREE MiB (4500), DRV_POLL s (30), DRV_SPACING s
# (45), DRV_MODULE_RE (kept for old drivers; IGNORED for slot counting since 2026-09-30: every
# mapformer.train_* process on a GPU counts, so concurrent drivers share one budget -- audit m3),
# DRV_DRYRUN=1 (log the launch instead of running it).
# DRV_GPUS (space-separated device indices, e.g. "0"; default every GPU) restricts drv_pick (2026-10-09).

DRV_MODULE_RE="${DRV_MODULE_RE:-mapformer[.]train_}"
DRV_MAXPG="${DRV_MAXPG:-2}"
DRV_MINFREE="${DRV_MINFREE:-4500}"
DRV_POLL="${DRV_POLL:-30}"
DRV_SPACING="${DRV_SPACING:-45}"

_drv_log() { echo "$(date '+%F %T') $*" >> "${LOG:-/dev/stderr}"; }

# real trainers on device $1 (python3 by comm, so no shell ever matches)
drv_ntrain() {
  ps -u "$USER" -o comm=,args= |
    awk -v d="--device cuda:$1" -v re="mapformer[.]train_" '$1=="python3" && $0 ~ re && index($0, d)' | wc -l
}
drv_freemem() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1"; }
drv_ngpu()    { nvidia-smi --query-gpu=index --format=csv,noheader | wc -l; }

# least-loaded device with a free slot and enough memory; empty if none
drv_pick() {  # [$1 max jobs per GPU] [$2 min free MiB]
  local maxpg="${1:-$DRV_MAXPG}" minfree="${2:-$DRV_MINFREE}" best="" bn=999 g n ng
  ng=$(drv_ngpu)
  for g in ${DRV_GPUS:-$(seq 0 $((ng - 1)))}; do
    n=$(drv_ntrain "$g")
    if [ "$n" -lt "$maxpg" ] && [ "$(drv_freemem "$g")" -gt "$minfree" ] && [ "$n" -lt "$bn" ]; then
      best=$g; bn=$n
    fi
  done
  echo "$best"
}
drv_wait_slot() {  # [$1 max jobs per GPU] [$2 min free MiB]
  local g=""
  while [ -z "$g" ]; do g=$(drv_pick "$@"); [ -z "$g" ] && sleep "$DRV_POLL"; done
  echo "$g"
}

# single-instance lock on fd 9; a refusal is logged because nohup discards stdout
drv_lock() {  # $1 = lock file
  exec 9>"$1"
  flock -n 9 || { _drv_log "REFUSED: $1 is held by another driver"; return 1; }
}

# record the training code's md5 on first use; refuse to continue if it changed
drv_md5_guard() {  # $1 = run dir, rest = module files relative to $REPO
  local r="$1"; shift
  mkdir -p "$r"
  ( cd "$REPO" && md5sum "$@" ) > "$r/.code_md5.now" || { _drv_log "ABORT: md5sum failed"; return 1; }
  if [ -f "$r/code_md5.txt" ]; then
    cmp -s "$r/code_md5.txt" "$r/.code_md5.now" ||
      { _drv_log "ABORT: training code changed since $r/code_md5.txt"; return 1; }
    rm -f "$r/.code_md5.now"
  else
    mv "$r/.code_md5.now" "$r/code_md5.txt"
  fi
}

# detached launch that survives session teardown; then give it time to claim GPU memory
drv_launch() {  # $1 = log file, rest = the command
  local log="$1"; shift
  if [ "${DRV_DRYRUN:-0}" = 1 ]; then _drv_log "DRYRUN: $*"; return 0; fi
  OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" setsid nohup "$@" > "$log" 2>&1 &
  _drv_log "launched pid $! -> $log"
  sleep "$DRV_SPACING"
}

# block until no python3 process carries $1 in its argv (the batch's run dir, with a slash)
drv_wait_dir() {
  while [ "$(ps -u "$USER" -o comm=,args= | awk -v r="$1" '$1=="python3" && index($0, r)' | wc -l)" -gt 0 ]; do
    sleep "${DRV_WAIT_POLL:-60}"
  done
}

drv_require() {  # files...; logs each missing one, fails if any is missing
  local m=0 f
  for f in "$@"; do [ -f "$f" ] || { _drv_log "MISSING $f"; m=$((m + 1)); }; done
  [ "$m" -eq 0 ]
}
drv_fail() { _drv_log "FAILED: $* -- done marker NOT set"; exit 1; }
drv_done() {  # $1 = marker, rest = artifacts that must exist first
  local mk="$1"; shift
  drv_require "$@" || drv_fail "artifacts missing, not setting $mk"
  touch "$mk"; _drv_log "DONE $mk"
}
