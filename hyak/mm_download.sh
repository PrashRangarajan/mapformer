#!/bin/bash
# Resumable download of the official Memory Maze offline dataset (Pasukonis et al. 2022, README of
# github.com/jurgisp/memory-maze: "Dataset download here", a public Google Drive folder with one folder per maze
# size, each holding eval.zip (1k episodes) and train-part0..9.zip (29k episodes); one npz per episode inside).
#
# Run as a CPU-only Slurm job (account, partition and output path on the sbatch command line, not here):
#   MM_DATA=<scratch dir> sbatch --export=ALL,MM_DATA=... --cpus-per-task=4 --mem=8G --time=08:00:00 \
#       --requeue --open-mode=append hyak/mm_download.sh
# Keeps the archives as zips (a few large files; GPFS is slow at many small ones); the gate reads members in place.
# Per file: <name>.zip.part grows with curl -C - (Drive serves byte ranges), then a full CRC test (zipfile.testzip),
# a member count and a field check on one member; only then the .part is renamed and <name>.ok written
# (artifact first, then marker). Re-running skips files with .ok. Order: what the gate needs first.
set -uo pipefail
: "${MM_DATA:?set MM_DATA to the download directory}"
PY="${MM_PY:-python3}"
PAR="${MM_PAR:-3}"
SIZES="${MM_SIZES:-15x15 9x9}"

# size  name  drive-file-id  (public ids from the README's folder, listed 2026-10-26)
LIST="
15x15 eval        1usfAYW48V0L0flQMB_yI9XVk5xSKh-Hm
9x9   eval        18YPeLnu_TkVx7r4mZjeQ9gSMPEI-jtiP
15x15 train-part0 19s80X0wUtQznkmYEzIwpJHevVQBbLCST
9x9   train-part0 1EdWafDjZG3VUo7CXTERx7fTx4IqWWNvC
15x15 train-part1 1yq-59zawM7N0XNhONmqR3Zs_xPCXBI8m
15x15 train-part2 1r19Lpy3iklyfGsl-m1bAZZVhRl5uRWLT
15x15 train-part3 1FZICWZkwOhG4rIIXk1P-BIWnJ2ZPj4Eu
15x15 train-part4 1HlbLM0kqeBHf1-_kd7shcJ9R-70g1jMo
15x15 train-part5 1HtjmGrMqkOfGpztfTteFSlGTXYPPfyQX
15x15 train-part6 11Dk2NWKcmMRh6sNcJeLqU_AQELawpBtP
15x15 train-part7 10yjJ_kHZ-LpTkceXD_vkoxxep9L32Qoy
15x15 train-part8 1tVxYcy9wBvihMi61q8fTG22XAJqB2f0H
15x15 train-part9 1cZDRsXYJQ3fUgDT0f-AhM3pmibtObHM5
9x9   train-part1 1Kyn8seqBiU8drJ8uyS2RSgeefLatUCzi
9x9   train-part2 1FaxOJJu6hbPjkab5dtCu-QWxMVDEcj5T
9x9   train-part3 1S6G-knZUG2V0Jhobseb_2-_V5n2jhVMS
9x9   train-part4 1br0-EZfh4aTY5E66_m-PXR9i0v1IA1-8
9x9   train-part5 17ZGVQ8fGVv9FlEQiLy-j_M9r4NlABY3i
9x9   train-part6 10vPLCDjv4AC35TkdwAkgUrrAvAhIQJ9v
9x9   train-part7 1vebIvddC4UG78YSpbGTJlr-Hl1ScZeGV
9x9   train-part8 1KmVoAofGWnwBJ0EqClYqWNBzENMA8riE
9x9   train-part9 1N4eiw0DV-HrxWSkhRDmBV6skGTiOUz5B
"

verify() {  # $1 zip path -> prints "members=N bytes=B" on success, exit 1 otherwise
  "$PY" -I - "$1" <<'PYEOF'
import sys, zipfile, io
import numpy as np
p = sys.argv[1]
z = zipfile.ZipFile(p)
names = [n for n in z.namelist() if n.endswith('.npz')]
bad = z.testzip()
if bad is not None:
    print('CRC FAIL', bad); sys.exit(1)
need = {'image', 'action', 'agent_pos', 'agent_dir', 'maze_layout', 'targets_pos', 'target_color', 'reward'}
for n in (names[0], names[len(names) // 2], names[-1]):
    ep = np.load(io.BytesIO(z.read(n)))
    miss = need - set(ep.files)
    if miss or ep['image'].shape[1:] != (64, 64, 3):
        print('FIELDS FAIL', n, sorted(miss), ep['image'].shape); sys.exit(1)
import os
print(f'members={len(names)} bytes={os.path.getsize(p)} T={ep["image"].shape[0]}')
PYEOF
}

fetch_one() {  # size name id
  local size="$1" name="$2" id="$3"
  local dir="$MM_DATA/$size"; mkdir -p "$dir"
  local out="$dir/$name.zip" part="$dir/$name.zip.part" ok="$dir/$name.ok"
  [ -f "$ok" ] && { echo "skip $size/$name (ok)"; return 0; }
  local url="https://drive.usercontent.google.com/download?id=$id&export=download&confirm=t"
  local nbad=0 t_end=$(( START + ${MM_WINDOW_S:-27000} ))
  while [ "$(date +%s)" -lt "$t_end" ]; do
    if [ ! -f "$out" ]; then
      # Honour Drive's per-file download quota: start or resume ONLY when a plain (non-range) GET serves the file
      # (its first bytes are the zip magic 'PK'); a quota page is HTML with status 200. Range requests are not used
      # to get around a quota. Back off MM_BACKOFF_S (default 1 h) and retry within the job window.
      local magic
      magic=$(curl -s -L --max-time 60 "$url" 2>/dev/null | head -c 2)
      if [ "$magic" != "PK" ]; then
        echo "$(date -Is) $size/$name: not served (quota or error page); retry in ${MM_BACKOFF_S:-3600} s"
        sleep "${MM_BACKOFF_S:-3600}"; continue
      fi
      curl -s -S -L -C - --retry 5 --retry-delay 30 -o "$part" "$url"
      local rc=$?
      if [ $rc -ne 0 ] && [ $rc -ne 33 ]; then echo "$(date -Is) $size/$name: curl rc $rc"; sleep 300; continue; fi
      mv "$part" "$out"
    fi
    local v
    if v=$(verify "$out" 2>&1 | tail -1); then
      echo "$v" > "$ok"; echo "$(date -Is) OK $size/$name $v"; return 0
    fi
    nbad=$((nbad + 1))
    echo "$(date -Is) $size/$name: verify failed ($v); deleting ($nbad/3)"
    rm -f "$out" "$part"
    [ "$nbad" -ge 3 ] && break
  done
  echo "$(date -Is) GAVE UP $size/$name"; return 1
}

START=$(date +%s)
echo "=== $(date -Is) on $(hostname): download to MM_DATA, sizes '$SIZES', $PAR parallel ==="
df -h "$MM_DATA" | tail -1
# shellcheck disable=SC2086
echo "$LIST" | awk -v s="$SIZES" 'BEGIN{n=split(s,a," "); for(i=1;i<=n;i++) want[a[i]]=1} NF==3 && ($1 in want)' |
{ while read -r size name id; do
  while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do sleep 5; done
  fetch_one "$size" "$name" "$id" &
done
wait  # inside the pipeline's subshell, which owns the background fetches
}
echo "=== $(date -Is) pass finished ==="
n_ok=$(ls "$MM_DATA"/*/*.ok 2>/dev/null | wc -l)
echo "ok markers: $n_ok"
df -h "$MM_DATA" | tail -1
