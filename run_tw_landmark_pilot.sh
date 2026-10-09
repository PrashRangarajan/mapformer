#!/usr/bin/env bash
# GPU pilot for TW_LANDMARK_PREREG.md (seed 150, OUTSIDE the batch's seeds 50-55; not reused). Five cells at 900 epochs:
#   MapWM 2L r=0   does a 2-layer path model learn the map in words at all (the O gate: reliance >= 0.2)?
#   MapWM 2L r=1   first look at the overshadowing cell
#   RoPE  2L r=1   does an index model learn name lookup (an induction route) within the budget?
#   MapWM 1L r=1   1-layer path model with names: name benefit ~0 expected (the one-layer argument)
#   RoPE  1L r=1   the one-layer argument's direct check: at the constant floor with names present
# plus a GPU reproduction check: MapWM 2L r=1 seed 151, 3 epochs, launched twice; losses and weights compared bitwise.
# Measures s/epoch and per-process GPU memory (sampled every 5 min) at the pilot's concurrency, then runs the readouts
# (CPU) and prints the per-run table. Launch (when GPUs are free):
#   cd /home/prashr/mapformer && setsid nohup bash run_tw_landmark_pilot.sh > /dev/null 2>&1 &
set -uo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
LOG="${TL_LOG:-$REPO/tw_landmark_pilot.log}"
source "$REPO/lib_driver.sh"
DRV_MAXPG="${TL_MAXPG:-4}"; DRV_SPACING="${TL_SPACING:-15}"; DRV_MINFREE="${TL_MINFREE:-7000}"
drv_lock "$REPO/.run_tw_landmark_pilot.lock" || exit 1
cd "$REPO/.."
R="${TL_RUNS:-$REPO/runs/tw_landmark_pilot}"; mkdir -p "$R/p0"   # TL_RUNS: dry-run test only
GUARD=(environment_textworld.py environment_tw_landmark.py train_tw_landmark.py tw_landmark_eval.py train.py
       train_variant.py model.py model_rank.py model_baseline_rope.py data_parallel.py environment.py rescore_hook.py
       stats_core.py analyze_tw_landmark.py)
echo "start $(date)" >> "$LOG"
drv_md5_guard "$R" "${GUARD[@]}" || exit 1
_drv_log "code version: $(cd "$REPO" && git rev-parse HEAD) $(cd "$REPO" && [ -z "$(git status --porcelain -- "${GUARD[@]}")" ] && echo clean || echo "DIRTY (a guarded file is modified or untracked)")"
( while :; do echo "$(date +%T) $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ';')" \
    >> "$R/gpu_mem.log"; sleep 300; done ) & MEMPID=$!
trap 'kill "$MEMPID" 2>/dev/null' EXIT
RUNS=()
launch() {  # arm layers rate seed epochs outdir
  local OUT="$6"
  [ -f "$OUT/train.json" ] && { echo "skip $OUT" >> "$LOG"; return; }
  if [ -n "$(ps -u "$USER" -o comm=,args= | awk -v o="$OUT" '$1=="python3" { for (i = 2; i < NF; i++) if ($i == "--output-dir" && $(i + 1) == o) { print; break } }')" ]; then
    echo "skip $OUT (already running)" >> "$LOG"; return
  fi
  mkdir -p "$OUT"; local G; G=$(drv_wait_slot)
  echo "$(date +%H:%M:%S) $1 L$2 r$3 s$4 e$5 -> cuda:$G" >> "$LOG"
  OMP_NUM_THREADS=2 drv_launch "$OUT.log" python3 -u -m mapformer.train_tw_landmark --arm "$1" --n-layers "$2" \
    --name-rate "$3" --seed "$4" --epochs "$5" --device "cuda:$G" --output-dir "$OUT"
}
for C in MapWM:2:0.0 MapWM:2:1.0 RoPE:2:1.0 MapWM:1:1.0 RoPE:1:1.0; do
  IFS=: read -r ARM L RATE <<< "$C"
  OUT="$R/p0/${ARM}_L${L}_r${RATE}_s150"; RUNS+=("$OUT")
  launch "$ARM" "$L" "$RATE" 150 900 "$OUT"
done
launch MapWM 2 1.0 151 3 "$R/repro_a"
drv_wait_dir "$R/repro_a"
launch MapWM 2 1.0 151 3 "$R/repro_b"
drv_wait_dir "$R/"
kill "$MEMPID" 2>/dev/null
REQ=(); for D in "${RUNS[@]}" "$R/repro_a" "$R/repro_b"; do REQ+=("$D/train.json"); done
drv_require "${REQ[@]}" || drv_fail "missing pilot runs"
python3 - "$R" <<'EOF' >> "$R/analysis.txt" 2>&1
import sys, torch
R = sys.argv[1]
a, b = (torch.load(f"{R}/repro_{k}/MapWM.pt", map_location="cpu", weights_only=False) for k in "ab")
same = a["losses"] == b["losses"] and all(torch.equal(a["model_state_dict"][k], b["model_state_dict"][k]) for k in a["model_state_dict"])
print(f"GPU reproduction (MapWM 2L r=1 s151, 3 epochs, launched twice): {'BITWISE IDENTICAL' if same else 'DIFFERENT'}"
      f" losses {a['losses']} vs {b['losses']}")
EOF
drv_md5_guard "$R" "${GUARD[@]}" || drv_fail "code changed before eval"
CUDA_VISIBLE_DEVICES="" python3 -u -m mapformer.tw_landmark_eval --runs "${RUNS[@]}" --threads 12 --out "$R/eval.json" \
  >> "$R/analysis.txt" 2>&1 || drv_fail eval
python3 - "$R" <<'EOF' >> "$R/analysis.txt" 2>&1
import json, re, sys, glob
R = sys.argv[1]; E = json.load(open(f"{R}/eval.json"))
print("\nrun | acc own / strip / uninf / named | reliance strip / uninf / named | name benefit | conflict path / name | class | final loss | s/epoch (last 50)")
for k, r in E.items():
    t = [float(x) for x in re.findall(r"\| ([0-9.]+)s$", open(f"{R}/p0/{k}.log").read(), re.M)][-10:]
    g = lambda f: r.get(f, float("nan"))
    print(f"{k} | {g('acc_own'):.4f} / {g('acc_strip'):.4f} / {g('acc_uninf'):.4f} / {g('acc_named'):.4f} | "
          f"{g('rel_strip'):+.4f} / {g('rel_uninf'):+.4f} / {g('rel_named'):+.4f} | {g('name_benefit'):+.4f} | "
          f"{g('conf_path'):.3f} / {g('conf_name'):.3f} | {r['cls']} | {r['final_loss']:.4f} | {sum(t) / max(1, len(t)):.2f}")
EOF
drv_done "$REPO/.tw_landmark_pilot_done" "$R/eval.json" "$R/analysis.txt"
