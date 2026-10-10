#!/usr/bin/env bash
# TW_LANDMARK existence runs after Amendment 2 (criterion (b) failed): can a 2- or 3-layer INDEX model learn name lookup
# in this task at all? Seed 150 (outside the batch), name rate 1, both on their own GPU:
#   RoPE 2L, 1800 epochs  (does the lookup appear with twice the budget?)
#   RoPE 3L,  900 epochs  (does a third layer give it?)
# Pass rule = Amendment 1's criterion (b), unchanged, applied to each run: name benefit >= 0.10 AND acc_named >= 0.85.
# Written and committed before launch; outcomes go into Amendment 3. Launch:
#   cd /home/prashr/mapformer && setsid nohup bash docs/audits/2026-10-10/tw_landmark_exist.sh > /dev/null 2>&1 &
set -uo pipefail
REPO=/home/prashr/mapformer
R=$REPO/runs/tw_landmark_pilot/exist; mkdir -p "$R"; LOG=$R/exist.log
exec 9>"$REPO/.tw_landmark_exist.lock"; flock -n 9 || { echo "$(date '+%F %T') REFUSED: lock held" >> "$LOG"; exit 1; }
log(){ echo "$(date '+%F %T') $*" >> "$LOG"; }
log "code version: $(cd "$REPO" && git rev-parse HEAD) $(cd "$REPO" && git diff --quiet && echo clean || echo DIRTY)"
cd /home/prashr
( while :; do echo "$(date +%T) $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ';')" >> "$R/gpu_mem.log"; sleep 30; done ) & MEMPID=$!
trap 'kill "$MEMPID" 2>/dev/null' EXIT
run(){ # name layers epochs gpu
  [ -f "$R/$1/train.json" ] && { log "skip $1 (done)"; return; }
  mkdir -p "$R/$1"
  OMP_NUM_THREADS=2 setsid nohup timeout 8h python3 -u -m mapformer.train_tw_landmark --arm RoPE --n-layers "$2" \
    --name-rate 1.0 --seed 150 --epochs "$3" --device "cuda:$4" --output-dir "$R/$1" > "$R/$1.log" 2>&1 < /dev/null &
  log "launched $1 (pid $!) on cuda:$4"
}
run RoPE_L2_r1.0_e1800_s150 2 1800 0
sleep 30
run RoPE_L3_r1.0_e900_s150 3 900 1
sleep 120
while ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /mapformer[.]train_tw_landmark/ && /exist/' | grep -q .; do sleep 120; done
for d in RoPE_L2_r1.0_e1800_s150 RoPE_L3_r1.0_e900_s150; do [ -f "$R/$d/train.json" ] || { log "FAILED: $d missing train.json"; exit 1; }; done
CUDA_VISIBLE_DEVICES="" python3 -u -m mapformer.tw_landmark_eval --runs "$R/RoPE_L2_r1.0_e1800_s150" "$R/RoPE_L3_r1.0_e900_s150" \
  --threads 12 --out "$R/eval.json" > "$R/eval_out.txt" 2>&1 || { log "FAILED: eval"; exit 1; }
python3 - "$R" <<'EOF' >> "$R/eval_out.txt" 2>&1
import json, sys
E = json.load(open(f"{sys.argv[1]}/eval.json"))
for k, r in E.items():
    ok = r["name_benefit"] >= 0.10 and r["acc_named"] >= 0.85
    print(f"{k}: own {r['acc_own']:.4f} strip {r['acc_strip']:.4f} uninf {r['acc_uninf']:.4f} named {r['acc_named']:.4f} "
          f"name benefit {r['name_benefit']:+.4f} conflict path/name {r['conf_path']:.3f}/{r['conf_name']:.3f} {r['cls']} "
          f"final loss {r['final_loss']:.4f} -> criterion (b) {'PASS' if ok else 'FAIL'}")
EOF
touch "$REPO/.tw_landmark_exist_done"; log "DONE"
