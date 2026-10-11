#!/usr/bin/env bash
# TW_LANDMARK budget pilot at 3 layers (after Amendment 3). Seed 150 (outside the batch), 2 runs per GPU:
#   MapWM 3L r=0 1200 ep, MapWM 3L r=1 1200 ep (cuda:0); RoPE 3L r=1 1200 ep, RoPE 3L r=1 1800 ep (cuda:1).
# Rules, fixed before launch (outcomes -> Amendment 4):
#   (a) MapWM 3L r=0 path-integrates: rel_strip >= 0.2 and acc_strip >= 0.90 (criterion (a) unchanged);
#   (b) RoPE 3L r=1 uses names: name benefit >= 0.10 and acc_named >= 0.85 (criterion (b) unchanged);
#   budget = the smaller of {1200, 1800} at which every run there is SOLVED (final-5% training loss < 0.05) and (a), (b)
#   hold; if neither, report and redesign. MapWM 3L r=1 is read only (name benefit, rel_uninf, O-ID numbers): no rule.
# Launch: cd /home/prashr/mapformer && setsid nohup bash docs/audits/2026-10-10/tw_landmark_budget.sh > /dev/null 2>&1 &
set -uo pipefail
REPO=/home/prashr/mapformer
R=$REPO/runs/tw_landmark_pilot/budget3; mkdir -p "$R"; LOG=$R/budget.log
exec 9>"$REPO/.tw_landmark_budget.lock"; flock -n 9 || { echo "$(date '+%F %T') REFUSED: lock held" >> "$LOG"; exit 1; }
log(){ echo "$(date '+%F %T') $*" >> "$LOG"; }
log "code version: $(cd "$REPO" && git rev-parse HEAD) $(cd "$REPO" && git diff --quiet && echo clean || echo DIRTY)"
cd /home/prashr
( while :; do echo "$(date +%T) $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ';')" >> "$R/gpu_mem.log"; sleep 30; done ) & MEMPID=$!
trap 'kill "$MEMPID" 2>/dev/null' EXIT
run(){ # name arm layers rate epochs gpu
  [ -f "$R/$1/train.json" ] && { log "skip $1 (done)"; return; }
  mkdir -p "$R/$1"
  OMP_NUM_THREADS=2 setsid nohup timeout 8h python3 -u -m mapformer.train_tw_landmark --arm "$2" --n-layers "$3" \
    --name-rate "$4" --seed 150 --epochs "$5" --device "cuda:$6" --output-dir "$R/$1" > "$R/$1.log" 2>&1 < /dev/null &
  log "launched $1 (pid $!) on cuda:$6"; sleep 30
}
RUNS=(MapWM_L3_r0.0_e1200_s150 MapWM_L3_r1.0_e1200_s150 RoPE_L3_r1.0_e1200_s150 RoPE_L3_r1.0_e1800_s150)
run MapWM_L3_r0.0_e1200_s150 MapWM 3 0.0 1200 0
run RoPE_L3_r1.0_e1800_s150 RoPE 3 1.0 1800 1
run MapWM_L3_r1.0_e1200_s150 MapWM 3 1.0 1200 0
run RoPE_L3_r1.0_e1200_s150 RoPE 3 1.0 1200 1
sleep 120
while ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /mapformer[.]train_tw_landmark/ && /budget3/' | grep -q .; do sleep 120; done
for d in "${RUNS[@]}"; do [ -f "$R/$d/train.json" ] || { log "FAILED: $d missing train.json"; exit 1; }; done
P=(); for d in "${RUNS[@]}"; do P+=("$R/$d"); done
CUDA_VISIBLE_DEVICES="" python3 -u -m mapformer.tw_landmark_eval --runs "${P[@]}" --threads 12 --out "$R/eval.json" \
  > "$R/eval_out.txt" 2>&1 || { log "FAILED: eval"; exit 1; }
python3 - "$R" <<'PY' >> "$R/eval_out.txt" 2>&1
import json, sys
E = json.load(open(f"{sys.argv[1]}/eval.json")); g = lambda r, k: r.get(k, float("nan"))
for k, r in E.items():
    print(f"{k}: own {g(r,'acc_own'):.4f} strip {g(r,'acc_strip'):.4f} uninf {g(r,'acc_uninf'):.4f} named {g(r,'acc_named'):.4f} "
          f"own_u05 {g(r,'acc_own_u05'):.4f} | name benefit {g(r,'name_benefit'):+.4f} | rel strip/uninf/named "
          f"{g(r,'rel_strip'):+.3f}/{g(r,'rel_uninf'):+.3f}/{g(r,'rel_named'):+.3f} | conflict {g(r,'conf_path'):.3f}/{g(r,'conf_name'):.3f} "
          f"| {r['cls']} final loss {r['final_loss']:.4f}")
a = E["MapWM_L3_r0.0_e1200_s150"]; A = a["rel_strip"] >= 0.2 and a["acc_strip"] >= 0.90
B = lambda r: r["name_benefit"] >= 0.10 and r["acc_named"] >= 0.85
print(f"(a) MapWM 3L r=0 path-integrates: {'PASS' if A else 'FAIL'}")
for e in (1200, 1800):
    rr = E[f"RoPE_L3_r1.0_e{e}_s150"]
    print(f"(b) RoPE 3L r=1 at {e}: {'PASS' if B(rr) else 'FAIL'}; SOLVED: {rr['cls'] == 'SOLVED'}")
ok1200 = A and B(E["RoPE_L3_r1.0_e1200_s150"]) and all(E[k]["cls"] == "SOLVED" for k in E if "e1200" in k)
ok1800 = A and B(E["RoPE_L3_r1.0_e1800_s150"]) and E["RoPE_L3_r1.0_e1800_s150"]["cls"] == "SOLVED"
print("BUDGET:", "1200" if ok1200 else ("1800 (MapWM arms read at 1200; their 1800 behaviour unmeasured)" if ok1800 else "NEITHER -- redesign"))
PY
touch "$REPO/.tw_landmark_budget_done"; log "DONE"
