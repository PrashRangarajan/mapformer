#!/usr/bin/env bash
# Pilot for RANK_TESTS_PREREG.md (seeds outside the batch; nothing reused). (a) ring R1 / R2 / R1DF at seed 120, 300 epochs;
# (b) the escape budget rule: S (Vanilla_r4mi from scratch) at seed 100 for 300 epochs -- SOLVED or not decides 300 vs 600;
# (c) end-to-end dry run of the escape --init-from path: E (lifted) and C (rank 2) at seed 0 for 3 epochs (not results);
# (d) timing and GPU memory (sampled every 20 s). Launch:
#   cd /home/prashr/mapformer && setsid nohup bash docs/audits/2026-10-11/rank_tests_pilot.sh > /dev/null 2>&1 &
set -uo pipefail
REPO=/home/prashr/mapformer; R=$REPO/runs/rank_tests_pilot; mkdir -p "$R"; LOG=$R/pilot.log
exec 9>"$REPO/.rank_tests_pilot.lock"; flock -n 9 || { echo "$(date '+%F %T') REFUSED: lock held" >> "$LOG"; exit 1; }
log(){ echo "$(date '+%F %T') $*" >> "$LOG"; }
log "code version: $(cd "$REPO" && git rev-parse HEAD) $(cd "$REPO" && git diff --quiet && echo clean || echo DIRTY)"
( while :; do echo "$(date +%T) $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ';')" >> "$R/gpu_mem.log"; sleep 20; done ) & MEMPID=$!
trap 'kill "$MEMPID" 2>/dev/null' EXIT
cd /home/prashr
L(){ OMP_NUM_THREADS=2 setsid nohup "$@" < /dev/null & }
for A in R1 R2 R1DF; do
  mkdir -p "$R/ring/${A}_s120"
  L python3 -u -m mapformer.train_ring_rank --arm "$A" --seed 120 --device cuda:0 --output-dir "$R/ring/${A}_s120" > "$R/ring/${A}_s120.log" 2>&1
  log "launched ring $A s120 on cuda:0"; sleep 15
done
mkdir -p "$R/S/p0/Vanilla_r4mi_s100"
L python3 -u -m mapformer.train_variant --variant Vanilla_r4mi --seed 100 --epochs 300 --lr 1e-3 --n-batches 98 --batch-size 16 \
  --n-steps 1024 --n-layers 1 --n-heads 2 --d-model 128 --n-landmarks 0 --schedule cosine --data-workers 3 --data-seed-offset 1 \
  --device cuda:1 --output-dir "$R/S/p0/Vanilla_r4mi_s100" > "$R/S_s100.log" 2>&1
log "launched S s100 (300 epochs) on cuda:1"; sleep 15
python3 -m mapformer.build_rank_lift --seeds 0 --out "$R/lifted" >> "$LOG" 2>&1 || log "FAILED lift"
mkdir -p "$R/dry/E" "$R/dry/C"
OMP_NUM_THREADS=2 python3 -u -m mapformer.train_variant --variant Vanilla_r4mi --seed 0 --epochs 3 --lr 1e-3 --n-batches 98 --batch-size 16 \
  --n-steps 1024 --n-layers 1 --n-heads 2 --d-model 128 --n-landmarks 0 --schedule cosine --data-workers 3 --data-seed-offset 1 \
  --device cuda:1 --output-dir "$R/dry/E" --init-from "$R/lifted/p0/Vanilla_r4mi_s0/Vanilla_r4mi.pt" > "$R/dry_E.log" 2>&1 || log "FAILED dry E"
OMP_NUM_THREADS=2 python3 -u -m mapformer.train_variant --variant Vanilla --seed 0 --epochs 3 --lr 1e-3 --n-batches 98 --batch-size 16 \
  --n-steps 1024 --n-layers 1 --n-heads 2 --d-model 128 --n-landmarks 0 --schedule cosine --data-workers 3 --data-seed-offset 1 \
  --device cuda:1 --output-dir "$R/dry/C" --init-from "$REPO/runs/rank_mi/p0/Vanilla_s0/Vanilla.pt" > "$R/dry_C.log" 2>&1 || log "FAILED dry C"
log "dry runs done"
sleep 60
while ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /mapformer[.]train_/ && /rank_tests_pilot/' | grep -q .; do sleep 60; done
python3 - "$R" <<'EOF' > "$R/pilot_out.txt" 2>&1
import sys, json, glob, torch, numpy as np
sys.path.insert(0, "/home/prashr")
from mapformer.stats_core import classify_run
from mapformer.analyze_rank_tests import epochs_to_solved
from mapformer.environment_cancel import GridWorldCancel
from mapformer.train_ring_rank import build
from mapformer.ring_rank_readouts import drift_kappa, head_class
R = sys.argv[1]; env = GridWorldCancel(size=32, seed=0, p_plus=0.5)
for A in ("R1", "R2", "R1DF"):
    b = torch.load(f"{R}/ring/{A}_s120/{A}.pt", map_location="cpu", weights_only=False); ev = json.load(open(f"{R}/ring/{A}_s120/eval.json"))
    m = build(A, env, 32); m.load_state_dict(b["model_state_dict"]); m.eval()
    c = classify_run(b["losses"]); k = drift_kappa(m, env.unified_vocab_size, 16, 0.5)
    print(f"ring {A:5s} s120: {c['registered']} tail {c['tail']:.4f} | acc T=128 {ev['eval']['128']['acc']:.4f} T=512 {ev['eval']['512']['acc']:.4f} "
          f"| epochs to SOLVED {epochs_to_solved(b['losses'], 300)} | kappa {[round(x, 4) for x in k]} {head_class(k)}")
b = torch.load(f"{R}/S/p0/Vanilla_r4mi_s100/Vanilla_r4mi.pt", map_location="cpu", weights_only=False); c = classify_run(b["losses"])
print(f"escape S s100 (rank 4 from scratch, 300 epochs): {c['registered']} tail {c['tail']:.4f} epochs to SOLVED {epochs_to_solved(b['losses'], 300)}"
      f" -> BUDGET RULE: {'300 epochs stand' if c['registered'] == 'SOLVED' else 'budget becomes 600 epochs (amendment before launch)'}")
for arm, v in (("E", "Vanilla_r4mi"), ("C", "Vanilla")):
    b = torch.load(f"{R}/dry/{arm}/{v}.pt", map_location="cpu", weights_only=False)
    print(f"dry run {arm}: 3 epochs, losses {[round(x, 4) for x in b['losses']]} (source rank-2 s0 final loss ~0.59)")
EOF
log "pilot done"; touch "$REPO/.rank_tests_pilot_done"
