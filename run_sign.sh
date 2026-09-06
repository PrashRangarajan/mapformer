#!/usr/bin/env bash
# THE SIGN ABLATION (axis A5). Pre-registration: SIGN_ABLATION_PREREG.md.
# Do not read the results without reading that file first -- the OOD-only case is
# explicitly NOT the predicted result and has a verdict written for it in advance.
set -u
REPO=/home/prashr/mapformer; R="$REPO/runs/sign"; LOG="$REPO/sign.log"
mkdir -p "$R/p0"; cd /home/prashr
echo "sign ablation start $(date)" > "$LOG"

EP=300; NB=98; BS=128; T=128; DM=128; NH=2; LR=1e-3
ARMS="Signed_r4 Abs_r4 Pos_r4 CARoPE_r4 Vanilla_r4 RoPE"
SEEDS="0 1 2 3 4 5 6 7 8 9 10 11"
MAXPG=4

# Count only REAL interpreters. `pgrep -f` matches the author's own shells as well
# as the script's, which once idled a waiter for two hours against zero jobs.
on_gpu(){ ps -u "$USER" -o comm=,args= \
          | awk -v g="cuda:$1" '$1=="python3" && /mapformer\.train_variant/ && index($0,g)' \
          | wc -l; }

launch(){ V="$1"; SEED="$2"
  OUT="$R/p0/${V}_s${SEED}"; mkdir -p "$OUT"
  [ -f "$OUT/${V}.pt" ] && { echo "skip $V s$SEED" >> "$LOG"; return; }
  GPU=""
  while :; do
    N0=$(on_gpu 0); N1=$(on_gpu 1)
    # balance to the LESS loaded device (rule 13: fill-first idles a whole GPU)
    if [ "$N0" -le "$N1" ] && [ "$N0" -lt "$MAXPG" ]; then GPU=0; break; fi
    if [ "$N1" -lt "$MAXPG" ]; then GPU=1; break; fi
    if [ "$N0" -lt "$MAXPG" ]; then GPU=0; break; fi
    sleep 20
  done
  echo "$(date +%H:%M) $V s$SEED -> cuda:$GPU (load $N0/$N1)" >> "$LOG"
  python3 -u -m mapformer.train_variant --variant "$V" --seed "$SEED" \
    --epochs $EP --n-batches $NB --batch-size $BS --n-steps $T --n-layers 1 \
    --n-heads $NH --d-model $DM --n-landmarks 0 --schedule cosine --lr $LR \
    --device "cuda:$GPU" --output-dir "$OUT" > "$R/${V}_s${SEED}.log" 2>&1 &
  sleep 6
}

# arm varies fastest, so a partial batch still spans every arm
for SEED in $SEEDS; do for V in $ARMS; do launch "$V" "$SEED"; done; done
wait
N=$(find "$R/p0" -name '*.pt' | wc -l)
echo "$(date +%H:%M) $N/72 checkpoints" >> "$LOG"
# `wait` returns regardless of child success -- verify the artifacts exist
[ "$N" -lt 60 ] && echo "TOO FEW CHECKPOINTS -- not aggregating" >> "$LOG"

python3 -u -m mapformer.eval_noise_refine --runs-dir "$R" \
  --variants $ARMS --noises 0.0 --seeds $SEEDS --lengths 128 512 1024 \
  --n-trials 100 --device cuda:0 --out "$REPO/_SIGN_RAW.md" \
  --title "Sign ablation, raw evaluation" >> "$LOG" 2>&1

if [ -s "$REPO/_SIGN_RAW.json" ]; then
  python3 -u -m mapformer.agg_sign --json "$REPO/_SIGN_RAW.json" \
    --runs-dir "$R" --out "$REPO/SIGN_ABLATION.md" >> "$LOG" 2>&1
  python3 -u -m mapformer.probe_sign --runs-dir "$R" --variants $ARMS \
    --seeds 0 1 2 --device cuda:0 --out "$REPO/SIGN_PROBE.md" >> "$LOG" 2>&1
else
  echo "$(date +%H:%M) EVAL FAILED -- no json" >> "$LOG"
fi
touch "$REPO/.sign_done"; echo "$(date) DONE" >> "$LOG"
