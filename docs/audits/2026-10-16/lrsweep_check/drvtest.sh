D=/tmp/claude-1002/-home-prashr-mapformer/224d7c23-b570-4513-8742-cf8a41acc4a3/scratchpad/lrsweep_check
C=$D/dry/mapformer; cd $C; export DRV_DRYRUN=1 DRV_GPUS="0 1"
L=$C/lrsweep.log; R=$C/runs/lrsweep
step(){ echo "=== $1" >> $L; }
# 1 resume: two done, one running
step "T1 resume"
for a in SO AO; do mkdir -p $R/lr1e-03/${a}_s1500; echo '{}' > $R/lr1e-03/${a}_s1500/eval.json; done
setsid nohup python3 -c "import time; time.sleep(75)" --output-dir $R/lr1e-03/SO_s1501 > /dev/null 2>&1 &
sleep 2; bash run_lrsweep.sh; echo "T1 launches: $(sed -n '/=== T1/,$p' $L | grep -c DRYRUN)" >> $L
# 2 dirty tree
step "T2 dirty"; echo "# x" >> train_gauge.py; bash run_lrsweep.sh; git checkout -q train_gauge.py
# 3 lock held
step "T3 lock"; ( flock 9; sleep 20 ) 9>$C/.run_lrsweep.lock & sleep 1; bash run_lrsweep.sh; wait
# 4 code changed since the md5 record
step "T4 md5"; echo "# y" >> train.py; git -c user.name=t -c user.email=t@t commit -qam "test change"; bash run_lrsweep.sh
git reset -q --hard HEAD~1
# 5 extension amendment: RATES += 1e-1 (fresh run dir)
step "T5 extension"; rm -rf $R; sed -i 's/^RATES=(1e-3 3e-3 1e-2 3e-2)/RATES=(1e-3 3e-3 1e-2 3e-2 1e-1)/' run_lrsweep.sh
git -c user.name=t -c user.email=t@t commit -qam "ext"; bash run_lrsweep.sh
echo "T5 launches: $(sed -n '/=== T5/,$p' $L | grep -c DRYRUN); analysis line: $(grep -c 'rates' run_lrsweep.sh)" >> $L
git reset -q --hard HEAD~1
touch $D/drvtest.done
