#!/bin/bash
cd /home/prashr
for spec in "RoPE 1 RoPE-1L" "MapWM 1 MapWM-1L_r2" "PoPE 4 PoPE-4L"; do set -- $spec
  PYTHONPATH=/tmp/claude-1002/-home-prashr-mapformer/11c678ec-9c7c-4954-8b14-36979f03e955/scratchpad/next_exp/pkg OMP_NUM_THREADS=2 python3 -u -m mapformer.train_dyck --arch $1 --n-layers $2 --n-heads 2 --rank 2 --seed 0 --device cuda:0 --output-dir /tmp/claude-1002/-home-prashr-mapformer/11c678ec-9c7c-4954-8b14-36979f03e955/scratchpad/next_exp/gate_repro/$3_s0 > /tmp/claude-1002/-home-prashr-mapformer/11c678ec-9c7c-4954-8b14-36979f03e955/scratchpad/next_exp/gate_repro/$3_s0.log 2>&1
done
echo finished > /tmp/claude-1002/-home-prashr-mapformer/11c678ec-9c7c-4954-8b14-36979f03e955/scratchpad/next_exp/gate_repro/.done
