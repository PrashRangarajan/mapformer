#!/bin/bash
cd /tmp/claude-1002/-home-prashr-mapformer/11c678ec-9c7c-4954-8b14-36979f03e955/scratchpad/next_exp/pkg
export OMP_NUM_THREADS=2
python3 -c "import mapformer, sys; print(mapformer.__file__)" > /tmp/claude-1002/-home-prashr-mapformer/11c678ec-9c7c-4954-8b14-36979f03e955/scratchpad/next_exp/gate_patched2/which.txt
run(){ out=$1; shift; python3 -u -m mapformer.train_dyck "$@" --n-heads 2 --rank 2 --seed 0 --device cuda:1 --output-dir /tmp/claude-1002/-home-prashr-mapformer/11c678ec-9c7c-4954-8b14-36979f03e955/scratchpad/next_exp/gate_patched2/$out > /tmp/claude-1002/-home-prashr-mapformer/11c678ec-9c7c-4954-8b14-36979f03e955/scratchpad/next_exp/gate_patched2/$out.log 2>&1; }
run RoPE-1L_s0 --arch RoPE --n-layers 1
run PoPE-4L_s0 --arch PoPE --n-layers 4
run RoPE-1L_tL32D12_s0 --arch RoPE --n-layers 1 --train-D 12
run MapWM-4L_r2_tL32D12_s0 --arch MapWM --n-layers 4 --train-D 12
run RoPE-4L_b32_mix_s0 --arch RoPE --n-layers 4 --rope-base 32 --train-D-set 4,5,6,7,8,9,10,11,12
run RoPE-4L_tL32D12_x3_s0 --arch RoPE --n-layers 4 --train-D 12 --n-sequences 1680000
echo finished > /tmp/claude-1002/-home-prashr-mapformer/11c678ec-9c7c-4954-8b14-36979f03e955/scratchpad/next_exp/gate_patched2/.done
