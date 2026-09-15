# SAMEBLOCK amendment 1: the compile check (`verify_addition_compile.py`)

Eager vs `torch.compile` training on identical batches and identical initialisation, 300 steps, role format,
bfloat16 autocast, the Cho recipe's learning rate. Run 2026-09-15 04:15 on an idle GPU.

| arm | final loss eager / compiled | max abs loss diff | median relative diff | s/step eager / compiled |
|---|---|---|---|---|
| ChoPos_signed | 1.3845 / 1.4035 | 0.0250 | 0.0013 | 0.089 / 0.031 (2.89x) |
| ChoPos_abs | 1.9891 / 1.9861 | 0.0139 | 0.0012 | 0.089 / 0.031 (2.88x) |
| ChoPos_rope | 2.0058 / 2.0083 | 0.0180 | 0.0009 | 0.076 / 0.029 (2.65x) |
| ChoPos_coupled | 0.3626 / 0.3626 | 0.0004 | 0.0001 | 0.065 / 0.028 (2.31x) |

**Rule, fixed in amendment 1 before this check:** median relative diff < 0.01, final losses within 5%, and speed-up
>= 1.2x for every arm.

**Result: PASS** for every arm. Seeds 1-2 run with `--compile`, which also switches on the vectorised generator.
Seed 0 used the original code.
