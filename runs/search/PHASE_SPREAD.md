## Per-pair phase spread of the position kernel (n=8 seeds, held-out episodes)

Circular sd of the kernel phase across query-key pairs. 0 = one kernel for every pair; ~1.97 is R~0.14, i.e. near-uniform on the circle.

| arm | circ sd of phase | amplitude-weighted | spread of per-offset mean phase |
|---|---|---|---|
| WM | 2.003 | 1.838 | 0.918 |
| EM single p0 | 0.000 | 0.000 | 0.000 |  (|phase| mean 0.000, one phase per block, 64 blocks, identical for every pair)
| EM separate q0/k0 | 0.000 | 0.000 | 0.000 |  (|phase| mean 1.603, one phase per block, 64 blocks, identical for every pair)

EM's phases are pair-independent BY CONSTRUCTION (they come from q0/k0, not from the tokens), and its content branch A_X is never rotated, so content can only rescale the shared kernel, not reshape it. WM's are set per pair by content, and vary systematically with the query's offset. This is the measured form of the shared-vs-per-pair contrast.
