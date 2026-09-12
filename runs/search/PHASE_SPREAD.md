## Per-pair phase spread of the position kernel (n=8 seeds; corrected 2026-09-11)

Circular sd of the kernel phase across query-key pairs, 29056 pairs per model. 0 = one kernel for every pair. **The finite-sample uniform ('no structure') null at this N is 3.267**, so a value near 2.0 is CONCENTRATED, not near-uniform.

| arm | circ sd of phase | amplitude-weighted | spread of per-offset mean phase |
|---|---|---|---|
| WM | 1.947 | 1.816 | 1.656 |
| EM single p0 | 0.000 | 0.000 | 0.000 |  (64 blocks, one phase each; \|phase\| mean 0.000)
| EM separate q0/k0 | 0.000 | 0.000 | 0.000 |  (64 blocks, one phase each; \|phase\| mean 1.603)
| **WM, UNTRAINED (control)** | **2.722** | 2.603 | 2.032 |  (3 inits; the same architecture, random weights)

**What this does and does not show.** EM's phases are pair-independent (measured 0.000, not assumed): they come from q0/k0, and EM's content branch is never rotated, so content can only rescale the shared kernel. WM's are set per pair. **But the untrained control scores HIGHER than the trained model**, so the spread is a property of the parameterisation, not something training discovers or uses. This supports 'WM CAN reshape per pair' and REFUTES any claim that WM's advantage comes from doing so.
