## T2 -- does length act only through kernel collisions? (VanillaEM_P0, paper task)

`N_coll` = prior keys at a different cell that the model's own position kernel ranks at least as high as the correct key. No free parameters.

| length | accuracy | mean collisions | median | prior keys |
|---|---|---|---|---|
| 256 | 0.974 | 1.2 | 0 | 135 |
| 512 | 0.974 | 4.2 | 0 | 265 |
| 1024 | 0.954 | 9.6 | 0 | 524 |
| 2048 | 0.933 | 41.4 | 0 | 1076 |

### The test: accuracy against collisions, per length

If length acts ONLY through collisions, every length falls on one curve.

| collisions | L=256 | L=512 | L=1024 | L=2048 |
|---|---|---|---|---|
| 0 | 0.983 (n=1933) | 0.989 (n=4299) | 0.981 (n=8207) | 0.964 (n=16684) |
| 1-2 | 0.917 (n=24) | 0.769 (n=78) | 0.704 (n=223) | 0.749 (n=394) |
| 3-8 | 0.933 (n=30) | 0.838 (n=68) | 0.789 (n=190) | 0.757 (n=378) |
| 9-32 | 0.769 (n=39) | 0.867 (n=83) | 0.809 (n=283) | 0.776 (n=447) |
| 33-128 | 0.704 (n=27) | 0.832 (n=125) | 0.802 (n=308) | 0.787 (n=687) |
| 129+ | - | 0.727 (n=44) | 0.766 (n=248) | 0.803 (n=1611) |
