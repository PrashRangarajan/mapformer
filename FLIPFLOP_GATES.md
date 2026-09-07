# Flip-Flop LM -- pre-flight gates (CPU, no training)

Published task (Liu et al. 2023; definition taken from CoPE sec 5.1, read first-hand). **Chance 0.500** -- binary, so every margin here is read against 0.5.

| scoring | split | marginal | o1 | o2 | o3 | o5 | last-bit | scored/token | dist to write | n |
|---|---|---|---|---|---|---|---|---|---|---|
| all reads | train  p_i=0.80 | 0.513 | 0.730 | 0.730 | 0.730 | 0.730 | 0.597 | 0.0520 | 19.0 (max 172) | 10640 |
| all reads | OOD dense p_i=0.98 | 0.505 | 0.704 | 0.704 | 0.703 | 0.686 | 0.516 | 0.0067 | 148.3 (max 510) | 1367 |
| all reads | OOD sparse p_i=0.10 | 0.500 | 0.752 | 0.752 | 0.752 | 0.752 | 0.951 | 0.2261 | 4.4 (max 34) | 46313 |
| final read only | train  p_i=0.80 | 0.532 | 0.538 | 0.540 | 0.528 | 0.585 | 0.585 | 0.0020 | 19.2 (max 108) | 400 |
| final read only | OOD dense p_i=0.98 | 0.522 | 0.442 | 0.490 | 0.477 | 0.446 | 0.525 | 0.0020 | 181.0 (max 510) | 400 |
| final read only | OOD sparse p_i=0.10 | 0.510 | 0.497 | 0.475 | 0.447 | 0.544 | 0.978 | 0.0020 | 4.3 (max 22) | 400 |

**The order-1 row under `all reads` is expected to fail, and it is the published task's own property**: two consecutive reads with no write between them return the same bit. Scoring the final read only removes it, at the cost of one scored position per sequence. Both readings are reported in the results rather than one being chosen quietly.

`dist to write` is what the OOD splits vary and is the reason the task exists: a fixed offset cannot address the governing write.
