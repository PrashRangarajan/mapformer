# P3: does the gate separate actions from observations?

The pre-registered discriminator (`GATED_PREREG.md`). **The floor is 1.35x**, taken from `GATE_PROBE.md`: Selective RoPE's gate measured exactly that on this contrast (0.560 on actions, 0.415 on observations) and was judged NOT to be suppression. Clearing 1.35x is the requirement, not the target.

| arm | gate on ACTIONS | gate on OBS | ratio | per-seed ratios | n |
|---|---|---|---|---|---|
| `Gated_r4` | 0.9873 ± 0.0028 | 0.2443 ± 0.0428 | **4.16x** | 4.73 3.33 3.69 4.05 3.46 3.65 5.44 4.93 | 8 |
| `Gated_r2` | 0.9426 ± 0.0540 | 0.5020 ± 0.2818 | **2.55x** | 1.76 0.96 3.38 2.13 0.92 2.03 5.30 3.89 | 8 |
| `Gated_r4_frozen` | 0.9820 ± 0.0000 | 0.9820 ± 0.0000 | **1.00x** | 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 | 8 |

`Gated_r4_frozen` is the control: its gate cannot learn, so its ratio must be **1.00x** by construction. Any other value there means the probe is wrong, not that the frozen gate separated anything.
