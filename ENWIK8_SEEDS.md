# enwik8 — does the PoPE x path-integration combination beat RoPE? (n=3)

36k iters, seq 512, deterministic val (fixed generator -- identical batches for every arm and checkpoint), r=4 for the path-integrating arm, param-matched to 0.03%. Values are the mean of the last 5 checkpoints. **Lower is better.**

| seed | RoPE | MapPoPE-Flat r4 | delta |
|---|---|---|---|
| 0 | 1.3864 | 1.3786 | **-0.0078** |
| 1 | 1.3837 | 1.3795 | **-0.0043** |
| 2 | 1.3840 | 1.3804 | **-0.0036** |

- mean **-0.0052**, sd 0.0023
- sign-consistent: **YES** (3/3 favour the combination)

> All seeds favour the combination. At this scale (295M tokens vs the
> paper's 100B) the effect is small but consistent -- the two orthogonal
> mechanisms compose on language, as they do on navigation.
