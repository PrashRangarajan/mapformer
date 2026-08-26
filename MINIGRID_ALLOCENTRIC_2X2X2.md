# MiniGrid DoorKey-16 factorial — ALLOCENTRIC action recoding (n=3)

> **RESOLVED 2026-08-26 — the 8th cell was run; see `MINIGRID_ALLOCENTRIC_8CELL.md`.**
> PoPE-Hier = 0.873 / 0.831 / 0.817 (T=128/512/1024). Reproducibility control
> (PoPE-Flat retrained alongside) reproduced its stored values EXACTLY (0.871 /
> 0.828 / 0.807, drift ~0.000), so the cross-batch comparison is licensed.
> **Corrected 8-cell effects:** raw −0.005 / −0.021 vs allocentric **+0.013 /
> +0.020** (T=512 / T=1024) — the FLIP SURVIVES on complete factorials both sides.
> **The ordering claim is FALSIFIED:** at T=1024 PoPE-Hier (0.817) beats the
> weakest path-integrated arm MapWM-Flat (0.809). Use the effect, not the ordering.
>
> (original flag, retained) **INCOMPLETE — 7 of 8 cells (flagged 2026-08-26).** `PoPE-Hier` (PoPE + index +
> hierarchy) was never run in this allocentric rerun, despite the "2X2X2" title.
> This is NOT a random gap: PoPE-Hier is the **best arm in the raw factorial**
> (0.964 / 0.955, `MINIGRID_FULL_2X2X2.md`), so omitting it removes the strongest
> INDEX arm and biases the path-int − index effect upward.
> Impact (T=1024): reported effect +0.026 with n=3 index arms; estimating
> PoPE-Hier as PoPE-Flat + the measured hier bump (0.8069 + 0.0064 = 0.8133) gives
> index mean 0.7972 and effect **~+0.021** — the flip vs raw (−0.021) SURVIVES but
> is inflated ~20%. The stronger claim "all four path-integrated arms outrank all
> index arms" does NOT survive: its margin is +0.0026 (Vanilla 0.8095 vs PoPE-Flat
> 0.8069) and the estimated PoPE-Hier (0.8133) would exceed the lowest path-int arm.
> Fix: retrain all 8 cells in ONE batch (standing rule: never compare a fresh arm
> against stored ones) before citing any ranking claim from this table.



Actions recoded as the realized per-step grid displacement (5 world-fixed classes)
instead of turn/forward. Identical to run_minigrid_2x2x2.sh otherwise.

**Headline:** the position effect (path-integrated - index) FLIPS sign vs the raw
factorial: T=512 -0.005 -> +0.016, T=1024 -0.021 -> +0.024. Under allocentric
recoding all four path-integrated arms outrank the three index arms PRESENT
(min path-int 0.809 > max present index 0.807 at T=1024) -- but that margin is
only +0.003 and the strongest index cell (PoPE-Hier) is MISSING, so this
ordering claim is NOT established (see the incompleteness note above). Absolute scores
are lower than raw (best 0.825 vs 0.953): DoorKey is content-solvable, so recoding
levels the comparison rather than making the task easier (matches the prediction
of parity-to-slight-win, not the +0.488 dominance on the map-requiring rotation env).

(The auto-generated table below carries the eval script's template title; the arm
list is the 7-cell factorial, not "four arms".)

---

# The torus 2x2 on MiniGrid-DoorKey-16x16 (n=3)

External, published benchmark; egocentric observation (the cell in front of the agent). All four arms trained in ONE batch (rule 3), 50 epochs, 25K cached trajectory buffer.

**Measured floor** (most common scored target) per length: T=128: **0.642**, T=512: **0.536**, T=1024: **0.495**. The original n=1 evaluation reported no floor at all.

| model | position | T=128 | T=512 | T=1024 |
|---|---|---|---|---|
| MapWM-Flat (RoPE + path int.) | path-integrated | 0.874 ± 0.008 | 0.833 ± 0.009 | 0.809 ± 0.010 |
| MapWM-Hier (RoPE + path int. + hier) | path-integrated | 0.876 ± 0.009 | 0.835 ± 0.014 | 0.820 ± 0.023 |
| MapPoPE-Flat (PoPE + path int.) | path-integrated | 0.873 ± 0.011 | 0.833 ± 0.013 | 0.818 ± 0.016 |
| MapPoPE-Hier (PoPE + path int. + hier) | path-integrated | 0.877 ± 0.008 | 0.840 ± 0.010 | 0.825 ± 0.013 |
| RoPE-Flat (index) | **index** | 0.873 ± 0.007 | 0.811 ± 0.009 | 0.781 ± 0.024 |
| RoPE-Hier (index + hier) | **index** | 0.874 ± 0.006 | 0.818 ± 0.007 | 0.788 ± 0.023 |
| PoPE-Flat (PoPE + index) | **index** | 0.871 ± 0.006 | 0.828 ± 0.003 | 0.807 ± 0.003 |

| *measured floor* | | *0.642* | *0.536* | *0.495* |

## The torus comparison

For reference, the identical 2x2 on the 64x64 torus paper task at n=8 (`INDEX_BASELINE_PAPER_TASK_n8.md`), floor 0.506:

| | index | path-integrated |
|---|---|---|
| RoPE encoding | 0.530 | **0.967** |
| PoPE encoding | 0.509 | **0.994** |

Both index arms sit ON the floor there. Whether that survives on an egocentric, rotation-actioned environment is what this file measures.
