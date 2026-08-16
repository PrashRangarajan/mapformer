> **PENDING RE-MEASUREMENT (2026-08-09).** An audit found the "Forward only"
> rows were NOT forward-only: `time_model` ran without `torch.no_grad()`, so each
> measurement also built the autograd graph. That cost scales with NODE COUNT
> rather than FLOPs, penalising the Python-loop models (MapEM-NC, TEMFaithful)
> far more than the cumsum models -- biasing exactly the comparison this
> benchmark exists to make, **in the direction that flatters the parallel-scan
> claim**. Fixed in `benchmark_timing.py`; the forward-only table below is stale
> and awaiting a re-run on a quiet GPU.
>
> The **forward+backward** table is unaffected (backward legitimately needs the
> graph) and remains the headline. Two labelling fixes also applied: the growth
> column now states the length span it actually covers (on an OOM-truncated row
> it was not L=2048/L=128, which understated the sequential penalty), and a
> broken f-string that printed a literal `{notes.get(v,'')}`.

# Wall-clock scaling with sequence length

batch=4, d_model=128, n_layers=2, median of 7 reps after 3 warmups, `torch.cuda.synchronize()` around every timed region.

**These models are NOT parameter-matched, so absolute times are not an architecture comparison.** What is comparable is each model's SCALING with L — the O(log T) parallel-scan claim.

## Forward + backward (training cost)

| variant | params | L=128 | L=256 | L=512 | L=1024 | L=2048 | L=2048/L=128 |
|---|---|---|---|---|---|---|---|
| Vanilla | 405,472 | 4.1 | 4.2 | 4.2 | 4.4 | 12.1 | **2.9x** |
| VanillaEM_P0 | 405,600 | 5.0 | 4.6 | 4.6 | 4.9 | 16.7 | **3.3x** |
| PlainFlat | 405,024 | 4.3 | 4.4 | 4.1 | 4.2 | 11.4 | **2.6x** |
| MapEM_NC_L | 430,016 | 28.7 | 52.8 | 102.2 | 204.2 | 415.6 | **14.5x** |
| TEMFaithful | 20,705 | 164.2 | 470.4 | 1508.2 | 5338.1 | 19743.2 | **120.2x** |

## Forward only

| variant | L=128 | L=256 | L=512 | L=1024 | L=2048 |
|---|---|---|---|---|---|
| Vanilla | 1.5 | 1.5 | 1.5 | 1.5 | 4.9 |
| VanillaEM_P0 | 1.3 | 1.4 | 1.4 | 1.7 | 6.5 |
| PlainFlat | 1.5 | 1.5 | 1.5 | 1.5 | 4.7 |
| MapEM_NC_L | 9.2 | 16.8 | 31.6 | 64.0 | 132.0 |
| TEMFaithful | 47.6 | 109.0 | 269.5 | 752.5 | 2343.9 |

## Findings

**The parallel-scan claim holds, measured for the first time in this repo.**
Across a 16x increase in sequence length (128 -> 2048), forward+backward cost:

| | scaling | L=2048 absolute |
|---|---|---|
| parallel (Vanilla / EM / PlainFlat) | **2.6x - 3.3x** | 11.4 - 16.7 ms |
| MapEM-NC (sequential matrix product) | **14.5x** | 415.6 ms |
| TEMFaithful (sequential RNN + Hopfield) | **120.2x** | 19,743 ms |

MapEM-NC is near-linear in L, exactly as eq. 18 requires once path integration
stops being an exponential-of-a-sum. At L=2048 Vanilla is **34x faster than
MapEM-NC** and **1632x faster than TEMFaithful** -- and TEMFaithful has 20x FEWER
parameters (20,705 vs 405,472), so the gap is not a capacity artefact.

### What the price of non-commutativity actually is

Read against `FAMILY_TREE_RESULTS.md`: on the task built specifically to justify
the non-commutative machinery, MapEM-NC bought **+0.005 (NC-L) to +0.014 (NC-NL)**
over the commutative control. It costs **34x the training time at L=2048**. That
is the trade, quantified on both axes.

### Qualifications, so this is not over-read

- **Parallel models are overhead-dominated below L=1024** (4.1 -> 4.4 ms from 128
  to 1024) and only jump at 2048. The O(L^2) attention cost is barely visible in
  this range; these numbers are about the path-integration term, not attention.
- **TEMFaithful's 120x includes Python-loop overhead.** A tuned sequential
  implementation would be faster in absolute terms. What is architectural is the
  SHAPE -- sequential-in-L versus parallel -- not the constant.
- **Models are not parameter-matched** (20,705 to 430,016). Absolute times across
  rows are not an architecture comparison; the within-row scaling ratio is.
