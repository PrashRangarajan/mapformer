# Does PoPE's pinned phase offset explain the MapPoPE interaction? -- results (registered, `DELTA_FREE_PREREG.md` + Amendment 1)

Batch 2026-10-20 10:26-14:00 on GPU 1 (shared with another user, <= 4 runs), 40 runs, seeds 1600-1607 (fresh). Paper
torus task, T = 128, rank 2, PAPER2X2's recipe (300 epochs, cosine, lr 1e-3, wd 0.05). Arms: PoPE-Flat (index phase,
delta pinned: zero init held at 0 by the clamp's dead zone), PoPE-FreeDelta (delta init U(-2 pi, 0), clamp removed),
MapPoPE-Flat (path phase, pinned), MapPoPE-FreeDelta (path, free), MapPoPE-FrozenDelta (path, delta frozen at the free
arms' draw). At each seed twins differ only in delta. Independent check before launch (`docs/audits/2026-10-20/
delta_free_audit.md`, Amendment 1); pilot on seed 1690 (never pooled). Analysis `analyze_delta_free.py` ->
`DELTA_FREE_ANALYSIS.txt`, `DELTA_FREE_VERDICTS.json`; recipe asserted on all 40 checkpoints. `experiment_audit`: all
arms converged, windowed classes agree. Own-stream retrace floor 0.938.
Account under test (`docs/paper/expansion/MAPPOPE.md`): PoPE's content-free offset delta, pinned at 0, fixes every
kernel's peak at zero phase difference; with a path phase that is the revisit target (helps), with an index phase it is
the query's own token (hurts: no multi-step retrace kernel).

**PRIMARY (registered): PEAK-PINNING SUPPORTED (path unchanged within noise).** Freeing delta raises the index model by
+0.122 [90% CI +0.076, +0.167] (8/8 seeds, sign-flip p 0.0078), concentrated in multi-step reversals (R3+ +0.180,
Holm p 0.031), and leaves the path model at ceiling (-0.018, p 0.31; SOLVED 6/8 vs 8/8, Fisher p 0.47).

| arm | T = 128 | T = 512 | T = 1024 | SOLVED |
|---|---|---|---|---|
| PoPE-Flat (index, pinned) | 0.649 | 0.615 | 0.599 | 0/8 |
| PoPE-FreeDelta (index, free) | **0.771** | 0.536 | 0.442 | 0/8 |
| MapPoPE-Flat (path, pinned) | 0.999 | 0.980 | 0.933 | 8/8 |
| MapPoPE-FreeDelta (path, free) | 0.981 | 0.901 | 0.831 | 6/8 |
| MapPoPE-FrozenDelta (path, random fixed) | 0.999 | 0.929 | 0.832 | 8/8 |

Interaction I = dP - dI = -0.140 [-0.186, -0.092] (0/8 seeds positive, p 0.0078); PAPER2X2's pinned score x phase
interaction was +0.154 (cross-batch, descriptive). The licensed sentence (Amendment 1 MAJOR 5) is the one above, not
"freeing delta reverses the interaction".

## Mechanism readouts (declared secondaries; eval-only probes, rebuilt logits exact)
- **Index kernel:** with delta pinned, 0% of content pairs prefer offset 3 over offset 1 (kernel argmax at offset 1 in
  1074/1088 pairs); freed, 50-100% per seed, with argmax spread over offsets 3, 4, 7, 8 (the retrace comb). Attention on
  the newest observation (never the target) 0.287 -> 0.003.
- **Where the index gain comes from:** R1 +0.039 (n.s.), R2 +0.117, R3+ +0.180 (Holm 0.031 each), unvisited-run +0.014;
  R3+ carries 0.093 of the 0.122.
- **Path target alignment:** C (same-cell key alignment) 0.987 pinned vs 0.616 free and 0.614 frozen; same-cell attention
  mass 0.909 vs 0.809 / 0.876. Accuracy at T = 128 stays at ceiling because the code itself (C0 0.993 / 0.798 / 0.884)
  still carries the map.
- **Delta (S4):** pinned arms 96-100% in the dead zone. Free arms moved mostly by weight decay (x0.447); the learned part
  beyond decay is largest in the fastest frequency quartile (index 0.447, path 0.371 rad).
- **S7 (diagnostic):** resetting delta to 0 on either half of the free index model's channels drops it below the pinned
  model (0.43 / 0.47 vs 0.649): the trained network relies on its learned offsets across the spectrum, so the reset is
  not a clean ablation of one mechanism.

## Beyond the training length (robustness, rule 10)
- **P1 (registered):** path frozen / free minus pinned at T = 1024: -0.101 (p 0.047, Holm 0.094) and -0.102 (Holm 0.195):
  both UNMEASURED by the registered rule; descriptively both lose robustness (T = 512: -0.052, -0.079).
- Index free minus pinned at T = 1024: -0.158 (p 0.016, Holm 0.094): the retrace kernel the free offset buys is
  length-specific, while the pinned recency kernel extrapolates (descriptive).
- Training (S6): the free index arm learns faster (loss AUC -0.131, p 0.0078); the free and frozen path arms slower
  (+0.122, +0.040).

## Reading
PoPE's polar score helps a path-integrated phase and hurts an index phase because its pinned offset fixes the attention
kernel's peak at zero phase difference. Freeing the offset lets the index model build the multi-step retrace kernel RoPE
uses (+0.122 at T = 128, mostly on multi-step reversals) and leaves the path model's accuracy at ceiling while its target
alignment falls (0.987 -> 0.62). The account is causal at T = 128; the costs beyond the training length are descriptive.
Scope: one layer, 2 heads, rank 2, the paper torus at T = 128.
