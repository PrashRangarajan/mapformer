---
name: Run one seed of everything first
description: For multi-variant multi-seed sweeps, complete seed 0 across all variants before starting seed 1, so an initial low-confidence table lands as fast as possible.
type: feedback
originSessionId: be30e775-ba9b-48e1-a763-b2488b550411
---
When designing run scripts for multi-variant × multi-seed sweeps, **outer loop is seed, inner loop is variant** — finish seed 0 across all variants before any variant gets seed 1, and finish seed 1 across all variants before any variant gets seed 2.

**Why:** Three reasons, in order of importance for *bigger* runs:
1. **Early failure detection.** If a variant has a config bug, NaN issue, or eval-script incompatibility, seed-0-first catches it within one run's worth of compute. Variant-first instead burns 3× that compute before the next variant exposes the same problem from a different angle. With ~hour-scale runs this is annoying; with workshop-scale runs (~hours-to-day per seed) it's expensive.
2. **Stoppable mid-pipeline.** A user noticing something off in the seed-0 table can stop / fix / restart without losing many seeds of work. Variant-first means stopping wastes more compute on already-confirmed variants.
3. **Early surprising-result detection.** Single-seed full-coverage table lets us see clear winners, dead ends, or unexpected wins fast, so we know which directions are worth the seed-1, seed-2 investment.

Old structure (`for variant: for seed`) instead delivers high-confidence numbers for one variant at a time, leaving most cells empty until very late. The cost is identical; only the ordering differs.

**How to apply:**
- Order training loops as `for seed in [0,1,2]: for variant in [...]: train(variant, seed)`, NOT the reverse.
- When a pipeline is splitting work across parallel GPU pairs, pair across-variant within a seed (RoPE-s0 with Vanilla-s0, then Level15-s0 with GSF-s0) rather than within-variant across seeds.
- After seed 0 lands across all variants, optionally pause / commit / push the seed-0-only aggregator MD so the table is reviewable even if seeds 1-2 take longer.
- This rule does NOT apply to single-variant deep sweeps (e.g. K=4 vs K=8 vs K=16 of one architecture) — there's no other-variant column to fill first.
