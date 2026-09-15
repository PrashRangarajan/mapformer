# TEM recency diagnostics

Written 2026-09-14, before training. Follows `TEM_RECENCY_PILOT.md`: the installed rewind scored 1.000,
but TEM from scratch stayed at chance (4/4). These pilots locate where from-scratch learning fails. Same
recipe as the pilot; 2 seeds each; `TEMRecency_Query` (non-committing query) throughout.

| id | arm | change | reading rule |
|---|---|---|---|
| D1 | `TEMRecency_Query --k-fixed 1` | every query asks for the most recent symbol | at chance again: the adaptation does not start learning at all (optimisation, not recency) |
| D2 | `TEMRecency_Query_CounterInstalled` | counter installed and frozen; per-query rewinds learned | learns: the counter was the blocker. Stays near the pilot: the per-query rewind search is |
| D3 | `TEMRecency_Query_Init1` | transitions initialised at scale 1.0, not 0.05 | learns: the near-identity start was the blocker |

These are diagnostics about TEM as adapted here, not comparisons with MapFormer.
