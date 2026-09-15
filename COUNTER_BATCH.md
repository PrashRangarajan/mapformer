# COUNTER batch: with the counter given, how does each model reach the k-th offset?

Written 2026-09-14, before training. Follows `TEM_RECENCY_DIAG.md` D2: with a perfect counter installed, TEM
found rewinds only for k <= 16, even though its per-query transform is a full orthogonal matrix, not a rank-4
bottleneck.

**Question.** When all three models get the IDENTICAL installed symbol counter (fixed frequencies from pi/2 to
2 pi/512; symbols +1, all else 0), does content phase (MapWM) find long offsets where a position-side rewind
(MapEM through its bottleneck, TEM through a full transform) does not?

## Arms: 5 x 4 seeds (0-3), one batch, recency recipe as in MONOTONE

| arm | counter | how the offset can be reached | trainable params |
|---|---|---|---|
| `WM_Counter` | installed, fixed | content phase in the query-key comparison | 221,401 |
| `EM_Counter` | installed + learned rank-4 increment (starts at zero) | query tokens learn a rewind through the bottleneck | 222,297 |
| `TEMRecency_Query_CounterInstalled` | installed, fixed | full orthogonal query transform per token | 376,243 |
| `VanillaEM_P0_r4` | learned from scratch | reference (MONOTONE: 0.641) | 222,361 |
| `Vanilla_r4` (MapWM) | learned from scratch | reference (MONOTONE: 0.994) | 222,233 |

Checked before launch:
- the installed counter equals the symbol count exactly;
- EM_Counter's learned increment is exactly zero at init;
- the TEM and WM/EM counter frequencies are identical.

## Reading rules (n=4: every contrast is exploratory whatever its MDE; no verdict is formally registered)

- **R1.** `WM_Counter` accuracy at k = 33-64 >= 0.9 means content phase finds long offsets, given the counter.
- **R2.** `EM_Counter` and TEM both < 0.5 at k = 33-64 means position-side rewinds fail at long offsets whether
  or not they pass a bottleneck. That bears on the "bottleneck" explanation.
- **R3.** `WM_Counter` - `EM_Counter`, paired by seed at T=1024, is reported with its MDE.
- **R4.** Each counter arm is compared with its from-scratch twin (`WM_Counter` vs `Vanilla_r4`,
  `EM_Counter` vs `VanillaEM_P0_r4`), to show how much learning the counter itself costs.

## Confounds stated up front

- The WM and EM layers differ in more than where the offset lives: WM has one score with content
  magnitudes and phases; EM has a content score times a position score.
- TEM has more parameters.
- The installed frequencies differ from the from-scratch arms' learned ones.
