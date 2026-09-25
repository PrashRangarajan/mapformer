# Dyck matched-depth gates (validate_dyck_mdepth.py)

**PASS: True**  (G1 validity, G3 sampler A2f = 1, best A2f floor at L32 D12 < 0.95)

## Training distributions (20,000 sequences each, the trainer's per-batch D rule)

| train | G1 | CE floor | closers at depth >4 | >8 | max depth | closing dist median / p90 / max |
|---|---|---|---|---|---|---|
| T4 (ladder) | True | 0.799 | 0.000 | 0.000 | 4 | 1 / 9 / 31 |
| T12 | True | 0.520 | 0.538 | 0.255 | 12 | 9 / 23 / 31 |
| Tmix 4-12 | True | 0.690 | 0.313 | 0.065 | 12 | 3 / 19 / 31 |

## Evaluation cells (the ladder's exact 512 sequences per cell)

| cell | CE floor | closers at depth >4 | >8 | closing dist median / p90 / max | forced-open share of scored | G3 sampler A2f / A2 | stack-free A2f / A2 |
|---|---|---|---|---|---|---|---|
| L32D4 | 0.802 | 0.000 | 0.000 | 1 / 9 / 31 | 0.013 | 1.000 / 0.996 | 0.535 / 0.535 |
| L32D12 | 0.520 | 0.537 | 0.254 | 9 / 23 / 31 | 0.322 | 1.000 / 0.940 | 0.543 / 0.543 |
| L128D12 | 0.904 | 0.543 | 0.234 | 1 / 27 / 127 | 0.032 | 1.000 / 0.994 | 0.515 / 0.515 |

## Floors: n-grams fitted on each training distribution, A2f / A2 (chance 0.500)

| train | order | L32D4 | L32D12 | L128D12 |
|---|---|---|---|---|
| T4 (ladder) | 1 | 0.531 / 0.531 | 0.531 / 0.531 | 0.508 / 0.508 |
| T4 (ladder) | 2 | 0.531 / 0.531 | 0.531 / 0.531 | 0.508 / 0.508 |
| T4 (ladder) | 3 | 0.563 / 0.563 | 0.562 / 0.562 | 0.515 / 0.515 |
| T4 (ladder) | 4 | 0.563 / 0.563 | 0.562 / 0.562 | 0.516 / 0.516 |
| T4 (ladder) | 5 | 0.595 / 0.595 | 0.588 / 0.582 | 0.523 / 0.523 |
| T4 (ladder) | 6 | 0.595 / 0.595 | 0.572 / 0.567 | 0.522 / 0.521 |
| T12 | 1 | 0.531 / 0.531 | 0.531 / 0.531 | 0.508 / 0.508 |
| T12 | 2 | 0.531 / 0.531 | 0.531 / 0.531 | 0.508 / 0.508 |
| T12 | 3 | 0.563 / 0.563 | 0.562 / 0.562 | 0.516 / 0.516 |
| T12 | 4 | 0.566 / 0.566 | 0.563 / 0.563 | 0.515 / 0.515 |
| T12 | 5 | 0.599 / 0.599 | 0.594 / 0.594 | 0.527 / 0.527 |
| T12 | 6 | 0.606 / 0.605 | 0.593 / 0.594 | 0.522 / 0.522 |
| Tmix 4-12 | 1 | 0.531 / 0.531 | 0.531 / 0.531 | 0.508 / 0.508 |
| Tmix 4-12 | 2 | 0.531 / 0.531 | 0.531 / 0.531 | 0.508 / 0.508 |
| Tmix 4-12 | 3 | 0.562 / 0.562 | 0.562 / 0.562 | 0.516 / 0.516 |
| Tmix 4-12 | 4 | 0.563 / 0.563 | 0.562 / 0.562 | 0.516 / 0.516 |
| Tmix 4-12 | 5 | 0.595 / 0.595 | 0.594 / 0.594 | 0.524 / 0.524 |
| Tmix 4-12 | 6 | 0.594 / 0.594 | 0.594 / 0.594 | 0.523 / 0.523 |

Best A2f floor (max over n-gram orders and the stack-free heuristic), to quote beside every cell:

- T4 (ladder): L32D4 0.595, L32D12 0.588, L128D12 0.523
- T12: L32D4 0.606, L32D12 0.594, L128D12 0.527
- Tmix 4-12: L32D4 0.595, L32D12 0.594, L128D12 0.524
