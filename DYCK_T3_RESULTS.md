# Dyck-2 replication -- `/home/prashr/mapformer/runs/dyck_t3`

Pre-registration: `DYCK_PREREG.md`. F1 = mean per-prefix valid-continuation F1 (Goodale et al.), 1024 sequences per cell. Floor = best n-gram (orders 1-6) from `DYCK_GATES.md`.

## Seeds, convergence (rule 10)

| arm | params | seeds | final CE (floor 0.800) | slope /1k steps, median seed | max slope |
|---|---|---|---|---|---|

## Headline cells (mean +/- sd over seeds; paper value in brackets)

| arm | L32_D4 | L128_D4 | L32_D12 | L128_D12 |
|---|---|---|---|---|
| n-gram floor | 0.904 | 0.896 | 0.903 | 0.884 |
| CE-optimal predictor | 0.911 | 0.933 | 0.753 | 0.944 |

## Full grids (ours / paper, rows D, columns L)

## Registered verdicts


**R6 contrasts at L128 D12 (paired by seed)**

| contrast | paper | delta | sd | MDE | seeds + | verdict |
|---|---|---|---|---|---|---|

**R7 floor reading at L128 D12** (best n-gram 0.884)


## Rule 9: r(final loss, F1) across all runs

- L32_D4: r = +nan
- L128_D4: r = +nan
- L32_D12: r = +nan
- L128_D12: r = +nan

## Mechanism (Fig 3f): cosines of W_in on bracket embeddings

| arm | cos('(' , ')') | cos('[' , ']') | abs cos('(' , '[') | mean norm brackets | norm BOS |
|---|---|---|---|---|---|
