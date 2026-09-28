# Code checkpoints rescored on the full val file -- results (2026-09-27)

Pre-registration `CODE_FULLVAL_PREREG.md`; driver `run_code_fullval.sh`; per-run JSONs
`runs/{code2048,code_decay,code}/FULLVAL_*.json`; full output `CODE_FULLVAL_ANALYSIS.txt`
(`python3 -m mapformer.analyze_code_fullval`). The `.final.pt` checkpoint of all 36 runs, scored on
the WHOLE val file in non-overlapping windows at the training length, replacing `best_val_bpc`
(min over 36 evaluations of 40 windows, ~5.7% of val).

## Verdict: no sign flips. Two claims keep significance; the rest stay UNMEASURED

| contrast | old (`best_val_bpc`) | **full val** | seeds | t-test p | registered |
|---|---|---|---|---|---|
| C1 position main (path integration costs, trained at 2048) | +0.0055 | **+0.0056** | 3/3 | **0.024** | keep |
| C1 encoding main | -0.0030 | -0.0033 | 1/3 | 0.31 | UNMEASURED |
| C1 MapPoPE - PoPE (the "reversal") | +0.0033 | +0.0034 | 3/3 | 0.19 | UNMEASURED |
| C1 MapPoPE - MapWM | -0.0052 | -0.0054 | 0/3 | 0.086 | UNMEASURED |
| C2 PoPE-Decay - RoPE-Decay (encoding costs, with envelope) | +0.0079 | **+0.0054** | 3/3 | **0.002** | keep |
| C2 MapPoPE-Decay - PoPE-Decay | +0.0063 | +0.0061 | 3/3 | 0.067 | UNMEASURED |
| C2 MapWM-Decay - RoPE-Decay ("RoPE-Decay best of eight") | +0.0034 | +0.0034 | 3/3 | 0.20 | UNMEASURED |
| envelope, RoPE (cross-batch) | -0.0130 | -0.0110 | all 3 improve | 0.15 | UNMEASURED |
| envelope, PoPE | -0.0085 | -0.0081 | all 3 improve | 0.052 | UNMEASURED |
| envelope, MapPoPE | -0.0071 | **-0.0059** | all 3 improve | **0.008** | keep |
| envelope, MapWM | -0.0191 | **-0.0170** | all 3 improve | **0.036** | keep |

n=3 throughout; the t-test p is the registered criterion (rule 5: the house |t| > 2.8 verdict is
printed in the analysis file and calls five more of these DETECTABLE at a 10.7% false-positive
rate). What changes: the ledger's B7 "reversal" (MapPoPE - PoPE) and B13 "adding path integration to
PoPE is detectably bad on clock tasks" are UNMEASURED on the better readout too; B8's PoPE-Decay -
RoPE-Decay now passes a t-test (was p 0.052); B9 "RoPE + envelope is the best of eight" stays
unsupported; B10's envelope holds for 2 of 4 arms. Every sign is unchanged, so nothing is withdrawn.

## Unexplained level shift (does not change any contrast's sign)
Full-val bpc sits **+0.048 to +0.051 above** `best_val_bpc` for every 2048-trained run, and **0.024 to
0.035 below** it for every 512-trained run. Min-selection bias alone would put full val ABOVE best
at both lengths, so the 512 shift is not selection; the trainer's 40 random windows are evidently
not a representative sample of the val file at either length (its sampling was not inspected). The
shift is nearly uniform across arms within a batch, which is why the contrasts survive, but its
per-run spread (up to 0.009 at 512) is the size of the contrasts themselves -- the reason this
rescore was needed. Absolute bpc numbers quoted from `best_val_bpc` should not be compared across
batches or with full-val numbers.
