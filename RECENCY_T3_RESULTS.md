# Recency T3: VOID -- the task is at ceiling, so it cannot test the rule

Pre-registration: `RECENCY_T3_PREREG.md`. 3 arms x 8 seeds, `run_recency.sh` recipe verbatim
(k_max 64, T = 1024, 300 epochs).

| arm | T=1024 (trained) | T=2048 | T=4096 (eval-only) | T=8192 (eval-only) |
|---|---|---|---|---|
| MapPoPE-Flat | 0.9997 | 0.9961 | 0.9887 | 0.9839 |
| MapPoPE_T3pi01 (phase) | 1.0000 | 1.0000 | 0.9992 | 0.9920 |
| MapPoPE_T3inert (twin) | 1.0000 | 0.9997 | 0.9974 | 0.9946 |

Every arm is between 0.984 and 1.000 at every length, including 8x the training length. The
registered contrast R1 (phase minus inert twin) is +0.0003 at T=2048, +0.0018 at 4096 and -0.0027 at
8192, all far inside their MDEs -- **as they must be when the baseline has 0.004 of headroom**.

**This is a ceiling, not a null**, and it is the trap this project already named (conditioning on a
ceiling shows ~0 by construction). The registered prediction R1 is therefore neither supported nor
refuted, and R4 -- the cross-task sign contrast that was the actual claim -- cannot be evaluated.
The batch is recorded as VOID for its registered purpose.

**Why the design failed.** I chose `run_recency.sh` verbatim for comparability, without checking that
the arms in THIS batch (all path-integrated, all with PoPE magnitudes) would saturate it. The earlier
recency numbers that motivated the choice, around 0.78-0.98, are `Vanilla`- and `Gated`-family arms;
MapPoPE-class arms had never been run on recency, so there was no basis for expecting headroom. A
pilot seed would have cost four minutes and shown it.

**What a valid version needs**: headroom at the training length. Options, in order of cost --
raise `k_max` well above 64, cut the model (1 head, d=64), or shorten training (fewer epochs) so the
baseline lands near 0.8. Not run here.

**Also recorded**: `run_recency_t3.sh` reported `missing=24` at the end. That was the completeness
check looking for `{variant}.json` while the trainer writes `{variant}_recency.json`; all 24 runs
existed. The check was wrong, not the batch.
