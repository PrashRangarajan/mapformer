# SPREAD -- is EM's recency search limited by QUERIES PER TOKEN, or by the number of tokens?

Pre-registered 2026-09-11, before any arm was trained. Follows `SEARCH_RESULTS.md`.

## What this tests

SEARCH found that single-`p0` EM solves a k-back query by giving that query's TOKEN a step that
rewinds the count, wrapped modulo each block's period; solved cells carry one and failed cells
never do. With one shared k, a 63-symbol rewind is found on 7/8 seeds. With 64 distinct k it is
found for about half the tokens. The two conditions differ in two ways at once: the number of
query tokens to serve (1 vs 64), and the number of queries each token gets (all vs 1/64).

This separates them by holding **queries per token** fixed across different **m** (the number of
distinct offsets) using the training budget.

| arm | k drawn from | epochs | queries per token over training |
|---|---|---|---|
| `m4_e300` | {1, 4, 16, 64} | 300 | ~403k |
| `m16_e300` | the 16-value set below | 300 | ~101k |
| `m16_e1200` | the same 16-value set | 1200 | ~403k |
| `m64_e1200` | 1..64 | 1200 | ~101k |
| `m64_e300` | 1..64 | 300 | ~25k -- the STORED `runs/dof/recency` P0 s0-7 |

16-value set: 1, 2, 3, 4, 6, 8, 11, 16, 22, 26, 32, 38, 45, 52, 58, 64 (contains the 4-value
set). ~5,376 scored queries per epoch (48 batches x 16 episodes x ~7), so queries per token is
`epochs * 5376 / m`. **Exposure-matched pairs: (`m16_e1200`, `m4_e300`) and
(`m64_e1200`, `m16_e300`).**

Everything else is the standard recency recipe: `VanillaEM_P0_r4`, k_max 64, min_gap 64, T 1024,
48 x 16 per epoch, cosine schedule over the arm's own budget, lr 1e-3, 1 layer, d 128, h 2,
8 seeds per arm, one batch. `m64_e300` is reused rather than retrained; a `repro` run at the
default settings must be BITWISE identical to the stored s0 first (it was, earlier today, but
`environment_recency` has been edited since).

## Readout, and the floor that forces it

**Primary: mean accuracy at T=1024 over k in {4, 16, 64}**, equally weighted -- the offsets every
arm trains on, minus k=1.

k=1 is excluded because the "ignore k, answer the most recent symbol" shortcut scores
`1/m + chance*(1 - 1/m)`, i.e. **0.297 at m=4, 0.121 at m=16, 0.077 at m=64** (measured in the
gates: 0.2815 / 0.1175 / 0.0574). Comparing whole-set accuracy across m would compare arms with
different floors. Conditioning on k removes it: within k in {4, 16, 64} the floor is chance
(0.0625) in every arm.

Secondary, reported but not registered as decisive: whole-trained-set accuracy (against each
arm's own stated floor), T=2048, per-k accuracy, epochs to loss < 0.5.

Gates: `validate_recency --k-set ...`, min_gap 64, T 1024, **800 episodes** -- the episode count
the committed `RECENCY_GATES_K64.md` used (n = 5683 scored answers per row here against its
5687). Files: `RECENCY_GATES_K4SET.md`, `RECENCY_GATES_K16SET.md`.

| set | marginal | o1 | o2 | o3 | o5 | most-recent (its floor) | oracle |
|---|---|---|---|---|---|---|---|
| m=4 {1,4,16,64} | 0.0667 | 0.0641 | 0.0620 | 0.0638 | 0.0694 | 0.2958 (0.2969) | 1.0000 |
| m=16 | 0.0693 | 0.0620 | 0.0613 | 0.0638 | 0.0613 | 0.1123 (0.1211) | 1.0000 |
| reference, k uniform 1..64 | 0.0663 | 0.0605 | 0.0704 | 0.0665 | 0.0641 | 0.0760 (0.0771) | 1.0000 |

Every column whose floor does not depend on m sits at chance (0.0625). The validator's own
most-recent verdict assumes k uniform on 1..k_max, so it flags both rows; the measured values
are the m-specific floors, which is exactly why the primary readout excludes k=1. (At 200
episodes the n-gram columns flag on the DEFAULT task too -- an estimator artifact at n~1421, not
a property of a k-set; that is why the gates were re-run at 800.) The default stream is
MD5-identical to the pre-edit environment.

## Predictions

- **S4-P1 (exposure).** Accuracy is set by queries per token. Both matched pairs agree within
  their MDE on the primary readout: |`m16_e1200` - `m4_e300`| and |`m64_e1200` - `m16_e300`|
  each below MDE (paired by seed, n=8, MDE = 2.8 sd / sqrt(8)).
  - CONFIRMED if both agree. The search limit is per-token supervision, and the number of
    tokens matters only through it.
  - REFUTED if `m16_e1200` is detectably BELOW `m4_e300`. Then budget does not substitute for
    spread, and the account narrows from total exposure to a WINDOW: a token must find its
    rewind before the kernel sharpens, and a longer budget does not restore what a token
    missed. Recorded as such, not as a null.
  - The reverse (matched arms detectably ABOVE) would mean more tokens help at fixed exposure,
    which nothing predicts.
- **S4-P2 (spread gradient at fixed budget).** `m4_e300` > `m16_e300` > `m64_e300` on the
  primary, with `m4_e300` - `m64_e300` DETECTABLE. If this fails, the whole framing is wrong
  and P1 is not interpreted.
- **S4-P3 (mechanism).** The fraction of trained query tokens carrying a wrapped rewind --
  `max_h sel_h - max_h sel0_h >= 0.5`, or the trough analogue with `selmin`, from
  `probe_anatomy` -- tracks accuracy across all arms and seeds, r >= 0.7, and follows exposure
  the same way the accuracy does. If accuracy moves without the rewind fraction moving, the
  per-token-rewind account of what training finds is incomplete.

## Analysis discipline

`stats_guard.paired` for every contrast, MDE beside each ("unmeasured", never "null");
`stats_guard.rule9` before any loss-matched reading; `ckpt_guard.compare_checkpoints` for the
repro run, and stored `m64_e300` is used only if it is bitwise identical. Registered verdicts
reported as met, split or refuted, including when obvious. `m64_e1200` and `m16_e1200` train
under a cosine schedule stretched to their own budget, so "budget" here means both more steps
and a slower decay; that is stated rather than controlled.
