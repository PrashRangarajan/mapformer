# PAIRORIGIN -- does PER-PAIR kernel freedom recover EM's varying-k deficit?

Pre-registered 2026-09-11, before any arm was trained. `EM_WM_THEORY.md` P2 -- the decisive test
of the kernel-sharing claim, and the one arm that has never existed.

## The question

Two accounts survive of why single-`p0` EM trails WM by **-0.375** on recency with `k` varying
per query (0/8 seeds; WM reaches loss < 0.5 on 8/8, EM on 0/8):

- **Kernel sharing.** EM has ONE position kernel for every query-key pair (measured phase spread
  across pairs 0.000, against WM's 1.947), so it cannot shape retrieval per query; its only
  handle is the query token's own step.
- **Per-token search.** The deficit is 64 independent wrapped rewinds, each trained by 1/64 of
  the queries. Supported by SEARCH (every solved cell rewinds via its query token; one shared k
  is found on 7/8 seeds) and SPREAD (at matched budget, fewer offsets is better, +0.422 8/8;
  4x budget takes the full task 0.578 -> 0.928).

`EMPair_r4` (`model_em_pairorigin.py`) separates them. It gives EM per-pair origins --
`q^p_t = p0 + W^q_out W^q_in x_t`, likewise for `k` -- while keeping the Hadamard composition,
rank, depth and task fixed. `W_out` is zero-initialised, so at step 0 it is EXACTLY
`VanillaEM_P0_r4`. It adds 2,048 parameters (+0.9%).

**This is a within-EM test.** It does not turn EM into WM: composition stays multiplicative and
the memory is still attention. It varies sharing, and only sharing.

## Arms and recipe

`EMPair_r4`, `VanillaEM_P0_r4`, `Vanilla_r4` (WM), **8 seeds each, ONE batch**, standard recency:
`k_max 64, min_gap 64, T 1024, 300 ep x 48 x 16, cosine, lr 1e-3, 1 layer, d 128, h 2`, no
`--fast-attn`. Output `runs/pairorigin/`. Chance 0.0625; the most-recent floor is 0.0771.

Primary readout: held-out accuracy at T=1024 (T=2048 secondary), MDE = 2.8 sd / sqrt(8) on
paired differences.

## Manipulation checks (run BEFORE the verdicts are read; a failure voids the arm)

1. **Same function at init.** `EMPair_r4` and `VanillaEM_P0_r4` built with the same seed must
   give identical logits (`ckpt_guard.assert_same_function_at_init`, tolerance 1e-6). If they
   differ, any effect could be an initialisation change (rule 31).
2. **The new pathway moves.** `assert_moved` on `q_origin_out` / `k_origin_out` after training;
   otherwise the arm is a parameter-count control, not a per-pair-kernel arm, and P1/P2 are not
   interpreted.
3. **Per-pair spread actually appears.** `probe_phase_spread` on the trained `EMPair_r4`
   checkpoints must give a phase spread across pairs > 0. Registered as a check, not a result.

## Predictions

- **P1 (kernel sharing).** `EMPair_r4 - VanillaEM_P0_r4` >= **+0.20** and DETECTABLE at n=8
  (MDE will be ~0.15 at the observed sd). CONFIRMED if both hold.
- **P2 (recovery).** `EMPair_r4 - Vanilla_r4` (WM) is within its MDE, i.e. per-pair freedom
  brings EM to WM's level. This is the strong form; P1 can hold while P2 fails, which would say
  sharing is part of the story and not all of it.
- **P3 (per-token search).** If the deficit is only per-token exposure, `EMPair_r4 - P0` is
  BELOW its MDE. P1 and P3 are mutually exclusive by construction; exactly one fires, or the
  contrast lands in between and both are reported as unresolved.
- **P4 (mechanism, not a verdict).** If P1 fires, the solved cells of `EMPair_r4` should show a
  LOWER per-token rewind fraction than `VanillaEM_P0_r4` (`probe_anatomy`): per-pair freedom
  should let it retrieve without moving the query token. If P1 fires with the rewind fraction
  UNCHANGED, the arm helped for some reason other than the one claimed, and that is recorded.

## Reading it against what is already known

EM matches or beats WM at a FIXED offset (fixed k=64: +0.191 at 2x length, 7/8, exploratory),
so P1 firing would say: sharing costs nothing when one kernel suffices, and costs 0.2+ when the
task needs a different offset per query. P3 firing would say the kernel axis is a description of
the function class with no bearing on what training finds -- and the account in
`EM_WM_THEORY.md` Sec 1c stands alone, as its own text already allows.

## Analysis discipline

`stats_guard.paired` with an MDE beside every contrast ("unmeasured", never "null");
`stats_guard.rule9` before any loss-matched reading (r(loss, acc) has run -0.945 to -0.986 on
this task, so accuracy contrasts here are largely fit contrasts and that must be stated);
`ckpt_guard.require_checkpoints`. Every registered verdict reported as met, split or refuted.
