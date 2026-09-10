# Pre-registration: EM vs WM on the recency (k-back) task

Written before any checkpoint in `runs/recency_em/` exists. Motivated by
`TALE_OF_TWO_ALGORITHMS.md`: reference [11] states that EM networks learn faster
than WM networks **"except on N-back task"**, and our recency task is a k-back
task. This is the one place the source names an exception, and we have the task
built, gated and with an 8-seed published baseline.

## Arms -- 3 x 8 seeds, ONE batch (rule 3)

| arm | what it is | params |
|---|---|---|
| `Vanilla_r4` | WM control, r=4 | 222,233 |
| `VanillaEM_P0_r4` | healthy EM: single `p_0`, r=4 | 222,361 |
| `VanillaEM_r4` | paper-faithful EM: separate `q_0^p`/`k_0^p`, r=4 | 222,489 |

Counts verified at construction. The +128 steps are one `d_model` vector each:
WM has no position origin, single-`p_0` EM has one, paper-faithful EM has two
(`q_0^p` and `k_0^p`). The +128 between the two EM arms IS the ablation, not a
confound to remove -- App. A.4's second vector is the thing under test. `Vanilla_r4` is retrained here rather than
read from `runs/recency/` -- rule 3, and because this batch drops `--fast-attn`.

**No `--fast-attn` on any arm.** The SDPA branch lives in `WMTransformerLayer`
(`model.py:231`), which `MapFormerEM` does not use, so the flag would give the WM
arm a different attention code path from the EM arms while the TF32 half applied
to all three. All arms therefore run the manual fp32 path. Consequence: these
numbers are not bit-comparable to the stored `Signed_r4` rows in
`RECENCY_RESULTS.md`; only the within-batch contrasts are claims.

Config otherwise identical to the published batch: `k_max=64`, `p_filler=0.5`,
`min_gap=64`, train `T=1024`, eval `T` in {1024, 2048}, 300 epochs cosine,
lr 1e-3, 48 batches of 16, 1 layer, d=128, 2 heads. Chance 0.0625, most-recent
shortcut floor 0.0771. Gates already PASS (`RECENCY_GATES_K64.md`).

## The ceiling, and what it forces

`Signed_r4` is **1.000 +/- 0.000 at T=1024** in the published batch. An EM-vs-WM
accuracy contrast there is zero by construction and could not have gone the other
way -- exactly the design error rule 11 was extended to cover. So:

**Primary readout: epochs-to-threshold on the training loss.** This is also the
axis [11] actually claims ("learning dynamics and sample complexity"), not final
accuracy. The per-epoch loss curve is already saved in the checkpoint under
`losses`; no code change. Thresholds 0.5 and 0.1 (published final losses are
0.020-0.110 for path-integrated arms, 2.15 for index arms).

**Secondary: accuracy at T=2048**, the only cell with headroom
(published 0.940 +/- 0.050). At n=8 that sd gives **MDE = 2.8*0.050/sqrt(8) =
0.050**. Anything smaller will be reported as *unmeasured*, not as a null.

**Tertiary: final training loss.**

## Predictions

**P1 -- the paper's own exception.** [11] predicts EM learns faster than WM on
ISR / 1D nav / 2D nav but **not** on N-back. If MapEM inherited RNN-EM's
inductive bias, we would see EM reach threshold no later than WM here only if the
exception fails to transfer. Registered prediction: **`VanillaEM_P0_r4` does NOT
reach threshold in fewer epochs than `Vanilla_r4`** (equal, or slower).
*Falsified if* EM is faster to both thresholds by more than the paired MDE.

**P2 -- our structural argument.** `TALE_OF_TWO_ALGORITHMS.md` argues MapEM has
no separate memory network, so it inherits neither RNN-EM's capacity advantage
nor its binding bias, and should simply match WM. Registered: **|EM_P0_r4 -
Vanilla_r4| at T=2048 is below the 0.050 MDE**, i.e. unmeasured in both
directions. *Falsified if* the contrast clears MDE either way -- and a clear win
in EITHER direction is the interesting outcome, since our four previous
EM-vs-WM cells all landed at +0.000 to +0.004.

**P3 -- the init pathology, fifth replication.** Prior four: paper task +0.089,
compositional +0.167, Match-Query +0.358, MiniGrid collapse (worst-to-second-worst
gap 0.137 -> 0.002 after fixing rank and origin). Registered: **`VanillaEM_P0_r4`
> `VanillaEM_r4`**, at T=2048 or in epochs-to-threshold. *Falsified if*
`VanillaEM_r4` matches or beats the single-`p_0` arm.

Note P3's effect grew with reliance on `A_P` across the previous four tasks. This
task's per-offset curve is FLAT in k for path-integrated arms out to k=64, i.e.
`A_P` carries the whole task, so P3 predicts a LARGE gap here. That is the one
place this batch could produce a surprise even though P1 and P2 both predict
nulls.

## What a null buys

P1 and P2 are both predictions of no difference, so this batch is powered to say
"unmeasured" and little else on those two -- stated up front rather than
discovered afterwards. It is run because (a) P3 is a real directional prediction
with a large predicted effect, (b) the k-back cell is the one the source names,
and it is cheap (~15 min/run, 24 runs), and (c) an EM win on a clock task would
falsify the structural argument in `TALE_OF_TWO_ALGORITHMS.md`, which currently
rests on architecture reading rather than measurement.
