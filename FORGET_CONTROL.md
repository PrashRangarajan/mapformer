# Is the forget gate's +0.086 the gate, or 259 parameters?

Torus paper task, held-out map, evaluated under the same noise it trained on.

## T=128

| p_action_noise | Vanilla | Forget | Forget_Frozen |
|---|---|---|---|
| 0 | 0.993 ± 0.017 | 0.991 ± 0.021 | 0.995 ± 0.010 |

## T=512

| p_action_noise | Vanilla | Forget | Forget_Frozen |
|---|---|---|---|
| 0 | 0.944 ± 0.028 | 0.967 ± 0.038 | 0.942 ± 0.036 |

## T=1024

| p_action_noise | Vanilla | Forget | Forget_Frozen |
|---|---|---|---|
| 0 | 0.834 ± 0.064 | 0.914 ± 0.041 | 0.818 ± 0.099 |


## Verdict: the gain needs a LIVE lambda, not the parameters

Paired, T=1024, n=8, all three arms in one batch.

| contrast | delta | sd | MDE | seeds + | verdict |
|---|---|---|---|---|---|
| `Forget` - `Vanilla` | +0.081 | 0.080 | 0.080 | 7/8 | DETECTABLE |
| `Forget_Frozen` - `Vanilla` | **-0.016** | 0.119 | 0.118 | 2/8 | unmeasured |
| `Forget` - `Forget_Frozen` | +0.097 | 0.106 | 0.105 | 7/8 | unmeasured |

**The pre-registered second branch fired.** `Forget_Frozen` -- gate parameters
present, initialisation matched to `Forget` at 0.0e+00, decay bias identically
zero -- lands on `Vanilla` (-0.016) and not on `Forget` (+0.097). So the +0.081 is
**not** parameter count and **not** initialisation shift. It needs lambda to be
trainable.

The direct `Forget` - `Forget_Frozen` contrast is 7/8 positive but formally
unmeasured, because `Forget_Frozen` is the highest-variance arm in the batch
(sd 0.099). The two flanking contrasts against `Vanilla` are what carry the
conclusion.

### What that leaves, and it is odd

A trainable lambda that **ends near zero** is worth +0.081, while the same
architecture with lambda **pinned at zero** is worth nothing. And FORGET_GATE.md
shows the gain is anti-correlated with the final lambda (r = -0.516), with five
of eight seeds ending slightly negative.

**Hypothesis, fitted after the fact and marked as such:** the gate is a
*transient training aid* rather than an inference mechanism. lambda is nonzero
*during* optimisation even though it relaxes toward zero -- the 60-step trace
taken before launch shows it rising 0 -> +0.034, against final values averaging
+0.012 with most seeds negative, i.e. a rise and then an anneal. An early recency
prior would make the first retrievals it has to learn the short-lag ones, which is
a curriculum; once the map is built the prior is no longer wanted and decays away.

**The test this needs is to log lambda over training**, which no checkpoint here
stores. A non-monotone trajectory -- rise then fall -- would support it; monotone
drift to its final value would not. Cheap to add and not yet run, so this is a
hypothesis and not a finding.

## Incidental: the torus IS bit-reproducible across batches

`Vanilla` and `Forget` were retrained here from scratch and match the earlier
batch to **0.0000 maximum per-seed drift** at T=1024. So the cross-batch
non-reproducibility recorded for Match-Query (MQ_RANK_2X2 vs LOOP_HEADROOM) is
**not** a general property of this codebase. The torus runs here do not pass
`--fast-attn`; the Match-Query batches do, which makes SDPA the leading suspect
and isolates it to a knob rather than the pipeline.
