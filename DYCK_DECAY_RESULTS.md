# Dyck decay: E1 REFUTED, and the refutation gives the better theory

Pre-registration: `DYCK_DECAY_PREREG.md`. I predicted a decay envelope would destroy long-range stack
tracking. It does -- on the INDEX row only. On the path-integrated row it IMPROVES it.

## Closer accuracy by distance to the bracket that must be closed (L=128, D=12, 8 seeds, chance 0.5)

| arm | d 1-2 | d 3-8 | d 9-32 | d 33+ | F1 | strict-F1 |
|---|---|---|---|---|---|---|
| MapPoPE | 0.997 | 0.996 | 0.971 | 0.730 | 0.927 | 0.871 |
| **MapPoPE + decay** | 0.999 | 0.994 | 0.968 | **0.778** | **0.956** | **0.926** |
| PoPE | 0.961 | 0.740 | 0.645 | 0.569 | 0.615 | 0.414 |
| **PoPE + decay** | 0.987 | 0.700 | **0.499** | **0.511** | 0.879 | 0.797 |
| *n-gram (no stack)* | *0.904* | *0.498* | *0.508* | *0.511* | *0.857* | *0.701* |

- **E1 REFUTED**: MapPoPE + decay is $+0.048$ at distance 33+, not below -- the sign is wrong for the
  prediction, which is what the test was for. Its F1 advantage over MapPoPE is **+0.029 with MDE
  0.034 (7/8), i.e. UNMEASURED**: "best Dyck arm" is directional, not established. The refutation
  does not depend on it.
- **E3 fires hard, on the row it was a secondary check for**: PoPE + decay is driven to EXACTLY the
  no-stack n-gram level beyond distance 8 (0.499 and 0.511 against the n-gram's 0.508 and 0.511). Its
  F1 rises to 0.879 all the same, because on this metric being reliably local beats being globally
  wrong -- a clean illustration of why the F1 needed the floor in the first place.
- **E4 FAILS on the index row** (corrected 2026-09-19 after audit). PoPE + decay at the TRAINING
  cell is 0.861 against PoPE's 0.911: **-0.0496, MDE 0.0024, 0/8 seeds** -- detectable damage where
  E4 registered that a loss "would mean the envelope is damaging the model rather than its reach".
  The probe shows why: at L32 D4 its closer accuracy at d 9-32 is already at chance. So on the index
  row the envelope does not merely trade reach for reliability, it removes long-range retrieval
  inside the training distribution as well. E4 holds on the path-integrated row (+0.0059).
- **E2 (graded damage), reported**: on the index row the loss IS graded in distance -- +0.026 at
  d 1-2, -0.040 at 3-8, **-0.146 at 9-32**, -0.058 at 33+ -- consistent with suppression of far
  pairs, but the d 9-32 collapse to chance happens at the training length too, which E4 catches.

## Why the same envelope helps one row and guts the other

The envelope decays in whatever metric the position variable defines. Measured on the trained MapPoPE + decay models by `probe_dyck_metric.py` (8 seeds; the first
version of these numbers, 0.238 / 0.862, came from an inline script over 4 seeds and one head), the
distance it actually uses, $|S_t - S_s|$, correlates with

| against | r |
|---|---|
| token distance $|t-s|$ | **0.278 +/- 0.109** |
| stack-depth difference | **0.755 +/- 0.251** |

**This is consistent with, not a measurement of, the cause.** The two arms differ in the position
mechanism AND in the decay metric at once (`model_pope_decay.py` hard-codes `|t-s|` for the index
class and `|S_t-S_s|` for the path class), so the metric is not isolated; and on Dyck the accumulator
cancels by construction, which makes the correlation close to definitional. The crossed arms -- a
path-integrated model decayed over `|t-s|`, an index model decayed over a state distance -- are the
test, and are NOT run. With that caveat: on Dyck the path-integrated accumulator encodes DEPTH, and a decay envelope over it says "prefer
keys at a similar depth" -- which is exactly the right prior for bracket matching, since the matching
open bracket sits at the same depth however far back it is. The index version's distance is literal
token distance, so the same envelope says "prefer recent keys", which is exactly wrong: it deletes the
long-range retrieval the task is made of.

## The theory this leaves

A decay envelope is not a locality prior. **It is a proximity prior in the space the position variable
defines**, and its value depends entirely on what that space means:

- index position: proximity = recency. Good when dependencies are local (Bach at 2-4x: PoPE 1.597 ->
  0.626), fatal when they are not (Dyck beyond distance 8: chance).
- path integration: proximity = nearness in the learned state. On Dyck that is stack depth and the
  prior is correct; on Bach the accumulator is a clock, so it degenerates to recency and behaves like
  the index version (0.622 vs 0.626, indistinguishable).

This subsumes the previous reading rather than contradicting it. "Decay and path integration do
different jobs" was too crude: they interact, because path integration determines the metric in which
decay operates. It also revises the Bach conclusion -- there the two repairs coincided not because
decay is universally sufficient, but because a clock accumulator makes state-proximity and recency the
same thing.

**What I got wrong, and why it was worth running**: I expected the envelope to suppress exactly the
far pairs Dyck needs. It does when distance is measured in tokens; when it is measured in the model's
own state, "far" means something else entirely. The prediction failed because I reasoned about the
envelope without asking what metric it was applied in -- the same mistake, in a new place, as reading
the omega base through RoPE intuitions.

## Audit corrections (2026-09-19)

An adversarial review of this line found three defects here, all now in the text above: E4 was
reported as holding when the index row fails it detectably; the "best Dyck arm" claim is inside its
MDE; and the metric account was stated as measured when the crossed arm that would isolate it has not
been run. The correlation probe is committed as `probe_dyck_metric.py`.
