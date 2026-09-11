# "A tale of two algorithms" vs our EM/WM results

> **AUDIT 2026-09-10 -- read `AUDIT_2026-09-10.md` first.** This file predates the recency batch
> and is stale on its headline. N-back HAS been run: EM - WM = -0.375 (0/8). The
> separate-`q0/k0` suspicion of App. A.4 is no longer "refuted four times" -- the
> separate form wins on recency, directionally, and does not replicate at detectable
> size on fresh seeds (+0.073, 9/16). And "WM = additive / sum / OR-like" is wrong:
> MapWM rotates content Q,K, so its position kernel has per-pair phases set by
> content.

Source now in the corpus: `papers/pdf/09_Whittington_2025_Tale_of_Two_Algorithms.pdf`,
extracted to `papers/txt/tale_two_algorithms.txt`. Whittington, Dorrell, Behrens,
Ganguli & El-Gaby, *Neuron* 113(2):321-333, Jan 2025 -- MapFormer's reference **[11]**.
Added by hand (paywalled at Cell Press despite being CC-BY; not on arXiv or bioRxiv,
`fetch.sh` cannot get it).

## The premise needs one correction

The paper does **not** claim EM and WM are equivalent *algorithms*. Verbatim (l. 535):

> "Our theory demonstrates that EM and WM systems implement the same computation
> (solve the same tasks) but use different neural algorithms/mechanisms. **These
> algorithms have tradeoffs.** Most notably, the WM solution requires more RNN
> neurons than the EM solution."

What is equivalent is the **solution**, and the equivalence is stated at exactly the
level our data confirms (l. 610):

> "Although EM and WM networks, once trained, have the **same generalization
> behavior** (due to the formal equivalence of the EM and WM solutions), their
> **learning dynamics and sample complexity differ** ... with EM networks learning
> faster (**except on N-back task**)."

So "why does one work better than the other" is a question the paper asks and answers,
not a tension with it. It gives two answers: a capacity trade-off and a learning-speed
trade-off, in the same direction (EM).

## Why its capacity result does not transfer to MapFormer's MapEM

The paper's EM advantage is a statement about **where the product space is paid for**.
Eq. 1 vs Eq. 2: EM's RNN state is `n_p` (position alone) and its memories sit in a
**separate synaptic memory network** with "additional neurons/parameters ... which we
do account for". WM's RNN state is `n_p * n_o` because it "tracks relative position to
each observation, rather than just position" -- bigger by a factor of `n_o`. At a fixed
*RNN-neuron* budget, EM therefore wins.

**MapFormer's MapEM adds no such memory network.** Attention is the memory, and it is
the same attention MapWM uses. Its only EM-ness is that the score factorises:

    (A_X (*) A_P)_ts = (q^x (x) q^p) . (k^x (x) k^p)

which is a **rank-one contraction** of the `d_x*d_p` product space -- the conjunction is
never materialised as state. So the `n_o` factor that makes RNN-EM cheaper never
appears. MapFormer's own Fig. 2 caption runs the size comparison the *other* way (WM's
conjunction states "smaller by a quadratic factor compared to the EM model"), which is
consistent: in a transformer it is EM that has to build the product, in the query.

Consequence: **MapFormer's "superior capacity of MapEM models in recall tasks"
(Fig. 11 / l. 2317) is a different claim about a different object** from the Neuron
paper's capacity result, and cites it for support it does not provide. It is also the
claim our vocab sweep failed to reproduce (`VOCAB_EM.md`: trimmed, EM - WM = +0.0000 at
n_obs=256; the apparent +0.060 was one collapsed `Vanilla_r4` seed).

A sharper way to say it: **both MapFormers are WM models** in the Neuron paper's sense.
Neither stores anything in weights at test time; neither has a Hopfield module. The
synapse-vs-activity axis the duality is *about* is absent from MapFormer entirely, and
what MapFormer calls EM vs WM is only whether position and content combine
multiplicatively or additively in the attention score.

## What our results say, read against the paper

| paper's claim | our measurement | verdict |
|---|---|---|
| same generalization once trained | MiniGrid allocentric r=4: EM - WM = **+0.0035**; vocab sweep trimmed at n_obs=256: **+0.0000** | **confirmed**, to 4 d.p. |
| EM learns faster (fast-binding inductive bias) | paper-faithful EM was *worse* and collapsed 1 seed in 8, traced entirely to the App. A.4 separate-`q0/k0` init (gap 0.137 -> 0.002 after fixing rank and origin) | **does not transfer** -- and it should not: the inductive bias the paper credits is the Hopfield binding MapEM does not have |
| EM has higher capacity at fixed neurons | `VOCAB_EM.md`, trimmed: no capacity advantage at any `n_obs` | **not reproduced**, for the structural reason above |
| N-back is the exception to EM-learns-faster | never run -- our recency task **is** a k-back task | **open, and cheap** |
| WM's compensating advantage is parallel planning over slots | no planning task survives here (all four were voided, `PLANNER_TASK_AUDIT.md`) | **untested** |
| position decodable from EM, activity slots from WM (Fig. 2G/2I) | not run; both MapFormers carry an explicit `theta`, so the prediction may not even be well posed here | **open** |

## The honest narrative

There is no EM-vs-WM performance story in our data, and the paper says there should not
be one at the level we measured (generalization after training). What there is:

1. an **initialisation pathology** in the paper-faithful EM -- App. A.4's separate
   `k0_p`/`q0_p`, which MapFormer itself flags as a suspicion, now refuted four times
   (+0.089 paper task, +0.167 compositional, +0.358 Match-Query, plus the MiniGrid
   collapse);
2. [AUDIT 2026-09-10: the 'sum / OR-like' reading of WM is WRONG -- MapWM rotates content Q,K] two **failure modes** -- product (AND-gate, no fallback, collapses) vs sum (OR-like,
   degrades gracefully) -- which make one algorithm look different from the other
   precisely when something goes wrong, and every regime we measured was that regime;
3. a **capacity claim that does not survive the RNN -> transformer translation**, because
   the resource it trades on is not present.
