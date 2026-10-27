# Does a recurrent network learn the long-walk torus map? LSTM / GRU baselines -- results (RNN_TORUS, registered, `RNN_TORUS_PREREG.md` + Amendment 1)

Batch 2026-10-26 13:30 -> 2026-10-27 09:53 (20 h 23 min, including evaluation) on **kalman** GPUs 0-1 (2 jobs per GPU),
code `782ba4f` (contains the batch commit `af6f7da`; clean; md5 guard over 107 files, `runs/rnn_torus/code_md5.txt`,
re-checked before every launch, before evaluation and before the analysis; `md5sum -c` clean afterwards, 107/107 OK),
torch 2.10.0+cu128, fp32 throughout (TF32 off). 64 launches for 64 runs, no failure or relaunch in `rnn_torus.log`; marker
`.rnn_torus_done`. Fresh seeds 3300-3307 (n = 8 per cell). Independent results-blind code check before launch
(`docs/audits/2026-10-26/rnn_torus_audit.md`: no BLOCKER, 3 MAJOR, 5 MINOR, folded into Amendment 1). Learning-rate pilot on
seed 3390, outside the batch (`runs/rnn_torus_pilot/`; never pooled). Registered output `RNN_TORUS_ANALYSIS.txt`,
`RNN_TORUS_VERDICTS.json`; evaluations `RNN_TORUS_{anchor,lr1e-03,lr3e-03}.json` (T = 512 / 1024 / 2048, 100 walks),
`..._STRATA.json`, own-map `..._OWN_s*.json`, context destruction `RNN_TORUS_CD.json`; run dirs `runs/rnn_torus/`.

**Checks done for this file (2026-10-27).** The analysis was re-run read-only: text identical, verdicts JSON unchanged
(md5). `experiment_audit` on the three roots (check [1] only, exit 2 on this layout as registered): windowed classes equal
the analysis's S7 in every cell. Every number below was re-derived independently from the evaluation, strata, own-map and
context-destruction JSONs and the checkpoints' loss curves (`docs/audits/2026-10-27/rnn_torus/verify_numbers.py` ->
`verify_numbers_out.txt`): all match the analysis to the printed precision, including every exact permutation p
(0.00016 = 2/12,870) and Fisher p.

**Task and recipe** = FACTORIAL / FACTORIAL_N8: 64 x 64 torus, 16 object types, p_empty 0.5, T = 1024 moves, directed
walks of 1-10 repeated moves, interleaved [a1, o1, a2, o2, ...] (vocabulary 21), loss on revisited observation positions
only; 900 epochs x 98 batches x 16 walks, AdamW wd 0.05, 5% warm-up + cosine to 0.1, grad clip 1.0, batch recipe asserted
per run against FACTORIAL_N8's path run. The RNNs run through `train_rnn` (registers them into `train_variant`; same data,
loss, optimiser, schedule and checkpoint code path). Anchors at lr 1e-3; RNN cells at the pilot-picked rates (below).

**Protocol fact that matters here: one training map per run, a new map at evaluation.** `train_variant` builds
`GridWorld(size 64, 16 types, p_empty 0.5, seed = --seed)`; the constructor draws the object map once from
`np.random.RandomState(seed)` (`environment.py`, `GridWorld.__init__`, `obs_map`), and every training batch is a fresh set
of walks on that one map (`ParallelBatchGenerator` workers receive the env object; only the walk RNG is reseeded per batch).
So each run trains on exactly one 64 x 64 map, its seed's, over 900 x 98 x 16 = 1.41M walks; at a seed every arm sees the
same map and the same walks. Evaluation rebuilds the environment with env seed 10000, a map no run trained on, and walks
from numpy seed 1234 + run seed (`eval_noise_refine.evaluate`). The "own map" secondary (S5) rebuilds it with env seed = run
seed (the training map) and the same evaluation walk seeds (fresh walks on the training map). This is MapFormer's protocol
and the anchors'; every held-out number is a transfer measurement.

## Reading (registered)

**PRIMARY: PATH WINS.** One path-integrated attention layer (MapWM, shared rank 4, 204,757 parameters) scores **0.9998**
on the held-out map (8/8 SOLVED) and beats every one of the six LSTM / GRU cells on both registered routes: accuracy
d **+0.350 to +0.428** (exact permutation p 0.00016 each) and SOLVED 8/8 vs 0/8 (Fisher p 0.00016 each). The selected best
RNN cell (training loss only) is the 4x LSTM at 3e-3: 0.5721, d +0.4277. Binding cell `LSTM_L1_h313@0.001`, separated on
both routes. Registered label, verbatim: "PATH WINS: path beats every RNN cell (6/6); selected best LSTM_L1_h313@0.003: acc
0.5721 vs path 0.9998 (d +0.4277, perm p 0.00016), SOLVED 0/8 vs 8/8 (Fisher p 0.00016); binding cell LSTM_L1_h313@0.001
(separated on both) [within 900 epochs; DESCENDING 5/8 in LSTM_L1_h313@0.003]". No route qualifier (Amendment 1 A2): every
RNN cell is separated on accuracy by far more than 0.05, not on SOLVED alone.

**MATCHED-PARAMETER READING (registered companion, the four ~205k cells): PATH WINS (4/4).** Selected best
`LSTM_L1_h157@0.003`: 0.6072 vs 0.9998 (d +0.3926, perm p 0.00016), SOLVED 0/8 vs 8/8 (Fisher p 0.00016); 0/8 descending.

| cell | layers x hidden | params | lr (pilot pick) | acc T = 1024, held-out map (sd) | own (training) map | own - held-out | SOLVED | classes (windowed) | final-5% loss |
|---|---|---|---|---|---|---|---|---|---|
| **Path_L1** (MapWM r4, anchor) | 1 x d 128 | 204,757 | 1e-3 | **0.9998** (0.0005) | 0.9998 | +0.000 | **8/8** | 8 SOLVED | 0.006 |
| RoPE_L1 (index, anchor) | 1 x d 128 | 203,925 | 1e-3 | 0.7243 (0.0172) | 0.7839 | +0.060 | 0/8 | 8 STALLED | 0.972 |
| LSTM_L2_h111 | 2 x 111 | 203,817 | 3e-3 | 0.6494 (0.0021) | 0.9358 | +0.286 | 0/8 | 8 STALLED | 0.268 |
| GRU_L2_h128 | 2 x 128 | 203,797 | 3e-3 | 0.6260 (0.0037) | 0.9333 | +0.307 | 0/8 | 8 STALLED | 0.278 |
| LSTM_L1_h157 | 1 x 157 | 205,377 | 3e-3 | 0.6072 (0.0045) | 0.9325 | +0.325 | 0/8 | 8 STALLED | 0.253 |
| GRU_L1_h181 | 1 x 181 | 205,637 | 1e-3 | 0.5819 (0.0061) | 0.9308 | +0.349 | 0/8 | 8 STALLED | 0.273 |
| LSTM_L1_h313 (4x) | 1 x 313 | 800,049 | 1e-3 | 0.5742 (0.0047) | 0.9742 | +0.400 | 0/8 | 8 STALLED | 0.099 |
| LSTM_L1_h313 (4x; Amendment 1 A3) | 1 x 313 | 800,049 | 3e-3 | 0.5721 (0.0051) | 0.9767 | +0.405 | 0/8 | 3 STALLED, 5 DESCENDING | 0.089 |

Own - held-out is positive on 8/8 seeds in every RNN cell (range per seed 0.284-0.411; paired sign-flip p 0.0078 per
cell) and in RoPE_L1 (0.045-0.076); path's is 0.0000-0.0003.

**Floors on the evaluation walks** (measured, model-free; analysis and pre-registration): blank 0.508, retrace 0.844
(short gap 0.910, long gap 0.498, wrap-only 0.515), wrap-only share 0.080; best action-stream n-gram 0.576 (quoted, RANK_MI).
Two independent maps agree on ~0.27 of cells (the context-destruction level).

## The central declared secondary: the RNNs learn their training map and do not transfer (S5, with S1-S3)

Every RNN cell fits its own training map (0.931-0.977 on fresh walks over it, far above retrace) and falls to 0.572-0.649
on a new map, **below index RoPE at one layer (S1: -0.075 to -0.152, perm p 0.00016 each; label RNN BELOW INDEX 1L for all
six) and below the map-free retrace floor (S2: -0.195 to -0.272, 0/8 seeds above in every cell)**. The gap own minus
held-out is +0.29 to +0.40 for the RNNs, +0.06 for index RoPE and 0.000 for the path model, which transfers completely
(0.9998 on both maps). The 4x LSTM fits its training map best (0.974-0.977) and transfers worst (0.572-0.574).

Revisit strata (S3, held-out map, mean over seeds; stratum floors: retrace short gap 0.910, blank long gap 0.498 /
wrap-only 0.515):

| cell | short gap (< 128 moves) | long gap (>= 128) | wrap-only |
|---|---|---|---|
| Path_L1 | 1.000 | 1.000 | 0.997 |
| RoPE_L1 | 0.774 | 0.462 | 0.479 |
| RNN cells (range) | 0.621-0.704 | 0.314-0.367 | 0.327-0.378 |

The RNNs are below retrace even on short-gap revisits (0.62-0.70 vs 0.910) and below the blank level on long-gap and
wrap-only revisits: on a new map they have learned neither the map nor the retrace rule that index attention partly
learns. They do read their context (S8: CONTEXT USED in every cell; destroyed 0.301-0.313 vs blank 0.508, intact
0.572-0.650 on the same 50 walks), so the held-out score is not a context-free guess, but the in-context use is weak.

Mechanism, as far as this batch shows it: with one fixed map per run, the cheapest way for a recurrent network to lower
the training loss is to store that map in its weights (the training loss rewards it on every walk), and a fixed-size state
need not learn to write and read a new map. Attention with a path-integrated phase has a content-addressable store in its
context and learns to use it; its training loss falls the same way on both maps. The pilot's low RNN training losses
(0.16-0.33 at epoch 300) were this memorisation, not map learning in context. This account is consistent with the numbers
but not tested: no RNN was trained on many maps (Scope).

**Post hoc (analysis, unregistered; `docs/audits/2026-10-27/rnn_torus/trainmap_probe.py` -> `trainmap_probe_out.{txt,json}`,
CPU, eval only, the first 10 of each seed's 100 evaluation walks; accuracy on them within 0.007 of the registered values).** Do the RNNs on a new map dead-reckon into the stored training map,
predicting the training map's object at the true cell? No: on revisits where the held-out, training and an unrelated map
(seed 20000) all differ (~1,500 per seed), RNN predictions equal the training map's content 0.095-0.103 of the time and
the unrelated map's 0.107-0.118 (training map below the unrelated map on 6-8 of 8 seeds per cell; index RoPE 0.107 vs
0.121; path 0.000 / 0.000). Their own-map accuracy therefore comes from something that does not carry over when the
observations change -- consistent with locating themselves on the training map by what they see, not by integrating the
moves -- but this probe does not identify the mechanism.

## Other declared secondaries

- **S4, out of distribution (rule 10; Amendment 1 A6: a drop is memory length or recurrent-state length generalisation,
  not separated).** T = 512 / 2048: path 1.000 / 0.985; RoPE_L1 0.758 / 0.552; RNN cells 0.604-0.686 / 0.532-0.605, each
  0.035-0.044 below its T = 1024 score at 2048.
- **S6, r(final-5% training loss, accuracy T = 1024) over the 64 runs: -0.050.** Training loss does not predict held-out
  accuracy in this batch (rule 2). Post hoc: over the 48 RNN runs alone r = **+0.70** (lower training loss, LOWER held-out
  accuracy), mostly between cells (the 4x LSTMs have the lowest loss and the lowest held-out accuracy; within-cell r -0.06
  to +0.56).
- **S7, run classes:** path 8 SOLVED; RoPE 8 STALLED; every RNN cell 8 STALLED except the 4x LSTM at 3e-3 (3 STALLED,
  5 DESCENDING; final-5% loss 0.086-0.090, its own-map accuracy 0.977). `experiment_audit`'s windowed classes agree.
- **S8, context destruction:** CONTEXT USED in all eight cells (path 0.260, RoPE 0.352, RNNs 0.301-0.313, all below
  blank - 0.10).
- **S9, capacity:** LSTM_L1_h313 (1e-3) - LSTM_L1_h157 (3e-3) = **-0.033** (perm p 0.00016); 0/8 vs 0/8. Four times the
  parameters transfer worse (and fit the training map better: 0.974 vs 0.933). Each at its own pilot-picked rate.

## Predictions (pre-registered, pilot-informed for the RNNs) vs outcomes

The RNN predictions were written after one pilot seed's training loss to epoch 300, which is loss on the run's own
training map. They read that loss as map learning and were contradicted wherever held-out behaviour was predicted.

| prediction | outcome | |
|---|---|---|
| Path_L1 0.999-1.000, 8/8 | 0.9998, 8/8 | held |
| RoPE_L1 0.70-0.75, 0/8, STALLED | 0.724, 0/8, 8 STALLED | held |
| ~205k RNN cells 0/8 SOLVED | 0/8 each | held |
| ~205k RNN cells held-out 0.84-0.92, above RoPE_L1, at or above retrace | 0.582-0.649, below RoPE_L1 and retrace | **contradicted** |
| 4x LSTM 0-3/8 SOLVED, 0.90-0.97, the selected best cell; stated risk: it solves 4-7/8 | 0/8 at both rates, 0.572-0.574; a 4x cell is selected (3e-3) | SOLVED and selection held; **accuracy contradicted**; the risk did not occur |
| PRIMARY PATH WINS, binding cell a 4x LSTM | PATH WINS, binding `LSTM_L1_h313@0.001`, both routes | held |
| Matched-parameter PATH WINS | PATH WINS (4/4) | held |
| S1 RNN ABOVE INDEX 1L (+0.10 to +0.25) for every cell | RNN BELOW INDEX 1L for every cell (-0.075 to -0.152) | **contradicted (sign reversed)** |
| S2 small cells within +/-0.07 of retrace, large above | -0.19 to -0.27, 0/8 above | **contradicted** |
| S3 short gap >= 0.95; long gap / wrap-only 0.6-0.9, above blank | 0.62-0.70; 0.31-0.38, below blank | **contradicted** |
| S4 T = 2048 below T = 1024 for every RNN cell | -0.035 to -0.044 in every cell | held |
| S5 own map >= held-out by 0.00-0.05 | +0.29 to +0.40 | **contradicted (the central finding)** |
| S6 r(loss, acc) strongly negative | -0.050 | **contradicted** |
| S8 CONTEXT USED everywhere | CONTEXT USED everywhere | held |
| S9 large - small LSTM +0.03 to +0.08 | -0.033 | **contradicted (sign reversed)** |

## Learning rates (pilot, seed 3390; rule written before the readout)

Lowest mean training loss over epochs 275-300 of the 900-epoch schedule, ties within 0.01 to the rate nearer 1e-3; no run
UNSTABLE. LSTM_L1_h313 1e-3 (0.160 vs 3e-3 0.165; inside the tie band, so Amendment 1 A3 ran both rates); LSTM_L1_h157 3e-3
(0.303; 3e-4 0.634, 1e-3 0.334, 1e-2 0.332: interior optimum); GRU_L1_h181 1e-3 (0.329 vs 0.327, tie); LSTM_L2_h111 3e-3
(0.308 vs 0.329); GRU_L2_h128 3e-3 (0.347; 3e-4 0.631, 1e-3 0.372, 1e-2 0.388: interior). The pick rule used training loss,
which here measures memorisation of the training map, so "best rate" means best at fitting the training map. The two 4x
cells (1e-3 and 3e-3, with 3x the effective decay at 3e-3) land within 0.002 of each other on the held-out map, so the
rate is not what separates the RNNs from the path model at 4x.

## Reading for the paper

One path-integrated attention layer beats LSTM and GRU baselines at matched parameters (one and two layers) and an LSTM
with four times the parameters on held-out maps, by +0.35 to +0.43 (0.9998 against 0.57-0.65). The recurrent networks fit
their training map (0.93-0.98) but do not transfer: on a new map they score below one index-RoPE layer and below the
map-free retrace rule, even on short-gap revisits. Under the paper's protocol (one training map per run, evaluation on a
new one) an integrator alone is not enough; what the path layer adds is a context store addressed by the integrated
position. This answers the review's B-M9 / A-M2 ("an RNN given the actions integrates by construction") for this protocol:
the RNN has the integration but not the transfer.

## Scope

- One length (T = 1024), one task, one recipe and budget (900 epochs), n = 8 per cell. Each RNN shape at one
  pilot-chosen rate (one seed, 300 of 900 epochs; the 4x LSTM at two), with rate and effective weight decay coupled.
- **One training map per run.** An RNN trained on many maps (a new map per walk, as in meta-learning setups such as TEM's
  environment switching) might learn to write and read a map in its state; that was not tested. This is a like-for-like
  comparison under the anchors' protocol, which is the paper's, not a claim about recurrent networks in general.
- SOLVED / STALLED / DESCENDING are classes of the training loss on the run's own map. For the RNNs a low training loss
  is memorisation, so these tags mean little here (none reached the 0.05 threshold in any case); the 4x LSTM at 3e-3 was
  still descending on 5/8 seeds, which would bring it closer to its training map, not to the held-out one.
- No existence construction (rule 14): "the RNNs do not learn to transfer within this budget and protocol", not "an RNN
  cannot represent an in-context map". The 4x GRU and 4x two-layer shapes, gated SSMs (Mamba) and the GPS-PE control
  (index attention given cumsum of the moves as an absolute code; review A-M2) were not run.
- RoPE_L1 is read in eval mode (Amendment 1 A5: eval mode under-reports one-layer attention below ceiling), so S1's
  RNN-below-index margins are lower bounds on the true gap; the primary is unaffected.
