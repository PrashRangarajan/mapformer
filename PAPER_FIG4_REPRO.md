# Do we reproduce the paper's Figure 4?

Sec 5.4 / Fig 4 makes four checkable claims about a trained MapFormer.
Each is tested here with the paper's own metric, on 8 seeds, at the
paper's r=2 and at r=4.

| claim | paper reports | r=2 (ours) | r=4 (ours) |
|---|---|---|---|
| **C1** ‖Δ_action‖ / ‖Δ_obs‖ | ≫ 1 <br><sub>observations leave position untouched</sub> | 25.0 ± 22.4 | 35.9 ± 19.0 |
| **C2** cos(Δ_left, Δ_right) | **−1** <br><sub>opposite actions cancel</sub> | -0.729 ± 0.373 | -0.996 ± 0.004 |
| **C3** \|cos(Δ_left, Δ_up)\| | **≫ 0** <br><sub>reported as a LIMITATION</sub> | 0.779 ± 0.276 | 0.174 ± 0.083 |
| **C4** ‖v_obs‖ / ‖v_action‖ | ≫ 1 <br><sub>only observations update content</sub> | 0.57 ± 0.06 | 0.60 ± 0.04 |

## Verdict

**C1 reproduces cleanly.** Action tokens rotate 25x more than observation tokens
at r=2 (36x at r=4). The model learns unsupervised which symbols move the agent,
which is the paper's central interpretability claim.

**C2 reproduces only weakly at r=2.** cos(opposite) is **-0.729 +/- 0.373** against
the paper's -1 -- the right sign, but with a seed spread that spans nearly half the
range. At r=4 it is **-0.996 +/- 0.004**, essentially exact. So the paper's clean
"opposite actions cancel" holds in our setup at r=4 and only approximately at the
rank the paper specifies.

**C3 reproduces, including the failure the paper itself reports.**
|cos(orthogonal)| = **0.779** at r=2, which is the "≫ 0" the paper documents and
attributes to a missing disentanglement constraint. This is not a defect in our
reproduction; it is the paper's own result.

**C4 DOES NOT REPRODUCE.** The paper reports that after training
"the norm of value embeddings in the attention layer becomes much bigger for
observations than actions (‖v_o‖ ≫ ‖v_a‖), implying that only observations
contribute in updating the state's content." We measure the ratio at **0.57 +/-
0.06** -- not merely short of ≫ 1, but **inverted**: our action tokens carry the
larger value norms. r=4 does not change this (0.60).

### What C4's failure probably is, and what would settle it

The leading hypothesis is that Fig. 4 shows an **EM** model, not WM. Sec 5.4's
framing is explicitly about EM's factorisation -- "this factorization in two
separate pools of neurons should allow EM to be more efficient than WM, as in the
former, neurons specialize for either position or observation" -- and every arm we
tested here is **MapWM**, which has no such separation by construction. Its
additive attention mixes position and content in one pool, so there is no
architectural reason for its value norms to split by token type.

That is checkable: train `VanillaEM` at r=2 in the same batch and re-measure. If
the ratio inverts to ≫ 1 on EM and stays < 1 on WM, C4 is an EM property that the
caption does not scope, and our reproduction is complete. If EM also comes back
< 1, we have a real discrepancy with the paper and it belongs in the record as one.

**Until that runs, the honest statement is: three of the paper's four Fig. 4
claims reproduce on MapWM, and the fourth is untested on the architecture it may
have been measured on.**

### One result the paper did not have

On C3, **widening the bottleneck does what the paper says would need an extra loss
term.** |cos(orthogonal)| falls **0.779 → 0.174** from r=2 to r=4, with no
bounded-energy constraint and no change to the objective. The paper's proposed
remedy for its own disentanglement failure is an added regulariser; a wider
bottleneck gets most of the way there for +384 parameters, and simultaneously
sharpens C2 from -0.729 to -0.996.

Inference only, 8 seeds, torus checkpoints from `runs/rank_sweep`. Δ_in is the r-dimensional Lie-algebra code (the paper's metric for C2/C3); Δ is the full per-head increment (C1).
