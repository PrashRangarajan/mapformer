# Does EM's capacity claim hold once the init pathology is removed?

## Why

The paper claims **superior capacity for MapEM in recall tasks** -- "memorizing large
sets of objects" -- with a structural mechanism: EM factorises what and where into
two spaces, so the conjunction lives on their **product**, and "the conjunction
states in the WM model are thus **smaller by a quadratic factor**". Our tensor-lift
identity derives exactly that: `(A_X . A_P)_ts = (q^x (x) q^p).(k^x (x) k^p)` is a
bilinear form on a `d_x*d_p` space.

**Every "EM is worse" result in this project used `VanillaEM`** -- paper-faithful
separate `q0/k0` -- which `MINIGRID_EM_FIX.md` just showed carries an init pathology:
`A_P` is not peaked at zero displacement at init, so the AND-gate multiplies content
by a random position mask. Fixed with shared `p_0` at `r=4`, EM matched WM exactly
and had a *tighter* distribution.

So the vocabulary sweep -- the one test aimed at the capacity claim -- has never been
run with a healthy EM arm.

## A second thing this must separate

The published sweep used the OLD recipe: `LinearLR(1.0->0.0)` from step one at
lr 3e-4. `COMP_HEADROOM.md` showed that recipe was worth `+0.160` on another task,
and `RECIPE_POWER.md` that fixing it cuts torus seed sd 3.5x. The `n_obs=256`
collapse I am attributing to init (`VanillaEM_P0` per-seed 0.910 / **0.502** / 0.906)
may simply be that recipe. **This batch runs cosine / lr 1e-3 throughout**, so if the
collapse disappears for `VanillaEM_P0`, it was never the origin vector there.

## Arms (5) x vocab (3) x seeds (8) = 120 runs, ONE batch

`Vanilla` (WM r=2) / `Vanilla_r4` (WM r=4, the matched-rank comparator) /
`VanillaEM` (the pathological arm every prior result used) / `VanillaEM_P0` (origin
fix only) / `VanillaEM_P0_r4` (both fixes).

`n_obs` in {16, 64, 256}. **4096 is excluded**: every arm sits at the 0.50 blank
floor there and it carries no signal in either direction.

## Predictions

**P1 -- the capacity claim.** `EM_P0_r4 - Vanilla_r4` should **grow with
vocabulary**. The shape is the claim, not any single cell: capacity should matter
more as there are more distinct bindings to store. *Refuted by* a flat or decreasing
profile, which would leave the quadratic-capacity claim with no support in this
project at any configuration.

**P2 -- no collapse anywhere.** `EM_P0_r4`'s worst-to-second-worst gap under 0.05 at
every vocabulary (it was 0.002 on MiniGrid, against 0.137 for the unfixed arm).

**P3 -- recipe or init.** If `VanillaEM_P0` at `n_obs=256` no longer collapses under
the better recipe, the published 0.502 seed was the recipe and my init attribution
was wrong there.

## Ceiling check, done in advance

At `n_obs=16` the published `Vanilla` is **0.997 at T=128** -- a ceiling, which
cannot show a gain of any size. **The primary contrast is `n_obs=256`** (published
Vanilla 0.913 at T=128), with 64 as the intermediate point that makes P1 a profile
rather than a pair. T=128 is the primary length: the paper's capacity claim is at
short `l`, where position is sharp and the AND-gate is not fighting drift.

## Reading rule

Per-seed minimum and the worst-to-second-worst gap before any mean, as in
`MINIGRID_EM_FIX`.
