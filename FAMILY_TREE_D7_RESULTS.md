
## Depth 7 closes the depth-5 caveat -- against the NC hypothesis

`FAMILY_TREE_RESULTS.md` recorded a caveat: 63 nodes in a 128-dim model may be
memorisable by content attention, which would let ANY position code coast and
mask a real non-commutativity advantage. Depth 7 quadruples the nodes to 255.

| effect (paired, per seed) | depth 5 (63 nodes) | depth 7 (255 nodes) |
|---|---|---|
| NC-L − commutative | +0.005, 3/3 | **+0.0017, 2/3** (one seed −0.003) |
| NC-NL − commutative | +0.014, 3/3 | **+0.0037, 2/3** (one seed −0.006) |
| path integration − index | +0.115 | **+0.180** |

The non-commutativity effect **shrank** with scale and lost its 3/3 consistency;
the path-integration effect **grew**. So the depth-5 null was not a
memorisation artifact -- more nodes made it stronger, not weaker.

Floors at depth 7: chance 0.1250, **hub baseline 0.144** (read every number
against the hub floor, not chance).

### What this settles

MapFormer's appendix B.2.2 argues that a commutative group CANNOT represent a
family tree, motivating `MapEM-NC` with exactly this example. That is correct
group theory. Measured at two scales on a structure with non-commutativity 1.000,
it does not translate into task performance: the commutative control matches the
non-commutative variants to within one standard deviation, while costing 34x less
at L=2048 (`TIMING_BENCHMARK.md`).

Path integration, meanwhile, is worth +0.180 here -- roughly 50x the
non-commutativity effect, and corroborating Match-Query on an unrelated
non-spatial structure.
