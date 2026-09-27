import sys; sys.path.insert(0, sys.argv[1]); from rep import apply
P = [
(r"""$8/8$ seeds, and trained at
  $T=1024$ training rarely finds the solution through a rank-$2$ bottleneck""",
r"""$8/8$ seeds, and when trained and tested at
  $T=1024$ training rarely finds the solution through a rank-$2$ bottleneck"""),
]
apply("mapformer_math.tex", P)
