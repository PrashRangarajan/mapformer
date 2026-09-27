import sys; sys.path.insert(0, sys.argv[1]); from rep import apply
P = [
(r"""recency deficit (measured for our shared $r=2$; for the per-head arm it is inferred).
Scope:""", r"""recency deficit. Scope:"""),
(r"""The representation exists
and is held;""", r"""The representation exists
and is held (measured for our shared $r=2$; for the per-head arm it is inferred);"""),
]
apply("axes_measured.tex", P)
