import sys; sys.path.insert(0, sys.argv[1]); from rep import apply
P = [
(r"""torus is $+0.189$ at $8\times$ the training length, larger than rank's $+0.085$. The
rank decides whether training \emph{finds} the solution, not whether it exists.""",
r"""torus is $+0.189$ at $8\times$ the training length, larger than rank's $+0.085$. At
rank $2$ training rarely \emph{finds} the solution, although it exists; which property
of the bottleneck is responsible is not separated."""),
(r"""signed, \textbf{and} rank $r\ge4$ per head to be found reliably & MapFormer at $r\ge4$. At $r=2$ (ours shared, or the paper's per head) training""",
r"""signed, \textbf{and} a shared $r\ge4$ to be found reliably (two heads tested) & MapFormer at $r\ge4$. At $r=2$ (ours shared, or per head) training"""),
(r"""initial weights, our shared $r=2$ solves $0/8$ seeds within $900$ epochs, the
  paper's per-head $r=2$ $2/8$ and $r=4$ $8/8$, while a rank-$2$ projection of a solved
  $r=4$ model scores $0.995$ and holds under training: a search deficit, in the paper's
  own design. Nothing yet explains why training finds the rank-$2$ solution so rarely
  when it occupies a $2$-plane either way, and $r=4$'s smaller initial $W_{\mathrm{out}}$
  is not separated from its rank.""",
r"""initial weights, our shared $r=2$ solves $0/8$ seeds within $900$ epochs, a
  per-head $r=2$ $2/8$ and a shared $r=4$ $8/8$, while a rank-$2$ projection of a solved
  $r=4$ model scores $0.995$ and holds under training: a search deficit (measured for
  our shared $r=2$, inferred for the per-head one). The paper states a per-head
  $W_{\mathrm{in}}$ but not $W_{\mathrm{out}}$'s shape: the per-head arm is its literal
  reading, and with a full $W_{\mathrm{out}}$ its $r=2$ at two heads is our $r=4$.
  Nothing yet explains why training finds the rank-$2$ solution so rarely
  when it occupies a $2$-plane either way, and against the per-head arm, per-head
  rank, cross-head sharing and $W_{\mathrm{out}}$'s per-entry scale are not separated."""),
]
apply("positional_review.tex", P)
