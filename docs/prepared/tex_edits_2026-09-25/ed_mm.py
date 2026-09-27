import sys; sys.path.insert(0, sys.argv[1]); from rep import apply
P = [
(r"""paper's per-head design each head reads its own two-dimensional accumulator).""",
r"""paper's per-head design, read literally, each head reads its own two-dimensional
accumulator)."""),
(r"""weights, $8$ seeds: our shared $r=2$ solves $0/8$ (accuracy $0.894$), the paper's
per-head $r=2$ $2/8$ ($0.885$), a shared $r=4$ $8/8$ ($0.998$); Fisher $p=0.0002$ and
$0.007$ against $r=4$. The per-head arm and $r=4$ have the same four latent
dimensions, so what decides it is the rank of each head's content-to-angle map. A
rank-$2$ projection of each solved $r=4$ model scores $0.9955$ frozen, and trained
onward it stays solved ($7/8$, as does the $r=4$ control). The solution is
expressible and stable at rank $2$ and training from scratch does not find it.
Budget-scoped (within $900$ epochs), one task and width, and $r=4$'s smaller initial
$W_{\mathrm{out}}$ (bound $0.5$ against $0.707$) is not separated from its rank
(\texttt{RANK\_MI\_RESULTS.md}, \texttt{RANK\_PROJ\_RESULTS.md}).""",
r"""weights, $8$ seeds: our shared $r=2$ solves $0/8$ (accuracy $0.894$), a
per-head $r=2$ $2/8$ ($0.885$), a shared $r=4$ $8/8$ ($0.998$); Fisher $p=0.0002$ and
$0.007$ against $r=4$, so $r=4$'s advantage is not its initial draws. The per-head arm
and $r=4$ have the same four latent dimensions and identical initial
$W_{\mathrm{in}}$, but which of per-head rank, cross-head sharing (perfectly
confounded with it at two heads) and $W_{\mathrm{out}}$'s per-entry scale (bound
$0.5$ against $0.707$) is responsible is not separated; the initial scale of the
angle increments is matched. The two $r=2$ arms are not distinguishable (Fisher
$p=0.47$). A rank-$2$ projection of each solved $r=4$ model scores $0.9955$ frozen,
and trained onward it stays solved ($7/8$, as does the $r=4$ control). The solution
is expressible and stable at rank $2$ (measured for our shared $r=2$; inferred for
the per-head one) and training from scratch does not find it. The paper states a
per-head $W_{\mathrm{in}}$ but not $W_{\mathrm{out}}$'s shape: the per-head arm is
its literal reading, and with a full $W_{\mathrm{out}}$ its $r=2$ at two heads is our
$r=4$. Budget-scoped (within $900$ epochs), two heads, one task, length and width
(\texttt{RANK\_MI\_RESULTS.md}, \texttt{RANK\_PROJ\_RESULTS.md})."""),
(r"""$T=1024$ the rank per head decides whether training finds the solution at all
  ($0/8$ shared $r=2$, $2/8$ the paper's per-head $r=2$, $8/8$ $r=4$) while a rank-$2$
  solution exists""",
r"""$T=1024$ training rarely finds the solution through a rank-$2$ bottleneck
  ($0/8$ shared $r=2$, $2/8$ a per-head $r=2$, $8/8$ a shared $r=4$; per-head rank,
  sharing and $W_{\mathrm{out}}$'s scale not separated) while a rank-$2$
  solution exists"""),
(r"""and holds, and training finds it on $0/8$ seeds (ours) and $2/8$ (the paper's
per-head $r=2$) against $8/8$ at $r=4$.""",
r"""and holds, and training finds it on $0/8$ seeds (ours) and $2/8$ (a per-head $r=2$,
the paper's literal reading) against $8/8$ at a shared $r=4$."""),
(r"""signed, \textbf{and} rank $r\ge4$ per head to be found reliably & MapFormer at $r\ge4$. At $r=2$ (ours shared or the paper's per head) training""",
r"""signed, \textbf{and} a shared $r\ge4$ to be found reliably (two heads tested) & MapFormer at $r\ge4$. At $r=2$ (ours shared or per head) training"""),
]
apply("mapformer_math.tex", P)
