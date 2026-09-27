import sys; sys.path.insert(0, sys.argv[1]); from rep import apply
P = [
# 1 abstract
(r"""different accumulator for each task. The \textbf{input-side rank} of the
content-to-phase map decides whether training \emph{finds} the solution, not whether
it exists: trained and tested at $T=1024$ for $900$ epochs from matched initial
weights, our shared $r=2$ solves $0/8$ seeds, the paper's per-head $r=2$ $2/8$ and
$r=4$ $8/8$, while a rank-$2$ solution exists ($0.9955$) and holds under training.""",
r"""different accumulator for each task. At \textbf{input-side rank} $2$ of the
content-to-phase map, training rarely \emph{finds} the solution, although it
exists: trained and tested at $T=1024$ for $900$ epochs from matched initial
weights, our shared $r=2$ solves $0/8$ seeds, a per-head $r=2$ $2/8$ and a shared
$r=4$ $8/8$, while a rank-$2$ solution exists ($0.9955$) and holds under training
(measured for our shared $r=2$). Which of per-head rank, cross-head sharing and
$W_{\mathrm{out}}$'s scale is responsible is not separated."""),
# 2 bottleneck is the separator
(r"""pass, so the finding that the per-head rank decides whether training finds the torus
solution ($\S$\ref{sec:rankmeas}) may be a finding""",
r"""pass, so the finding that training rarely finds the torus solution through a
rank-$2$ bottleneck ($\S$\ref{sec:rankmeas}) may be a finding"""),
# 3 F2
(r"""$r=32$. Trained and tested at $T=1024$ from matched initial weights, the rank per
  head decides whether training finds the solution within $900$ epochs ($0/8$, $2/8$,
  $8/8$), while a rank-$2$ solution exists and holds ($\S$\ref{sec:rankmeas}).""",
r"""$r=32$. Trained and tested at $T=1024$ from matched initial weights, training finds
  the solution within $900$ epochs on $0/8$ seeds at our shared $r=2$, $2/8$ at a
  per-head $r=2$ and $8/8$ at a shared $r=4$, while a rank-$2$ solution exists and holds
  ($\S$\ref{sec:rankmeas}); what separates the per-head $r=2$ from $r=4$ is not isolated."""),
# 4 main paragraph, first half
(r"""solves $0/8$ (accuracy $0.894$), the paper's per-head $r=2$ $2/8$ ($0.885$) and a
shared $r=4$ $8/8$ ($0.998$); Fisher $p=0.0002$ and $0.007$ against $r=4$. The
per-head arm and $r=4$ have the same four latent dimensions, so what separates them is
the rank of each head's content-to-angle map. The solution""",
r"""solves $0/8$ (accuracy $0.894$), a per-head $r=2$ $2/8$ ($0.885$) and a
shared $r=4$ $8/8$ ($0.998$); Fisher $p=0.0002$ and $0.007$ against $r=4$, so $r=4$'s
advantage is not its initial draws. The per-head arm and $r=4$ have the same four
latent dimensions and identical initial $W_{\mathrm{in}}$, but differ in three ways at
once --- the rank of each head's content-to-angle map, whether the heads read a shared
latent (perfectly confounded with it at two heads), and $W_{\mathrm{out}}$'s per-entry
scale --- and which is responsible is not separated; the initial scale of the angle
increments is matched. The two $r=2$ arms are not distinguishable ($2/8$ against
$0/8$, Fisher $p=0.47$): unmeasured. The solution"""),
# 5 main paragraph, second half
(r"""recency deficit. Scope: one task, one width, one recipe, budget-scoped (runs still
descending count as unsolved), and $r=4$'s smaller initial $W_{\mathrm{out}}$ (bound
$0.5$ against $0.707$) is not separated from its rank. \textbf{Which $r=2$}: our
bottleneck is shared across heads and the paper's is per head, so at two heads our
$r=2$ has half the paper's latent dimensions; the per-head arm above is the paper's
design, and it fails too. The seed""",
r"""recency deficit (measured for our shared $r=2$; for the per-head arm it is inferred).
Scope: two heads, one task, one length, one width, one recipe, budget-scoped (runs
still descending count as unsolved). \textbf{Which $r=2$}: our bottleneck is shared
across heads and the paper's $W_{\mathrm{in}}$ is per head, so at two heads our $r=2$
has half the paper's latent dimensions. The paper does not state $W_{\mathrm{out}}$'s
shape: the per-head arm above is its literal reading, and with a full
$W_{\mathrm{out}}$ the paper's $r=2$ at two heads is our shared $r=4$. The seed"""),
# 6 keep the bottleneck
(r"""through, and its rank per head decides whether training finds the torus solution at
all ($\S$\ref{sec:rankmeas}); $r=4$ costs $384$ parameters.""",
r"""through, and at rank $2$ training rarely finds the torus solution within budget
($\S$\ref{sec:rankmeas}); a shared $r=4$, which finds it on $8/8$ seeds, costs $384$
parameters."""),
# 7 conclusion
(r"""length. The rank per head decides whether training finds the torus solution --- $0/8$,
$2/8$ and $8/8$ seeds for our shared $r=2$, the paper's per-head $r=2$ and $r=4$,
trained and tested at $T=1024$ --- although a rank-$2$ solution exists and is held, so
the deficit is one of search.""",
r"""length. Through a rank-$2$ bottleneck training rarely finds the torus solution ---
$0/8$, $2/8$ and $8/8$ seeds for our shared $r=2$, a per-head $r=2$ and a shared $r=4$,
trained and tested at $T=1024$ --- although a rank-$2$ solution exists and is held, so
the deficit is one of search; which property of the bottleneck is responsible is not
separated."""),
]
apply("axes_measured.tex", P)
