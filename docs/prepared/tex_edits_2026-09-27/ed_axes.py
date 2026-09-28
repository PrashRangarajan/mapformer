import sys; sys.path.insert(0, sys.argv[1]); from rep import apply
P = [
# abstract
(r"""(measured for our shared $r=2$). Which of per-head rank, cross-head sharing and
$W_{\mathrm{out}}$'s scale is responsible is not separated.""",
r"""(measured for our shared $r=2$). A separation batch identifies the responsible property
as the per-head rank of the map: a block-diagonal $r=4$ (rank $2$ per head) solves $2/8$
and a per-head $r=4$ $8/8$ (Fisher and permutation $p=0.0070$); cross-head sharing and
$W_{\mathrm{out}}$'s per-entry scale are individually unmeasured."""),
# abstract stamp
(r"""``built to want a clock'' on which the ordering ``inverts'' (a signed rewind solves it
exactly, and nothing inverts).
\end{abstract}""",
r"""``built to want a clock'' on which the ordering ``inverts'' (a signed rewind solves it
exactly, and nothing inverts).

\smallskip\noindent\emph{Corrected 2026-09-27.} The 2026-09-25 text said which property
of the rank bottleneck is responsible was not separated; it is per-head rank
(\texttt{RANK\_SEP\_RESULTS.md}). A search-aid batch at $r=2$ (\texttt{LOOP\_RANK\_RESULTS.md})
is added to $\S$\ref{sec:rankmeas}.
\end{abstract}"""),
# F2
(r"""  per-head $r=2$ and $8/8$ at a shared $r=4$, while a rank-$2$ solution exists and holds
  ($\S$\ref{sec:rankmeas}); what separates the per-head $r=2$ from $r=4$ is not isolated.""",
r"""  per-head $r=2$ and $8/8$ at a shared $r=4$, while a rank-$2$ solution exists and holds
  ($\S$\ref{sec:rankmeas}). What separates them is per-head rank: a block-diagonal $r=4$
  (rank $2$ per head) solves $2/8$ and a per-head $r=4$ $8/8$ (Fisher and permutation
  $p=0.0070$); sharing and $W_{\mathrm{out}}$'s per-entry scale are unmeasured."""),
# sec:rankmeas
(r"""latent dimensions and identical initial $W_{\mathrm{in}}$, but differ in three ways at
once --- the rank of each head's content-to-angle map, whether the heads read a shared
latent (perfectly confounded with it at two heads), and $W_{\mathrm{out}}$'s per-entry
scale --- and which is responsible is not separated; the initial scale of the angle
increments is matched. The two $r=2$ arms are not distinguishable ($2/8$ against
$0/8$, Fisher $p=0.47$): unmeasured.""",
r"""latent dimensions and identical initial $W_{\mathrm{in}}$, but differ in three ways at
once --- the rank of each head's content-to-angle map, whether the heads read a shared
latent, and $W_{\mathrm{out}}$'s per-entry scale. A second batch, built the same way,
separates them with two more arms: a block-diagonal $r=4$ (the shared $r=4$ with its
cross-head blocks held at zero, so rank $2$ per head at $r=4$'s $W_{\mathrm{out}}$ scale)
solves $2/8$ ($0.948$), and a per-head $r=4$ (rank $4$ per head, separate latents) $8/8$
($0.999$). The split is entirely by per-head rank: every arm at rank $2$ per head solves
$0$--$2$ of $8$, every arm at rank $4$ solves $8$ of $8$, and total latent size does not
track it. Per-head rank fires (per-head against block-diagonal $r=4$, Fisher and permutation
$p=0.0070$, Holm $0.028$); sharing (per-head against shared $r=4$, $8/8$ against $8/8$,
permutation $p=0.59$) and $W_{\mathrm{out}}$'s per-entry scale (block-diagonal $r=4$ against
per-head $r=2$, $2/8$ against $2/8$, permutation $p=0.24$) are unmeasured. The
initial-angle-scale worry is answered rather than merely matched: zeroing the cross-head
blocks halves the block-diagonal arm's initial increment (std $0.208$), but our shared and
the per-head $r=2$ start at normal scale ($0.335$, $0.328$) and fail the same way, so low
initial scale is not necessary for failure. The two original $r=2$ arms are not
distinguishable ($2/8$ against $0/8$, Fisher $p=0.47$): unmeasured."""),
(r"""recency deficit. Scope: two heads, one task, one length, one width, one recipe, budget-scoped (runs
still descending count as unsolved).""",
r"""recency deficit. A search aid partly recovers rank $2$ but does not reach rank $4$: looping
our shared $r=2$ block $4\times$ at identical parameters and initialisation raises accuracy
from $0.894$ to $0.973$ (permutation $p=0.0034$) and solves $2/8$; four real layers ($4\times$
the parameters) reach $0.990$ and $5/8$, beating the loop ($+0.017$, permutation $p=0.0009$);
$r=4$ is at $0.998$ and $8/8$. Final losses form three regimes, and nothing at rank $2$ enters
$r=4$'s on any seed. The registered verdict is unmeasured, and uncertain rather than negative:
the $0.05$ solved cutoff falls inside the aids' spread, and at $0.08$ the loop would have met
its solved-count condition. Since depth beats the matched-parameter loop, this is not search
at constant capacity. Scope: two heads, one task, one length, one width, one recipe, budget-scoped (runs
still descending count as unsolved)."""),
# keep the bottleneck
(r"""($\S$\ref{sec:rankmeas}); a shared $r=4$, which finds it on $8/8$ seeds, costs $384$
parameters.""",
r"""($\S$\ref{sec:rankmeas}); what buys it is per-head rank $4$, and a shared $r=4$, which
finds it on $8/8$ seeds, costs $384$ parameters."""),
# combinations caption
(r"""\caption{Combinations the frame identifies, ordered by how much evidence already points at them. None has been built.}""",
r"""\caption{Combinations the frame identifies, ordered by how much evidence already points at them. Only the last has been built (Match-Query, one batch); on the torus the loop has been tested at $r=2$ only, where it helps accuracy but does not reach $r=4$ ($\S$\ref{sec:rankmeas}).}"""),
# conclusion
(r"""the deficit is one of search; which property of the bottleneck is responsible is not
separated.""",
r"""the deficit is one of search; the responsible property is the per-head rank of the map
($p=0.0070$), with cross-head sharing and $W_{\mathrm{out}}$'s per-entry scale unmeasured,
and looping or deepening the $r=2$ model partly recovers it without reaching rank $4$."""),
(r"""$16$-epoch torus recipe and the $r=2$ basis reading).
""",
r"""$16$-epoch torus recipe and the $r=2$ basis reading).
\emph{Corrected 2026-09-27:} it also said the responsible property of the bottleneck was not
separated; it is per-head rank.
"""),
]
apply("axes_measured.tex", P)
