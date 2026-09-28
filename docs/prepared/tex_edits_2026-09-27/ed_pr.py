import sys; sys.path.insert(0, sys.argv[1]); from rep import apply
P = [
# abstract stamp
(r"""the ordering ``inverts'' on a clock task (the recency task does not need a clock, and
nothing inverts).
\end{abstract}""",
r"""the ordering ``inverts'' on a clock task (the recency task does not need a clock, and
nothing inverts).

\smallskip\noindent\emph{Corrected 2026-09-27.} The 2026-09-25 text said which property of
the rank bottleneck is responsible was not separated. A separation batch
(\texttt{RANK\_SEP\_RESULTS.md}) since identified it as the per-head rank of the
content-to-phase map; cross-head sharing and $W_{\mathrm{out}}$'s per-entry scale are
individually unmeasured.
\end{abstract}"""),
# C3
(r"""At
rank $2$ training rarely \emph{finds} the solution, although it exists; which property
of the bottleneck is responsible is not separated.""",
r"""At
rank $2$ per head training rarely \emph{finds} the solution, although it exists. The
responsible property is the per-head rank of the map (block-diagonal arms at rank $4$
against rank $2$ per head: $8/8$ against $2/8$ seeds, Fisher and permutation
$p=0.0070$); cross-head sharing and $W_{\mathrm{out}}$'s per-entry scale are individually
unmeasured."""),
# capability table
(r"""\emph{absolute location in a bounded map} & signed, \textbf{and} a shared $r\ge4$ to be found reliably (two heads tested) & MapFormer at $r\ge4$. At $r=2$ (ours shared, or per head) training usually fails to find the torus solution within budget --- $0/8$ and $2/8$ against $8/8$ at $T{=}1024$ --- though a rank-$2$ solution exists: search, not capacity \\[3pt]""",
r"""\emph{absolute location in a bounded map} & signed, \textbf{and} per-head rank $\ge4$ to be found reliably (two heads tested) & MapFormer at per-head rank $\ge4$, shared or per head. At per-head rank $2$ (ours shared, per head, or block-diagonal $r=4$) training usually fails to find the torus solution within budget --- $0/8$, $2/8$, $2/8$ against $8/8$, $8/8$ at per-head rank $4$, $T{=}1024$ --- though a rank-$2$ solution exists: search, not capacity \\[3pt]"""),
# conclusion stamp
(r"""$r=2$ fails by a skewed basis. See the corrected abstract.
""",
r"""$r=2$ fails by a skewed basis. See the corrected abstract.
\emph{Corrected 2026-09-27:} the rank bottleneck's responsible property, listed as
unseparated, is per-head rank ($\S$\ref{sec:open}).
"""),
# open question
(r"""\item \textbf{The input-side rank.} Trained and tested at $T=1024$ from matched
  initial weights, our shared $r=2$ solves $0/8$ seeds within $900$ epochs, a
  per-head $r=2$ $2/8$ and a shared $r=4$ $8/8$, while a rank-$2$ projection of a solved
  $r=4$ model scores $0.995$ and holds under training: a search deficit (measured for
  our shared $r=2$, inferred for the per-head one). The paper states a per-head
  $W_{\mathrm{in}}$ but not $W_{\mathrm{out}}$'s shape: the per-head arm is its literal
  reading, and with a full $W_{\mathrm{out}}$ its $r=2$ at two heads is our $r=4$.
  Nothing yet explains why training finds the rank-$2$ solution so rarely
  when it occupies a $2$-plane either way, and against the per-head arm, per-head
  rank, cross-head sharing and $W_{\mathrm{out}}$'s per-entry scale are not separated.""",
r"""\item \textbf{The input-side rank: which property is settled, why is not.} Trained and
  tested at $T=1024$ from matched initial weights, five bottlenecks split entirely by the
  rank of each head's map: at per-head rank $2$ our shared $r=2$ solves $0/8$ seeds within
  $900$ epochs, a per-head $r=2$ $2/8$ and a block-diagonal $r=4$ $2/8$; at per-head rank
  $4$ a shared $r=4$ and a per-head $r=4$ both solve $8/8$. Per-head rank fires (the two
  block-diagonal arms, Fisher and permutation $p=0.0070$); sharing one latent across heads
  ($8/8$ against $8/8$) and $W_{\mathrm{out}}$'s per-entry scale ($2/8$ against $2/8$) are
  unmeasured. A rank-$2$ projection of a solved $r=4$ model scores $0.9955$ and holds under
  training, so this is a search deficit (measured for our shared $r=2$). The paper states a
  per-head $W_{\mathrm{in}}$ but not $W_{\mathrm{out}}$'s shape: the per-head arm is its
  literal reading, and with a full $W_{\mathrm{out}}$ its $r=2$ at two heads is our $r=4$.
  What stays open is why training finds the rank-$2$ solution so rarely. A first probe
  (\texttt{LOOP\_RANK\_RESULTS.md}) gives partial support to the search account: looping the
  $r=2$ block $4\times$ at identical parameters raises accuracy from $0.894$ to $0.973$
  (permutation $p=0.0034$) but solves only $2/8$, and four real layers reach $0.990$ and
  $5/8$; neither reaches $r=4$'s loss regime on any seed, and the registered verdict is
  unmeasured. Scope: two heads, one length, one recipe."""),
]
apply("positional_review.tex", P)
# follow-up: "four real layers" -> "four real layers ($4\times$ the parameters)" in the open-question item
