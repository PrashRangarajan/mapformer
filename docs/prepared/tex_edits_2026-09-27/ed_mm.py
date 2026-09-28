import sys; sys.path.insert(0, sys.argv[1]); from rep import apply
P = [
# abstract stamp
(r"""are the $16$-epoch recipe, converged $+0.243$ with the encoding detectable at every
length.
\end{abstract}""",
r"""are the $16$-epoch recipe, converged $+0.243$ with the encoding detectable at every
length.

\smallskip\noindent\emph{Corrected 2026-09-27.} (iv) Rank separation
(\texttt{RANK\_SEP\_RESULTS.md}): the property of the bottleneck the 2026-09-25 text left
unseparated is the per-head rank of the content-to-angle map; cross-head sharing and
$W_{\mathrm{out}}$'s per-entry scale are individually unmeasured. (v) Recursion on the torus
(\texttt{LOOP\_RANK\_RESULTS.md}): at $r=2$, four real layers beat the matched-parameter loop.
\end{abstract}"""),
# quote box
(r"""\textbf{Tested at matched length (2026-09-24), and the heading above holds.} Trained""",
r"""\textbf{Tested at matched length (2026-09-24, separated 2026-09-25), and the heading above holds.} Trained"""),
(r"""$W_{\mathrm{in}}$, but which of per-head rank, cross-head sharing (perfectly
confounded with it at two heads) and $W_{\mathrm{out}}$'s per-entry scale (bound
$0.5$ against $0.707$) is responsible is not separated; the initial scale of the
angle increments is matched. The two $r=2$ arms are not distinguishable (Fisher
$p=0.47$).""",
r"""$W_{\mathrm{in}}$, but differ in per-head rank, cross-head sharing and
$W_{\mathrm{out}}$'s per-entry scale (bound $0.5$ against $0.707$). Two more arms from
the same initial weights separate them: a block-diagonal $r=4$ (rank $2$ per head, at
$r=4$'s $W_{\mathrm{out}}$ scale) solves $2/8$ ($0.948$) and a per-head $r=4$ $8/8$
($0.999$). The split is entirely by per-head rank, which fires (per-head against
block-diagonal $r=4$, Fisher and permutation $p=0.0070$); sharing ($8/8$ against $8/8$)
and $W_{\mathrm{out}}$'s scale ($2/8$ against $2/8$) are unmeasured. Low initial angle
scale is not necessary for failure: the block-diagonal arm starts at std $0.208$, but our
shared and the per-head $r=2$ start at $0.335$ and $0.328$ and fail the same way. The two
original $r=2$ arms are not distinguishable (Fisher $p=0.47$)."""),
(r"""$r=4$. Budget-scoped (within $900$ epochs), two heads, one task, length and width
(\texttt{RANK\_MI\_RESULTS.md}, \texttt{RANK\_PROJ\_RESULTS.md}).""",
r"""$r=4$. Budget-scoped (within $900$ epochs), two heads, one task, length and width
(\texttt{RANK\_MI\_RESULTS.md}, \texttt{RANK\_SEP\_RESULTS.md}, \texttt{RANK\_PROJ\_RESULTS.md}).
Looping the $r=2$ block $4\times$ at identical parameters partly recovers it ($0.973$,
$2/8$) and four real layers more ($0.990$, $5/8$), but neither enters $r=4$'s loss regime
on any seed (\texttt{LOOP\_RANK\_RESULTS.md}; registered verdict unmeasured)."""),
# prior art A6 item
(r"""  ($0/8$ shared $r=2$, $2/8$ a per-head $r=2$, $8/8$ a shared $r=4$; per-head rank,
  sharing and $W_{\mathrm{out}}$'s scale not separated) while a rank-$2$""",
r"""  ($0/8$ shared $r=2$, $2/8$ a per-head $r=2$ and $2/8$ a block-diagonal $r=4$, against
  $8/8$ for a shared and a per-head $r=4$: it is per-head rank, $p=0.0070$, with sharing and
  $W_{\mathrm{out}}$'s scale unmeasured) while a rank-$2$"""),
# (ii') heading and contrast
(r"""\emph{(ii$'$) A6 is thinner than a blank column suggests, but still open.}""",
r"""\emph{(ii$'$) A6 is thinner than a blank column suggests; one torus measurement now sits in it.}"""),
(r"""and holds, and training finds it on $0/8$ seeds (ours) and $2/8$ (a per-head $r=2$,
the paper's literal reading) against $8/8$ at a shared $r=4$.""",
r"""and holds, and training finds it on $0/8$ seeds (ours), $2/8$ (a per-head $r=2$,
the paper's literal reading) and $2/8$ (a block-diagonal $r=4$) against $8/8$ at a shared
and at a per-head $r=4$. The deciding property is the per-head rank of the map
($p=0.0070$); sharing and $W_{\mathrm{out}}$'s scale are unmeasured. Scope: two heads, one
task, one length."""),
# capability table
(r"""\emph{absolute location in a bounded map} & signed, \textbf{and} a shared $r\ge4$ to be found reliably (two heads tested) & MapFormer at $r\ge4$. At $r=2$ (ours shared or per head) training usually fails to find the torus solution within budget, though a rank-$2$ solution exists: search, not a skewed basis or capacity \\[3pt]""",
r"""\emph{absolute location in a bounded map} & signed, \textbf{and} per-head rank $\ge4$ to be found reliably (two heads tested) & MapFormer at per-head rank $\ge4$, shared or per head. At per-head rank $2$ (ours shared, per head, or block-diagonal $r=4$) training usually fails to find the torus solution within budget --- $0/8$, $2/8$, $2/8$ against $8/8$, $8/8$ at per-head rank $4$, $T{=}1024$ --- though a rank-$2$ solution exists: search, not a skewed basis or capacity \\[3pt]"""),
# anchors
(r"""trained at $1024$, solved $8/8$ against $0/8$ \\[2pt]""",
r"""trained at $1024$, solved $0/8$, $2/8$, $2/8$ at per-head rank $2$ against $8/8$, $8/8$ at per-head rank $4$ \\[2pt]"""),
(r"""Recursion        & block reuse & $0$ & $+0.056$ parity ($16/16$); matches or beats $3\times$ params \\[2pt]""",
r"""Recursion        & block reuse & $0$ & $+0.056$ parity ($16/16$); matches $3\times$ params on Match-Query. Torus $T{=}1024$, $r{=}2$: $4$ real layers beat the loop ($+0.017$, perm.\ $p=0.0009$; $5/8$ against $2/8$ solved) \\[2pt]"""),
]
apply("mapformer_math.tex", P)
