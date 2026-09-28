import sys; sys.path.insert(0, sys.argv[1]); from rep import apply
P = [
# abstract
(r"""training rarely finds a torus solution that exists at rank 2 ($0/8$ and $2/8$ seeds against $8/8$ at a shared $r=4$). Explicit""",
r"""training rarely finds a torus solution that exists at rank 2 ($0/8$ and $2/8$ seeds against $8/8$ at per-head rank 4; the deciding property is per-head rank). Explicit"""),
(r"""\emph{Corrected 25 September 2026}: rank was described as a skewed basis that $r=4$ repairs; it is a search
deficit.
\end{abstract}""",
r"""\emph{Corrected 25 September 2026}: rank was described as a skewed basis that $r=4$ repairs; it is a search
deficit. \emph{Corrected 27 September 2026}: the responsible property of the rank bottleneck, previously
unseparated, is per-head rank; a torus loop batch and the Dyck-2 matched-depth result are added.
\end{abstract}"""),
# rank paragraph
(r"""seeds, a per-head $r=2$ $2/8$ and a shared $r=4$ $8/8$ (Fisher $p=0.0002$ and $0.007$), so $r=4$'s advantage is not
its initial draws; the two $r=2$ arms are not distinguishable ($p=0.47$).""",
r"""seeds, a per-head $r=2$ $2/8$ and a shared $r=4$ $8/8$ (Fisher $p=0.0002$ and $0.007$), so $r=4$'s advantage is not
its initial draws; the two $r=2$ arms are not distinguishable ($p=0.47$). Two more arms from the same initial
weights separate the cause: a block-diagonal $r=4$ (cross-head blocks held at zero, so rank 2 per head at $r=4$'s
$W_{\mathrm{out}}$ scale) solves $2/8$ ($0.948$) and a per-head $r=4$ $8/8$ ($0.999$). The split is entirely by
per-head rank, which fires (per-head against block-diagonal $r=4$, Fisher and permutation $p=0.0070$); sharing
($8/8$ against $8/8$) and $W_{\mathrm{out}}$'s per-entry scale ($2/8$ against $2/8$) are unmeasured."""),
(r"""one); training rarely finds it. Against the per-head $r=2$, $r=4$ has the same four latent dimensions and identical
initial $W_{\mathrm{in}}$, but per-head rank, cross-head sharing (confounded with it at two heads) and
$W_{\mathrm{out}}$'s per-entry scale are not separated; the initial angle-increment scale is matched. This""",
r"""one); training rarely finds it. The initial-angle-scale worry is answered rather than merely matched: the
block-diagonal arm starts at half the usual increment scale (std $0.208$), but our shared and the per-head $r=2$
start at normal scale ($0.335$, $0.328$) and fail the same way. Looping the $r=2$ block four times at identical
parameters partly recovers it ($0.894$ to $0.973$, permutation $p=0.0034$, $2/8$ solved) and four real layers
more ($0.990$, $5/8$), but neither reaches $r=4$'s loss regime on any seed (registered verdict unmeasured). This"""),
# recursion
(r"""Unpaired, the loop against three real layers is $+0.032$: not distinguishable from them, at a third of the parameters.""",
r"""Unpaired, the loop against three real layers is $+0.032$: not distinguishable from them, at a third of the parameters, on Match-Query; on the torus at $T=1024$ with $r=2$, four real layers beat the loop ($+0.017$, permutation $p=0.0009$; $5/8$ against $2/8$ solved)."""),
(r"""(single) batch. On the torus the loop's training-length gain is $+0.052$ raw but $+0.006$ at matched loss, with
$r(\text{loss},\text{accuracy})=-0.956$ at that length.""",
r"""(single) batch. On the torus at $T=128$ the loop's training-length gain is $+0.052$ raw but $+0.006$ at matched loss, with
$r(\text{loss},\text{accuracy})=-0.956$ at that length. At $T=1024$ with $r=2$, where there is headroom, the loop adds
$+0.079$ (permutation $p=0.0034$) and solves $2/8$ against $0/8$; the $0.05$ solved cutoff falls inside its spread
(at $0.08$ it solves $6/8$), so the registered verdict, unmeasured, is uncertain rather than negative."""),
# limitations
(r"""where one was run (code, rank) the extrapolation effect did not survive as such.""",
r"""where one was run (code, rank, Dyck depth) the extrapolation effect did not survive as such."""),
(r"""\item \textbf{Language modelling only at small scale.} The sequence results are replications and length effects;
the enwik8""",
r"""\item \textbf{Language modelling only at small scale.} The sequence results are replications and length effects;
the one that got a matched-distribution control, Dyck-2's 4-layer position effect at $3\times$ the training depth,
closed under it ($+0.002$ at ceiling), leaving depth substitution: at matched depth path integration adds
$+0.353$, $+0.130$, $+0.045$ and $+0.024$ at 1--4 layers, so one layer of it is worth about three of attention;
the enwik8"""),
(r"""full $W_{\mathrm{out}}$ the paper's $r=2$ at two heads is our $r=4$.
\end{itemize}""",
r"""full $W_{\mathrm{out}}$ the paper's $r=2$ at two heads is our $r=4$. Sharing itself was tested (per-head against
shared $r=4$, $8/8$ against $8/8$, permutation $p=0.59$) and is unmeasured; per-head rank is what fires.
\end{itemize}"""),
# conclusion
(r"""where our shared and a per-head $r=2$ usually fail to find a solution that exists.""",
r"""where every arm at rank 2 per head usually fails to find a solution that exists and every arm at rank 4 per head finds it."""),
(r"""(on the torus its gain vanishes at matched loss; Match-Query was
not loss-matched).""",
r"""(on the torus at $T=128$ its gain vanishes at matched loss; at $T=1024$, $r=2$ it partly recovers the rank deficit and
real depth does better; Match-Query was not loss-matched)."""),
(r"""basis, cited the paired loop interaction, and stated the Bach kernel account as a mechanism; all three are
corrected above.
""",
r"""basis, cited the paired loop interaction, and stated the Bach kernel account as a mechanism; all three are
corrected above.

\emph{Corrected 27 September 2026.} The 25 September text left the responsible property of the rank bottleneck
unseparated (it is per-head rank), scoped the torus loop to $T=128$ without saying so, and did not say that
Dyck-2's out-of-distribution depth effect closes at matched depth.
"""),
]
apply("report/report_short.tex", P)
