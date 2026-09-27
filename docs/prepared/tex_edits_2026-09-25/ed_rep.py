import sys; sys.path.insert(0, sys.argv[1]); from rep import apply
P = [
# R1 intro bullet
(r"""both detectable. The paper's rank-2 bottleneck shows the
same pattern on the torus: trained and tested at $T=1024$ from matched initial weights, our shared $r=2$
finds the solution on $0/8$ seeds within 900 epochs, the paper's per-head $r=2$ on $2/8$ and $r=4$ on
$8/8$, although a rank-2 solution exists and is held under training.""",
r"""both detectable. The rank-2 bottleneck shows the
same pattern on the torus: trained and tested at $T=1024$ from matched initial weights, our shared $r=2$
finds the solution on $0/8$ seeds within 900 epochs, a per-head $r=2$ on $2/8$ and a shared $r=4$ on
$8/8$, although a rank-2 solution exists and is held under training; which of per-head rank, cross-head
sharing and $W_{\mathrm{out}}$'s scale is responsible is not separated."""),
# R2 contributions table
(r"""At rank 2 per head (ours shared, and the paper's per-head design) training rarely""",
r"""At rank 2 per head (ours shared, and a per-head reading of the paper's design) training rarely"""),
# R3 claim 4 opening
(r"""training, yet our shared $r=2$ finds it on $0/8$ seeds within 900 epochs, the paper's per-head $r=2$ on
$2/8$, and $r=4$ on $8/8$.""",
r"""training, yet our shared $r=2$ finds it on $0/8$ seeds within 900 epochs, a per-head $r=2$ on
$2/8$, and a shared $r=4$ on $8/8$."""),
# R4 main paragraph, first half
(r"""our shared $r=2$ solves $0/8$ seeds, the
paper's per-head $r=2$ $2/8$ and a shared $r=4$ $8/8$ (Fisher $p=0.0002$ and $0.007$ against $r=4$; accuracy
$+0.104$ and $+0.112$, permutation $p=0.0005$ and $0.007$). The per-head arm and $r=4$ have the same four
latent dimensions, so what decides the outcome is the rank of each head's content-to-angle map. The
solution exists""",
r"""our shared $r=2$ solves $0/8$ seeds, a
per-head $r=2$ $2/8$ and a shared $r=4$ $8/8$ (Fisher $p=0.0002$ and $0.007$ against $r=4$; accuracy
$+0.104$ and $+0.112$, permutation $p=0.0005$ and $0.007$), so $r=4$'s advantage is not its initial draws.
The per-head arm and $r=4$ have the same four latent dimensions and identical initial $W_{\mathrm{in}}$, but
differ in three ways at once: the rank of each head's content-to-angle map, whether the heads read a shared
latent (perfectly confounded with it at two heads), and $W_{\mathrm{out}}$'s per-entry scale. Which is
responsible is not separated; the initial scale of the angle increments is matched. The two $r=2$ arms are
not distinguishable ($2/8$ against $0/8$, Fisher $p=0.47$; accuracy 95\% CI $[-0.123,+0.100]$): unmeasured. The
solution exists"""),
# R4 main paragraph, second half
(r"""while the paper's bottleneck is
per head ($W_{\mathrm{in}}\in\R^{d\times n_h\times r}$), so at two heads our $r=2$ has half the paper's
latent dimensions. The per-head arm is the paper's design, and it has the search problem too: ``use $r=4$''
is advice about MapFormer's design, not about this reimplementation.""",
r"""while the paper's $W_{\mathrm{in}}$ is
per head ($W_{\mathrm{in}}\in\R^{d\times n_h\times r}$), so at two heads our $r=2$ has half the paper's
latent dimensions. The paper does not state $W_{\mathrm{out}}$'s shape: the per-head arm is its literal
reading, and with a full $W_{\mathrm{out}}$ the paper's $r=2$ at two heads is our shared $r=4$. Existence and
holding were measured for our shared $r=2$; for the per-head arm they are inferred."""),
# R5 caption
(r"""Reading: the per-head rank, not the number of latent dimensions, lines up with whether training finds the solution; a rank-2 solution exists (0.9955 frozen) and is held.}""",
r"""Reading: at the same four latent dimensions and identical initial $W_{\mathrm{in}}$, the shared $r=4$ finds the solution and the per-head $r=2$ rarely does; per-head rank, cross-head sharing and $W_{\mathrm{out}}$'s per-entry scale differ together between them and are not separated. A rank-2 solution exists (0.9955 frozen) and is held (measured for the shared $r=2$).}"""),
# R6 table row
(r"""the paper's per-head $r=2$ & 4 & 2 & $2/8$ & $0.885\pm0.131$ \\""",
r"""per-head $r=2$ (the paper, read literally) & 4 & 2 & $2/8$ & $0.885\pm0.131$ \\"""),
# R7 scope bullet
(r"""$r=4$'s $W_{\mathrm{out}}$ starts at a smaller scale (bound $0.5$ against $0.707$), which
is not separated from its rank.""",
r"""$r=4$'s $W_{\mathrm{out}}$ starts at a smaller per-entry scale (bound $0.5$ against $0.707$; the initial
angle-increment scale is matched), which is not separated from its rank or from cross-head sharing."""),
# R8 limitation
(r"""\item \textbf{Our bottleneck is shared across heads; the paper's is per head.} At two heads our $r=2$
has half the paper's latent dimensions; the matched-length rank test includes the paper's per-head design.""",
r"""\item \textbf{Our bottleneck is shared across heads; the paper's $W_{\mathrm{in}}$ is per head.} At two heads
our $r=2$ has half the paper's latent dimensions. The paper does not state $W_{\mathrm{out}}$'s shape; the
matched-length rank test includes its literal per-head reading, and with a full $W_{\mathrm{out}}$ the paper's
$r=2$ at two heads is our $r=4$."""),
# R9 conclusion
(r"""the torus, where the paper's own per-head $r=2$ usually fails to find a solution that exists.""",
r"""the torus, where our shared and a per-head $r=2$ usually fail to find a solution that exists."""),
]
apply("report/report.tex", P)
