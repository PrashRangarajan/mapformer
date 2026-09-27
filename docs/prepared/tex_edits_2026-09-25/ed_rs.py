import sys; sys.path.insert(0, sys.argv[1]); from rep import apply
P = [
(r"""shared budget, and at rank 2 per head --- the paper's design --- training rarely finds a torus solution that
exists at rank 2 ($0/8$ and $2/8$ seeds against $8/8$ at $r=4$).""",
r"""shared budget, and at rank 2 per head (our shared bottleneck, or a per-head one read literally from the paper)
training rarely finds a torus solution that exists at rank 2 ($0/8$ and $2/8$ seeds against $8/8$ at a shared $r=4$)."""),
(r"""our shared $r=2$ solves $0/8$
seeds, the paper's per-head $r=2$ $2/8$ and a shared $r=4$ $8/8$ (Fisher $p=0.0002$ and $0.007$); a rank-2
projection of each solved $r=4$ model scores $0.9955$ frozen and stays solved when trained onward ($7/8$, as does
an $r=4$ control). The rank-2 solution exists and is held; training rarely finds it. This implementation shares its
bottleneck across heads where the paper's is per head, so our $r=2$ has half the paper's latent dimensions; the
paper's own per-head $r=2$ fails too. Scope: one task and width, within 900 epochs, and $r=4$'s smaller initial
$W_{\mathrm{out}}$ is not separated from its rank.""",
r"""our shared $r=2$ solves $0/8$
seeds, a per-head $r=2$ $2/8$ and a shared $r=4$ $8/8$ (Fisher $p=0.0002$ and $0.007$), so $r=4$'s advantage is not
its initial draws; the two $r=2$ arms are not distinguishable ($p=0.47$). A rank-2
projection of each solved $r=4$ model scores $0.9955$ frozen and stays solved when trained onward ($7/8$, as does
an $r=4$ control). The rank-2 solution exists and is held (measured for our shared $r=2$, inferred for the per-head
one); training rarely finds it. Against the per-head $r=2$, $r=4$ has the same four latent dimensions and identical
initial $W_{\mathrm{in}}$, but per-head rank, cross-head sharing (confounded with it at two heads) and
$W_{\mathrm{out}}$'s per-entry scale are not separated; the initial angle-increment scale is matched. This
implementation shares its bottleneck across heads where the paper's $W_{\mathrm{in}}$ is per head, so our $r=2$ has
half the paper's latent dimensions; the paper does not state $W_{\mathrm{out}}$'s shape, the per-head arm is its
literal reading, and with a full $W_{\mathrm{out}}$ the paper's $r=2$ at two heads is our $r=4$. Scope: two heads,
one task, length and width, within 900 epochs."""),
(r"""\item \textbf{Our bottleneck is shared across heads; the paper's is per head.} The matched-length rank test
includes the paper's per-head design.""",
r"""\item \textbf{Our bottleneck is shared across heads; the paper's $W_{\mathrm{in}}$ is per head.} The paper does
not state $W_{\mathrm{out}}$'s shape; the matched-length rank test includes its literal per-head reading, and with a
full $W_{\mathrm{out}}$ the paper's $r=2$ at two heads is our $r=4$."""),
(r"""where the paper's own per-head $r=2$ usually fails to find a solution that exists.""",
r"""where our shared and a per-head $r=2$ usually fail to find a solution that exists."""),
]
apply("report/report_short.tex", P)
