import sys; sys.path.insert(0, sys.argv[1]); from rep import apply
P = [
# ---- abstract: rank
(r"""$8/8$, although a rank-2 solution exists and is held under training; which of per-head rank, cross-head
sharing and $W_{\mathrm{out}}$'s scale is responsible is not separated.
% src: RECENCY_EM_RESULTS.md (F1); WARM_RESULTS.md (F9); SEARCH_RESULTS.md (F12); PAIRSPLIT_RESULTS.md (F19); RANK_MI_RESULTS.md; RANK_PROJ_RESULTS.md""",
r"""$8/8$, although a rank-2 solution exists and is held under training. Two more arms separate the
cause: it is the per-head rank of the content-to-angle map (a block-diagonal $r=4$, rank 2 per head, $2/8$;
a per-head $r=4$ $8/8$; Fisher and permutation $p=0.0070$), with cross-head sharing and $W_{\mathrm{out}}$'s
per-entry scale individually unmeasured.
% src: RECENCY_EM_RESULTS.md (F1); WARM_RESULTS.md (F9); SEARCH_RESULTS.md (F12); PAIRSPLIT_RESULTS.md (F19); RANK_MI_RESULTS.md; RANK_SEP_RESULTS.md; RANK_PROJ_RESULTS.md"""),
# ---- abstract: recursion, Dyck, stamp
(r"""trains unreliably (on Match-Query, in an unpaired analysis); on the torus its gain vanishes at matched
loss. Transfer across a change of structure is not tested, and the sequence-modelling results (Dyck-2,
Bach, Indirect Indexing, enwik8) are replications and length effects at small scale, not language-model
quality. \emph{Corrected 25 September 2026}: rank was described as a skewed basis that $r=4$ repairs; it
is a search deficit (Section~\ref{sec:c3-rank}).""",
r"""trains unreliably (on Match-Query, in an unpaired analysis); on the torus at $T=128$ its gain vanishes at matched
loss, and at $T=1024$ it partly recovers the rank-2 deficit, where four real layers do better. Transfer across a change of structure is not tested, and the sequence-modelling results (Dyck-2,
Bach, Indirect Indexing, enwik8) are replications and length effects at small scale, not language-model
quality; on Dyck-2 the depth-extrapolation effect closes when models are trained at the test depth, leaving
depth substitution (one layer of path integration is worth about three of attention). \emph{Corrected 25 September 2026}: rank was described as a skewed basis that $r=4$ repairs; it
is a search deficit (Section~\ref{sec:c3-rank}). \emph{Corrected 27 September 2026}: the responsible property
of the rank bottleneck, previously unseparated, is per-head rank; the torus loop and Dyck matched-depth
batches are added (Sections~\ref{sec:c5-loop} and~\ref{sec:dyck})."""),
# ---- contributions table
(r"""At rank 2 per head (ours shared, and a per-head reading of the paper's design) training rarely finds a torus solution that exists and is held at rank 2; $r=4$ finds it on every seed & ours & LieRE (different axis) \\""",
r"""At rank 2 per head (ours shared, a per-head reading of the paper's design, and a block-diagonal $r=4$) training rarely finds a torus solution that exists and is held at rank 2; at rank 4 per head, shared or per head, it is found on every seed; the deciding property is per-head rank & ours & LieRE (different axis) \\"""),
(r"""Recursion adds accuracy to path integration on one task (unpaired; the super-additive paired estimate is single-batch and withdrawn); recursion's torus gain is convergence & ours & recursion for depth: Mixture-of-Recursions \\""",
r"""Recursion adds accuracy to path integration on one task (unpaired; the super-additive paired estimate is single-batch and withdrawn); recursion's torus gain is convergence at $T=128$; at $T=1024$, $r=2$ it partly recovers the rank deficit, and four real layers beat it & ours & recursion for depth: Mixture-of-Recursions \\"""),
# ---- sec:c3-rank body
(r"""$+0.104$ and $+0.112$, permutation $p=0.0005$ and $0.007$), so $r=4$'s advantage is not its initial draws.
The per-head arm and $r=4$ have the same four latent dimensions and identical initial $W_{\mathrm{in}}$, but
differ in three ways at once: the rank of each head's content-to-angle map, whether the heads read a shared
latent (perfectly confounded with it at two heads), and $W_{\mathrm{out}}$'s per-entry scale. Which is
responsible is not separated; the initial scale of the angle increments is matched. The two $r=2$ arms are
not distinguishable""",
r"""$+0.104$ and $+0.112$, permutation $p=0.0005$ and $0.007$), so $r=4$'s advantage is not its initial draws.
The per-head arm and $r=4$ have the same four latent dimensions and identical initial $W_{\mathrm{in}}$, but
differ in three ways at once: the rank of each head's content-to-angle map, whether the heads read a shared
latent, and $W_{\mathrm{out}}$'s per-entry scale. A second batch built the same way (pre-registered,
\texttt{RANK\_SEP\_PREREG.md}) separates them with two more arms: a block-diagonal $r=4$ (the shared $r=4$
with its cross-head blocks held at zero, so rank 2 per head at $r=4$'s $W_{\mathrm{out}}$ scale) solves $2/8$
($0.948$), and a per-head $r=4$ (rank 4 per head, separate latents) solves $8/8$ ($0.999$). The split is
entirely by per-head rank: every arm at rank 2 per head solves 0--2 of 8, every arm at rank 4 solves 8 of 8,
and total latent size does not track it (the per-head $r=2$ and the shared $r=4$ both have four dimensions and
land on opposite sides). Per-head rank fires (per-head against block-diagonal $r=4$, both unshared: Fisher and
permutation $p=0.0070$, Holm $0.028$). Sharing (per-head against shared $r=4$: $8/8$ against $8/8$,
permutation $p=0.59$) and $W_{\mathrm{out}}$'s per-entry scale (block-diagonal $r=4$ against per-head $r=2$:
$2/8$ against $2/8$, permutation $p=0.24$) are unmeasured. The initial-angle-scale worry is answered rather
than merely matched: zeroing the cross-head blocks halves the block-diagonal arm's initial increment (std
$0.208$), but our shared and the per-head $r=2$ start at normal scale ($0.335$, $0.328$) and fail the same
way, so low initial scale is not necessary for failure. A reproduction run inside the batch matched the
stored shared-$r=4$ seed on all 900 per-epoch losses. The two original $r=2$ arms are
not distinguishable"""),
(r"""% src: RANK_MI_RESULTS.md, RANK_MI_ANALYSIS.txt; RANK_PROJ_RESULTS.md (S1 STABLE); RANK_MATCHED_RESULTS.md (continuation)""",
r"""% src: RANK_MI_RESULTS.md, RANK_MI_ANALYSIS.txt; RANK_SEP_RESULTS.md, RANK_SEP_PREREG.md; RANK_PROJ_RESULTS.md (S1 STABLE); RANK_MATCHED_RESULTS.md (continuation)"""),
# ---- table
(r"""Floor 0.506. Reading: at the same four latent dimensions and identical initial $W_{\mathrm{in}}$, the shared $r=4$ finds the solution and the per-head $r=2$ rarely does; per-head rank, cross-head sharing and $W_{\mathrm{out}}$'s per-entry scale differ together between them and are not separated. A rank-2 solution exists (0.9955 frozen) and is held (measured for the shared $r=2$).}""",
r"""Floor 0.506. The first three arms are one batch, the last two a second batch built the same way (reproduction exact); the second batch reports means only. Reading: the split is entirely by rank per head; per-head rank fires (per-head against block-diagonal $r=4$, Fisher and permutation $p=0.0070$), while sharing (per-head against shared $r=4$) and $W_{\mathrm{out}}$'s per-entry scale (block-diagonal $r=4$ against per-head $r=2$) are unmeasured. A rank-2 solution exists (0.9955 frozen) and is held (measured for the shared $r=2$).}"""),
(r"""\begin{tabular}{@{}lcccc@{}}
\toprule
arm & latent dims & rank per head & solved & accuracy $T=1024$ \\
\midrule
our shared $r=2$ & 2 & 2 & $0/8$ & $0.894\pm0.070$ \\
per-head $r=2$ (the paper, read literally) & 4 & 2 & $2/8$ & $0.885\pm0.131$ \\
shared $r=4$ & 4 & 4 & $8/8$ & $0.998\pm0.005$ \\
\bottomrule
\end{tabular}
% src: RANK_MI_RESULTS.md""",
r"""\begin{tabular}{@{}lccccc@{}}
\toprule
arm & latent dims & shared & rank per head & solved & accuracy $T=1024$ \\
\midrule
our shared $r=2$ & 2 & yes & 2 & $0/8$ & $0.894\pm0.070$ \\
per-head $r=2$ (the paper, read literally) & 4 & no & 2 & $2/8$ & $0.885\pm0.131$ \\
block-diagonal $r=4$ (\arm{Vanilla\_r4mibd}) & 4 & no & 2 & $2/8$ & $0.948$ \\
shared $r=4$ & 4 & yes & 4 & $8/8$ & $0.998\pm0.005$ \\
per-head $r=4$ (\arm{Vanilla\_r4ph}) & 8 & no & 4 & $8/8$ & $0.999$ \\
\bottomrule
\end{tabular}
% src: RANK_MI_RESULTS.md; RANK_SEP_RESULTS.md"""),
# ---- scope bullet
(r"""heads), one recipe. $r=4$'s $W_{\mathrm{out}}$ starts at a smaller per-entry scale (bound $0.5$ against $0.707$; the initial
angle-increment scale is matched), which is not separated from its rank or from cross-head sharing.""",
r"""heads), one recipe. $r=4$'s $W_{\mathrm{out}}$ starts at a smaller per-entry scale (bound $0.5$ against $0.707$). The
separation batch tested that scale (block-diagonal $r=4$ against per-head $r=2$, both rank 2 per head): $2/8$
against $2/8$, unmeasured, so it is not shown to matter; per-head rank is the factor that fires. Untested: rank
3 per head, more heads, and whether a rank-2 arm can be brought to the solution by another route (init,
schedule, curriculum; for depth and looping see Section~\ref{sec:c5-loop})."""),
# ---- recursion summary paragraph
(r"""is associated with, not separated from, more reliable training. On the torus the loop's training-length gain
is $+0.052$ raw but $+0.006$ at matched loss. An explicit""",
r"""is associated with, not separated from, more reliable training. On the torus at $T=128$ the loop's training-length gain
is $+0.052$ raw but $+0.006$ at matched loss; at $T=1024$ with $r=2$ it adds $+0.079$ (permutation $p=0.0034$),
and four real layers add more. An explicit"""),
# ---- recursion headroom paragraph
(r"""\paragraph{Where there is headroom.} On the torus a one-layer path-integrated model is already near
ceiling, so a loop has nothing measurable to add there. Match-Query""",
r"""\paragraph{Where there is headroom.} On the torus at $T=128$ with $r=4$ a one-layer path-integrated model is already near
ceiling, so a loop has little to add there; at $T=1024$ with $r=2$ there is headroom, and the loop adds
measurably (below). Match-Query"""),
(r"""the loop against three real layers is $+0.032$ (se $0.105$, t $0.30$): not distinguishable from them, at a
third of the parameters.""",
r"""the loop against three real layers is $+0.032$ (se $0.105$, t $0.30$): not distinguishable from them, at a
third of the parameters, on this task (on the torus at $T=1024$, $r=2$, four real layers beat the loop; below)."""),
# ---- torus paragraph: add T=1024 r=2
(r"""training length (Table~\ref{tab:l15loop}) is a separate matter (Appendix~\ref{app:loops}).
% src: L15_LOOP_2X2.md (C24, B2); RECIPE_POWER.md (C25)""",
r"""training length (Table~\ref{tab:l15loop}) is a separate matter (Appendix~\ref{app:loops}).
% src: L15_LOOP_2X2.md (C24, B2); RECIPE_POWER.md (C25)

\paragraph{On the torus at $T=1024$, rank 2: the loop partly recovers the rank deficit.} A pre-registered
batch asked whether a search aid recovers rank 2 (Section~\ref{sec:c3-rank}). Trained and tested at $T=1024$
for 900 epochs, eight seeds, one batch: our shared $r=2$ at one layer (204,373 parameters) solves $0/8$ at
accuracy $0.894$; the same block looped four times, bit-identical in parameters and initialisation, solves
$2/8$ at $0.973$ ($+0.079$, 95\% CI $[+0.026,+0.131]$, permutation $p=0.0034$; solved count Fisher $p=0.47$);
four real $r=2$ layers (799,189 parameters) solve $5/8$ at $0.990$ (Fisher $0.0256$, permutation $0.0012$);
$r=4$ at one layer solves $8/8$ at $0.998$. Four real layers beat the matched-parameter loop ($+0.017$,
permutation $p=0.0009$; $5/8$ against $2/8$, Fisher $0.31$). Final losses form three regimes (one-layer $r=2$
$0.05$--$0.68$, the two aids $0.011$--$0.13$, $r=4$ $0.002$--$0.010$): extra search moves rank 2 out of its
regime but into $r=4$'s on no seed. The registered verdict is unmeasured (the loop had to solve at least
$4/8$), and it is uncertain rather than negative: the $0.05$ solved cutoff, calibrated on a 300-epoch batch,
falls inside the aids' spread, and at $0.08$ the loop solves $6/8$ (Fisher $0.041$ against one layer). Since
parameters plus compute beat compute alone here, this is not ``search at constant capacity''. Scope: one task,
length and recipe; one loop count; several looped and deep runs were still descending.
% src: LOOP_RANK_RESULTS.md, LOOP_RANK_PREREG.md"""),
# ---- prior art
(r"""What this report adds is the unpaired loop gain with path integration on one task, and the
decomposition of the torus gain into convergence.""",
r"""What this report adds is the unpaired loop gain with path integration on one task, the
decomposition of the $T=128$ torus gain into convergence, and a torus case at $T=1024$, $r=2$ where the loop
partly recovers a search deficit but real depth beats it (its verdict threshold-sensitive)."""),
# ---- length section
(r"""(\texttt{CODE\_RESULTS.md}), and on the torus rank's matched-length effect is on search. The sign, the
filter, the forget gate, PoPE's wrapping and rotation/allocentric recording have no matched-length control.
% src: CODE_RESULTS.md; RANK_MI_RESULTS.md;""",
r"""(\texttt{CODE\_RESULTS.md}); on the torus rank's matched-length effect is on search; and on Dyck-2 the clearest
instance, on the depth axis: the 4-layer position effect trained at depth 4 and scored at depth 12 closes to
$+0.002$ at ceiling when models are trained at depth 12 (Section~\ref{sec:dyck}). The sign, the
filter, the forget gate, PoPE's wrapping and rotation/allocentric recording have no matched-length control.
% src: CODE_RESULTS.md; RANK_MI_RESULTS.md; DYCK_MDEPTH_RESULTS.md;"""),
# ---- Dyck table 1 caption and headers
(r"""\caption{Dyck-2 F1 (ours, 8 seeds; paper value in brackets). Every model fits the training cell; the
split appears only out of distribution. The \emph{n-gram floor}""",
r"""\caption{Dyck-2 F1 (ours, 8 seeds; paper value in brackets), trained at $L32\,D4$. Every model fits the training cell; the
split appears only out of distribution: the $D12$ columns are $3\times$ the training depth (depth extrapolation). The \emph{n-gram floor}"""),
(r"""arm & $L32\,D4$ (trained) & $L128\,D4$ & $L32\,D12$ & $L128\,D12$ \\
\midrule
MapPoPE-1L & \textbf{0.988} & \textbf{0.923}""",
r"""arm & $L32\,D4$ (trained) & $L128\,D4$ & $L32\,D12$ (depth-OOD) & $L128\,D12$ (depth-OOD) \\
\midrule
MapPoPE-1L & \textbf{0.988} & \textbf{0.923}"""),
# ---- Dyck table 2
(r"""sequence is correct; chance $0.000$.}""",
r"""sequence is correct; chance $0.000$. Trained at $L32\,D4$: the $L128\,D12$ columns are $3\times$ the training depth and $4\times$ its length.}"""),
(r"""arm & $L32\,D4$ & $L128\,D12$ & $L32\,D4$ & $L128\,D12$ \\
\midrule
MapPoPE-1L & \textbf{0.994}""",
r"""arm & $L32\,D4$ & $L128\,D12$ (OOD) & $L32\,D4$ & $L128\,D12$ (OOD) \\
\midrule
MapPoPE-1L & \textbf{0.994}"""),
# ---- three things follow
(r"""Three things follow. First, the paper's real claim survives and sharpens: a \emph{one-layer} path-integrated
model learns the stack at the training distribution ($0.98$--$0.99$) where one-layer index models do not
($0.63$--$0.65$), and index models need a second layer to learn it, exactly as the transformer theory predicts
--- a distinction""",
r"""Three things follow. First, the one-layer half of the paper's claim survives: a \emph{one-layer} path-integrated
model learns the stack at the training distribution ($0.98$--$0.99$) where one-layer index models do not
($0.63$--$0.65$), and index models need more layers to learn it. That is a claim about parameter efficiency
(depth substitution), not about something depth cannot buy; the matched-depth batch below measures it
--- a distinction"""),
# ---- new matched-depth paragraph after Reach
(r"""(\texttt{DYCK\_FAR\_PROBE.md}; that evaluation was run but uncommitted when this section was first written, and is committed now).
% src: DYCK_STACK_PROBE.md""",
r"""(\texttt{DYCK\_FAR\_PROBE.md}; that evaluation was run but uncommitted when this section was first written, and is committed now).
% src: DYCK_STACK_PROBE.md

\paragraph{At matched depth the depth-extrapolation effect closes; depth substitution survives.} A depth
ladder of the same arms (RoPE, PoPE, MapWM, MapPoPE at 1--4 layers, trained at $L32\,D4$) showed a position
effect at $L32\,D12$ at every depth, including 4 layers; but $D12$ is $3\times$ the training depth. A
pre-registered batch trained every arm at $L32\,D12$ and scored it there, on bracket-closing memory averaged over
distance (chance $0.500$, best trivial predictor $0.594$), eight seeds, one batch, with two of the ladder's 4-layer
runs reproduced exactly inside it. At 4 layers and $3\times$ budget every arm is at ceiling (index
$0.997$--$0.998$, path-integrated $1.000$): the position effect is $+0.002$ ($8/8$), a ceiling difference far
below the pre-registered $0.05$, and the verdict is that the effect \emph{closes}. The ladder's 4-layer effect
was depth extrapolation. What survives at matched depth and $1\times$ budget is depth substitution: path
integration minus index position is $+0.353$, $+0.130$, $+0.045$ and $+0.024$ at 1, 2, 3 and 4 layers (each
$8/8$), so one layer of path integration is worth roughly three layers of attention. The earlier readings that
the index arms plateau and that depth closes part of the gap and then stops are dead: at $1\times$ the index
arms keep climbing from 3 to 4 layers ($+0.018$ RoPE, within its \mde; $+0.025$ PoPE), the $1\times$ budget limits them ($3\times$
minus $1\times$ is $+0.021$, $8/8$), and at $3\times$ they reach ceiling by 4 layers. Training over a mixture of
depths (4--12) keeps a real effect at 4 layers, $+0.110$ at $L32\,D12$ ($8/8$) and $+0.043$ at $L32\,D4$; at
$L128\,D12$ it is unmeasured. Index base 32 against base 10000 is within the \mde\ at 4 layers ($+0.001$,
$-0.001$).
% src: DYCK_MDEPTH_RESULTS.md, DYCK_MDEPTH_PREREG.md, DYCK_LADDER_RESULTS.md"""),
# ---- limitations: language
(r"""\item \textbf{Language modelling only at small scale.} The sequence results here are replications and
length effects (Dyck-2, Bach, Indirect Indexing); the enwik8""",
r"""\item \textbf{Language modelling only at small scale.} The sequence results here are replications and
length effects (Dyck-2, Bach, Indirect Indexing); the one sequence result that got a matched-distribution
control, Dyck-2's 4-layer position effect at $3\times$ the training depth, closed under it, leaving depth
substitution (Section~\ref{sec:dyck}); the enwik8"""),
# ---- limitations: OOD length
(r"""where one was run (code, rank) the extrapolation effect did not survive as such.
\item \textbf{Recipe dependence""",
r"""where one was run (code, rank, Dyck depth) the extrapolation effect did not survive as such.
\item \textbf{Recipe dependence"""),
# ---- limitations: bottleneck
(r"""matched-length rank test includes its literal per-head reading, and with a full $W_{\mathrm{out}}$ the paper's
$r=2$ at two heads is our $r=4$.
\item \textbf{Holdability""",
r"""matched-length rank test includes its literal per-head reading, and with a full $W_{\mathrm{out}}$ the paper's
$r=2$ at two heads is our $r=4$. Sharing itself was tested (per-head against shared $r=4$: $8/8$ against $8/8$,
Fisher $p=1.00$, permutation $p=0.59$) and is unmeasured; the rank of each head's map is what fires.
\item \textbf{Holdability"""),
# ---- conclusion
(r"""the torus, where our shared and a per-head $r=2$ usually fail to find a solution that exists.""",
r"""the torus, where every arm at rank 2 per head usually fails to find a solution that exists and every arm
at rank 4 per head finds it."""),
(r"""association with more reliable optimisation (on the torus its gain vanishes at matched loss; Match-Query was not
loss-matched).""",
r"""association with more reliable optimisation (on the torus at $T=128$ its gain vanishes at matched loss; at
$T=1024$, $r=2$ it partly recovers the rank deficit and real depth does better; Match-Query was not
loss-matched)."""),
(r"""two-part account of the Bach collapse; all four are corrected above.
""",
r"""two-part account of the Bach collapse; all four are corrected above.

\emph{Corrected 27 September 2026.} The 25 September text left the responsible property of the rank
bottleneck unseparated (it is per-head rank; sharing and $W_{\mathrm{out}}$'s scale are unmeasured), said a
loop has nothing to add on the torus (at $T=1024$, $r=2$ it does, and real depth does more), and did not say
that Dyck-2's out-of-distribution depth effect closes at matched depth.
"""),
]
apply("report/report.tex", P)
# follow-up after the end-to-end read (applied separately)
# caption "one batch" -> "two batches"; column "shared" -> "heads share a latent"; \date line -> "corrected 25 and 27 September 2026"
