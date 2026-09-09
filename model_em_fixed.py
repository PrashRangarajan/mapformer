"""MapFormer-EM corrected to the paper's eq. 3 --- single origin vector p_0.

RETRACTION of this file's earlier contents
------------------------------------------
An earlier version of this module claimed our EMTransformerLayer had the
Hadamard product in the wrong place, and "fixed" it by softmaxing each branch
before the product. That was WRONG. Reading the paper directly, eq. 3 is

    Att(Q_g, K_g, V) = softmax( A_X  o  A_P ) V ,
    A_X = Q_X K_X^T / sqrt(d),      A_P = P . P^T / sqrt(d)

i.e. RAW scaled scores, softmax applied AFTER the elementwise product. Our
ORIGINAL EMTransformerLayer already does exactly this. The earlier "fix" was
built on a summariser's paraphrase, trained consistently worse on the paper's
own task (loss ~1.19 vs the original reaching 0.083), and is withdrawn.

CORRECTION (2026-08-09): this is an ABLATION, not a bug fix
-----------------------------------------------------------
An earlier version of this docstring called separate q0_pos/k0_pos "the REAL
deviation" from the paper. That is ALSO wrong. Appendix A.4, verbatim:

    "compared to TEM and TEM-t that use a single neural population p_t to
     encode position, our MapFormers use two separate initial vectors k0p and
     q0p. This distinction is optional and mimics the formalism of EM-SSMs
     defined in eq. 12, meaning that we could set k0p = q0p = p0 without loss
     of generality. However, we suspect this separation to be beneficial
     because it would create sparser attention values."

So separate q0/k0 IS the paper's implementation; our original code was faithful.
The main text's P* = [p_0, ..., p_0] is the simplified presentation.

What this module therefore is: an ABLATION of an optional design choice that the
paper states as a SUSPICION and never measures. Our measurement contradicts it
(paper-task held-out revisit accuracy, n=3 seeds, same training batch):

    separate q0/k0  0.898 +/- 0.108   (seeds 0.778 / 0.931 / 0.986)
    single p_0      0.987 +/- 0.012   (seeds 0.9995 / 0.975 / 0.986)

The separation does not create useful sparsity at this scale; it destabilises
training. Both configurations must stay in the results -- the separate version
is the paper's architecture, the single-p_0 version is our ablation of it.

The geometric reason the single-p_0 form is better behaved:

    P := R_{theta_PI} P* ,   P* = [p_0, ..., p_0] ,   A_P = P . P^T

so A_P[i,j] = p_0^T R(theta_j - theta_i) p_0 is an AUTOCORRELATION kernel:
necessarily maximal and positive at zero displacement -- a proper "same place"
detector.

With separate q0_pos and k0_pos,
A_P[i,j] = sum_c |q0_c||k0_c| cos(dtheta_c + psi_c) with psi_c the phase offset
between them, so the peak is displaced and A_P[i,i] can be NEGATIVE. Measured on
a trained VanillaEM:

    zero-displacement A_P   ours: mean -0.0091, 50% negative, row-max on 16% of rows
                            paper (single p_0): mean +0.0442, 0% negative, row-max on 100%

q0_pos and k0_pos had learned to be nearly anti-aligned (cos = -0.73 on head 0),
inverting the position kernel. That is the defect worth fixing.
"""
import torch
import torch.nn as nn

from .model import MapFormerEM, _apply_rope


class MapFormerEM_SingleP0(MapFormerEM):
    """MapFormer-EM with the paper's single origin vector: A_P = P . P^T.

    Identical to MapFormerEM except q0_pos/k0_pos are replaced by one p0_pos
    used on both sides, making A_P an autocorrelation kernel peaked at zero
    displacement. The layer (softmax(A_X o A_P) V) is unchanged -- it was
    already correct.
    """

    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1,
                 dropout=0.1, grid_size=64, bottleneck_r=2):
        super().__init__(vocab_size, d_model, n_heads, n_layers, dropout,
                         grid_size, bottleneck_r)
        # one origin, used for BOTH query and key sides (paper eq. 3)
        self.p0_pos = nn.Parameter(self.q0_pos.detach().clone())
        del self.q0_pos, self.k0_pos

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, L = tokens.shape
        x = self.token_emb(tokens)
        cos_a, sin_a = self.path_integrator(self.action_to_lie(x))

        p0 = self.p0_pos.unsqueeze(0).unsqueeze(2).expand(B, -1, L, -1)
        p = _apply_rope(p0, cos_a, sin_a)          # P = R_theta P*
        q_pos = k_pos = p                          # A_P = P . P^T

        causal_mask = torch.triu(
            torch.ones(L, L, device=tokens.device, dtype=torch.bool), diagonal=1)
        for layer in self.layers:
            x = layer(x, q_pos, k_pos, causal_mask)
        return self.out_proj(self.out_norm(x))


def _single_p0_rank(r):
    """Shared origin AND a wider bottleneck -- the combination that did not exist.

    MINIGRID_EM.md measured EM's failure precisely: it MATCHES WM on 7 of 8 seeds
    and collapses on one (0.666 against a pack at 0.810-0.834; worst-to-second-worst
    gap 0.144 for EM_r4 against 0.007 for WM_r4). Two independent init pathologies
    are candidates and they were never separable:

      separate q0/k0  A_P is not peaked at zero displacement at init, so the
                      AND-gate multiplies content by a random position mask and the
                      gradient signal dies. Shared p0 is worth +0.358 on Match-Query.
      r=2             a skewed action basis (|cos(N,E)| 0.78 against 0.17 at r=4).

    `VanillaEM_P0` fixes only the first, `VanillaEM_r4` only the second. This fixes
    both, which is what the 2x2 needs.
    """
    class _P(MapFormerEM_SingleP0):
        def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1,
                     dropout=0.1, grid_size=64, bottleneck_r=2, **kw):
            super().__init__(vocab_size, d_model, n_heads, n_layers, dropout,
                             grid_size, r)
            assert self.action_to_lie.w_in.out_features == r, "rank lost in construction"
            assert hasattr(self, "p0_pos") and not hasattr(self, "q0_pos"), "origin not shared"
    _P.__name__ = f"MapFormerEM_SingleP0_r{r}"
    _P.__doc__ = f"MapFormer-EM, shared origin (paper eq. 3), bottleneck rank r={r}."
    return _P


MapFormerEM_SingleP0_r4 = _single_p0_rank(4)
