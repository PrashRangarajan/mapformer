"""MapFormer's phase generator is a SUBMODEL of Selective RoPE's -- constructed.

Rather than assert the inclusion, build the Selective RoPE parameters that
reproduce MapFormer's angle exactly, and check the outputs agree.

    MapFormer      theta_t = omega (*) W_out W_in  sum_{u<=t} x_u
    Selective RoPE theta_t = tau sum_{u<=t} [ sigma(W_g x_u) (*) (kappa * W_w x)_u ]

Four steps, each landing on a point in Selective RoPE's own parameter space:

  1  kappa := delta at the LAST tap. The conv is left-padded by K-1, so
     out[t] = sum_j w[j] u[t+j-3]; weights (0,0,0,1) give out[t] = u[t] and K
     collapses 4 -> 1.
  2  W_g := 0. The gate is then the CONSTANT sigma(b_g) for ANY bias -- it does
     not need saturating, and the constant is absorbed in step 3. (An earlier
     version of the math note called this a limit. It is not; it is exact.)
  3  W_omega := diag(vec omega) W_out W_in / (tau * sigma(b_g)). Legal because
     the full-rank parameterisation does not CONSTRAIN rank, and weight_norm
     represents any matrix (rows g_i v_i/||v_i||, with g_i = 0 for a zero row).
  4  bias := 0, or MapFormer picks up an index-RoPE component.

Two things this makes precise, both refinements to the math note:

  * The inclusion is at the level of the FUNCTION theta(x_1..t), not the state
    dimension. Selective RoPE realises MapFormer with a 64-dimensional state
    confined to an r-dimensional subspace. So coherence rank must be defined on
    the COMPOSITE token->phase map A.B, not on A alone -- otherwise this
    construction would read rho = 64 while computing a rank-2 function.
  * The chain RoPE < MapFormer < Selective RoPE is NOT uniformly exact. RoPE
    sits inside Selective RoPE unconditionally (W_omega := 0, bias := b gives
    theta_t = tau (t+1) b), but inside MapFormer only CONDITIONALLY: Delta has
    no bias term, so Delta_t == const requires the learned embeddings to share a
    projection, which is a hypothesis about training rather than a setting.
"""
import numpy as np
import torch

from mapformer.model import MapFormerWM
from mapformer.model_selective import MapFormerWM_SRoPEGen


def main():
    torch.manual_seed(0)
    V, d, H, r = 21, 128, 2, 2
    mf = MapFormerWM(vocab_size=V, d_model=d, n_heads=H, n_layers=1,
                     grid_size=64, bottleneck_r=r).eval()
    sr = MapFormerWM_SRoPEGen(vocab_size=V, d_model=d, n_heads=H, n_layers=1,
                              grid_size=64).eval()
    C = H * mf.n_blocks
    sr.token_emb.load_state_dict(mf.token_emb.state_dict())

    with torch.no_grad():
        sr.angle.conv.conv.weight.zero_()                       # 1
        sr.angle.conv.conv.weight[:, 0, -1] = 1.0
        sr.angle.gate.weight.zero_()                            # 2
        g = torch.sigmoid(sr.angle.gate.bias)
        tau = sr.angle.log_temp.exp()
        A = (mf.path_integrator.omega.reshape(C, 1)             # 3
             * (mf.action_to_lie.w_out.weight @ mf.action_to_lie.w_in.weight))
        torch.nn.utils.parametrize.remove_parametrizations(sr.angle.proj, "weight")
        sr.angle.proj.weight.copy_(A / (tau * g).reshape(C, 1))
        sr.angle.proj.bias.zero_()                              # 4

        tok = torch.randint(0, V, (4, 64))
        x = mf.token_emb(tok)
        cos_m, sin_m = mf.path_integrator(mf.action_to_lie(x))
        th_s = sr.angle(x).transpose(1, 2)
        dc = (cos_m - torch.cos(th_s)).abs().max().item()
        ds = (sin_m - torch.sin(th_s)).abs().max().item()

    print(f"max |cos diff| {dc:.3e}   max |sin diff| {ds:.3e}")
    print(f"composite rank: MapFormer {np.linalg.matrix_rank(A.numpy(), tol=1e-6)}"
          f", the SRoPE W_omega realising it "
          f"{np.linalg.matrix_rank((A/(tau*g).reshape(C,1)).numpy(), tol=1e-6)}"
          f"  (of {C} channels)")
    assert max(dc, ds) < 1e-3, "construction failed"

    with torch.no_grad():                                       # RoPE, via the bias
        sr.angle.proj.weight.zero_()
        sr.angle.proj.bias.copy_(torch.linspace(0.1, 1.0, C))
        th = sr.angle(x)[0, :, 0, 0]
        ratio = th / torch.arange(1, x.shape[1] + 1).float()
    print(f"RoPE special case: theta_t/(t+1) constant to "
          f"{(ratio - ratio.mean()).abs().max().item():.3e}")
    print("PASS")


if __name__ == "__main__":
    main()
