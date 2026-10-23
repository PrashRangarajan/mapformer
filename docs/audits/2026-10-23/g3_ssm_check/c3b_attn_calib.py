"""Calibration for C3: the same move-angle Adam-sensitivity (L1 Jacobian) on WRAP_EXISTENCE's attention constructions
AO_bal and SO_bal (HELD at lr 1e-3, 300 epochs), rebuilt from SO_s600..s602 with the committed wrap_construct."""
import torch
from mapformer.environment_cancel import GridWorldCancel
from mapformer.model_wrap_exist import MapFormerWM_SignedAO_r4, wrap_construct

R = "/home/prashr/mapformer/runs/wrap_existence/p0"


def sensitivity(m):
    ps = [m.token_emb.weight, m.action_to_lie.w_in.weight, m.action_to_lie.w_out.weight, m.path_integrator.omega]
    A = (m.step_table()[m.act] * m.path_integrator.omega).reshape(-1)
    tot = []
    for i in range(A.numel()):
        g = torch.autograd.grad(A[i], ps, retain_graph=True, allow_unused=True)
        tot.append(sum(float(x.abs().sum()) for x in g if x is not None))
    t = torch.tensor(tot)
    return float(t.mean()), float(A.detach().abs().mean())


env = GridWorldCancel(size=32, p_plus=0.5)
kw = dict(vocab_size=env.unified_vocab_size, d_model=128, n_heads=2, n_layers=1, grid_size=32,
          action_ids=(env.action_offset, env.action_offset + 1))
for s in (600, 601, 602):
    b = torch.load(f"{R}/SO_s{s}/SO.pt", map_location="cpu", weights_only=False)
    so = MapFormerWM_SignedAO_r4(**kw); so.load_state_dict(b["model_state_dict"])
    ao, _ = wrap_construct(so, kw["vocab_size"], kw["action_ids"], mode="balanced", d_model=128, n_heads=2,
                           n_layers=1, grid_size=32)
    sb, _ = wrap_construct(so, kw["vocab_size"], kw["action_ids"], mode="signed_balanced", d_model=128, n_heads=2,
                           n_layers=1, grid_size=32)
    a, aa = sensitivity(ao); c, ca = sensitivity(sb); o, oa = sensitivity(so)
    print(f"s{s}: source sens {o:.1f} (|ang| {oa:.2f}); SO_bal {c:.1f} (|ang| {ca:.2f}); AO_bal {a:.1f} (|ang| {aa:.2f});"
          f" ratio AO_bal / SO_bal {a / c:.2f}; max|W_in| AO {float(ao.action_to_lie.w_in.weight.abs().max()):.3f} "
          f"max|W_out| AO {float(ao.action_to_lie.w_out.weight.abs().max()):.3f}")
