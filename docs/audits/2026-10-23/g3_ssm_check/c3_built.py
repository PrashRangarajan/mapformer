"""Auditor check C3: the committed pilot constructions (built_s2183, built_s2180, built_s2183_all), re-read independently.
(a) every tensor except w_in, w_out, omega bitwise equal to the source; (b) angle table mod 2 pi vs the source, from raw
weights; (c) constraint; (d) frozen accuracy vs source on the registered stream (CPU); (e) scale / sensitivity of the
move angles to an Adam-sized step (L1 norm of d angle / d theta over the step-path parameters) and |angle| magnitudes."""
import math
import torch
from mapformer.construct_ssm_sign import load_source
from mapformer.environment_steps import GridWorldSteps
from mapformer.model_ssm_sign import build
from mapformer.train_cancel import evaluate

P = "/home/prashr/mapformer/runs/ssm_sign_pilot"
SRC = {"built_s2183": f"{P}/signed_moves_m32_lr3e-3_s2183/signed.pt",
       "built_s2180": f"{P}/signed_moves_m32_s2180/signed.pt",
       "built_s2183_all": f"{P}/signed_moves_m32_lr3e-3_s2183/signed.pt"}
F = {"signed": lambda z: z, "abs": torch.abs, "softplus": torch.nn.functional.softplus, "exp": torch.exp}


def angles_raw(sd, mode, act=(0, 1)):
    E = sd["token_emb.weight"].double()
    s = E @ sd["action_to_lie.w_in.weight"].double().T @ sd["action_to_lie.w_out.weight"].double().T
    return F[mode](s) * sd["path_integrator.omega"].double().reshape(-1)


def wrap(x):
    return torch.remainder(x + math.pi, 2 * math.pi) - math.pi


def sensitivity(m):
    """L1 norm of the Jacobian of the 2 x 64 move angles w.r.t. move embeddings, w_in, w_out, omega: the angle change
    an Adam step of size 1 per coordinate (sign-aligned) could make. Multiply by lr for a per-step bound."""
    ps = [m.token_emb.weight, m.action_to_lie.w_in.weight, m.action_to_lie.w_out.weight, m.path_integrator.omega]
    A = m.angle_table()[m.act].reshape(-1)
    tot = torch.zeros(A.numel(), dtype=torch.float64)
    for i in range(A.numel()):
        g = torch.autograd.grad(A[i], ps, retain_graph=True, allow_unused=True)
        tot[i] = sum(float(x.abs().sum()) for x in g if x is not None)
    return float(tot.mean()), float(tot.max())


for d, sp in SRC.items():
    src, b = load_source(sp)
    cfg = b["config"]
    env = GridWorldSteps(size=cfg["size"], seed=10000, p_first=0.5, deltas=tuple(cfg["deltas"]))
    Ssd = src.state_dict()
    src_acc = evaluate(src, env, 128, 400, "cpu", seed=10 ** 6)[0]
    A_src = angles_raw(Ssd, "signed")
    print(f"== {d}  source acc {src_acc:.4f}; source |move angle| mean {float(A_src[:2].abs().mean()):.3f} "
          f"max {float(A_src[:2].abs().max()):.3f}; source sens {sensitivity(src)}")
    for k in ("abs", "softplus", "exp", "signed"):
        bb = torch.load(f"{P}/{d}/{k}_built.pt", map_location="cpu", weights_only=False)
        c = bb["config"]
        m = build(k, c["steps"], c["vocab_size"], c["grid_size"], action_ids=tuple(c["action_ids"]))
        m.load_state_dict(bb["model_state_dict"]); m.eval()
        sd = m.state_dict()
        changed = sorted(n for n in sd if not torch.equal(sd[n], Ssd[n]))
        A = angles_raw(sd, k)
        if c["steps"] == "moves":
            mask = torch.zeros(A.shape[0], dtype=torch.bool); mask[[0, 1]] = True
            A = A * mask[:, None]
            ref = A_src * mask[:, None]
        else:
            ref = A_src.clone(); ref[2:] = 0.0            # observations annihilated: 0 mod 2 pi
        err = float(wrap(A - ref).abs().max())
        st = F[k](sd["token_emb.weight"].double() @ sd["action_to_lie.w_in.weight"].double().T
                  @ sd["action_to_lie.w_out.weight"].double().T)
        rows = st[:2] if c["steps"] == "moves" else st
        cons = {"abs": bool((rows >= 0).all()), "softplus": bool((rows > 0).all()), "exp": bool((rows > 0).all()),
                "signed": True}[k]
        acc = evaluate(m, env, 128, 400, "cpu", seed=10 ** 6)[0]
        s_mean, s_max = sensitivity(m)
        print(f"  {k:8s} changed {changed}  angle err {err:.1e}  constraint {cons}  frozen acc {acc:.4f} "
              f"(src {src_acc:.4f}, d {acc - src_acc:+.4f})  |move angle| mean {float(A[:2].abs().mean()):.3f} "
              f"max {float(A[:2].abs().max()):.3f}  omega max {float(sd['path_integrator.omega'].abs().max()):.2f} "
              f"min {float(sd['path_integrator.omega'].abs().min()):.2f}  sens mean {s_mean:.2f} max {s_max:.2f} "
              f"| stored frozen check {c['construction'].get('frozen_acc')}")
