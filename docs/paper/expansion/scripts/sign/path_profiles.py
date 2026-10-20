"""SEARCH_THEORY A2 (eval-only, CPU, exploratory): the revisit loss along straight paths from the |.| start to a solution, on
the REAL ring model with the content of a solved signed model (runs/wrap_mech/p0/SO_s800-807) held fixed. Balanced gauge
(latents c e1, c e2; W_out columns carry the angles; omega' = max(|omega|, 1)), so a straight line in W_out is a straight line
in angle space, channel by channel.
  S_lin   signed, angles (1 - tau) |default| + tau a*_signed      (minus-move angles cross zero; reaches the source's code)
  A_lin   |.|,    angles (1 - tau) |default| + tau a*_wrapped     (stays >= 0; each channel's sum passes pi on the way)
  S_lat   signed, the collective latent route at the START's W_out: z(-)(tau) = c((1 - tau) e2 - tau e1), so
          a(-)(tau) = (1 - tau) a(-)_0 - tau a(+)_0 in every channel at once (ends at a cancelling code with the start's map)
  A_lat   |.|, the same latent route (ends at a(-) = a(+): a clock)
Loss = revisit cross-entropy (the training loss) on 64 fixed walks (np seed 4242), T = 128, eval mode; acc on the same walks.
  python3 path_profiles.py > path_profiles_out.txt
"""
import sys, json, math
from pathlib import Path
sys.path.insert(0, "/home/prashr")
import numpy as np
import torch
import torch.nn.functional as F

from mapformer.environment_cancel import GridWorldCancel
from mapformer.model_wrap_exist import MapFormerWM_SignedAO_r4, wrap_construct, action_angles
from mapformer.train_wrap_mech import set_move_angles

torch.set_num_threads(4)
R = Path("/home/prashr/mapformer/runs/wrap_mech/p0")
TAUS = np.round(np.linspace(0, 1, 21), 3)


def models(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    env = GridWorldCancel(size=32, seed=seed, p_plus=0.5)
    kw = dict(d_model=128, n_heads=2, n_layers=1, grid_size=32)
    act = (env.action_offset, env.action_offset + 1)
    src = MapFormerWM_SignedAO_r4(env.unified_vocab_size, action_ids=act, **kw)
    src.load_state_dict(torch.load(R / f"SO_s{seed}/SO.pt", map_location="cpu", weights_only=False)["model_state_dict"])
    a_sol, _ = wrap_construct(src, env.unified_vocab_size, act, mode="balanced", **kw)
    s_sol, _ = wrap_construct(src, env.unified_vocab_size, act, mode="signed_balanced", **kw)
    default = torch.tensor(json.load(open(R / f"SO_s{seed}/eval.json"))["default_angles"])
    return a_sol.eval(), s_sol.eval(), default


@torch.no_grad()
def loss_acc(m, tok, rev):
    lg = m(tok[:, :-1]); msk = rev[:, 1:].reshape(-1); y = tok[:, 1:].reshape(-1)[msk]
    lg = lg.reshape(-1, lg.shape[-1])[msk]
    return float(F.cross_entropy(lg, y)), float((lg.argmax(-1) == y).float().mean())


def main():
    np.random.seed(4242)
    env = GridWorldCancel(size=32, seed=4242, p_plus=0.5)
    tok, _om, rev, _loc = env.generate_batch(64, 128)
    summ = {}
    for seed in range(800, 808):
        a_sol, s_sol, default = models(seed)
        with torch.no_grad():
            a_star = action_angles(a_sol).clone(); s_star = action_angles(s_sol).clone()
        st = default.abs()
        prof = {}
        for name, m, target in (("S_lin", s_sol, s_star), ("A_lin", a_sol, a_star)):
            vals = []
            for t in TAUS:
                set_move_angles(m, (1 - t) * st + t * target)
                vals.append(loss_acc(m, tok, rev))
            prof[name] = vals
        for name, m in (("S_lat", s_sol), ("A_lat", a_sol)):
            set_move_angles(m, st)
            W = m.action_to_lie.w_in.weight.detach().clone()
            e = m.token_emb.weight.detach().double(); c = float((e[m.act] @ W.double().T).norm(dim=1).mean())
            vals = []
            for t in TAUS:
                L = torch.zeros(2, W.shape[0], dtype=torch.float64); L[0, 0] = c; L[1, 1] = c * (1 - t); L[1, 0] = -c * t
                with torch.no_grad():
                    m.action_to_lie.w_in.weight.copy_((torch.linalg.pinv(e[m.act]) @ L).T.float())
                vals.append(loss_acc(m, tok, rev))
            prof[name] = vals
        summ[seed] = prof
        print(f"seed {seed}  (loss/acc at tau = 0, 0.1, ..., 1)")
        for name, vals in prof.items():
            print(f"  {name}: " + " ".join(f"{vals[i][0]:.2f}/{vals[i][1]:.2f}" for i in range(0, 21, 2)))
            l = np.array([v[0] for v in vals])
            bar = float(np.max(l - np.minimum.accumulate(l)))           # largest rise above the running minimum
            print(f"         monotone-violation (max rise over running min) {bar:.3f}; argmin tau {TAUS[int(l.argmin())]}; "
                  f"loss start {l[0]:.2f} end {l[-1]:.3f}")
    json.dump({str(k): v for k, v in summ.items()}, open(Path(__file__).with_name("path_profiles.json"), "w"))
    print("\n== mean over seeds: loss at tau = 0, 0.1, ..., 1 ==")
    for name in ("S_lin", "A_lin", "S_lat", "A_lat"):
        L = np.array([[v[0] for v in summ[s][name]] for s in summ]); A = np.array([[v[1] for v in summ[s][name]] for s in summ])
        print(f"  {name} loss: " + " ".join(f"{x:.2f}" for x in L.mean(0)[::2]))
        print(f"  {name} acc : " + " ".join(f"{x:.2f}" for x in A.mean(0)[::2]))
        rises = [float(np.max(l - np.minimum.accumulate(l))) for l in L]
        print(f"  {name} max rise over running min per seed: " + " ".join(f"{r:.2f}" for r in rises))


if __name__ == "__main__":
    main()
