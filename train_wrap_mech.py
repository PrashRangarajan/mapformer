"""Trainer for MECHANISM_PREREG.md: does the non-negative step's learnability cost come from the START it imposes?
Ring (environment_cancel, 32 cells, p_plus 0.5), RING_DIAL recipe (T = 128, batch 128, 98 batches, lr 1e-3, cosine, 300
epochs, 1 layer, 2 heads, d 128). Observation-step-free models (SignedAO_r4 / AbsAO_r4, `model_wrap_exist.py`). Arms:
  SO / AO            default init (as WRAP_EXISTENCE)
  SO_same            SignedAO_r4 whose two move angles are set, in the balanced gauge, to THIS seed's default-init angles
                     (gauge control for SO)
  SO_absmag          SignedAO_r4 started at |default-init angles| (the start the |.| constraint imposes), balanced gauge
  AO_absmag          AbsAO_r4 started at |default-init angles| in the balanced gauge (gauge control for AO; same angles as AO)
  AO_high            AbsAO_r4 started with large move angles: a(+1), a(-1) iid U(1.7, 4.5) rad per channel (sum > pi)
  SO_high            SignedAO_r4 from the SAME large angles as AO_high (Amendment 1: matched-start sign contrast)
  AO_mid             AbsAO_r4 started with a(+1), a(-1) iid U(0.9, 1.5) (sum < pi, ~5x the default magnitude; Amendment 1
                     mobility control)
Balanced gauge (`set_move_angles`): W_in re-solved so the two move latents are c e1, c e2 (c = this init's mean move-latent
norm), omega' = max(|omega|, 1), W_out[:, :2] = (target angle / omega') / c, W_out[:, 2:] = 0 -- WRAP_EXISTENCE Amendment 1's
construction applied at initialisation. Every other weight is the seed's default init (identical across arms at a seed).
Evaluation on a fresh stream (np seed 10**6) at T and 4T. Writes <arm>.pt, eval.json (with the initial angles).
  python3 -m mapformer.train_wrap_mech --arm SO_absmag --seed 800 --device cuda:0 --output-dir runs/wrap_mech/p0/SO_absmag_s800
"""
import argparse, json
from pathlib import Path

import numpy as np
import torch

from mapformer.environment_cancel import GridWorldCancel
from mapformer.model_wrap_exist import MapFormerWM_SignedAO_r4, MapFormerWM_AbsAO_r4, action_angles, wrap_share
from mapformer.train import train
from mapformer.train_cancel import evaluate

ARMS = ("SO", "AO", "SO_same", "SO_absmag", "AO_absmag", "AO_high", "SO_high", "AO_mid")
EVAL_SEED = 10 ** 6


@torch.no_grad()
def set_move_angles(m, ang):
    """Set the model's move angles to `ang` (2, C) in the balanced gauge; returns the max angle error (rad, mod 2 pi)."""
    e = m.token_emb.weight.double(); W_in = m.action_to_lie.w_in.weight.double(); r = W_in.shape[0]
    lat = e[m.act] @ W_in.T; c = float(lat.norm(dim=1).mean())
    L = torch.zeros(2, r, dtype=torch.float64); L[0, 0] = c; L[1, 1] = c
    m.action_to_lie.w_in.weight.copy_((torch.linalg.pinv(e[m.act]) @ L).T.float())
    om = m.path_integrator.omega.double().reshape(-1).abs().clamp_min(1.0)
    m.path_integrator.omega.copy_(om.reshape(m.path_integrator.omega.shape).float())
    W = torch.zeros_like(m.action_to_lie.w_out.weight, dtype=torch.float64); W[:, :2] = (ang.double() / om).T / c
    m.action_to_lie.w_out.weight.copy_(W.float())
    return float(torch.angle(torch.exp(1j * (action_angles(m).double() - ang.double()))).abs().max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--p-plus", type=float, default=0.5)
    ap.add_argument("--size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--n-batches", type=int, default=98)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--n-steps", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--data-workers", type=int, default=3)
    ap.add_argument("--n-trials", type=int, default=400)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output-dir", required=True)
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    env = GridWorldCancel(size=a.size, seed=a.seed, p_plus=a.p_plus)
    kw = dict(vocab_size=env.unified_vocab_size, d_model=128, n_heads=2, n_layers=1, grid_size=a.size,
              action_ids=(env.action_offset + 0, env.action_offset + 1))
    signed = a.arm.startswith("SO")
    model = (MapFormerWM_SignedAO_r4 if signed else MapFormerWM_AbsAO_r4)(**kw)
    # the seed's default SIGNED move angles (identical for SO and AO at a seed: same draws; AO's are their |.|)
    with torch.no_grad():
        st = model.action_to_lie.w_out(model.action_to_lie.w_in(model.token_emb.weight[model.act]))      # (2, C) signed
        ang0 = st * model.path_integrator.omega.reshape(-1)
    err = None
    if a.arm in ("SO_same",):
        err = set_move_angles(model, ang0)
    elif a.arm in ("SO_absmag", "AO_absmag"):
        err = set_move_angles(model, ang0.abs())
    elif a.arm in ("AO_high", "SO_high", "AO_mid"):
        g = np.random.RandomState(a.seed + 7919)                     # same draw for AO_high and SO_high at a seed
        lo, hi = (0.9, 1.5) if a.arm == "AO_mid" else (1.7, 4.5)
        err = set_move_angles(model, torch.from_numpy(g.uniform(lo, hi, ang0.shape)).float())
    init_angles = action_angles(model)
    assert len(model.layers) == 1
    print(f"{a.arm} seed={a.seed} params={sum(p.numel() for p in model.parameters()):,} init angle err {err} "
          f"init |angle| mean {float(init_angles.abs().mean()):.3f} init wrap {wrap_share(model)} | init |W_out| max "
          f"{float(model.action_to_lie.w_out.weight.abs().max()):.3f} |W_in| max {float(model.action_to_lie.w_in.weight.abs().max()):.3f}",
          flush=True)
    losses = train(model, env, n_epochs=a.epochs, lr=a.lr, batch_size=a.batch_size, n_steps=a.n_steps,
                   n_batches=a.n_batches, device=a.device, schedule="cosine", data_workers=a.data_workers)
    torch.save({"model_state_dict": model.state_dict(), "losses": losses, "variant": a.arm, "seed": a.seed,
                "config": {"args": dict(vars(a)), "vocab_size": env.unified_vocab_size,
                           "action_offset": env.action_offset}}, out / f"{a.arm}.pt")
    te = GridWorldCancel(size=a.size, seed=10000, p_plus=a.p_plus)
    ev = {}
    for T in (a.n_steps, 4 * a.n_steps):
        acc, nll = evaluate(model, te, T, a.n_trials, a.device, seed=EVAL_SEED)
        ev[str(T)] = {"acc": acc, "nll": nll}
        print(f"fresh maps T={T}: acc {acc:.4f} nll {nll:.4f}", flush=True)
    model.cpu()
    json.dump({"eval": ev, "final_loss": float(np.mean(losses[-max(1, len(losses) // 20):])), "wrap": wrap_share(model),
               "init_angle_err": err, "init_angles": init_angles.tolist(), "default_angles": ang0.tolist()},
              open(out / "eval.json", "w"), indent=2)


if __name__ == "__main__":
    main()
