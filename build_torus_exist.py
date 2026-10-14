"""TORUS_EXIST: build the warm-start checkpoints from a trained SignedAO4_r4 torus model (CPU or GPU).
For each mode in (balanced, signed_balanced, [min]) writes <out>/<arm>/<variant>.pt with the source's config and seed (so
train_variant --init-from accepts it), and <out>/<arm>/construct.json with the construction report and the frozen
held-out accuracy of source and construction (eval_noise_refine.evaluate: env seed 10000, T 1024, same trajectories).
  python3 -m mapformer.build_torus_exist --src runs/torus_exist/p0/SO_s700/SignedAO4_r4.pt --out runs/torus_exist/init/s700
"""
import argparse, json, os

import torch

from mapformer.environment import GridWorld
from mapformer.eval_noise_refine import evaluate
from mapformer.model_wrap_exist import wrap_construct, wrap_share
from mapformer.train_variant import VARIANT_MAP

ARMS = {"AO_bal": ("balanced", "AbsAO4_r4"), "SO_bal": ("signed_balanced", "SignedAO4_r4"), "AO_wrap": ("min", "AbsAO4_r4")}
PAIRS = ((0, 1), (2, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", nargs="+", default=["AO_bal", "SO_bal"])
    ap.add_argument("--T", type=int, default=1024)
    ap.add_argument("--n-trials", type=int, default=50)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    b = torch.load(a.src, map_location="cpu", weights_only=False)
    assert b["variant"] == "SignedAO4_r4", b["variant"]
    c = b["config"]; kw = dict(d_model=c["d_model"], n_heads=c["n_heads"], n_layers=c["n_layers"], grid_size=c["grid_size"])
    src = VARIANT_MAP["SignedAO4_r4"](vocab_size=c["vocab_size"], **kw); src.load_state_dict(b["model_state_dict"])
    src = src.to(a.device).eval()
    env = GridWorld(size=c["grid_size"], n_obs_types=c.get("n_obs_types", 16), p_empty=c.get("p_empty", 0.5), seed=10000)
    src_acc = evaluate(src, env, a.T, a.n_trials, 0.0, a.device, seed=1234)[0]
    for arm in a.arms:
        mode, variant = ARMS[arm]
        m, rep = wrap_construct(src.cpu(), c["vocab_size"], (0, 1, 2, 3), mode=mode, **kw)
        src.to(a.device)
        m2 = VARIANT_MAP[variant](vocab_size=c["vocab_size"], **kw); m2.load_state_dict(m.state_dict())   # the registered class
        m2 = m2.to(a.device).eval()
        frz = evaluate(m2, env, a.T, a.n_trials, 0.0, a.device, seed=1234)[0]
        d = os.path.join(a.out, arm); os.makedirs(d, exist_ok=True)
        torch.save({"model_state_dict": {k: v.cpu() for k, v in m2.state_dict().items()}, "losses": [], "variant": variant,
                    "seed": b["seed"], "config": dict(c, constructed_from=a.src, construction=mode)}, os.path.join(d, f"{variant}.pt"))
        info = {"arm": arm, "mode": mode, "variant": variant, "construction": rep, "source_acc": src_acc, "frozen_acc": frz,
                "source_wrap": wrap_share(src.cpu(), pairs=PAIRS), "frozen_wrap": wrap_share(m2.cpu(), pairs=PAIRS)}
        json.dump(info, open(os.path.join(d, "construct.json"), "w"), indent=2)
        src.to(a.device)
        print(f"{arm}: source acc {src_acc:.4f} frozen {frz:.4f} | angle err {rep['angle_err_max']:.1e} | |W_out| "
              f"{rep['w_out_absmax_signed']:.2f} -> {rep['w_out_absmax_built']:.2f} | wrap {info['source_wrap'][1]:.2f} -> "
              f"{info['frozen_wrap'][1]:.2f} | latent cond {rep['latent_cond_signed']:.1f}", flush=True)


if __name__ == "__main__":
    main()
