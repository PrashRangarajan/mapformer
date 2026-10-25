"""Evaluation of image-torus checkpoints (IMAGE_TORUS_PREREG.md). One JSON per run, `eval_image.json` beside the
checkpoint. Every readout scores revisit targets (class or blank token, argmax over the unified vocabulary) exactly as
eval_noise_refine does, on its stream: held-out map GridWorld(64, 16, 0.5, env seed 10000), np.random.seed(1234 + run
seed), 100 walks. Walks are scored in fixed chunks of CHUNK; before chunk i the instance generator is reseeded to
IMG_SEED + 1000 * run seed + i, so every arm at a seed sees the same images on the same walk (paired).

Readouts (key -> meaning):
  test|1024            REGISTERED: test-split images, T = 1024 (in distribution); strata short-gap / long-gap / wrap
  test|512, test|2048  test split, other lengths (2048 is out of distribution, rule 10)
  train|1024           train-split images on the held-out map (perception transfer gap = train - test)
  own|1024             the run's own training map (env seed = run seed), test images
  zeroobj|1024         object-observation steps zeroed at eval (actions and blank keep theirs): leak L = zeroobj - test
  shuffle|1024         context destruction: observation tokens permuted among the input's observation slots (targets
                       unchanged) -- a map-reading model must fall to about the blank floor
  step                 step-map statistics on 10 walks (path arms): rms step of actions / objects / blank, and the
                       within-class (per-instance) rms deviation of object steps; ratios to the action rms
  probe                image arms: linear probe (16-way softmax regression, LBFGS) on the trained encoder's features,
                       fitted on all train-split images, scored on all test-split images

  python3 -m mapformer.eval_image_torus --ckpt runs/image_torus/I_MapWM_s3100/I_MapWM.pt --device cuda:0
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from mapformer.environment import GridWorld
from mapformer.eval_rank_strata import kinds_vec
from mapformer import model_image as MI
from mapformer import image_bank

ENV_SEED, N_TRIALS, CHUNK, IMG_SEED = 10000, 100, 4, 777_000_000
STRATA = ("plain_lag<128", "plain_lag>=128", "wrap")


def load(ckpt, dev, banks=None):
    b = torch.load(ckpt, map_location="cpu", weights_only=False)
    c = b["config"]
    if c["obs"] == "image":
        banks = banks or image_bank.load_banks()
        assert c["bank_md5"] == image_bank.bank_md5(), (ckpt, c["bank_md5"])
    m = MI.build(c["arm"], c["obs"], b["seed"], n_layers=c["n_layers"], banks=banks, grid_size=c["grid_size"],
                 seed_rng=False)
    m.load_state_dict(b["model_state_dict"])               # strict; the image banks are non-persistent buffers
    return m.to(dev).eval(), b


def walks(env_seed, np_seed, n, T, size=64):
    env = GridWorld(size=size, n_obs_types=MI.N_CLS, p_empty=0.5, seed=env_seed)
    np.random.seed(np_seed)
    return [env.generate_trajectory(T)[::2] for _ in range(n)]      # (tokens, revisit) per walk


class _ZeroObj:
    """Forward hook on action_to_lie: object-observation steps -> 0 (actions and blank unchanged)."""

    def __init__(self, m):
        self.keep = None
        self.h = m.action_to_lie.register_forward_hook(self._hook) if hasattr(m, "action_to_lie") else None

    def _hook(self, mod, inp, out):
        if self.keep is None:
            return out
        return torch.where(self.keep[:, :, None, None], out, torch.zeros_like(out))

    def close(self):
        if self.h is not None:
            self.h.remove()


def shuffle_obs(inp, rng):
    """Context destruction: permute each row's observation tokens among its observation slots; actions untouched."""
    inp = inp.clone()
    for r in range(inp.shape[0]):
        o = inp[r, 1::2].clone(); inp[r, 1::2] = o[torch.from_numpy(rng.permutation(o.numel()))]
    return inp


@torch.no_grad()
def score(m, W, seed, dev, split="test", zero_obj=False, shuffle=False, size=64):
    emb = MI.image_emb(m)
    if emb is not None:
        MI.set_split(m, split)
    zo = _ZeroObj(m) if zero_obj else None
    hit = {k: 0 for k in ("all",) + STRATA}; cnt = dict.fromkeys(hit, 0); nll = 0.0
    rng = np.random.default_rng(99_000 + seed)
    for i in range(0, len(W), CHUNK):
        tok = torch.stack([w[0] for w in W[i:i + CHUNK]]); rev = torch.stack([w[1] for w in W[i:i + CHUNK]])
        inp = tok[:, :-1].clone()
        if shuffle:
            inp = shuffle_obs(inp, rng)
        if emb is not None:
            emb.reseed(IMG_SEED + 1000 * seed + i // CHUNK)
        inp = inp.to(dev)
        if zo is not None:
            zo.keep = (inp < MI.N_ACT) | (inp == MI.BLANK)
        lp = F.log_softmax(m(inp).float(), -1).cpu()
        if zo is not None:
            zo.keep = None
        tgt, msk = tok[:, 1:], rev[:, 1:]
        pred = lp.argmax(-1)
        for r in range(tok.shape[0]):
            idx = torch.nonzero(msk[r]).flatten()
            if idx.numel() == 0:
                continue
            h = (pred[r, idx] == tgt[r, idx]).numpy()
            nll += float(-lp[r, idx, tgt[r, idx]].double().sum())
            new_u, lag = kinds_vec(tok[r], size)
            st = (idx // 2).numpy()
            wrap, short = new_u[st], lag[st] < 128
            for k, sel in (("all", np.ones_like(h, bool)), ("wrap", wrap), ("plain_lag<128", ~wrap & short),
                           ("plain_lag>=128", ~wrap & ~short)):
                hit[k] += int(h[sel].sum()); cnt[k] += int(sel.sum())
    if zo is not None:
        zo.close()
    if emb is not None:
        MI.set_split(m, None)                                         # back to auto (eval mode -> test)
    return {"acc": hit["all"] / max(1, cnt["all"]), "n": cnt["all"], "nll": nll / max(1, cnt["all"]),
            "strata": {k: {"acc": hit[k] / max(1, cnt[k]), "n": cnt[k]} for k in STRATA}}


@torch.no_grad()
def step_stats(m, W, seed, dev):
    if not hasattr(m, "action_to_lie"):
        return None
    emb = MI.image_emb(m)
    if emb is not None:
        MI.set_split(m, "test"); emb.reseed(IMG_SEED + 1000 * seed + 500)
    try:
        return _step_stats(m, W, dev)
    finally:
        if emb is not None:
            MI.set_split(m, None)


def _step_stats(m, W, dev):
    tok = torch.stack([w[0] for w in W])[:, :-1].to(dev)
    d = MI.step_of(m, tok).flatten(2).double().cpu()                   # (B, L, H * n_blocks)
    t = tok.cpu()
    act, blank = t < MI.N_ACT, t == MI.BLANK
    obj = (t >= MI.N_ACT) & ~blank
    rms = lambda sel: float(d[sel].pow(2).sum(-1).mean().sqrt()) if bool(sel.any()) else None
    dev2, n = 0.0, 0
    for c in range(MI.N_ACT, MI.BLANK):
        sel = t == c
        if int(sel.sum()) > 1:
            x = d[sel]; dev2 += float((x - x.mean(0)).pow(2).sum()); n += x.shape[0]
    ra = rms(act)
    inst = (dev2 / n) ** 0.5 if n else None
    out = {"rms_action": ra, "rms_object": rms(obj), "rms_blank": rms(blank), "within_class_rms_object": inst}
    out["object_over_action"] = out["rms_object"] / ra if ra else None
    out["instance_over_action"] = inst / ra if (ra and inst is not None) else None
    return out


@torch.no_grad()
def _features(enc, imgs, dev, bs=4096):
    return torch.cat([enc(imgs[i:i + bs].to(dev).float()[:, None] / 255.0).cpu() for i in range(0, len(imgs), bs)])


def probe(m, banks, dev, iters=200):
    """Linear probe of the trained encoder: train-split features -> 16 classes, scored on the test split."""
    enc = MI.image_emb(m).encoder
    Xtr = _features(enc, banks["train"]["images"], dev).double(); ytr = banks["train"]["labels"]
    Xte = _features(enc, banks["test"]["images"], dev).double(); yte = banks["test"]["labels"]
    mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-6)
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    W = torch.zeros(Xtr.shape[1], MI.N_CLS, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(MI.N_CLS, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(Xtr @ W + b, ytr) + 1e-4 * W.pow(2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        pte = (Xte @ W + b).argmax(1); ptr = (Xtr @ W + b).argmax(1)
    per = [float((pte[yte == k] == k).double().mean()) for k in range(MI.N_CLS)]
    return {"test_acc": float((pte == yte).double().mean()), "train_acc": float((ptr == ytr).double().mean()),
            "per_class_test": per}


def evaluate_run(ckpt, dev, banks=None, n_trials=N_TRIALS, lengths=(1024, 512, 2048), do_probe=True):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    t0 = time.time()
    m, blob = load(ckpt, dev, banks)
    s, c = blob["seed"], blob["config"]
    gs = c["grid_size"]
    emb = MI.image_emb(m)
    d0 = dict(emb.draws) if emb is not None else None
    R = {"ckpt": os.path.abspath(ckpt), "cell": blob["variant"], "seed": s, "arm": c["arm"], "obs": c["obs"],
         "n_layers": c["n_layers"], "bank_md5": c.get("bank_md5"), "losses": [float(x) for x in blob["losses"]],
         "acc": {}, "strata": {}}
    for T in lengths:
        W = walks(ENV_SEED, 1234 + s, n_trials, T, gs)
        r = score(m, W, s, dev, "test")
        R["acc"][f"test|{T}"] = r["acc"]; R["acc"][f"test|{T}|nll"] = r["nll"]; R["acc"][f"test|{T}|n"] = r["n"]
        if T == 1024:
            R["strata"]["test|1024"] = r["strata"]
            W1024 = W
    d1 = dict(emb.draws) if emb is not None else None
    if emb is not None:
        R["acc"]["train|1024"] = score(m, W1024, s, dev, "train")["acc"]
    R["acc"]["own|1024"] = score(m, walks(s, 1234 + s, n_trials, 1024, gs), s, dev, "test")["acc"]
    if hasattr(m, "action_to_lie"):
        R["acc"]["zeroobj|1024"] = score(m, W1024, s, dev, "test", zero_obj=True)["acc"]
    R["acc"]["shuffle|1024"] = score(m, W1024, s, dev, "test", shuffle=True)["acc"]
    R["step"] = step_stats(m, W1024[:10], s, dev)
    if emb is not None:
        R["draws_eval_main"] = {k: d1[k] - d0[k] for k in d0}             # the registered readouts' draws
        R["train_draws"] = c.get("train_draws")
        if do_probe:
            R["probe"] = probe(m, banks or image_bank.load_banks(), dev)
    R["eval_wall_s"] = time.time() - t0
    return R


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-trials", type=int, default=N_TRIALS)
    ap.add_argument("--lengths", nargs="+", type=int, default=[1024, 512, 2048])
    ap.add_argument("--no-probe", action="store_true")
    ap.add_argument("--skip-done", action="store_true", help="skip runs whose eval_image.json exists")
    a = ap.parse_args(argv)
    torch.backends.cuda.matmul.allow_tf32 = False          # as in training (Amendment 1, m1)
    torch.backends.cudnn.allow_tf32 = False
    assert a.lengths[0] == 1024, "1024 (the registered length) must come first"
    banks = None
    for ck in a.ckpt:
        out = os.path.join(os.path.dirname(ck), "eval_image.json")
        if a.skip_done and os.path.exists(out):
            print(f"skip {ck}"); continue
        blob_obs = torch.load(ck, map_location="cpu", weights_only=False)["config"]["obs"]
        if blob_obs == "image" and banks is None:
            banks = image_bank.load_banks()
        R = evaluate_run(ck, a.device, banks, a.n_trials, tuple(a.lengths), not a.no_probe)
        tmp = out + ".tmp"
        json.dump(R, open(tmp, "w"), indent=1); os.replace(tmp, out)
        print(f"{R['cell']} s{R['seed']}: test|1024 {R['acc']['test|1024']:.4f}  "
              + "  ".join(f"{k} {v:.4f}" for k, v in R["acc"].items() if "|n" not in k and k != "test|1024"
                          and not k.endswith("nll"))
              + (f"  probe {R['probe']['test_acc']:.4f}" if "probe" in R else "") + f"  ({R['eval_wall_s']:.0f} s)",
              flush=True)


if __name__ == "__main__":
    sys.exit(main())
