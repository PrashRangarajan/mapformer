"""Per-run readouts for TW_NORMSTEP_PREREG.md, from the checkpoint (CPU).

  drift    the clock readout of docs/audits/2026-09-27/tw_clock_probe.py, generalised to every step arm: on 40
           held-out walks (map 10000, np seed 0, T=1024), the phase theta = omega * cumsum(step) at each object
           slot is compared with the phase at the first visit to the same cell; a (head, block) channel DRIFTS if
           its mean |wrapped dtheta| over all revisit pairs exceeds 1.0 rad (random ~ pi/2). Count of 64. A map
           channel returns to the same phase at the same cell; a clock channel does not. Also (Amendment 1) the same
           count with the phase integrated over CORE positions only and over OPTIONAL positions only (drift_opt:
           the per-word clock readout; MapWM on the committed text-world runs: 0/64 on all 8 seeds).
  bias_step  ||W beta|| / mean ||direction-word step|| (NormStep only): the size of the beta parameter's step.
           0 by construction for NormStepNB. Descriptive: a step shared by all tokens can also come from a shared
           component of gamma * norm(e), and a token can cancel W beta through its own embedding (Amendment 1).
  move     mean ||step|| over non-direction words / mean over the 12 direction words (probe_textworld's ratio).
  ablate   held-out accuracy (T=1024, 200 trials, the trainer's eval stream) with every non-direction word's
           step zeroed (secondary: a per-move gauge between verb and direction word makes it gauge-dependent).
"""
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

from mapformer.environment_textworld import TextWorld, VERBS
from mapformer.model_textstep import step_of
from mapformer.train_tw_normstep import build, EVAL_SEED, HELDOUT

wrap = lambda x: (x + np.pi) % (2 * np.pi) - np.pi


def load(ck):
    b = torch.load(ck, map_location="cpu", weights_only=False); a = b["config"]["args"]
    env = TextWorld(size=a["size"], seed=a["seed"])
    m = build(b["arm"], env, a["n_layers"], a["size"]); m.load_state_dict(b["model_state_dict"])
    return m.eval(), b["arm"]


def core_mask(te, t, slots):
    """Per-move CORE positions (exactly one each per move: verb, direction word, the 2 seeing words, object, '.');
    every other position (adverb, fillers, aside sentence) is OPTIONAL, present a variable number of times.
    Amendment 1 (audit prototype, verified on the committed text-world runs)."""
    dirset = set(i for a in range(4) for i in te.dir_ids[a]); verbset = set(te.idx[w] for w in VERBS)
    core = np.zeros(len(t), bool)
    for i in slots:
        core[i] = True; core[i - 2:i] = True
        if i + 1 < len(t):
            core[i + 1] = True
        j = i - 3
        while t[j] not in dirset:
            j -= 1
        core[j] = True
        while t[j] not in verbset:
            j -= 1
        core[j] = True
    return core


@torch.no_grad()
def drift(m, n_walks=40, T=1024, thr=1.0):
    """-> {full, core, opt}: drift channel counts with the phase integrated over all / core / optional positions
    only. full is the registered readout; opt is the per-WORD clock readout (Amendment 1): a step on optional
    words cannot be absorbed by a per-move gauge, so any opt drift is time leaking into the phase."""
    te = TextWorld(size=64, seed=HELDOUT); om = m.path_integrator.omega.detach().numpy()
    np.random.seed(0); dif = {"full": [], "core": [], "opt": []}
    for _ in range(n_walks):
        tok, obs, _rev = te.generate_trajectory(T); locs = te.visited_locations
        d = step_of(m, tok[None])[0].numpy(); t = tok.numpy()
        slots = np.nonzero(obs.numpy())[0][:len(locs)]; core = core_mask(te, t, slots)
        ths = {"full": np.cumsum(d, 0) * om[None], "core": np.cumsum(d * core[:, None, None], 0) * om[None],
               "opt": np.cumsum(d * (~core)[:, None, None], 0) * om[None]}
        first = {}
        for k, i in enumerate(slots):
            L = tuple(locs[k])
            if L in first:
                for key, th in ths.items():
                    dif[key].append(wrap(th[i] - th[first[L]]))
            else:
                first[L] = i
    out = {}
    for key, v in dif.items():
        md = np.abs(np.array(v)).mean(0); out[key] = int((md > thr).sum()); out[key + "_rad"] = float(md.mean())
    out["channels"] = int(md.size)
    return out


@torch.no_grad()
def table(m, env):
    ids = torch.arange(env.unified_vocab_size)[None]
    D = step_of(m, ids)[0].reshape(env.unified_vocab_size, -1).numpy()
    n = np.linalg.norm(D, axis=1); dirs = [i for a in range(4) for i in env.dir_ids[a]]
    other = [i for i in range(env.unified_vocab_size) if i not in dirs]
    out = {"move": float(n[other].mean() / n[dirs].mean()), "bias_step": None}
    ln = getattr(m, "step_ln", None)
    if ln is not None:
        b = ln.bias if ln.bias is not None else torch.zeros(m.d_model)
        out["bias_step"] = float(m.action_to_lie(b[None, None])[0, 0].reshape(-1).norm() / n[dirs].mean())
    return out


@torch.no_grad()
def ablate(m, env, T=1024, n_trials=200):
    dirs = torch.tensor([i for a in range(4) for i in env.dir_ids[a]])
    orig = m.step if hasattr(m, "step") else None
    def masked(tokens, x):
        s = orig(tokens, x) if orig else m.action_to_lie(x)
        return s * torch.isin(tokens, dirs)[..., None, None].to(s.dtype)
    if orig is None:                                   # MapWM: route through the same masking
        f = m.action_to_lie; keep = {}
        def hook(mod, inp, out):
            return out * keep["k"][..., None, None].to(out.dtype)
        h = f.register_forward_hook(hook)
    else:
        m.step = masked
    te = TextWorld(size=64, seed=HELDOUT); np.random.seed(EVAL_SEED); ok = tot = 0
    for _ in range(n_trials):
        tok, _o, rev = te.generate_trajectory(T); inp = tok[None, :-1]
        if orig is None:
            keep["k"] = torch.isin(inp, dirs)
        lg = m(inp)[0]; msk = rev[1:]
        ok += int((lg.argmax(-1)[msk] == tok[1:][msk]).sum()); tot += int(msk.sum())
    if orig is None:
        h.remove()
    else:
        del m.step
    return ok / tot


@torch.no_grad()
def own_acc(m, map_seed, T=1024, n_trials=100):
    """Accuracy on the run's own TRAINING map (env seed = its seed), eval stream np seed 10**6 (Amendment 1 secondary:
    training uses one fixed map per seed, so own-map minus held-out accuracy measures memorisation)."""
    te = TextWorld(size=64, seed=map_seed); np.random.seed(EVAL_SEED); ok = tot = 0
    for _ in range(n_trials):
        tok, _o, rev = te.generate_trajectory(T); lg = m(tok[None, :-1])[0]; msk = rev[1:]
        ok += int((lg.argmax(-1)[msk] == tok[1:][msk]).sum()); tot += int(msk.sum())
    return ok / tot


@torch.no_grad()
def mode_gap(m, n_walks=40, T=1024, n_drop=3):
    """Amendment 1 secondary (found in the pilot, docs/audits/2026-10-03/dropout_mode_check_out.txt): accuracy on
    the same 40 held-out walks (map 10000, np seed 10**6) in eval mode and in train mode (all dropout on, as in
    training; mean over n_drop dropout seeds). Clock-type solutions depend on attention-probability dropout."""
    te = TextWorld(size=64, seed=HELDOUT); np.random.seed(EVAL_SEED)
    W = [te.generate_trajectory(T) for _ in range(n_walks)]
    def acc():
        ok = tot = 0
        for tok, _o, rev in W:
            lg = m(tok[None, :-1])[0]; msk = rev[1:]
            ok += int((lg.argmax(-1)[msk] == tok[1:][msk]).sum()); tot += int(msk.sum())
        return ok / tot
    m.eval(); a_eval = acc(); a_tr = []
    for k in range(n_drop):
        m.train(); torch.manual_seed(k); a_tr.append(acc())
    m.eval()
    return a_eval, float(np.mean(a_tr))


def readouts(ck, do_ablate=True):
    m, arm = load(ck); env = TextWorld(size=64, seed=0)
    dr = drift(m)
    r = {"arm": arm, "drift": dr["full"], "drift_core": dr["core"], "drift_opt": dr["opt"], "channels": dr["channels"],
         "drift_rad": dr["full_rad"], "drift_opt_rad": dr["opt_rad"],
         **table(m, env)}
    if do_ablate:
        r["ablate_nondir"] = ablate(m, env)
    r["acc_own"] = own_acc(m, torch.load(ck, map_location="cpu", weights_only=False)["seed"])
    r["acc40_eval"], r["acc40_train"] = mode_gap(m)
    return r


if __name__ == "__main__":
    print(json.dumps(readouts(sys.argv[1]), indent=1))
