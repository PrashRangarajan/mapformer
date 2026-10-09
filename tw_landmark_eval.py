"""Per-run readouts for TW_LANDMARK_PREREG.md, from a checkpoint (CPU by default; --device for a GPU).

Eval stream: held-out map (env seed 10000), np seed 10**6, N_WALKS walks at T=1024 words. Every walk is rendered under
every condition from the same RNG state (environment_tw_landmark.render_conditions), so conditions differ ONLY in the
names. Scored targets: revisit object slots of moves k < K1, where K1 is the number of moves whose object slot fits in
the rate-1 rendering (the shortest), so every condition and every training cell is scored on the SAME targets.

Conditions (r = the run's training name rate):
  own     names at rate r, consistent      in-distribution accuracy (registered I readout)
  strip   no names                         'names stripped' (= own at r = 0)
  uninf   names at rate r, each FRESH      training form, names carry no information (= strip at r = 0)
  named   names at rate 1, consistent      every revisit name-solvable
  conf    names at rate 1, cue conflict    on conflict targets whose true and alt objects differ and are both non-blank,
                                           conf_path = share of argmax predictions at the object slot equal to the true
                                           cell's object, conf_name = share equal to the object of the cell whose name
                                           was shown (alt); the rest predict a third word
Theta reliance (path models): acc - acc with every direction word's step replaced by the mean step of the 12
direction words (theta then carries no displacement; the per-move common component, the aside / role offsets and
the name steps are untouched), per condition: rel_strip, rel_uninf, rel_named, rel_own. Secondary rel_all: also every
name token's step and the mark's step by the name mean (no name identity in theta).
In-distribution probe (Amendment 1, D1): acc_own_u05 / rel_own_u05 on the own rendering's targets at cells that are NOT
landmarks at rate 0.5 (defined at every rate by the coupled draws). MC-dropout acc_own_mc / acc_strip_mc (D3).
Also: acc_own split by landmark / unnamed cell (rate r naming), name benefit = acc_named - acc_uninf, the step table
(path models), drift channels, run class from the training losses.

Usage: python3 -m mapformer.tw_landmark_eval --runs RUN_DIR [RUN_DIR ...] --out OUT.json [--n-walks 200]
       (each RUN_DIR holds <arm>.pt). Under rescore_hook it gives the dropout re-scored readouts (one-layer correction).
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from mapformer.environment_tw_landmark import TextWorldLandmark, render_conditions
from mapformer.environment_textworld import VERBS
from mapformer.stats_core import classify_run
from mapformer.train_variant import VARIANT_MAP

HELDOUT, EVAL_SEED, T = 10000, 10**6, 1024
VARIANT = {"MapWM": "Vanilla_r4", "RoPE": "RoPE"}
wrap = lambda x: (x + np.pi) % (2 * np.pi) - np.pi
MC = True          # --no-mc (the re-scored pass) skips the MC-dropout readouts


def load(ck):
    b = torch.load(ck, map_location="cpu", weights_only=False); a = b["config"]["args"]
    env = TextWorldLandmark(size=a["size"], seed=a["seed"], name_rate=a["name_rate"])
    m = VARIANT_MAP[VARIANT[a["arm"]]](vocab_size=env.unified_vocab_size, d_model=128, n_heads=2,
                                       n_layers=a["n_layers"], grid_size=a["size"])
    m.load_state_dict(b["model_state_dict"]); m.eval()
    assert len(m.layers) == a["n_layers"]
    return m, a, b["losses"]


def conds_for(r):
    return {"own": (r, "consistent"), "strip": (0.0, "consistent"), "uninf": (r, "fresh"),
            "named": (1.0, "consistent"), "conf": (1.0, "conflict")}


def build_eval(r, n_walks, n_tokens=T, map_seed=HELDOUT):
    """-> {cond: (tokens (N, T), targets)} plus '_slots' ({cond: per walk, the rendering's slots}); targets: list of
    (walk, pos, true tok, landmark-at-rate-r, conflict, alt obj, landmark-at-rate-0.5) for revisit slots with move
    index < K1. The last field (Amendment 1, D1) is defined for every training rate by the coupled draws: the
    in-distribution probe scores the 'own' rendering on cells that are NOT landmarks at rate 0.5. CPU only,
    deterministic."""
    env = TextWorldLandmark(size=64, seed=map_seed)
    cd = conds_for(r); keys = list(dict.fromkeys(list(cd.values()) + [(0.5, "consistent")]))
    np.random.seed(EVAL_SEED)
    out = {c: ([], []) for c in cd}; slots_all = {c: [] for c in cd}
    for w in range(n_walks):
        R = render_conditions(env, n_tokens, keys)
        K1 = len(R[(1.0, "consistent")][3])
        land = {s[1]: s[3] for s in R[(r, "consistent")][3]}
        land05 = {s[1]: s[3] for s in R[(0.5, "consistent")][3]}
        for c, key in cd.items():
            t, _o, _rv, slots = R[key]
            out[c][0].append(t); slots_all[c].append(slots)
            for (pos, k, cell, _l, rev, conf, alt) in slots:
                if rev and k < K1:
                    out[c][1].append((w, pos, int(t[pos]), bool(land[k]), conf, alt, bool(land05[k])))
    E = {c: (torch.stack(v[0]), v[1]) for c, v in out.items()}
    E["_slots"] = slots_all
    return E


class DirMean:
    """Forward hook on action_to_lie. mode None: unchanged. mode 'dir': every direction word's step replaced by the
    mean step of the 12 direction words (theta reliance). mode 'all': also every name token and the mark by the mean
    over names + mark (rel_all). The means are model constants (unweighted over the ids)."""
    def __init__(self, m, dirs, names):
        self.mode, self.tok = None, None
        with torch.no_grad():
            def grp(ids):
                ids_t = torch.tensor(ids)
                return ids_t, m.action_to_lie(m.token_emb(ids_t[None]))[0].mean(0)      # (H, n_blocks)
            self.groups = {"dir": [grp(dirs)], "all": [grp(dirs), grp(names)]}
        self.h = m.action_to_lie.register_forward_hook(self._hook)

    def _hook(self, mod, inp, out):
        if self.mode is None:
            return out
        out = out.clone()
        for ids_t, mean in self.groups[self.mode]:
            msk = torch.isin(self.tok, ids_t.to(self.tok.device))
            out[msk] = mean.to(out.dtype).to(out.device)
        return out


@torch.no_grad()
def predict(m, toks, dev, bs=20, hook=None):
    """-> argmax prediction (N, T-1) at every input position; the hook (if any) is told the batch's tokens."""
    preds = []
    for i in range(0, len(toks), bs):
        x = toks[i:i + bs, :-1].to(dev)
        if hook is not None:
            hook.tok = x
        preds.append(m(x).argmax(-1).cpu())
    return torch.cat(preds)


def acc_of(pred, targets, sel=lambda t: True):
    ok = [int(pred[t[0], t[1] - 1]) == t[2] for t in targets if sel(t)]
    return (float(np.mean(ok)) if ok else float("nan")), len(ok)


@torch.no_grad()
def step_table(m, env):
    ids = torch.arange(env.unified_vocab_size)[None]
    D = m.action_to_lie(m.token_emb(ids))[0].reshape(env.unified_vocab_size, -1).numpy()
    om = m.path_integrator.omega.detach().numpy().reshape(-1)
    X = D * om[None]                                           # omega-scaled (gauge-free in the (Delta, omega) scale)
    n = np.linalg.norm(X, axis=1)
    dirs = [i for a in range(4) for i in env.dir_ids[a]]
    names = env.name_ids
    base_other = [i for i in range(env.base_vocab_size) if i not in dirs]
    A = {a: X[env.dir_ids[a]].mean(0) for a in range(4)}; c = np.mean([A[a] for a in range(4)], 0)
    Rr = {a: A[a] - c for a in range(4)}
    opp = (np.linalg.norm(Rr[0] + Rr[1]) / max(np.linalg.norm(Rr[0]), 1e-12)
           + np.linalg.norm(Rr[2] + Rr[3]) / max(np.linalg.norm(Rr[2]), 1e-12)) / 2
    nm = X[names]; nmean = nm.mean(0)
    dn = n[dirs].mean()
    return {"move": float(n[base_other].mean() / dn), "opp_minus_common": float(opp),
            "name_step": float(n[names].mean() / dn),
            "name_identity_step": float(np.linalg.norm(nm - nmean[None], axis=1).mean() / dn),
            "mark_step": float(n[env.mark_id] / dn), "common_over_dir": float(np.linalg.norm(c) / dn)}


@torch.no_grad()
def drift(m, toks, targets_slots, thr=1.0):
    """Clock readout (tw_normstep_readouts.drift's 'full' count, on the given rendering): for every revisit, the
    wrapped phase difference omega * cumsum(step) between the revisit's object slot and the first visit's, mean |.|
    per channel over revisit pairs; count of channels > thr rad, and the mean in rad."""
    om = m.path_integrator.omega.detach().numpy()
    dif = []
    for w, (tok, slots) in enumerate(zip(toks, targets_slots)):
        d = m.action_to_lie(m.token_emb(tok[None]))[0].numpy()
        th = np.cumsum(d, 0) * om[None]
        first = {}
        for (pos, k, cell, *_r) in slots:
            if cell in first:
                dif.append(wrap(th[pos] - th[first[cell]]))
            else:
                first[cell] = pos
    md = np.abs(np.array(dif)).mean(0)
    return {"drift": int((md > thr).sum()), "drift_rad": float(md.mean()), "channels": int(md.size)}


def ckpt_md5(run_dir):
    arm = [f for f in os.listdir(run_dir) if f.endswith(".pt")]
    assert len(arm) == 1, (run_dir, arm)
    return hashlib.md5(open(os.path.join(run_dir, arm[0]), "rb").read()).hexdigest()


def readouts(run_dir, n_walks, dev, cache):
    arm = [f for f in os.listdir(run_dir) if f.endswith(".pt")]
    assert len(arm) == 1, (run_dir, arm)
    m, a, losses = load(os.path.join(run_dir, arm[0])); m.to(dev)
    r = float(a["name_rate"])
    if r not in cache:
        cache[r] = build_eval(r, n_walks)
    E = cache[r]
    env = TextWorldLandmark(size=64, seed=HELDOUT)
    c = classify_run(losses)
    out = {"arm": a["arm"], "n_layers": a["n_layers"], "name_rate": r, "seed": a["seed"], "cls": c["registered"],
           "tail": float(c["tail"]), "final_loss": float(np.mean(losses[-max(1, len(losses) // 20):])),
           "n_targets": len(E["own"][1]), "n_walks": n_walks, "ckpt_md5": ckpt_md5(run_dir)}
    path = hasattr(m, "action_to_lie")
    dirs = [i for d in range(4) for i in env.dir_ids[d]]
    hook = DirMean(m, dirs, env.name_ids + [env.mark_id]) if path else None
    for cond in conds_for(r):
        toks, tg = E[cond]
        p = predict(m, toks, dev, hook=hook)
        out[f"acc_{cond}"], _ = acc_of(p, tg)
        if cond == "own":
            out["acc_own_land"], out["n_own_land"] = acc_of(p, tg, lambda t: t[3])
            out["acc_own_unnamed"], out["n_own_unnamed"] = acc_of(p, tg, lambda t: not t[3])
            # Amendment 1 (D1): the in-distribution probe -- own rendering, cells unnamed at rate 0.5
            out["acc_own_u05"], out["n_own_u05"] = acc_of(p, tg, lambda t: not t[6])
        if cond == "conf":
            # conflict targets with two DIFFERENT, NON-BLANK objects: with a blank on either side the base rate of
            # 'nothing' alone makes a constant predictor look like a name- or path-follower (e2e test: 0.31 vs 0.52)
            nothing = env.idx["nothing"]
            cf = [t for t in tg if t[4] and t[5] != t[2] and t[2] != nothing and t[5] != nothing]
            out["n_conf"] = len(cf)
            out["conf_path"] = float(np.mean([int(p[t[0], t[1] - 1]) == t[2] for t in cf])) if cf else float("nan")
            out["conf_name"] = float(np.mean([int(p[t[0], t[1] - 1]) == t[5] for t in cf])) if cf else float("nan")
        if path and cond in ("own", "strip", "uninf", "named"):
            for mode, key in (("dir", f"rel_{cond}"), ("all", f"rel_all_{cond}")):
                if mode == "all" and cond != "uninf":
                    continue
                hook.mode = mode
                q = predict(m, toks, dev, hook=hook)
                hook.mode = None
                out[key] = out[f"acc_{cond}"] - acc_of(q, tg)[0]
                if cond == "own" and mode == "dir":
                    out["rel_own_u05"] = out["acc_own_u05"] - acc_of(q, tg, lambda t: not t[6])[0]
        if cond in ("own", "strip") and MC:
            # Amendment 1 (D3): MC-dropout accuracy (train mode, all dropout on, mean of 3 dropout seeds): valid at any
            # depth, unlike the one-layer x1/(1-p) re-score. Read it from the PLAIN pass (rescore_hook's pre-hooks stay
            # on in train mode, so the re-scored pass's MC values are double-scaled and not used).
            mc = []
            for k in range(3):
                m.train(); torch.manual_seed(k)
                mc.append(acc_of(predict(m, toks, dev), tg)[0])
            m.eval()
            out[f"acc_{cond}_mc"] = float(np.mean(mc))
    out["name_benefit"] = out["acc_named"] - out["acc_uninf"]
    if path:
        hook.h.remove(); m.cpu()
        out.update(step_table(m, env))
        out.update(drift(m, E["own"][0][:40], E["_slots"]["own"][:40]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-walks", type=int, default=200)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--no-mc", action="store_true")
    a = ap.parse_args()
    global MC
    MC = not a.no_mc
    torch.set_num_threads(a.threads)
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    cache = {}
    for rd in a.runs:
        key = os.path.basename(rd.rstrip("/"))
        # Amendment 1 (N5): reuse a stored readout only if it was computed from THIS checkpoint with these walks
        if key in res and res[key].get("ckpt_md5") == ckpt_md5(rd) and res[key].get("n_walks") == a.n_walks:
            continue
        res[key] = readouts(rd, a.n_walks, a.device, cache)
        json.dump(res, open(a.out + ".tmp", "w"), indent=1); os.replace(a.out + ".tmp", a.out)
        r = res[key]
        print(f"{key}: own {r['acc_own']:.4f} strip {r['acc_strip']:.4f} uninf {r['acc_uninf']:.4f} "
              f"named {r['acc_named']:.4f} rel_uninf {r.get('rel_uninf', float('nan')):.4f} {r['cls']}", flush=True)


if __name__ == "__main__":
    main()
