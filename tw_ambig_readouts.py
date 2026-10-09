"""Per-run readouts for TW_AMBIG_PREREG.md, from the checkpoint (device: --device, default cuda if free else cpu).
Every readout uses the held-out map (env seed 10000) and fixed walk streams; the arm's own stream type (tagged for the
oracle arms; identical random draws).

  acc_split    the registered eval walks (np seed 10**6, 200 walks, T=1024, eval mode): accuracy recomputed (must equal
               eval.json's within 0.002) and split by revisit gap: CONTAMINATED if a non-movement direction word lies
               between the most recent earlier visit of the target's cell and the target, else CLEAN. A context-free step
               is wrong by construction only on contaminated gaps.
  rescored     the same walks with every attention layer's o_proj input x 1/(1-p) (rescore_hook's correction, hooks on
               this model only; all layer classes here are model.WMTransformerLayer, a KNOWN class). A ONE-layer
               correction: for the 2-layer arms (HSR, CF2, RoPE2) it compounds and is not a better estimate.
  reliance     THETA RELIANCE (GAIN_PHASE's readout, text version): acc - acc with every position's step replaced by the
               mean step of its word (over the 200 walks), and every DIRECTION word's step (both roles) by the mean
               over all direction-word positions: theta then advances by a word-count clock and carries no position.
               First 100 walks. ~0 for a model that does not use theta.
  swap         decoy swap test (docs/audits/2026-09-27/swap_test.py's readout): at a direction word, set it to north in
               one copy and south in the other (keeping its role tag on tagged streams) and take the mean |difference|
               of omega * cumsum(step) over the 64 channels, 15 tokens later. Per class: move (pooled over frames) and
               each non-movement class (nat, lead_near, lead_far, trail_near, trail_far); ratio_c = nm_c / move
               (0 = non-movement uses ignored, 1 = treated as moves; 1.00 for context-free arms by construction).
               Walk stream np seed 321, 40 walks, up to 60 occurrences per class.
  atword       the same swap read AT the direction word (offset 0): gate-inside (the word's own step suppressed) vs
               cancel-later (the step is taken and undone at a later cue).
  drift        revisit phase comparison (tw_normstep_readouts.drift's readout on this task): 40 walks (np seed
               10**6 + 1); for each revisit of a cell vs its first visit, wrapped omega * (cumsum step) difference at the
               object slots. drift = channels (of 64) with mean |dtheta| > 1 rad (random ~ pi/2); nm_drift_rad = mean
               |dtheta| with the phase integrated over NON-MOVEMENT sentence positions only (the map drift the
               ambiguity causes); core_drift_rad = over all other positions.
  disp         per-move displacement in phase units, gauge-free: u_NS = (mean omega*Delta at move-role north words -
               south) / 2, u_WE likewise, from the walks. disp = (||u_NS|| + ||u_WE||) / 2 rad; sharp = share of the 64
               channels whose max(|u_NS|, |u_WE|) exceeds 0.5 rad (fine, cell-resolving channels) -- the 'compromise
               step' readout (does MapWM move its map to coarse channels that one wrong step barely shifts?).
`python3 -m mapformer.tw_ambig_readouts CKPT [device]` prints the dict.
"""
import json
import sys

import numpy as np
import torch

from mapformer.environment_tw_ambig import TextWorldAmbig, ALL_DIR, NM_CLASSES
from mapformer.model_tw_ambig import ARMS, build, steps
from mapformer.train_tw_ambig import EVAL_SEED, HELDOUT

wrap = lambda x: (x + np.pi) % (2 * np.pi) - np.pi
DIR_A = {w: a for a, ws in enumerate([ALL_DIR[0:3], ALL_DIR[3:6], ALL_DIR[6:9], ALL_DIR[9:12]]) for w in ws}


def pick_device():
    if not torch.cuda.is_available():
        return "cpu"
    free = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
    return f"cuda:{int(np.argmax(free))}"


def load(ck, dev):
    b = torch.load(ck, map_location="cpu", weights_only=False); a = b["config"]["args"]
    tagged = ARMS[b["arm"]][2]
    te = TextWorldAmbig(size=a["size"], seed=HELDOUT, p_nm=a["p_nm"], tag_roles=tagged)
    m = build(b["arm"], te, a["size"]); m.load_state_dict(b["model_state_dict"])
    return m.to(dev).eval(), b, te


def walk_set(te, n, seed, T=1024):
    np.random.seed(seed); W = []
    for _ in range(n):
        tok, obs, rev = te.generate_trajectory(T)
        W.append(dict(tok=tok, obs=obs.numpy(), rev=rev.numpy(), ctx=list(te.ctx), nm=np.array(te.nm_mask, bool),
                      locs=[tuple(map(int, l)) for l in te.visited_locations]))
    return W


def gap_labels(w):
    """Per object slot index: 'clean' / 'contam' for revisit targets (module docstring)."""
    nm_dir = set(i for i, r, f in w["ctx"] if r == "nm"); lab = {}; seen = 0; last = {}; k = 0
    for i in range(len(w["tok"])):
        if i in nm_dir:
            seen += 1
        if w["obs"][i] and k < len(w["locs"]):
            c = w["locs"][k]
            if w["rev"][i]:
                lab[i] = "contam" if seen > last.get(c, seen) else "clean"
            last[c] = seen; k += 1
    return lab


@torch.no_grad()
def acc_split(m, W, dev):
    out = {"clean": [], "contam": []}
    for w in W:
        tok = w["tok"].to(dev); pred = m(tok[None, :-1])[0].argmax(-1).cpu()
        for i, g in gap_labels(w).items():
            out[g].append(int(pred[i - 1]) == int(w["tok"][i]))
    a = out["clean"] + out["contam"]
    return {"acc": float(np.mean(a)), "acc_clean": float(np.mean(out["clean"])) if out["clean"] else None,
            "acc_contam": float(np.mean(out["contam"])) if out["contam"] else None,
            "n_clean": len(out["clean"]), "n_contam": len(out["contam"])}


@torch.no_grad()
def rescored(m, W, dev):
    hs = [L.o_proj.register_forward_pre_hook(lambda mod, a, s=1.0 / (1.0 - L.dropout.p): (a[0] * s,) + tuple(a[1:]))
          for L in m.layers]
    try:
        return acc_split(m, W, dev)["acc"]
    finally:
        for h in hs:
            h.remove()


@torch.no_grad()
def reliance(m, W, dev, te):
    if steps(m, W[0]["tok"][None].to(dev)) is None:
        return None
    base = (m.base_id if hasattr(m, "base_id") else torch.arange(te.unified_vocab_size, device=dev))
    dirs = torch.tensor(te.dir_word_ids, device=dev)
    S = torch.zeros(te.unified_vocab_size, *m.path_integrator.omega.shape, device=dev, dtype=torch.float64)
    C = torch.zeros(te.unified_vocab_size, device=dev, dtype=torch.float64)
    for w in W:
        tok = w["tok"].to(dev)[None, :-1]; d = steps(m, tok)[0].double(); b = base[tok[0]]
        S.index_add_(0, b, d); C.index_add_(0, b, torch.ones_like(b, dtype=torch.float64))
    mean = S / C.clamp_min(1)[:, None, None]
    isdir = torch.isin(torch.arange(te.unified_vocab_size, device=dev), dirs)
    dmean = S[isdir].sum(0) / C[isdir].sum()
    mean[isdir] = dmean
    state = {}

    def pre(mod, args):
        return (mean[state["b"]].to(args[0].dtype)[None],)
    h = m.path_integrator.register_forward_pre_hook(pre)
    try:
        ok = tot = 0
        for w in W:
            tok = w["tok"].to(dev); inp = tok[None, :-1]; state["b"] = base[inp[0]]
            pred = m(inp)[0].argmax(-1); msk = torch.as_tensor(w["rev"][1:], device=dev)
            ok += int((pred[msk] == tok[1:][msk]).sum()); tot += int(msk.sum())
    finally:
        h.remove()
    return ok / tot


@torch.no_grad()
def theta(m, tok):
    return (torch.cumsum(steps(m, tok), 1) * m.path_integrator.omega)


@torch.no_grad()
def swap(m, W, dev, te, offsets=(15, 0), n_max=60):
    """-> {f"{cls}@{off}": mean |dtheta|}, cls in move + NM_CLASSES; ratios added by the caller."""
    tagged = hasattr(m, "base_id"); idN, idS = te.idx["north"], te.idx["south"]
    kN, kS = ALL_DIR.index("north"), ALL_DIR.index("south")
    acc = {(c, o): [] for c in ["move"] + NM_CLASSES for o in offsets}
    for w in W:
        tok = w["tok"]
        for i, role, f in w["ctx"]:
            c = "move" if role == "move" else f
            if len(acc[(c, offsets[0])]) >= n_max or i < 5 or i + max(offsets) + 1 >= len(tok):
                continue
            x = tok[:i + max(offsets) + 1].clone(); y = x.clone()
            if role == "nm" and tagged:
                x[i] = te.tag_offset + kN; y[i] = te.tag_offset + kS
            else:
                x[i] = idN; y[i] = idS
            th = theta(m, torch.stack([x, y]).to(dev))
            for o in offsets:
                acc[(c, o)].append(float((th[0, i + o] - th[1, i + o]).abs().mean()))
    return {f"{c}@{o}": (float(np.mean(v)) if v else None) for (c, o), v in acc.items()} | \
        {f"n_{c}": len(acc[(c, offsets[0])]) for c in ["move"] + NM_CLASSES}


@torch.no_grad()
def drift_disp(m, W, dev, te, thr=1.0):
    om = m.path_integrator.omega.detach()
    dif = {"full": [], "nm": [], "core": []}
    dsum = {a: 0 for a in range(4)}; dcnt = {a: 0 for a in range(4)}
    for w in W:
        d = steps(m, w["tok"][None].to(dev))[0]
        nmm = torch.as_tensor(w["nm"], device=dev)[:, None, None].to(d.dtype)
        ths = {"full": (torch.cumsum(d, 0) * om).cpu().numpy(), "nm": (torch.cumsum(d * nmm, 0) * om).cpu().numpy(),
               "core": (torch.cumsum(d * (1 - nmm), 0) * om).cpu().numpy()}
        slots = np.nonzero(w["obs"])[0][:len(w["locs"])]; first = {}
        for k, i in enumerate(slots):
            L = w["locs"][k]
            if L in first:
                for key, th in ths.items():
                    dif[key].append(wrap(th[i] - th[first[L]]))
            else:
                first[L] = i
        dd = (d * om).cpu().numpy(); words = [te.vocab[int(t)] if int(t) < te.tag_offset else None for t in w["tok"]]
        for i, role, f in w["ctx"]:
            if role == "move":
                a = DIR_A[words[i]]; dsum[a] = dsum[a] + dd[i]; dcnt[a] += 1
    out = {}
    for key, v in dif.items():
        md = np.abs(np.array(v)).mean(0)
        out["drift" if key == "full" else f"{key}_drift_ch"] = int((md > thr).sum())
        out[f"{key}_drift_rad"] = float(md.mean())
    u = {a: dsum[a] / max(1, dcnt[a]) for a in range(4)}
    uNS, uWE = (u[0] - u[1]) / 2, (u[2] - u[3]) / 2
    out["disp"] = float((np.linalg.norm(uNS) + np.linalg.norm(uWE)) / 2)
    out["sharp"] = float((np.maximum(np.abs(uNS), np.abs(uWE)) > 0.5).mean())
    return out


def readouts(ck, dev=None, n_acc=200):
    dev = dev or pick_device()
    m, b, te = load(ck, dev)
    W = walk_set(te, n_acc, EVAL_SEED)
    r = {"arm": b["arm"], "seed": b["seed"], **acc_split(m, W, dev)}
    r["acc_rescored"] = rescored(m, W, dev)
    if steps(m, W[0]["tok"][None].to(dev)) is None:                      # index arms: accuracy readouts only
        return r
    r["acc_sub"] = reliance(m, W[:100], dev, te)
    r["reliance"] = acc_split(m, W[:100], dev)["acc"] - r["acc_sub"]
    sw = swap(m, walk_set(te, 40, 321), dev, te)
    r.update(sw)
    mv = sw["move@15"]
    r["move_change"] = mv
    for c in NM_CLASSES:
        r[f"ratio_{c}"] = (sw[f"{c}@15"] / mv) if (mv and sw[f"{c}@15"] is not None) else None
        r[f"ratio0_{c}"] = (sw[f"{c}@0"] / sw["move@0"]) if (sw["move@0"] and sw[f"{c}@0"] is not None) else None
    r.update(drift_disp(m, walk_set(te, 40, EVAL_SEED + 1), dev, te))
    if hasattr(m, "ctx_alpha"):
        r["alpha"] = float(m.ctx_alpha.detach().cpu())
    return r


if __name__ == "__main__":
    print(json.dumps(readouts(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None), indent=1))
