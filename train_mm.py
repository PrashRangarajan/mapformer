"""Train, score and probe one Memory Maze run (MEMORY_MAZE_PREREG.md). Resumable per epoch.

  python3 -m mapformer.train_mm --arm P_E --seed 3600 --data <array dir> --out <run dir> [--device cuda]

Run dir: ckpt.pt (model, optimiser, scheduler, epoch, history; replaced atomically every epoch), log.txt,
results.json (written last, atomically) and DONE (marker after results.json). A run with DONE is skipped.
Randomness: data order and the training blind-phase draws come from numpy generators seeded by (seed, epoch, batch),
so a resumed run sees the same data as an uninterrupted one; dropout is 0 (no RNG in the forward pass). On GPU the
fp32 SDPA backward is not bitwise deterministic, so a resumed run is a statistical, not bitwise, continuation (the
CPU resume test in test_mm is exact). TF32 is OFF for matmuls and cuDNN (the GRU arm is fp32 too; Amendment 2).
Provenance: md5 of every code file at START (config.md5_start) and END (md5); a mismatch is recorded (md5_changed).

Training: train-part0 indices 0-1599 (mm_data.TRAIN_RANGE), batch 16 episodes, blind schedule B 16 / S 16 with a random
phase per sample, InfoNCE: the query at s_t against the keys of all 1001 frames of the episode, positive f_t, frames
u != t in the same pose bin as t removed from the negatives, averaged over sighted steps PLUS averaged over post-blind
steps (`--loss-steps both`, the default since Amendment 1, pilot-informed). 600 epochs (Amendment 1). AdamW, lr 3e-4,
weight decay 0.05 on all parameters, linear warmup over 2% of steps then cosine to 0, grad-norm clip 1.0, fp32
(step features stored fp32 since Amendment 2). Arm P_E_frozen installs the fitted E map frozen (Amendment 2).
Scoring: the gate's schedule (mm_data.sighted_mask, phase from the archive index) on the official eval split
(primary) and on train-part0 1600-2899 (held-out secondary); readouts R1 / R2 on the gate's scored blind re-entries.
"""
import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import mapformer.model as MM
from mapformer import mm_data as D
from mapformer import model_mm as M

CODE_FILES = ("train_mm.py", "model_mm.py", "mm_data.py", "mm_gate.py", "mm_gate_v2.py", "model.py", "lie_groups.py")
HERE = os.path.dirname(os.path.abspath(__file__))


def _fp32_precision(op):
    """cuDNN fp32 precision for op from the C++ state (building nn.GRU imports torch.backends.cudnn.rnn, which shadows
    the Python attribute torch.backends.cudnn.rnn but not the setting itself)."""
    try:
        return torch._C._get_fp32_precision_getter("cuda", op)
    except Exception:
        return None


def md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


# ---------------------------------------------------------------- data
class Split:
    """Rows [lo, hi) of a prepared split, as device tensors. arch_idx = archive index of each row (the gate's ep_idx)."""

    def __init__(self, data_dir, split, lo, hi, device, need_tvec=False):
        S = D.load_split(data_dir, split)
        hi = min(hi, S["emb"].shape[0])
        self.name, self.lo, self.n = split, lo, hi - lo
        self.arch_idx = np.arange(lo, hi)
        act = np.asarray(S["act"][lo:hi]).astype(np.int64)
        pos = np.asarray(S["pos"][lo:hi]).astype(np.float64)
        dr = np.asarray(S["dir"][lo:hi]).astype(np.float64)
        self.act, self.pos, self.dir = act, pos, dr
        self.emb = torch.from_numpy(np.array(S["emb"][lo:hi])).to(device)                       # fp16
        e = np.stack([D.e_feature(a, d_) for a, d_ in zip(act, dr)])
        self.feats = {
            "e": torch.from_numpy(e).to(device=device, dtype=torch.float32),
            "odo": torch.from_numpy(np.stack([D.odo_feature(p) for p in pos])).to(device),
            "raw": torch.from_numpy(np.stack([D.raw_feature(a) for a in act])).to(device=device, dtype=torch.float32),
        }
        pb = [D.pose_bins(p, d_) for p, d_ in zip(pos, dr)]
        self.pose_np = np.stack([x[0] for x in pb]); self.cid_np = np.stack([x[1] for x in pb])
        self.pose = torch.from_numpy(self.pose_np).to(device)
        self.layout = torch.from_numpy(np.asarray(S["layout"][lo:hi]).astype(np.float32)).to(device) if need_tvec else None
        self.tvec = torch.from_numpy(np.array(S["tvec"][lo:hi])).to(device) if need_tvec else None
        self._rows = None
        self.device = device

    def batch(self, idx, sighted_np):
        dev = self.device
        feats = {k: v[idx].float() for k, v in self.feats.items()}
        return self.emb[idx].float(), torch.from_numpy(sighted_np).to(dev), feats, self.pose[idx]

    def gate_masks(self):
        return np.stack([D.sighted_mask(1001, int(a)) for a in self.arch_idx])

    def rows(self):
        """Scored blind re-entries per row under the gate's schedule (cached)."""
        if self._rows is None:
            sm = self.gate_masks()
            self._rows = [D.scored_steps(self.pose_np[i], self.cid_np[i], sm[i]) for i in range(self.n)]
        return self._rows


def info_nce(logits, pose, sighted, steps="sighted"):
    """logits (B, 1000, 1001); pose (B, 1001); sighted (B, 1001). CE with same-pose negatives masked, averaged over:
    steps="sighted": every sighted t >= 1 (the loss as first registered; failed in pilot wave 1);
    steps="both":    (the registered default since Amendment 1) mean over sighted steps + mean over post-blind steps (first sighted step after a blind run), so
                     the steps that need memory carry as much weight as all the others together."""
    B, Tq, Tk = logits.shape
    pt = pose[:, 1:]
    same = pt.unsqueeze(-1) == pose.unsqueeze(1)                              # (B, 1000, 1001)
    tgt = torch.arange(1, Tk, device=logits.device)
    same[:, torch.arange(Tq, device=logits.device), tgt] = False               # the positive stays
    lg = logits.masked_fill(same, float("-inf"))
    ce = F.cross_entropy(lg.reshape(-1, Tk), tgt.repeat(B), reduction="none").view(B, Tq)
    w = sighted[:, 1:].float()
    loss = (ce * w).sum() / w.sum().clamp_min(1.0)
    if steps == "both":
        pb = (sighted[:, 1:] & ~sighted[:, :-1]).float()
        loss = loss + (ce * pb).sum() / pb.sum().clamp_min(1.0)
    return loss


# ---------------------------------------------------------------- scoring
@torch.no_grad()
def score(model, sp, bs, mode="normal"):
    """Per scored step: R1 (nearest frame's pose bin = target's) and R2 (target beats every other-pose frame).
    mode: normal | frames_swapped (input frames from the next episode in the batch, keys own) | steps_shuffled."""
    model.eval()
    sm_all = sp.gate_masks()
    rows = sp.rows()
    out = []
    g = np.random.default_rng(12345)
    for i0 in range(0, sp.n, bs):
        idx = np.arange(i0, min(i0 + bs, sp.n))
        emb, sighted, feats, pose = sp.batch(idx, sm_all[idx])
        emb_in = emb
        if mode == "frames_swapped":
            emb_in = torch.roll(emb, 1, dims=0) if len(idx) > 1 else emb.flip(1)
        if mode == "steps_shuffled":
            perm = torch.from_numpy(np.concatenate([[0], 1 + g.permutation(1000)])).to(emb.device)
            feats = {k: v[:, perm] for k, v in feats.items()}
        out_h = model.hidden(emb_in, sighted, feats)
        logits = model.io.logits(out_h, emb)
        for j, i in enumerate(idx):
            P = sp.pose_np[i]
            lg = logits[j].float().cpu().numpy()
            for r in rows[i]:
                t = r["t"]; l = lg[t - 1]
                r1 = P[int(np.argmax(l))] == P[t]
                other = P != P[t]
                r2 = bool(l[t] > l[other].max())
                out.append((int(i), t, r["gap"], r["back"], r["nonlocal_"], bool(r1), r2))
    return out


def summarise(sc):
    a = np.array([x[4:] for x in sc], dtype=bool) if sc else np.zeros((0, 3), bool)
    gap = np.array([x[2] for x in sc]) if sc else np.zeros(0)
    res = {}
    for st, m in (("nonlocal", a[:, 0] if len(a) else a), ("all", np.ones(len(a), bool)), ("longgap", gap >= D.LONG_GAP)):
        n = int(m.sum())
        res[st] = dict(n=n, R1=float(a[m, 1].mean()) if n else float("nan"), R2=float(a[m, 2].mean()) if n else float("nan"))
    per_ep = {}
    for x in sc:
        if x[4]:
            per_ep.setdefault(x[0], []).append(x[5])
    res["nonlocal_episode_mean_R1"] = float(np.mean([np.mean(v) for v in per_ep.values()])) if per_ep else float("nan")
    return res


# ---------------------------------------------------------------- probe (Pasukonis et al. 2022 / Deng et al. 2023)
class ProbeMLP(nn.Module):
    def __init__(self, din, dout, width=1024, depth=4):
        super().__init__()
        layers, d = [], din
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.LayerNorm(width, eps=1e-3), nn.ELU()]   # Table E.3 LayerNorm eps
            d = width
        layers.append(nn.Linear(d, dout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def states(model, sp, bs, blinded):
    """Probe states at frame tokens f_0..f_1000 (model.probe_states: final hidden after the last LayerNorm for transformer
    arms, the 184-d GRU hidden for the GRU): (n, 1001, d), fp16 on device."""
    model.eval()
    sm = sp.gate_masks() if blinded else np.ones((sp.n, 1001), bool)
    hs = []
    for i0 in range(0, sp.n, bs):
        idx = np.arange(i0, min(i0 + bs, sp.n))
        emb, sighted, feats, _ = sp.batch(idx, sm[idx])
        hs.append(model.probe_states(emb, sighted, feats)[:, 0::2].half())
    return torch.cat(hs)


def probe_inputs(H, sp, ii, tt):
    pos = torch.from_numpy(sp.pos[ii, tt]).float().to(H.device)
    dr = torch.from_numpy(sp.dir[ii, tt]).float().to(H.device)
    return torch.cat([H[ii, tt].float(), pos, dr], dim=-1)


def run_probe(Htr, sp_tr, Hev, sp_ev, steps, seed, log):
    """Walls (maze_layout, BCE, accuracy over tiles) and Objects (targets_vec, MSE) probes on frozen states plus
    agent_pos / agent_dir; trained on training-split states (all t), scored on eval steps 501-1000 (the second half,
500 steps). Table E.3 epsilons (Adam 1e-5, LayerNorm 1e-3) since Amendment 2."""
    dev = Htr.device
    torch.manual_seed(seed)
    g = np.random.default_rng(seed)
    din = Htr.shape[-1] + 4
    res = {}
    ev_i = np.repeat(np.arange(sp_ev.n), 500); ev_t = np.tile(np.arange(501, 1001), sp_ev.n)   # second half: 500 steps
    for target in ("walls", "objects"):
        dout = 225 if target == "walls" else 12
        net = ProbeMLP(din, dout).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=0.0, eps=1e-5)   # Table E.3 Adam eps
        for s in range(steps):
            ii = g.integers(0, sp_tr.n, 1024); tt = g.integers(0, 1001, 1024)
            x = probe_inputs(Htr, sp_tr, ii, tt)
            if target == "walls":
                y = sp_tr.layout[ii].reshape(1024, 225)
                loss = F.binary_cross_entropy_with_logits(net(x), y)
            else:
                y = sp_tr.tvec[ii, tt].reshape(1024, 12)
                loss = F.mse_loss(net(x), y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        net.eval()
        acc_sum, sq_obj, sq_coord, n = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for k in range(0, len(ev_i), 8192):
                ii, tt = ev_i[k:k + 8192], ev_t[k:k + 8192]
                p = net(probe_inputs(Hev, sp_ev, ii, tt))
                if target == "walls":
                    y = sp_ev.layout[ii].reshape(len(ii), 225)
                    acc_sum += float(((p > 0).float() == y).float().mean(1).sum())
                else:
                    y = sp_ev.tvec[ii, tt].reshape(len(ii), 6, 2)
                    e2 = (p.view(len(ii), 6, 2) - y) ** 2
                    sq_obj += float(e2.sum(-1).mean(-1).sum()); sq_coord += float(e2.mean((1, 2)).sum())
                n += len(ii)
        if target == "walls":
            res["walls_acc"] = 100.0 * acc_sum / n
        else:
            res["objects_mse_sqdist"] = sq_obj / n          # mean over objects of the squared distance
            res["objects_mse_coord"] = sq_coord / n         # mean over objects and coordinates
        log(f"probe {target}: {res}")
    return res


def constant_probe(sp_tr, sp_ev):
    """Training-set mean predictions (the published 'constant baseline'), scored like the probes."""
    lay_mean = sp_tr.layout.mean(0).reshape(225)
    y = sp_ev.layout.reshape(sp_ev.n, 225)
    walls = 100.0 * float(((lay_mean > 0.5).float() == y).float().mean())
    tv_mean = sp_tr.tvec.reshape(-1, 6, 2).mean(0)
    ev = sp_ev.tvec[:, 501:1001]
    e2 = (ev - tv_mean) ** 2
    return dict(walls_acc=walls, objects_mse_sqdist=float(e2.sum(-1).mean()), objects_mse_coord=float(e2.mean()))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=M.ARMS)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=600)            # Amendment 1 (pilot-informed)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--train-range", type=int, nargs=2, default=list(D.TRAIN_RANGE))
    ap.add_argument("--held-range", type=int, nargs=2, default=list(D.HELD_RANGE))
    ap.add_argument("--eval-n", type=int, default=1000)
    ap.add_argument("--install", choices=["none", "frozen", "trainable"], default="none")
    ap.add_argument("--probe-steps", type=int, default=10000)
    ap.add_argument("--no-probe", action="store_true")
    ap.add_argument("--no-held", action="store_true")
    ap.add_argument("--eval-bs", type=int, default=16)
    ap.add_argument("--loss-steps", choices=["sighted", "both"], default="both")   # Amendment 1 (pilot-informed)
    ap.add_argument("--val-every", type=int, default=0,
                    help="PILOTS ONLY: every k epochs, R1 on non-local blind re-entries of --val-range (training-split "
                         "episodes excluded from --train-range; never a scoring split)")
    ap.add_argument("--val-range", type=int, nargs=2, default=[1500, 1600])
    ap.add_argument("--pilot", action="store_true",
                    help="PILOTS: no scoring on eval / held splits; the probe is scored on --val-range steps 500-1000; "
                         "the model-free constant probe baseline is still computed on eval (metric calibration)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    done = os.path.join(args.out, "DONE")
    if os.path.exists(done):
        print("DONE exists; skip"); return
    logf = open(os.path.join(args.out, "log.txt"), "a")

    def log(m):
        s = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {m}"
        print(s, flush=True); logf.write(s + "\n"); logf.flush()

    dev = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False      # Amendment 2: fp32 everywhere, GRU (cuDNN) included
    torch.backends.cudnn.allow_tf32 = False
    for k in ("rnn", "conv"):                          # torch >= 2.9 per-op precision API: say 'ieee' explicitly
        if hasattr(getattr(torch.backends.cudnn, k, None), "fp32_precision"):
            getattr(torch.backends.cudnn, k).fp32_precision = "ieee"
    md5_start = {f: md5_file(os.path.join(HERE, f)) for f in CODE_FILES}
    MM.USE_SDPA = True                     # fast attention; dropout 0, so no RNG difference (rule 19 note in prereg)
    torch.manual_seed(args.seed)
    model = M.build(args.arm).to(dev)
    if args.arm == "P_E_frozen":
        assert args.install in ("none", "frozen"), "P_E_frozen is the frozen install by definition"
        args.install = "frozen"
    if args.install != "none":
        W = D.e_map_as_matrix(np.load(os.path.join(args.data, "e_map.npy")))
        model.install_e_map(W, frozen=(args.install == "frozen"), seed=args.seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    tr = Split(args.data, "train0", *args.train_range, dev, need_tvec=not args.no_probe)
    nb = math.ceil(tr.n / args.batch)
    total = args.epochs * nb
    warm = max(1, int(0.02 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    ck = os.path.join(args.out, "ckpt.pt")
    ep0, hist, t_ep, val = 0, [], [], []
    vsp = None
    if args.val_every:
        assert args.val_range[0] >= args.train_range[1] and args.val_range[1] <= D.HELD_RANGE[0], "val must be unseen, unscored"
        vsp = Split(args.data, "train0", *args.val_range, dev)
    if os.path.exists(ck):
        st = torch.load(ck, map_location=dev)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        ep0, hist, t_ep, val = st["epoch"], st["hist"], st["t_ep"], st.get("val", [])
        log(f"resumed at epoch {ep0}")
    cfg = dict(vars(args), n_params=M.n_params(model), n_trainable=M.n_params(model, True), n_train=tr.n,
               steps_per_epoch=nb, total_steps=total, warmup=warm, sdpa=True, dropout=0.0,
               gru_hidden=getattr(model, "h", None), md5_start=md5_start,
               tf32=dict(matmul=torch.backends.cuda.matmul.allow_tf32, cudnn=torch.backends.cudnn.allow_tf32,
                         rnn=_fp32_precision("rnn")))
    log(f"config {json.dumps(cfg)}")
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    for ep in range(ep0, args.epochs):
        model.train()
        t0 = time.time()
        g = np.random.default_rng([args.seed, ep])
        order = g.permutation(tr.n)
        tot = 0.0
        for b in range(nb):
            idx = np.sort(order[b * args.batch:(b + 1) * args.batch])
            gb = np.random.default_rng([args.seed, ep, b])
            sm = np.stack([D.random_sighted_mask(1001, gb) for _ in idx])
            emb, sighted, feats, pose = tr.batch(idx, sm)
            logits, _ = model(emb, sighted, feats)
            loss = info_nce(logits, pose, sighted, args.loss_steps)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step()
            tot += loss.item()
        hist.append(tot / nb); t_ep.append(time.time() - t0)
        if vsp is not None and (ep + 1) % args.val_every == 0:
            v = summarise(score(model, vsp, args.eval_bs))
            val.append((ep + 1, v["nonlocal"]["R1"], v["all"]["R1"]))
            log(f"val epoch {ep+1}: nonlocal R1 {v['nonlocal']['R1']:.4f} all R1 {v['all']['R1']:.4f}")
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), epoch=ep + 1,
                        hist=hist, t_ep=t_ep, val=val), ck + ".tmp")
        os.replace(ck + ".tmp", ck)
        log(f"epoch {ep+1}/{args.epochs} loss {hist[-1]:.4f} ({t_ep[-1]:.1f}s)")
        if os.environ.get("MM_STOP_AFTER") and ep + 1 == int(os.environ["MM_STOP_AFTER"]) and ep + 1 < args.epochs:
            log("MM_STOP_AFTER: simulated preemption"); return      # test hook (test_mm resume check)
    peak = torch.cuda.max_memory_allocated(dev) / 2 ** 30 if dev.type == "cuda" else None

    # ---- scoring
    t0 = time.time()
    res = dict(config=cfg, hist=hist, val=val, sec_per_epoch=float(np.mean(t_ep)) if t_ep else None, peak_gib=peak)
    if not args.pilot:
        ev = Split(args.data, "eval", 0, args.eval_n, dev, need_tvec=not args.no_probe)
        res["eval"] = {m: summarise(score(model, ev, args.eval_bs, m)) for m in ("normal", "frames_swapped", "steps_shuffled")}
        res["eval_rows"] = score(model, ev, args.eval_bs)   # per-step rows (episode row, t, gap, back, nonlocal, R1, R2)
        log(f"eval: {json.dumps(res['eval']['normal'])}")
        if not args.no_held:
            held = Split(args.data, "train0", *args.held_range, dev)
            res["held"] = {"normal": summarise(score(model, held, args.eval_bs))}
            log(f"held: {json.dumps(res['held']['normal'])}")
            del held
    else:
        ev = Split(args.data, "train0", *args.val_range, dev, need_tvec=True)
    res["score_sec"] = time.time() - t0

    # ---- offline probe (standard secondary)
    if not args.no_probe:
        t0 = time.time()
        res["probe"] = {"constant": constant_probe(tr, ev)}
        if args.pilot:
            res["probe"]["constant_eval_calibration"] = constant_probe(tr, Split(args.data, "eval", 0, args.eval_n, dev,
                                                                                 need_tvec=True))
        for blinded in (True, False):
            Htr = states(model, tr, args.eval_bs, blinded)
            Hev = states(model, ev, args.eval_bs, blinded)
            res["probe"]["blinded" if blinded else "unblinded"] = run_probe(Htr, tr, Hev, ev, args.probe_steps,
                                                                           args.seed, log)
            del Htr, Hev
        res["probe_sec"] = time.time() - t0

    res["md5"] = {f: md5_file(os.path.join(HERE, f)) for f in CODE_FILES}
    res["md5_changed"] = sorted(f for f in CODE_FILES if res["md5"][f] != md5_start[f])
    if res["md5_changed"]:
        log(f"WARNING: code changed during the run: {res['md5_changed']}")
    res["env"] = dict(torch=torch.__version__, python=platform.python_version(),
                      gpu=torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu")
    with open(os.path.join(args.out, "results.json.tmp"), "w") as f:
        json.dump(res, f)
    os.replace(os.path.join(args.out, "results.json.tmp"), os.path.join(args.out, "results.json"))
    open(done, "w").close()
    log("done")


if __name__ == "__main__":
    main()
