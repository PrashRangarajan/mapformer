"""Frozen frame encoder and array prep for the Memory Maze batch (MEMORY_MAZE_PREREG.md).

  python3 -I mm_encoder.py prep --zips <dir with 15x15/{train-part0,eval}.zip> --out <array dir> [--device cuda]
  python3 -I mm_encoder.py prep --smoke --zips <dir with 15x15/train-part0.zip (tiny)> --out <dir> --device cpu

Stage 1: a convolutional autoencoder (64x64x3 -> 256 -> 64x64x3, MSE) trained on every frame of train-part0 indices
0-299 ONLY (mm_data.ENC_RANGE), fixed seed, then frozen. Not time-contrastive: a time-contrastive objective would learn
pose adjacency (review A-M1). Stage 2: every episode of train-part0 (0-2899) and eval (0-999) is embedded and written
as a few large arrays (mm_data docstring). Stage 3: the least-squares E map on train-part0 0-299 (rule-14 install only).
Each stage writes its artifact first and a marker last; a rerun skips finished stages (preemption-safe).
"""
import argparse
import hashlib
import io
import json
import os
import sys
import time
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import mm_data as D  # noqa: E402

EMB = 256
AE_SEED = 20261027
KEYS = ("image", "action", "agent_pos", "agent_dir", "maze_layout", "targets_vec")


class Encoder(nn.Module):
    def __init__(self, emb=EMB):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(), nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(), nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(),
            nn.Flatten(), nn.Linear(256 * 4 * 4, emb))

    def forward(self, x):          # x: (N, 3, 64, 64) float in [0, 1]
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, emb=EMB):
        super().__init__()
        self.fc = nn.Linear(emb, 256 * 4 * 4)
        self.net = nn.Sequential(
            nn.ReLU(), nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(), nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(), nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid())

    def forward(self, z):
        return self.net(self.fc(z).view(-1, 256, 4, 4))


def to_float(img_u8, device):
    return torch.from_numpy(np.ascontiguousarray(img_u8)).to(device).permute(0, 3, 1, 2).float().div_(255.0)


def iter_range(path, lo, hi, keys=KEYS):
    z = zipfile.ZipFile(path)
    names = sorted(x for x in z.namelist() if x.endswith(".npz"))
    hi = min(hi, len(names))
    for i in range(lo, hi):
        ep = np.load(io.BytesIO(z.read(names[i])))
        out = {k: ep[k] for k in keys}
        out["name"] = names[i]
        out["idx"] = i
        yield out


def n_members(path):
    return sum(1 for x in zipfile.ZipFile(path).namelist() if x.endswith(".npz"))


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def train_ae(train_zip, enc_range, out, device, epochs, batch, log):
    ck = os.path.join(out, "ae_ckpt.pt")
    final = os.path.join(out, "encoder.pt")
    if os.path.exists(final):
        log("AE: encoder.pt exists, skip"); return
    t0 = time.time()
    frames = np.concatenate([ep["image"] for ep in iter_range(train_zip, *enc_range, keys=("image",))])
    log(f"AE: {len(frames)} frames from train-part0 {enc_range}, {time.time()-t0:.0f}s")
    torch.manual_seed(AE_SEED)
    enc, dec = Encoder().to(device), Decoder().to(device)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-3)
    ep0, hist = 0, []
    if os.path.exists(ck):
        st = torch.load(ck, map_location=device)
        enc.load_state_dict(st["enc"]); dec.load_state_dict(st["dec"]); opt.load_state_dict(st["opt"])
        ep0, hist = st["epoch"], st["hist"]
        log(f"AE: resumed at epoch {ep0}")
    g = np.random.default_rng(AE_SEED)
    perms = [g.permutation(len(frames)) for _ in range(epochs)]   # fixed order, resume-exact
    for e in range(ep0, epochs):
        tot, n = 0.0, 0
        for i in range(0, len(frames), batch):
            x = to_float(frames[np.sort(perms[e][i:i + batch])], device)
            loss = ((dec(enc(x)) - x) ** 2).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tot += loss.item() * len(x); n += len(x)
        hist.append(tot / n)
        torch.save(dict(enc=enc.state_dict(), dec=dec.state_dict(), opt=opt.state_dict(), epoch=e + 1, hist=hist), ck)
        log(f"AE epoch {e+1}/{epochs} mse {hist[-1]:.5f} ({time.time()-t0:.0f}s)")
    torch.save(dict(enc=enc.state_dict(), dec=dec.state_dict(), hist=hist, seed=AE_SEED, enc_range=list(enc_range)), final)


@torch.no_grad()
def embed_split(zip_path, split, lo, hi, out, device, log, batch=1001):
    marker = os.path.join(out, f"{split}.done")
    if os.path.exists(marker):
        log(f"embed {split}: done, skip"); return
    st = torch.load(os.path.join(out, "encoder.pt"), map_location=device)
    enc = Encoder().to(device).eval(); enc.load_state_dict(st["enc"])
    dec = Decoder().to(device).eval(); dec.load_state_dict(st["dec"])
    n = min(hi, n_members(zip_path)) - lo
    T = 1001
    mm = {
        "emb": np.lib.format.open_memmap(os.path.join(out, f"{split}_emb.npy"), "w+", np.float16, (n, T, EMB)),
        "act": np.lib.format.open_memmap(os.path.join(out, f"{split}_act.npy"), "w+", np.int8, (n, T)),
        "pos": np.lib.format.open_memmap(os.path.join(out, f"{split}_pos.npy"), "w+", np.float32, (n, T, 2)),
        "dir": np.lib.format.open_memmap(os.path.join(out, f"{split}_dir.npy"), "w+", np.float32, (n, T, 2)),
        "layout": np.lib.format.open_memmap(os.path.join(out, f"{split}_layout.npy"), "w+", np.uint8, (n, 15, 15)),
        "tvec": np.lib.format.open_memmap(os.path.join(out, f"{split}_tvec.npy"), "w+", np.float32, (n, T, 6, 2)),
    }
    names, rec_mse, layout_const = [], [], True
    t0 = time.time()
    for i, ep in enumerate(iter_range(zip_path, lo, lo + n)):
        x = to_float(ep["image"], device)
        z = enc(x)
        mm["emb"][i] = z.half().cpu().numpy()
        if i < 50:
            rec_mse.append(float(((dec(z) - x) ** 2).mean()))
        a = ep["action"].argmax(1).astype(np.int8)
        a[ep["action"].sum(1) == 0] = -1
        mm["act"][i] = a
        mm["pos"][i] = ep["agent_pos"]; mm["dir"][i] = ep["agent_dir"]
        L = ep["maze_layout"]
        layout_const &= bool((L == L[0]).all())
        mm["layout"][i] = L[0]
        mm["tvec"][i] = ep["targets_vec"]
        names.append(ep["name"])
        if (i + 1) % 200 == 0:
            log(f"embed {split}: {i+1}/{n} ({time.time()-t0:.0f}s)")
    for v in mm.values():
        v.flush()
    with open(os.path.join(out, f"{split}_names.txt"), "w") as f:
        f.write("\n".join(names) + "\n")
    info = dict(split=split, lo=lo, n=n, recon_mse_first50=float(np.mean(rec_mse)), layout_constant=layout_const,
                first=names[0], last=names[-1])
    with open(os.path.join(out, f"{split}_info.json"), "w") as f:
        json.dump(info, f, indent=1)
    open(marker, "w").close()
    log(f"embed {split}: {info}")


def encoder_quality(out, enc_range, n_eval):
    """Pose leak of the frozen embedding (review A-M1), on EVAL frames:
    (a) cross-maze ridge probes from the embedding to heading bin (8) and to cell (225), fitted on train0 enc_range
        frames (every 5th), scored on the first n_eval eval episodes (every 5th frame), with majority / uniform chance;
    (b) single-frame pose identification within an episode (the gate's 'ident'): at every frame t >= 16 of the first
        n_eval eval episodes, the nearest earlier frame (s <= t - 16) in embedding L2 has the same pose bin."""
    tr, ev = D.load_split(out, "train0"), D.load_split(out, "eval")

    def feats(S, rows, step):
        X, H, C = [], [], []
        for i in rows:
            e = np.asarray(S["emb"][i], np.float32)[::step]
            pose, cid = D.pose_bins(np.asarray(S["pos"][i]), np.asarray(S["dir"][i]))
            X.append(e); H.append((pose % 8)[::step]); C.append(cid[::step])
        return np.concatenate(X), np.concatenate(H), np.concatenate(C)

    Xt, Ht, Ct = feats(tr, range(*enc_range), 5)
    mu, sd = Xt.mean(0), Xt.std(0) + 1e-6
    Z = np.hstack([(Xt - mu) / sd, np.ones((len(Xt), 1))]).astype(np.float64)
    G = Z.T @ Z + 1.0 * np.eye(Z.shape[1])
    Wh = np.linalg.solve(G, Z.T @ np.eye(8)[Ht]); Wc = np.linalg.solve(G, Z.T @ np.eye(225)[Ct])
    n_eval = min(n_eval, ev["emb"].shape[0])
    Xe, He, Ce = feats(ev, range(n_eval), 5)
    Ze = np.hstack([(Xe - mu) / sd, np.ones((len(Xe), 1))])
    q = dict(heading_acc=float((np.argmax(Ze @ Wh, 1) == He).mean()), heading_majority=float(np.bincount(He).max() / len(He)),
             cell_acc=float((np.argmax(Ze @ Wc, 1) == Ce).mean()), cell_uniform=1 / 225)
    hit, n = 0, 0
    for i in range(n_eval):
        e = np.asarray(ev["emb"][i], np.float32)
        pose, _ = D.pose_bins(np.asarray(ev["pos"][i]), np.asarray(ev["dir"][i]))
        d2 = (e * e).sum(1)[:, None] + (e * e).sum(1)[None] - 2 * e @ e.T
        for t in range(16, 1001):
            u = int(np.argmin(d2[: t - 15, t]))
            hit += pose[u] == pose[t]; n += 1
    q["single_frame_ident_R1"] = float(hit / max(n, 1))
    q["n_eval_episodes"] = n_eval
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep"])
    ap.add_argument("--zips", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    logf = open(os.path.join(args.out, "prep.log"), "a")

    def log(m):
        s = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {m}"
        print(s, flush=True); logf.write(s + "\n"); logf.flush()

    tz = os.path.join(args.zips, "15x15", "train-part0.zip")
    ez = os.path.join(args.zips, "15x15", "eval.zip")
    enc_range = (0, 1) if args.smoke else D.ENC_RANGE
    train_ae(tz, enc_range, args.out, args.device, 1 if args.smoke else args.epochs, args.batch, log)
    embed_split(tz, "train0", 0, 2 if args.smoke else 2900, args.out, args.device, log)
    if not os.path.exists(ez) and not args.smoke:
        raise SystemExit(f"missing {ez}: refusing to embed anything else as 'eval'")
    embed_split(ez if os.path.exists(ez) else tz, "eval", 0, 2 if args.smoke else 1000, args.out, args.device, log)
    emap = os.path.join(args.out, "e_map.npy")
    if not os.path.exists(emap):
        S = D.load_split(args.out, "train0")
        r = range(*enc_range)
        M = D.fit_e_map([S["act"][i] for i in r], [S["dir"][i] for i in r], [S["pos"][i] for i in r])
        np.save(emap, M)
        log(f"E map fitted on train0 {enc_range}")
    qual = os.path.join(args.out, "encoder_quality.json")
    if not os.path.exists(qual):
        q = encoder_quality(args.out, enc_range, 2 if args.smoke else 200)
        with open(qual, "w") as f:
            json.dump(q, f, indent=1)
        log(f"encoder quality: {q}")
    man = {f: md5(os.path.join(args.out, f)) for f in sorted(os.listdir(args.out))
           if f.endswith((".npy", ".pt", ".txt")) and f != "ae_ckpt.pt"}
    man["code"] = {f: md5(os.path.join(os.path.dirname(os.path.abspath(__file__)), f))
                   for f in ("mm_encoder.py", "mm_data.py", "mm_gate.py", "mm_gate_v2.py")}
    with open(os.path.join(args.out, "MANIFEST.json"), "w") as f:
        json.dump(man, f, indent=1)
    open(os.path.join(args.out, "PREP.done"), "w").close()
    log("prep done")


if __name__ == "__main__":
    main()
