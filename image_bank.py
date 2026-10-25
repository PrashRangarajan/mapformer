"""Image bank for the image-observation torus (IMAGE_TORUS_PREREG.md): 16 classes, 28x28 uint8, train / test split.

Classes 0-9 are MNIST digits 0-9; classes 10-15 are six Fashion-MNIST classes (FASHION_CLASSES: trouser, dress,
sandal, sneaker, bag, ankle boot). 16 = GridWorld's default n_obs_types, so the class-label stream, the revisit mask
and every floor are those of the token task at the same seed (the token stream IS GridWorld's; the model swaps each
class token for an image of that class). The four top-wear classes (t-shirt, pullover, coat, shirt) are left out:
they are the confusable ones, and the bridge asks about the map, not about fine-grained perception.

Raw files: the canonical gzipped IDX files (md5s below = torchvision's), downloaded 2026-10-25 into
data/image_torus/raw/{mnist,fashion}/ (gitignored, 42 MB). The bank is cached as data/image_torus/bank16.npz
(88 MB, gitignored) and rebuilt from the raw files if absent; bank_md5() hashes the arrays, so a checkpoint records
exactly which images it saw. Parsing is plain numpy (torchvision is not installed and is not needed).

  python3 -m mapformer.image_bank          # build / verify the cache, print counts and md5
"""
import gzip
import hashlib
import os

import numpy as np
import torch

REPO = "/home/prashr/mapformer"
RAW = f"{REPO}/data/image_torus/raw"
CACHE = f"{REPO}/data/image_torus/bank16.npz"
N_CLASSES = 16
FASHION_CLASSES = (1, 3, 5, 7, 8, 9)          # trouser, dress, sandal, sneaker, bag, ankle boot -> classes 10..15
CLASS_NAMES = [str(d) for d in range(10)] + ["trouser", "dress", "sandal", "sneaker", "bag", "ankle_boot"]
RAW_MD5 = {
    "mnist/train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "mnist/train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "mnist/t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "mnist/t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
    "fashion/train-images-idx3-ubyte.gz": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
    "fashion/train-labels-idx1-ubyte.gz": "25c81989df183df01b3e8a0aad5dffbe",
    "fashion/t10k-images-idx3-ubyte.gz": "bef4ecab320f06d8554ea6380940ec79",
    "fashion/t10k-labels-idx1-ubyte.gz": "bb300cfdad3c16e7a12a480ee83cd310",
}


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _idx(path, magic):
    with gzip.open(path, "rb") as f:
        raw = f.read()
    m = int.from_bytes(raw[0:4], "big")
    assert m == magic, (path, m, magic)
    n = int.from_bytes(raw[4:8], "big")
    if magic == 2051:
        r, c = int.from_bytes(raw[8:12], "big"), int.from_bytes(raw[12:16], "big")
        a = np.frombuffer(raw, dtype=np.uint8, offset=16)
        assert a.size == n * r * c and (r, c) == (28, 28), (path, a.size, n, r, c)
        return a.reshape(n, r, c).copy()
    a = np.frombuffer(raw, dtype=np.uint8, offset=8)
    assert a.size == n, (path, a.size, n)
    return a.copy()


def _split(name):
    tag = "train" if name == "train" else "t10k"
    mi = _idx(f"{RAW}/mnist/{tag}-images-idx3-ubyte.gz", 2051)
    ml = _idx(f"{RAW}/mnist/{tag}-labels-idx1-ubyte.gz", 2049).astype(np.int64)
    fi = _idx(f"{RAW}/fashion/{tag}-images-idx3-ubyte.gz", 2051)
    fl = _idx(f"{RAW}/fashion/{tag}-labels-idx1-ubyte.gz", 2049).astype(np.int64)
    keep = np.isin(fl, FASHION_CLASSES)
    remap = {c: 10 + i for i, c in enumerate(FASHION_CLASSES)}
    imgs = np.concatenate([mi, fi[keep]])
    labs = np.concatenate([ml, np.array([remap[int(c)] for c in fl[keep]], dtype=np.int64)])
    order = np.argsort(labs, kind="stable")                  # sorted by class: class k is imgs[off[k]:off[k]+cnt[k]]
    return imgs[order], labs[order]


def build_cache(path=CACHE):
    for rel, want in RAW_MD5.items():
        got = _md5(f"{RAW}/{rel}")
        if got != want:
            raise SystemExit(f"{RAW}/{rel}: md5 {got} != canonical {want}")
    tr_i, tr_l = _split("train")
    te_i, te_l = _split("test")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez(tmp, train_images=tr_i, train_labels=tr_l, test_images=te_i, test_labels=te_l)
    os.replace(tmp, path)


def _arrays(path=CACHE):
    if not os.path.exists(path):
        build_cache(path)
    z = np.load(path)
    return {k: z[k] for k in ("train_images", "train_labels", "test_images", "test_labels")}


def bank_md5(path=CACHE):
    """md5 over the four arrays (bytes, in a fixed order): identifies the exact images, independent of npz framing."""
    a = _arrays(path); h = hashlib.md5()
    for k in ("train_images", "train_labels", "test_images", "test_labels"):
        h.update(k.encode()); h.update(np.ascontiguousarray(a[k]).tobytes())
    return h.hexdigest()


def load_split(split, path=CACHE):
    """{'images': uint8 (N,28,28), 'labels': int64 (N,), 'off': int64 (16,), 'cnt': int64 (16,)}, sorted by class."""
    assert split in ("train", "test"), split
    a = _arrays(path)
    imgs, labs = a[f"{split}_images"], a[f"{split}_labels"]
    cnt = np.bincount(labs, minlength=N_CLASSES)
    assert len(cnt) == N_CLASSES and (cnt > 0).all(), cnt
    off = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    assert (np.diff(labs) >= 0).all()
    return {"images": torch.from_numpy(imgs), "labels": torch.from_numpy(labs),
            "off": torch.from_numpy(off.astype(np.int64)), "cnt": torch.from_numpy(cnt.astype(np.int64))}


def load_banks(path=CACHE):
    return {"train": load_split("train", path), "test": load_split("test", path)}


def synthetic_banks(n_per_class=(5, 3), fill=(10, 200)):
    """Tiny constant-image banks for unit tests: every train image has value fill[0] + class, every test image
    fill[1] + class (so the encoder input reveals split and class)."""
    out = {}
    for split, n, v in (("train", n_per_class[0], fill[0]), ("test", n_per_class[1], fill[1])):
        labs = torch.arange(N_CLASSES).repeat_interleave(n)
        imgs = (v + labs).to(torch.uint8)[:, None, None].expand(-1, 28, 28).contiguous()
        cnt = torch.full((N_CLASSES,), n, dtype=torch.int64)
        out[split] = {"images": imgs, "labels": labs, "off": torch.arange(N_CLASSES) * n, "cnt": cnt}
    return out


if __name__ == "__main__":
    b = load_banks()
    for s in ("train", "test"):
        print(s, tuple(b[s]["images"].shape), "per class", b[s]["cnt"].tolist())
    print("bank md5", bank_md5(), "cache", CACHE, f"{os.path.getsize(CACHE) / 2**20:.1f} MiB")
