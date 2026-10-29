"""Dump the non-tensor content of checkpoints into one tracked sidecar per run dir.

Written for the D2 disk cleanup (2026-10-29, `docs/cleanup/D2_MANIFEST.md`): before a
checkpoint is deleted, everything in it that is not a weight tensor -- the per-epoch
training losses, the run config, the stored evaluation results, variant / seed / sizes --
is written to `runs/<top>/CKPT_META.jsonl`, where `<top>` is the first path component
under `runs/`. JSON lines, not `.json`, so that no aggregator globbing `*.json` in a run
dir picks it up.

Sidecar format (`"format": "mapformer-ckpt-meta/1"`), one JSON object per line:

    line 1:  {"format": ..., "written": "2026-10-29", "doc": "docs/cleanup/D2_MANIFEST.md",
              "dir": "runs/<top>", "note": ..., "n_entries": N}
    line 2+: {"path": "<path of the .pt relative to runs/<top>>",
              "md5": "<md5 of the .pt>", "bytes": <size>,
              "state_dicts": {"<key>": <number of tensor entries>, ...},
              "meta": {<every other top-level key, JSON-encoded>}}

Encoding of `meta`: Python floats are written with `repr`, so they round-trip exactly
(`json.loads` gives back the same float); NaN / Infinity are written as the bare tokens
`NaN` / `Infinity` (Python's json reads them); dict keys become strings (so the
`results` dict `{512: {...}}` reads back as `{"512": {...}}`); tuples become lists;
a tensor found outside a state dict is `{"__tensor__": [...], "dtype": ..., "shape": [...]}`
when it has at most 64 elements, else `{"__tensor_omitted__": [shape], "dtype": ...}`;
any other object is `{"__repr__": "<repr>"}`. A top-level value that is a mapping holding
a tensor of more than 64 elements (model / optimizer state) is not dumped; its entry
count is listed under `state_dicts`. A checkpoint that IS a bare state dict lists
`"<root>"` there. The weights are the only thing not kept.

Readers: `ckpt_sidecar.load_ckpt_meta(pt_path)` returns the `meta` dict of a deleted
checkpoint (or the live `.pt`'s content when it still exists).

Usage: python3 docs/cleanup/extract_ckpt_meta.py <file listing .pt paths relative to the repo>
       [--min-meta]  (write no sidecar for a dir of bare state dicts)
       [--check]     (re-read every sidecar entry and compare it with torch.load of the .pt)
"""
import argparse
import hashlib
import json
import math
import os
import sys
from collections import OrderedDict, defaultdict

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FORMAT = "mapformer-ckpt-meta/1"
SMALL = 64


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def _has_big_tensor(v):
    if torch.is_tensor(v):
        return v.numel() > SMALL
    if isinstance(v, dict):
        return any(_has_big_tensor(x) for x in v.values())
    if isinstance(v, (list, tuple)):
        return any(_has_big_tensor(x) for x in v)
    return False


def _count_tensors(v):
    if torch.is_tensor(v):
        return 1
    if isinstance(v, dict):
        return sum(_count_tensors(x) for x in v.values())
    if isinstance(v, (list, tuple)):
        return sum(_count_tensors(x) for x in v)
    return 0


def enc(v):
    if v is None or isinstance(v, (bool, int, str)):
        return v
    if isinstance(v, float):
        return v
    if torch.is_tensor(v):
        if v.numel() <= SMALL:
            return {"__tensor__": v.detach().cpu().tolist(), "dtype": str(v.dtype),
                    "shape": list(v.shape)}
        return {"__tensor_omitted__": list(v.shape), "dtype": str(v.dtype)}
    if isinstance(v, dict):
        return {str(k): enc(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [enc(x) for x in v]
    try:
        import numpy as np
        if isinstance(v, np.generic):
            return v.item()
        if isinstance(v, np.ndarray):
            return {"__ndarray__": v.tolist(), "dtype": str(v.dtype)}
    except ImportError:
        pass
    return {"__repr__": repr(v)}


def extract(ck):
    """-> (state_dicts, meta)."""
    sds, meta = OrderedDict(), OrderedDict()
    if not isinstance(ck, dict):
        if hasattr(ck, "state_dict"):
            sds["<module>"] = len(ck.state_dict())
            return sds, meta
        meta["<value>"] = enc(ck)
        return sds, meta
    root_tensors = 0
    for k, v in ck.items():
        if torch.is_tensor(v) and v.numel() > SMALL:
            root_tensors += 1
        elif isinstance(v, dict) and _has_big_tensor(v):
            sds[str(k)] = _count_tensors(v)
        elif torch.is_tensor(v) and all(torch.is_tensor(x) for x in ck.values()):
            root_tensors += 1          # a small tensor inside a bare state dict
        else:
            meta[str(k)] = enc(v)
    if root_tensors:
        sds["<root>"] = root_tensors
    return sds, meta


def top_of(rel):
    parts = rel.split("/")
    assert parts[0] == "runs" and len(parts) >= 3, rel
    return parts[1], "/".join(parts[2:])


def dump_jsonl(path, header, entries):
    header = OrderedDict(header, n_entries=len(entries))
    with open(path, "w") as f:
        f.write(json.dumps(header) + "\n")
        for k, v in entries.items():
            row = OrderedDict(path=k)
            row.update(v)
            f.write(json.dumps(row, separators=(",", ":"), allow_nan=True) + "\n")


def read_jsonl(path):
    """-> (header, {path: entry})."""
    with open(path) as f:
        lines = [json.loads(l) for l in f if l.strip()]
    entries = OrderedDict((r.pop("path"), r) for r in lines[1:])
    assert lines[0].get("n_entries") == len(entries), path
    return lines[0], entries


def same(a, b):
    if isinstance(a, float) and isinstance(b, float):
        return (math.isnan(a) and math.isnan(b)) or a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("listfile")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--min-meta", action="store_true",
                    help="skip writing a sidecar for a dir whose checkpoints hold no "
                         "non-tensor content at all (bare state dicts)")
    a = ap.parse_args()
    paths = [l.strip() for l in open(a.listfile) if l.strip()]
    by_top = defaultdict(OrderedDict)
    for rel in paths:
        top, sub = top_of(rel)
        by_top[top][sub] = rel
    if a.check:
        bad = n = 0
        for top, subs in by_top.items():
            side = os.path.join(REPO, "runs", top, "CKPT_META.jsonl")
            if not os.path.exists(side):
                print(f"[check] no sidecar for runs/{top} ({len(subs)} files)")
                continue
            H, E = read_jsonl(side)
            assert H["format"] == FORMAT
            if set(E) != set(subs):
                print(f"[check] KEY MISMATCH runs/{top}: sidecar {len(E)} vs list {len(subs)}")
                bad += 1
            for sub, rel in subs.items():
                e = E[sub]
                ck = torch.load(os.path.join(REPO, rel), map_location="cpu", weights_only=False)
                sds, meta = extract(ck)
                reread = json.loads(json.dumps(meta, allow_nan=True))
                ok = (same(e["meta"], reread) and e["state_dicts"] == dict(sds)
                      and e["md5"] == md5(os.path.join(REPO, rel))
                      and e["bytes"] == os.path.getsize(os.path.join(REPO, rel)))
                # the losses must be bit-identical to the checkpoint's own list
                if ok and isinstance(ck, dict) and "losses" in ck:
                    ok = same(e["meta"]["losses"], [float(x) for x in ck["losses"]])
                n += 1
                if not ok:
                    bad += 1
                    print(f"[check] MISMATCH {rel}")
        print(f"[check] {n} entries compared, {bad} mismatches")
        sys.exit(1 if bad else 0)

    for top, subs in by_top.items():
        entries = OrderedDict()
        any_meta = False
        for sub, rel in subs.items():
            p = os.path.join(REPO, rel)
            ck = torch.load(p, map_location="cpu", weights_only=False)
            sds, meta = extract(ck)
            any_meta |= bool(meta)
            entries[sub] = OrderedDict(md5=md5(p), bytes=os.path.getsize(p),
                                       state_dicts=sds, meta=meta)
        if a.min_meta and not any_meta:
            print(f"runs/{top}: {len(entries)} checkpoints, bare state dicts only -- no sidecar")
            continue
        header = OrderedDict(
            format=FORMAT, written="2026-10-29", doc="docs/cleanup/D2_MANIFEST.md",
            dir=f"runs/{top}",
            note=("Non-tensor content of the checkpoints listed on the following lines, dumped before "
                  "they were deleted; format in docs/cleanup/extract_ckpt_meta.py; read "
                  "with ckpt_sidecar.load_ckpt_meta()."))
        out = os.path.join(REPO, "runs", top, "CKPT_META.jsonl")
        dump_jsonl(out, header, entries)
        print(f"runs/{top}: {len(entries)} entries -> {os.path.relpath(out, REPO)} "
              f"({os.path.getsize(out)} bytes)")


if __name__ == "__main__":
    main()
