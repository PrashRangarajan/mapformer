"""Build a byte-level Python corpus + genuine bracket annotations.

Drop-in for train_hourglass_enwik8.py's --data (a flat uint8 byte file), so the
code-modelling arm reuses the exact enwik8 scaffold and recipe.

Two things make this more than `cat *.py`:

1. SPLIT BY FILE, never by byte offset. Python source is full of near-duplicate
   boilerplate; a contiguous byte split would put a file's own idioms on both
   sides. Files are hashed and de-duplicated first, then assigned whole to one
   split.

2. Bracket annotations come from Python's OWN tokenizer, not a regex. A regex
   would count brackets inside strings, comments and f-strings, and would
   mis-assign every nesting depth after the first docstring. Rule 7: the gate
   must call the task code -- here the "task code" for what counts as a bracket
   is CPython's tokenizer, so we call it.

For each genuine closing-bracket OP token we record, as absolute byte offsets
into the split's stream:
    pos       byte offset of the closer
    kind      0=')' 1=']' 2='}'
    open_pos  byte offset of its matching opener
    depth     how many brackets are open at that moment (1 = outermost)
Distance is pos - open_pos. Eval filters to closers whose opener is inside the
same crop -- a closer whose opener the model never saw is not a memory test.

Usage:
  python3 -m mapformer.build_code_corpus --out-dir data --target-mb 100
"""

import argparse
import hashlib
import io
import os
import random
import site
import sysconfig
import tokenize
from pathlib import Path

import numpy as np

OPENERS = {"(": 0, "[": 1, "{": 2}
CLOSERS = {")": 0, "]": 1, "}": 2}


def source_roots():
    roots = [sysconfig.get_paths()["stdlib"]]
    try:
        roots.extend(site.getsitepackages())
    except Exception:
        pass
    seen, out = set(), []
    for r in roots:
        r = os.path.realpath(r)
        if r not in seen and os.path.isdir(r):
            seen.add(r)
            out.append(r)
    return out


def collect_files(roots, max_bytes=1_000_000):
    """Every .py under the roots, de-duplicated by content hash.

    max_bytes drops generated blobs (protobuf stubs, giant tables); they are
    real Python but their bracket statistics are not.
    """
    by_hash = {}
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d != "__pycache__"]
            for fn in filenames:
                if not fn.endswith(".py"):
                    continue
                p = os.path.join(dirpath, fn)
                try:
                    if os.path.getsize(p) > max_bytes:
                        continue
                    raw = open(p, "rb").read()
                except OSError:
                    continue
                if not raw:
                    continue
                h = hashlib.sha256(raw).hexdigest()
                if h not in by_hash:
                    by_hash[h] = p
    return sorted(by_hash.values())


def annotate(text):
    """Genuine bracket tokens for one source string.

    Returns (encoded_bytes, [(closer_byte_off, kind, opener_byte_off, depth)]).
    Byte offsets are into the returned bytes, so they stay exact for non-ASCII
    source: tokenize reports columns in CHARACTERS of the decoded line.
    """
    data = text.encode("utf-8")

    # Line starts must be computed by splitting on "\n" ONLY. str.splitlines()
    # also breaks on \x0b \x0c \x1c \x1d \x1e \x85 \u2028 \u2029 -- and FORM FEED
    # (\x0c) is common in Python source as a section separator, while CPython's
    # tokenizer does not treat it as a line break. Using splitlines() here put
    # every line number after a form feed off by one, which gate G2 caught as
    # 188/84,466 val annotations pointing at the wrong byte.
    lines = text.split("\n")
    line_start, off = [], 0
    for ln in lines:
        line_start.append(off)
        off += len(ln.encode("utf-8")) + 1      # +1 for the "\n" itself

    def boff(row, col):
        i = row - 1
        if i >= len(lines):
            return len(data)
        return line_start[i] + len(lines[i][:col].encode("utf-8"))

    stack, recs = [], []
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type != tokenize.OP:
            continue
        s = tok.string
        if s in OPENERS:
            stack.append((boff(*tok.start), OPENERS[s]))
        elif s in CLOSERS:
            if not stack:
                continue
            op_off, op_kind = stack.pop()
            if op_kind != CLOSERS[s]:
                continue          # malformed; skip rather than guess
            recs.append((boff(*tok.start), CLOSERS[s], op_off, len(stack) + 1))
    return data, recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data"))
    ap.add_argument("--target-mb", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--test-frac", type=float, default=0.05)
    args = ap.parse_args()

    roots = source_roots()
    print("roots:")
    for r in roots:
        print("   ", r)
    files = collect_files(roots)
    print(f"{len(files)} unique .py files after de-duplication")

    rng = random.Random(args.seed)
    rng.shuffle(files)

    budget = args.target_mb * 1024 * 1024
    kept, total, skipped = [], 0, 0
    for p in files:
        if total >= budget:
            break
        try:
            text = open(p, "r", encoding="utf-8").read()
            data, recs = annotate(text)
        except (UnicodeDecodeError, SyntaxError, tokenize.TokenError,
                IndentationError, ValueError, OSError):
            skipped += 1
            continue
        if not data:
            continue
        kept.append((p, data, recs))
        total += len(data)
    print(f"{len(kept)} files tokenized, {skipped} skipped (unparseable), "
          f"{total/1048576:.1f} MB")

    n = len(kept)
    n_val = int(n * args.val_frac)
    n_test = int(n * args.test_frac)
    splits = {"val": kept[:n_val],
              "test": kept[n_val:n_val + n_test],
              "train": kept[n_val + n_test:]}

    outdir = Path(args.out_dir)
    outdir.mkdir(exist_ok=True)
    manifest = {}
    for name, items in splits.items():
        stream, allrecs, cursor = [], [], 0
        for p, data, recs in items:
            for (cl, kind, op, depth) in recs:
                allrecs.append((cursor + cl, kind, cursor + op, depth))
            stream.append(data)
            cursor += len(data)
        blob = b"".join(stream)
        arr = np.frombuffer(blob, dtype=np.uint8)
        arr.tofile(outdir / f"code_{name}.bin")
        if allrecs:
            a = np.array(allrecs, dtype=np.int64)
            np.savez_compressed(outdir / f"code_{name}_brackets.npz",
                                pos=a[:, 0], kind=a[:, 1],
                                open_pos=a[:, 2], depth=a[:, 3])
        manifest[name] = {"files": [p for p, _, _ in items],
                          "bytes": len(blob), "closers": len(allrecs)}
        print(f"  {name:5s} {len(blob)/1048576:7.2f} MB  "
              f"{len(items):5d} files  {len(allrecs):8d} annotated closers")

    import json
    with open(outdir / "code_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {outdir}/code_{{train,val,test}}.bin + brackets + manifest")


if __name__ == "__main__":
    main()
