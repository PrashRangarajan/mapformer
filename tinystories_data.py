"""TinyStories (V2, GPT-4 split) as a WORD-level token stream, for the step-table question on natural text:
which tokens does a path-integrated LM choose to move its phase on?

Tokens: lowercase words (internal apostrophe kept: don't, i'll), numbers, every other non-space character alone
(punctuation, quotes), '\\n' as its own token, '<eos>' at each story end. Vocabulary: '<eos>', '<unk>', '\\n', then the
most frequent tokens of the TRAIN file, VOCAB in total. Output (uint16): data/tinystories/train.bin, val.bin (the
official valid file), vocab.json, stats.json.

  python3 -m mapformer.tinystories_data            # from /home/prashr; ~10-20 min on 16 processes
"""
import collections, json, os, re, sys
from multiprocessing import Pool
import numpy as np

REPO = "/home/prashr/mapformer"
D = f"{REPO}/data/tinystories"
TRAIN, VALID = f"{D}/TinyStoriesV2-GPT4-train.txt", f"{D}/TinyStoriesV2-GPT4-valid.txt"
VOCAB = 8192
SEP = "<|endoftext|>"
TOK = re.compile(r"[a-z]+(?:'[a-z]+)*|[0-9]+|\n|[^\sa-z0-9]")


def tokenize(story):
    s = re.sub(r"\n\s*\n+", "\n", story.strip().lower())       # collapse blank lines to one newline token
    return TOK.findall(s) + ["<eos>"]


def stories(path, chunk=4000):
    buf, n = [], 0
    with open(path, encoding="utf-8") as f:
        part = []
        for line in f:
            if line.strip() == SEP:
                st = "".join(part).strip()
                if st:
                    buf.append(st)
                part = []
                if len(buf) == chunk:
                    yield buf; buf = []
            else:
                part.append(line)
        st = "".join(part).strip()
        if st:
            buf.append(st)
    if buf:
        yield buf


def count_chunk(chunk):
    c = collections.Counter()
    for st in chunk:
        c.update(tokenize(st))
    return c


def encode_chunk(args):
    chunk, idx = args
    unk = idx["<unk>"]
    out = []
    for st in chunk:
        out.extend(idx.get(t, unk) for t in tokenize(st))
    return np.asarray(out, dtype=np.uint16)


def main():
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    with Pool(nproc) as pool:
        counts = collections.Counter()
        for c in pool.imap_unordered(count_chunk, stories(TRAIN), chunksize=1):
            counts.update(c)
        special = ["<eos>", "<unk>", "\n"]
        words = [w for w, _ in counts.most_common() if w not in special][: VOCAB - len(special)]
        itos = special + words
        idx = {w: i for i, w in enumerate(itos)}
        assert len(itos) <= 65535
        stats = {"vocab": len(itos), "train_types": len(counts), "train_tokens_counted": sum(counts.values())}
        covered = sum(counts[w] for w in itos if w in counts)
        stats["train_unk_rate"] = 1 - covered / stats["train_tokens_counted"]
        for name, path in (("val", VALID), ("train", TRAIN)):
            parts = pool.imap(encode_chunk, ((ch, idx) for ch in stories(path)), chunksize=1)
            arr = np.concatenate(list(parts))
            assert arr.max() < len(itos)
            arr.tofile(f"{D}/{name}.bin")
            stats[f"{name}_tokens"] = int(arr.size)
            stats[f"{name}_unk_rate"] = float((arr == idx["<unk>"]).mean())
            stats[f"{name}_stories"] = int((arr == idx["<eos>"]).sum())
            print(name, stats[f"{name}_tokens"], f"unk {stats[f'{name}_unk_rate']:.5f}", flush=True)
    json.dump(itos, open(f"{D}/vocab.json", "w"))
    json.dump(stats, open(f"{D}/stats.json", "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
