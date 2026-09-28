import sys
def apply(path, pairs):
    s = open(path).read()
    for i, (old, new) in enumerate(pairs):
        n = s.count(old)
        if n != 1:
            sys.exit(f"{path}: edit {i} matched {n} times:\n{old[:120]}")
        s = s.replace(old, new)
    open(path, 'w').write(s)
    print(f"{path}: {len(pairs)} edits applied")
