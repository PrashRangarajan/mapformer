---
name: Relative paths in `python3 -m mapformer.X` resolve to the PARENT directory
description: Run scripts cd to /home/prashr so the module imports; every relative path inside then resolves there and silently finds nothing. Four debugging rounds.
metadata:
  type: feedback
---

`python3 -m mapformer.X` must run from **`/home/prashr`** (the package parent), so
any relative path inside the module — `runs/foo/bar.pt`, `paper_figures/...` —
resolves to `/home/prashr/`, not the repo. **It fails silently**: globs match
nothing, aggregators emit a table of `—`, and nothing errors.

This has cost four separate debugging rounds.

**The fix that works for modules** is an absolute constant, not a `cd`:

```python
REPO = "/home/prashr/mapformer"
ck = f"{REPO}/runs/{sub}/{tag}_s{s}/{var}.pt"
```

For heredocs and one-off scripts, `cd "$REPO"` first. For anything invoked as
`python3 -m`, the `cd` is not available to you — use the constant.

**Tell:** an aggregator that runs cleanly and reports zero rows, or `n=0` in every
cell. That is a path bug, not an empty batch. Check one path with `ls` before
believing the table.
