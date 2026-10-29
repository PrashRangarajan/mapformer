"""Read a checkpoint's non-tensor content, from the .pt or, once it is deleted, its sidecar.

The 2026-10-29 disk cleanup (`docs/cleanup/D2_MANIFEST.md`) deleted checkpoints whose
weights no analysis needs but whose training losses and config some aggregators read
(`agg_alias.py`, `agg_visits.py`, `agg_miniworld.py`, `experiment_audit.py`). Before
deletion their non-tensor content went to `runs/<top>/CKPT_META.jsonl`
(format: `docs/cleanup/extract_ckpt_meta.py`). `load_ckpt_meta(pt)` returns:

  - the `torch.load` result of `pt` when the file exists (unchanged behaviour);
  - otherwise the sidecar's `meta` dict for that checkpoint (losses, config, results,
    variant, ...; dict keys are strings there, e.g. results["512"]);
  - otherwise None.

Losses read from a sidecar are bit-identical to the checkpoint's (floats are stored with
repr); this was verified on every entry before deletion.
"""
import json
import os

_CACHE = {}


def _sidecar_entry(pt):
    real = os.path.realpath(pt)
    d = os.path.dirname(real)
    while True:
        side = os.path.join(d, "CKPT_META.jsonl")
        if os.path.exists(side):
            if side not in _CACHE:
                with open(side) as f:
                    rows = [json.loads(l) for l in f if l.strip()]
                _CACHE[side] = {r["path"]: r for r in rows[1:]}
            rel = os.path.relpath(real, d)
            e = _CACHE[side].get(rel)
            if e is not None:
                return e
        parent = os.path.dirname(d)
        if parent == d or os.path.basename(d) == "runs":
            return None
        d = parent


def has_ckpt(pt):
    """True if the checkpoint exists, or its non-tensor content survives in a sidecar."""
    return os.path.exists(pt) or _sidecar_entry(pt) is not None


def load_ckpt_meta(pt, **torch_load_kwargs):
    if os.path.exists(pt):
        import torch
        torch_load_kwargs.setdefault("map_location", "cpu")
        return torch.load(pt, **torch_load_kwargs)
    e = _sidecar_entry(pt)
    return None if e is None else e["meta"]
