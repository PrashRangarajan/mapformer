"""Eval-time attention-dropout scale correction (docs/theory/2026-10-04/00_PLAN.md step 1; mechanism verified in
docs/audits/2026-10-04/dropout_scale_check_out.txt).

Models trained with inverted dropout on the attention probabilities (p = 0.1) can come to expect the 1/(1-p) scale of
the typical training-time attention output. `install(scale)` patches nn.Module.eval so that every attention layer
(any module with `o_proj` and `dropout` whose class is a known layer whose o_proj input is softmax(scores) @ V) gets
a forward pre-hook multiplying o_proj's input by `scale` (default 1/(1-p) of that layer). Nothing else changes;
scale=None leaves the model untouched (control). The registered eval scripts are run unmodified via `run`:

    python3 -m mapformer.rescore_hook --scale auto -- mapformer.eval_rank_strata --runs-dir ... --out NEW.json
"""
import atexit
import runpy
import sys

import torch.nn as nn

KNOWN = {"mapformer.model.WMTransformerLayer", "mapformer.model.EMTransformerLayer",
         "mapformer.model_pope.WMTransformerLayer_PoPE", "mapformer.model_codes.EMPosOnlyLayer"}
STATS = {"hooked": 0, "fired": 0, "skipped": set()}


def install(scale):
    orig_eval = nn.Module.eval

    def eval_(self):
        out = orig_eval(self)
        for mod in self.modules():
            if getattr(mod, "_rescore_hooked", False) or not (hasattr(mod, "o_proj") and hasattr(mod, "dropout")):
                continue
            name = f"{type(mod).__module__}.{type(mod).__name__}"
            if name not in KNOWN:
                STATS["skipped"].add(name); continue
            s = (1.0 / (1.0 - mod.dropout.p)) if scale == "auto" else float(scale)

            def pre(m, args, s=s):
                STATS["fired"] += 1
                return (args[0] * s,) + tuple(args[1:])
            mod.o_proj.register_forward_pre_hook(pre); mod._rescore_hooked = True; STATS["hooked"] += 1
        return out
    nn.Module.eval = eval_


def _report():
    print(f"[rescore_hook] layers hooked {STATS['hooked']}, hook calls {STATS['fired']}, unknown layer classes skipped "
          f"{sorted(STATS['skipped'])}", file=sys.stderr)


if __name__ == "__main__":
    i = sys.argv.index("--")
    opts = sys.argv[1:i]; scale = opts[opts.index("--scale") + 1]
    if scale != "none":
        install(scale); atexit.register(_report)
    mod = sys.argv[i + 1]; sys.argv = [mod] + sys.argv[i + 2:]
    runpy.run_module(mod, run_name="__main__", alter_sys=True)
