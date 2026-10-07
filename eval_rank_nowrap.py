"""Run mapformer.eval_nd unchanged (via runpy) with the RANK_NOWRAP arms registered in VARIANT_MAP.

    python3 -m mapformer.eval_rank_nowrap --runs-dir ... (eval_nd's own arguments)

eval_nd imports VARIANT_MAP from train_variant at module level, i.e. the dict mutated by train_rank_nowrap."""
import runpy
import sys

from mapformer import train_rank_nowrap  # noqa: F401  (registers Vanilla_r2ph_om32 / Vanilla_r3ph_om32)

if __name__ == "__main__":
    sys.argv = ["mapformer.eval_nd"] + sys.argv[1:]
    runpy.run_module("mapformer.eval_nd", run_name="__main__", alter_sys=True)
