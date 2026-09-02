"""Wire ParallelBatchGenerator into train.py + train_variant.py, then PROVE the
serial path is untouched.

Kept separate from the module itself because it edits files the 2x2 batch is
still spawning processes from. Editing a module mid-batch turns a within-batch
comparison into a between-code one, so this refuses to run while those processes
are alive, and it reverts itself if the serial fingerprint moves.
"""
import argparse, os, subprocess, sys

TRAIN = "mapformer/train.py"
VARIANT = "mapformer/train_variant.py"

EDITS = [
    (TRAIN,
     "    aux_coef: float = 0.0,\n    schedule: str = \"linear\",\n) -> list[float]:",
     "    aux_coef: float = 0.0,\n    schedule: str = \"linear\",\n"
     "    data_workers: int = 0,\n) -> list[float]:"),
    (TRAIN,
     "    losses = []\n    for epoch in range(n_epochs):",
     "    # Optional parallel trajectory generation. Generation is 79-95% of an\n"
     "    # epoch at the standard config and is single-threaded, so this is where\n"
     "    # the wall time is. OFF by default: the parallel path seeds each batch by\n"
     "    # its INDEX and so draws a DIFFERENT sample from the same generator than\n"
     "    # the serial path does. Same distribution, not the same stream -- a\n"
     "    # parallel run therefore will not reproduce a stored serial checkpoint.\n"
     "    wants_positions = hasattr(model, \"_batch_positions\")\n"
     "    gen = None\n"
     "    if data_workers > 0:\n"
     "        from .data_parallel import ParallelBatchGenerator\n"
     "        gen = ParallelBatchGenerator(\n"
     "            env, batch_size, n_steps, n_workers=data_workers,\n"
     "            base_seed=torch.initial_seed() % (2 ** 31),\n"
     "            p_transition_noise=p_transition_noise,\n"
     "            want_locations=wants_positions)\n"
     "\n    losses = []\n    for epoch in range(n_epochs):"),
    (TRAIN,
     "        wants_positions = hasattr(model, \"_batch_positions\")\n\n"
     "        for _ in range(n_batches):",
     "        for _ in range(n_batches):"),
    (TRAIN,
     "            tokens, obs_mask, revisit_mask, all_locations = env.generate_batch(\n"
     "                batch_size, n_steps, p_transition_noise=p_transition_noise,\n"
     "            )",
     "            if gen is not None:\n"
     "                tokens, obs_mask, revisit_mask, all_locations = gen.next_batch()\n"
     "            else:\n"
     "                tokens, obs_mask, revisit_mask, all_locations = env.generate_batch(\n"
     "                    batch_size, n_steps, p_transition_noise=p_transition_noise,\n"
     "                )"),
    (TRAIN, "\n    return losses", "\n    if gen is not None:\n        gen.close()\n"
     "\n    return losses"),
    (VARIANT, "        schedule=args.schedule,",
     "        schedule=args.schedule,\n        data_workers=args.data_workers,"),
]


def apply(revert=False):
    for path, old, new in EDITS:
        a, b = (new, old) if revert else (old, new)
        t = open(path).read()
        if t.count(a) != 1:
            print(f"REFUSING: anchor appears {t.count(a)} times in {path}:\n  "
                  f"{a.splitlines()[0][:70]}")
            return False
        open(path, "w").write(t.replace(a, b, 1))
    return True


def add_cli():
    t = open(VARIANT).read()
    if "--data-workers" in t:
        return True
    anchor = '    parser.add_argument("--schedule"'
    if t.count(anchor) != 1:
        print(f"REFUSING: --schedule argument anchor appears {t.count(anchor)} times")
        return False
    ins = ('    parser.add_argument("--data-workers", type=int, default=0,\n'
           '                        help="Parallel trajectory-generation workers. 0 "\n'
           '                             "(default) uses the serial path and is "\n'
           '                             "byte-identical to every existing checkpoint. "\n'
           '                             ">0 is ~3.4x faster at 6 workers but draws a "\n'
           '                             "DIFFERENT sample from the same generator, so "\n'
           '                             "runs are reproducible among themselves and NOT "\n'
           '                             "against stored serial checkpoints (rule 3).")\n')
    open(VARIANT, "w").write(t.replace(anchor, ins + anchor, 1))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    live = subprocess.run(["pgrep", "-u", os.environ.get("USER", ""), "-f",
                           "train_var" "iant"], capture_output=True, text=True)
    n = len([x for x in live.stdout.split() if x])
    if n and not a.force:
        print(f"REFUSING: {n} train_variant processes are still running. Editing a "
              f"module mid-batch makes later runs a different code path.")
        sys.exit(2)

    if not apply() or not add_cli():
        print("patch not applied; nothing changed"); sys.exit(3)
    print("patch applied; verifying the serial path is byte-identical")

    r = subprocess.run([sys.executable, "-m", "mapformer.dp_reference", "--check",
                        "--out", a.reference, "--device", a.device])
    if r.returncode != 0:
        print("SERIAL PATH CHANGED -- reverting")
        subprocess.run(["git", "checkout", "--", TRAIN, VARIANT])
        sys.exit(4)
    print("serial path verified unchanged; wiring is live")


if __name__ == "__main__":
    main()
