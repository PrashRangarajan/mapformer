"""Trainer entry for EXPTORUS_PREREG.md: train_variant, unchanged, with ExpAO4_r4 / SoftplusAO4_r4 registered first.

  python3 -m mapformer.train_exptorus --variant ExpAO4_r4 <train_variant's arguments>

Every argument, the recipe, the seeding (torch / numpy seed, then the GridWorld, then the model), the data stream and the
checkpoint format are train_variant.main()'s; this module only calls model_exptorus.register() before it. The name starts
with `train_` on purpose: lib_driver.sh counts GPU slots by `mapformer.train_` in a python3 process's argv.
"""
from mapformer.model_exptorus import register

register()

from mapformer.train_variant import main  # noqa: E402  (after register, so --variant's choices include the new names)

if __name__ == "__main__":
    main()
