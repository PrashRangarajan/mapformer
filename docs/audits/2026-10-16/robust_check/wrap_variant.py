"""Run train_variant.main() recording md5 of every training batch (tokens) and the AdamW groups. python3 wrap_variant.py OUT -- args"""
import sys, json, hashlib


def run():
    out = sys.argv[1]; sys.argv = ["train_variant"] + sys.argv[3:]
    import torch
    import mapformer.data_parallel as DP
    rec = {"batches": [], "groups": []}
    orig = DP.ParallelBatchGenerator.next_batch
    def nb(self):
        b = orig(self); rec["batches"].append(hashlib.md5(b[0].numpy().tobytes() + b[2].numpy().tobytes()).hexdigest()); return b
    DP.ParallelBatchGenerator.next_batch = nb
    _O = torch.optim.AdamW
    class R(_O):
        def __init__(self, p, *a, **kw):
            super().__init__(p, *a, **kw); rec["groups"] = [(g["weight_decay"], g["lr"], len(g["params"])) for g in self.param_groups]
    torch.optim.AdamW = R
    import mapformer.train_variant as TV
    TV.main()
    json.dump(rec, open(out, "w"))


if __name__ == "__main__":
    run()
