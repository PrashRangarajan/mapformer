"""Run mapformer.train_gauge.main() with AdamW instrumented: records per-group weight_decay, #params per group,
and the lr actually used at every optimizer step. Usage: python3 wrap_gauge.py OUTJSON -- <train_gauge args>"""
import sys, json


def run():
    out_json = sys.argv[1]; sys.argv = ["train_gauge"] + sys.argv[3:]
    import torch, mapformer
    rec = {"pkg": mapformer.__file__, "groups": [], "lrs": []}
    _Orig = torch.optim.AdamW

    class AdamWRec(_Orig):
        def __init__(self, params, *a, **kw):
            super().__init__(params, *a, **kw)
            for g in self.param_groups:
                rec["groups"].append({"weight_decay": g["weight_decay"], "lr": g["lr"], "n_tensors": len(g["params"]),
                                      "n_params": sum(p.numel() for p in g["params"])})

        def step(self, *a, **kw):
            rec["lrs"].append(self.param_groups[0]["lr"])
            return super().step(*a, **kw)
    torch.optim.AdamW = AdamWRec
    import mapformer.train_gauge as G
    G.main()
    json.dump(rec, open(out_json, "w"))
    print("pkg", rec["pkg"], "groups", rec["groups"], "nsteps", len(rec["lrs"]))


if __name__ == "__main__":
    run()
