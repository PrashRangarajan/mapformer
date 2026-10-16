import sys, torch, json
a = torch.load(sys.argv[1], map_location="cpu", weights_only=False); b = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
sa, sb = a["model_state_dict"], b["model_state_dict"]
assert sa.keys() == sb.keys()
neq = [k for k in sa if not torch.equal(sa[k], sb[k])]
print("losses equal:", a["losses"] == b["losses"], "| tensors differing:", len(neq), neq[:5])
if len(sys.argv) > 3:
    ea = json.load(open(sys.argv[3])); eb = json.load(open(sys.argv[4])); print("eval.json equal:", ea == eb)
