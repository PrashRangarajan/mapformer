import os, shutil, tempfile
import mapformer.analyze_ssm_sign as A
tmp = tempfile.mkdtemp(dir=os.environ["S"])
base = {("A", "abs", 32): 1, ("A", "abs", 8): 1, ("A", "softplus", 32): 1, ("A", "exp", 32): 1, ("A", "softplus", 8): 1,
        ("A", "exp", 8): 1, ("A", "index", 32): 2, ("A", "index", 8): 0, ("B", "abs", 32): 1, ("B", "softplus", 32): 1,
        ("B", "exp", 32): 1}
lo = {("A", "index", 32): 0.70, ("A", "index", 8): 0.70}
A._write(tmp, base, lo, {"abs": {"k": 8}, "softplus": {"k": 8}, "exp": {"k": 8}})
R, miss = A.load(tmp)
v = A.analyse(R, out=lambda *a, **k: None)
print("reading:", v["reading"])
print("gate m=32:", v["gate"][32])
print("A abs|32 label stored:", v["A"]["abs|32"]["label"], "| softplus|32:", v["A"]["softplus|32"]["label"])
print("joint (3e-3):", v["joint"])
print("joint (1e-3):", v["joint_lr1e-3"])
# rule-2 set: which blocks enter r(loss, acc)?
fin = [k for k, r in R.items() if not r["div"] and k[0] != "H"]
print("blocks in rule-2 r:", sorted({k[0] for k in fin}))
shutil.rmtree(tmp)
