"""Functional CoPE (Listing 1 formula, through the repo's CoPE module) with hand-set gates and table, for the
construction-level checks of COPE_DIAGNOSIS.md. Content logit TIED to the gate (logit = log(g/(1-g))), as in CoPE.
Readout: argmax of attention mass per observation id (values = token identity), as c3.py."""
import math
import numpy as np
import torch

from lib_cope import strata
from mapformer.model_mono_baselines import CoPE


def make_table(M, npos=2048, shape="tri2"):
    cp = CoPE(npos, 1)
    with torch.no_grad():
        cp.pos_emb.zero_()
        if shape == "tri2":
            cp.pos_emb[0, 0] = torch.tensor([M if k % 2 == 0 else -M for k in range(npos)], dtype=torch.float32)
    return cp


@torch.no_grad()
def func_eval(G, cp, trajs, rows_out=True):
    """G: (V, V) gate table [query type, key type]. Returns acc, mean -log(share of the obs-key attention mass on the correct obs), rows."""
    lg = torch.log(G / (1 - G))
    ok = tot = 0; loss = 0.0; rows = []
    for tok, rev in trajs:
        x = tok[:-1]; L = len(x)
        logits = lg[x[:, None], x[None, :]].float()
        mask = torch.ones(L, L).tril().log()
        al = (logits + mask)[None, None]
        q = torch.ones(1, 1, L, 1)
        z = cp(q, al)[0, 0]
        att = torch.softmax(al[0, 0] + z, -1)
        tgt = tok[1:]; m = rev[1:]
        st = {r["t"]: r for r in strata(tok)}
        for i in torch.nonzero(m).flatten().tolist():
            w = torch.zeros(19).index_add_(0, x[: i + 1], att[i, : i + 1]); w[0] = w[1] = 0
            c = int(w.argmax() == tgt[i]); ok += c; tot += 1
            loss -= math.log(max(float(w[tgt[i]] / w.sum()), 1e-12))   # mass among obs keys
            if rows_out:
                rows.append(dict(st[i // 2], ok=c))
    return ok / tot, loss / tot, rows


def gate_table(g_plus, g_minus, g_obs):
    G = torch.full((19, 19), float(g_obs))
    G[:, 0] = g_plus; G[:, 1] = g_minus
    return G
