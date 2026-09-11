#!/usr/bin/env python3
"""Exact parameter counts for every architecture in the double hold-out, on CPU.

The manuscript quotes a parameter count in prose. This emits it from the same
construction path the experiments use, so the number in the paper comes from the
model that was actually trained rather than from an estimate.
"""
import os, sys, json, numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trainer as T

DATA = os.path.expanduser("~/metagnn/work/crc_624")
FEAT = os.path.join(DATA, "reaction_features")
N_RXN = 10600
ARCH = {"A": dict(hidden=128, layers=2, heads=4), "B": dict(hidden=256, layers=3, heads=8)}

import pandas as pd
ids = pd.read_csv(os.path.join(DATA, "clinical_metadata.tsv"), sep="\t")["tcga_barcode"].tolist()
torch.manual_seed(2024)
d = T.UnifiedDataset(DATA, ids[:2], "cohort_220", bipartite_only=False, feature_override_dir=FEAT)
s = d[0]
rin, min_ = s["reaction"].x.shape[1], s["metabolite"].x.shape[1]
ets = list(s.edge_types)
print("rxn_in_dim=%d met_in_dim=%d n_metabolites=%d" % (rin, min_, s["metabolite"].x.shape[0]))
print("edge types:", ets)

class ReactionMLP(nn.Module):
    def __init__(self, in_dim, hidden=256, dropout=0.2, emb=0):
        super().__init__(); self.emb = nn.Embedding(N_RXN, emb) if emb else None
        dd = in_dim + emb
        self.net = nn.Sequential(nn.Linear(dd, hidden), nn.ELU(), T.MCDropout(dropout),
                                 nn.Linear(hidden, hidden // 2), nn.ELU(), T.MCDropout(dropout),
                                 nn.Linear(hidden // 2, 1), nn.Sigmoid())

def n_par(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

out = {}
for k, a in ARCH.items():
    m = T.MetaGNN(rxn_in_dim=rin, met_in_dim=min_, hidden_dim=a["hidden"],
                  n_layers=a["layers"], heads=a["heads"], dropout=0.2, edge_types=ets)
    out["gnn_" + k] = n_par(m)
out["mlp"] = n_par(ReactionMLP(rin, emb=0))
out["mlp_emb"] = n_par(ReactionMLP(rin, emb=64))
out["mlp_emb_embedding_only"] = N_RXN * 64
out["mlp_emb_params_per_reaction"] = 64

for k, v in out.items():
    print("%-28s %10d" % (k, v))
json.dump(out, open(os.path.expanduser("~/metagnn/out/param_counts.json"), "w"), indent=1)
print("\nwrote out/param_counts.json")
