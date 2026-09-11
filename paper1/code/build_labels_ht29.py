#!/usr/bin/env python3
"""Rebuild the activity labels from the Human Metabolic Atlas cell-line reconstructions, and write
the tissue-matched (HT29-only) variant used by the robustness check.

The deposited label vector is the union of eleven NCI-60 cell-line reconstructions from the Human1
publication data (11models.mat; Robinson et al. 2020), matched to Recon3D by reaction identifier in
the canonical order of the training arrays. Only one of the eleven, HT29, is colorectal. This script

  1. rebuilds that union with the same rule as the data-generation pipeline (a reaction is active in
     a model when either bound is nonzero; a model contributes the active reactions whose identifier
     is a Recon3D identifier) and checks that it reproduces the deposited vector exactly, so that the
     rule below is known to be the rule the labels were built with;
  2. writes the HT29-only vector, activity_labels_ht29.pt, in the same format as the deposited file,
     together with a JSON summary of both vectors.

Usage:  python3 build_labels_ht29.py --mat 11models.mat --deposited activity_pseudolabels.pt
        (or --deposited rxn_context.npz, whose 'labels' array is the same vector)
The reference order comes from data/recon3d_aligned.json.gz.
"""
import argparse, gzip, json, os
import numpy as np, scipy.io as sio, torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--mat", required=True, help="11models.mat from the Human1 publication data")
ap.add_argument("--deposited", required=True, help="activity_pseudolabels.pt or rxn_context.npz")
ap.add_argument("--out", default=os.path.join(HERE, "data", "activity_labels_ht29.pt"))
ap.add_argument("--summary", default=os.path.join(HERE, "data", "labels_ht29.json"))
A = ap.parse_args()

ids = [r["id"] for r in json.load(gzip.open(os.path.join(HERE, "data", "recon3d_aligned.json.gz")))["reactions"]]
idx = {r: i for i, r in enumerate(ids)}
mat = sio.loadmat(A.mat, squeeze_me=True, struct_as_record=False)
keys = [k for k in mat if not k.startswith("__")]

def active_vector(m):
    rx = [str(x) for x in m.rxns]
    lb = np.asarray(m.lb, float).ravel(); ub = np.asarray(m.ub, float).ravel()
    act = (np.abs(lb) > 1e-9) | (np.abs(ub) > 1e-9)
    v = np.zeros(len(ids), dtype=np.int64); matched = 0
    for r, a in zip(rx, act):
        if r in idx:
            matched += 1
            if a: v[idx[r]] = 1
    return v, matched, len(rx)

per, matched = {}, {}
for k in keys:
    v, n_match, n_rxn = active_vector(mat[k])
    per[k] = v; matched[k] = dict(model_id=str(getattr(mat[k], "id", k)), n_reactions=n_rxn, n_matched=n_match, n_active_matched=int(v.sum()))
union = np.maximum.reduce(list(per.values()))

if A.deposited.endswith(".npz"):
    y = np.asarray(np.load(A.deposited, allow_pickle=True)["labels"]).astype(int)
else:
    y = np.asarray(torch.load(A.deposited, map_location="cpu", weights_only=False)).astype(int)
same = bool((union == y).all())
print(f"union rebuilt: {int(union.sum())} active; deposited: {int(y.sum())}; identical: {same}")
if not same:
    raise SystemExit("the rebuilt union does not reproduce the deposited labels; the rule differs")

ht = per["HT29"]
torch.save(torch.tensor(ht, dtype=torch.float32), A.out)
summary = dict(
    rule="active in a model when either bound is nonzero; matched to Recon3D by reaction identifier",
    models=matched, union_active=int(union.sum()), union_reproduces_deposited=same,
    ht29_active=int(ht.sum()), ht29_subset_of_union=bool((ht <= union).all()),
    union_active_not_ht29=int(((union == 1) & (ht == 0)).sum()),
    ht29_prevalence=float(ht.mean()), union_prevalence=float(union.mean()),
    out=os.path.basename(A.out))
json.dump(summary, open(A.summary, "w"), indent=1)
print(f"HT29-only: {int(ht.sum())} active, written to {A.out}")
