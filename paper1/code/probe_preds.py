#!/usr/bin/env python3
"""Is a model's held-out score patient-specific, or one shared ranking?

Every cell saves per-patient scores over all reactions plus the held-out mask.
If collapsing those scores to their cohort mean, a deliberately patient-invariant
predictor, reproduces or beats the per-patient AUROC, then whatever the model
learned is a property of the reaction and not of the patient.

Writes out/collapse.json so the manuscript's collapse table is generated rather
than transcribed, and records the cell basis of every model-level average.
"""
import numpy as np, glob, json, os, statistics as st
from sklearn.metrics import roc_auc_score
import torch

D = os.path.expanduser("~/metagnn/out/dh")
OUT = os.path.expanduser("~/metagnn/out/collapse.json")
y = np.asarray(torch.load(os.path.expanduser("~/metagnn/work/crc_624/activity_pseudolabels.pt"),
                          map_location="cpu", weights_only=False)).astype(int)

rng = np.random.default_rng(0)
cells = {}
print("%-24s %9s %9s %9s %9s %9s" % ("cell", "perpat", "cohmean", "diff", "shuffled", "interpat_r"))
print("-" * 78)
for f in sorted(glob.glob(D + "/*_preds.npz")):
    tag = os.path.basename(f).replace("_preds.npz", "")
    z = np.load(f); P = z["scores"]; m = z["heldout_mask"]
    if m.sum() < 50:
        continue
    per = float(np.mean([roc_auc_score(y[m], P[i, m]) for i in range(len(P))]))
    coh = float(roc_auc_score(y[m], P[:, m].mean(0)))
    perm = rng.permutation(len(P))
    shuf = float(np.mean([roc_auc_score(y[m], P[perm[i], m]) for i in range(len(P))]))
    C = np.corrcoef(P[:, m]); r = float(np.median(C[np.triu_indices(len(C), 1)]))
    cells[tag] = dict(perpat=per, cohmean=coh, diff=coh - per, shuffled=shuf, interpat_r=r,
                      n_patients=int(len(P)), n_heldout=int(m.sum()))
    print("%-24s %9.4f %9.4f %+9.4f %9.4f %9.4f" % (tag, per, coh, coh - per, shuf, r))

# model-level means, over exactly the cells that exist, with the basis recorded
def parse(tag):
    # e.g. gnn_B_mb0.0_p0_r0  /  mlp_emb_mb0.0_p3_r2
    parts = tag.split("_mb")
    model = parts[0]
    rest = parts[1].split("_")
    return model, float(rest[0]), int(rest[1][1:]), int(rest[2][1:])

roll = {}
for tag, v in cells.items():
    model, lam, pf, rf = parse(tag)
    if lam != 0.0:
        continue
    roll.setdefault(model, []).append((tag, v))
summary = {}
for model, items in sorted(roll.items()):
    items.sort()
    summary[model] = dict(
        n_cells=len(items),
        cells=[t for t, _ in items],
        perpat=st.mean(v["perpat"] for _, v in items),
        cohmean=st.mean(v["cohmean"] for _, v in items),
        shuffled=st.mean(v["shuffled"] for _, v in items),
        interpat_r=st.mean(v["interpat_r"] for _, v in items),
        # the paired collapse effect: mean over cells of (cohort mean - per patient)
        paired_diff=st.mean(v["cohmean"] - v["perpat"] for _, v in items),
        paired_diff_sd=(st.stdev([v["cohmean"] - v["perpat"] for _, v in items])
                        if len(items) > 1 else None),
    )

print()
print("%-12s %5s %9s %9s %10s %9s" % ("model", "n", "perpat", "cohmean", "paired_d", "shuffled"))
print("-" * 60)
for m, s in summary.items():
    print("%-12s %5d %9.4f %9.4f %+10.4f %9.4f"
          % (m, s["n_cells"], s["perpat"], s["cohmean"], s["paired_diff"], s["shuffled"]))

json.dump(dict(cells=cells, summary=summary), open(OUT, "w"), indent=1)
print("\nwrote", OUT)
