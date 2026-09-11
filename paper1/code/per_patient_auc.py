#!/usr/bin/env python3
"""Per-patient held-out AUROC for every substitution arm, so the paper can do inference.

A cell's headline number is a mean over test patients. Reporting only that mean throws
away the sampling unit the design actually has, and leaves no way to separate three
sources of variation that a reviewer will want separated:

  between patients within a trained model   (does the substitution change this model's
                                             scores, patient by patient)
  between reaction folds                    (does it hold on different held-out reactions)
  between patient folds                     (does it hold on different training sets)

This writes one number per (cell, arm, patient), which is small, and lets all three be
estimated downstream without moving the prediction arrays off the machine.

Beside the AUROC, the average precision of the same held-out scores is written for every
(cell, arm, patient) under "auprc", with the positive-class prevalence of the cell's
held-out reactions under "prevalence", which is the average precision of a random
ranking. The existing keys are unchanged.

The prediction files are read from ~/metagnn/out/{dh,ctrl,rewire} and the labels from
~/metagnn/work/crc_624, as double_holdout.py writes them; P1_OUTD and P1_DATA override
the two roots and P1_OUT the output file.
"""
import os, re, json, glob
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import torch

OUTD = os.path.expanduser(os.environ.get("P1_OUTD", "~/metagnn/out"))
DH = os.path.join(OUTD, "dh")
CT = os.path.join(OUTD, "ctrl")
RW = os.path.join(OUTD, "rewire")
OUT = os.path.expanduser(os.environ.get("P1_OUT", os.path.join(OUTD, "per_patient.json")))
DATA = os.path.expanduser(os.environ.get("P1_DATA", "~/metagnn/work/crc_624"))

y = np.asarray(torch.load(os.path.join(DATA, "activity_pseudolabels.pt"),
                          map_location="cpu", weights_only=False)).astype(int)

PAT = re.compile(r"^gnn_B_mb0\.0_p(?P<pf>\d+)_r(?P<rf>\d+)"
                 r"(?:_(?P<mode>mean|zero|indicator|rewire\d+))?_preds\.npz$")

rows = {}
for d in (DH, CT, RW):
    for f in sorted(glob.glob(os.path.join(d, "*_preds.npz"))):
        m = PAT.match(os.path.basename(f))
        if not m:
            continue
        mode = m.group("mode") or "real"
        key = (int(m.group("pf")), int(m.group("rf")))
        z = np.load(f, allow_pickle=True)
        P, mk = z["scores"], z["heldout_mask"]
        pids = [str(x) for x in z["patient_ids"]]
        auc = [float(roc_auc_score(y[mk], P[i, mk])) for i in range(len(P))]
        apr = [float(average_precision_score(y[mk], P[i, mk])) for i in range(len(P))]
        rows.setdefault(f"p{key[0]}r{key[1]}", {})[mode] = dict(
            patient_ids=pids, auroc=auc, n_heldout=int(mk.sum()),
            auprc=apr, prevalence=float(y[mk].mean()))
        print("p%dr%d %-10s n_pat=%3d  mean=%.4f  sd=%.4f  auprc=%.4f (prevalence %.4f)"
              % (key[0], key[1], mode, len(auc), np.mean(auc), np.std(auc, ddof=1), np.mean(apr), y[mk].mean()), flush=True)

json.dump(rows, open(OUT, "w"))
cells = sorted(rows)
print("\ncells with all three feature modes:",
      [c for c in cells if {"real", "mean", "zero"} <= set(rows[c])])
print("cells with a rewiring:",
      [c for c in cells if any(k.startswith("rewire") for k in rows[c])])
print("cells with the presence-only arm:", [c for c in cells if "indicator" in rows[c]])
print("wrote", OUT, "(%.1f KB)" % (os.path.getsize(OUT) / 1024))
