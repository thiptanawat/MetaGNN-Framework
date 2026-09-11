#!/usr/bin/env python3
"""How much of the cohort-mean collapse gain is noise averaging?

Collapsing every patient's saved score vector to the cohort mean removes two things at
once: whatever is patient-specific in the scores, and the Monte Carlo dropout noise
that remains in each saved score after averaging T forward passes. Averaging over a
hundred-odd patients removes the second even when there is no first, and against a
patient-invariant label that alone raises AUROC. The collapse gain is therefore an upper
bound on the cost of patient identity unless the noise part is measured.

Each cell saves, for every patient and reaction, the mean over T = 30 passes (scores)
and their standard deviation (uncertainties). The standard error of a saved score is
uncertainties / sqrt(T). This script builds a noise-only cohort: the cohort-mean score
vector plus, for each patient, Gaussian noise of that patient's own standard errors.
Those pseudo-patients differ only by noise. The gain from collapsing them is the
noise-averaging component; whatever the observed gain exceeds it by is what patient
identity costs. The Gaussian form is an approximation to the dropout noise and is stated
as one; the draws are independent across patients, as the dropout passes are, and independent
across reactions, which is the approximation.

Writes out/collapse_noise.json.
"""
import numpy as np, glob, json, os, statistics as st
from sklearn.metrics import roc_auc_score
import torch

D = os.path.expanduser("~/metagnn/out/dh")
OUT = os.path.expanduser("~/metagnn/out/collapse_noise.json")
T = 30; REPS = 5
y = np.asarray(torch.load(os.path.expanduser("~/metagnn/work/crc_624/activity_pseudolabels.pt"),
                          map_location="cpu", weights_only=False)).astype(int)
rng = np.random.default_rng(0)
cells = {}
print("%-24s %8s %8s %9s %9s %9s" % ("cell", "perpat", "cohmean", "observed", "noiseonly", "identity"))
print("-" * 74)
for f in sorted(glob.glob(D + "/*_preds.npz")):
    tag = os.path.basename(f).replace("_preds.npz", "")
    if "_mb0.0_" not in tag:
        continue
    z = np.load(f); P = z["scores"][:, z["heldout_mask"]]; U = z["uncertainties"][:, z["heldout_mask"]]
    ym = y[z["heldout_mask"]]
    if ym.sum() < 10:
        continue
    per = float(np.mean([roc_auc_score(ym, P[i]) for i in range(len(P))]))
    S = P.mean(0); coh = float(roc_auc_score(ym, S))
    se = U / np.sqrt(T)
    noise_gains = []
    for _ in range(REPS):
        Q = S[None, :] + rng.normal(size=P.shape) * se
        per_q = float(np.mean([roc_auc_score(ym, Q[i]) for i in range(len(Q))]))
        noise_gains.append(float(roc_auc_score(ym, Q.mean(0))) - per_q)
    ng = float(np.mean(noise_gains))
    # Monte Carlo error of the noise term itself, from the spread of the replicate draws
    ng_sd = float(np.std(noise_gains, ddof=1)) if REPS > 1 else 0.0
    cells[tag] = dict(perpat=per, cohmean=coh, observed_gain=coh - per, noise_only_gain=ng,
                      noise_gain_replicates=[float(x) for x in noise_gains],
                      noise_gain_sd=ng_sd, noise_gain_mc_se=ng_sd / np.sqrt(REPS),
                      identity_component=(coh - per) - ng, n_patients=int(len(P)),
                      mean_se=float(se.mean()), between_patient_sd=float(P.std(0).mean()))
    print("%-24s %8.4f %8.4f %+9.4f %+9.4f %+9.4f" % (tag, per, coh, coh - per, ng, (coh - per) - ng))

summary = {}
for model in ("mlp", "mlp_emb", "gnn_B"):
    items = sorted((t, v) for t, v in cells.items() if t.split("_mb")[0] == model)
    if not items:
        continue
    obs = [v["observed_gain"] for _, v in items]; noi = [v["noise_only_gain"] for _, v in items]
    ide = [v["identity_component"] for _, v in items]
    mcse = [v["noise_gain_mc_se"] for _, v in items]
    summary[model] = dict(n_cells=len(items), cells=[t for t, _ in items],
                          observed_gain=st.mean(obs), noise_only_gain=st.mean(noi),
                          # Monte Carlo error of the mean noise term over the cells (draws are independent)
                          noise_only_mc_se=float(np.sqrt(sum(x * x for x in mcse)) / len(mcse)),
                          identity_component=st.mean(ide),
                          identity_sd=(st.stdev(ide) if len(ide) > 1 else None),
                          identity_all_positive=all(x > 0 for x in ide),
                          identity_n_positive=sum(1 for x in ide if x > 0))
print()
for m, s in summary.items():
    print("%-8s n=%2d observed %+.4f  noise-only %+.4f  identity %+.4f (sd %s, %d/%d positive)"
          % (m, s["n_cells"], s["observed_gain"], s["noise_only_gain"], s["identity_component"],
             ("%.4f" % s["identity_sd"]) if s["identity_sd"] is not None else "na",
             s["identity_n_positive"], s["n_cells"]))
json.dump(dict(cells=cells, summary=summary, T=T, replicates=REPS), open(OUT, "w"), indent=1)
print("\nwrote", OUT)
