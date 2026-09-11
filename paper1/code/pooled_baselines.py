#!/usr/bin/env python3
"""Pooled, frozen logistic baselines on every cell: one model per cell, scored on every test patient.

The two per-patient linear rows of naive_baselines_allfolds.py ("own expression, fitted"
and "network + own expression") fit a separate logistic regression for every test
patient: the design matrix is built from that patient's own expression column, the fit
uses the training reactions and the patient-invariant labels, and the model is scored on
the same patient's held-out reactions. Each test patient therefore receives a model
adapted to their own column, which is a different object from the graph model, a single
model trained once on the training patients and frozen before any test patient is seen.

This script adds the frozen counterpart. For every (patient fold, reaction fold) cell one
l2-penalized logistic regression is fitted on the stacked rows of (training patient,
training reaction) pairs, the features of a row built from that training patient's own
expression column, and the model is then frozen and scored on each test patient's own
rows on the held-out reactions of the cell. Two designs are fitted:

  expr_pooled_lr          the three expression columns of the per-patient rows, the value,
                          log(1 + value) and a nonzero flag, built from each patient's own
                          column
  structure_plus_pooled   the 69 network-feature columns of naive_baselines_allfolds.py
                          (copied verbatim: six topological and 63 annotation columns)
                          followed by the same three expression columns; the per-patient
                          "network + own expression" row appends two of them, the value
                          and the flag, so the two rows differ by the log column as well
                          as by the fitting unit

The standardization is fitted on the stacked training rows. The regularization strength
is chosen, as for the per-patient rows, from the same C grid by the same three-fold
inner cross-validation over training reactions with the same seed, grouped by reaction so
that the rows of one reaction never fall on both sides of an inner split; the inner score
is the AUROC over the pooled rows of the inner held-out reactions. A fixed C = 1 fit is
kept beside the tuned one under the "fixed" regime, as in the per-patient file. The
held-out figure of a cell is, exactly as for every other row, the AUROC of each test
patient's own held-out rows averaged over the test patients of the cell; the average
precision is recorded beside it under the parallel "auprc" key together with the
positive-class prevalence of the cell's held-out reactions.

The stacked design of every training patient would run to about three million rows per
cell. The fit therefore uses a fixed, seeded subsample of the training patients: at most
MAX_TRAIN_PATIENTS of them (60 by default) drawn without replacement from the cell's
training patients by numpy's default generator seeded with 2024 plus the patient fold
index, sorted by barcode, and identical for every reaction fold and both designs of that
patient fold. The rule and the selected barcodes are recorded in the output.

Environment variables are the ones naive_baselines_allfolds.py reads: P1_DATA (the cohort
directory), P1_ALIGNED (recon3d_aligned.json), P1_OUT (default results/naive_pooled.json
next to this script's directory), P1_EXPR_TRANSFORM (none or rank) and NJOBS (parallel
inner fits; each holds a copy of the stacked design). The --cells option restricts a run
to a list of cells such as p0r0 p0r1 for a smoke test, and --max_train_patients changes
the subsample size (recorded in the output). Cells already in the output file are
skipped, so the grid can be filled in several runs.
"""
import json, os, sys, time, argparse, numpy as np, pandas as pd, torch, h5py
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--cells", nargs="*", default=None, help="restrict to these cells, e.g. p0r0 p0r1 (default: all)")
ap.add_argument("--max_train_patients", type=int, default=60, help="size of the seeded training-patient subsample")
A = ap.parse_args()

D = os.path.expanduser(os.environ.get("P1_DATA", "~/metagnn/work/crc_624"))
OUT = os.path.expanduser(os.environ.get("P1_OUT", os.path.join(os.path.dirname(HERE), "results", "naive_pooled.json")))
ALIGNED = os.path.expanduser(os.environ.get("P1_ALIGNED", "~/metagnn/out/recon3d_aligned.json"))
EXPR_TRANSFORM = os.environ.get("P1_EXPR_TRANSFORM", "none")
N = 10600; SEED = 2024; NJOBS = int(os.environ.get("NJOBS", "16"))
CS = [0.01, 0.1, 1.0, 10.0]
MAX_TRAIN_PATIENTS = A.max_train_patients
y = np.asarray(torch.load(D + "/activity_pseudolabels.pt", map_location="cpu",
                          weights_only=False)).astype(int)

# ---- folds, identical to double_holdout.py and naive_baselines_allfolds.py ---------------
meta = pd.read_csv(D + "/clinical_metadata.tsv", sep="\t")
ids = meta["tcga_barcode"].tolist(); proj = meta["project"].fillna("U").tolist()
pk = StratifiedKFold(5, shuffle=True, random_state=SEED); pfolds = []
for tr, te in pk.split(ids, proj):
    tr_ids = [ids[i] for i in tr]
    trn, val = train_test_split(tr_ids, test_size=0.15, random_state=SEED)
    pfolds.append(dict(train=trn, val=val, test=[ids[i] for i in te]))

def _transform(M):
    if EXPR_TRANSFORM == "none":
        return M
    if EXPR_TRANSFORM == "rank":
        from scipy.stats import rankdata
        M = M.copy()
        for i in range(M.shape[0]):
            nz = M[i] > 0
            if nz.any(): M[i, nz] = rankdata(M[i, nz]) / nz.sum()
        return M
    raise ValueError(EXPR_TRANSFORM)

def load(ps):
    return _transform(np.stack([h5py.File(f"{D}/reaction_features/{p}.h5")["X_R"][:, 0] for p in ps]))

has = load(pfolds[0]["train"][:20]).max(0) > 0
rk = StratifiedKFold(3, shuffle=True, random_state=SEED)
rfolds = list(rk.split(np.zeros(N), 2 * y + has.astype(int)))
# P1_FAMILY_MAP: reaction folds drawn family by family (family_folds.py), and the inner partition
# that tunes C grouped by family, as in naive_baselines_allfolds.py
FAMILY_MAP = os.environ.get("P1_FAMILY_MAP")
FAMILY = None
if FAMILY_MAP:
    import sys as _sys; _sys.path.insert(0, HERE)
    import family_folds as FF
    FAMILY, FAMILY_SCHEME = FF.load_family_map(os.path.expanduser(FAMILY_MAP))
    rfolds = FF.family_rfolds(FAMILY, y, has, n_folds=3, seed=SEED)

# ---- structural and annotation columns, verbatim from naive_baselines_allfolds.py -----------
AL = json.load(open(ALIGNED))["reactions"]; assert len(AL) == N
deg = {}
for e in ["substrate_of", "produces", "shared_metabolite"]:
    t = torch.load(f"{D}/edge_indices/{e}.pt", map_location="cpu", weights_only=False).numpy()
    d = np.zeros(N); src = t[0] if e != "produces" else t[1]
    np.add.at(d, src[src < N], 1); deg[e] = d
sub = pd.Series([(r.get("subsystem") or "(unknown)") for r in AL])
top = [s for s in sub.value_counts().index if s != "(unknown)"][:60]
n_met = np.array([len(r["metabolites"]) for r in AL], float)
TOPO = np.column_stack(
    [deg["substrate_of"], deg["produces"], deg["shared_metabolite"], np.log1p(deg["shared_metabolite"]),
     n_met, (n_met == 1).astype(float)])
ANNOT = np.column_stack(
    [np.array([1.0 if r["gene_sets"] else 0.0 for r in AL]),
     np.array([float(r["n_genes"]) for r in AL]),
     np.array([1.0 if (r.get("lower_bound") or 0) < 0 else 0.0 for r in AL])]
    + [(sub == s).values.astype(float) for s in top])
STRUCT = np.column_stack([TOPO, ANNOT])
print(f"topology {TOPO.shape[1]} columns, annotation {ANNOT.shape[1]} columns, "
      f"structure {STRUCT.shape[1]} columns; expression-bearing reactions {int(has.sum())}", flush=True)

def expr_cols(v):
    return np.column_stack([v, np.log1p(v), (v > 0).astype(float)])

DESIGNS = {
    "expr_pooled_lr": expr_cols,
    "structure_plus_pooled": lambda v: np.column_stack([STRUCT, expr_cols(v)]),
}

# ---- the pooled fit ---------------------------------------------------------------------
def stacked(Xp, build, rxn):
    """Rows (patient, reaction) for every patient in Xp and every reaction in rxn, patient-major."""
    X = np.vstack([build(v)[rxn] for v in Xp])
    return X, np.tile(y[rxn], len(Xp))

def inner_splits(n_patients, rtr):
    """The per-patient rows' inner partition of the training reactions (three stratified folds,
    seed 2024), lifted to the stacked rows so that every row of a reaction is on one side."""
    n_r = len(rtr); out = []
    if FAMILY is not None:
        from sklearn.model_selection import GroupKFold
        splits = GroupKFold(3).split(np.zeros(n_r), y[rtr], groups=FAMILY[rtr])
    else:
        splits = StratifiedKFold(3, shuffle=True, random_state=SEED).split(np.zeros(n_r), y[rtr])
    for itr, ite in splits:
        out.append((np.concatenate([itr + p * n_r for p in range(n_patients)]),
                    np.concatenate([ite + p * n_r for p in range(n_patients)])))
    return out

def fit_pooled(Xtr_sub, Xte, build, rtr, rte, tuned):
    """Fit one frozen model on the stacked training rows and score every test patient on their own
    held-out rows. Returns the mean per-patient AUROC, the mean per-patient average precision and
    the regularization strength used."""
    X, t = stacked(Xtr_sub, build, rtr)
    base = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000))
    if tuned:
        m = GridSearchCV(base, {"logisticregression__C": CS}, cv=inner_splits(len(Xtr_sub), rtr),
                         scoring="roc_auc", n_jobs=NJOBS, refit=True)
        m.fit(X, t); C = float(m.best_params_["logisticregression__C"]); model = m.best_estimator_
    else:
        model = base; model.fit(X, t); C = 1.0
    aucs, aps = [], []
    for v in Xte:
        s = model.predict_proba(build(v)[rte])[:, 1]
        aucs.append(roc_auc_score(y[rte], s)); aps.append(average_precision_score(y[rte], s))
    return float(np.mean(aucs)), float(np.mean(aps)), C, int(len(t))

R = {"fixed": {}, "tuned": {}, "auprc": {"fixed": {}, "tuned": {}}}
if os.path.exists(OUT):
    R = json.load(open(OUT))
    for k in ("fixed", "tuned"):
        R.setdefault(k, {})
    R.setdefault("auprc", {}); R["auprc"].setdefault("fixed", {}); R["auprc"].setdefault("tuned", {})

def put(regime, key, cell, auc, apr):
    R[regime].setdefault(key, {})[cell] = auc
    R["auprc"][regime].setdefault(key, {})[cell] = apr

R["_subsample"] = dict(rule="at most MAX_TRAIN_PATIENTS training patients of the patient fold, drawn without "
                            "replacement by numpy.random.default_rng(2024 + patient fold) and sorted by barcode; "
                            "the same patients for every reaction fold and both designs",
                       max_train_patients=MAX_TRAIN_PATIENTS, patients=R.get("_subsample", {}).get("patients", {}))
R["_inner_cv"] = ("three stratified folds over training reactions, seed 2024, the partition of the per-patient "
                  "rows, lifted to the stacked rows so that every row of a reaction is on one side; "
                  "scored by the AUROC over the pooled rows of the inner held-out reactions")
for p, sp in enumerate(pfolds):
    cells_here = [f"p{p}r{rf}" for rf in range(len(rfolds))]
    if A.cells is not None and not any(c in A.cells for c in cells_here):
        continue
    rng = np.random.default_rng(SEED + p)
    n_sub = min(MAX_TRAIN_PATIENTS, len(sp["train"]))
    sub_ids = sorted(rng.choice(sp["train"], size=n_sub, replace=False).tolist())
    R["_subsample"]["patients"][f"p{p}"] = sub_ids
    Xtr_sub = load(sub_ids); Xte = load(sp["test"])
    for rf, (rtr, rte) in enumerate(rfolds):
        cell = f"p{p}r{rf}"
        if A.cells is not None and cell not in A.cells:
            continue
        if all(cell in R[reg].get(k, {}) and cell in R["auprc"][reg].get(k, {})
               for reg in ("fixed", "tuned") for k in DESIGNS):
            print(f"{cell} present, skipped", flush=True); continue
        t0 = time.time()
        R.setdefault("_heldout_prevalence", {})[cell] = float(y[rte].mean())
        R.setdefault("_n_heldout", {})[cell] = int(len(rte))
        for regime, tuned in (("fixed", False), ("tuned", True)):
            for key, build in DESIGNS.items():
                auc, apr, C, n_rows = fit_pooled(Xtr_sub, Xte, build, rtr, rte, tuned)
                put(regime, key, cell, auc, apr)
                R.setdefault("_C_selected", {}).setdefault(regime, {}).setdefault(key, {})[cell] = C
                R.setdefault("_n_train_rows", {})[cell] = n_rows
        R["_n_test_patients"] = R.get("_n_test_patients", {}); R["_n_test_patients"][f"p{p}"] = len(sp["test"])
        R["_n_train_patients_used"] = R.get("_n_train_patients_used", {}); R["_n_train_patients_used"][f"p{p}"] = n_sub
        R["_columns"] = dict(topology=TOPO.shape[1], annotation=ANNOT.shape[1], structure=STRUCT.shape[1],
                             expression=3, structure_plus_expression=STRUCT.shape[1] + 3)
        R["_C_grid"] = CS; R["_family_map"] = (os.path.basename(FAMILY_MAP) if FAMILY_MAP else None)
        R["_data"] = D; R["_expr_transform"] = EXPR_TRANSFORM; R["_n_active_labels"] = int(y.sum())
        os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
        json.dump(R, open(OUT, "w"), indent=1)
        f, t, fa, ta = R["fixed"], R["tuned"], R["auprc"]["fixed"], R["auprc"]["tuned"]
        print(f"{cell} done in {time.time()-t0:.0f} s ({n_sub} training patients, {R['_n_train_rows'][cell]} rows) | "
              f"fixed: expr {f['expr_pooled_lr'][cell]:.4f} s+expr {f['structure_plus_pooled'][cell]:.4f} | "
              f"tuned: expr {t['expr_pooled_lr'][cell]:.4f} (C={R['_C_selected']['tuned']['expr_pooled_lr'][cell]:g}) "
              f"s+expr {t['structure_plus_pooled'][cell]:.4f} (C={R['_C_selected']['tuned']['structure_plus_pooled'][cell]:g}) | "
              f"AUPRC tuned: expr {ta['expr_pooled_lr'][cell]:.4f} s+expr {ta['structure_plus_pooled'][cell]:.4f} "
              f"(prevalence {R['_heldout_prevalence'][cell]:.4f})", flush=True)
print("wrote", OUT)
