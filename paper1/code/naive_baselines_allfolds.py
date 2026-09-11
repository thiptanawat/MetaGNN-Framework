#!/usr/bin/env python3
"""The linear reference points on every cell of the grid, with the structure floor ablated.

The two earlier suites (naive_baselines.py, naive_baselines2.py) ran at patient fold 0
only, so their rows in the ladder rested on three cells while the graph rows rested on
every cell. This script runs every linear predictor on all patient folds crossed with all
reaction folds, on the byte-identical fold construction, so that each linear row can be
paired cell by cell with the graph rows.

Two things are added on the way.

  The structure floor is ablated into what is topology and what is curated annotation.
  Of the 69 columns, six are topological and come from the edges and the stoichiometric
  matrix alone (three degrees, a log degree, the metabolite count, and whether the
  reaction touches a single metabolite, which is what an exchange reaction is), and 63
  are curated annotation (whether a gene rule exists, how many genes it names, the
  reversibility bound, and 60 subsystem indicators). Each block is fitted alone and
  together.

  The regularization strength is tuned by inner cross-validation on train reactions,
  for every fitted predictor, expression-only and structural alike, so that a
  69-column design and a 3-column design are not handed the same penalty. The fixed
  C = 1 fits of the earlier suites are kept beside the tuned ones.

No patient's label ever enters a fit: the labels are patient-invariant and the fit uses
train reactions only. Output: out/naive_allfolds.json keyed by regime, predictor, cell.

Beside every AUROC the file also records the average precision (the area under the
precision-recall curve as scikit-learn computes it) of the same scores, under a parallel
top-level key "auprc" with the identical regime, predictor, cell layout, together with the
positive-class prevalence of every cell's held-out reaction set ("_heldout_prevalence"),
which is the value average precision takes for a random ranking. The per-patient rows
average the per-patient average precision over the test patients of the cell, as the
AUROC rows do. The AUROC layout is unchanged.
"""
import json, os, re, sys, time, numpy as np, pandas as pd, torch, h5py
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score

# Environment overrides, used by the robustness checks: P1_DATA points at a data directory whose
# labels differ (the HT29-only label), P1_OUT at the file to write, P1_ALIGNED at the aligned
# reference network, and P1_EXPR_TRANSFORM=rank replaces every patient's expression column by its
# within-patient percentile among the expression-bearing reactions (zeros stay zero), a second
# normalization for the expression rows.
D = os.path.expanduser(os.environ.get("P1_DATA", "~/metagnn/work/crc_624"))
OUT = os.path.expanduser(os.environ.get("P1_OUT", "~/metagnn/out/naive_allfolds.json"))
ALIGNED = os.path.expanduser(os.environ.get("P1_ALIGNED", "~/metagnn/out/recon3d_aligned.json"))
EXPR_TRANSFORM = os.environ.get("P1_EXPR_TRANSFORM", "none")
N = 10600; SEED = 2024; NJOBS = int(os.environ.get("NJOBS", "16"))
CS = [0.01, 0.1, 1.0, 10.0]
y = np.asarray(torch.load(D + "/activity_pseudolabels.pt", map_location="cpu",
                          weights_only=False)).astype(int)

# ---- folds, identical to double_holdout.py ---------------------------------------------
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
# P1_FAMILY_MAP: a families_*.json from build_families.py. The reaction folds are then drawn family
# by family (family_folds.py) and the inner cross-validation that tunes C groups the training
# reactions by family, so that no relative of a held-out or inner-held-out reaction is fitted.
FAMILY_MAP = os.environ.get("P1_FAMILY_MAP")
FAMILY = None
if FAMILY_MAP:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import family_folds as FF
    FAMILY, FAMILY_SCHEME = FF.load_family_map(os.path.expanduser(FAMILY_MAP))
    rfolds = FF.family_rfolds(FAMILY, y, has, n_folds=3, seed=SEED)
    for _r in FF.fold_report(FAMILY, rfolds, y, has):
        print("family fold", _r, flush=True)

# ---- structural and annotation columns ---------------------------------------------------
# Annotation comes from recon3d_aligned.json, the reference network in the canonical order of the
# training arrays (see build_aligned_recon3d.py); the BiGG JSON is never indexed by position.
AL = json.load(open(ALIGNED))["reactions"]; assert len(AL) == N
deg = {}
for e in ["substrate_of", "produces", "shared_metabolite"]:
    t = torch.load(f"{D}/edge_indices/{e}.pt", map_location="cpu", weights_only=False).numpy()
    d = np.zeros(N); src = t[0] if e != "produces" else t[1]
    np.add.at(d, src[src < N], 1); deg[e] = d
# Subsystem indicators: the 60 most frequent subsystems over the whole network, a label-free rule;
# reactions whose subsystem is unknown (no annotation, or the 289 that could not be aligned) get an
# all-zero row rather than an indicator of their own.
sub = pd.Series([(r.get("subsystem") or "(unknown)") for r in AL])
top = [s for s in sub.value_counts().index if s != "(unknown)"][:60]
n_met = np.array([len(r["metabolites"]) for r in AL], float)
# Topology: read from the edges and the stoichiometric matrix only. An exchange reaction is one that
# touches a single metabolite, which is a structural property; the identifier prefix is not used.
TOPO = np.column_stack(
    [deg["substrate_of"], deg["produces"], deg["shared_metabolite"], np.log1p(deg["shared_metabolite"]),
     n_met, (n_met == 1).astype(float)])
# Annotation: curated fields of the reference network. Reversibility is a curated flux bound and so
# belongs here; the 289 unaligned reactions carry no bound and are counted as irreversible, and no
# column marks them.
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

# ---- a nearest-neighbor label lookup: similarity from stoichiometry and gene rules only ---------
# For every held-out reaction, the k = 5 most similar training reactions by a label-free similarity
# (the mean of the Jaccard overlap of compartment-stripped metabolite sets and, where both rules are
# nonempty, of gene sets), and the score is their similarity-weighted mean training label. Under the
# random split this is what a duplicate can be recovered by; under the family split every twin has
# left the training set and the lookup has to fall back on weaker relatives.
import scipy.sparse as _sps
def _stripped(mid): return mid.rsplit("_", 1)[0]
_met_ids = {}; _rows = []; _cols = []
for i, r in enumerate(AL):
    for m in (r.get("metabolites") or {}):
        j = _met_ids.setdefault(_stripped(m), len(_met_ids)); _rows.append(i); _cols.append(j)
MET = _sps.csr_matrix((np.ones(len(_rows)), (_rows, _cols)), shape=(N, len(_met_ids)))
MET.data[:] = 1.0; MET.sum_duplicates(); MET.data[:] = 1.0
_gene_ids = {}; _rows = []; _cols = []
for i, r in enumerate(AL):
    for gs in (r.get("gene_sets") or []):
        for g in (gs if isinstance(gs, (list, tuple)) else [gs]):
            j = _gene_ids.setdefault(re.sub(r"_AT\d+$", "", str(g)), len(_gene_ids)); _rows.append(i); _cols.append(j)
GEN = _sps.csr_matrix((np.ones(len(_rows)), (_rows, _cols)), shape=(N, max(1, len(_gene_ids))))
GEN.sum_duplicates(); GEN.data[:] = 1.0
def nearest_lookup(rtr, rte, k=5):
    def jac(M):
        inter = (M[rte] @ M[rtr].T).toarray()
        sa = np.asarray(M[rte].sum(1)).ravel()[:, None]; sb = np.asarray(M[rtr].sum(1)).ravel()[None, :]
        den = sa + sb - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(den > 0, inter / den, 0.0), (sa > 0) & (sb > 0)
    jm, _ = jac(MET); jg, okg = jac(GEN)
    sim = np.where(okg, 0.5 * jm + 0.5 * jg, jm)
    top = np.argpartition(-sim, k, axis=1)[:, :k]
    w = np.take_along_axis(sim, top, 1); lab = y[rtr][top]
    with np.errstate(divide="ignore", invalid="ignore"):
        score = np.where(w.sum(1) > 0, (w * lab).sum(1) / w.sum(1), y[rtr].mean())
    return score

def fit_lr(X, rtr, rte, tuned):
    base = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000))
    if tuned:
        if FAMILY is not None:
            from sklearn.model_selection import GroupKFold
            inner = list(GroupKFold(3).split(np.zeros(len(rtr)), y[rtr], groups=FAMILY[rtr]))
        else:
            inner = StratifiedKFold(3, shuffle=True, random_state=SEED)
        m = GridSearchCV(base, {"logisticregression__C": CS}, cv=inner, scoring="roc_auc", n_jobs=1)
    else:
        m = base
    m.fit(X[rtr], y[rtr])
    s = m.predict_proba(X[rte])[:, 1]
    return float(roc_auc_score(y[rte], s)), float(average_precision_score(y[rte], s))

def per_patient(Xte, build, rtr, rte, tuned):
    res = Parallel(n_jobs=NJOBS)(delayed(fit_lr)(build(xp), rtr, rte, tuned) for xp in Xte)
    return float(np.mean([r[0] for r in res])), float(np.mean([r[1] for r in res]))

def both(scores, rte):
    """AUROC and average precision of one score vector on the held-out reactions."""
    return float(roc_auc_score(y[rte], scores)), float(average_precision_score(y[rte], scores))

R = {"fixed": {}, "tuned": {}, "auprc": {"fixed": {}, "tuned": {}}}
if os.path.exists(OUT):
    R = json.load(open(OUT))
    R.setdefault("fixed", {}); R.setdefault("tuned", {})
    R.setdefault("auprc", {}); R["auprc"].setdefault("fixed", {}); R["auprc"].setdefault("tuned", {})

def put(regime, key, cell, val):
    """Store an (AUROC, average precision) pair: the AUROC in the regime block, the average
    precision under the parallel "auprc" block."""
    auc, ap = val
    R[regime].setdefault(key, {})[cell] = auc
    R["auprc"][regime].setdefault(key, {})[cell] = ap

for p, sp in enumerate(pfolds):
    Xtr_mean = load(sp["train"]).mean(0)
    Xte = load(sp["test"])
    for rf, (rtr, rte) in enumerate(rfolds):
        cell = f"p{p}r{rf}"
        if all(cell in R["tuned"].get(k, {}) and cell in R["auprc"]["tuned"].get(k, {})
               for k in ("structure_plus_perpat",)):
            print(f"{cell} present, skipped", flush=True); continue
        t0 = time.time()
        R.setdefault("_heldout_prevalence", {})[cell] = float(y[rte].mean())
        R.setdefault("_n_heldout", {})[cell] = int(len(rte))
        # unfitted rows, regime-free but stored under both for convenience
        for regime in ("fixed", "tuned"):
            put(regime, "indicator", cell, both(has[rte].astype(float), rte))
            pp = [both(s[rte], rte) for s in Xte]
            put(regime, "expr_perpat_rank", cell, (float(np.mean([v[0] for v in pp])), float(np.mean([v[1] for v in pp]))))
            put(regime, "expr_cohortmean_rank", cell, both(Xtr_mean[rte], rte))
            put(regime, "degree_shared", cell, both(deg["shared_metabolite"][rte], rte))
            rate = pd.Series(y[rtr]).groupby(sub.values[rtr]).mean(); glob_rate = float(y[rtr].mean())
            put(regime, "subsystem_prevalence", cell, both(np.array([rate.get(s, glob_rate) for s in sub.values[rte]]), rte))
            put(regime, "nearest_lookup_k5", cell, both(nearest_lookup(rtr, rte), rte))
        for regime, tuned in (("fixed", False), ("tuned", True)):
            put(regime, "topology_only", cell, fit_lr(TOPO, rtr, rte, tuned))
            put(regime, "annotation_only", cell, fit_lr(ANNOT, rtr, rte, tuned))
            put(regime, "structure_only", cell, fit_lr(STRUCT, rtr, rte, tuned))
            put(regime, "expr_cohortmean_lr", cell, fit_lr(expr_cols(Xtr_mean), rtr, rte, tuned))
            put(regime, "structure_plus_cohortmean", cell,
                fit_lr(np.column_stack([STRUCT, Xtr_mean, (Xtr_mean > 0).astype(float)]), rtr, rte, tuned))
            put(regime, "expr_perpat_lr", cell, per_patient(Xte, expr_cols, rtr, rte, tuned))
            put(regime, "structure_plus_perpat", cell,
                per_patient(Xte, lambda xp: np.column_stack([STRUCT, xp, (xp > 0).astype(float)]), rtr, rte, tuned))
        R["_n_test_patients"] = R.get("_n_test_patients", {}); R["_n_test_patients"][f"p{p}"] = len(sp["test"])
        R["_columns"] = dict(topology=TOPO.shape[1], annotation=ANNOT.shape[1], structure=STRUCT.shape[1])
        R["_C_grid"] = CS
        R["_data"] = D; R["_expr_transform"] = EXPR_TRANSFORM; R["_n_active_labels"] = int(y.sum())
        R["_family_map"] = (os.path.basename(FAMILY_MAP) if FAMILY_MAP else None)
        json.dump(R, open(OUT, "w"), indent=1)
        f, t = R["fixed"], R["tuned"]; ta = R["auprc"]["tuned"]
        print(f"{cell} done in {time.time()-t0:.0f} s | fixed: topo {f['topology_only'][cell]:.4f} "
              f"annot {f['annotation_only'][cell]:.4f} struct {f['structure_only'][cell]:.4f} "
              f"s+coh {f['structure_plus_cohortmean'][cell]:.4f} s+per {f['structure_plus_perpat'][cell]:.4f} "
              f"| tuned: struct {t['structure_only'][cell]:.4f} s+coh {t['structure_plus_cohortmean'][cell]:.4f} "
              f"s+per {t['structure_plus_perpat'][cell]:.4f} e+coh {t['expr_cohortmean_lr'][cell]:.4f} "
              f"e+per {t['expr_perpat_lr'][cell]:.4f} | AUPRC tuned: struct {ta['structure_only'][cell]:.4f} "
              f"s+per {ta['structure_plus_perpat'][cell]:.4f} e+per {ta['expr_perpat_lr'][cell]:.4f} "
              f"(prevalence {R['_heldout_prevalence'][cell]:.4f})", flush=True)
print("wrote", OUT)
