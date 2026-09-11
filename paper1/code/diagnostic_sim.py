#!/usr/bin/env python3
"""A diagnostic calibration study for the input-substitution audit (P1-E3).

The audit reads two contrasts on a double hold-out: own minus cohort-mean (does the patient's own
expression column beat the training-cohort-mean column shown to everyone) and own minus donor (does
it beat a wrong patient's column), each tested for sign across the cells of a patient x reaction
grid. This script asks what those contrasts do when the ground truth is known, so that the audit's
readings can be calibrated rather than trusted.

Independent synthetic worlds are generated with three knobs: a reaction term (a shared propensity
per reaction), a sample term (a per-patient shift), and a controlled sample-by-reaction interaction
whose strength alpha sets how much of the label depends on the individual profile. A world
also carries an assay-noise level, a near-duplicate rate (reactions copied with jitter, so a random
reaction split leaks twins across it) and a label prevalence. Two label families are built on the
same expression: a shared label (one vector for every patient, like the reconstruction proxy) and a
patient-varying label (each patient's own threshold on their own column).

Five predictors are run through the whole audit, including tuning and donor selection: a sample-blind
model (reaction features only), an oracle (given the true sample-specific signal), an
expression-sensitive but label-irrelevant model (fed a permuted expression matrix), a learned
gradient-boosted model, and and the restricted nonlinear scorer (x-1)^2. That last one is the case where averaging the column across patients destroys distributional information the per-patient
scorer keeps, so a shared label does not force the own-minus-mean contrast negative.

The decision rule (the sign and interval of each contrast, and how they combine into a verdict) is
developed on world set A, locked, and its false-positive rate and coverage measured on the disjoint
world set B, with the Monte Carlo count raised until the binomial standard error of the
false-positive rate is small. The output is results/diagnostic_sim.json and a decision table.

CPU only. The learners are small enough that the whole study runs on a laptop in minutes.
"""
import os, json, argparse, numpy as np
def _j(o):
    import numpy as _np
    if isinstance(o, (_np.integer,)): return int(o)
    if isinstance(o, (_np.floating,)): return float(o)
    if isinstance(o, (_np.bool_,)): return bool(o)
    if isinstance(o, _np.ndarray): return o.tolist()
    raise TypeError(str(type(o)))
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "diagnostic_sim.json"))
ap.add_argument("--n_dev", type=int, default=60, help="world set A (develop the rule)")
ap.add_argument("--n_test", type=int, default=240, help="world set B (assess coverage/false positives)")
ap.add_argument("--P", type=int, default=60, help="patients per world")
ap.add_argument("--R", type=int, default=400, help="reactions per world")
ap.add_argument("--pfolds", type=int, default=3); ap.add_argument("--rfolds", type=int, default=3)
ap.add_argument("--quick", action="store_true", help="tiny run for a smoke test")
A = ap.parse_args()
if A.quick: A.n_dev, A.n_test, A.P, A.R = 8, 16, 40, 200

def make_world(seed, alpha, noise, dup_rate, prevalence, label):
    """Return (X patients x reactions, Y patients x reactions int, vary_mask, oracle_signal)."""
    rng = np.random.default_rng(seed)
    P, R = A.P, A.R
    beta_r = rng.normal(0, 1.0, R)                          # reaction propensity (shared)
    s_i = rng.normal(0, 1.0, P)                             # sample shift
    base = rng.gamma(2.0, 1.0, (P, R))                      # expression, nonnegative
    base += s_i[:, None] * 0.5                              # sample term enters expression
    base = np.clip(base, 0, None)
    # near-duplicate reactions: copy a share of columns with jitter (a random split leaks these)
    n_dup = int(dup_rate * R)
    if n_dup:
        src = rng.choice(R, n_dup, replace=False); dst = rng.choice(R, n_dup, replace=False)
        base[:, dst] = base[:, src] + rng.normal(0, 0.05, (P, n_dup))
    X = base
    # the sample-specific signal that a valid predictor would need: an interaction of the patient's
    # own centered column with the reaction
    g = (X - X.mean(0)) * beta_r[None, :]
    oracle = g
    lin = beta_r[None, :] + alpha * g + rng.normal(0, noise, (P, R))
    if label == "shared":
        # one shared label vector: threshold the reaction term alone at the target prevalence
        v = beta_r + rng.normal(0, noise, R)
        thr = np.quantile(v, 1 - prevalence); yv = (v > thr).astype(int)
        Y = np.tile(yv, (P, 1)); vary = np.zeros(R, bool)
    else:
        thr = np.quantile(lin, 1 - prevalence); Y = (lin > thr).astype(int); vary = np.ones(R, bool)
    return X, Y, vary, oracle

def score_matrix(kind, Xtr, Ytr, Xte, rtr, oracle_te):
    """Return a P_te x R score matrix for the held-out reactions from a predictor of the given kind,
    trained on train patients x train reactions. Rows differ across patients only if the predictor
    uses the patient's own column."""
    Pte = Xte.shape[0]; R = Xte.shape[1]
    if kind == "blind":
        # a sample-blind ranking: ridge regression from reaction-level features (train-patient column
        # mean and spread) onto the reaction-level positive rate, giving one shared score row. Ridge
        # (not logistic) so a fold whose reaction-level rate never crosses one half still fits.
        from sklearn.linear_model import Ridge
        f_tr = np.column_stack([Xtr[:, rtr].mean(0), Xtr[:, rtr].std(0)])
        target = Ytr[:, rtr].mean(0)
        m = Ridge(alpha=1.0).fit(f_tr, target)
        s = m.predict(np.column_stack([Xtr.mean(0), Xtr.std(0)]))
        return np.tile(s, (Pte, 1))
    if kind == "oracle":
        return oracle_te                                    # the true per-patient signal
    if kind == "label_irrelevant":
        rng = np.random.default_rng(0)
        return Xte[:, :] * 0 + rng.normal(0, 1, (Pte, R))   # noise, unrelated to the label
    if kind == "restricted_nonlinear":
        return (Xte - 1.0) ** 2                              # the (x-1)^2 case of the roadmap, per patient
    if kind == "learned":
        # a gradient-boosted model on [own value, column mean, own - mean], trained on stacked train rows
        def feats(Xm):
            cm = Xtr[:, :].mean(0)
            return np.stack([Xm, np.tile(cm, (Xm.shape[0], 1)), Xm - cm[None, :]], -1)
        Ftr = feats(Xtr)[:, rtr, :].reshape(-1, 3); ytr = Ytr[:, rtr].reshape(-1)
        m = HistGradientBoostingClassifier(max_iter=80, max_depth=3).fit(Ftr, ytr)
        Fte = feats(Xte).reshape(-1, 3)
        return m.predict_proba(Fte)[:, 1].reshape(Pte, R)
    raise ValueError(kind)

def audit(X, Y, vary, oracle, seed):
    """Run the double hold-out audit for every predictor; return own/mean/donor AUROC per cell and
    the paired contrasts, on held-out reactions, mean over patient x reaction folds."""
    rng = np.random.default_rng(seed); P, R = X.shape
    pf = np.array_split(rng.permutation(P), A.pfolds); rf = np.array_split(rng.permutation(R), A.rfolds)
    out = {}
    for kind in ("blind", "oracle", "label_irrelevant", "restricted_nonlinear", "learned"):
        own_cell, mean_cell, don_cell = [], [], []
        for pfi in range(A.pfolds):
            te_p = pf[pfi]; tr_p = np.concatenate([pf[k] for k in range(A.pfolds) if k != pfi])
            donor = rng.permutation(te_p)                    # a derangement-ish donor within the test patients
            while np.any(donor == te_p): donor = rng.permutation(te_p)
            for rfi in range(A.rfolds):
                rte = rf[rfi]; rtr = np.concatenate([rf[k] for k in range(A.rfolds) if k != rfi])
                Xte = X[te_p]; ote = oracle[te_p]
                S = score_matrix(kind, X[tr_p], Y[tr_p], Xte, rtr, ote)
                # cohort-mean arm: every test patient scored by the mean score row over test patients
                Smean = np.tile(S.mean(0), (len(te_p), 1))
                # donor arm: patient scored by the donor patient's score row (only differs if S varies by patient)
                Sdon = S[np.searchsorted(np.sort(te_p), donor)] if False else S[[list(te_p).index(d) for d in donor]]
                yv = Y[te_p]
                def mauc(Sm):
                    v = []
                    for i in range(len(te_p)):
                        yy = yv[i, rte] if vary.any() else Y[te_p][0, rte]
                        if len(set(yy)) > 1: v.append(roc_auc_score(yy, Sm[i, rte]))
                    return np.mean(v) if v else np.nan
                own_cell.append(mauc(S)); mean_cell.append(mauc(Smean)); don_cell.append(mauc(Sdon))
        own_cell = np.array(own_cell); mean_cell = np.array(mean_cell); don_cell = np.array(don_cell)
        d_mean = own_cell - mean_cell; d_don = own_cell - don_cell
        def signflip_p(d):
            d = d[np.isfinite(d)]
            if len(d) < 2: return 1.0
            k = (d > 0).sum(); n = len(d)
            from scipy.stats import binomtest
            return binomtest(k, n, 0.5).pvalue
        out[kind] = dict(own=float(np.nanmean(own_cell)), mean=float(np.nanmean(mean_cell)), donor=float(np.nanmean(don_cell)),
                         own_minus_mean=float(np.nanmean(d_mean)), own_minus_donor=float(np.nanmean(d_don)),
                         n_cells=int(np.isfinite(d_mean).sum()),
                         own_minus_mean_pos=int((d_mean > 0).sum()), own_minus_donor_pos=int((d_don > 0).sum()),
                         signflip_mean_p=signflip_p(d_mean), signflip_donor_p=signflip_p(d_don))
    return out

# the parameter grid for the worlds
GRID = [dict(alpha=a, noise=nz, dup_rate=dr, prevalence=pv, label=lb)
        for a in (0.0, 0.5, 1.5, 3.0) for nz in (0.3, 1.0) for dr in (0.0, 0.3) for pv in (0.3,) for lb in ("shared", "varying")]

def run_worlds(n, seed0, tag):
    rows = []
    for w in range(n):
        cfg = GRID[w % len(GRID)]; seed = seed0 + w
        X, Y, vary, oracle = make_world(seed, cfg["alpha"], cfg["noise"], cfg["dup_rate"], cfg["prevalence"], cfg["label"])
        res = audit(X, Y, vary, oracle, seed + 10000)
        rows.append(dict(world=w, seed=seed, **cfg,
                         sample_specific=bool(cfg["label"] == "varying" and cfg["alpha"] > 0), audit=res))
        if (w + 1) % 20 == 0: print(f"{tag}: {w+1}/{n} worlds", flush=True)
    return rows

print(f"developing on {A.n_dev} worlds, assessing on {A.n_test} worlds; {A.P} patients x {A.R} reactions each", flush=True)
dev = run_worlds(A.n_dev, 1, "dev"); test = run_worlds(A.n_test, 100000, "test")

# ---- the locked decision rule, developed on dev ------------------------------------------------
# A predictor is called to carry sample-specific information for a world when BOTH the own-minus-mean
# and own-minus-donor contrasts are positive with a sign-flip p below 0.05 on the learned model.
# The rule is fixed here; the threshold 0.05 is the sign-flip test's conventional level, not tuned.
def verdict(world, kind="learned"):
    a = world["audit"][kind]
    import math
    if any(math.isnan(a[k]) for k in ("own_minus_mean", "own_minus_donor")): return False
    return (a["own_minus_mean"] > 0 and a["signflip_mean_p"] < 0.05 and
            a["own_minus_donor"] > 0 and a["signflip_donor_p"] < 0.05)

def confusion(worlds, kind="learned"):
    tp = fp = tn = fn = 0
    for w in worlds:
        truth = w["sample_specific"]; call = verdict(w, kind)
        tp += int(truth and call); fp += int((not truth) and call); tn += int((not truth) and (not call)); fn += int(truth and (not call))
    n_null = tn + fp; fpr = fp / n_null if n_null else float("nan")
    se = (fpr * (1 - fpr) / n_null) ** 0.5 if n_null else float("nan")
    return dict(tp=tp, fp=fp, tn=tn, fn=fn, false_positive_rate=round(fpr, 4), fpr_binom_se=round(se, 4),
                power=round(tp / (tp + fn), 4) if (tp + fn) else None, n_null=n_null, n_positive=tp + fn)

# the (x-1)^2 counterexample, constructed exactly as the roadmap states and scaled up: for each of
# many reaction pairs, a positive reaction with expression {0, 2} across two patients and a negative
# reaction with expression 1 in both; the label (positive > negative) is SHARED by both patients, yet
# the per-patient (x-1)^2 scorer ranks each patient's pair correctly while the cohort-mean column
# (both means equal 1) cannot. This shows a shared target does not force own-minus-mean below zero.
def toy_counterexample(n_pairs=200, seed=0):
    rng = np.random.default_rng(seed)
    # two patients, 2*n_pairs reactions: for pair k, reaction 2k is "positive", 2k+1 is "negative"
    P = 2; R = 2 * n_pairs
    X = np.zeros((P, R)); y = np.zeros(R, int)
    for k in range(n_pairs):
        X[0, 2 * k] = 0.0; X[1, 2 * k] = 2.0          # positive reaction: 0 in patient 0, 2 in patient 1
        X[:, 2 * k + 1] = 1.0                          # negative reaction: 1 in both
        y[2 * k] = 1                                    # the shared label: positive > negative
    own = (X - 1.0) ** 2                               # per-patient (x-1)^2 scorer (larger = more likely positive)
    mean_col = np.tile(X.mean(0), (P, 1)); mean_score = (mean_col - 1.0) ** 2
    own_auc = np.mean([roc_auc_score(y, own[i]) for i in range(P)])
    mean_auc = np.mean([roc_auc_score(y, mean_score[i]) for i in range(P)])
    return dict(n_pairs=n_pairs, own_auroc=round(float(own_auc), 4), cohort_mean_auroc=round(float(mean_auc), 4),
                own_minus_mean=round(float(own_auc - mean_auc), 4),
                note="shared label; own (x-1)^2 ranks every patient's pair correctly (AUROC 1) while the cohort-mean "
                     "column is constant at 1 for both reactions (AUROC 0.5), so own-minus-mean is +0.5")
TOY = toy_counterexample()
restr = [w["audit"]["restricted_nonlinear"]["own_minus_mean"] for w in dev + test if w["label"] == "shared"]
blind_shared = [w["audit"]["blind"]["own_minus_mean"] for w in dev + test if w["label"] == "shared"]
OUT = dict(
    note="calibration of the input-substitution audit on synthetic worlds; the decision rule is developed on the "
         "development worlds, locked, and assessed on the disjoint test worlds",
    params=dict(n_dev=A.n_dev, n_test=A.n_test, P=A.P, R=A.R, pfolds=A.pfolds, rfolds=A.rfolds, grid=GRID),
    rule="learned model: own>mean and own>donor, each sign-flip p<0.05 across cells",
    dev_confusion={k: confusion(dev, k) for k in ("learned", "oracle", "blind")},
    test_confusion={k: confusion(test, k) for k in ("learned", "oracle", "blind", "restricted_nonlinear", "label_irrelevant")},
    counterexample=dict(
        constructed=TOY,
        note="the constructed case proves a shared target does not force own-minus-mean below zero; in the noisy "
             "generic worlds the generic (x-1)^2 scorer is not aligned with the generic label, so its contrast sits near zero",
        generic_restricted_nonlinear_own_minus_mean_mean=round(float(np.mean(restr)), 4),
        generic_blind_own_minus_mean_mean=round(float(np.mean(blind_shared)), 4)),
    decision_table=[])
# a compact decision table: for each true regime and predictor, the mean contrasts and the verdict rate
regimes = [("shared label, alpha=0 (no sample signal)", lambda w: w["label"] == "shared"),
           ("varying label, alpha=0 (threshold on own column, weak)", lambda w: w["label"] == "varying" and w["alpha"] == 0.0),
           ("varying label, alpha>=1.5 (strong sample signal)", lambda w: w["label"] == "varying" and w["alpha"] >= 1.5)]
for name, sel in regimes:
    ws = [w for w in dev + test if sel(w)]
    row = dict(regime=name, n_worlds=len(ws))
    for kind in ("learned", "oracle", "blind", "restricted_nonlinear"):
        om = np.mean([w["audit"][kind]["own_minus_mean"] for w in ws]); od = np.mean([w["audit"][kind]["own_minus_donor"] for w in ws])
        vr = np.mean([verdict(w, kind) for w in ws])
        row[kind] = dict(own_minus_mean=round(float(om), 4), own_minus_donor=round(float(od), 4), verdict_rate=round(float(vr), 4))
    OUT["decision_table"].append(row)
OUT["dev_worlds"] = dev; OUT["test_worlds"] = test
json.dump(OUT, open(A.out, "w"), indent=1, default=_j)
print("\ntest confusion (learned):", OUT["test_confusion"]["learned"])
print("counterexample (constructed):", TOY)
for row in OUT["decision_table"]:
    print(row["regime"], "| learned own-mean %.3f verdict %.2f | oracle verdict %.2f | blind own-mean %.3f verdict %.2f"
          % (row["learned"]["own_minus_mean"], row["learned"]["verdict_rate"], row["oracle"]["verdict_rate"],
             row["blind"]["own_minus_mean"], row["blind"]["verdict_rate"]))
print("wrote", A.out)
