"""metabench: five checks for benchmarks whose labels are shared across samples.

Motivation
----------
When a benchmark's target vector is identical for every sample, holding out samples
does not hide a single label. A model with one free parameter per entity can then
reproduce the target exactly and score at ceiling without ever reading a sample.
These five functions detect that situation and quantify what a reported score is
actually made of. They are deliberately framework-agnostic: each takes numpy arrays
that any pipeline already produces.

Each check answers one question:

    memorization_gap      is the model reproducing entity identity rather than
                          generalizing? Requires an entity-axis hold-out.
    indicator_floor       how much AUROC is available for free from the fact that
                          some entities receive input and others do not?
    structure_floor       how much is available from network topology alone, with
                          no sample data of any kind?
    cohort_mean_gain      does replacing every sample's input with the training
                          cohort mean change accuracy? If it improves, sample
                          identity is costing you.
    dispersion_ratio      does the output depend on the input at all?

See docs/PROTOCOL.md for how to wire them into an existing evaluation.
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score

__all__ = ["memorization_gap", "indicator_floor", "structure_floor",
           "cohort_mean_gain", "dispersion_ratio", "entity_folds"]


def entity_folds(y, has_input=None, n_folds=3, seed=2024):
    """Stratified folds over the ENTITY axis (reactions, genes, targets).

    Holding out entities as well as samples is the joint row-and-column exclusion
    long established for relational prediction. Stratifying on the label crossed
    with input availability keeps each fold comparable.
    """
    from sklearn.model_selection import StratifiedKFold
    y = np.asarray(y).astype(int)
    strat = y if has_input is None else 2 * y + np.asarray(has_input).astype(int)
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), strat))


def memorization_gap(scores, y, train_mask):
    """AUROC on entities seen in training minus AUROC on entities held out.

    `scores` is (n_samples, n_entities) for held-out samples; `y` is the shared
    label vector; `train_mask` marks entities whose labels entered the loss.

    A gap near zero means the model is using its input. A large gap means it is
    using entity identity, which a sample-only split cannot reveal.
    """
    scores = np.asarray(scores, float); y = np.asarray(y).astype(int)
    m = np.asarray(train_mask, bool)
    if scores.ndim == 1: scores = scores[None, :]
    def au(mask):
        v = [roc_auc_score(y[mask], s[mask]) for s in scores if len(set(y[mask])) > 1]
        return float(np.mean(v)) if v else float("nan")
    tr, ho = au(m), au(~m)
    return {"train_entities": round(tr, 4), "heldout_entities": round(ho, 4),
            "gap": round(tr - ho, 4),
            "verdict": ("memorizing entity identity" if tr - ho > 0.05
                        else "gap is small; consistent with genuine generalization")}


def indicator_floor(y, has_input):
    """AUROC of a binary flag for whether an entity receives any input at all.

    Equals 1/2 + 1/2 (TPR - FPR), the informedness identity. This carries no
    information from the measurement itself, so any score below it is worse than
    knowing nothing, and any score above it should be credited against it rather
    than against 0.5.
    """
    y = np.asarray(y).astype(int); h = np.asarray(has_input).astype(bool)
    tpr, fpr = h[y == 1].mean(), h[y == 0].mean()
    return {"auroc": round(float(roc_auc_score(y, h.astype(float))), 4),
            "closed_form": round(float(0.5 + 0.5 * (tpr - fpr)), 4),
            "prevalence_with_input": round(float(y[h].mean()), 4),
            "prevalence_without_input": round(float(y[~h].mean()), 4)}


def structure_floor(features, y, folds, C=1.0):
    """AUROC of logistic regression on entity features containing NO sample data.

    Pass topological or otherwise sample-independent per-entity features. Whatever
    this reaches is inherited for free by every method scored on the benchmark.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    X = np.asarray(features, float); y = np.asarray(y).astype(int); out = []
    for tr, te in folds:
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, C=C))
        m.fit(X[tr], y[tr]); out.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
    return {"per_fold": [round(v, 4) for v in out], "mean": round(float(np.mean(out)), 4)}


def cohort_mean_gain(scores_real, scores_cohort_mean, y, heldout_mask=None):
    """Accuracy with each sample's own input, against every sample shown the
    training-cohort mean.

    A positive `gain` means substituting one average vector for everybody made the
    model BETTER, so sample identity has negative marginal value. Under a target
    that does not vary across samples this is the expected direction, and measuring
    it tells you whether the benchmark can reward personalization at all.
    """
    y = np.asarray(y).astype(int)
    m = np.ones(len(y), bool) if heldout_mask is None else np.asarray(heldout_mask, bool)
    def au(S):
        S = np.atleast_2d(np.asarray(S, float))
        return float(np.mean([roc_auc_score(y[m], s[m]) for s in S]))
    a, b = au(scores_real), au(scores_cohort_mean)
    return {"own_input": round(a, 4), "cohort_mean_input": round(b, 4),
            "gain_from_substitution": round(b - a, 4),
            "verdict": ("sample identity has NEGATIVE marginal value" if b > a
                        else "sample identity helps")}


def dispersion_ratio(scores, mc_std, n_mc):
    """Between-sample spread of the output against its own sampling noise.

    `scores` is (n_samples, n_entities) posterior means, `mc_std` the matching
    Monte Carlo standard deviations over `n_mc` stochastic passes. A value at the
    no-signal null (near 1) means the output does not vary with the sample beyond
    what resampling alone produces: the model is not reading its input.
    """
    S = np.asarray(scores, float); U = np.asarray(mc_std, float)
    rho = float(S.std(0).mean() / (U.mean() / np.sqrt(n_mc)))
    C = np.corrcoef(S)
    r = float(np.median(C[np.triu_indices(len(C), 1)])) if len(C) > 1 else float("nan")
    return {"rho": round(rho, 4), "median_inter_sample_r": round(r, 4),
            "verdict": ("output is effectively input-independent" if rho < 1.2
                        else "output varies with the input")}
