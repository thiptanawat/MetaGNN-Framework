#!/usr/bin/env python3
"""A calibration study for the input-substitution audit on synthetic worlds (P1-E3).

The audit reads two contrasts on a double hold-out: own minus cohort-mean (does a predictor given
each patient's own expression column beat the same predictor given the training-cohort mean column)
and own minus donor (does it beat the same predictor given a wrong patient's real column), each read
for sign across the cells of a patient x reaction grid. This script asks what those contrasts do
when the ground truth is known, so that a reading on the real benchmark can be judged against
constructed cases rather than trusted.

Every arm here is an input substitution, as it is in the paper. The cohort-mean arm replaces each
patient's expression column, training patients included, with the mean column of the training
patients and refits the predictor on those inputs; the donor arm gives each patient another
patient's real column, a derangement among the training patients and a separate derangement among
the held-out patients, and refits on those. A predictor that fits nothing simply reads the
substituted input. This is not an averaging of a fitted model's outputs, which is a different
intervention and is not what the paper's arms do.

The worlds carry a reaction term (a shared propensity per reaction), a sample term (a per-patient
shift) and a controlled sample-by-reaction interaction whose strength alpha sets how much of the
label depends on the individual profile; also an assay-noise level, a near-duplicate rate
(reactions copied with jitter, so a random reaction split leaks twins) and a label prevalence. Two
label families are built on the same expression: a shared label (one vector for every patient, like
the reconstruction proxy) and a patient-varying label (each patient's own threshold on their own
column).

Five predictors run through the whole audit: a sample-blind reference (reaction-level features of
the unsubstituted training patients, so it is the same in every arm by construction), an oracle
handed the true sample-specific signal, an expression-sensitive but label-irrelevant scorer (each
patient's own column read through a fixed random permutation of the reactions, so its output moves
with the patient and carries nothing about the label), a learned gradient-boosted model whose
number of boosting rounds is selected on an inner partition of the training reactions withheld
from the loss, which is the paper's matched selection rule, and the restricted nonlinear scorer
(x-1)^2, the case in which averaging the column across patients destroys distributional information
the per-patient scorer keeps.

The paired contrasts are tested with the exact sign-flip test the paper uses: every assignment of
signs to the per-cell differences is enumerated and the two-sided p is the share whose absolute
mean reaches the observed one. A difference of exactly zero is invariant under a flip and so
contributes nothing; a set of contrasts that are all zero has p = 1, as it should.

Two facts about the design are recorded rather than hidden. Under a shared label a donor
substitution cannot change any predictor's mean per-patient AUROC, because the rows are permuted
against one label vector and the multiset of (input, label) pairs the model is fitted on is
unchanged; the own-minus-donor contrast is therefore identically zero on shared-label worlds, and
those worlds test the cohort-mean contrast only. The informative null for the donor contrast is the
varying-label world with alpha = 0. The output reports the false-positive count on each null
subset separately, and reports development and test worlds separately.

The decision rule (both contrasts positive with exact sign-flip p below 0.05 on the learned model)
is applied to world set A during development and its behavior measured on the disjoint world set B;
world seeds are listed in the output so that any world can be regenerated.

CPU only. The learners are small enough that the whole study runs on a laptop in under an hour.
"""
import os, json, argparse, itertools, numpy as np


def _j(o):
    import numpy as _np
    if isinstance(o, (_np.integer,)): return int(o)
    if isinstance(o, (_np.floating,)): return float(o)
    if isinstance(o, (_np.bool_,)): return bool(o)
    if isinstance(o, _np.ndarray): return o.tolist()
    raise TypeError(str(type(o)))


from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "diagnostic_sim.json"))
ap.add_argument("--n_dev", type=int, default=60, help="world set A (develop the rule)")
ap.add_argument("--n_test", type=int, default=240, help="world set B (assess false positives and detection)")
ap.add_argument("--P", type=int, default=60, help="patients per world")
ap.add_argument("--R", type=int, default=400, help="reactions per world")
ap.add_argument("--pfolds", type=int, default=3); ap.add_argument("--rfolds", type=int, default=3)
ap.add_argument("--inner_frac", type=float, default=0.2, help="share of training reactions withheld from the loss for selection")
ap.add_argument("--rounds", default="20,80,200", help="candidate boosting rounds the matched rule selects among")
ap.add_argument("--quick", action="store_true", help="tiny run for a smoke test")
ap.add_argument("--retable", action="store_true",
                help="recompute the summaries (confusions, null subsets, decision tables) from the per-world "
                     "records of an existing output file, without rerunning the worlds")
A = ap.parse_args()
if A.quick: A.n_dev, A.n_test, A.P, A.R = 8, 16, 40, 200
ROUNDS = sorted(int(x) for x in A.rounds.split(","))
ARMS = ("own", "mean", "donor")
KINDS = ("blind", "oracle", "label_irrelevant", "restricted_nonlinear", "learned")


def make_world(seed, alpha, noise, dup_rate, prevalence, label):
    """Return (X patients x reactions, Y patients x reactions int, vary flag, beta_r)."""
    rng = np.random.default_rng(seed)
    P, R = A.P, A.R
    beta_r = rng.normal(0, 1.0, R)                          # reaction propensity (shared)
    s_i = rng.normal(0, 1.0, P)                             # sample shift
    base = rng.gamma(2.0, 1.0, (P, R))                      # expression, nonnegative
    base += s_i[:, None] * 0.5                              # sample term enters expression
    base = np.clip(base, 0, None)
    n_dup = int(dup_rate * R)
    if n_dup:
        src = rng.choice(R, n_dup, replace=False); dst = rng.choice(R, n_dup, replace=False)
        base[:, dst] = base[:, src] + rng.normal(0, 0.05, (P, n_dup))
    X = base
    g = (X - X.mean(0)) * beta_r[None, :]                   # the sample-specific signal
    lin = beta_r[None, :] + alpha * g + rng.normal(0, noise, (P, R))
    if label == "shared":
        v = beta_r + rng.normal(0, noise, R)
        thr = np.quantile(v, 1 - prevalence); yv = (v > thr).astype(int)
        Y = np.tile(yv, (P, 1)); vary = False
    else:
        thr = np.quantile(lin, 1 - prevalence); Y = (lin > thr).astype(int); vary = True
    return X, Y, vary, beta_r


def oracle_signal(Xin, beta_r, cm):
    """The true sample-specific signal computed from whatever column the arm hands the predictor,
    centered on the training-cohort mean column the world was built around."""
    return (Xin - cm[None, :]) * beta_r[None, :]


def derange(rng, idx):
    p = rng.permutation(len(idx))
    while np.any(p == np.arange(len(idx))): p = rng.permutation(len(idx))
    return idx[p]


def substituted_inputs(arm, X, tr_p, te_p, rng):
    """The expression columns each arm hands the predictor, for training and held-out patients."""
    cm = X[tr_p].mean(0)
    if arm == "own":
        return X[tr_p], X[te_p], cm
    if arm == "mean":
        return np.tile(cm, (len(tr_p), 1)), np.tile(cm, (len(te_p), 1)), cm
    if arm == "donor":
        return X[derange(rng, tr_p)], X[derange(rng, te_p)], cm
    raise ValueError(arm)


def exact_signflip_p(d):
    """Two-sided exact sign-flip p over all 2^n sign assignments of the finite per-cell differences.
    A zero difference is invariant under a flip and contributes nothing; all-zero contrasts give 1."""
    d = np.asarray([x for x in d if np.isfinite(x)], float)
    n = len(d)
    if n == 0: return float("nan")
    obs = abs(d.mean())
    if obs == 0: return 1.0
    count = 0
    for signs in itertools.product((1.0, -1.0), repeat=n):
        if abs((d * np.array(signs)).mean()) >= obs - 1e-12: count += 1
    return count / 2 ** n


def per_patient_auroc(S, Yte, rte, vary):
    v = []
    for i in range(S.shape[0]):
        yy = Yte[i, rte] if vary else Yte[0, rte]
        if len(set(yy)) > 1: v.append(roc_auc_score(yy, S[i, rte]))
    return float(np.mean(v)) if v else float("nan")


def learned_scores(Xtr_in, Ytr, Xte_in, inner_perm, rte, cm):
    """The learned model under the matched selection rule: an inner partition of the training
    reactions (the same one for every arm of the cell) is withheld from the loss and used to pick
    the number of boosting rounds, then the model at that many rounds scores the held-out
    reactions of the held-out patients."""
    def feats(Xm):
        return np.stack([Xm, np.tile(cm, (Xm.shape[0], 1)), Xm - cm[None, :]], -1)
    n_in = max(1, int(round(A.inner_frac * len(inner_perm))))
    r_val, r_fit = inner_perm[:n_in], inner_perm[n_in:]
    Ftr = feats(Xtr_in); Fte = feats(Xte_in)
    F_fit = Ftr[:, r_fit, :].reshape(-1, 3); y_fit = Ytr[:, r_fit].reshape(-1)
    F_val = Ftr[:, r_val, :].reshape(-1, 3); y_val = Ytr[:, r_val].reshape(-1)
    if len(set(y_fit)) < 2:
        return np.full((Xte_in.shape[0], Xte_in.shape[1]), 0.5), None
    m = HistGradientBoostingClassifier(max_iter=max(ROUNDS), max_depth=3, random_state=0).fit(F_fit, y_fit)
    # one fit, read at each candidate number of rounds on the withheld reactions
    best, best_auc, chosen = None, -1.0, None
    stages_val = list(m.staged_predict_proba(F_val)); stages_te = list(m.staged_predict_proba(Fte.reshape(-1, 3)))
    for r in ROUNDS:
        k = min(r, len(stages_val)) - 1
        auc = roc_auc_score(y_val, stages_val[k][:, 1]) if len(set(y_val)) > 1 else 0.5
        if auc > best_auc: best_auc, chosen, best = auc, r, stages_te[k][:, 1]
    return best.reshape(Xte_in.shape[0], Xte_in.shape[1]), chosen


def audit(X, Y, vary, beta_r, seed):
    """Run every predictor through every arm on every cell; return per-kind contrasts and tests."""
    rng = np.random.default_rng(seed); P, R = X.shape
    pf = np.array_split(rng.permutation(P), A.pfolds); rf = np.array_split(rng.permutation(R), A.rfolds)
    cells = {kind: {arm: [] for arm in ARMS} for kind in KINDS}
    chosen_rounds = {arm: [] for arm in ARMS}
    for pfi in range(A.pfolds):
        te_p = pf[pfi]; tr_p = np.concatenate([pf[k] for k in range(A.pfolds) if k != pfi])
        # the sample-blind reference reads reaction-level features of the unsubstituted training
        # patients and is therefore identical in every arm; it is a reference, not an audited input
        f_tr_full = np.column_stack([X[tr_p].mean(0), X[tr_p].std(0)])
        for rfi in range(A.rfolds):
            rte = rf[rfi]; rtr = np.concatenate([rf[k] for k in range(A.rfolds) if k != rfi])
            cell_rng = np.random.default_rng(seed * 100 + pfi * 10 + rfi)
            perm_react = cell_rng.permutation(R)             # the label-irrelevant scorer's fixed permutation
            # the inner partition of training reactions withheld from the loss is drawn once per cell and
            # shared by every arm, as the paper's loss mask is; only the donor derangement is arm-specific
            inner_perm = cell_rng.permutation(rtr)
            blind = Ridge(alpha=1.0).fit(f_tr_full[rtr], Y[tr_p][:, rtr].mean(0)).predict(f_tr_full)
            S_blind = np.tile(blind, (len(te_p), 1))
            for arm in ARMS:
                arm_rng = np.random.default_rng(seed * 100 + pfi * 10 + rfi + {"own": 0, "mean": 1000, "donor": 2000}[arm])
                Xtr_in, Xte_in, cm = substituted_inputs(arm, X, tr_p, te_p, arm_rng)
                Yte = Y[te_p]
                for kind in KINDS:
                    if kind == "blind":
                        S = S_blind
                    elif kind == "oracle":
                        S = oracle_signal(Xte_in, beta_r, cm)
                    elif kind == "label_irrelevant":
                        S = Xte_in[:, perm_react]
                    elif kind == "restricted_nonlinear":
                        S = (Xte_in - 1.0) ** 2
                    else:
                        S, rounds = learned_scores(Xtr_in, Y[tr_p], Xte_in, inner_perm, rte, cm)
                        chosen_rounds[arm].append(rounds)
                    cells[kind][arm].append(per_patient_auroc(S, Yte, rte, vary))
    out = {}
    for kind in KINDS:
        own = np.array(cells[kind]["own"]); mean = np.array(cells[kind]["mean"]); don = np.array(cells[kind]["donor"])
        d_mean = own - mean; d_don = own - don
        out[kind] = dict(own=float(np.nanmean(own)), mean=float(np.nanmean(mean)), donor=float(np.nanmean(don)),
                         own_minus_mean=float(np.nanmean(d_mean)), own_minus_donor=float(np.nanmean(d_don)),
                         n_cells=int(np.isfinite(d_mean).sum()),
                         own_minus_mean_cells=[float(x) for x in d_mean], own_minus_donor_cells=[float(x) for x in d_don],
                         own_minus_mean_pos=int((d_mean > 0).sum()), own_minus_mean_zero=int((d_mean == 0).sum()),
                         own_minus_donor_pos=int((d_don > 0).sum()), own_minus_donor_zero=int((d_don == 0).sum()),
                         signflip_mean_p=exact_signflip_p(d_mean), signflip_donor_p=exact_signflip_p(d_don))
    out["learned"]["rounds_chosen"] = {arm: [int(r) for r in chosen_rounds[arm] if r is not None] for arm in ARMS}
    return out


GRID = [dict(alpha=a, noise=nz, dup_rate=dr, prevalence=pv, label=lb)
        for a in (0.0, 0.5, 1.5, 3.0) for nz in (0.3, 1.0) for dr in (0.0, 0.3) for pv in (0.3,) for lb in ("shared", "varying")]


def run_worlds(n, seed0, tag):
    rows = []
    for w in range(n):
        cfg = GRID[w % len(GRID)]; seed = seed0 + w
        X, Y, vary, beta_r = make_world(seed, cfg["alpha"], cfg["noise"], cfg["dup_rate"], cfg["prevalence"], cfg["label"])
        res = audit(X, Y, vary, beta_r, seed + 10000)
        rows.append(dict(world=w, seed=seed, **cfg,
                         sample_specific=bool(cfg["label"] == "varying" and cfg["alpha"] > 0),
                         null_kind=("shared_label" if cfg["label"] == "shared" else
                                    ("varying_label_no_signal" if cfg["alpha"] == 0 else None)),
                         audit=res))
        if (w + 1) % 10 == 0: print(f"{tag}: {w+1}/{n} worlds", flush=True)
    return rows


if A.retable:
    _prev = json.load(open(A.out))
    dev, test = _prev["dev_worlds"], _prev["test_worlds"]
    A.n_dev, A.n_test = len(dev), len(test)
    print(f"recomputing the summaries of {A.out} from its {A.n_dev} development and {A.n_test} test worlds", flush=True)
else:
    print(f"developing on {A.n_dev} worlds, assessing on {A.n_test} worlds; {A.P} patients x {A.R} reactions each; "
          f"boosting rounds selected among {ROUNDS} on {A.inner_frac:.0%} of training reactions withheld from the loss", flush=True)
    dev = run_worlds(A.n_dev, 1, "dev"); test = run_worlds(A.n_test, 100000, "test")


# ---- the decision rule, fixed on the development worlds ----------------------------------------
def verdict(world, kind="learned"):
    a = world["audit"][kind]
    import math
    if any(math.isnan(a[k]) for k in ("own_minus_mean", "own_minus_donor", "signflip_mean_p", "signflip_donor_p")): return False
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


def null_subsets(worlds, kind="learned"):
    """False positives on each kind of null world, with the structural fact about the donor contrast."""
    out = {}
    for nk, label in (("shared_label", "shared label: the donor contrast is identically zero for a fixed predictor and "
                                       "zero up to the arithmetic of a refit for the learned one, so only the "
                                       "cohort-mean contrast can fire the rule"),
                      ("varying_label_no_signal", "varying label with alpha = 0: both contrasts are free to move, the "
                                                  "informative null for the donor contrast")):
        ws = [w for w in worlds if w["null_kind"] == nk]
        fp = sum(verdict(w, kind) for w in ws)
        don_zero = sum(1 for w in ws if w["audit"][kind]["own_minus_donor_zero"] == w["audit"][kind]["n_cells"])
        max_abs_don = max((abs(w["audit"][kind]["own_minus_donor"]) for w in ws), default=float("nan"))
        out[nk] = dict(n=len(ws), false_positives=int(fp), what=label,
                       worlds_with_all_donor_contrasts_zero=int(don_zero),
                       max_abs_donor_contrast=float(max_abs_don),
                       mean_contrast_positive_sig=int(sum(1 for w in ws if w["audit"][kind]["own_minus_mean"] > 0
                                                          and w["audit"][kind]["signflip_mean_p"] < 0.05)))
    return out


def toy_counterexample(n_pairs=200, seed=0):
    P = 2; R = 2 * n_pairs
    X = np.zeros((P, R)); y = np.zeros(R, int)
    for k in range(n_pairs):
        X[0, 2 * k] = 0.0; X[1, 2 * k] = 2.0
        X[:, 2 * k + 1] = 1.0
        y[2 * k] = 1
    own = (X - 1.0) ** 2
    mean_col = np.tile(X.mean(0), (P, 1)); mean_score = (mean_col - 1.0) ** 2
    own_auc = np.mean([roc_auc_score(y, own[i]) for i in range(P)])
    mean_auc = np.mean([roc_auc_score(y, mean_score[i]) for i in range(P)])
    return dict(n_pairs=n_pairs, own_auroc=round(float(own_auc), 4), cohort_mean_auroc=round(float(mean_auc), 4),
                own_minus_mean=round(float(own_auc - mean_auc), 4),
                note="shared label; own (x-1)^2 ranks every patient's pair correctly (AUROC 1) while the cohort-mean "
                     "input is constant at 1 for both reactions (AUROC 0.5), so own-minus-mean is +0.5")


TOY = toy_counterexample()


def decision_table(worlds):
    regimes = [("shared label, alpha=0 (no sample signal)", lambda w: w["label"] == "shared"),
               ("varying label, alpha=0 (threshold on own column, weak)", lambda w: w["label"] == "varying" and w["alpha"] == 0.0),
               ("varying label, alpha=0.5 (weak sample signal)", lambda w: w["label"] == "varying" and 0.0 < w["alpha"] < 1.5),
               ("varying label, alpha>=1.5 (strong sample signal)", lambda w: w["label"] == "varying" and w["alpha"] >= 1.5)]
    rows = []
    for name, sel in regimes:
        ws = [w for w in worlds if sel(w)]
        row = dict(regime=name, n_worlds=len(ws))
        for kind in KINDS:
            om = np.mean([w["audit"][kind]["own_minus_mean"] for w in ws]); od = np.mean([w["audit"][kind]["own_minus_donor"] for w in ws])
            vr = np.mean([verdict(w, kind) for w in ws])
            row[kind] = dict(own_minus_mean=round(float(om), 4), own_minus_donor=round(float(od), 4), verdict_rate=round(float(vr), 4))
        rows.append(row)
    return rows


restr = [w["audit"]["restricted_nonlinear"]["own_minus_mean"] for w in test if w["label"] == "shared"]
blind_shared = [w["audit"]["blind"]["own_minus_mean"] for w in test if w["label"] == "shared"]
OUT = dict(
    note="calibration of the input-substitution audit on synthetic worlds: every arm is an input substitution with "
         "the predictor refitted on the substituted inputs, the learned model's boosting rounds are selected on "
         "reactions withheld from the loss, and the contrasts are tested with the exact two-sided sign-flip test. "
         "The decision rule is applied on the development worlds and its behavior measured on the disjoint test "
         "worlds; every summary below says which set it was computed on.",
    params=(_prev["params"] if A.retable else
            dict(n_dev=A.n_dev, n_test=A.n_test, P=A.P, R=A.R, pfolds=A.pfolds, rfolds=A.rfolds,
                 inner_frac=A.inner_frac, rounds=ROUNDS, grid=GRID,
                 dev_seeds=[1, A.n_dev], test_seeds=[100000, 100000 + A.n_test - 1])),
    rule="learned model: own>mean and own>donor, each exact two-sided sign-flip p<0.05 over the cells",
    arms="own: each patient's own column; mean: every patient given the training-cohort mean column, predictor "
         "refitted on those inputs; donor: every patient given another patient's real column (a derangement among "
         "training patients and one among held-out patients), predictor refitted on those inputs",
    dev_confusion={k: confusion(dev, k) for k in KINDS},
    test_confusion={k: confusion(test, k) for k in KINDS},
    test_null_subsets={k: null_subsets(test, k) for k in ("learned", "oracle", "label_irrelevant")},
    dev_null_subsets={k: null_subsets(dev, k) for k in ("learned",)},
    counterexample=dict(
        constructed=TOY,
        note="the constructed case shows a shared target does not force own-minus-mean below zero; in the noisy "
             "generic worlds the generic (x-1)^2 scorer is not aligned with the generic label, so its contrast sits near zero",
        generic_restricted_nonlinear_own_minus_mean_mean_test=round(float(np.mean(restr)), 4),
        generic_blind_own_minus_mean_mean_test=round(float(np.mean(blind_shared)), 4)),
    decision_table=decision_table(test),
    dev_decision_table=decision_table(dev))
OUT["dev_worlds"] = dev; OUT["test_worlds"] = test
json.dump(OUT, open(A.out, "w"), indent=1, default=_j)
print("\ntest confusion (learned):", OUT["test_confusion"]["learned"])
print("test null subsets (learned):", json.dumps(OUT["test_null_subsets"]["learned"], indent=1))
print("counterexample (constructed):", TOY)
for row in OUT["decision_table"]:
    print(row["regime"], "| learned own-mean %.3f own-donor %.3f verdict %.2f | oracle verdict %.2f | blind own-mean %.3f"
          % (row["learned"]["own_minus_mean"], row["learned"]["own_minus_donor"], row["learned"]["verdict_rate"],
             row["oracle"]["verdict_rate"], row["blind"]["own_minus_mean"]))
print("wrote", A.out)
