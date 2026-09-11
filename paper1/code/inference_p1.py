#!/usr/bin/env python3
"""Inference for the paired comparisons, on the unit the design actually randomizes.

Three sources of variation sit in these experiments and they are not interchangeable.

  Between patients, inside one trained model. In the substituted arms every patient is
  shown an identical input, so their scores differ only by Monte Carlo dropout. Patients
  there are repeats of one prediction, not independent draws, and a test over them would
  be a test of dropout noise. We report their spread as a precision statistic and never
  as evidence about an arm.

  Between reaction folds and between patient folds. A cell is one retraining on one
  training set evaluated on one held-out reaction set. That is the unit a fresh run of
  this protocol would resample, so it is the unit we test on.

With fifteen cells at most, and three in the smallest arm, the parametric options are weak
and most of them assume more than the design supports. We therefore report an exact
sign-flip randomization test, which
enumerates every assignment of signs to the observed per-cell differences and needs no
distributional assumption, alongside a two-way decomposition that says how much of the
variation sits between patient folds, between reaction folds, and in the residual.

The randomization test has a floor: with n cells the smallest attainable two-sided p is
2 / 2^n. For an arm with three cells that floor is 0.25, so a large effect there cannot
be certified by this test however large it is, and we say so rather than reaching for a
test whose assumptions the design does not meet.

Cells that share a patient fold or a reaction fold are not exchangeable with one another
under the null in the way independent draws would be, and the two-way decomposition
below is exactly the measure of how far they depart from it. So every comparison is
also tested at the block level: signs flipped for a whole patient fold at a time, for a
whole reaction fold at a time, and a one-sample t-test on the fold means, whose n is
the number of folds rather than the number of cells. Those are the conservative
readings; the cell-level exact test is reported beside them, not instead of them.
"""
import os, re, json, math, itertools, statistics as st

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")
PP = json.load(open(os.path.join(RES, "per_patient.json")))
OUT = os.path.join(RES, "inference.json")


def cell_means(mode):
    return {c: st.mean(v[mode]["auroc"]) for c, v in PP.items() if mode in v}


def within_sd(mode):
    return {c: st.stdev(v[mode]["auroc"]) for c, v in PP.items() if mode in v}


def signflip(d):
    """Exact two-sided randomization test: enumerate all 2^n sign assignments."""
    n = len(d)
    obs = abs(st.mean(d))
    hits = sum(1 for s in itertools.product((-1, 1), repeat=n)
               if abs(st.mean([si * di for si, di in zip(s, d)])) >= obs - 1e-15)
    return hits / 2 ** n, 2.0 / 2 ** n


def block_signflip(cells, d, which):
    """Exact sign-flip test with signs assigned per block (patient fold or reaction fold)."""
    lab = [c.split("r")[0] if which == "p" else "r" + c.split("r")[1] for c in cells]
    blocks = sorted(set(lab)); nb = len(blocks)
    obs = abs(st.mean(d))
    hits = 0
    for s in itertools.product((-1, 1), repeat=nb):
        sign = dict(zip(blocks, s))
        if abs(st.mean([sign[l] * x for l, x in zip(lab, d)])) >= obs - 1e-15:
            hits += 1
    return hits / 2 ** nb, 2.0 / 2 ** nb, nb


def fold_means_t(cells, d, which):
    """One-sample t-test on the per-fold means of the differences, n = number of folds."""
    from scipy import stats
    lab = [c.split("r")[0] if which == "p" else "r" + c.split("r")[1] for c in cells]
    groups = {}
    for l, x in zip(lab, d):
        groups.setdefault(l, []).append(x)
    means = [st.mean(v) for _, v in sorted(groups.items())]
    if len(means) < 2 or st.pstdev(means) == 0:
        return dict(n=len(means), means=means, t=None, p=None)
    tt = stats.ttest_1samp(means, 0.0)
    return dict(n=len(means), means=means, mean=st.mean(means), sd=st.stdev(means),
                t=float(tt.statistic), p=float(tt.pvalue),
                all_same_sign=all(m < 0 for m in means) or all(m > 0 for m in means))


def _ols_ss(cells, d, factors):
    """Residual sum of squares of an additive least-squares fit on the named factors."""
    import numpy as np
    labs = [(c.split("r")[0], "r" + c.split("r")[1]) for c in cells]
    cols = [np.ones(len(d))]
    for fi, use in enumerate(factors):
        if not use:
            continue
        lv = sorted({l[fi] for l in labs})[1:]          # drop one level, keep full rank
        for level in lv:
            cols.append(np.array([1.0 if l[fi] == level else 0.0 for l in labs]))
    X = np.column_stack(cols); y = np.asarray(d, float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ beta
    return float(r @ r)


def twoway(cells, d):
    """Decompose the per-cell differences by patient fold and reaction fold.

    The substitution cells are not always a complete crossing, so marginal sums of
    squares are non-orthogonal and would double-count. Each factor's share is therefore
    the reduction in residual sum of squares from adding it to a model that already
    contains the other one, which is what a reader should be told when the design is
    unbalanced.
    """
    labs = [(c.split("r")[0], "r" + c.split("r")[1]) for c in cells]
    npf = len({l[0] for l in labs}); nrf = len({l[1] for l in labs})
    gm = st.mean(d)
    ss_t = sum((x - gm) ** 2 for x in d)
    if not ss_t or npf < 2 or nrf < 2:
        return dict(n_patient_folds=npf, n_reaction_folds=nrf, balanced=None,
                    frac_patient_fold=None, frac_reaction_fold=None, frac_residual=None)
    rss_both = _ols_ss(cells, d, (True, True))
    rss_no_p = _ols_ss(cells, d, (False, True))
    rss_no_r = _ols_ss(cells, d, (True, False))
    balanced = len({tuple(sorted(l[1] for l in labs if l[0] == p))
                    for p in {l[0] for l in labs}}) == 1
    return dict(n_patient_folds=npf, n_reaction_folds=nrf, balanced=balanced,
                frac_patient_fold=(rss_no_p - rss_both) / ss_t,
                frac_reaction_fold=(rss_no_r - rss_both) / ss_t,
                frac_residual=rss_both / ss_t)


def repro(mode):
    """Does a mode's per-patient AUROC ordering survive an independent retraining?

    If the between-patient spread inside a cell were patient signal it would reproduce
    across cells trained on the same patients. If it is run noise it will not.
    """
    import numpy as np
    cells = sorted(c for c in PP if mode in PP[c])
    rs = []
    for i in range(len(cells)):
        for j in range(i + 1, len(cells)):
            ci, cj = cells[i], cells[j]
            if ci.split("r")[0] != cj.split("r")[0]:
                continue                              # same patient fold only
            pi = PP[ci][mode]["patient_ids"]; pj = PP[cj][mode]["patient_ids"]
            if pi != pj:
                continue
            rs.append(float(np.corrcoef(PP[ci][mode]["auroc"], PP[cj][mode]["auroc"])[0, 1]))
    return dict(n_pairs=len(rs), r_mean=(st.mean(rs) if rs else None),
                r_min=(min(rs) if rs else None), r_max=(max(rs) if rs else None))


def compare(a, b, label, basis=None):
    A, B = cell_means(a), cell_means(b)
    cells = sorted(set(A) & set(B) & (set(basis) if basis else set(A)))
    if not cells:
        return None
    d = [A[c] - B[c] for c in cells]
    return compare_d(cells, d, label, arms=[a, b], within=(within_sd(a), within_sd(b)))


def compare_d(cells, d, label, arms=None, within=None):
    p, floor = signflip(d)
    m, s = st.mean(d), (st.stdev(d) if len(d) > 1 else None)
    bp, bpf, nbp = block_signflip(cells, d, "p")
    br, brf, nbr = block_signflip(cells, d, "r")
    out = dict(
        label=label, arms=arms, cells=cells, n=len(cells),
        per_cell=dict(zip(cells, [round(x, 6) for x in d])),
        mean=m, sd=s,
        cohens_dz=(m / s if s else None),   # signed, on purpose
        signflip_p=p, signflip_floor=floor,
        signflip_at_floor=(p <= floor + 1e-12), signflip_certifiable=(p < 0.05),
        all_same_sign=all(x < 0 for x in d) or all(x > 0 for x in d),
        n_negative=sum(1 for x in d if x < 0), n_positive=sum(1 for x in d if x > 0),
        block_patient=dict(p=bp, floor=bpf, n_blocks=nbp),
        block_reaction=dict(p=br, floor=brf, n_blocks=nbr),
        t_patient_folds=fold_means_t(cells, d, "p"),
        t_reaction_folds=fold_means_t(cells, d, "r"),
        variance=twoway(cells, d))
    if within:
        wa, wb = within
        out["within_cell_sd"] = dict(a=st.mean([wa[c] for c in cells]), b=st.mean([wb[c] for c in cells]))
        out["ratio_between_to_within"] = (s / st.mean([wa[c] for c in cells]) if s else None)
    return out


R = {"reproducibility": {m: repro(m) for m in ("real", "mean", "zero")},
     "_note": "Cells are the inference unit. Patients inside a substituted cell receive "
              "an identical input and differ only by Monte Carlo dropout.",
     "comparisons": {}}
# The three substitution comparisons share one basis: the cells where all three arms
# have run, so that the steps of the ladder telescope on identical cells.
BASIS = sorted(set(cell_means("real")) & set(cell_means("mean")) & set(cell_means("zero")))
R["basis"] = BASIS
for a, b, lab, basis in [("real", "mean", "patient identity: own expression minus cohort mean", BASIS),
                         ("mean", "zero", "aggregate expression: cohort mean minus zeroed", BASIS),
                         ("real", "zero", "expression in total: own minus zeroed", BASIS),
                         ("real", "rewire7", "network identity: real minus degree-preserving rewiring", None),
                         ("real", "rewire11", "network identity, second permutation seed: real minus rewiring 11", None),
                         ("rewire7", "rewire11", "two rewirings against each other: seed 7 minus seed 11", None),
                         ("indicator", "zero", "presence pattern: indicator minus zeroed", None),
                         ("mean", "indicator", "magnitudes: cohort mean minus indicator", None),
                         ("real", "indicator", "own expression minus indicator", None)]:
    c = compare(a, b, lab, basis)
    if c:
        R["comparisons"][f"{a}_vs_{b}"] = c

# the collapse comparison lives in collapse.json and has its own cell basis
col = os.path.join(RES, "collapse.json")
if os.path.exists(col):
    K = json.load(open(col))["cells"]
    # Unsubstituted cells only: <model>_mb0.0_p<i>_r<j> with no feature-mode suffix.
    CELL = re.compile(r"^(?P<m>.+?)_mb0\.0_p(?P<p>\d+)_r(?P<r>\d+)$")
    for model in ("mlp", "mlp_emb", "gnn_B"):
        d, cells = [], []
        for tag, v in sorted(K.items()):
            m = CELL.match(tag)
            if m and m.group("m") == model:
                d.append(v["cohmean"] - v["perpat"])
                cells.append("p%sr%s" % (m.group("p"), m.group("r")))
        if len(d) > 1:
            R["comparisons"][f"collapse_{model}"] = compare_d(
                cells, d, f"cohort-mean collapse, {model}", arms=["cohmean", "perpat"])

# the noise-corrected collapse: what remains of the collapse gain after the part that
# averaging Monte Carlo dropout noise alone would produce is removed (collapse_noise.py)
cn = os.path.join(RES, "collapse_noise.json")
if os.path.exists(cn):
    KN = json.load(open(cn))["cells"]
    CELL = re.compile(r"^(?P<m>.+?)_mb0\.0_p(?P<p>\d+)_r(?P<r>\d+)$")
    for model in ("mlp", "mlp_emb", "gnn_B"):
        d, cells = [], []
        for tag, v in sorted(KN.items()):
            m = CELL.match(tag)
            if m and m.group("m") == model:
                d.append(v["identity_component"]); cells.append("p%sr%s" % (m.group("p"), m.group("r")))
        if len(d) > 1:
            R["comparisons"][f"collapse_identity_{model}"] = compare_d(
                cells, d, f"cohort-mean collapse net of noise averaging, {model}",
                arms=["identity component", "zero"])

# the linear reference points, paired cell by cell with the graph arms on the same basis
na = os.path.join(RES, "naive_allfolds.json")
if os.path.exists(na):
    NA = json.load(open(na))
    for regime in ("fixed", "tuned"):
        G = NA.get(regime, {})
        def lin(k):
            return {c: v for c, v in G.get(k, {}).items()}
        pairs = [("structure_only", "expr_cohortmean_lr", "structure over fitted expression, linear"),
                 ("topology_only", "expr_cohortmean_lr", "topology alone over fitted expression, linear"),
                 ("annotation_only", "expr_cohortmean_lr", "annotation alone over fitted expression, linear"),
                 ("structure_plus_perpat", "structure_plus_cohortmean", "patient identity, linear with structure"),
                 ("expr_perpat_lr", "expr_cohortmean_lr", "patient identity, linear expression only"),
                 ("structure_plus_cohortmean", "structure_only", "aggregate expression, linear")]
        for a, b, lab in pairs:
            A, B = lin(a), lin(b)
            cells = sorted(set(A) & set(B) & set(BASIS))
            if len(cells) > 1:
                R["comparisons"][f"lin_{regime}_{a}_vs_{b}"] = compare_d(
                    cells, [A[c] - B[c] for c in cells], f"{lab} [{regime} C]", arms=[a, b])
        # graph arm minus its matched linear counterpart, paired on the basis cells
        gm = cell_means("mean"); gz = cell_means("zero"); gr = cell_means("real")
        for garm, gvals, lk, lab in [("zero", gz, "structure_only", "graph model over linear, no expression"),
                                     ("mean", gm, "structure_plus_cohortmean", "graph model over linear, aggregate expression"),
                                     ("real", gr, "structure_plus_perpat", "graph model over linear, own expression"),
                                     ("mean", gm, "structure_only", "graph model with cohort mean over the no-patient linear model"),
                                     ("real", gr, "structure_only", "graph model with own expression over the no-patient linear model")]:
            Lk = lin(lk)
            cells = sorted(set(gvals) & set(Lk) & set(BASIS))
            if len(cells) > 1:
                R["comparisons"][f"gnn_{garm}_vs_lin_{regime}_{lk}"] = compare_d(
                    cells, [gvals[c] - Lk[c] for c in cells], f"{lab} [{regime} C]", arms=[f"gnn {garm}", lk])

# the pooled, frozen linear rows (pooled_baselines.py): one model per cell fitted on the stacked
# rows of training patients and frozen, against the per-patient rows, the cohort-mean rows and the
# graph model's own-expression arm, paired on the basis cells
npd = os.path.join(RES, "naive_pooled.json")
if os.path.exists(npd) and os.path.exists(na):
    NP_ = json.load(open(npd)); NA = json.load(open(na))
    for regime in ("fixed", "tuned"):
        P_ = NP_.get(regime, {}); G = NA.get(regime, {})
        pairs = [("expr_pooled_lr", "expr_perpat_lr", "pooled frozen expression model over the per-patient one"),
                 ("expr_pooled_lr", "expr_cohortmean_lr", "pooled frozen expression model over the cohort-mean one"),
                 ("structure_plus_pooled", "structure_plus_perpat", "pooled frozen network + expression model over the per-patient one"),
                 ("structure_plus_pooled", "structure_plus_cohortmean", "pooled frozen network + expression model over the cohort-mean one")]
        for a, b, lab in pairs:
            A, B = P_.get(a, {}), G.get(b, {})
            cells = sorted(set(A) & set(B) & set(BASIS))
            if len(cells) > 1:
                R["comparisons"][f"lin_{regime}_{a}_vs_{b}"] = compare_d(
                    cells, [A[c] - B[c] for c in cells], f"{lab} [{regime} C]", arms=[a, b])
        gr = cell_means("real")
        for garm, gvals, lk, lab in [("real", gr, "structure_plus_pooled", "graph model over the pooled frozen linear model, own expression")]:
            Lk = P_.get(lk, {})
            cells = sorted(set(gvals) & set(Lk) & set(BASIS))
            if len(cells) > 1:
                R["comparisons"][f"gnn_{garm}_vs_lin_{regime}_{lk}"] = compare_d(
                    cells, [gvals[c] - Lk[c] for c in cells], f"{lab} [{regime} C]", arms=[f"gnn {garm}", lk])

# the graph arms under inner-validation-reaction early stopping (results/ivr/): the contrast under
# the selection rule that asks the test question, the same arms under the main rule on the same
# cells, and each ivr arm against its main-rule counterpart and against the linear rows
import glob as _glob
def cells_from_dir(sub, mode, suffix):
    """Per-cell held-out AUROC of one arm from the cell result files of a results subdirectory."""
    out = {}
    pat = re.compile(r"^gnn_B_mb0\.0_p(?P<p>\d+)_r(?P<r>\d+)" + ("" if mode == "real" else "_" + mode) + suffix + r"\.json$")
    for f in _glob.glob(os.path.join(RES, sub, "*.json")):
        m = pat.match(os.path.basename(f))
        if m:
            out["p%sr%s" % (m.group("p"), m.group("r"))] = json.load(open(f))["heldout_AUROC_perpatient"]
    return out
vr, vm = cells_from_dir("ivr", "real", "_ivr"), cells_from_dir("ivr", "mean", "_ivr")
if vr and vm:
    cells = sorted(set(vr) & set(vm))
    if len(cells) > 1:
        R["comparisons"]["ivr_real_vs_mean"] = compare_d(cells, [vr[c] - vm[c] for c in cells],
            "patient identity under inner-validation stopping: own expression minus cohort mean", arms=["ivr real", "ivr mean"])
        gr, gm = cell_means("real"), cell_means("mean")
        cm = sorted(set(cells) & set(gr) & set(gm))
        if len(cm) > 1:
            R["comparisons"]["main_real_vs_mean_on_ivr_cells"] = compare_d(cm, [gr[c] - gm[c] for c in cm],
                "patient identity under the main selection rule, on the cells rerun with inner-validation stopping", arms=["real", "mean"])
            R["comparisons"]["ivr_real_vs_main_real"] = compare_d(cm, [vr[c] - gr[c] for c in cm],
                "own-expression arm: inner-validation stopping minus the main rule", arms=["ivr real", "real"])
            R["comparisons"]["ivr_mean_vs_main_mean"] = compare_d(cm, [vm[c] - gm[c] for c in cm],
                "cohort-mean arm: inner-validation stopping minus the main rule", arms=["ivr mean", "mean"])
        # the zeroed, presence-only and wrong-patient arms under the same rule, where they ran
        vz, vi, vp = cells_from_dir("ivr", "zero", "_ivr"), cells_from_dir("ivr", "indicator", "_ivr"), cells_from_dir("ivr", "permute", "_ivr")
        for key, a, b, lab, arms in [("ivr_mean_vs_zero", vm, vz, "aggregate expression under inner-validation stopping: cohort mean minus zeroed", ["ivr mean", "ivr zero"]),
                                     ("ivr_real_vs_zero", vr, vz, "expression in total under inner-validation stopping: own minus zeroed", ["ivr real", "ivr zero"]),
                                     ("ivr_indicator_vs_zero", vi, vz, "presence under inner-validation stopping: indicator minus zeroed", ["ivr indicator", "ivr zero"]),
                                     ("ivr_mean_vs_indicator", vm, vi, "magnitude under inner-validation stopping: cohort mean minus indicator", ["ivr mean", "ivr indicator"]),
                                     ("ivr_real_vs_permute", vr, vp, "wrong-patient check under inner-validation stopping: own minus permuted column", ["ivr real", "ivr permute"]),
                                     ("ivr_permute_vs_mean", vp, vm, "wrong-patient arm against the cohort mean under inner-validation stopping", ["ivr permute", "ivr mean"])]:
            cc = sorted(set(a) & set(b))
            if len(cc) > 1:
                R["comparisons"][key] = compare_d(cc, [a[c] - b[c] for c in cc], lab, arms=arms)
        if os.path.exists(na):
            NA = json.load(open(na)); G = NA.get("tuned", {})
            NP_ = json.load(open(npd)).get("tuned", {}) if os.path.exists(npd) else {}
            for garm, gvals, src, lk, lab in [("real", vr, G, "structure_plus_perpat", "ivr graph model over the per-patient linear model, own expression"),
                                              ("real", vr, NP_, "structure_plus_pooled", "ivr graph model over the pooled frozen linear model, own expression"),
                                              ("mean", vm, G, "structure_plus_cohortmean", "ivr graph model over the linear model, cohort mean"),
                                              ("real", vr, G, "structure_only", "ivr graph model with own expression over the no-patient linear model"),
                                              ("mean", vm, G, "structure_only", "ivr graph model with cohort mean over the no-patient linear model")]:
                Lk = src.get(lk, {})
                cl = sorted(set(gvals) & set(Lk))
                if len(cl) > 1:
                    R["comparisons"][f"ivr_{garm}_vs_lin_tuned_{lk}"] = compare_d(cl, [gvals[c] - Lk[c] for c in cl], lab + " [tuned C]", arms=[f"ivr {garm}", lk])
vz = globals().get("vz") or {}; vi = globals().get("vi") or {}; vp = globals().get("vp") or {}
# the family-disjoint reaction split (results/fam/): the same arms refitted with biochemically
# related reactions withheld together. The primary own-minus-cohort-mean contrast is recomputed here,
# every arm is priced against the random-split arm of the same name, and the difference of the two
# contrasts says what the random split was worth on the question the paper asks.
fr, fm = cells_from_dir("fam", "real", "_ivr_fam"), cells_from_dir("fam", "mean", "_ivr_fam")
if fr and fm:
    cells = sorted(set(fr) & set(fm))
    if len(cells) > 1:
        R["comparisons"]["fam_real_vs_mean"] = compare_d(cells, [fr[c] - fm[c] for c in cells],
            "patient identity on the family-disjoint split: own expression minus cohort mean", arms=["fam real", "fam mean"])
        fz = cells_from_dir("fam", "zero", "_ivr_fam"); fi = cells_from_dir("fam", "indicator", "_ivr_fam")
        fp = cells_from_dir("fam", "permute", "_ivr_fam")
        for key, a, b, lab, arms in [("fam_mean_vs_zero", fm, fz, "aggregate expression on the family-disjoint split: cohort mean minus zeroed", ["fam mean", "fam zero"]),
                                     ("fam_real_vs_zero", fr, fz, "expression in total on the family-disjoint split: own minus zeroed", ["fam real", "fam zero"]),
                                     ("fam_indicator_vs_zero", fi, fz, "presence on the family-disjoint split: indicator minus zeroed", ["fam indicator", "fam zero"]),
                                     ("fam_mean_vs_indicator", fm, fi, "magnitude on the family-disjoint split: cohort mean minus indicator", ["fam mean", "fam indicator"]),
                                     ("fam_real_vs_permute", fr, fp, "wrong-patient check on the family-disjoint split: own minus permuted column", ["fam real", "fam permute"]),
                                     ("fam_permute_vs_mean", fp, fm, "wrong-patient arm against the cohort mean on the family-disjoint split", ["fam permute", "fam mean"])]:
            cc = sorted(set(a) & set(b))
            if len(cc) > 1:
                R["comparisons"][key] = compare_d(cc, [a[c] - b[c] for c in cc], lab, arms=arms)
        # what the random split was worth: each arm, and then the contrast of the two contrasts
        for key, a, b, lab, arms in [("fam_real_vs_rand_real", fr, vr, "own-expression arm: family-disjoint split minus random split", ["fam real", "ivr real"]),
                                     ("fam_mean_vs_rand_mean", fm, vm, "cohort-mean arm: family-disjoint split minus random split", ["fam mean", "ivr mean"]),
                                     ("fam_zero_vs_rand_zero", fz, vz, "zeroed arm: family-disjoint split minus random split", ["fam zero", "ivr zero"]),
                                     ("fam_indicator_vs_rand_indicator", fi, vi, "presence-only arm: family-disjoint split minus random split", ["fam indicator", "ivr indicator"]),
                                     ("fam_permute_vs_rand_permute", fp, vp, "wrong-patient arm: family-disjoint split minus random split", ["fam permute", "ivr permute"])]:
            cc = sorted(set(a) & set(b))
            if len(cc) > 1:
                R["comparisons"][key] = compare_d(cc, [a[c] - b[c] for c in cc], lab, arms=arms)
        cc = sorted(set(fr) & set(fm) & set(vr) & set(vm))
        if len(cc) > 1:
            R["comparisons"]["fam_pat_minus_rand_pat"] = compare_d(
                cc, [(fr[c] - fm[c]) - (vr[c] - vm[c]) for c in cc],
                "patient identity: the family-disjoint contrast minus the random-split contrast on the same cells",
                arms=["fam real minus mean", "ivr real minus mean"])

# the nearest-family label lookup on each split: copying the label of the most similar training
# reaction, which is exactly what a near-duplicate lets a model do
narf = os.path.join(RES, "naive_allfolds_rerun.json")
naff = os.path.join(RES, "naive_allfolds_fam.json")
if os.path.exists(narf) and os.path.exists(naff):
    def _co(b): return b.get("cells") if isinstance(b, dict) and "cells" in b else (b if isinstance(b, dict) else None)
    _lr = _co((json.load(open(narf)).get("tuned") or {}).get("nearest_lookup_k5") or {})
    _lf = _co((json.load(open(naff)).get("tuned") or {}).get("nearest_lookup_k5") or {})
    if _lr and _lf:
        cc = sorted(set(_lr) & set(_lf))
        if len(cc) > 1:
            R["comparisons"]["lookup_fam_vs_rand"] = compare_d(
                cc, [_lf[c] - _lr[c] for c in cc],
                "nearest-family label lookup: family-disjoint split minus random split",
                arms=["lookup (family folds)", "lookup (random folds)"])

# the linear reference rows refitted on the family-disjoint folds: the no-patient floor against the
# graph model on each split, and the change in that gap, which is the quantity that says how much of
# the graph model's standing on the random split rested on reactions with a near-twin in training
naf = os.path.join(RES, "naive_allfolds_fam.json")
if fr and fm and os.path.exists(naf) and os.path.exists(na):
    _NAF = json.load(open(naf)).get("tuned", {}); _NAR = json.load(open(na)).get("tuned", {})
    def _cells_of(b): return b.get("cells") if isinstance(b, dict) and "cells" in b else (b if isinstance(b, dict) else None)
    _sof, _sor = _cells_of(_NAF.get("structure_only") or {}), _cells_of(_NAR.get("structure_only") or {})
    if _sof:
        cc = sorted(set(fr) & set(_sof))
        if len(cc) > 1:
            R["comparisons"]["fam_gnn_real_vs_lin_structure_only"] = compare_d(
                cc, [fr[c] - _sof[c] for c in cc],
                "graph model with own expression over the no-patient linear model, family-disjoint split",
                arms=["fam real", "structure_only (family folds)"])
        for key, a, lab in (("fam_gnn_mean_vs_lin_structure_only", fm,
                             "graph model with the cohort mean over the no-patient linear model, family-disjoint split"),):
            cc = sorted(set(a) & set(_sof))
            if len(cc) > 1:
                R["comparisons"][key] = compare_d(cc, [a[c] - _sof[c] for c in cc], lab,
                                                  arms=["fam mean", "structure_only (family folds)"])
        if _sor:
            cc = sorted(set(_sof) & set(_sor))
            if len(cc) > 1:
                R["comparisons"]["fam_lin_structure_only_vs_rand"] = compare_d(
                    cc, [_sof[c] - _sor[c] for c in cc],
                    "the no-patient linear model: family-disjoint split minus random split",
                    arms=["structure_only (family folds)", "structure_only (random folds)"])
            cc = sorted(set(fr) & set(vr) & set(_sof) & set(_sor))
            if len(cc) > 1:
                R["comparisons"]["fam_gnn_lin_gap_change"] = compare_d(
                    cc, [(fr[c] - _sof[c]) - (vr[c] - _sor[c]) for c in cc],
                    "the graph model's standing against the no-patient linear model: the family-disjoint gap minus the random-split gap",
                    arms=["fam gnn minus structure_only", "random gnn minus structure_only"])

# the synthetic patient-varying positive control (results/synth/), on all held-out reactions and on
# the held-out reactions whose label varies between patients
for tagsuf, name in (("_synth1_ivr", "synth_full_ivr"), ("_synth0.5_ivr", "synth_half_ivr"), ("_synth1", "synth_full_main"), ("_synth0.5", "synth_half_main")):
    sr, sm = cells_from_dir("synth", "real", tagsuf), cells_from_dir("synth", "mean", tagsuf)
    cells = sorted(set(sr) & set(sm))
    if len(cells) > 1:
        R["comparisons"][name + "_real_vs_mean"] = compare_d(cells, [sr[c] - sm[c] for c in cells],
            f"synthetic patient-varying target ({name}): own expression minus cohort mean, all held-out reactions", arms=["real", "mean"])
        def _vary(mode):
            out = {}
            pat = re.compile(r"^gnn_B_mb0\.0_p(?P<p>\d+)_r(?P<r>\d+)" + ("" if mode == "real" else "_" + mode) + tagsuf + r"\.json$")
            for f in _glob.glob(os.path.join(RES, "synth", "*.json")):
                m = pat.match(os.path.basename(f))
                if m: out["p%sr%s" % (m.group("p"), m.group("r"))] = json.load(open(f)).get("heldout_vary_AUROC")
            return out
        vr_, vm_ = _vary("real"), _vary("mean")
        R["comparisons"][name + "_real_vs_mean_vary"] = compare_d(cells, [vr_[c] - vm_[c] for c in cells],
            f"synthetic patient-varying target ({name}): own expression minus cohort mean, varying held-out reactions", arms=["real", "mean"])

json.dump(R, open(OUT, "w"), indent=1)
print("%-58s %5s %10s %9s %8s %10s" % ("comparison", "n", "mean", "sd", "d_z", "signflip p"))
print("-" * 106)
for k, c in R["comparisons"].items():
    print("%-58s %5d %+10.4f %9.4f %8.1f %10.4f%s"
          % (c["label"][:58], c["n"], c["mean"], c["sd"] or 0, c["cohens_dz"] or 0,
             c["signflip_p"],
             "  (at the floor, %s)" % ("certifiable" if c["signflip_p"] < 0.05 else "NOT certifiable")
             if c.get("signflip_at_floor") else ""))
print("\nwrote", os.path.relpath(OUT, HERE))
