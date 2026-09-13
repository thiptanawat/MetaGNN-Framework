#!/usr/bin/env python3
"""Frozen statistics for the Language-Model Evidence manuscript.

Every number quoted in the manuscript is produced here and written to results_frozen.json,
so each claim in the text maps to one key in one file.

Design note. Within an arm, the `reaction` prompts are identical for every patient by
construction (the arm shows no patient data). The spread of the model's answers across
those repeated identical prompts is therefore a direct measurement of serving
nondeterminism, and it is the reference against which the patient-versus-stranger
comparison has to be read.
"""
import os, json, sys, numpy as np
from sklearn.metrics import roc_auc_score
from scipy import stats as st
import _paths as PATHS

D = np.load(PATHS.data('rxn_context.npz'), allow_pickle=True)
LAB, HAS, X, PIDS = D['labels'], D['has'], D['X'], D['pids']
# the cohort percentile printed in every patient prompt: the argsort rank within the cohort at each
# reaction, scaled to 0..100, exactly as the prompt builder computes it (no tie correction)
_order = np.argsort(X, axis=0); _ranks = np.empty_like(_order)
for _j in range(X.shape[1]): _ranks[_order[:, _j], _j] = np.arange(X.shape[0])
PCT = 100.0 * _ranks / (X.shape[0] - 1)

def load(out):
    raw = json.load(open(PATHS.run(out, 'raw.json'))); des = json.load(open(PATHS.run(out, 'design.json')))
    IDX = np.array(des['reactions']); PSEL = np.array(des['patients']); ARMS = des['arms']
    M = {}
    for arm in ARMS:
        A_ = np.full((len(PSEL), len(IDX)), np.nan)
        for a, pj in enumerate(PSEL):
            for b, ri in enumerate(IDX):
                v = raw.get(f'{arm}|{pj}|{ri}')
                if v and v['prob'] is not None: A_[a, b] = v['prob']
        M[arm] = A_
    return M, IDX, PSEL, ARMS, des

def auc_matrix(S, y):
    """Per-row AUROC of a score matrix against one label vector, vectorized through mid-ranks
    (the Mann-Whitney identity); rows with a non-finite entry fall back to sklearn on their finite cells."""
    S = np.asarray(S, float); y = np.asarray(y).astype(bool)
    out = np.full(S.shape[0], np.nan)
    npos = int(y.sum()); nneg = int((~y).sum())
    if npos == 0 or nneg == 0: return out
    fin = np.isfinite(S).all(axis=1)
    if fin.any():
        R = st.rankdata(S[fin], axis=1)
        out[fin] = (R[:, y].sum(axis=1) - npos * (npos + 1) / 2.0) / (npos * nneg)
    for i in np.where(~fin)[0]:
        ok = np.isfinite(S[i])
        out[i] = roc_auc_score(y[ok], S[i][ok]) if len(set(y[ok].tolist())) > 1 else np.nan
    return out

def boot_patients(fn, n_pat, B=2000, seed=7):
    """Patient-level bootstrap: patients are the independent unit."""
    rng = np.random.default_rng(seed)
    v = [fn(rng.integers(0, n_pat, n_pat)) for _ in range(B)]
    v = np.array([x for x in v if np.isfinite(x)])
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

def analyze(out, tag):
    M, IDX, PSEL, ARMS, des = load(out)
    y = LAB[IDX].astype(int); hx = HAS[IDX]
    Xs = X[np.ix_(PSEL, IDX)]
    PCTS = PCT[np.ix_(PSEL, IDX)]
    R = dict(tag=tag, out=out, n_reactions=len(IDX), n_expression_bearing=int(hx.sum()),
             n_patients=len(PSEL), arms=ARMS, minutes=des.get('minutes'),
             n_calls=int(len(ARMS) * len(PSEL) * len(IDX)),
             n_calls_note='cells in the design; runs interrupted by a restart were resumed, so the '
                          'per-invocation counts in design.json are smaller',
             # calls the serving endpoint answered with a transport error (HTTP 502/503) and that were
             # re-sent with the identical prompt once it returned; zero for a run with no outage
             resent_calls=int(des.get('resent_calls', 0) or 0),
             sampling=des.get('sampling'), label_base_rate=round(float(y.mean()), 4),
             model=des.get('model'))

    def auc_rows(S, mask=None):
        m = np.ones(len(y), bool) if mask is None else mask
        v = []
        for i in range(len(PSEL)):
            ok = m & np.isfinite(S[i])
            v.append(roc_auc_score(y[ok], S[i][ok]) if len(set(y[ok])) > 1 else np.nan)
        return np.array(v)

    R['raw_expression_auroc'] = round(float(np.mean(auc_rows(Xs))), 4)
    R['raw_expression_auroc_sd'] = round(float(np.std(auc_rows(Xs))), 4)
    R['raw_expression_auroc_expr'] = round(float(np.mean(auc_rows(Xs, hx))), 4)
    # the second number the prompt prints, the cohort percentile of the value, ranked with no model:
    # a within-reaction rank across patients carries none of the between-reaction magnitude that the
    # raw value carries, so it is a different baseline from the raw ranking (reactions shown no value
    # keep percentile zero, as in the prompt builder's arrays)
    Pz = PCTS.copy(); Pz[:, ~hx] = 0.0
    R['raw_percentile_auroc'] = round(float(np.mean(auc_rows(Pz))), 4)
    R['raw_percentile_auroc_expr'] = round(float(np.mean(auc_rows(Pz, hx))), 4)
    R['indicator_floor'] = round(float(roc_auc_score(y, hx.astype(float))), 4)
    tpr, fpr = hx[y == 1].mean(), hx[y == 0].mean()
    R['indicator_floor_closed_form'] = round(float(0.5 + 0.5 * (tpr - fpr)), 4)
    # the same references as the area under the precision-recall curve, for which a constant
    # predictor scores the prevalence (label_base_rate)
    from sklearn.metrics import average_precision_score as _aps
    def ap_rows(S, mask=None):
        m = np.ones(len(y), bool) if mask is None else mask
        v = []
        for i in range(len(PSEL)):
            ok = m & np.isfinite(S[i])
            v.append(_aps(y[ok], S[i][ok]) if len(set(y[ok])) > 1 else np.nan)
        return np.array(v)
    R['raw_expression_auprc'] = round(float(np.mean(ap_rows(Xs))), 4)
    R['indicator_auprc'] = round(float(_aps(y, hx.astype(float))), 4)
    R['constant_auprc'] = round(float(y.mean()), 4)

    # calls the endpoint answered with a transport error and that were re-sent (see llm_probe.py
    # --repair): their share per arm, and the arm's accuracy with those cells excluded, so that the
    # repair's largest possible effect is on record
    try:
        bad = set(json.load(open(PATHS.run(out, 'resent_keys.json')))['keys'])
    except FileNotFoundError:
        bad = set()
    R['resent'] = {}
    if bad:
        for a in ARMS:
            keys = [f'{a}|{pj}|{ri}' for pj in PSEL for ri in IDX]
            nb = sum(1 for k in keys if k in bad)
            if not nb: continue
            # which patients the re-sent cells belong to, and how the arm reads with and without them
            per_pat = {}
            for ia, pj in enumerate(PSEL):
                c = sum(1 for ri in IDX if f'{a}|{pj}|{ri}' in bad)
                if c: per_pat[int(pj)] = c
            v = []
            for i in range(len(PSEL)):
                ok = np.isfinite(M[a][i]); v.append(roc_auc_score(y[ok], M[a][i][ok]) if len(set(y[ok])) > 1 else np.nan)
            v = np.array(v); aff = np.array([int(pj) in per_pat for pj in PSEL])
            R['resent'][a] = dict(n=nb, share_of_arm=round(nb / len(keys), 4),
                                  n_patients_affected=int(aff.sum()), cells_per_affected_patient=per_pat,
                                  auroc_all=round(float(np.nanmean(v)), 4),
                                  auroc_unaffected=(round(float(np.nanmean(v[~aff])), 4) if (~aff).sum() else None),
                                  auroc_affected=(round(float(np.nanmean(v[aff])), 4) if aff.sum() else None))
    R['surviving_per_patient'] = {a: dict(min=int(np.isfinite(M[a]).sum(1).min()), max=int(np.isfinite(M[a]).sum(1).max()))
                                  for a in ARMS}
    R['per_arm'] = {}
    per_patient_auc = {}
    def median_r(S, mask):
        """Median off-diagonal correlation between patients' score vectors over the masked reactions."""
        Sm = S[:, mask]; full = np.isfinite(Sm).all(1)
        if full.sum() < 2: return float('nan')
        C = np.corrcoef(Sm[full]); return float(np.median(C[np.triu_indices(len(C), 1)]))
    for a in ARMS:
        S = M[a]; v = auc_rows(S); per_patient_auc[a] = v
        R['per_arm'][a] = dict(
            parsed=round(float(np.isfinite(S).mean()), 4),
            auroc=round(float(np.nanmean(v)), 4), auroc_sd=round(float(np.nanstd(v)), 4),
            auroc_ci=[round(x, 4) for x in boot_patients(lambda ix: np.nanmean(v[ix]), len(PSEL))],
            auroc_expr=round(float(np.nanmean(auc_rows(S, hx))), 4),
            auprc=round(float(np.nanmean(ap_rows(S))), 4),
            # agreement between patients is read on the reactions where their prompts can differ;
            # on the rest the prompts are byte-identical and agree by construction
            median_inter_patient_r=round(median_r(S, hx), 4),
            median_inter_patient_r_all=round(median_r(S, np.ones(len(IDX), bool)), 4),
            mean_p=round(float(np.nanmean(S)), 4),
            sd_p=round(float(np.nanstd(S)), 4),
            n_distinct_values=int(len(np.unique(S[np.isfinite(S)]))))

    # ---- test-retest floor: the reaction arm repeats one identical prompt across patients ----
    if 'reaction' in ARMS:
        Rn = M['reaction']; pr = []
        for i in range(len(PSEL)):
            for j in range(i + 1, len(PSEL)):
                m = np.isfinite(Rn[i]) & np.isfinite(Rn[j])
                pr.append((float((Rn[i][m] == Rn[j][m]).mean()), float(np.abs(Rn[i][m] - Rn[j][m]).mean())))
        n_cells_both = int(sum(int((np.isfinite(Rn[i]) & np.isfinite(Rn[j])).sum())
                              for i in range(len(PSEL)) for j in range(i + 1, len(PSEL))))
        pr = np.array(pr)
        # the same floor restricted to the expression-bearing reactions, the set on which the
        # movement statistics below and the inter-patient agreement are computed
        pe = []
        for i in range(len(PSEL)):
            for j in range(i + 1, len(PSEL)):
                m = np.isfinite(Rn[i]) & np.isfinite(Rn[j]) & hx
                pe.append((float((Rn[i][m] == Rn[j][m]).mean()), float(np.abs(Rn[i][m] - Rn[j][m]).mean())))
        pe = np.array(pe)
        R['determinism'] = dict(n_prompt_pairs=len(pr), n_cell_pairs=n_cells_both,
                                identical=round(float(pr[:, 0].mean()), 4),
                                mean_abs_diff=round(float(pr[:, 1].mean()), 6),
                                identical_expr=round(float(pe[:, 0].mean()), 4),
                                mean_abs_diff_expr=round(float(pe[:, 1].mean()), 6),
                                note='identical prompts, temperature 0, repeated across the patient axis; '
                                     'the _expr figures are on the expression-bearing reactions only')

    # ---- test-retest floor on the patient prompt itself, where the arms carry repeats of it ----
    reps = [a for a in ARMS if a.startswith('patient_r')]
    if 'patient' in ARMS and reps:
        P0 = M['patient']; pr = []; pa = []
        for a in reps:
            m = np.isfinite(P0) & np.isfinite(M[a]) & hx[None, :]
            pr.append((float((P0[m] == M[a][m]).mean()), float(np.abs(P0[m] - M[a][m]).mean()), int(m.sum())))
            ma = np.isfinite(P0) & np.isfinite(M[a])      # the same pair over all reactions
            pa.append((float((P0[ma] == M[a][ma]).mean()), float(np.abs(P0[ma] - M[a][ma]).mean()), int(ma.sum())))
        R['determinism_patient_prompt'] = dict(
            n_repeats=len(reps), n_pairs=int(sum(x[2] for x in pr)),
            identical=round(float(np.mean([x[0] for x in pr])), 4),
            mean_abs_diff=round(float(np.mean([x[1] for x in pr])), 6),
            n_pairs_all=int(sum(x[2] for x in pa)),
            identical_all=round(float(np.mean([x[0] for x in pa])), 4),
            mean_abs_diff_all=round(float(np.mean([x[1] for x in pa])), 6),
            n_distinct_values=int(len(np.unique(P0[np.isfinite(P0) & hx[None, :]]))),
            note='the patient prompt re-sent verbatim; identical and mean_abs_diff are on the expression-bearing '
                 'reactions only, the _all figures over all reactions')

    # ---- the decisive comparison ----
    if 'patient' in ARMS and 'shuffled' in ARMS:
        P, S = M['patient'], M['shuffled']
        ex = np.isfinite(P) & np.isfinite(S) & hx[None, :]
        d = np.abs(P[ex] - S[ex])
        rp = P - np.nanmean(P, axis=0, keepdims=True)
        rs = S - np.nanmean(S, axis=0, keepdims=True)
        m2 = np.isfinite(rp) & np.isfinite(rs) & hx[None, :]
        rres = float(np.corrcoef(rp[m2], rs[m2])[0, 1])

        def resid_r(ix):
            a, b = rp[ix][:, hx], rs[ix][:, hx]
            k = np.isfinite(a) & np.isfinite(b)
            return np.corrcoef(a[k], b[k])[0, 1] if k.sum() > 10 else np.nan
        lo, hi = boot_patients(resid_r, len(PSEL))

        dv = per_patient_auc['patient'] - per_patient_auc['shuffled']
        t, p = st.ttest_rel(per_patient_auc['patient'], per_patient_auc['shuffled'], nan_policy='omit')
        w = st.wilcoxon(per_patient_auc['patient'], per_patient_auc['shuffled']) if len(PSEL) > 5 else None
        # TOST equivalence on the paired AUROC difference, at the conventional 0.05 and at 0.02,
        # about a tenth of the distance between chance and the raw-expression ranking in the
        # representative design, which is the benchmark's whole usable range there
        se = np.nanstd(dv, ddof=1) / np.sqrt(np.isfinite(dv).sum()); dfree = np.isfinite(dv).sum() - 1
        def tost(marg):
            p_low = st.t.sf((np.nanmean(dv) + marg) / se, dfree)      # H0: diff <= -margin
            p_high = st.t.cdf((np.nanmean(dv) - marg) / se, dfree)    # H0: diff >= +margin
            return p_low, p_high
        marg = 0.05; p_low, p_high = tost(marg)
        p_low2, p_high2 = tost(0.02)
        ci90 = (float(np.nanmean(dv) - st.t.ppf(0.95, dfree) * se), float(np.nanmean(dv) + st.t.ppf(0.95, dfree) * se))

        # responsiveness: how far each arm moves away from the patient-blind arm
        mv = {}
        if 'reaction' in ARMS:
            Rn = M['reaction']
            for a in ('patient', 'shuffled'):
                mm = np.isfinite(M[a]) & np.isfinite(Rn) & hx[None, :]
                mv[a] = round(float(np.abs(M[a][mm] - Rn[mm]).mean()), 6)

        R['patient_vs_shuffled'] = dict(
            n_pairs=int(ex.sum()),
            identical=round(float((P[ex] == S[ex]).mean()), 4),
            mean_abs_diff=round(float(d.mean()), 4),
            arm_correlation=round(float(np.corrcoef(P[ex], S[ex])[0, 1]), 4),
            residual_correlation=round(rres, 4),
            residual_correlation_ci=[round(lo, 4), round(hi, 4)],
            between_patient_sd_real=round(float(np.nanstd(P[:, hx], axis=0).mean()), 4),
            between_patient_sd_stranger=round(float(np.nanstd(S[:, hx], axis=0).mean()), 4),
            auroc_diff_mean=round(float(np.nanmean(dv)), 4),
            auroc_diff_sd=round(float(np.nanstd(dv, ddof=1)), 4),
            paired_t=round(float(t), 3), paired_p=round(float(p), 4),
            wilcoxon_p=(round(float(w.pvalue), 4) if w is not None else None),
            tost_margin=marg, tost_p_lower=float(f'{p_low:.3g}'), tost_p_upper=float(f'{p_high:.3g}'),
            tost_equivalent=bool(max(p_low, p_high) < 0.05),
            tost_margin_narrow=0.02, tost_narrow_p_lower=float(f'{p_low2:.3g}'), tost_narrow_p_upper=float(f'{p_high2:.3g}'),
            tost_narrow_equivalent=bool(max(p_low2, p_high2) < 0.05),
            auroc_diff_ci90=[round(ci90[0], 4), round(ci90[1], 4)],
            mean_move_from_reaction_arm=mv)

        # each data-bearing arm against the patient-blind arm, per patient, paired: what adding a
        # number is worth, and what adding a stranger's number is worth
        R['arms_vs_blind'] = {}
        if 'reaction' in ARMS:
            for a in ('patient', 'shuffled'):
                dd = per_patient_auc[a] - per_patient_auc['reaction']
                ok = np.isfinite(dd)
                if ok.sum() > 2:
                    tt = st.ttest_rel(per_patient_auc[a][ok], per_patient_auc['reaction'][ok])
                    se_ = np.std(dd[ok], ddof=1) / np.sqrt(ok.sum()); q = st.t.ppf(0.95, ok.sum() - 1)
                    R['arms_vs_blind'][a] = dict(n=int(ok.sum()), mean_diff=round(float(dd[ok].mean()), 4),
                                                 sd=round(float(np.std(dd[ok], ddof=1)), 4),
                                                 paired_p=float(f'{tt.pvalue:.3g}'),
                                                 ci90=[round(float(dd[ok].mean() - q * se_), 4), round(float(dd[ok].mean() + q * se_), 4)])
        # the between-patient standard deviation of the scores, own against stranger's, with a
        # patient-level bootstrap interval on the difference
        def sd_diff(ix):
            with np.errstate(all='ignore'):
                return float(np.nanmean(np.nanstd(P[ix][:, hx], axis=0)) - np.nanmean(np.nanstd(S[ix][:, hx], axis=0)))
        try:
            sdlo, sdhi = boot_patients(sd_diff, len(PSEL))
            R['between_patient_sd_diff'] = dict(diff=round(sd_diff(np.arange(len(PSEL))), 4),
                                                ci95=[round(sdlo, 4), round(sdhi, 4)],
                                                note='own minus stranger, mean over expression-bearing reactions; patient bootstrap')
        except (IndexError, ValueError):
            R['between_patient_sd_diff'] = None      # too few finite cells (the reasoning run)
        # would a coarse answer grid alone explain the gap to the raw ranking? bin the raw
        # expression to as many quantile levels as the arm has distinct values and score the bins
        R['raw_expression_binned'] = {}
        for a in ('patient', 'shuffled'):
            k = R['per_arm'][a]['n_distinct_values']
            if k < 2: continue
            qs = np.quantile(Xs[:, hx], np.linspace(0, 1, k + 1)[1:-1])
            # bin codes start at 1: code 0 is what the reactions without a value carry
            Xb = Xs.copy(); Xb[:, hx] = np.digitize(Xs[:, hx], qs) + 1
            R['raw_expression_binned'][a] = dict(levels=int(k), auroc=round(float(np.mean(auc_rows(Xb))), 4))
        # a reaction-level bootstrap: the intervals above resample patients only, so this says how
        # the headline comparisons behave when the reaction sample is resampled instead
        rng_r = np.random.default_rng(23); B_r = 1000
        own_ge_ind = own_ge_raw = 0; rd = []; ra = {a: [] for a in ARMS}; rob = []; rsb = []
        for _ in range(B_r):
            jx = rng_r.integers(0, len(IDX), len(IDX))
            yb = y[jx]
            if len(set(yb)) < 2: continue
            hb = hx[jx]
            def auc_b(Sm):
                return auc_matrix(Sm[:, jx], yb)
            ab = {a: auc_b(M[a]) for a in ARMS}
            for a in ARMS: ra[a].append(float(np.nanmean(ab[a])))
            raw_b = float(np.nanmean(auc_b(Xs))); ind_b = float(roc_auc_score(yb, hb.astype(float)))
            own_ge_ind += float(np.nanmean(ab['patient'])) >= ind_b
            own_ge_raw += float(np.nanmean(ab['patient'])) >= raw_b
            rd.append(float(np.nanmean(ab['patient'] - ab['shuffled'])))
            if 'reaction' in ARMS:
                rob.append(float(np.nanmean(ab['patient'] - ab['reaction'])))
                rsb.append(float(np.nanmean(ab['shuffled'] - ab['reaction'])))
        rd = np.array(rd); rob = np.array(rob); rsb = np.array(rsb)
        if len(rd) > 10: R['reaction_bootstrap'] = dict(
            B=int(len(rd)), p_own_ge_indicator=round(own_ge_ind / len(rd), 4), p_own_ge_raw=round(own_ge_raw / len(rd), 4),
            own_minus_stranger_ci95=[round(float(np.percentile(rd, 2.5)), 4), round(float(np.percentile(rd, 97.5)), 4)],
            arm_ci95={a: [round(float(np.percentile(ra[a], 2.5)), 4), round(float(np.percentile(ra[a], 97.5)), 4)] for a in ARMS},
            own_minus_blind_ci95=([round(float(np.percentile(rob, 2.5)), 4), round(float(np.percentile(rob, 97.5)), 4)] if len(rob) else None),
            stranger_minus_blind_ci95=([round(float(np.percentile(rsb, 2.5)), 4), round(float(np.percentile(rsb, 97.5)), 4)] if len(rsb) else None),
            p_own_minus_blind_le0=(round(float((rob <= 0).mean()), 4) if len(rob) else None),
            p_stranger_minus_blind_le0=(round(float((rsb <= 0).mean()), 4) if len(rsb) else None),
            note='reactions resampled with replacement, patients fixed')
        # a crossed bootstrap: patients and reactions resampled together, arms kept paired within
        # every cell, as a sensitivity analysis on the paired differences; neither one-way bootstrap
        # above speaks to the joint uncertainty over both sampling axes
        rng_c = np.random.default_rng(29); B_c = 1000
        cd = []; cob = []; csb = []
        for _ in range(B_c):
            ix = rng_c.integers(0, len(PSEL), len(PSEL)); jx = rng_c.integers(0, len(IDX), len(IDX))
            yb = y[jx]
            if len(set(yb)) < 2: continue
            def auc_c(Sm):
                return auc_matrix(Sm[np.ix_(ix, jx)], yb)
            ac = {a: auc_c(M[a]) for a in ARMS}
            cd.append(float(np.nanmean(ac['patient'] - ac['shuffled'])))
            if 'reaction' in ARMS:
                cob.append(float(np.nanmean(ac['patient'] - ac['reaction'])))
                csb.append(float(np.nanmean(ac['shuffled'] - ac['reaction'])))
        cd = np.array(cd); cob = np.array(cob); csb = np.array(csb)
        if len(cd) > 10: R['crossed_bootstrap'] = dict(
            B=int(len(cd)),
            own_minus_stranger_ci90=[round(float(np.percentile(cd, 5)), 4), round(float(np.percentile(cd, 95)), 4)],
            own_minus_stranger_ci95=[round(float(np.percentile(cd, 2.5)), 4), round(float(np.percentile(cd, 97.5)), 4)],
            own_minus_stranger_within_002=bool(np.percentile(cd, 5) > -0.02 and np.percentile(cd, 95) < 0.02),
            own_minus_stranger_within_005=bool(np.percentile(cd, 5) > -0.05 and np.percentile(cd, 95) < 0.05),
            own_minus_blind_ci90=([round(float(np.percentile(cob, 5)), 4), round(float(np.percentile(cob, 95)), 4)] if len(cob) else None),
            stranger_minus_blind_ci90=([round(float(np.percentile(csb, 5)), 4), round(float(np.percentile(csb, 95)), 4)] if len(csb) else None),
            note='patients and reactions resampled with replacement together; arms paired within cells')

        # does the score track the number shown?
        m3 = np.isfinite(P) & hx[None, :]
        R['score_vs_shown_value'] = dict(
            pearson=round(float(np.corrcoef(P[m3], Xs[m3])[0, 1]), 4),
            spearman=round(float(st.spearmanr(P[m3], Xs[m3]).statistic), 4))

    # ---- presence versus magnitude: what part of the response does the number drive? ----
    if 'patient' in ARMS and 'reaction' in ARMS:
        P, Rn = M['patient'], M['reaction']
        m = np.isfinite(P) & np.isfinite(Rn)
        # (i) whole sample: how much of the patient-arm score is explained by the patient-blind
        #     score, by a presence indicator, and by the magnitude shown, nested in that order
        rows = []
        for i in range(len(PSEL)):
            for j in range(len(IDX)):
                if m[i, j]: rows.append((P[i, j], Rn[i, j], float(hx[j]), Xs[i, j], PCTS[i, j]))
        Ar = np.array(rows)
        yv = Ar[:, 0]
        def r2(cols):
            Z = np.column_stack([np.ones(len(Ar))] + [Ar[:, c] for c in cols])
            b, *_ = np.linalg.lstsq(Z, yv, rcond=None)
            return float(1 - ((yv - Z @ b) ** 2).sum() / ((yv - yv.mean()) ** 2).sum())
        # The presence indicator and the value are collinear by construction (the value is zero
        # wherever the indicator is), so a sequential R-squared depends on the entry order. Both
        # orders are reported and no ratio between them is formed.
        R['variance_decomposition'] = dict(
            n=len(Ar),
            r2_reaction_identity=round(r2([1]), 4),
            r2_plus_presence=round(r2([1, 2]), 4),
            r2_plus_magnitude=round(r2([1, 2, 3]), 4),
            r2_plus_magnitude_first=round(r2([1, 3]), 4),
            r2_presence_magnitude_corr=round(float(np.corrcoef(Ar[:, 2], Ar[:, 3])[0, 1]), 4),
            mean_p_shown=round(float(yv[Ar[:, 2] == 1].mean()), 4),
            mean_p_not_shown=round(float(yv[Ar[:, 2] == 0].mean()), 4),
            mean_p_blind_shown=round(float(Ar[Ar[:, 2] == 1, 1].mean()), 4),
            mean_p_blind_not_shown=round(float(Ar[Ar[:, 2] == 0, 1].mean()), 4),
            note='nested least squares; columns are the patient-blind score, an indicator that a '
                 'value was shown, and the value itself; both entry orders are given because the '
                 'two columns are collinear by construction')
        # (ii) restricted to expression-bearing reactions, where presence is constant
        e = Ar[Ar[:, 2] == 1]
        if len(e) > 20:
            ye = e[:, 0]
            def r2e(cols):
                Z = np.column_stack([np.ones(len(e))] + [e[:, c] for c in cols])
                b, *_ = np.linalg.lstsq(Z, ye, rcond=None)
                return float(1 - ((ye - Z @ b) ** 2).sum() / ((ye - ye.mean()) ** 2).sum())
            R['variance_decomposition']['expr_only_r2_reaction_identity'] = round(r2e([1]), 4)
            R['variance_decomposition']['expr_only_r2_plus_magnitude'] = round(r2e([1, 3]), 4)
            # the prompt line carries the value and its cohort percentile; the same restricted fit
            # with the percentile in place of the value, and with both, says which the score follows
            R['variance_decomposition']['expr_only_r2_plus_percentile'] = round(r2e([1, 4]), 4)
            R['variance_decomposition']['expr_only_r2_plus_both'] = round(r2e([1, 3, 4]), 4)
            R['variance_decomposition']['expr_only_spearman_percentile'] = round(float(st.spearmanr(e[:, 0], e[:, 4]).statistic), 4)
            R['variance_decomposition']['expr_only_n'] = int(len(e))

    # ---- strata: does the model reproduce the indicator floor internally? ----
    RX = PATHS.reactions()
    sub = np.array([(RX[i].get('subsystem') or '') for i in IDX])
    istr = np.array([s.startswith('Transport') or 'xchange' in s for s in sub])
    R['strata'] = {}
    for a in ARMS:
        s_ = np.nanmean(M[a], axis=0); ok = np.isfinite(s_); e = {}
        for nm, m in [('expression_bearing', hx), ('no_expression', ~hx),
                      ('transport_exchange', istr), ('metabolic', ~istr)]:
            mm = m & ok
            if mm.sum() > 5 and len(set(y[mm])) > 1:
                e[nm] = dict(n=int(mm.sum()), label_active=round(float(y[mm].mean()), 3),
                             mean_p=round(float(s_[mm].mean()), 4),
                             auroc=round(float(roc_auc_score(y[mm], s_[mm])), 4))
        e['mean_p_label_active'] = round(float(np.nanmean(s_[ok & (y == 1)])), 4)
        e['mean_p_label_inactive'] = round(float(np.nanmean(s_[ok & (y == 0)])), 4)
        R['strata'][a] = e
    return R

def rescore_label(out, tag, y_alt, label_name):
    """Every arm of a run scored against an alternative label vector (the tissue-matched HT29-only
    label of the companion robustness check), with the same raw-expression and indicator references,
    the patient-level bootstrap interval of each arm, and the paired differences between arms."""
    M, IDX, PSEL, ARMS, des = load(out)
    y = y_alt[IDX].astype(int); hx = HAS[IDX]; Xs = X[np.ix_(PSEL, IDX)]
    def auc_rows(S):
        v = []
        for i in range(len(PSEL)):
            ok = np.isfinite(S[i])
            v.append(roc_auc_score(y[ok], S[i][ok]) if len(set(y[ok])) > 1 else np.nan)
        return np.array(v)
    R = dict(tag=tag, out=out, label=label_name, n_reactions=len(IDX), n_patients=len(PSEL),
             label_base_rate=round(float(y.mean()), 4), model=des.get('model'),
             raw_expression_auroc=round(float(np.nanmean(auc_rows(Xs))), 4),
             indicator_floor=round(float(roc_auc_score(y, hx.astype(float))), 4), per_arm={})
    rows = {}
    for a in ARMS:
        if a not in M: continue
        v = auc_rows(M[a]); rows[a] = v
        lo, hi = boot_patients(lambda ix: float(np.nanmean(v[ix])), len(PSEL))
        R['per_arm'][a] = dict(auroc=round(float(np.nanmean(v)), 4), auroc_sd=round(float(np.nanstd(v)), 4),
                               auroc_ci=[round(lo, 4), round(hi, 4)])
    for a, b_, nm in (('patient', 'shuffled', 'own_minus_stranger'), ('patient', 'reaction', 'own_minus_blind'),
                      ('shuffled', 'reaction', 'stranger_minus_blind')):
        if a in rows and b_ in rows:
            d = rows[a] - rows[b_]; d = d[np.isfinite(d)]
            rng = np.random.default_rng(11)
            bs = np.array([float(np.mean(d[rng.integers(0, len(d), len(d))])) for _ in range(2000)])
            R[nm] = dict(mean=round(float(d.mean()), 4), sd=round(float(d.std()), 4), n=int(len(d)),
                         ci95=[round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)],
                         ci90=[round(float(np.percentile(bs, 5)), 4), round(float(np.percentile(bs, 95)), 4)],
                         n_positive=int((d > 0).sum()), n_negative=int((d < 0).sum()))
    return R

def format_arm(out, tag, ref_tag='B_balanced'):
    """A prompt-format variant: same reactions, same patients, different rendering of the evidence."""
    M, IDX, PSEL, ARMS, des = load(out)
    y = LAB[IDX].astype(int); hx = HAS[IDX]
    R = dict(tag=tag, out=out, format=des.get('format'), n_reactions=len(IDX),
             n_patients=len(PSEL), n_calls=int(len(ARMS) * len(PSEL) * len(IDX)),
             reference_run=ref_tag, per_arm={})
    for a in ARMS:
        S = M[a]
        v = []
        for i in range(len(PSEL)):
            ok = np.isfinite(S[i])
            v.append(roc_auc_score(y[ok], S[i][ok]) if len(set(y[ok])) > 1 else np.nan)
        v = np.array(v)
        full = np.isfinite(S).all(1)
        def _med_r(mask):
            # median off-diagonal correlation between patients' score vectors on the masked reactions,
            # the same statistic as analyze(): expression-bearing reactions by default, all reactions as _all
            Sm = S[full][:, mask]
            C = np.corrcoef(Sm) if full.sum() > 1 and mask.sum() > 1 else np.array([[1.0]])
            offd = C[np.triu_indices(len(C), 1)] if len(C) > 1 else np.array([np.nan])
            return round(float(np.median(offd)), 4)
        e = dict(parsed=round(float(np.isfinite(S).mean()), 4),
                 auroc=round(float(np.nanmean(v)), 4),
                 auroc_ci=[round(x, 4) for x in boot_patients(lambda ix: np.nanmean(v[ix]), len(PSEL))],
                 median_inter_patient_r=_med_r(hx),
                 median_inter_patient_r_all=_med_r(np.ones(len(IDX), bool)),
                 mean_p=round(float(np.nanmean(S)), 4),
                 n_distinct_values=int(len(np.unique(S[np.isfinite(S)]))))
        s_ = np.nanmean(S, axis=0); ok = np.isfinite(s_)
        for nm, m in (('mean_p_expression_bearing', hx), ('mean_p_no_expression', ~hx)):
            mm = m & ok
            if mm.sum() > 5: e[nm] = round(float(s_[mm].mean()), 4)
        R['per_arm'][a] = e
    if 'patient' in ARMS and 'shuffled' in ARMS:
        P, S = M['patient'], M['shuffled']
        ex = np.isfinite(P) & np.isfinite(S) & hx[None, :]
        R['patient_vs_shuffled'] = dict(
            n_pairs=int(ex.sum()),
            identical=round(float((P[ex] == S[ex]).mean()), 4),
            mean_abs_diff=round(float(np.abs(P[ex] - S[ex]).mean()), 4),
            between_patient_sd_real=round(float(np.nanstd(P[:, hx], axis=0).mean()), 4),
            between_patient_sd_stranger=round(float(np.nanstd(S[:, hx], axis=0).mean()), 4))
    return R


if __name__ == '__main__':
    def _has(run, fn):
        try: PATHS.run(run, fn); return True
        except FileNotFoundError: return False
    # Which collection carries each design. Every design was first collected in blocked arm order
    # (every patient-blind call, then every own-data call, then every stranger's-data call). Designs A
    # and E on the main backbone, Design A on the second backbone and the third backbone were then
    # re-collected with the arms interleaved in a seeded random order and the complete reply archived
    # (same seed, reactions, patients and donors). The interleaved collection is the primary evidence
    # wherever it exists; the blocked collection keeps its own tag and is compared with it arm by arm.
    def pick(primary, fallback, tag, fallback_tag):
        """(out, tag) pairs: the primary collection under the design's tag, the fallback under its own tag;
        when the primary is absent the fallback carries the design's tag and a flag records that."""
        pairs = []
        if _has(primary, 'design.json'):
            pairs.append((primary, tag))
            if _has(fallback, 'design.json'): pairs.append((fallback, fallback_tag))
        elif _has(fallback, 'design.json'):
            pairs.append((fallback, tag)); COLLECTION_FALLBACK[tag] = fallback
        return pairs
    COLLECTION_FALLBACK = {}
    runs = pick('repr_il', 'repr', 'A_representative', 'A_blocked')
    runs += [('pilot', 'B_balanced')]
    if _has('b2', 'design.json'): runs.append(('b2', 'C_backbone2'))
    # Design A repeated on a second backbone (Gemma 4 31B-it): the same seed, reactions and patients, served locally
    runs += pick('repr_b2il', 'repr_b2', 'A_backbone2', 'A_backbone2_blocked')
    # a third backbone (Mistral Small 3.2 24B), same seed, reactions, patients and donors
    runs += pick('repr_b3il', 'repr_b3', 'A_backbone3', 'A_backbone3_blocked')
    # Design A on the main backbone with an independently chosen donor schedule (donor seed 4048)
    if _has('repr_il_d2', 'design.json'): runs.append(('repr_il_d2', 'A_donor2'))
    # Design E (the 40-patient replication) under the same second donor schedule
    if _has('repr40_il_d2', 'design.json'): runs.append(('repr40_il_d2', 'E_donor2'))
    if _has('det_patient', 'design.json'): runs.append(('det_patient', 'F_replicate'))
    # the 40-patient replication of Design A: same reactions, same prompts, same backbone,
    # four times the patients, with the donor assignment recorded
    runs += pick('repr40_il', 'repr40', 'E_representative40', 'E_blocked')
    import os
    if _has('think', 'design.json'): runs.append(('think', 'D_reasoning'))
    OUT = {'runs': {}, 'collections': {tag: out for out, tag in runs}, 'collection_fallback': COLLECTION_FALLBACK}
    for out, tag in runs:
        try:
            OUT['runs'][tag] = analyze(out, tag); print(f'ok {tag}', flush=True)
        except Exception as e:
            print(f'skip {out}: {type(e).__name__}: {e}', flush=True)
    BY_TAG = {tag: out for out, tag in runs}
    # The cross-session floor. Design E re-sends, in a later session and a separate process, the
    # byte-identical prompts of Design A for the ten patients the two designs share (the patient
    # and reaction arms; the stranger's arm draws different donors). Comparing the two answers is
    # a test-retest measurement of the patient prompt itself on 3,000 cells, which neither the
    # within-session repeat of the patient-blind prompt nor the small replicate arm provides.
    def cell_agreement(MA, ME, hxA, arms):
        XS = {}
        for a in arms:
            if a not in MA or a not in ME: continue
            A_ = MA[a]; E_ = ME[a]
            m_all = np.isfinite(A_) & np.isfinite(E_); m_hx = m_all & hxA[None, :]
            XS[a] = dict(n_cells=int(m_all.sum()), n_cells_expr=int(m_hx.sum()),
                         identical=round(float((A_[m_all] == E_[m_all]).mean()), 4),
                         mean_abs_diff=round(float(np.abs(A_[m_all] - E_[m_all]).mean()), 6),
                         identical_expr=round(float((A_[m_hx] == E_[m_hx]).mean()), 4),
                         mean_abs_diff_expr=round(float(np.abs(A_[m_hx] - E_[m_hx]).mean()), 6))
        return XS
    def time_gap_hours(outA, outE, PA, PE, IA):
        """Median hours between the answers to the same (patient, reaction, arm) cell in two collections,
        from the archived sent_at timestamps when both archives carry them."""
        try:
            ra = json.load(open(PATHS.run(outA, 'raw.json'))); re_ = json.load(open(PATHS.run(outE, 'raw.json')))
        except FileNotFoundError:
            return None
        gaps = []
        for a in ('patient', 'reaction'):
            for pj in PA:
                for ri in IA:
                    k = f'{a}|{pj}|{ri}'; va = ra.get(k, {}).get('meta') or {}; ve = re_.get(k, {}).get('meta') or {}
                    if va.get('sent_at') and ve.get('sent_at'): gaps.append((float(ve['sent_at']) - float(va['sent_at'])) / 3600.0)
        if gaps:
            return dict(median_hours=round(float(np.median(gaps)), 2), min_hours=round(float(np.min(gaps)), 2), max_hours=round(float(np.max(gaps)), 2), n=len(gaps), basis='per-call timestamps in both archives')
        # The first collections carry no per-call timestamps. Their completion time (the archive's last
        # write on the collection host, results/collection_times.json) stands in for the send time of
        # every call, so the gap is measured from the end of the first collection to each call of the second.
        try:
            done = json.load(open(PATHS._first(os.path.join(PATHS.ROOT, 'results', 'collection_times.json'), 'results/collection_times.json', 'collection_times.json'))).get(outA)
        except FileNotFoundError:
            done = None
        if done is None: return None
        sent = [float(v['meta']['sent_at']) for v in re_.values() if isinstance(v, dict) and (v.get('meta') or {}).get('sent_at')]
        if not sent: return None
        gaps = [(t - float(done)) / 3600.0 for t in sent]
        return dict(median_hours=round(float(np.median(gaps)), 2), min_hours=round(float(np.min(gaps)), 2), max_hours=round(float(np.max(gaps)), 2), n=len(gaps), basis='from the completion of the first collection to each call of the second')
    if 'A_representative' in OUT['runs'] and 'E_representative40' in OUT['runs']:
        MA, IA, PA, ARA, _ = load(BY_TAG['A_representative']); ME, IE, PE, ARE, _ = load(BY_TAG['E_representative40'])
        if list(IA) == list(IE) and set(PA.tolist()) <= set(PE.tolist()):
            pos = {int(p): i for i, p in enumerate(PE)}; hxA = HAS[IA]
            ME_A = {a: np.vstack([ME[a][pos[int(p)]] for p in PA]) for a in ME}
            XS = cell_agreement(MA, ME_A, hxA, ('patient', 'reaction'))
            OUT['cross_session'] = dict(designs=['A_representative', 'E_representative40'],
                                        collections=[BY_TAG['A_representative'], BY_TAG['E_representative40']],
                                        n_patients=int(len(PA)), n_reactions=int(len(IA)), arms=XS,
                                        time_gap=time_gap_hours(BY_TAG['A_representative'], BY_TAG['E_representative40'], PA, PE, IA),
                                        note='byte-identical prompts answered in two sessions; the same patients and reactions')
    # Each interleaved re-collection answers the byte-identical prompts of its blocked collection (same
    # seed, reactions, patients and donors) in a seeded random order across arms, in a later session: the
    # check on the blocked-order confound, arm by arm, and a further cross-session measurement.
    OUT['interleaved_vs_blocked'] = {}
    for tag, btag in (('A_representative', 'A_blocked'), ('A_backbone2', 'A_backbone2_blocked'),
                      ('A_backbone3', 'A_backbone3_blocked'), ('E_representative40', 'E_blocked')):
        if tag in OUT['runs'] and btag in OUT['runs']:
            MB, IB, PB, ARB, _ = load(BY_TAG[btag]); MI, II, PI, ARI, _ = load(BY_TAG[tag])
            if list(IB) == list(II) and list(PB) == list(PI):
                XS = cell_agreement(MB, MI, HAS[IB], ('patient', 'reaction', 'shuffled'))
                rb, ri_ = OUT['runs'][btag], OUT['runs'][tag]
                OUT['interleaved_vs_blocked'][tag] = dict(
                    blocked=btag, collections=[BY_TAG[btag], BY_TAG[tag]], arms=XS,
                    auroc_blocked={a: rb['per_arm'][a]['auroc'] for a in rb['per_arm']},
                    auroc_interleaved={a: ri_['per_arm'][a]['auroc'] for a in ri_['per_arm']},
                    own_minus_stranger_blocked=rb.get('patient_vs_shuffled', {}).get('auroc_diff_mean'),
                    own_minus_stranger_interleaved=ri_.get('patient_vs_shuffled', {}).get('auroc_diff_mean'),
                    time_gap=time_gap_hours(BY_TAG[btag], BY_TAG[tag], PB, PI, IB),
                    note='byte-identical prompts, blocked order then interleaved order, two sessions')
    # the second donor schedule against the primary one: the same reactions, patients and blind/own
    # prompts, a different stranger at every swapped cell
    for _primary, _second, _key in (('A_representative', 'A_donor2', 'donor_schedules'),
                                    ('E_representative40', 'E_donor2', 'donor_schedules_E')):
        if _primary in OUT['runs'] and _second in OUT['runs']:
            r1, r2 = OUT['runs'][_primary], OUT['runs'][_second]
            OUT[_key] = dict(
                own_minus_stranger=[r1['patient_vs_shuffled']['auroc_diff_mean'], r2['patient_vs_shuffled']['auroc_diff_mean']],
                own_minus_stranger_ci90=[r1['patient_vs_shuffled']['auroc_diff_ci90'], r2['patient_vs_shuffled']['auroc_diff_ci90']],
                own_minus_stranger_p=[r1['patient_vs_shuffled'].get('paired_p'), r2['patient_vs_shuffled'].get('paired_p')],
                stranger_auroc=[r1['per_arm']['shuffled']['auroc'], r2['per_arm']['shuffled']['auroc']],
                own_auroc=[r1['per_arm']['patient']['auroc'], r2['per_arm']['patient']['auroc']],
                blind_auroc=[r1['per_arm']['reaction']['auroc'], r2['per_arm']['reaction']['auroc']],
                between_patient_sd_stranger=[r1['patient_vs_shuffled']['between_patient_sd_stranger'], r2['patient_vs_shuffled']['between_patient_sd_stranger']],
                equivalent_002=[r1['patient_vs_shuffled'].get('tost_narrow_equivalent'), r2['patient_vs_shuffled'].get('tost_narrow_equivalent')],
                equivalent_005=[r1['patient_vs_shuffled'].get('tost_equivalent'), r2['patient_vs_shuffled'].get('tost_equivalent')],
                note='two independently drawn donor schedules on the same reactions and patients')
    # prompt-format variants
    OUT['formats'] = {}
    for out, tag in [('fmt_cat', 'categorical'), ('fmt_pct', 'percentile')]:
        if not _has(out, 'design.json'): continue
        try:
            OUT['formats'][tag] = format_arm(out, tag); print(f'ok format:{tag}', flush=True)
        except Exception as e:
            print(f'skip {out}: {type(e).__name__}: {e}', flush=True)
    # every reaction-activity run rescored against the tissue-matched label (HT29 alone), when the
    # companion's label file is present; the runs themselves are untouched
    _lab = os.path.join(PATHS.ROOT, '..', 'paper1', 'data', 'activity_labels_ht29.pt')
    if not os.path.exists(_lab) and not os.environ.get('METABENCH_ALLOW_MISSING_HT29'):
        raise FileNotFoundError(f'{_lab} is missing: the tissue-matched label (Table 5) is built by '
                                'paper1/code/build_labels_ht29.py in the shared repository; set '
                                'METABENCH_ALLOW_MISSING_HT29=1 to write results without it')
    if os.path.exists(_lab):
        try:
            import torch
            y_ht = np.asarray(torch.load(_lab, map_location='cpu', weights_only=True)).astype(int)
            OUT['label_ht29'] = dict(source='paper1/data/activity_labels_ht29.pt', n_active=int(y_ht.sum()),
                                     prevalence=round(float(y_ht.mean()), 4),
                                     indicator_floor_network=round(float(roc_auc_score(y_ht, HAS.astype(float))), 4), runs={})
            for out, tag in runs:
                if tag in ('F_replicate', 'D_reasoning', 'A_donor2', 'E_donor2') or tag.endswith('_blocked'): continue
                try:
                    OUT['label_ht29']['runs'][tag] = rescore_label(out, tag, y_ht, 'HT29 only'); print(f'ok ht29:{tag}', flush=True)
                except Exception as e:
                    print(f'skip ht29 {out}: {type(e).__name__}: {e}', flush=True)
        except Exception as e:
            print(f'skip ht29 rescoring: {type(e).__name__}: {e}', flush=True)
    # positive controls
    OUT['positive_control'] = {}
    _pc3 = ['pc_b3il'] if _has('pc_b3il', 'raw.json') else ['pc_b3', 'pc_echo_b3']
    # the value-echo control (copy the printed three-decimal expression value) is a separate run per backbone
    for tag, outs in [('backbone1', ['pc', 'pc_echo_b1', 'pc_echoval']), ('backbone2', ['pc_b2', 'pc_echo_b2', 'pc_echoval_b2']),
                      ('backbone3', _pc3 + ['pc_echoval_b3'])]:
        r = {}
        for out in outs:
            if _has(out, 'raw.json'): r.update(json.load(open(PATHS.run(out, 'raw.json'))))
        if not r: continue
        e = {}
        # The prompt prints the percentile rounded to an integer, so the answer key is the printed
        # integer: `read` is scored against round(p) > 50, `compare` against the two printed integers,
        # and a `compare` trial whose two printed integers are equal has no correct answer and is
        # excluded (counted under n_tied). The collection script stored a key computed from the
        # unrounded percentile; it is not used here.
        def key(task, x):
            p1 = int(round(x['pct'])); p2 = int(round(x['pct2'])) if x.get('pct2') is not None else None
            if task == 'read': return p1 > 50
            if task == 'echo': return p1
            if task == 'echoval': return float(f"{x['val']:.3f}")     # the value as the prompt prints it
            return None if p1 == p2 else (1 if p1 > p2 else 2)
        for task in ('read', 'compare', 'echo', 'echoval'):
            rows = [(x['ans'], key(task, x), x) for k, x in r.items() if k.startswith(task + '|') and x['ans'] is not None]
            tot = len([k for k in r if k.startswith(task + '|')])
            if not tot: continue
            n_tied = sum(1 for _, t, _ in rows if t is None)
            v = [(a, t) for a, t, _ in rows if t is not None]
            acc = float(np.mean([(abs(a - t) < 1.0 if task == 'echo' else abs(a - t) < 0.0005 if task == 'echoval' else a == t) for a, t in v]))
            if task in ('echo', 'echoval'):
                hard = []
                # degenerate-answer check: share taken by the single most common answer
                import collections as _c
                modal = _c.Counter(a for a, _ in v).most_common(1)[0][1] / len(v)
                e[f'{task}_modal_share'] = round(float(modal), 4)
                if task == 'echoval':
                    # how far the wrong copies are from the printed value, in units of the last printed digit
                    _wrong = [abs(a - t) for a, t in v if abs(a - t) >= 0.0005]
                    e['echoval_median_abs_error_wrong'] = (round(float(np.median(_wrong)), 4) if _wrong else None)
                    e['echoval_within_one_digit'] = round(float(np.mean([abs(a - t) < 0.0015 for a, t in v])), 4)
            elif task == 'read':
                hard = [(a, t) for a, t, x in rows if t is not None and abs(round(x['pct']) - 50) < 10]
            else:
                hard = [(a, t) for a, t, x in rows if t is not None and abs(round(x['pct']) - round(x['pct2'])) < 10]
            e[task] = dict(n=tot, parsed=len(v) + n_tied, n_scored=len(v), n_tied=n_tied,
                           accuracy=round(acc, 4), n_hard=len(hard),
                           accuracy_hard=(round(float(np.mean([a == t for a, t in hard])), 4) if hard else None))
            if task == 'compare':
                errs = [a for a, t in v if a != t]
                e[task]['n_errors'] = len(errs); e[task]['errors_answering_first'] = int(sum(1 for a in errs if a == 1))
            if task in ('read', 'compare'):
                import collections as _c
                e[task]['modal_answer_share'] = round(float(_c.Counter(a for a, _ in v).most_common(1)[0][1] / len(v)), 4)
        OUT['positive_control'][tag] = e
    # cohort composition, computed from the released files rather than quoted
    import csv as _csv, collections as _co
    _rx = PATHS.reactions()
    _msi = _co.Counter((r['msi_status'] or '').strip()
                       for r in _csv.DictReader(open(PATHS.data('clinical_metadata_msi.tsv')), delimiter='\t'))
    OUT['cohort'] = dict(
        n_gpr_rule=int(sum(1 for r in _rx if r['has_gene'])),
        n_no_gpr_rule=int(sum(1 for r in _rx if not r['has_gene'])),
        n_gene_and_expression=int(sum(1 for r, h in zip(_rx, HAS) if r['has_gene'] and h)),
        n_gene_no_expression=int(sum(1 for r, h in zip(_rx, HAS) if r['has_gene'] and not h)),
        n_expression_no_gene=int(sum(1 for r, h in zip(_rx, HAS) if (not r['has_gene']) and h)),
        n_aligned=int(sum(1 for r in _rx if r['aligned'])), n_unaligned=int(sum(1 for r in _rx if not r['aligned'])),
        msi_h=int(_msi['MSI-H']), msi_l=int(_msi['MSI-L']), mss=int(_msi['MSS']),
        not_evaluable=int(_msi['NOT EVALUABLE']), n_annotated=int(sum(_msi.values())))
    # network-level reference values
    OUT['network'] = dict(
        n_reactions=int(len(LAB)), n_active=int(LAB.sum()), base_rate=round(float(LAB.mean()), 4),
        n_expression_bearing=int(HAS.sum()), n_patients=int(len(PIDS)),
        prevalence_expression_bearing=round(float(LAB[HAS].mean()), 4),
        prevalence_no_expression=round(float(LAB[~HAS].mean()), 4),
        indicator_floor=round(float(roc_auc_score(LAB, HAS.astype(float))), 4),
        raw_expression_auroc=round(float(np.mean([roc_auc_score(LAB, X[i]) for i in range(len(PIDS))])), 4),
        raw_expression_auroc_sd=round(float(np.std([roc_auc_score(LAB, X[i]) for i in range(len(PIDS))])), 4))
    json.dump(OUT, open(PATHS.out('results_frozen.json'), 'w'), indent=1)
    print('\nwrote results_frozen.json')
