#!/usr/bin/env python3
"""Statistics for the frozen percentile-only interface (msi_probe2.py collections).

Implements docs/ANALYSIS_PLAN_2026-09-10.md, Section 2, for every collection under results/msi2/:
own and donor AUROC with the own-minus-donor contrast; a patient bootstrap that redraws the donor
derangement inside every replicate (the population interval), beside the naive fixed-schedule
bootstrap; five donor schedules by re-indexing the own scores (a donor prompt is byte-identical
to the donor's own prompt, which the archived real donor calls of schedule 11 confirm); a full
profile-label alignment permutation null; AUPRC, Brier score, clipped log loss, calibration slope
and intercept and a reliability curve; the same for the frozen logistic reference, the
single-feature rule and the prevalence constant; serving variation from the same-session repeats;
the reading control; failed-call accounting with the prevalence-imputation sensitivity; and, for
the tool configuration, numeric fidelity to the stated tool probability. Calibration maps are
fitted out of fold on the development collection and applied unchanged to the external
collections of the same backbone and configuration.

Writes results/msi2/stats.json.
"""
import os, re, json, glob, numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import _paths as PATHS

ROOT = os.path.join(PATHS.ROOT, 'results', 'msi2'); EXT = os.path.join(PATHS.ROOT, 'results', 'external')
SCORER = json.load(open(os.path.join(EXT, 'frozen_scorer.json')))
SEEDS_DONOR = (11, 22, 33, 44, 55); B_BOOT = 2000; N_PERM = 10000; CLIP = (0.005, 0.995)

def derangement(n, seed):
    r = np.random.default_rng(seed)
    while True:
        p = r.permutation(n)
        if not np.any(p == np.arange(n)): return p
def auc(y, s):
    return float(roc_auc_score(y, s)) if len(set(y.tolist())) > 1 else float('nan')
def logit(p):
    p = np.clip(p, CLIP[0], CLIP[1]); return np.log(p / (1 - p))
def calib_slope_intercept(y, p):
    z = logit(p).reshape(-1, 1)
    m = LogisticRegression(C=1e6, max_iter=5000).fit(z, y)
    return float(m.coef_[0][0]), float(m.intercept_[0])
def reliability(y, p, nb=10):
    order = np.argsort(p); bins = np.array_split(order, nb)
    return [dict(n=int(len(b)), mean_pred=float(p[b].mean()), obs_rate=float(y[b].mean())) for b in bins if len(b)]
def score_block(y, p):
    p = np.asarray(p, float); pc = np.clip(p, CLIP[0], CLIP[1])
    out = dict(auroc=auc(y, p), auprc=float(average_precision_score(y, p)), brier=float(brier_score_loss(y, pc)),
               log_loss=float(log_loss(y, pc, labels=[0, 1])), n=int(len(y)), n_pos=int(y.sum()),
               n_distinct=int(len(set(np.round(p, 6).tolist()))), modal_share=float(np.max(np.unique(np.round(p, 6), return_counts=True)[1]) / len(p)))
    if out['n_distinct'] > 1:
        s, i = calib_slope_intercept(y, p); out.update(calib_slope=s, calib_intercept=i)
    out['reliability'] = reliability(y, p)
    return out

def percentiles_for(des):
    """The panel percentiles of the evaluated patients, in the collection's own convention."""
    if des['cohort'] == 'tcga':
        D = np.load(PATHS.data('rxn_context.npz'), allow_pickle=True); X = D['X']
        P = np.stack([100.0 * (rankdata(X[:, j]) - 1.0) / (X.shape[0] - 1) for j in des['panel']], axis=1)
        return P[np.array(des['keep'])]
    Z = np.load(os.path.join(EXT, f"{des['cohort']}_panel.npz"), allow_pickle=True)
    pos = [list(Z['panel_ids']).index(i) for i in des['panel_ids']]
    return Z['pct'][:, pos].astype(np.float64)[np.array(des['keep'])]

def analyze(d):
    raw = json.load(open(os.path.join(d, 'raw.json'))); des = json.load(open(os.path.join(d, 'design.json')))
    n = len(des['keep']); y = np.array(des['y']); P = percentiles_for(des)
    get = lambda arm, a: raw.get(f'{arm}|{a}', {}).get('val')
    s_own = np.array([get('own', a) for a in range(n)], dtype=object)
    s_don = np.array([get('donor', a) for a in range(n)], dtype=object)
    ok_own = np.array([v is not None for v in s_own]); ok_don = np.array([v is not None for v in s_don])
    R = dict(dir=os.path.basename(d), cohort=des['cohort'], endpoint=des['endpoint'], config=des['config'], model=des['model'],
             n=n, n_pos=int(y.sum()), prevalence=float(y.mean()), failed_calls=des.get('failed_calls'), minutes=des.get('minutes'),
             n_own_evaluable=int(ok_own.sum()), n_donor_evaluable=int(ok_don.sum()))
    prev = float(y.mean())
    so = np.array([float(v) if v is not None else prev for v in s_own]); sd = np.array([float(v) if v is not None else prev for v in s_don])
    both = ok_own & ok_don
    # paired evaluable analysis (primary) and the prevalence-imputed sensitivity
    for tag, mask, S_o, S_d in (('paired_evaluable', both, so, sd), ('prevalence_imputed', np.ones(n, bool), so, sd)):
        yy = y[mask]
        blk = dict(own=score_block(yy, S_o[mask]), donor=score_block(yy, S_d[mask]))
        blk['own_minus_donor_auroc'] = blk['own']['auroc'] - blk['donor']['auroc']
        R[tag] = blk
    # the real donor calls against the re-indexed own scores (byte-identical prompts)
    DON = np.array(des['donor']); reidx = so[DON]
    m = both & ok_own[DON]
    R['donor_call_vs_reindexed_own'] = dict(n=int(m.sum()), mean_abs_diff=float(np.mean(np.abs(sd[m] - reidx[m]))), share_identical=float(np.mean(np.abs(sd[m] - reidx[m]) < 1e-9)))
    # five donor schedules by re-indexing the own scores (label-blind derangements), on own-evaluable patients
    idx = np.where(ok_own)[0]; yo = y[idx]; s = so[idx]
    sched = {}
    for sd_ in SEEDS_DONOR:
        D_ = derangement(len(idx), sd_); sched[str(sd_)] = dict(auroc_donor=auc(yo, s[D_]), own_minus_donor=auc(yo, s) - auc(yo, s[D_]))
    vals = [v['own_minus_donor'] for v in sched.values()]
    R['schedules'] = dict(per_seed=sched, own_auroc=auc(yo, s), own_minus_donor_mean=float(np.mean(vals)), own_minus_donor_min=float(np.min(vals)), own_minus_donor_max=float(np.max(vals)))
    # bootstrap: patients resampled, the derangement redrawn inside every replicate (primary interval)
    rng = np.random.default_rng(7); diffs = []; diffs_fixed = []; own_b = []; don_b = []
    D11 = derangement(len(idx), 11)
    for b in range(B_BOOT):
        bi = rng.integers(0, len(idx), len(idx)); yb = yo[bi]
        if len(set(yb.tolist())) < 2: continue
        perm = derangement(len(bi), int(rng.integers(0, 2**31 - 1)))
        a_o = auc(yb, s[bi]); a_d = auc(yb, s[bi][perm]); diffs.append(a_o - a_d); own_b.append(a_o); don_b.append(a_d)
        diffs_fixed.append(a_o - auc(yb, s[D11][bi]))
    diffs = np.array(diffs); diffs_fixed = np.array(diffs_fixed)
    R['bootstrap'] = dict(B=int(len(diffs)), own_ci95=[float(np.percentile(own_b, 2.5)), float(np.percentile(own_b, 97.5))],
                          donor_ci95=[float(np.percentile(don_b, 2.5)), float(np.percentile(don_b, 97.5))],
                          own_minus_donor_ci95=[float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
                          own_minus_donor_ci95_fixed_schedule=[float(np.percentile(diffs_fixed, 2.5)), float(np.percentile(diffs_fixed, 97.5))],
                          note='patients resampled with replacement; the donor derangement redrawn among the resampled patients in every replicate; '
                               'the fixed-schedule interval keeps schedule 11 and is descriptive')
    lo, hi = R['bootstrap']['own_minus_donor_ci95']
    R['verdict'] = ('advantage' if lo > 0 else 'reversed' if hi < 0 else 'inconclusive')
    # alignment permutation null: full permutations of the profile-label alignment, fixed points allowed
    prng = np.random.default_rng(13); a_obs = auc(yo, s); d_obs = a_obs - auc(yo, s[D11]); null_a = []; null_d = []
    for _ in range(N_PERM):
        pi = prng.permutation(len(idx)); sp = s[pi]; null_a.append(auc(yo, sp)); null_d.append(auc(yo, sp) - auc(yo, sp[D11]))
    null_a = np.array(null_a); null_d = np.array(null_d)
    R['permutation_null'] = dict(n_perm=N_PERM, own_auroc=a_obs, p_own_auroc=float((np.sum(null_a >= a_obs) + 1) / (N_PERM + 1)),
                                 own_minus_donor=d_obs, p_own_minus_donor=float((np.sum(null_d >= d_obs) + 1) / (N_PERM + 1)),
                                 null_diff_sd=float(null_d.std()))
    # references on the identical patients and the identical schedule
    EP = SCORER['endpoints']['msih_vs_nonmsih' if des['endpoint'] == 'nonmsih' else 'msih_vs_mss']
    coef = np.array(EP['coef']); b0 = EP['intercept']
    lr = 1.0 / (1.0 + np.exp(-(P / 100.0 @ coef + b0)))
    sf = EP['single_feature']; rule = (P[:, sf['panel_position']] / 100.0) * sf['direction']
    R['references'] = dict(logistic=dict(own=score_block(y, lr), donor=score_block(y, lr[DON]), own_minus_donor_auroc=auc(y, lr) - auc(y, lr[DON]),
                                         schedules={str(sd_): auc(y, lr) - auc(y, lr[derangement(n, sd_)]) for sd_ in SEEDS_DONOR}),
                           single_feature=dict(reaction_id=sf['reaction_id'], own_auroc=auc(y, rule), donor_auroc=auc(y, rule[DON]), own_minus_donor_auroc=auc(y, rule) - auc(y, rule[DON])),
                           prevalence_constant=dict(auroc=0.5, brier=float(prev * (1 - prev)), development_prevalence=EP['prevalence']))
    # the model's own values against the reference on the same patients
    R['own_vs_logistic_auroc_gap'] = R['paired_evaluable']['own']['auroc'] - auc(y[both], lr[both])
    # serving variation: same-session repeats of the own prompt
    rep = [(float(so[a]), float(get('own_repeat', a))) for a in range(des.get('repeat_n', 0)) if get('own_repeat', a) is not None and ok_own[a]]
    if rep:
        dd = np.array([abs(a - b) for a, b in rep])
        R['repeat'] = dict(n=int(len(rep)), median_abs_change=float(np.median(dd)), mean_abs_change=float(dd.mean()), share_identical=float(np.mean(dd < 1e-9)), max_abs_change=float(dd.max()))
    # the reading control
    cnt = [(a, get('count', a)) for a in des.get('count_patients', [])]
    if cnt:
        truth = {a: int(np.sum(np.round(P[a]) >= 75)) for a, _ in cnt}
        okc = [(a, v) for a, v in cnt if v is not None]
        R['count_control'] = dict(n=int(len(cnt)), n_answered=int(len(okc)), exact=float(np.mean([abs(v - truth[a]) < 0.5 for a, v in okc])) if okc else None,
                                  within_one=float(np.mean([abs(v - truth[a]) <= 1 for a, v in okc])) if okc else None,
                                  spearman=float(np.corrcoef(rankdata([v for _, v in okc]), rankdata([truth[a] for a, _ in okc]))[0, 1]) if len(okc) > 2 else None)
    # tool configuration: fidelity to the stated probability and the tool's own performance
    if des['config'] == 'tool':
        tp = np.array([raw.get(f'own|{a}', {}).get('tool_prob') for a in range(n)], dtype=object)
        okt = np.array([v is not None for v in tp]) & ok_own
        fid = np.abs(so[okt] - np.array([float(v) for v in tp[okt]]))
        R['tool'] = dict(n=int(okt.sum()), mean_abs_deviation=float(fid.mean()), share_within_0005=float(np.mean(fid <= 0.005)), max_abs_deviation=float(fid.max()),
                         tool_only=score_block(y[okt], np.array([float(v) for v in tp[okt]])), tool_only_own_minus_donor_auroc=auc(y, lr) - auc(y, lr[DON]))
    R['_own_scores'] = dict(y=y.tolist(), own=[(None if not ok_own[a] else