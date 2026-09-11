#!/usr/bin/env python3
"""Fold the MSI patient-level experiment into results_frozen.json."""
import json, re, collections
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import average_precision_score, brier_score_loss
from scipy import stats as st
import _paths as PATHS

D = np.load(PATHS.data('rxn_context.npz'), allow_pickle=True); X = D['X']; HAS = D['has']
try:
    from aligned_rxn import ALIGNED
    EXPR = np.where(HAS & ALIGNED)[0]           # the pool the panel is chosen from (msi_probe.py)
except Exception:
    EXPR = np.where(HAS)[0]
def analyze(run):
    """Every MSI statistic for one run directory (msi = the main backbone, msi_b2 = the second)."""
    raw = json.load(open(PATHS.run(run, 'raw.json'))); des = json.load(open(PATHS.run(run, 'design.json')))
    keep = np.array(des['keep']); Y = np.array(des['y']); PANEL = np.array(des['panel'])
    F = X[np.ix_(keep, PANEL)]

    def boot_auc(y, s, B=2000, seed=7):
        rng = np.random.default_rng(seed); v = []
        for _ in range(B):
            ix = rng.integers(0, len(y), len(y))
            if len(set(y[ix])) > 1: v.append(roc_auc_score(y[ix], s[ix]))
        return round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)

    M = dict(k=int(des['k']), n_patients=int(len(keep)), n_msi_high=int(Y.sum()),
             n_mss=int((1 - Y).sum()), prevalence=round(float(Y.mean()), 4),
             panel_ids=des['panel_ids'], arms={})

    for arm in ('own', 'swap'):
        s = np.array([raw.get(f'{arm}|{int(p)}', {}).get('val') for p in keep], dtype=object)
        ok = np.array([v is not None for v in s])
        sv = np.array([float(v) for v in s[ok]]); yv = Y[ok]
        cnt = collections.Counter(sv.tolist())
        M['arms'][arm] = dict(
            n=int(ok.sum()), parsed=round(float(ok.mean()), 4),
            auroc=round(float(roc_auc_score(yv, sv)), 4), auroc_ci=list(boot_auc(yv, sv)),
            n_distinct_values=int(len(cnt)), modal_share=round(float(cnt.most_common(1)[0][1] / len(sv)), 4),
            modal_value=float(cnt.most_common(1)[0][0]),
            mean_p=round(float(sv.mean()), 4), sd_p=round(float(sv.std()), 4),
            mean_p_msih=round(float(sv[yv == 1].mean()), 4), mean_p_mss=round(float(sv[yv == 0].mean()), 4))

    # how many distinct free-text rationales accompany the modal probability?
    mv = M['arms']['own']['modal_value']
    reasons, complete = [], []
    for p in keep:
        e = raw.get(f'own|{int(p)}')
        if not e or e['val'] != mv or not e.get('raw'): continue
        m = re.search(r'"reason"\s*:\s*"(.*?)(?:"|$)', e['raw'], re.S)
        if m:
            reasons.append(' '.join(m.group(1).split()))
            # the archive keeps the first 300 characters of each reply; a rationale is complete only if its
            # closing quote is inside that prefix
            complete.append(bool(re.search(r'"reason"\s*:\s*".*?"', e['raw'], re.S)))
    # with the complete archive, a cut rationale is the model's own (finish_reason 'length'), not the
    # archive's; the two are distinguished here
    _fin = [raw[f'own|{int(p)}'].get('meta', {}).get('finish_reason') for p in keep
            if f'own|{int(p)}' in raw and raw[f'own|{int(p)}'].get('meta')]
    M['full_archive'] = bool(des.get('full_archive', False))
    if _fin:
        M['own_finish_length_n'] = int(sum(1 for f in _fin if f == 'length'))
        M['own_finish_n'] = int(len(_fin))
    M['reason_n'] = len(reasons)
    M['reason_distinct'] = len(set(reasons))
    M['reason_examples'] = reasons[:3]
    M['reason_complete_n'] = int(sum(complete))
    M['reason_truncated_n'] = int(len(complete) - sum(complete))
    _cd = sorted(set(r for r, c in zip(reasons, complete) if c))
    M['reason_complete_distinct'] = len(_cd)
    M['reason_complete_digit'] = int(sum(1 for r in _cd if re.search(r'[0-9]', r)))
    M['reason_complete_percentile'] = int(sum(1 for r in _cd if re.search(r'[0-9]', r) and re.search(r'percentile', r, re.I)))
    _ad = sorted(set(reasons))
    M['reason_any_digit_all'] = int(sum(1 for r in _ad if re.search(r'[0-9]', r)))
    M['reason_any_percentile_all'] = int(sum(1 for r in _ad if re.search(r'[0-9]', r) and re.search(r'percentile', r, re.I)))
    # completeness is not random: the complete rationales are the short ones
    M['reason_mean_len_complete'] = round(float(np.mean([len(r) for r, c in zip(reasons, complete) if c])), 1) if any(complete) else None
    M['reason_mean_len_truncated'] = round(float(np.mean([len(r) for r, c in zip(reasons, complete) if not c])), 1) if not all(complete) else None

    # supervised reference on the identical features
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, C=1.0))
    oof = np.zeros(len(Y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=2024).split(F, Y):
        oof[te] = pipe.fit(F[tr], Y[tr]).predict_proba(F[te])[:, 1]
    M['logistic_auroc'] = round(float(roc_auc_score(Y, oof)), 4)
    M['logistic_auroc_ci'] = list(boot_auc(Y, oof))
    # Discrimination and calibration against the assay call beside the constant-prevalence predictor:
    # area under the precision-recall curve (a constant predictor scores the prevalence), the Brier
    # score (mean squared error of the probability; the constant predictor scores p(1-p)), and
    # calibration in the large (mean returned probability minus the observed prevalence).
    prev = float(Y.mean())
    M['constant'] = dict(auprc=round(prev, 4), brier=round(prev * (1 - prev), 4), auroc=0.5)
    for arm in ('own', 'swap'):
        s_ = np.array([raw.get(f'{arm}|{int(p)}', {}).get('val') for p in keep], dtype=object)
        ok_ = np.array([v is not None for v in s_])
        sv_ = np.clip(np.array([float(v) for v in s_[ok_]]), 0, 1); yv_ = Y[ok_]
        M['arms'][arm]['auprc'] = round(float(average_precision_score(yv_, sv_)), 4)
        M['arms'][arm]['brier'] = round(float(brier_score_loss(yv_, sv_)), 4)
        M['arms'][arm]['calibration_in_the_large'] = round(float(sv_.mean() - yv_.mean()), 4)
        # reliability by returned value: observed positive rate at each distinct probability
        _vals = sorted(set(sv_.tolist()))
        M['arms'][arm]['reliability'] = [dict(p=round(float(v), 4), n=int((sv_ == v).sum()),
                                              observed=round(float(yv_[sv_ == v].mean()), 4)) for v in _vals]
    M['logistic_auprc'] = round(float(average_precision_score(Y, oof)), 4)
    M['logistic_brier'] = round(float(brier_score_loss(Y, oof)), 4)
    M['logistic_calibration_in_the_large'] = round(float(oof.mean() - Y.mean()), 4)
    # A fold-inductive version of the reference: the panel is chosen from the training patients of
    # each fold alone (the K highest-variance expression-bearing reactions over those patients) and
    # frozen for the held-out patients, so no covariate information of the scored patients enters
    # the selection. The run as archived used a panel chosen over the whole cohort (transductive).
    oofi = np.zeros(len(Y)); _panels = []
    Xk = X[keep]
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=2024).split(F, Y):
        sel = EXPR[np.argsort(-Xk[np.ix_(tr, EXPR)].std(axis=0))[:int(des['k'])]]
        _panels.append(sorted(int(v) for v in sel))
        oofi[te] = pipe.fit(Xk[np.ix_(tr, sel)], Y[tr]).predict_proba(Xk[np.ix_(te, sel)])[:, 1]
    M['logistic_auroc_inductive'] = round(float(roc_auc_score(Y, oofi)), 4)
    M['logistic_auroc_inductive_ci'] = list(boot_auc(Y, oofi))
    M['logistic_inductive_panel_overlap'] = round(float(np.mean([len(set(pn) & set(PANEL.tolist())) / len(PANEL) for pn in _panels])), 4)
    # The same reference fitted on the swapped panels: each patient's features are the donor's panel
    # that the swap arm showed the model (the archived 'src' of every swap record). A supervised
    # learner given a stranger's panel should fall to chance against this patient's label, which is
    # what shows that the swap arm can detect patient-specific information when it is present.
    _src = [raw.get(f'swap|{int(p)}', {}).get('src') for p in keep]
    if all(v is not None for v in _src):
        Fs = X[np.ix_([int(v) for v in _src], PANEL)]
        oofs = np.zeros(len(Y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=2024).split(Fs, Y):
            oofs[te] = pipe.fit(Fs[tr], Y[tr]).predict_proba(Fs[te])[:, 1]
        M['logistic_auroc_swapped'] = round(float(roc_auc_score(Y, oofs)), 4)
        M['logistic_auroc_swapped_ci'] = list(boot_auc(Y, oofs))
        # The swap diagnostic that matches the model's: the reference fitted on the patients' own
        # training panels and applied, frozen, to the held-out patients' own panels (oof above) and to
        # the donor panels the swap arm showed the model. The same fixed scorer sees both arms.
        ooff = np.zeros(len(Y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=2024).split(F, Y):
            ooff[te] = pipe.fit(F[tr], Y[tr]).predict_proba(Fs[te])[:, 1]
        M['logistic_auroc_frozen_swap'] = round(float(roc_auc_score(Y, ooff)), 4)
        M['logistic_auroc_frozen_swap_ci'] = list(boot_auc(Y, ooff))

    # The own-versus-swap comparison itself: a patient-level paired bootstrap of the AUROC difference,
    # a two-sided bootstrap p, and the same two equivalence margins used for the reaction-activity designs
    # (a difference is declared equivalent to zero at a margin when the 90 % interval lies inside it).
    _own = np.array([raw.get(f'own|{int(p)}', {}).get('val') for p in keep], dtype=object)
    _swp = np.array([raw.get(f'swap|{int(p)}', {}).get('val') for p in keep], dtype=object)
    _ok = np.array([a is not None and b is not None for a, b in zip(_own, _swp)])
    _o = np.array([float(v) for v in _own[_ok]]); _s = np.array([float(v) for v in _swp[_ok]]); _y = Y[_ok]
    _rng = np.random.default_rng(11); _d = []
    for _ in range(4000):
        ix = _rng.integers(0, len(_y), len(_y))
        if len(set(_y[ix])) > 1: _d.append(roc_auc_score(_y[ix], _o[ix]) - roc_auc_score(_y[ix], _s[ix]))
    _d = np.array(_d); _obs = float(roc_auc_score(_y, _o) - roc_auc_score(_y, _s))
    _p_two = float(2 * min((_d <= 0).mean(), (_d >= 0).mean()))
    _lo90, _hi90 = float(np.percentile(_d, 5)), float(np.percentile(_d, 95))
    M['own_vs_swap'] = dict(n=int(_ok.sum()), auroc_diff=round(_obs, 4),
                            ci95=[round(float(np.percentile(_d, 2.5)), 4), round(float(np.percentile(_d, 97.5)), 4)],
                            ci90=[round(_lo90, 4), round(_hi90, 4)], boot_p=round(_p_two, 4),
                            equivalent_005=bool(_lo90 > -0.05 and _hi90 < 0.05),
                            equivalent_002=bool(_lo90 > -0.02 and _hi90 < 0.02),
                            note='patient-level paired bootstrap, 4000 resamples; equivalence read from the 90 % interval')
    uni = [roc_auc_score(Y, F[:, j]) for j in range(F.shape[1])]
    jb = int(np.argmax([max(u, 1 - u) for u in uni]))
    M['best_univariate_auroc'] = round(float(max(max(uni), 1 - min(uni))), 4)
    M['best_univariate_id'] = des['panel_ids'][jb]
    M['best_univariate_direction'] = 'high expression predicts MSI-H' if uni[jb] >= 0.5 else 'low expression predicts MSI-H'
    M['best_univariate_note'] = 'computed in sample over the whole cohort, therefore optimistic'
    # the same single-feature rule chosen inside each training fold and scored out of fold
    oof1 = np.zeros(len(Y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=2024).split(F, Y):
        u = [roc_auc_score(Y[tr], F[tr, j]) for j in range(F.shape[1])]
        j = int(np.argmax([max(x, 1 - x) for x in u])); sign = 1.0 if u[j] >= 0.5 else -1.0
        oof1[te] = sign * F[te, j]
    M['best_univariate_auroc_oof'] = round(float(roc_auc_score(Y, oof1)), 4)
    # When the same reaction with the same sign is chosen in every training fold, the out-of-fold score
    # vector is the feature itself and its AUROC coincides with the in-sample value by construction; the
    # out-of-fold value then certifies the stability of the choice, not a different accuracy.
    _chosen = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=2024).split(F, Y):
        u = [roc_auc_score(Y[tr], F[tr, j]) for j in range(F.shape[1])]
        j = int(np.argmax([max(x, 1 - x) for x in u])); _chosen.append((des['panel_ids'][j], 1 if u[j] >= 0.5 else -1))
    M['best_univariate_folds_agree'] = bool(len(set(_chosen)) == 1)
    M['best_univariate_fold_choices'] = [f'{a}:{b:+d}' for a, b in _chosen]

    # aggregation control
    cp = [int(p) for p in des['count_patients']]
    # The answer key is what the table printed: the cohort percentile of each value, rounded to the
    # integer the prompt shows, at or above 75. (The collection script also recorded the 75th-percentile
    # expression values, whose criterion differs from the printed one on a few patients; it is not used.)
    _order = np.argsort(X, axis=0); _ranks = np.empty_like(_order)
    for _j in range(X.shape[1]): _ranks[_order[:, _j], _j] = np.arange(X.shape[0])
    _PCT = 100.0 * _ranks / (X.shape[0] - 1)
    truth = {p: int((np.round(_PCT[p, PANEL]) >= 75).sum()) for p in cp}
    pred = [(raw[f'count|{p}']['val'], truth[p]) for p in cp
            if f'count|{p}' in raw and raw[f'count|{p}']['val'] is not None]
    if pred:
        a = np.array([x for x, _ in pred]); b = np.array([t for _, t in pred])
        M['arms']['count'] = dict(n=len(pred), exact=round(float((a == b).mean()), 4),
                                  within_two=round(float((np.abs(a - b) <= 2).mean()), 4),
                                  mae=round(float(np.abs(a - b).mean()), 4),
                                  spearman=round(float(st.spearmanr(a, b).statistic), 4),
                                  truth_mean=round(float(b.mean()), 2), pred_mean=round(float(a.mean()), 2),
                                  truth_sd=round(float(b.std()), 2), pred_sd=round(float(a.std()), 2))


    M['run'] = run; M['model'] = des.get('model')
    return M

R = json.load(open(PATHS.data('results_frozen.json')))
def _has(run):
    try: PATHS.run(run, 'raw.json'); return True
    except FileNotFoundError: return False
# The MSI probe was collected up to three times per backbone: first with a 300-character archive of
# each reply (msi, msi_b2, msi_b3), then with the complete reply archived (msi_il, msi_b2il, msi_b3il),
# both with the own arm sent before the stranger's arm, and then with the two arms interleaved in one
# seeded random order and the complete archive (msi_int, msi_b2int, msi_b3int). The latest collection
# that exists is primary and takes the backbone's key; the earlier ones keep their own keys
# (<key>_blocked for the complete-archive blocked collection, <key>_first for the prefix-archive one).
R['msi_collections'] = {}
def _fold(interleaved, blocked, first, key):
    chain = [(interleaved, 'interleaved'), (blocked, 'blocked'), (first, 'first')]
    have = [(r, kind) for r, kind in chain if _has(r)]
    for k in (key, key + '_blocked', key + '_first'): R.pop(k, None); R['msi_collections'].pop(k, None)
    if not have: return
    r0, kind0 = have[0]
    R[key] = analyze(r0); R[key]['collection_kind'] = kind0; R['msi_collections'][key] = r0
    print(f'ok {r0} -> {key} ({kind0} collection)')
    for r, kind in have[1:]:
        kk = key + '_' + kind
        R[kk] = analyze(r); R[kk]['collection_kind'] = kind; R['msi_collections'][kk] = r
        print(f'ok {r} -> {kk}')
_fold('msi_int', 'msi_il', 'msi', 'msi')
_fold('msi_b2int', 'msi_b2il', 'msi_b2', 'msi_backbone2')
_fold('msi_b3int', 'msi_b3il', 'msi_b3', 'msi_backbone3')
# the interleaved collection is primary on every backbone that has the probe at all
_keys = [k for k in ('msi', 'msi_backbone2', 'msi_backbone3') if k in R]
R['msi_interleaved'] = bool(_keys) and all(R[k].get('collection_kind') == 'interleaved' for k in _keys)
M = R['msi']
json.dump(R, open(PATHS.out('results_frozen.json'), 'w'), indent=1)
print(json.dumps({k: v for k, v in M.items() if k != 'panel_ids'}, indent=1))
