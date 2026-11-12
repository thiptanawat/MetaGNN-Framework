#!/usr/bin/env python3
"""Analyses of the own-versus-donor contrast that the plan of 13 November 2026 did not specify.

They were added after the prespecified statistics (replication_stats.py) had been read, and the
manuscript reports them as exploratory. Nothing here changes results/replication_frozen.json.

The question. The donor arm prints, for every reaction, the value and the cohort percentile of
another patient, redrawn at every reaction, so a donor prompt set is a mosaic of patients while an
own prompt set is one patient's profile. A scorer that only ranks the printed numbers can therefore
score differently in the two arms without knowing anything about the patient. This script measures
that difference with no model:

1. printed value: the per-patient AUROC of the printed expression value (three decimals, as the
   prompt prints it) against the label, for the patient's own values and for the values each donor
   schedule prints, with the paired own-minus-donor difference. A reaction that is shown no value
   keeps zero, as in stats.analyze;
2. printed percentile: the same for the printed cohort percentile (an integer, as printed);
3. the language model's own-minus-donor difference beside those two, the per-patient association
   between them, and the paired difference between the model's contrast and that of the printed value.

The same three quantities are computed for the earlier primary design (Design E) from its archive.
Writes results/replication_posthoc.json.
"""
import os, sys, json
import numpy as np
from scipy import stats as st

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import stats as S
import _paths as PATHS

R10 = lambda x: None if x is None or not np.isfinite(x) else round(float(x), 10)


def paired(dv):
    """Mean paired difference over patients, its standard deviation, the 90% Student interval and the
    two-sided paired t test, as replication_stats.paired reports them."""
    dv = np.asarray(dv, float); dv = dv[np.isfinite(dv)]; n = len(dv)
    m = float(dv.mean()); sd = float(dv.std(ddof=1)); se = sd / np.sqrt(n)
    h = float(st.t.ppf(0.95, n - 1)) * se
    p = float(2 * st.t.sf(abs(m / se), n - 1)) if se > 0 else float('nan')
    return dict(n=int(n), mean=R10(m), sd=R10(sd), ci90=[R10(m - h), R10(m + h)], paired_p=R10(p),
                within_002=bool(m - h > -0.02 and m + h < 0.02), excludes_zero=bool(m - h > 0 or m + h < 0))


def assoc(a, b):
    """Pearson and Spearman correlation over patients, with the 90% interval of the Pearson
    coefficient by the Fisher transformation."""
    ok = np.isfinite(a) & np.isfinite(b); a, b = a[ok], b[ok]; n = len(a)
    r = float(np.corrcoef(a, b)[0, 1]); z = np.arctanh(r); h = 1.6448536269514722 / np.sqrt(n - 3)
    return dict(n=int(n), pearson=R10(r), pearson_ci90=[R10(np.tanh(z - h)), R10(np.tanh(z + h))],
                spearman=R10(float(st.spearmanr(a, b).statistic)))


def model_matrix(raw, arm, patients, reactions):
    M = np.full((len(patients), len(reactions)), np.nan)
    for a, pj in enumerate(patients):
        for b, ri in enumerate(reactions):
            v = raw.get(f'{arm}|{pj}|{ri}')
            if v and v['prob'] is not None: M[a, b] = v['prob']
    return M


def contrast(own, donor):
    """Per-patient AUROC of two score matrices and their paired difference."""
    return dict(own_auroc=R10(np.nanmean(own)), donor_auroc=R10(np.nanmean(donor)), diff=paired(own - donor))


def one_design(patients, reactions, donors, raw, arms):
    """donors: schedule -> array (patient position, reaction position) of donor patient indices;
    arms: schedule -> key of the donor arm in the archive."""
    patients = np.asarray(patients); reactions = np.asarray(reactions)
    y = S.LAB[reactions].astype(int); hx = S.HAS[reactions]
    Xo = np.round(S.X[np.ix_(patients, reactions)].astype(float), 3)
    Po = np.round(S.PCT[np.ix_(patients, reactions)].astype(float)); Po[:, ~hx] = 0.0
    a_val_own = S.auc_matrix(Xo, y); a_pct_own = S.auc_matrix(Po, y)
    a_llm_own = S.auc_matrix(model_matrix(raw, 'patient', patients, reactions), y)
    out = dict(n_patients=int(len(patients)), n_reactions=int(len(reactions)), n_expression_bearing=int(hx.sum()),
               printed_value={}, printed_percentile={}, language_model={}, association={}, model_minus_printed_value={},
               model_minus_printed_percentile={})
    d_val, d_pct, d_llm = {}, {}, {}
    for s, D in donors.items():
        D = np.asarray(D)
        Xd = np.round(S.X[D, reactions[None, :].repeat(len(patients), 0)].astype(float), 3)
        Pd = np.round(S.PCT[D, reactions[None, :].repeat(len(patients), 0)].astype(float)); Pd[:, ~hx] = 0.0
        a_val = S.auc_matrix(Xd, y); a_pct = S.auc_matrix(Pd, y)
        a_llm = S.auc_matrix(model_matrix(raw, arms[s], patients, reactions), y)
        out['printed_value'][s] = contrast(a_val_own, a_val); out['printed_percentile'][s] = contrast(a_pct_own, a_pct)
        out['language_model'][s] = contrast(a_llm_own, a_llm)
        d_val[s], d_pct[s], d_llm[s] = a_val_own - a_val, a_pct_own - a_pct, a_llm_own - a_llm
        out['association'][s] = dict(model_with_printed_value=assoc(d_llm[s], d_val[s]),
                                     model_with_printed_percentile=assoc(d_llm[s], d_pct[s]))
        out['model_minus_printed_value'][s] = paired(d_llm[s] - d_val[s])
        out['model_minus_printed_percentile'][s] = paired(d_llm[s] - d_pct[s])
    if len(donors) > 1:
        k = list(donors)
        out['printed_value']['mean_of_schedules'] = paired(np.mean([d_val[s] for s in k], axis=0))
        out['printed_percentile']['mean_of_schedules'] = paired(np.mean([d_pct[s] for s in k], axis=0))
        out['language_model']['mean_of_schedules'] = paired(np.mean([d_llm[s] for s in k], axis=0))
    return out


def main():
    OUT = dict(note='exploratory analyses of the own-versus-donor contrast, not specified in REPLICATION_PLAN_2026-11-13.md and '
                    'added after the prespecified statistics had been read: the contrast of the printed value and of the '
                    'printed percentile with no model, beside the contrast of the language model')
    # the replication, first wording
    raw = json.load(open(PATHS.run('replication/w1', 'raw.json'))); des = json.load(open(PATHS.run('replication/w1', 'design.json')))
    master = json.load(open(PATHS.run('replication', 'design_master.json')))
    pos = {p: i for i, p in enumerate(master['patients'])}; assert master['reactions'] == des['reactions']
    rows = [pos[p] for p in des['patients']]
    donors = {s: np.asarray(master['donors'][s])[rows] for s in ('d1', 'd2', 'd3')}
    # the archive's own record of the first schedule must agree with the master file
    chk = np.array([[des['swap_source'][f'{p}|{r}'] for r in des['reactions']] for p in des['patients']])
    assert (chk == donors['d1']).all(), 'donor schedule d1 of the archive differs from design_master.json'
    OUT['replication'] = one_design(des['patients'], des['reactions'], donors, raw,
                                    dict(d1='shuffled', d2='shuffled_d2', d3='shuffled_d3'))
    # the earlier primary design (Design E), from its archive
    for cand in ('repr40_il', 'repr40'):      # the interleaved collection is the primary one
        try:
            rawE = json.load(open(PATHS.run(cand, 'raw.json'))); desE = json.load(open(PATHS.run(cand, 'design.json')))
        except FileNotFoundError:
            continue
        if 'swap_source' not in desE: continue
        dE = np.array([[desE['swap_source'][f'{p}|{r}'] for r in desE['reactions']] for p in desE['patients']])
        OUT['design_E'] = dict(run=cand, **one_design(desE['patients'], desE['reactions'], dict(d1=dE), rawE, dict(d1='shuffled')))
        break
    p = os.path.join(PATHS.ROOT, 'results', 'replication_posthoc.json')
    json.dump(OUT, open(p, 'w'), indent=1); print('wrote', p)
    for name in ('replication', 'design_E'):
        if name not in OUT: continue
        G = OUT[name]; print(name, G['n_patients'], 'patients')
        for kind in ('language_model', 'printed_value', 'printed_percentile'):
            for s, v in G[kind].items():
                d = v['diff'] if 'diff' in v else v
                print(f"  {kind:20s} {s:18s} own {v.get('own_auroc', float('nan')) or float('nan'):.4f} donor {v.get('donor_auroc', float('nan')) or float('nan'):.4f} "
                      f"diff {d['mean']:+.4f} [{d['ci90'][0]:+.4f}, {d['ci90'][1]:+.4f}] p {d['paired_p']:.4f}")
        for s, v in G['association'].items():
            print(f"  association {s}: value r {v['model_with_printed_value']['pearson']:+.3f} {v['model_with_printed_value']['pearson_ci90']}, "
                  f"percentile r {v['model_with_printed_percentile']['pearson']:+.3f}")
        for s, v in G['model_minus_printed_value'].items():
            print(f"  model minus printed value {s}: {v['mean']:+.4f} [{v['ci90'][0]:+.4f}, {v['ci90'][1]:+.4f}]")
        for s, v in G['model_minus_printed_percentile'].items():
            print(f"  model minus printed percentile {s}: {v['mean']:+.4f} [{v['ci90'][0]:+.4f}, {v['ci90'][1]:+.4f}]")


if __name__ == '__main__':
    main()
