#!/usr/bin/env python3
"""The graph scorer on the second cohort, beside the first cohort's.

Plan: Section 6 of paper1/SECOND_COHORT_PLAN_2026-11-13.md, fixed before any such run. The cells are the
ones code/double_holdout_ctrl.py wrote under the family-disjoint reaction split, the matched selection
rule (20% of each cell's training reactions withheld from the loss and used for stopping) and the default
training seed: results/cohort2/fam/ for the second cohort, results/fam/ for the first.

For each cohort the primary quantity is the paired difference over the 15 cells of the 5 x 3 grid between
the own-expression arm and the cohort-mean arm, AUROC on held-out reactions of held-out patients. The
reading is the one the plan fixed: the mean, the standard deviation over cells, the range and the number
of negative cells; a mean within 0.02 AUROC of zero is read as agreement with the first cohort. The exact
sign-flip p value over cells is given beside them, as for the linear rows. Cells share patients and
reactions and are not independent, so all of this is descriptive.

The cells are crossed, so the sign-flip test is also given with the signs flipped for a whole patient fold
at a time and for a whole reaction fold at a time, as code/inference_p1.py gives it for the first cohort;
with five and three folds these tests cannot fall below 0.0625 and 0.25. The two fold-level values were
added to this script on 14 November 2026, after the outcome of Section 6 had been read, so that the second
cohort is described as the first is. They change no other quantity.

The first cohort's contrast is recomputed here from its cell files and must equal the one
code/inference_p1.py stores (results/inference.json, fam_real_vs_mean); the script stops if it does not.

Writes data/results_cohort2_graph.json.
"""
import glob, itertools, json, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
DIRS = dict(cohort1=os.path.join(ROOT, 'results', 'fam'), cohort2=os.path.join(ROOT, 'results', 'cohort2', 'fam'))
ARMS = dict(own='real', cohort_mean='mean')
GRID = [f'p{p}r{r}' for p in range(5) for r in range(3)]
DEFAULT_SEED = 2024
MARGIN = 0.02
R10 = lambda x: round(float(x), 10)


def cells(directory, mode):
    """Cell -> the result record of one arm, default training seed only (files of the seed replicates
    carry a _ts<seed> tag and do not match)."""
    pat = re.compile(r'^gnn_B_mb0\.0_p(?P<p>\d+)_r(?P<r>\d+)' + ('' if mode == 'real' else '_' + mode) + r'_ivr_fam\.json$')
    out = {}
    for f in glob.glob(os.path.join(directory, '*.json')):
        m = pat.match(os.path.basename(f))
        if m: out['p%sr%s' % (m.group('p'), m.group('r'))] = json.load(open(f))
    return out


def signflip_p(d):
    """Exact two-sided sign-flip p value of the mean over cells (as code/second_cohort_stats.py)."""
    d = np.asarray(d, float); obs = abs(d.mean()); n = len(d); hits = 0
    for s in itertools.product((1.0, -1.0), repeat=n):
        hits += abs((d * np.asarray(s)).mean()) >= obs - 1e-15
    return hits / 2 ** n


def block_signflip_p(d, which):
    """Exact two-sided sign-flip p value with one sign per patient fold ('p') or per reaction fold ('r')
    (as code/inference_p1.py, block_signflip); returns the p value and the smallest value it can take."""
    lab = [c.split('r')[0] if which == 'p' else 'r' + c.split('r')[1] for c in GRID]
    blocks = sorted(set(lab)); d = np.asarray(d, float); obs = abs(d.mean()); hits = 0
    for s in itertools.product((1.0, -1.0), repeat=len(blocks)):
        sign = dict(zip(blocks, s))
        hits += abs(np.mean([sign[l] * x for l, x in zip(lab, d)])) >= obs - 1e-15
    return hits / 2 ** len(blocks), 2.0 / 2 ** len(blocks)


def summarize(name):
    arm = {k: cells(DIRS[name], mode) for k, mode in ARMS.items()}
    missing = {k: [c for c in GRID if c not in arm[k]] for k in ARMS}
    if any(missing.values()):
        return None, missing
    out = dict(directory=os.path.relpath(DIRS[name], ROOT), cells=GRID, arms={})
    for k, mode in ARMS.items():
        recs = [arm[k][c] for c in GRID]
        # every cell must be what the plan names: this arm, the matched rule, the family map, the default seed
        for c, r in zip(GRID, recs):
            ok = (r.get('feature_mode') == mode and r.get('inner_val_rxn') == 0.2 and r.get('family_tag') == 'fam'
                  and r.get('train_seed', DEFAULT_SEED) == DEFAULT_SEED and r.get('model') == 'gnn_B' and r.get('lambda_mb') == 0.0)
            if not ok: sys.exit(f'{name} {k} {c}: the cell file is not a cell of the planned run')
        a = np.array([r['heldout_AUROC_perpatient'] for r in recs])
        out['arms'][k] = dict(
            feature_mode=mode, per_cell={c: R10(x) for c, x in zip(GRID, a)}, mean=R10(a.mean()), sd=R10(a.std(ddof=1)),
            min=R10(a.min()), max=R10(a.max()),
            epochs=dict(min=int(min(r['epochs'] for r in recs)), max=int(max(r['epochs'] for r in recs)),
                        median=float(np.median([r['epochs'] for r in recs]))),
            n_train=sorted({int(r['n_train']) for r in recs}), n_test=sorted({int(r['n_test']) for r in recs}),
            n_heldout_rxn=sorted({int(r['n_heldout_rxn']) for r in recs}), data=sorted({str(r.get('data')) for r in recs}),
            expr_transform=sorted({str(r.get('expr_transform')) for r in recs}),
            baseline_rawexpr_mean=R10(np.mean([r['baseline_rawexpr'] for r in recs])),
            baseline_indicator_mean=R10(np.mean([r['baseline_indicator'] for r in recs])))
    d = np.array([arm['own'][c]['heldout_AUROC_perpatient'] - arm['cohort_mean'][c]['heldout_AUROC_perpatient'] for c in GRID])
    out['contrast'] = dict(own='own expression', cohort_mean='cohort-mean expression', n_cells=len(GRID),
                           per_cell={c: R10(x) for c, x in zip(GRID, d)}, mean=R10(d.mean()), sd=R10(d.std(ddof=1)),
                           min=R10(d.min()), max=R10(d.max()), n_negative=int((d < 0).sum()), n_positive=int((d > 0).sum()),
                           signflip_p=R10(signflip_p(d)), within_margin=bool(abs(d.mean()) < MARGIN))
    for which, key in (('p', 'patient_folds'), ('r', 'reaction_folds')):
        pv, floor = block_signflip_p(d, which)
        out['contrast']['signflip_p_' + key] = R10(pv); out['contrast']['signflip_floor_' + key] = R10(floor)
    return out, None


def main():
    OUT = dict(note='the graph scorer under the family-disjoint split and the matched selection rule, default training seed, '
                    'own-expression arm against cohort-mean arm on the 15 cells of the 5 x 3 grid; AUROC on held-out reactions '
                    'of held-out patients; Section 6 of SECOND_COHORT_PLAN_2026-11-13.md',
               margin=MARGIN, primary='paired difference over cells, own minus cohort mean, on the second cohort', cohorts={})
    for name in ('cohort1', 'cohort2'):
        S, missing = summarize(name)
        if S is None:
            n = sum(len(v) for v in missing.values())
            sys.exit(f'{name}: {n} cell files are missing under {os.path.relpath(DIRS[name], ROOT)}: {missing}')
        OUT['cohorts'][name] = S
    # the first cohort's contrast as code/inference_p1.py stores it
    try:
        ref = json.load(open(os.path.join(ROOT, 'results', 'inference.json')))['comparisons']['fam_real_vs_mean']
    except (FileNotFoundError, KeyError):
        ref = None
    c1 = OUT['cohorts']['cohort1']['contrast']
    if ref is not None:
        if abs(ref['mean'] - c1['mean']) > 1e-9 or abs(ref['sd'] - c1['sd']) > 1e-9 or ref['n_negative'] != c1['n_negative']:
            sys.exit('the first cohort contrast differs from results/inference.json (fam_real_vs_mean)')
        if (abs(ref['signflip_p'] - c1['signflip_p']) > 1e-9 or abs(ref['block_patient']['p'] - c1['signflip_p_patient_folds']) > 1e-9
                or abs(ref['block_reaction']['p'] - c1['signflip_p_reaction_folds']) > 1e-9):
            sys.exit('the first cohort sign-flip p values differ from results/inference.json (fam_real_vs_mean)')
        OUT['first_cohort_check'] = ('equal to results/inference.json, comparisons.fam_real_vs_mean (mean, standard deviation, negative cells, '
                                     'and the three sign-flip p values)')
    c2 = OUT['cohorts']['cohort2']['contrast']
    OUT['reading'] = dict(mean=c2['mean'], sd=c2['sd'], min=c2['min'], max=c2['max'], n_negative=c2['n_negative'], n_cells=c2['n_cells'],
                          within_margin=c2['within_margin'],
                          agreement='the mean lies within %.2f AUROC of zero, which the plan reads as agreement with the first cohort' % MARGIN
                          if c2['within_margin'] else
                          'the mean does not lie within %.2f AUROC of zero, which the plan does not read as agreement with the first cohort' % MARGIN,
                          second_minus_first=R10(c2['mean'] - c1['mean']))
    rec = os.path.join(ROOT, 'data', 'cohort2_graph_run_record.json')     # where and when the cells were computed
    if os.path.exists(rec): OUT['run_record'] = json.load(open(rec))
    dst = os.path.join(ROOT, 'data', 'results_cohort2_graph.json')
    json.dump(OUT, open(dst, 'w'), indent=1)
    print('wrote data/results_cohort2_graph.json; second cohort, own minus cohort mean over cells:', c2['mean'], 'sd', c2['sd'],
          'range', c2['min'], c2['max'], 'negative cells', c2['n_negative'], 'of', c2['n_cells'], '| first cohort:', c1['mean'], 'sd', c1['sd'])


if __name__ == '__main__':
    main()
