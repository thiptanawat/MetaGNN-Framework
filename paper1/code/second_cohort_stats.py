#!/usr/bin/env python3
"""The linear rungs of the decomposition on a second cohort, beside the first cohort's.

Plan: paper1/SECOND_COHORT_PLAN_2026-11-13.md. The rows are the ones naive_baselines_allfolds.py and
pooled_baselines.py compute on every cell of the 5 x 3 grid, under the tuned regime that the
decomposition table reports, on the random reaction split and on the family-disjoint split. The
second cohort's files are in results/cohort2/ under the names the first cohort's files carry in
results/. Rows that take no patient input (the indicator of a resolved gene rule, topology,
annotation, the network features) do not depend on the cohort: the label vector and the reaction
folds are the same, so those rows are a check that the two runs share them.

A contrast is the paired difference over cells between a row built from each test patient's own
expression and the same row built from the cohort-mean expression. Cells share patients and
reactions and are not independent; the standard deviation, the range, the count of negative cells
and the sign-flip p value over cells are descriptive.

Writes data/results_cohort2.json.
"""
import itertools, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
FILES = dict(random=('naive_allfolds.json', 'naive_pooled.json'), family=('naive_allfolds_fam.json', 'naive_pooled_fam.json'))
ROWS = ['indicator', 'expr_perpat_rank', 'expr_cohortmean_rank', 'expr_perpat_lr', 'expr_cohortmean_lr',
        'topology_only', 'annotation_only', 'structure_only', 'structure_plus_cohortmean', 'structure_plus_perpat',
        'expr_pooled_lr', 'structure_plus_pooled']
CONTRASTS = dict(ranked=('expr_perpat_rank', 'expr_cohortmean_rank'), fitted=('expr_perpat_lr', 'expr_cohortmean_lr'),
                 network=('structure_plus_perpat', 'structure_plus_cohortmean'))
REGIME = 'tuned'
R10 = lambda x: round(float(x), 10)


def load(cohort, split):
    base = os.path.join(ROOT, 'results') if cohort == 'cohort1' else os.path.join(ROOT, 'results', 'cohort2')
    rows, meta = {}, {}
    for fn in FILES[split]:
        d = json.load(open(os.path.join(base, fn)))
        for k, v in d[REGIME].items():
            rows[k] = {c: float(x) for c, x in v.items()}
        for k in ('_n_test_patients', '_n_train_patients_used', '_data', '_family_map', '_expr_transform'):
            if k in d: meta[fn + ':' + k] = d[k]
    return rows, meta


def signflip_p(d):
    """Exact two-sided sign-flip p value of the mean over cells."""
    d = np.asarray(d, float); obs = abs(d.mean()); n = len(d); hits = 0
    for s in itertools.product((1.0, -1.0), repeat=n):
        hits += abs((d * np.asarray(s)).mean()) >= obs - 1e-15
    return hits / 2 ** n


def contrast(rows, a, b):
    cells = sorted(set(rows[a]) & set(rows[b]))
    d = np.array([rows[a][c] - rows[b][c] for c in cells])
    return dict(own=a, cohort_mean=b, n_cells=len(cells), mean=R10(d.mean()), sd=R10(d.std(ddof=1)),
                min=R10(d.min()), max=R10(d.max()), n_negative=int((d < 0).sum()), signflip_p=R10(signflip_p(d)))


OUT = dict(note='linear rungs of the decomposition on the second cohort beside the first; tuned regime; AUROC, mean over cells',
           regime=REGIME, primary='network contrast on the family-disjoint split of the second cohort', cohorts={})
for cohort in ('cohort1', 'cohort2'):
    OUT['cohorts'][cohort] = {}
    for split in ('random', 'family'):
        try:
            rows, meta = load(cohort, split)
        except FileNotFoundError as e:
            sys.exit(f'missing: {e.filename}')
        OUT['cohorts'][cohort][split] = dict(
            rows={r: dict(n_cells=len(rows[r]), mean=R10(np.mean(list(rows[r].values()))),
                          sd=R10(np.std(list(rows[r].values()), ddof=1))) for r in ROWS if r in rows},
            contrasts={k: contrast(rows, a, b) for k, (a, b) in CONTRASTS.items() if a in rows and b in rows},
            meta={k: (v if not isinstance(v, dict) else {kk: v[kk] for kk in sorted(v)[:3]}) for k, v in meta.items()})
# rows without patient input must agree between the cohorts
chk = {}
for split in ('random', 'family'):
    for r in ('topology_only', 'annotation_only', 'structure_only'):
        a = OUT['cohorts']['cohort1'][split]['rows'].get(r); b = OUT['cohorts']['cohort2'][split]['rows'].get(r)
        if a and b: chk[f'{split}:{r}'] = R10(abs(a['mean'] - b['mean']))
OUT['patient_free_rows_abs_difference'] = chk
try:
    OUT['second_cohort_expression_qc'] = json.load(open(os.path.join(ROOT, 'data', 'cohort2_expression_qc.json')))
except FileNotFoundError:
    pass
json.dump(OUT, open(os.path.join(ROOT, 'data', 'results_cohort2.json'), 'w'), indent=1)
p = OUT['cohorts']['cohort2']['family']['contrasts']['network']
print('wrote data/results_cohort2.json; primary (second cohort, family-disjoint, network + own minus network + cohort mean):',
      p['mean'], 'sd', p['sd'], 'range', p['min'], p['max'], 'negative cells', p['n_negative'], 'of', p['n_cells'])
