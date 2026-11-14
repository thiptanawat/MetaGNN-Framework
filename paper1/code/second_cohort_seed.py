#!/usr/bin/env python3
"""Outside the plan: the training seed on the second cohort, at one cell.

Section 6 of paper1/SECOND_COHORT_PLAN_2026-11-13.md fixes the default training seed for the graph scorer on
the second cohort. On the first cohort the family-disjoint split was also rerun at two further training
seeds on cell p0r0, with the folds fixed (code/gpu_queue8.sh; data/results_p1.json, robustness.fam.seed_replicates).
This script reads the same check on the second cohort: the same cell, the same two seeds, both arms,
everything else as in Section 6 (code/gpu_queue_cohort2_seed.sh). The check was made after the outcome of
Section 6 was known. It is no part of the plan and enters none of its quantities. This script was written
before any of its runs had finished.

For each cohort and each seed it gives the two arms at the cell and their difference, own minus cohort
mean, AUROC on held-out reactions of held-out patients; then the spread of the difference over the seeds
(largest minus smallest) and whether all differences carry one sign. One cell is one cell: the figures
describe how much a single cell moves with the seed and say nothing about the mean over the grid.

The first cohort's differences are recomputed from its cell files and must equal those of
data/results_p1.json; the script stops if they do not.

Reads results/cohort2/fam/ and results/fam/; writes data/results_cohort2_seed.json.
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
DIRS = dict(cohort1=os.path.join(ROOT, 'results', 'fam'), cohort2=os.path.join(ROOT, 'results', 'cohort2', 'fam'))
CELL = (0, 0); SEEDS = (2025, 2026); DEFAULT_SEED = 2024
R10 = lambda x: round(float(x), 10)


def cell(directory, mode, seed):
    """The result record of one arm of the cell at one training seed; the default seed's file carries no seed tag."""
    name = 'gnn_B_mb0.0_p%d_r%d' % CELL + ('' if mode == 'real' else '_' + mode) + ('' if seed == DEFAULT_SEED else '_ts%d' % seed) + '_ivr_fam.json'
    path = os.path.join(directory, name)
    if not os.path.exists(path): return None
    r = json.load(open(path))
    ok = (r.get('feature_mode') == mode and r.get('inner_val_rxn') == 0.2 and r.get('family_tag') == 'fam' and r.get('model') == 'gnn_B'
          and r.get('lambda_mb') == 0.0 and r.get('train_seed', DEFAULT_SEED) == seed and (r.get('pfold'), r.get('rfold')) == CELL)
    if not ok: sys.exit(f'{os.path.relpath(path, ROOT)}: not the cell, arm and seed its name says')
    return r


def summarize(name):
    out = dict(directory=os.path.relpath(DIRS[name], ROOT), by_seed={})
    for seed in (DEFAULT_SEED,) + SEEDS:
        own, mean = cell(DIRS[name], 'real', seed), cell(DIRS[name], 'mean', seed)
        if own is None or mean is None:
            if seed == DEFAULT_SEED: sys.exit(f'{name}: the default-seed files of the cell are missing')
            continue
        a, b = own['heldout_AUROC_perpatient'], mean['heldout_AUROC_perpatient']
        out['by_seed'][str(seed)] = dict(own=R10(a), cohort_mean=R10(b), difference=R10(a - b),
                                         epochs_own=int(own['epochs']), epochs_cohort_mean=int(mean['epochs']),
                                         n_train=int(own['n_train']), n_test=int(own['n_test']))
    d = [v['difference'] for v in out['by_seed'].values()]
    out.update(seeds=[int(s) for s in out['by_seed']], differences=d, min=R10(min(d)), max=R10(max(d)), spread=R10(max(d) - min(d)),
               one_sign=bool(all(x > 0 for x in d) or all(x < 0 for x in d)),
               own_min=R10(min(v['own'] for v in out['by_seed'].values())), own_max=R10(max(v['own'] for v in out['by_seed'].values())),
               cohort_mean_min=R10(min(v['cohort_mean'] for v in out['by_seed'].values())),
               cohort_mean_max=R10(max(v['cohort_mean'] for v in out['by_seed'].values())))
    return out


def main():
    OUT = dict(note='outside SECOND_COHORT_PLAN_2026-11-13.md: cell p%dr%d of the family-disjoint split under the matched selection rule, '
                    'retrained at further training seeds with the folds fixed, own-expression arm against cohort-mean arm; AUROC on '
                    'held-out reactions of held-out patients; made on the second cohort after the outcome of Section 6 was known' % CELL,
               cell='p%dr%d' % CELL, default_seed=DEFAULT_SEED, further_seeds=list(SEEDS), cohorts={})
    for name in ('cohort1', 'cohort2'):
        OUT['cohorts'][name] = summarize(name)
    if len(OUT['cohorts']['cohort2']['seeds']) < 2:
        sys.exit('no seed replicate of the cell under results/cohort2/fam (files tagged _ts<seed>)')
    # the first cohort's differences as code/build_results_p1.py stores them
    try:
        ref = json.load(open(os.path.join(ROOT, 'data', 'results_p1.json')))['robustness']['fam']['seed_replicates']
    except (FileNotFoundError, KeyError):
        ref = None
    if ref is not None:
        c1 = OUT['cohorts']['cohort1']['by_seed']
        want = {str(DEFAULT_SEED): ref['default_seed']['delta'], **{s: v['delta'] for s, v in ref['by_seed'].items()}}
        if set(want) != set(c1) or any(abs(want[s] - c1[s]['difference']) > 1e-9 for s in want):
            sys.exit('the first cohort\'s seed replicates differ from data/results_p1.json (robustness.fam.seed_replicates)')
        OUT['first_cohort_check'] = 'equal to data/results_p1.json, robustness.fam.seed_replicates'
    rec = os.path.join(ROOT, 'data', 'cohort2_seed_run_record.json')      # where and when the four runs were made
    if os.path.exists(rec): OUT['run_record'] = json.load(open(rec))
    json.dump(OUT, open(os.path.join(ROOT, 'data', 'results_cohort2_seed.json'), 'w'), indent=1)
    for name in ('cohort2', 'cohort1'):
        c = OUT['cohorts'][name]
        print(name, 'cell', OUT['cell'], 'own minus cohort mean by seed:', {s: v['difference'] for s, v in c['by_seed'].items()},
              'spread', c['spread'], 'one sign:', c['one_sign'])


if __name__ == '__main__':
    main()
