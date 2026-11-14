#!/usr/bin/env python3
"""The graph scorer on the second cohort at a second training seed, on the full grid.

Plan: Section 7 of paper1/SECOND_COHORT_PLAN_2026-11-13.md, fixed with this script before any of its runs
except those of cell p0r0, which the check of 14 November had already made (code/second_cohort_seed.py).
The cells are the ones code/double_holdout_ctrl.py writes with --train_seed 2025 under the settings of
Section 6 (code/gpu_queue_cohort2_seed2025.sh): files tagged _ts2025 under results/cohort2/fam/.

Primary quantity, as in Section 6: the paired difference over the 15 cells between the own-expression arm
and the cohort-mean arm at the second seed, AUROC on held-out reactions of held-out patients, with its
mean, standard deviation over cells, range and number of negative cells. A mean within 0.02 AUROC of zero
is read as agreement with the first cohort. The exact sign-flip p values are given with single cells
flipped, with whole patient folds flipped and with whole reaction folds flipped, as descriptions.

Beside it, also fixed in Section 7 and descriptive:
  - the same summary for the per-cell mean of the two seeds' differences (seeds 2024 and 2025);
  - how far the two seeds agree: the difference between their grid means, the mean absolute change of a
    cell's difference between the seeds, the number of cells whose difference keeps its sign, and the
    Pearson and Spearman correlations of the two seeds' differences across the 15 cells;
  - training length, at each seed: the epochs of the cohort-mean arm minus those of the own-expression arm
    cell by cell, the correlation of the difference in AUROC with that gap across the cells, and the mean
    difference over the cells whose two arms stopped within two epochs of each other.

Reads results/cohort2/fam/ (the default-seed files of Section 6 and the files tagged _ts2025); writes
data/results_cohort2_graph_seed.json. It stops if a cell of the second seed is missing.
"""
import glob, json, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import second_cohort_graph as SG          # the grid, the margin and the sign-flip tests of Section 6

DIR = os.path.join(ROOT, 'results', 'cohort2', 'fam')
SEED2 = 2025
R10 = SG.R10; GRID = SG.GRID


def cells_at(mode, seed):
    """Cell -> the result record of one arm at one training seed (the default seed's files carry no seed tag)."""
    tag = '' if seed == SG.DEFAULT_SEED else '_ts%d' % seed
    pat = re.compile(r'^gnn_B_mb0\.0_p(?P<p>\d+)_r(?P<r>\d+)' + ('' if mode == 'real' else '_' + mode) + re.escape(tag) + r'_ivr_fam\.json$')
    out = {}
    for f in glob.glob(os.path.join(DIR, '*.json')):
        m = pat.match(os.path.basename(f))
        if not m: continue
        r = json.load(open(f)); c = 'p%sr%s' % (m.group('p'), m.group('r'))
        ok = (r.get('feature_mode') == mode and r.get('inner_val_rxn') == 0.2 and r.get('family_tag') == 'fam' and r.get('model') == 'gnn_B'
              and r.get('lambda_mb') == 0.0 and r.get('train_seed', SG.DEFAULT_SEED) == seed)
        if not ok: sys.exit(f'{os.path.basename(f)}: not the arm and seed its name says')
        out[c] = r
    return out


def describe(d):
    d = np.asarray(d, float)
    out = dict(per_cell={c: R10(x) for c, x in zip(GRID, d)}, mean=R10(d.mean()), sd=R10(d.std(ddof=1)), min=R10(d.min()), max=R10(d.max()),
               n_cells=len(GRID), n_negative=int((d < 0).sum()), n_positive=int((d > 0).sum()), signflip_p=R10(SG.signflip_p(d)),
               within_margin=bool(abs(d.mean()) < SG.MARGIN))
    for which, key in (('p', 'patient_folds'), ('r', 'reaction_folds')):
        pv, floor = SG.block_signflip_p(d, which)
        out['signflip_p_' + key] = R10(pv); out['signflip_floor_' + key] = R10(floor)
    return out


def one_seed(seed):
    arm = {k: cells_at(mode, seed) for k, mode in SG.ARMS.items()}
    missing = {k: [c for c in GRID if c not in arm[k]] for k in SG.ARMS}
    if any(missing.values()):
        sys.exit(f'training seed {seed}: {sum(len(v) for v in missing.values())} cell files are missing under results/cohort2/fam: {missing}')
    out = dict(train_seed=seed, arms={})
    for k in SG.ARMS:
        recs = [arm[k][c] for c in GRID]; a = np.array([r['heldout_AUROC_perpatient'] for r in recs])
        out['arms'][k] = dict(per_cell={c: R10(x) for c, x in zip(GRID, a)}, mean=R10(a.mean()), sd=R10(a.std(ddof=1)),
                              epochs=dict(min=int(min(r['epochs'] for r in recs)), max=int(max(r['epochs'] for r in recs)),
                                          median=float(np.median([r['epochs'] for r in recs]))))
    d = np.array([arm['own'][c]['heldout_AUROC_perpatient'] - arm['cohort_mean'][c]['heldout_AUROC_perpatient'] for c in GRID])
    out['contrast'] = describe(d)
    # training length: epochs of the cohort-mean arm minus epochs of the own-expression arm, cell by cell, and how the
    # difference in AUROC goes with it across the cells (Section 6 left open whether the longer training of the
    # cohort-mean arm accounts for the difference between the arms)
    gap = np.array([arm['cohort_mean'][c]['epochs'] - arm['own'][c]['epochs'] for c in GRID], float)
    out['training_length'] = dict(epoch_gap_per_cell={c: int(x) for c, x in zip(GRID, gap)}, mean_epoch_gap=R10(gap.mean()),
                                  pearson_difference_with_epoch_gap=(R10(np.corrcoef(d, gap)[0, 1]) if gap.std() > 0 else None),
                                  spearman_difference_with_epoch_gap=(R10(np.corrcoef(ranks(d), ranks(gap))[0, 1]) if gap.std() > 0 else None),
                                  cells_with_equal_length=int((gap == 0).sum()),
                                  mean_difference_where_gap_at_most_2=(R10(d[np.abs(gap) <= 2].mean()) if (np.abs(gap) <= 2).any() else None),
                                  n_cells_where_gap_at_most_2=int((np.abs(gap) <= 2).sum()))
    return out, d


def ranks(x):
    """Average ranks (ties share the mean rank), for the Spearman correlation."""
    x = np.asarray(x, float); order = np.argsort(x, kind='mergesort'); r = np.empty(len(x)); i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]: j += 1
        r[order[i:j + 1]] = (i + j) / 2.0 + 1; i = j + 1
    return r


def main():
    S1, d1 = one_seed(SG.DEFAULT_SEED); S2, d2 = one_seed(SEED2)
    OUT = dict(note='the graph scorer on the second cohort under the family-disjoint split and the matched selection rule, own-expression arm '
                    'against cohort-mean arm on the 15 cells of the 5 x 3 grid, at the training seed of Section 6 and at a second training seed; '
                    'AUROC on held-out reactions of held-out patients; Section 7 of SECOND_COHORT_PLAN_2026-11-13.md',
               margin=SG.MARGIN, primary='paired difference over cells, own minus cohort mean, at the second training seed',
               seeds={str(SG.DEFAULT_SEED): S1, str(SEED2): S2},
               both_seeds=dict(what='per cell, the mean of the two seeds\' differences', **describe((d1 + d2) / 2.0)),
               between_seeds=dict(grid_mean_second_minus_first=R10(d2.mean() - d1.mean()),
                                  mean_abs_change_of_a_cell=R10(np.abs(d2 - d1).mean()), max_abs_change_of_a_cell=R10(np.abs(d2 - d1).max()),
                                  cells_with_one_sign=int(((d1 < 0) == (d2 < 0)).sum()), n_cells=len(GRID),
                                  pearson=R10(np.corrcoef(d1, d2)[0, 1]), spearman=R10(np.corrcoef(ranks(d1), ranks(d2))[0, 1])))
    c = S2['contrast']
    OUT['reading'] = dict(mean=c['mean'], sd=c['sd'], n_negative=c['n_negative'], n_cells=c['n_cells'], within_margin=c['within_margin'],
                          agreement='the mean lies within %.2f AUROC of zero, which the plan reads as agreement with the first cohort' % SG.MARGIN
                          if c['within_margin'] else
                          'the mean does not lie within %.2f AUROC of zero, which the plan does not read as agreement with the first cohort' % SG.MARGIN)
    # the default seed's contrast here must be the one of Section 6
    ref = os.path.join(ROOT, 'data', 'results_cohort2_graph.json')
    if os.path.exists(ref):
        r6 = json.load(open(ref))['cohorts']['cohort2']['contrast']
        if abs(r6['mean'] - S1['contrast']['mean']) > 1e-9 or r6['n_negative'] != S1['contrast']['n_negative']:
            sys.exit('the default-seed contrast differs from data/results_cohort2_graph.json')
        OUT['default_seed_check'] = 'equal to data/results_cohort2_graph.json (mean, negative cells)'
    rec = os.path.join(ROOT, 'data', 'cohort2_graph_seed_run_record.json')     # where and when the cells of the second seed were computed
    if os.path.exists(rec): OUT['run_record'] = json.load(open(rec))
    json.dump(OUT, open(os.path.join(ROOT, 'data', 'results_cohort2_graph_seed.json'), 'w'), indent=1)
    b = OUT['both_seeds']; bs = OUT['between_seeds']
    print(f"second cohort, own minus cohort mean over cells: seed {SG.DEFAULT_SEED} {S1['contrast']['mean']:+.4f} (negative {S1['contrast']['n_negative']}), "
          f"seed {SEED2} {c['mean']:+.4f} sd {c['sd']:.4f} (negative {c['n_negative']}), both {b['mean']:+.4f} sd {b['sd']:.4f} (negative {b['n_negative']}); "
          f"between seeds: pearson {bs['pearson']:+.2f}, cells with one sign {bs['cells_with_one_sign']} of {bs['n_cells']}")


if __name__ == '__main__':
    main()
