#!/usr/bin/env python3
"""The graph scorer on the first cohort at a second training seed, on the full grid.

Plan: paper1/FIRST_COHORT_SEED_PLAN_2026-11-15.md, fixed with this script before any of its runs. The cells
are the ones code/double_holdout_ctrl.py writes with --train_seed 2025 under the settings of the primary
grid (code/gpu_queue_cohort1_seed2025.sh), all made on the second workstation: results/fam_seed2025/.

Primary quantity: the paired difference over the 15 cells between the own-expression arm and the
cohort-mean arm at the second seed, AUROC on held-out reactions of held-out patients, with its mean,
standard deviation over cells, range and number of negative cells. A mean within 0.02 AUROC of zero is
read as agreement with the first seed. The exact sign-flip p values are given with single cells flipped,
with whole patient folds flipped and with whole reaction folds flipped, as descriptions.

Beside it, also fixed in the plan and descriptive: the same summary for the per-cell mean of the two
seeds' differences; how far the two seeds agree (difference between the grid means, mean and largest
absolute change of a cell, cells that keep their sign, Pearson and Spearman correlations across the
cells); training length at each seed; and, at cell p0r0 and the second seed, the two arms and their
difference on the two workstations (the first from the one-cell check under results/fam/).

The first seed's grid was run on the first workstation and the second seed's on the second, so the
comparison between the seeds is one between two seeds on two workstations (Section 4 of the plan).

Reads results/fam/ and results/fam_seed2025/; writes data/results_cohort1_graph_seed.json. It stops if a
cell of the second seed is missing. The summaries are computed by the functions of
code/second_cohort_graph_seed.py and code/second_cohort_graph.py, unchanged.
"""
import glob, json, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import second_cohort_graph as SG                 # the grid, the margin and the sign-flip tests
import second_cohort_graph_seed as S2            # describe() and ranks()

SEED2 = 2025
DIR1 = os.path.join(ROOT, 'results', 'fam')                  # first seed's grid and the one-cell check, first workstation
DIR2 = os.path.join(ROOT, 'results', 'fam_seed2025')         # second seed's grid, second workstation
R10 = SG.R10; GRID = SG.GRID


def cells_at(directory, mode, seed):
    """Cell -> the result record of one arm at one training seed (the default seed's files carry no seed tag)."""
    tag = '' if seed == SG.DEFAULT_SEED else '_ts%d' % seed
    pat = re.compile(r'^gnn_B_mb0\.0_p(?P<p>\d+)_r(?P<r>\d+)' + ('' if mode == 'real' else '_' + mode) + re.escape(tag) + r'_ivr_fam\.json$')
    out = {}
    for f in glob.glob(os.path.join(directory, '*.json')):
        m = pat.match(os.path.basename(f))
        if not m: continue
        r = json.load(open(f)); c = 'p%sr%s' % (m.group('p'), m.group('r'))
        ok = (r.get('feature_mode') == mode and r.get('inner_val_rxn') == 0.2 and r.get('family_tag') == 'fam' and r.get('model') == 'gnn_B'
              and r.get('lambda_mb') == 0.0 and r.get('train_seed', SG.DEFAULT_SEED) == seed and str(r.get('data', '')).rstrip('/').endswith('crc_624'))
        if not ok: sys.exit(f'{os.path.basename(f)}: not the cohort, arm and seed its place and name say')
        out[c] = r
    return out


def one_seed(directory, seed):
    arm = {k: cells_at(directory, mode, seed) for k, mode in SG.ARMS.items()}
    missing = {k: [c for c in GRID if c not in arm[k]] for k in SG.ARMS}
    if any(missing.values()):
        sys.exit(f'training seed {seed}: {sum(len(v) for v in missing.values())} cell files are missing under {os.path.relpath(directory, ROOT)}: {missing}')
    out = dict(train_seed=seed, directory=os.path.relpath(directory, ROOT), arms={})
    for k in SG.ARMS:
        recs = [arm[k][c] for c in GRID]; a = np.array([r['heldout_AUROC_perpatient'] for r in recs])
        out['arms'][k] = dict(per_cell={c: R10(x) for c, x in zip(GRID, a)}, mean=R10(a.mean()), sd=R10(a.std(ddof=1)),
                              epochs=dict(min=int(min(r['epochs'] for r in recs)), max=int(max(r['epochs'] for r in recs)),
                                          median=float(np.median([r['epochs'] for r in recs]))))
    d = np.array([arm['own'][c]['heldout_AUROC_perpatient'] - arm['cohort_mean'][c]['heldout_AUROC_perpatient'] for c in GRID])
    out['contrast'] = S2.describe(d)
    gap = np.array([arm['cohort_mean'][c]['epochs'] - arm['own'][c]['epochs'] for c in GRID], float)
    out['training_length'] = dict(epoch_gap_per_cell={c: int(x) for c, x in zip(GRID, gap)}, mean_epoch_gap=R10(gap.mean()),
                                  pearson_difference_with_epoch_gap=(R10(np.corrcoef(d, gap)[0, 1]) if gap.std() > 0 else None),
                                  spearman_difference_with_epoch_gap=(R10(np.corrcoef(S2.ranks(d), S2.ranks(gap))[0, 1]) if gap.std() > 0 else None),
                                  cells_with_equal_length=int((gap == 0).sum()),
                                  mean_difference_where_gap_at_most_2=(R10(d[np.abs(gap) <= 2].mean()) if (np.abs(gap) <= 2).any() else None),
                                  n_cells_where_gap_at_most_2=int((np.abs(gap) <= 2).sum()))
    return out, d, arm


def main():
    S1, d1, _ = one_seed(DIR1, SG.DEFAULT_SEED); S2_, d2, arm2 = one_seed(DIR2, SEED2)
    S1['workstation'] = 'first'; S2_['workstation'] = 'second'
    OUT = dict(note='the graph scorer on the first cohort under the family-disjoint split and the matched selection rule, own-expression arm against '
                    'cohort-mean arm on the 15 cells of the 5 x 3 grid, at the training seed of the primary grid (first workstation) and at a second '
                    'training seed (second workstation); AUROC on held-out reactions of held-out patients; FIRST_COHORT_SEED_PLAN_2026-11-15.md',
               margin=SG.MARGIN, primary='paired difference over cells, own minus cohort mean, at the second training seed',
               seeds={str(SG.DEFAULT_SEED): S1, str(SEED2): S2_},
               both_seeds=dict(what='per cell, the mean of the two seeds\' differences', **S2.describe((d1 + d2) / 2.0)),
               between_seeds=dict(what='second seed on the second workstation against first seed on the first workstation',
                                  grid_mean_second_minus_first=R10(d2.mean() - d1.mean()),
                                  mean_abs_change_of_a_cell=R10(np.abs(d2 - d1).mean()), max_abs_change_of_a_cell=R10(np.abs(d2 - d1).max()),
                                  cells_with_one_sign=int(((d1 < 0) == (d2 < 0)).sum()), n_cells=len(GRID),
                                  pearson=R10(np.corrcoef(d1, d2)[0, 1]), spearman=R10(np.corrcoef(S2.ranks(d1), S2.ranks(d2))[0, 1])))
    c = S2_['contrast']
    OUT['reading'] = dict(mean=c['mean'], sd=c['sd'], n_negative=c['n_negative'], n_cells=c['n_cells'], within_margin=c['within_margin'],
                          agreement='the mean lies within %.2f AUROC of zero, which the plan reads as agreement with the first seed' % SG.MARGIN
                          if c['within_margin'] else
                          'the mean does not lie within %.2f AUROC of zero, which the plan does not read as agreement with the first seed' % SG.MARGIN)
    # cell p0r0 at the second seed on the two workstations: the one-cell check made it on the first
    chk = {k: cells_at(DIR1, mode, SEED2).get('p0r0') for k, mode in SG.ARMS.items()}
    if all(chk.values()):
        a1 = {k: chk[k]['heldout_AUROC_perpatient'] for k in SG.ARMS}; a2 = {k: arm2[k]['p0r0']['heldout_AUROC_perpatient'] for k in SG.ARMS}
        OUT['cell_p0r0_second_seed_on_two_workstations'] = dict(
            first_workstation=dict(own=R10(a1['own']), cohort_mean=R10(a1['cohort_mean']), difference=R10(a1['own'] - a1['cohort_mean'])),
            second_workstation=dict(own=R10(a2['own']), cohort_mean=R10(a2['cohort_mean']), difference=R10(a2['own'] - a2['cohort_mean'])))
    # the first seed's contrast here must be the one already published
    ref = os.path.join(ROOT, 'data', 'results_cohort2_graph.json')
    if os.path.exists(ref):
        r1 = json.load(open(ref))['cohorts']['cohort1']['contrast']
        if abs(r1['mean'] - S1['contrast']['mean']) > 1e-9 or r1['n_negative'] != S1['contrast']['n_negative']:
            sys.exit('the first seed\'s contrast differs from data/results_cohort2_graph.json')
        OUT['first_seed_check'] = 'equal to data/results_cohort2_graph.json (mean, negative cells)'
    rec = os.path.join(ROOT, 'data', 'cohort1_graph_seed_run_record.json')      # where and when the cells of the second seed were computed
    if os.path.exists(rec): OUT['run_record'] = json.load(open(rec))
    json.dump(OUT, open(os.path.join(ROOT, 'data', 'results_cohort1_graph_seed.json'), 'w'), indent=1)
    b = OUT['both_seeds']; bs = OUT['between_seeds']
    print(f"first cohort, own minus cohort mean over cells: seed {SG.DEFAULT_SEED} {S1['contrast']['mean']:+.4f} (negative {S1['contrast']['n_negative']}), "
          f"seed {SEED2} {c['mean']:+.4f} sd {c['sd']:.4f} (negative {c['n_negative']}), both {b['mean']:+.4f} sd {b['sd']:.4f} (negative {b['n_negative']}); "
          f"between seeds: pearson {bs['pearson']:+.2f}, cells with one sign {bs['cells_with_one_sign']} of {bs['n_cells']}")


if __name__ == '__main__':
    main()
