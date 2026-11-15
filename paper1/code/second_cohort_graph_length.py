#!/usr/bin/env python3
"""The two arms of the graph scorer on the second cohort at equal training length.

Plan: Section 8 of paper1/SECOND_COHORT_PLAN_2026-11-13.md, fixed with this script before any of its runs.
Under the stopping rule of Sections 6 and 7 the cohort-mean arm was trained longer than the own-expression arm
(a median of 28 epochs against 16 at the training seed 2024, 22 against 16 at the seed 2025). With a patience of
15 epochs the scored checkpoints were those of epochs 13 and 1 at the median (7 and 1 at the second seed).
Section 8 removes the stopping rule: code/double_holdout_len.py trains every cell of the 5 x 3 grid, both arms,
at both training seeds, for 13 epochs and scores it after 1, 7 and 13 epochs with the weights of that epoch
(code/gpu_queue_cohort2_len.sh). The cells are the files tagged _len_ep<E> under results/cohort2/fam_len/.

Primary quantity: at each training seed and each of the three lengths, the paired difference over the 15 cells
between the own-expression arm and the cohort-mean arm trained for the same number of epochs, AUROC on held-out
reactions of held-out patients, with its mean, standard deviation over cells, range and number of negative
cells. No test is fixed; the sign-flip p values of Section 6 are given as descriptions.

Readings fixed in Section 8:
  - margin: a mean within 0.02 AUROC of zero is read as in Sections 6 and 7, and the number of the
    seed-by-length means that lie within it is reported;
  - training length: at each seed the reference length is the epoch of the cohort-mean arm's scored checkpoint
    at the median under the stopping rule (13 at the seed 2024, 7 at the seed 2025). If the mean difference at
    that length has the sign of the planned difference of that seed and at least half its size, the longer
    training of the cohort-mean arm is read as not accounting for most of the planned difference at that seed;
    otherwise it is read as accounting for most of it.

Beside it, descriptive: each arm's mean AUROC at each length, the share of the planned difference that each
equal-length difference amounts to, and the mean validation AUROC of each arm after every epoch.

Reads results/cohort2/fam_len/ and data/results_cohort2_graph_seed.json (the planned differences and the
epochs of Sections 6 and 7); writes data/results_cohort2_graph_length.json. It stops if a file of a requested
seed is missing. Without arguments both seeds are required; --seed 2024 analyses that seed alone, which
Section 8 provides for when the runs of the second seed cannot be completed.
"""
import argparse, glob, json, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import second_cohort_graph as SG          # the grid, the margin and the sign-flip tests of Section 6
import second_cohort_graph_seed as SS     # describe() of Section 7

DIR = os.path.join(ROOT, 'results', 'cohort2', 'fam_len')
LENGTHS = (1, 7, 13); SEEDS = (SG.DEFAULT_SEED, SS.SEED2); PATIENCE = 15
REFERENCE = {SG.DEFAULT_SEED: 13, SS.SEED2: 7}     # the cohort-mean arm's scored checkpoint at the median under the stopping rule
HALF = 0.5
R10 = SG.R10; GRID = SG.GRID


def cells_at(mode, seed, length):
    """Cell -> the result record of one arm at one training seed, scored after `length` epochs."""
    tag = '' if seed == SG.DEFAULT_SEED else '_ts%d' % seed
    pat = re.compile(r'^gnn_B_mb0\.0_p(?P<p>\d+)_r(?P<r>\d+)' + ('' if mode == 'real' else '_' + mode) + re.escape(tag) + r'_ivr_fam_len_ep%d\.json$' % length)
    out = {}
    for f in glob.glob(os.path.join(DIR, '*.json')):
        m = pat.match(os.path.basename(f))
        if not m: continue
        r = json.load(open(f)); c = 'p%sr%s' % (m.group('p'), m.group('r'))
        ok = (r.get('feature_mode') == mode and r.get('inner_val_rxn') == 0.2 and r.get('family_tag') == 'fam' and r.get('model') == 'gnn_B'
              and r.get('lambda_mb') == 0.0 and r.get('train_seed') == seed and r.get('epochs') == length and r.get('stopping') == 'none'
              and r.get('score_epochs') == list(LENGTHS) and len(r.get('val_curve', [])) == length
              and r.get('pfold') == int(m.group('p')) and r.get('rfold') == int(m.group('r')))
        if not ok: sys.exit(f'{os.path.basename(f)}: not the arm, seed and length its name says')
        out[c] = r
    return out


def one_seed(seed, planned):
    """The equal-length differences of one training seed, beside the planned difference of that seed."""
    med = {k: planned['arms'][k]['epochs']['median'] for k in SG.ARMS}
    if med['cohort_mean'] - PATIENCE != REFERENCE[seed] or med['own'] - PATIENCE != LENGTHS[0]:
        sys.exit(f'training seed {seed}: the median epochs of the planned runs ({med}) do not give the lengths fixed in Section 8')
    D0 = planned['contrast']['mean']
    out = dict(train_seed=seed, reference_length=REFERENCE[seed],
               planned=dict(difference_mean=R10(D0), own_mean=planned['arms']['own']['mean'], cohort_mean_mean=planned['arms']['cohort_mean']['mean'],
                            epochs_median=dict(own=med['own'], cohort_mean=med['cohort_mean']),
                            scored_epoch_median=dict(own=med['own'] - PATIENCE, cohort_mean=med['cohort_mean'] - PATIENCE)),
               lengths={})
    last = None
    for L in LENGTHS:
        arm = {k: cells_at(mode, seed, L) for k, mode in SG.ARMS.items()}
        missing = {k: [c for c in GRID if c not in arm[k]] for k in SG.ARMS}
        if any(missing.values()):
            sys.exit(f'training seed {seed}, {L} epochs: {sum(len(v) for v in missing.values())} cell files are missing under results/cohort2/fam_len: {missing}')
        entry = dict(epochs=L, arms={})
        for k in SG.ARMS:
            a = np.array([arm[k][c]['heldout_AUROC_perpatient'] for c in GRID])
            entry['arms'][k] = dict(per_cell={c: R10(x) for c, x in zip(GRID, a)}, mean=R10(a.mean()), sd=R10(a.std(ddof=1)),
                                    val_AUROC_mean=R10(np.mean([arm[k][c]['val_AUROC'] for c in GRID])))
        d = np.array([arm['own'][c]['heldout_AUROC_perpatient'] - arm['cohort_mean'][c]['heldout_AUROC_perpatient'] for c in GRID])
        entry['contrast'] = SS.describe(d)
        entry['share_of_planned_difference'] = R10(d.mean() / D0)
        out['lengths'][str(L)] = entry; last = arm
    # the validation AUROC after every epoch, from the records of the longest length (they hold the whole course)
    out['validation_course_mean'] = {k: [R10(x) for x in np.mean([last[k][c]['val_curve'] for c in GRID], axis=0)] for k in SG.ARMS}
    ref = out['lengths'][str(REFERENCE[seed])]
    share = ref['share_of_planned_difference']
    out['reference'] = dict(length=REFERENCE[seed], mean=ref['contrast']['mean'], share_of_planned_difference=share,
                            keeps_half=bool(share >= HALF), within_margin=ref['contrast']['within_margin'])
    return out


def reading(S):
    """The two readings of Section 8 as sentences, from the summaries of the analysed seeds."""
    seeds = sorted(S, key=int)
    within = [S[s]['lengths'][str(L)]['contrast']['within_margin'] for s in seeds for L in LENGTHS]
    n, k = len(within), sum(within)
    margin = ('all %d equal-length means lie within %.2f AUROC of zero' % (n, SG.MARGIN) if k == n else
              'none of the %d equal-length means lies within %.2f AUROC of zero' % (n, SG.MARGIN) if k == 0 else
              '%d of the %d equal-length means lie within %.2f AUROC of zero' % (k, n, SG.MARGIN))
    keeps = [S[s]['reference']['keeps_half'] for s in seeds]
    if len(seeds) == 1:
        length = ('at the reference length the difference has the sign of the planned difference and at least half its size, so the longer training '
                  'of the cohort-mean arm is not read as accounting for most of the planned difference' if keeps[0] else
                  'at the reference length the difference is less than half the planned difference or of the other sign, so the longer training of '
                  'the cohort-mean arm is read as accounting for most of the planned difference')
    elif all(keeps):
        length = ('at both seeds the difference at the reference length has the sign of the planned difference and at least half its size, so the '
                  'longer training of the cohort-mean arm is not read as accounting for most of the planned difference')
    elif not any(keeps):
        length = ('at both seeds the difference at the reference length is less than half the planned difference or of the other sign, so the '
                  'longer training of the cohort-mean arm is read as accounting for most of the planned difference')
    else:
        a, b = (seeds[0], seeds[1]) if keeps[0] else (seeds[1], seeds[0])
        length = (f'at the training seed {a} the difference at the reference length has the sign of the planned difference and at least half its size, '
                  f'and at the training seed {b} it does not, so the two seeds do not agree on whether the longer training of the cohort-mean arm '
                  'accounts for most of the planned difference')
    return dict(n_within_margin=k, n_means=n, margin=margin, keeps_half={s: bool(x) for s, x in zip(seeds, keeps)}, training_length=length)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, choices=SEEDS, default=None, help='analyse this training seed alone; the default is both')
    A = ap.parse_args()
    planned = json.load(open(os.path.join(ROOT, 'data', 'results_cohort2_graph_seed.json')))['seeds']
    seeds = SEEDS if A.seed is None else (A.seed,)
    S = {str(s): one_seed(s, planned[str(s)]) for s in seeds}
    OUT = dict(note='the graph scorer on the second cohort under the family-disjoint split, own-expression arm against cohort-mean arm on the 15 '
                    'cells of the 5 x 3 grid, both arms trained for the same number of epochs with no early stopping and no checkpoint '
                    'selection and scored after 1, 7 and 13 epochs; AUROC on held-out reactions of held-out patients; Section 8 of '
                    'SECOND_COHORT_PLAN_2026-11-13.md',
               margin=SG.MARGIN, half=HALF, lengths=list(LENGTHS), patience_of_the_stopping_rule=PATIENCE,
               primary='paired difference over cells, own minus cohort mean, both arms trained for the same number of epochs',
               seeds_analysed=[int(s) for s in S], seeds=S, reading=reading(S))
    rec = os.path.join(ROOT, 'data', 'cohort2_graph_length_run_record.json')     # where and when the runs were made
    if os.path.exists(rec): OUT['run_record'] = json.load(open(rec))
    json.dump(OUT, open(os.path.join(ROOT, 'data', 'results_cohort2_graph_length.json'), 'w'), indent=1)
    for s in S:
        row = ', '.join(f"{L} epochs {S[s]['lengths'][str(L)]['contrast']['mean']:+.4f}" for L in LENGTHS)
        print(f"second cohort, equal training length, seed {s}: own minus cohort mean over cells: {row}; planned {S[s]['planned']['difference_mean']:+.4f}")
    print(OUT['reading']['margin'] + '; ' + OUT['reading']['training_length'])


if __name__ == '__main__':
    main()
