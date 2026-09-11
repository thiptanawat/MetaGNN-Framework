#!/usr/bin/env python3
"""The crossed bootstrap of the own-minus-stranger contrast at 10,000 draws under recorded seeds.

stats.py reports a crossed bootstrap (patients and reactions resampled together, arms paired within
cells) with 1,000 draws under one seed. One of its 90% endpoints, for the 40-patient representative
design, lies within a thousandth of the narrower 0.02 equivalence margin, so the Monte Carlo error of
the endpoint matters for the verdict. This script repeats that bootstrap with 10,000 draws under
three recorded seeds for every design that has one, and records the endpoints, their range over
seeds, and whether the "within 0.02" reading is the same under every seed.

Writes results/crossed_boot_10k.json. Reads the same run files as stats.py.
"""
import os, json, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stats as S
import _paths as PATHS

SEEDS = (29, 1029, 2029); B = 10000
RUNS = [('repr_il', 'A_representative'), ('repr', 'A_blocked'), ('repr_b2il', 'A_backbone2'), ('repr_b3il', 'A_backbone3'),
        ('repr40_il', 'E_representative40'), ('repr40', 'E_blocked'), ('repr40_il_d2', 'E_donor2'), ('repr_il_d2', 'A_donor2')]
OUT = dict(note='patients and reactions resampled with replacement together, arms paired within cells; percentile '
                'intervals of the mean over patients of the paired own-minus-stranger AUROC difference; B draws per seed',
           B=B, seeds=list(SEEDS), designs={})
for run, tag in RUNS:
    try:
        M, IDX, PSEL, ARMS, des = S.load(run)
    except FileNotFoundError:
        continue
    y = S.LAB[IDX].astype(int)
    per_seed = {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed); cd = []
        for _ in range(B):
            ix = rng.integers(0, len(PSEL), len(PSEL)); jx = rng.integers(0, len(IDX), len(IDX)); yb = y[jx]
            if len(set(yb)) < 2: continue
            ap = S.auc_matrix(M['patient'][np.ix_(ix, jx)], yb); asx = S.auc_matrix(M['shuffled'][np.ix_(ix, jx)], yb)
            cd.append(float(np.nanmean(ap - asx)))
        cd = np.array(cd)
        per_seed[str(seed)] = dict(n_draws=int(len(cd)),
                                   ci90=[round(float(np.percentile(cd, 5)), 4), round(float(np.percentile(cd, 95)), 4)],
                                   ci95=[round(float(np.percentile(cd, 2.5)), 4), round(float(np.percentile(cd, 97.5)), 4)],
                                   within_002_90=bool(np.percentile(cd, 5) > -0.02 and np.percentile(cd, 95) < 0.02),
                                   within_002_95=bool(np.percentile(cd, 2.5) > -0.02 and np.percentile(cd, 97.5) < 0.02),
                                   within_005_90=bool(np.percentile(cd, 5) > -0.05 and np.percentile(cd, 95) < 0.05),
                                   mean=round(float(cd.mean()), 4))
    hi90 = [v['ci90'][1] for v in per_seed.values()]; lo90 = [v['ci90'][0] for v in per_seed.values()]
    hi95 = [v['ci95'][1] for v in per_seed.values()]; lo95 = [v['ci95'][0] for v in per_seed.values()]
    OUT['designs'][tag] = dict(run=run, n_patients=int(len(PSEL)), n_reactions=int(len(IDX)), per_seed=per_seed,
                               ci90_upper_range=[min(hi90), max(hi90)], ci90_lower_range=[min(lo90), max(lo90)],
                               ci95_upper_range=[min(hi95), max(hi95)], ci95_lower_range=[min(lo95), max(lo95)],
                               within_002_90_all_seeds=all(v['within_002_90'] for v in per_seed.values()),
                               within_002_90_any_seed=any(v['within_002_90'] for v in per_seed.values()),
                               within_002_95_all_seeds=all(v['within_002_95'] for v in per_seed.values()),
                               within_005_90_all_seeds=all(v['within_005_90'] for v in per_seed.values()))
    d = OUT['designs'][tag]
    print(f"{tag:20s} n={d['n_patients']:3d}x{d['n_reactions']:3d}  90%: lower {d['ci90_lower_range']} upper {d['ci90_upper_range']}  "
          f"within 0.02 (90%) all seeds {d['within_002_90_all_seeds']}  95%: {d['ci95_lower_range']} {d['ci95_upper_range']}", flush=True)
json.dump(OUT, open(os.path.join(PATHS.ROOT, 'results', 'crossed_boot_10k.json'), 'w'), indent=1)
print('wrote results/crossed_boot_10k.json')
