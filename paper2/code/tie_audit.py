#!/usr/bin/env python3
"""How many printed percentiles sit on exactly tied expression values.

The prompts print, beside each reaction's expression value, its percentile within the cohort.
That percentile was computed from an argsort over the 624 patients, which assigns tied values
distinct consecutive ranks in an order that carries no information. This audit counts, for the
entries the manuscript evaluates, how many belong to a group of patients whose stored expression
value at that reaction is exactly equal (and, separately, equal after the three-decimal rounding
the prompt prints), how far apart the printed percentiles of tied values were, and what the
midrank convention adopted for every later collection would print instead.

Writes results/tie_audit.json. Reads only the released rxn_context.npz and the design files.
"""
import json, numpy as np
from scipy import stats as st
import _paths as PATHS

D = np.load(PATHS.data('rxn_context.npz'), allow_pickle=True)
X, HAS = D['X'], D['has']
N = X.shape[0]
_order = np.argsort(X, axis=0); _ranks = np.empty_like(_order)
for j in range(X.shape[1]): _ranks[_order[:, j], j] = np.arange(N)
PCT = 100.0 * _ranks / (N - 1)                                   # the convention the collections printed
PCT_MID = 100.0 * (st.rankdata(X, axis=0) - 1.0) / (N - 1)       # the midrank convention of every later collection

def audit(pats, rxns, expr_only):
    """Entries = evaluated patients x panel reactions (expression-bearing only when expr_only)."""
    pats = np.asarray(pats); rxns = np.asarray(rxns)
    if expr_only: rxns = rxns[HAS[rxns]]
    n_entries = len(pats) * len(rxns)
    tied_exact = tied_round = 0; spread = []; shift = []; groups_exact = 0
    for j in rxns:
        col = X[:, j]; colr = np.round(col, 3)
        _, inv, cnt = np.unique(col, return_inverse=True, return_counts=True)
        _, invr, cntr = np.unique(colr, return_inverse=True, return_counts=True)
        te = cnt[inv] > 1; tr = cntr[invr] > 1
        tied_exact += int(te[pats].sum()); tied_round += int(tr[pats].sum())
        for g in np.unique(inv[pats][te[pats]]):
            members = np.where(inv == g)[0]; groups_exact += 1
            p = np.round(PCT[members, j]); spread.append(float(p.max() - p.min()))
        shift.extend(np.abs(np.round(PCT[pats, j]) - np.round(PCT_MID[pats, j]))[te[pats]].tolist())
    return dict(n_patients=int(len(pats)), n_reactions=int(len(rxns)), n_entries=int(n_entries),
                tied_exact=int(tied_exact), tied_exact_share=round(100.0 * tied_exact / n_entries, 2),
                tied_after_rounding=int(tied_round), tied_after_rounding_share=round(100.0 * tied_round / n_entries, 2),
                extra_ties_from_rounding=int(tied_round - tied_exact),
                tie_groups_exact=int(groups_exact),
                printed_percentile_spread_within_tie_max=(round(max(spread), 1) if spread else 0.0),
                printed_percentile_spread_within_tie_mean=(round(float(np.mean(spread)), 2) if spread else 0.0),
                midrank_shift_of_tied_entries_mean=(round(float(np.mean(shift)), 2) if shift else 0.0),
                midrank_shift_of_tied_entries_max=(round(float(np.max(shift)), 1) if shift else 0.0))

R = dict(note='ties in the stored reaction-level expression across the 624 patients at the reactions each design shows; '
              'an entry is tied when at least one other patient has the identical stored value (exact) or the identical '
              'value after rounding to three decimals (the precision the prompt prints)',
         convention_collected='argsort rank over 624 patients, ties broken by patient order (no information)',
         convention_new='midrank (scipy.stats.rankdata average) over the cohort; identical values print identical percentiles',
         designs={})
des = json.load(open(PATHS.run('msi_int', 'design.json')))
R['designs']['msi_panel'] = audit(des['keep'], des['panel'], expr_only=False)
for run, tag in (('repr_il', 'A_representative'), ('repr40_il', 'E_representative40')):
    d = json.load(open(PATHS.run(run, 'design.json')))
    R['designs'][tag] = audit(d['patients'], d['reactions'], expr_only=True)
# the cohort-wide picture: share of all patient x expression-bearing-reaction entries that are tied
te_all = 0; n_all = 0
for j in np.where(HAS)[0]:
    _, inv, cnt = np.unique(X[:, j], return_inverse=True, return_counts=True)
    te_all += int((cnt[inv] > 1).sum()); n_all += N
R['cohort_expression_bearing'] = dict(n_entries=int(n_all), tied_exact=int(te_all), tied_exact_share=round(100.0 * te_all / n_all, 2))
import os
os.makedirs(os.path.join(PATHS.ROOT, 'results'), exist_ok=True)
json.dump(R, open(os.path.join(PATHS.ROOT, 'results', 'tie_audit.json'), 'w'), indent=1)
for k, v in R['designs'].items():
    print(f"{k:20s} entries {v['n_entries']:6d}  tied exact {v['tied_exact']:5d} ({v['tied_exact_share']}%)  after rounding {v['tied_after_rounding']:5d} ({v['tied_after_rounding_share']}%)  "
          f"printed spread within ties max {v['printed_percentile_spread_within_tie_max']} mean {v['printed_percentile_spread_within_tie_mean']}")
print('cohort', R['cohort_expression_bearing'])
