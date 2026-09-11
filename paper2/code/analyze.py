#!/usr/bin/env python3
"""Analysis for the Paper 2 pilot: apply Paper 1's diagnostics to a language model's annotations.

Reports, per arm: AUROC against the input-independent HMA labels (per patient and pooled), against the
raw-expression baseline and the information-free indicator floor computed on the identical reaction set;
the median inter-patient agreement of the model's own outputs; and the decisive patient-vs-shuffled
comparison restricted to expression-bearing reactions, where the two prompts actually differ.
"""
import json, sys, numpy as np
from sklearn.metrics import roc_auc_score

out = sys.argv[1] if len(sys.argv) > 1 else 'pilot'
raw = json.load(open(f'{out}/raw.json')); des = json.load(open(f'{out}/design.json'))
D = np.load('rxn_context.npz', allow_pickle=True)
LAB, HAS, X = D['labels'], D['has'], D['X']
IDX = np.array(des['reactions']); PSEL = np.array(des['patients']); ARMS = des['arms']
y = LAB[IDX].astype(int); hasx = HAS[IDX]

def mat(arm):
    """[patients x reactions] of p_active; nan where the call failed."""
    M = np.full((len(PSEL), len(IDX)), np.nan)
    for a, pj in enumerate(PSEL):
        for b, ri in enumerate(IDX):
            v = raw.get(f'{arm}|{pj}|{ri}')
            if v and v['prob'] is not None: M[a, b] = v['prob']
    return M

def auc(sc, mask=None):
    m = np.ones(len(y), bool) if mask is None else mask
    ok = m & ~np.isnan(sc)
    return roc_auc_score(y[ok], sc[ok]) if len(set(y[ok])) > 1 else np.nan

print(f'design: {len(IDX)} reactions ({int(hasx.sum())} expression-bearing), {len(PSEL)} patients, '
      f'arms {ARMS}, {des["n"]} calls in {des["minutes"]} min')
print(f'label base rate in the sample: {y.mean():.3f}\n')

# ---- reference points on the identical reaction set ----
Xs = X[np.ix_(PSEL, IDX)]
raw_expr = float(np.mean([auc(Xs[i]) for i in range(len(PSEL))]))
raw_expr_x = float(np.mean([auc(Xs[i], hasx) for i in range(len(PSEL))]))
ind = auc(hasx.astype(float))
print(f'{"reference":<34} {"AUROC(all)":>11} {"AUROC(expr)":>12}')
print(f'{"raw expression, per patient":<34} {raw_expr:>11.4f} {raw_expr_x:>12.4f}')
print(f'{"indicator floor (any expression?)":<34} {ind:>11.4f} {"n/a":>12}')
print()

M = {a: mat(a) for a in ARMS}
print(f'{"arm":<12} {"parsed":>7} {"AUROC(all)":>11} {"AUROC(expr)":>12} {"pooled":>8} {"median r":>9} {"mean p":>7}')
for a in ARMS:
    Ma = M[a]; ok = np.isfinite(Ma).mean()
    per = [auc(Ma[i]) for i in range(len(PSEL))]
    perx = [auc(Ma[i], hasx) for i in range(len(PSEL))]
    flat_s = Ma.ravel(); flat_y = np.tile(y, len(PSEL))
    good = ~np.isnan(flat_s)
    if good.sum() < 10 or len(set(flat_y[good])) < 2:
        print(f'{a:<12} {ok:>7.1%}   (too few parsed calls yet)'); continue
    pooled = roc_auc_score(flat_y[good], flat_s[good])
    full = np.isfinite(Ma).all(1)
    C = np.corrcoef(Ma[full]) if full.sum() > 1 else np.array([[1.0]])
    med_r = float(np.median(C[np.triu_indices(len(C), 1)])) if len(C) > 1 else np.nan
    print(f'{a:<12} {ok:>7.1%} {np.nanmean(per):>11.4f} {np.nanmean(perx):>12.4f} {pooled:>8.4f} {med_r:>9.4f} {np.nanmean(Ma):>7.3f}')

# ---- the decisive test: does giving the model THIS patient's data change anything? ----
if 'patient' in ARMS and 'shuffled' in ARMS:
    P, S = M['patient'], M['shuffled']
    both = np.isfinite(P) & np.isfinite(S)
    ex = both & hasx[None, :]          # only where the two prompts actually differ
    ident = float((P[ex] == S[ex]).mean())
    d = np.abs(P[ex] - S[ex])
    print(f'\npatient vs shuffled, expression-bearing reactions only (n = {int(ex.sum())} pairs):')
    print(f'  identical answers          : {ident:.1%}')
    print(f'  mean |difference|          : {d.mean():.4f}   (score range 0-1)')
    print(f'  correlation of the two arms : {np.corrcoef(P[ex], S[ex])[0,1]:.4f}')
    # between-patient variation actually attributable to the patient's own data
    def resid(A_):
        A_ = A_.copy()
        return A_ - np.nanmean(A_, axis=0, keepdims=True)
    rp, rs = resid(P), resid(S)
    m2 = np.isfinite(rp) & np.isfinite(rs) & hasx[None, :]
    if m2.sum() > 10:
        print(f'  residual correlation (patient vs shuffled): {np.corrcoef(rp[m2], rs[m2])[0,1]:+.4f}')
    bp = np.nanstd(P[:, hasx], axis=0).mean(); bs = np.nanstd(S[:, hasx], axis=0).mean()
    print(f'  between-patient sd: real data {bp:.4f} | shuffled data {bs:.4f}')

# ---- does the model track the expression value it was shown? ----
if 'patient' in ARMS:
    P = M['patient']; m3 = np.isfinite(P) & hasx[None, :]
    if m3.sum() > 10:
        print(f'\ncorrelation between the model score and the expression value shown: '
              f'{np.corrcoef(P[m3], Xs[m3])[0,1]:+.4f}')
json.dump({'raw_expression': raw_expr, 'indicator': ind,
           'arms': {a: float(np.nanmean([auc(M[a][i]) for i in range(len(PSEL))])) for a in ARMS}},
          open(f'{out}/summary.json', 'w'), indent=1)

# ---- breakdown: where does the model agree or disagree with the label set? ----
RX = json.load(open('Recon3D.json'))['reactions']
sub = np.array([(RX[i].get('subsystem') or '') for i in IDX])
istr = np.array([s.startswith('Transport') or 'xchange' in s for s in sub])
for a in ARMS:
    Ma = M[a]
    if not np.isfinite(Ma).any(): continue
    s_ = np.nanmean(Ma, axis=0)
    ok = np.isfinite(s_)
    print(f'\n[{a}] mean p(active) by true label: active {np.nanmean(s_[ok & (y == 1)]):.3f} | '
          f'inactive {np.nanmean(s_[ok & (y == 0)]):.3f}')
    for nm, m in [('GPR-bearing', hasx), ('no GPR', ~hasx), ('transport/exchange', istr), ('metabolic', ~istr)]:
        mm = m & ok
        if mm.sum() > 5 and len(set(y[mm])) > 1:
            print(f'   {nm:20s} n={mm.sum():4d}  label active {y[mm].mean():.2f}  '
                  f'mean p {s_[mm].mean():.3f}  AUROC {roc_auc_score(y[mm], s_[mm]):.3f}')
