#!/usr/bin/env python3
"""Figures for Paper 1, drawn from data/results_p1.json so they cannot drift from the text.

Every number in every figure is read from the results file; nothing is typed in. Wherever a bar or a
summary rests on the 15 cells of the substitution basis (5 patient folds x 3 reaction folds), the
per-cell values are drawn over it, one marker per cell: the marker shape gives the reaction fold and
the five patient folds are spread across the height of the row, patient fold 0 at the top.

Figure numbers follow the order of the captions in manuscript/body_results.tex:
    Figure 1  fig_memorization.png    what the standard protocol rewards (three panels)
    Figure 2  fig_ladder.png          the decomposition ladder, per-cell markers on every bar
    Figure 3  fig_phenotype.png       the phenotype probe, one marker and interval per reaction fold
    Figure 4  fig_personalization.png the seven paired estimates as per-cell dot-and-interval rows

Each PNG is laid out at the text width (W inches, labels included), so fonts print at their nominal
size when the figure is included at \\linewidth.
"""
import json, os, textwrap, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MS = "ms" if os.path.isdir(os.path.join(HERE, "ms")) else "manuscript"
FIG = os.path.join(HERE, MS, "figures"); os.makedirs(FIG, exist_ok=True)

# One palette across all four figures, keyed to the predictor family rather than to the panel, so
# that a color means the same thing wherever it appears. The first figure's caption states it.
PAL = {
    'input':      '#d8dce2',   # not a predictor: the transcriptome, or the model's own input
    'naive':      '#8a93a0',   # unfitted rankings and the presence-only indicator
    'lin_expr':   '#5b6470',   # logistic regression on expression columns alone
    'lin_struct': '#4a3aa7',   # logistic regression on network features
    'mlp':        '#1baf7a',   # the per-reaction feature model
    'memo':       '#d9a017',   # the memorization control
    'gnn_real':   '#eb6834',   # graph model, each patient's own expression
    'gnn_mean':   '#2a78d6',   # graph model, one cohort-average vector
    'gnn_zero':   '#9dc3ec',   # graph model, expression zeroed
    'gnn_rew':    '#e34948',   # graph model, degree-preserving rewired network
}
PALE = {'input', 'gnn_zero'}   # light fills: annotate them in dark ink
RED = '#e34948'
INK = '#222222'
plt.rcParams.update({'font.size':8,'axes.labelsize':8,'axes.titlesize':8.5,'xtick.labelsize':7.5,
                     'ytick.labelsize':7.5,'legend.fontsize':7,'axes.spines.top':False,
                     'axes.spines.right':False,'figure.dpi':300,'savefig.dpi':300,
                     'savefig.bbox':'tight','savefig.pad_inches':0.02,'font.family':'DejaVu Sans'})
R = json.load(open(os.path.join(HERE, 'data', 'results_p1.json')))
L, NB, G, S, K = R['ladder'], R['naive_baselines'], R['grid'], R['substitution'], R.get('collapse', {})
# When the linear reference points exist on every cell, draw the ladder on the common basis so
# that every bar, linear or graph, rests on the identical cells.
LB = R.get('ladder_basis')
if LB and LB.get('structure_only') is not None:
    L = dict(L); L.update({k: v for k, v in LB.items() if v is not None})
KN = (R.get('collapse_noise') or {}).get('summary', {})
I = R.get('inference') or {}
# The ladder's linear rows are the tuned-C regime of the all-folds linear suite; its per-cell values
# are the ones drawn over those bars.
TUNED = ((R.get('naive_allfolds') or {}).get('regimes') or {}).get('tuned') or {}
CELLS = sorted(S['basis'])                       # 'p0r0' ... 'p4r2', the substitution basis
NPF, NRF = R['protocol']['pfolds'], R['protocol']['rfolds']
def ptitle(ax, l, t): ax.set_title(f'{l}  {t}', loc='left', fontweight='bold', pad=6)
W = 6.3

# ---- per-cell markers: shape = reaction fold, lane = patient fold --------------------------------
RF_MARK = ['o', 's', '^']

def _pf_rf(key):
    i = key.index('r'); return int(key[1:i]), int(key[i + 1:])

def draw_cells(ax, y, cells, face='white', edge=INK, ms=2.9, lane=0.10, alpha=0.9, mew=0.55, zorder=4):
    """One marker per cell at x = the cell's value; patient folds are lanes across the row."""
    for k, v in cells.items():
        pf, rf = _pf_rf(k)
        ax.plot(v, y + ((NPF - 1) / 2.0 - pf) * lane, marker=RF_MARK[rf], ms=ms, mfc=face, mec=edge,
                mew=mew, alpha=alpha, ls='none', zorder=zorder)

def fold_handles(face='#9a9a9a', edge=INK, ms=4):
    return [Line2D([], [], marker=RF_MARK[r], ms=ms, mfc=face, mec=edge, mew=0.6, ls='none',
                   label=f'reaction fold {r}') for r in range(NRF)]

def _mean(cells): return float(np.mean(list(cells.values())))
def _sd(cells):   return float(np.std(list(cells.values()), ddof=1))

def signed(v, nd=4):
    """A signed value with the typographic minus the tick labels use; a value that rounds to zero
    carries no sign."""
    if abs(v) < 0.5 * 10 ** (-nd):
        return f'{0:.{nd}f}'
    return f'{v:+.{nd}f}'.replace('-', '−')

# ---- the per-cell values behind each ladder row --------------------------------------------------
LIN_CELLS = {'indicator': 'indicator', 'expr_perpat': 'expr_perpat_rank',
             'expr_cohortmean': 'expr_cohortmean_rank', 'expr_perpat_lr': 'expr_perpat_lr',
             'expr_cohortmean_lr': 'expr_cohortmean_lr', 'topology_only': 'topology_only',
             'structure_only': 'structure_only', 'structure_plus_cohortmean': 'structure_plus_cohortmean',
             'structure_plus_perpat': 'structure_plus_perpat'}

def lin_cells(key):
    blk = TUNED.get(LIN_CELLS[key])
    if not blk or not blk.get('cells'):
        return None
    c = {x: blk['cells'][x] for x in CELLS}
    assert abs(_mean(c) - L[key]) < 1e-9, f'{key}: per-cell mean disagrees with the ladder value'
    return c

def gnn_cells(arm, sub_key):
    """Per-cell AUROC of a graph-model arm on the basis.

    The results file stores the graph arms' cell values only as paired differences against the
    tuned network-features linear row (inference block, six decimals); adding that row's own
    per-cell values back recovers each cell's AUROC to those six decimals. The mean of the
    recovered values is checked against the arm's stored mean.
    """
    d = (I.get(f'gnn_{arm}_vs_lin_tuned_structure_only') or {}).get('per_cell')
    so = (TUNED.get('structure_only') or {}).get('cells')
    if not d or not so:
        return None
    c = {x: so[x] + d[x] for x in CELLS}
    assert abs(_mean(c) - S[sub_key]['mean']) < 1e-5, f'gnn {arm}: recovered mean disagrees'
    return c

def ivr_cells(arm):
    """Per-cell AUROC of a graph-model arm under inner-validation stopping, recovered the same way
    from its paired difference against the no-patient linear row."""
    d = (I.get(f'ivr_{arm}_vs_lin_tuned_structure_only') or {}).get('per_cell')
    so = (TUNED.get('structure_only') or {}).get('cells')
    if not d or not so or set(d) != set(CELLS):
        return None
    c = {x: so[x] + d[x] for x in CELLS}
    _iv = (R.get('robustness') or {}).get('ivr') or {}
    ref = (_iv.get('real' if arm == 'real' else 'cohort_mean') or {}).get('mean')
    assert ref is None or abs(_mean(c) - ref) < 1e-5, f'ivr {arm}: recovered mean disagrees'
    return c

# ---- Figure 1: memorization + diagnostics --------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(W, 2.6), gridspec_kw=dict(width_ratios=[1.45, 1.0, 1.0]))
ax = axes[0]
tr = [G['mlp']['train_rxn'], G['mlp_emb']['train_rxn'], G['gnn_B']['train_rxn']]
ho = [G['mlp']['heldout_rxn'], G['mlp_emb']['heldout_rxn'], G['gnn_B']['heldout_rxn']]
fam = ['mlp', 'memo', 'gnn_real']
x = np.arange(3) * 1.25; w = 0.36
ax.bar(x-w/2, tr, w, color=PAL['input'], edgecolor='#5b6470', lw=0.8, label='train reactions')
ax.bar(x+w/2, ho, w, color=[PAL[f] for f in fam], label='held-out reactions')
for i,(a,b) in enumerate(zip(tr,ho)):
    # when the two bars are of a height, stagger the labels so they do not run together; the
    # held-out label starts at its bar's left edge so that it never runs into the train bar
    lift = 0.09 if abs(a - b) < 0.06 else 0.0
    ax.text(x[i]-w/2, a+0.02, f'{a:.4f}', ha='center', fontsize=6.0)
    ax.text(x[i]+0.02, b+0.02+lift, f'{b:.4f}', ha='left', fontsize=6.0)
ax.axhline(0.5, color='#999', lw=0.8, ls=':')
ax.set_xticks(x); ax.set_xticklabels(['mlp', 'mlp+emb', 'gnn'])
ax.set_ylim(0, 1.40); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_ylabel('AUROC')
ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(-0.02, 1.04),
          handlelength=1.1, borderaxespad=0, labelspacing=0.25)
ax.set_xlim(-0.7, x[-1] + 0.75)
ptitle(ax, 'A', 'AUROC by reaction set')

ax = axes[1]
x = np.arange(3)
gaps = [G['mlp']['gap_abs'], G['mlp_emb']['gap'], G['gnn_B']['gap']]
ax.bar(x, gaps, 0.55, color=[PAL[f] for f in fam])
for i,g in enumerate(gaps): ax.text(i, g+0.014, f'{g:.4f}', ha='center', fontsize=7)
ax.set_xticks(x); ax.set_xticklabels(['mlp', 'mlp+emb', 'gnn'])
ax.set_ylabel('mean |train minus held-out|'); ax.set_ylim(0, 0.60)
ptitle(ax, 'B', 'Memorization gap')

ax = axes[2]
rho = [S['real']['rho'], S['cohort_mean']['rho'], S['zero']['rho']]
ax.bar(np.arange(3), rho, 0.55, color=[PAL[k] for k in ('gnn_real', 'gnn_mean', 'gnn_zero')])
ax.axhline(S['rho_null'], color=RED, lw=1.3, ls='--', label='no-signal null')
ax.legend(frameon=False, loc='upper right', fontsize=6.6, handlelength=1.4)
for i,v in enumerate(rho): ax.text(i, v+0.08, f'{v:.2f}', ha='center', fontsize=7)
ax.set_xticks(np.arange(3)); ax.set_xticklabels(['unsub-\nstituted', 'cohort\nmean', 'zeroed'])
ax.set_ylabel(r'dispersion ratio $\rho$'); ax.set_ylim(0, 3.4)
ptitle(ax, 'C', 'Dispersion ratio')
fig.tight_layout(w_pad=2.0); fig.savefig(os.path.join(FIG, 'fig_memorization.png')); plt.close(fig)

# ---- Figure 2: the decomposition ladder ----------------------------------------------------------
# Ordered as Table 5 is: the rows that use no model, the fitted linear rows, then the graph arms.
# Each bar is the mean over the 15 cells and carries its 15 per-cell values as markers.
_iv0 = (R.get('robustness') or {}).get('ivr') or {}
fig, ax = plt.subplots(figsize=(W, (5.9 if _iv0.get('zero') and _iv0['zero']['n'] == len(_iv0.get('basis') or []) else 5.6) if _iv0.get('real') else 5.0), layout='constrained')
rows = [('presence-only indicator',                     L['indicator'],                 'naive',      'shared vector', lin_cells('indicator')),
        ("each patient's own expression, ranked",       L['expr_perpat'],               'naive',      'own column',          lin_cells('expr_perpat')),
        ('cohort-mean expression, ranked',              L['expr_cohortmean'],           'naive',      'shared vector', lin_cells('expr_cohortmean')),
        ("each patient's own expression, linear",       L['expr_perpat_lr'],            'lin_expr',   'own column',          lin_cells('expr_perpat_lr')),
        ('cohort-mean expression, linear',              L['expr_cohortmean_lr'],        'lin_expr',   'shared vector', lin_cells('expr_cohortmean_lr')),
        ('topology only, linear',                       L['topology_only'],             'lin_struct', 'no patient data',     lin_cells('topology_only')),
        ('network features only, linear',               L['structure_only'],            'lin_struct', 'no patient data',     lin_cells('structure_only')),
        ('network features + cohort-mean expr, linear', L['structure_plus_cohortmean'], 'lin_struct', 'shared vector', lin_cells('structure_plus_cohortmean')),
        ('network features + own expression, linear',   L['structure_plus_perpat'],     'lin_struct', 'own column',          lin_cells('structure_plus_perpat')),
        ('graph model, expression zeroed',              L['gnn_zero'],                  'gnn_zero',   'no patient data',     gnn_cells('zero', 'zero')),
        ('graph model, cohort-mean expression',         L['gnn_cohort_mean'],           'gnn_mean',   'shared vector', gnn_cells('mean', 'cohort_mean')),
        ("graph model, patient's own expression",       L['gnn_real'],                  'gnn_real',   'own column',          gnn_cells('real', 'real'))]
_iv = (R.get('robustness') or {}).get('ivr') or {}
HAS_IVR = bool(_iv.get('real') and _iv.get('cohort_mean') and set(_iv['basis']) == set(CELLS))
# the zeroed arm under the same rule, when it ran on every cell: its per-cell values are the
# cohort-mean arm's less the paired cohort-mean-minus-zeroed differences
HAS_IVR_ZERO = False
if HAS_IVR and _iv.get('zero') and _iv['zero']['n'] == len(CELLS):
    _dz = (I.get('ivr_mean_vs_zero') or {}).get('per_cell') or {}
    _mc = ivr_cells('mean')
    if _mc and set(_dz) == set(CELLS):
        _zc = {x: _mc[x] - _dz[x] for x in CELLS}
        assert abs(_mean(_zc) - _iv['zero']['mean']) < 1e-5, 'ivr zero: recovered mean disagrees'
        rows.append(('graph model, expression zeroed,\nselected on withheld reactions', _iv['zero']['mean'], 'gnn_zero', 'no patient data', _zc))
        HAS_IVR_ZERO = True
if HAS_IVR:
    rows += [('graph model, cohort-mean expression,\nselected on withheld reactions', _iv['cohort_mean']['mean'], 'gnn_mean', 'shared vector', ivr_cells('mean')),
             ("graph model, patient's own expression,\nselected on withheld reactions", _iv['real']['mean'], 'gnn_real', 'own column', ivr_cells('real'))]
yy = np.arange(len(rows))[::-1]
ax.barh(yy, [r[1] for r in rows], color=[PAL[r[2]] for r in rows], height=0.66, zorder=1)
for y, r in zip(yy, rows):
    cells = r[4]
    if cells:
        draw_cells(ax, y, cells)
    x_lab = (max(cells.values()) if cells else r[1]) + 0.007
    ax.text(x_lab, y, f'{r[1]:.4f}', va='center', fontsize=7.5, zorder=6)
    ax.text(0.503, y, r[3], va='center', fontsize=5.6,
            color='#333' if r[2] in PALE else 'white', style='italic', zorder=6)
ax.set_yticks(yy); ax.set_yticklabels([r[0] for r in rows])
ax.set_xlim(0.5, 0.99); ax.set_xlabel('AUROC on held-out reactions')
ax.axvline(0.5, color='#999', lw=0.8, ls=':')
d = S['patient_identity']
n_neg = (I.get('real_vs_mean') or {}).get('n_negative')
XA = 0.945
# one arrow per own-versus-cohort-mean pair of graph arms: the grid's rule (the two main arms) and,
# when the full grid exists, the matched rule (the two arms selected on withheld reactions)
_pairs = [(yy[11], yy[10], d['delta'])]                    # own arm (row 11) against cohort-mean arm (row 10)
if HAS_IVR:
    _pairs.append((yy[len(rows) - 1], yy[len(rows) - 2], _iv['patient_identity']['delta']))   # the last two rows
for _ylo, _yhi, _delta in _pairs:
    ax.annotate('', xy=(XA, _ylo), xytext=(XA, _yhi), arrowprops=dict(arrowstyle='->', color=RED, lw=1.6))
    ax.text(XA - 0.008, (_ylo + _yhi) / 2, signed(_delta), color=RED, fontsize=7.5, fontweight='bold',
            ha='right', va='center')
sign_note = f"; negative in {n_neg} of {d['n']} cells" if n_neg is not None else ''
if HAS_IVR:
    _pi = _iv['patient_identity']
    _txt = (f"individual-versus-cohort-mean contrast, paired over {d['n']} cells:\n"
            f"grid's rule {signed(d['delta'])} AUROC (sd {d['sd']:.4f}; negative in {n_neg} of {d['n']} cells)\n"
            f"selected on withheld reactions {signed(_pi['delta'])} AUROC\n"
            f"(sd {_pi['sd']:.4f}; negative in {_pi['n_negative']} of {_pi['n']} cells)")
    _ytxt = yy[-1] - 1.7; _ylim0 = -2.85
else:
    _txt = (f"individual-versus-cohort-mean contrast, paired over {d['n']} cells: {signed(d['delta'])} AUROC\n"
            f"(sd {d['sd']:.4f} across cells{sign_note})")
    _ytxt = yy[-1] - 1.15; _ylim0 = -1.75
ax.text(0.5 + 0.006, _ytxt, _txt, color=RED, fontsize=(6.8 if HAS_IVR else 7.3), fontweight='bold', ha='left', va='center', linespacing=1.25)
ax.set_ylim(_ylim0, len(rows) - 0.4)
leg = ax.legend(handles=fold_handles(face='white'), loc='upper right', bbox_to_anchor=(0.995, 0.995),
                frameon=True, framealpha=0.95, edgecolor='#cccccc', fontsize=6.8,
                title=f'one marker per cell ({len(CELLS)} cells):\nshape gives the reaction fold, the\n'
                      f'{NPF} patient folds run top to bottom', title_fontsize=6.6,
                handletextpad=0.5, labelspacing=0.35, borderpad=0.6)
leg.get_title().set_ha('left')
fig.savefig(os.path.join(FIG, 'fig_ladder.png')); plt.close(fig)

# ---- Figure 3: a phenotype the benchmark cannot see ----------------------------------------------
# One marker and its own patient-level bootstrap interval per reaction fold, the mean over the
# folds as a black tick without an interval, and one diamond for a quantity fitted once because it
# does not vary with the reaction fold. No envelope is drawn.
PH = R.get('phenotype')
PMU = R.get('phenotype_multi') or {}
if PH:
    def _entries(per_fold, key):
        return [e for e in (per_fold.get(key) or []) if e.get('auroc') is not None]

    def _agg_mean(blk, key, entries):
        m = ((blk.get('aggregate') or {}).get(key) or {}).get('mean')
        if m is None:
            m = float(np.mean([e['auroc'] for e in entries]))
        assert abs(m - np.mean([e['auroc'] for e in entries])) < 1e-9, f'{key}: fold mean disagrees'
        return m

    groups = []   # (header, [row, ...]); row = (label, family, kind, payload, mean)
    A = PH['arms']
    n_ref, p_ref = A['real']['n_patients'], A['real']['n_pos']
    rows = []
    if PH.get('genes'):
        rows.append(('transcriptome before\nGPR mapping', 'input', 'single',
                     (PH['genes']['auroc'], PH['genes']['ci_lo'], PH['genes']['ci_hi']), None))
    inp = _entries(PH['per_fold'], 'input')
    if inp:
        rows.append(('model input:\nreaction-level expression', 'naive', 'single',
                     (inp[0]['auroc'], inp[0]['lo'], inp[0]['hi']), None))
    for k, lab, fam_ in (('real', "graph model,\npatient's own expression", 'gnn_real'),
                         ('real_on_rewired_patients', "graph model, own expression,\nrescored on the rewired\narm's patients", 'gnn_real'),
                         ('rewire7', 'graph model,\nrewired network', 'gnn_rew'),
                         ('mean', 'graph model,\ncohort-mean expression', 'gnn_mean'),
                         ('zero', 'graph model,\nexpression zeroed', 'gnn_zero')):
        ent = _entries(PH['per_fold'], k)
        if not ent:
            continue
        n_lab = lab + (f" ({A[k]['n_pos']} of {A[k]['n_patients']})"
                       if A[k]['n_patients'] != n_ref else '')
        rows.append((n_lab, fam_, 'folds', ent, _agg_mean(PH, k, ent)))
    # headers carry the attribute and, in parentheses, positives of annotated patients
    groups.append((f"A  microsatellite instability, {PH['target']} ({p_ref} of {n_ref} unless stated)", rows))
    for letter, attr, name in (('B', 'cin_vs_gs', None), ('C', 'sex', 'sex'),
                               ('D', 'site', 'anatomical site')):
        M = PMU.get(attr)
        if not M:
            continue
        rows = []
        inp = _entries(M['per_fold'], 'input')
        if inp:
            rows.append(('model input:\nreaction-level expression', 'naive', 'single',
                         (inp[0]['auroc'], inp[0]['lo'], inp[0]['hi']), None))
        for k, lab, fam_ in (('real', "graph model,\npatient's own expression", 'gnn_real'),
                             ('mean', 'graph model,\ncohort-mean expression', 'gnn_mean')):
            ent = _entries(M['per_fold'], k)
            if not ent:
                continue
            if len(ent) < NRF:
                lab += f"\n(reaction fold{'s' if len(ent) > 1 else ''} "
                lab += ', '.join(str(e['fold']) for e in ent) + ' only)'
            rows.append((lab, fam_, 'folds', ent, _agg_mean(M, k, ent)))
        # a short multi-line title, since the small panels are narrow
        one = f"{letter}  {name}, {M['label']}" if name else None
        if one and len(one) <= 24:
            lines = [one]
        elif name:
            lines = [f"{letter}  {name},"] + textwrap.wrap(M['label'], 24)
        else:
            parts = textwrap.wrap(M['label'], 24)
            lines = [f"{letter}  {parts[0]}"] + parts[1:]
        groups.append(('\n'.join(lines + [f"({M['n_pos']} of {M['n']})"]), rows))

    LANE = 0.23

    def draw_probe(ax, rows, ms=4.2, value_fs=7.0, label_fs=7.2, pad=0.012):
        """Rows top to bottom; per-fold markers in three lanes, the fold mean as a black tick."""
        y, ticks, tlabs = 0.0, [], []
        for lab, fam_, kind, payload, mean in rows:
            if lab.count('\n') >= 2:
                y -= 0.2
            ticks.append(y); tlabs.append(lab)
            if kind == 'single':
                v, lo, hi = payload
                ax.errorbar([v], [y], xerr=[[v - lo], [hi - v]], fmt='none', ecolor='#555',
                            elinewidth=0.8, capsize=1.5, capthick=0.8, zorder=3)
                ax.plot(v, y, marker='D', ms=ms, mfc=PAL[fam_], mec='#333', mew=0.5, ls='none', zorder=5)
                x_hi, x_val = hi, v
            else:
                for e in payload:
                    dy = ((NRF - 1) / 2.0 - e['fold']) * LANE
                    ax.errorbar([e['auroc']], [y + dy], xerr=[[e['auroc'] - e['lo']], [e['hi'] - e['auroc']]],
                                fmt='none', ecolor='#555', elinewidth=0.8, capsize=1.5, capthick=0.8, zorder=3)
                    ax.plot(e['auroc'], y + dy, marker=RF_MARK[e['fold']], ms=ms, mfc=PAL[fam_],
                            mec='#333', mew=0.5, ls='none', zorder=5)
                if len(payload) > 1:
                    ax.plot(mean, y, marker='|', ms=11, mec='black', mew=1.4, ls='none', zorder=6)
                x_hi, x_val = max(e['hi'] for e in payload), mean
            ax.text(x_hi + pad, y, f'{x_val:.3f}', va='center', fontsize=value_fs, zorder=6)
            if lab.count('\n') >= 2:
                y -= 0.2
            y -= 1.0
        ax.set_yticks(ticks); ax.set_yticklabels(tlabs, fontsize=label_fs)
        ax.set_ylim(y + 0.45, 0.55)
        ax.axvline(0.5, color='#999', lw=0.9, ls=':', zorder=0)
        return tlabs

    def _units(rows):
        return len(rows) + 0.4 * sum(1 for r in rows if r[0].count('\n') >= 2)

    XLAB = ("out-of-fold AUROC from each arm's per-patient score vector\n"
            "(in parentheses: positives of annotated patients)")
    top_header, top_rows = groups[0]
    small = groups[1:]   # the further attributes, as a row of small panels under the main one
    u_top, u_bot = _units(top_rows), max([_units(r) for _, r in small] + [0])
    fig = plt.figure(figsize=(W, 0.27 * (u_top + u_bot) + (1.75 if small else 1.1)), layout='constrained')
    if small:
        gs = fig.add_gridspec(2, len(small), height_ratios=[u_top + 0.6, u_bot + 1.0])
        ax = fig.add_subplot(gs[0, :])
    else:
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel(XLAB)
    draw_probe(ax, top_rows)
    ax.set_title(top_header, loc='left', fontsize=7.6, fontweight='bold', pad=4)
    ax.set_xlim(0.3, 1.075); ax.set_xticks(np.round(np.arange(3, 11) / 10.0, 1))
    first_labels = None
    for i, (header, rows) in enumerate(small):
        axi = fig.add_subplot(gs[1, i])
        labs = draw_probe(axi, rows, ms=3.9, value_fs=6.6, label_fs=7.0, pad=0.015)
        # the three small panels carry identical rows; the labels are written once, on the left
        if first_labels is None:
            first_labels = labs
        elif labs == first_labels:
            axi.set_yticklabels([])
            axi.tick_params(axis='y', length=0)
        axi.set_title(header, loc='left', fontsize=7.0, fontweight='bold', pad=4, linespacing=1.15)
        axi.set_xlim(0.3, 1.16); axi.set_xticks([0.3, 0.5, 0.7, 0.9])
        axi.tick_params(axis='x', labelsize=7.0)
        if i == len(small) // 2:
            axi.set_xlabel(XLAB)
    handles = fold_handles(ms=4.2) + [
        Line2D([], [], marker='|', ms=11, mec='black', mew=1.4, ls='none',
               label='mean over the reaction folds shown, no interval'),
        Line2D([], [], marker='D', ms=4.2, mfc='#9a9a9a', mec=INK, mew=0.6, ls='none',
               label='one fit, does not vary with the reaction fold'),
        Line2D([], [], color='#555', lw=0.8, marker='|', ms=5, mew=0.8,
               label='patient-level bootstrap percentile interval')]
    fig.legend(handles=handles, loc='outside lower center', ncol=3, frameon=False, fontsize=6.8,
               handletextpad=0.5, columnspacing=1.4)
    fig.savefig(os.path.join(FIG, 'fig_phenotype.png')); plt.close(fig)

# ---- Figure 4: the personalization effect, seven ways --------------------------------------------
# Every row is a paired per-cell difference, drawn as its 15 per-cell values, the mean over cells
# and a bar of one standard deviation across cells. Four rows substitute the model input and refit;
# three re-score the unsubstituted model's saved predictions after the fact (hollow markers).
def _cells_from(key, ref, sign=1.0, tol=2e-6):
    pc = (I.get(key) or {}).get('per_cell')
    if not pc:
        return None
    c = {x: sign * pc[x] for x in CELLS if x in pc}
    if len(c) != len(CELLS):
        return None
    assert abs(_mean(c) - sign * ref) < tol, f'{key}: per-cell mean disagrees with the summary'
    return c

INPUT_SUB, POSTHOC = 'input', 'posthoc'
eff = []   # (label, mean, family, kind, per-cell dict or None)
np_cells = None
if TUNED.get('expr_perpat_rank') and TUNED.get('expr_cohortmean_rank'):
    np_cells = {x: TUNED['expr_perpat_rank']['cells'][x] - TUNED['expr_cohortmean_rank']['cells'][x] for x in CELLS}
    assert abs(_mean(np_cells) - L['nonparam_patient_effect']) < 1e-9
eff.append(("expression only, ranked\n(no model)", L['nonparam_patient_effect'], 'naive', INPUT_SUB, np_cells))
if L.get('lin_patient_expr_only') is not None:
    eff.append(("expression only, linear,\nfitted per patient", L['lin_patient_expr_only'], 'lin_expr', INPUT_SUB,
                _cells_from('lin_tuned_expr_perpat_lr_vs_expr_cohortmean_lr', L['lin_patient_expr_only'])))
if L.get('lin_patient_struct_expr') is not None:
    eff.append(("network features + expression,\nlinear, fitted per patient", L['lin_patient_struct_expr'], 'lin_struct', INPUT_SUB,
                _cells_from('lin_tuned_structure_plus_perpat_vs_structure_plus_cohortmean', L['lin_patient_struct_expr'])))
# the same two linear comparisons under the fixed-model protocol: one model per cell fitted on the
# training patients' own columns and frozen for the test patients (the pooled rows), against the
# cohort-mean fit, which is a fixed model already
PD = (((R.get('naive_pooled') or {}).get('deltas') or {}).get('tuned') or {})
if PD.get('pooled_vs_cohort_expr'):
    eff.append(("expression only, linear,\npooled and frozen", PD['pooled_vs_cohort_expr']['delta'], 'lin_expr', INPUT_SUB,
                _cells_from('lin_tuned_expr_pooled_lr_vs_expr_cohortmean_lr', PD['pooled_vs_cohort_expr']['delta'])))
if PD.get('pooled_vs_cohort_struct'):
    eff.append(("network features + expression,\nlinear, pooled and frozen", PD['pooled_vs_cohort_struct']['delta'], 'lin_struct', INPUT_SUB,
                _cells_from('lin_tuned_structure_plus_pooled_vs_structure_plus_cohortmean', PD['pooled_vs_cohort_struct']['delta'])))
eff.append(("graph model, retrained", S['patient_identity']['delta'], 'gnn_real', INPUT_SUB,
            _cells_from('real_vs_mean', S['patient_identity']['delta'])))
# the same retraining selected on reactions withheld from the loss (inner-validation stopping),
# drawn when it covers the whole substitution basis
_iv = (R.get('robustness') or {}).get('ivr') or {}
if _iv.get('patient_identity') and set(_iv['basis']) == set(CELLS):
    eff.append(("graph model, retrained,\nselected on withheld reactions", _iv['patient_identity']['delta'], 'gnn_real', INPUT_SUB,
                _cells_from('ivr_real_vs_mean', _iv['patient_identity']['delta'])))
# the same retraining again on the family-disjoint reaction split, where a held-out reaction has no
# near-twin in training; it rests on its own cells, so its per-cell markers are its own
_fa = (R.get('robustness') or {}).get('fam') or {}
if _fa.get('patient_identity'):
    _fp = (I.get('fam_real_vs_mean') or {}).get('per_cell')
    eff.append(("graph model, retrained,\nrelated reactions withheld together", _fa['patient_identity']['delta'],
                'gnn_real', INPUT_SUB, dict(_fp) if _fp else None))
# The collapse rows enter net of what averaging dropout noise alone produces, where that
# correction exists; otherwise as the raw collapse gain. Their natural direction is cohort mean
# minus per patient, so the sign is reversed to the personalization direction.
for m, lab, fam_ in (('mlp', 'feature model', 'mlp'),
                     ('mlp_emb', 'memorization control', 'memo'),
                     ('gnn_B', 'graph model', 'gnn_real')):
    if m in KN:
        eff.append((lab + '\n(net of noise averaging, exploratory)', -KN[m]['identity_component'], fam_, POSTHOC,
                    _cells_from(f'collapse_identity_{m}', KN[m]['identity_component'], sign=-1.0)))
    elif m in K:
        eff.append((lab, -K[m]['paired_diff'], fam_, POSTHOC,
                    _cells_from(f'collapse_{m}', K[m]['paired_diff'], sign=-1.0)))

HEADERS = {INPUT_SUB: ('Input substitution',
                       "each patient's own expression column against one\n"
                       "cohort-mean column; the model is refit on that input"),
           POSTHOC:   ('Post-hoc output averaging',
                       "the unsubstituted model's saved per-patient predictions\n"
                       "collapsed to their cohort mean; nothing is retrained")}
fig, ax = plt.subplots(figsize=(W, 4.6), layout='constrained')
order = [e for e in eff if e[3] == INPUT_SUB] + [e for e in eff if e[3] == POSTHOC]
ys, y_hdr, y = {}, {}, 0.0
for kind in (INPUT_SUB, POSTHOC):
    y -= 0.55
    y_hdr[kind] = y          # the bold group name sits here, its description just below
    y -= 1.35
    for e in order:
        if e[3] == kind:
            ys[e[0]] = y; y -= 1.0
    y -= 0.05
y_bot = y + 0.5
ax.axhspan(y_bot, y_hdr[POSTHOC] + 0.55, color='#f1f1f1', zorder=0)
xs_all, nums = [], []
for e in order:
    lab, mean, fam_, kind, cells = e
    yv = ys[lab]
    hollow = kind == POSTHOC
    if cells:
        draw_cells(ax, yv, cells, face='white' if hollow else PAL[fam_], edge=PAL[fam_] if hollow else INK,
                   ms=3.1, lane=0.115, alpha=0.75 if not hollow else 0.95, mew=0.7 if hollow else 0.45, zorder=3)
        sd = _sd(cells)
        ax.errorbar([mean], [yv], xerr=[[sd], [sd]], fmt='none', ecolor='black', elinewidth=1.0,
                    capsize=2.2, capthick=1.0, zorder=2)
        xs_all += list(cells.values()) + [mean - sd, mean + sd]
    else:
        sd = None; xs_all.append(mean)
    ax.plot(mean, yv, marker='D', ms=5.2, mfc='white' if hollow else PAL[fam_],
            mec=PAL[fam_] if hollow else 'black', mew=1.3 if hollow else 0.8, ls='none', zorder=5)
    nums.append(signed(mean) + (f' ({sd:.4f})' if sd is not None else ''))
lo, hi = min(xs_all), max(xs_all)
span = hi - lo
# room on the right so that the group headers end before the zero line
ax.set_xlim(lo - 0.05 * span, hi + 0.16 * span)
for kind, yh in y_hdr.items():
    name, desc = HEADERS[kind]
    ax.text(lo - 0.035 * span, yh, name, fontsize=7.4, fontweight='bold', color=INK, ha='left', va='center', zorder=6,
            bbox=dict(facecolor='white' if kind == INPUT_SUB else '#f1f1f1', edgecolor='none', pad=1.5))
    ax.text(lo - 0.035 * span, yh - 0.62, desc, fontsize=6.5, color='#444', ha='left', va='center',
            linespacing=1.2, zorder=6, bbox=dict(facecolor='white' if kind == INPUT_SUB else '#f1f1f1', edgecolor='none', pad=1.5))
ax.axvline(0, color='#333', lw=1.0, zorder=1)
ax.set_yticks([ys[e[0]] for e in order]); ax.set_yticklabels([e[0] for e in order])
ax.set_ylim(y_bot, 0.0)
# the per-row mean and standard deviation as a column of labels on the right
ax2 = ax.twinx()
ax2.set_ylim(ax.get_ylim()); ax2.set_yticks([ys[e[0]] for e in order]); ax2.set_yticklabels(nums, fontsize=7.2)
ax2.tick_params(axis='y', length=0, pad=6)
for sp in ax2.spines.values(): sp.set_visible(False)
_ncells = sorted({len(e[4]) for e in eff if e[4]} | {len(CELLS)})
_hdr = (f'mean (sd)\nover {len(CELLS)} cells' if len(_ncells) == 1
        else f'mean (sd)\nover {min(_ncells)} to {max(_ncells)} cells')
ax.text(1.0, y_hdr[INPUT_SUB] + 0.05, _hdr, transform=ax.get_yaxis_transform(),
        ha='left', va='center', fontsize=7.0, color='#444', linespacing=1.2)
ax.set_xlabel('individual-versus-cohort-mean contrast in held-out AUROC\n'
              '(negative: the own column, or own predictions, scored below the cohort average)')
handles = fold_handles(face='#9a9a9a', ms=3.6) + [
    Line2D([], [], marker='D', ms=5.2, mfc='#9a9a9a', mec='black', mew=0.8, ls='none',
           label='mean over the cells, input substitution'),
    Line2D([], [], marker='D', ms=5.2, mfc='white', mec='#9a9a9a', mew=1.3, ls='none',
           label='mean over the cells, post-hoc output averaging'),
    Line2D([], [], color='black', lw=1.0, marker='|', ms=6, mew=1.0,
           label='one standard deviation across the cells')]
fig.legend(handles=handles, loc='outside lower center', ncol=3, frameon=False, fontsize=6.8,
           handletextpad=0.5, columnspacing=1.4,
           title=f'one marker per cell: shape gives the reaction fold, '
                 f'the {NPF} patient folds run top to bottom within a row', title_fontsize=6.8)
fig.savefig(os.path.join(FIG, 'fig_personalization.png')); plt.close(fig)

print('figures written to', os.path.relpath(FIG, HERE))

# ---- Figure 5: the family-disjoint reaction split -------------------------------------------------
# Two panels on the same cells. Left: what withholding a reaction's relatives costs each arm, as the
# paired per-cell difference between the two splits. Right: the four contrasts the paper reads,
# drawn on both splits, so that what the harder split does to each conclusion is visible at once.
FAMB = (R.get('robustness') or {}).get('fam') or {}
IVRB = (R.get('robustness') or {}).get('ivr') or {}
if FAMB.get('versus_random_split') and FAMB.get('patient_identity'):
    ARMS = [('real', 'own expression', 'gnn_real'), ('cohort_mean', 'cohort mean', 'gnn_mean'),
            ('indicator', 'presence only', 'naive'), ('permute', "another patient's column", 'memo'),
            ('zero', 'expression zeroed', 'gnn_zero')]
    CON = [('patient_identity', "own expression\nminus cohort mean", 'gnn_real'),
           ('own_minus_permuted', "own column minus\nanother patient's", 'memo'),
           ('presence_over_zero', 'presence only\nminus zeroed', 'naive'),
           ('aggregate_expression', 'cohort mean\nminus zeroed', 'gnn_mean')]
    fig, axes = plt.subplots(1, 2, figsize=(W, 3.05), gridspec_kw={'width_ratios': [1.0, 1.18]})

    ax = axes[0]
    rows = [(lab, FAMB['versus_random_split'][k], fam) for k, lab, fam in ARMS if k in FAMB['versus_random_split']]
    for i, (lab, blk, fam) in enumerate(rows):
        y = len(rows) - 1 - i
        m, sdv, pc = blk['delta'], blk.get('sd') or 0.0, blk['per_cell']
        ax.barh(y, m, height=0.60, color=PAL[fam], edgecolor=INK, linewidth=0.5, zorder=2)
        ax.plot([m - sdv, m + sdv], [y, y], color=INK, lw=1.0, zorder=5)
        draw_cells(ax, y, pc, face='white', edge=INK, ms=2.6, lane=0.085)
    ax.axvline(0, color=INK, lw=0.8, zorder=3)
    ax.set_yticks(range(len(rows)))
    # the mean rides in the tick label, so no number sits on top of a per-cell marker
    ax.set_yticklabels([f'{lab}\n{signed(blk["delta"])}' for lab, blk, _ in rows][::-1])
    ax.set_xlabel('family-disjoint minus random split (AUROC)')
    _lo = min(min(b['per_cell'].values()) for _, b, _ in rows)
    ax.set_xlim(_lo - 0.012, 0.010)
    ax.set_ylim(-1.30, len(rows) - 0.42)         # the same vertical framing as panel b
    ax.legend(handles=fold_handles(), loc='lower left', frameon=False, handletextpad=0.4,
              borderpad=0.1, labelspacing=0.25, fontsize=6.8)
    ptitle(ax, 'a', 'the cost of withholding relatives')

    ax = axes[1]
    rows = [(lab, IVRB.get(k), FAMB.get(k), fam) for k, lab, fam in CON if FAMB.get(k)]
    H = 0.27
    _all = [v for _, rb, fb, _ in rows for blk in (rb, fb) if blk for v in blk['per_cell'].values()]
    _xt = max(_all) + 0.026                      # one column for the values, clear of every marker
    for i, (lab, rb, fb, fam) in enumerate(rows):
        y = len(rows) - 1 - i
        for blk, off, face in ((rb, +H, 'white'), (fb, -H, PAL[fam])):
            if not blk: continue
            m, sdv = blk['delta'], blk.get('sd') or 0.0
            ax.barh(y + off, m, height=0.50, color=face, edgecolor=INK, linewidth=0.5, zorder=2)
            ax.plot([m - sdv, m + sdv], [y + off, y + off], color=INK, lw=1.0, zorder=5)
            draw_cells(ax, y + off, blk['per_cell'], face='white' if face != 'white' else '#c8ccd2',
                       edge=INK, ms=2.4, lane=0.050)
            ax.text(_xt, y + off, signed(m), fontsize=6.8, ha='left', va='center', color=INK, zorder=6)
    ax.axvline(0, color=INK, lw=0.8, zorder=3)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows][::-1])
    ax.set_xlabel('paired difference in held-out AUROC')
    ax.set_xlim(min(_all) - 0.016, _xt + 0.036)
    ax.set_ylim(-1.30, len(rows) - 0.42)         # an empty strip under the last row for the key
    ptitle(ax, 'b', 'the contrasts on both splits')
    ax.legend(handles=[Line2D([], [], marker='s', ms=5, mfc='white', mec=INK, ls='none', label='random split'),
                       Line2D([], [], marker='s', ms=5, mfc='#9a9a9a', mec=INK, ls='none', label='family-disjoint split')],
              loc='lower left', frameon=False, handletextpad=0.4, borderpad=0.1, labelspacing=0.25,
              fontsize=6.8)
    fig.tight_layout(w_pad=1.2); fig.savefig(os.path.join(FIG, 'fig_family.png')); plt.close(fig)
    print('wrote fig_family.png')

# ---- Supplementary Figure S1: where the inputs and the label come from ------------------------------
# A flow of counts from the network to the label, every number read from results_p1.json and
# data/labels_ht29.json, so that the provenance figure cannot disagree with the text.
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
_C = R['cohort']
_LBJ = os.path.join(HERE, 'data', 'labels_ht29.json')
_LB = json.load(open(_LBJ)) if os.path.exists(_LBJ) else None
if _LB:
    fig, ax = plt.subplots(figsize=(W, 5.4)); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')
    TS, BS = 6.4, 5.7      # title and body font sizes, chosen so that no line outruns its box
    def box(x, y, w, h, title, body, face='white', edge=INK, lw=0.9):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.3,rounding_size=1.0',
                                    facecolor=face, edgecolor=edge, lw=lw, zorder=2))
        ax.text(x + w / 2, y + h - 1.8, title, ha='center', va='top', fontsize=TS, fontweight='bold', color=INK, zorder=3)
        ax.text(x + w / 2, y + 1.4, body, ha='center', va='bottom', fontsize=BS, color='#333', linespacing=1.3, zorder=3)
    def arrow(x0, y0, x1, y1, label=None, lx=None, ly=None, color='#555'):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=8, lw=0.9, color=color, zorder=1,
                                     shrinkA=0, shrinkB=0))
        if label: ax.text(lx, ly, label, fontsize=5.6, color='#444', ha='center', va='center', zorder=3,
                          bbox=dict(facecolor='white', edgecolor='none', pad=0.6))
    n = lambda v: f'{v:,}'
    ms = sorted(_LB['models'].values(), key=lambda v: v['n_reactions'])
    mm = sorted(v['n_matched'] for v in _LB['models'].values())
    # three columns of width 31: the network at x=1, the cohort at x=34.5, the label at x=68
    X0, X1, X2, BW = 1, 34.5, 68, 31
    # left column: the network
    box(X0, 86, BW, 12, 'Recon3D v3', f"{n(_C['n_reactions'])} reactions\n{n(_C['n_metabolites'])} metabolites")
    box(X0, 64, BW, 16, 'aligned to the BiGG export', f"{n(_C['n_reactions_aligned_to_json'])} placed by identifier\nor stoichiometric signature\n{n(_C['n_reactions_unaligned'])} keep gene table and\nstoichiometry only")
    box(X0, 44, BW, 14, 'gene rule', f"{n(_C['n_gpr_rule'])} with at least one gene\n{n(_C['n_no_gpr_rule'])} without")
    box(X0, 18, BW, 20, 'expression column', f"{n(_C['n_gene_and_expression'])} receive a value\n{n(_C['n_gene_no_expression'])} with a gene receive none\n{n(_C['n_expression_no_gene'])} without a gene receive one\nmask identical under\nthree definitions")
    xa = X0 + BW / 2
    arrow(xa, 86, xa, 80.3); arrow(xa, 64, xa, 58.3); arrow(xa, 44, xa, 38.3)
    # middle column: the cohort, feeding the expression column
    box(X1, 44, BW, 14, 'TCGA-COAD and TCGA-READ', f"{n(_C['n_patients'])} patients, one primary\ntumor sample each")
    arrow(X1 + 2, 44, X0 + BW + 0.3, 32, label='$\\log_2(\\mathrm{TPM}+1)$ through the\ngene rules: min over AND,\nmax over OR', lx=X1 + BW / 2 + 1, ly=33)
    # right column: the label
    box(X2, 86, BW, 12, 'Human1 NCI-60 models', f"{len(ms)} cell-line models, {n(ms[0]['n_reactions'])} to\n{n(ms[-1]['n_reactions'])} reactions each")
    box(X2, 64, BW, 16, 'matched to Recon3D', f"{n(mm[0])} to {n(mm[-1])} reactions per\nmodel, every matched reaction\nwith a nonzero bound")
    box(X2, 44, BW, 14, 'union label (the proxy)', f"{n(_LB['union_active'])} active of {n(_C['n_reactions'])} ({100 * _LB['union_prevalence']:.1f}%)\nidentical for every patient")
    box(X2, 18, BW, 20, 'colorectal-only label (HT29)', f"{n(_LB['ht29_active'])} active ({100 * _LB['ht29_prevalence']:.1f}%)\n{n(_LB['union_active_not_ht29'])} union reactions active\nin no colorectal model\nused in the label check only")
    xb = X2 + BW / 2
    arrow(xb, 86, xb, 80.3); arrow(xb, 64, xb, 58.3); arrow(xb, 44, xb, 38.3)
    # what a scorer is given: the network, the expression column and the shared label
    box(X1, 2, BW, 12, 'what a scorer is given', 'the network, one expression column\nper patient, and the shared label')
    arrow(X0 + BW, 22, X1 + 3, 14, color='#888'); arrow(X2, 48, X1 + BW - 3, 14, color='#888')
    fig.savefig(os.path.join(FIG, 'fig_provenance.png')); plt.close(fig)
    print('wrote fig_provenance.png')
