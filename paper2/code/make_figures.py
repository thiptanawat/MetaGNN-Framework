#!/usr/bin/env python3
"""Figures for the Language-Model Evidence manuscript. Every panel is drawn from results_frozen.json or from the
raw response files, so no figure can drift from the text."""
import json, collections
import _paths as PATHS
import os as _os
FIGDIR = _os.path.join(PATHS.ROOT,
                       'manuscript' if _os.path.isdir(_os.path.join(PATHS.ROOT, 'manuscript')) else 'ms',
                       'figures')
_os.makedirs(FIGDIR, exist_ok=True)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import roc_auc_score, roc_curve

BLUE, ORANGE, AQUA, VIOLET, RED = '#2a78d6', '#eb6834', '#1baf7a', '#4a3aa7', '#e34948'
GREY, LGREY = '#5b6470', '#d8dce2'
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8.5,
                     'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
                     'font.family': 'DejaVu Sans'})
R = json.load(open(PATHS.data('results_frozen.json')))
RUNS = R['runs']; W = 6.3
# the run directory behind each design's primary collection (the interleaved one where it exists)
COLL = R.get('collections') or {}
RUN_A = COLL.get('A_representative', 'repr'); RUN_MSI = (R.get('msi_collections') or {}).get('msi', 'msi')
def ptitle(ax, letter, t):
    ax.set_title(f'{letter}  {t}', loc='left', fontweight='bold', pad=6)

# ---------------------------------------------------------------- Figure 1: design
fig, ax = plt.subplots(figsize=(W, 2.5)); ax.axis('off')
ax.set_xlim(0, 10); ax.set_ylim(0, 4)
rows = [('reaction', 'ID, name, subsystem, equation, reversibility, GPR rule', 'no patient data', LGREY, GREY),
        ('patient',  'the same, plus this patient\'s own expression', 'value + percentile', '#cfe0f6', BLUE),
        ('shuffled', 'the same, plus a stranger\'s value, redrawn per reaction',
         'presented as this patient\'s', '#fbd9c9', ORANGE)]
for i, (nm, txt, note, fc, ec) in enumerate(rows):
    y = 3.0 - i * 1.05
    ax.add_patch(FancyBboxPatch((0.15, y - 0.38), 6.85, 0.76, boxstyle='round,pad=0.04',
                                fc=fc, ec=ec, lw=1.1))
    ax.text(0.38, y + 0.10, nm, fontsize=8.5, fontweight='bold', family='monospace', va='center')
    ax.text(0.38, y - 0.16, txt, fontsize=7.3, va='center')
    ax.text(7.2, y + 0.08, note, fontsize=7.4, va='center', color=ec, style='italic')
    ax.text(7.2, y - 0.18, {'reaction': 'identical prompt for every patient',
                             'patient': 'the patient\'s own measurement',
                             'shuffled': 'same format, a mosaic of strangers'}[nm],
            fontsize=6.8, va='center', color=GREY)
ax.annotate('', xy=(0.05, 3.42), xytext=(0.05, 0.55),
            arrowprops=dict(arrowstyle='-', lw=1.0, color=GREY))
ax.text(0.02, 3.72, 'One zero-shot call per reaction $\\times$ patient $\\times$ arm; temperature 0; '
                    'scored against a reconstruction-membership proxy',
        fontsize=7.4, color='black')
fig.savefig(FIGDIR + '/fig_design.png'); plt.close(fig)

# ---------------------------------------------------------------- Figure 2: accuracy ladder
# The designs drawn are the ones present in the frozen results, in this order.
ALL_TAGS = [('A_representative', 'A', 'Representative'), ('B_balanced', 'B', 'Label-balanced'),
            ('C_backbone2', 'C', 'Balanced, backbone two'), ('E_representative40', 'E', 'Representative'),
            ('A_backbone2', 'S', 'Second backbone'), ('A_backbone3', 'T', 'Third backbone')]
MODEL_NAME = {'qwen3-8-27b': 'Qwen3.8-27B', 'gemma-4-31b-it': 'Gemma 4 31B-it',
              'mistral-small-3.2-24b-instruct': 'Mistral Small 3.2 24B'}
DES = [(k, l, ti) for k, l, ti in ALL_TAGS if k in RUNS and 'patient_vs_shuffled' in RUNS[k]]
tags = [k for k, _, _ in DES]; lab = [l for _, l, _ in DES]
# up to four panels in one row; five or six go on two rows of three
_ncol = len(DES) if len(DES) <= 4 else 3; _nrow = 1 if len(DES) <= 4 else 2
fig, axes = plt.subplots(_nrow, _ncol, figsize=(W + 1.1 * max(0, _ncol - 3), 2.35 * _nrow + 0.25 * (_nrow - 1)),
                         sharey=True, squeeze=False)
axes = axes.reshape(-1)
for ax in axes[len(DES):]: ax.axis('off')
titles = {k: ti for k, _, ti in DES}
for k, (tag, ax) in enumerate(zip(tags, axes)):
    r = RUNS[tag]
    names = ['reaction\nonly', '+ own', "+ stranger's"]
    keys = ['reaction', 'patient', 'shuffled']
    vals = [r['per_arm'][a]['auroc'] for a in keys]
    los = [r['per_arm'][a]['auroc'] - r['per_arm'][a]['auroc_ci'][0] for a in keys]
    his = [r['per_arm'][a]['auroc_ci'][1] - r['per_arm'][a]['auroc'] for a in keys]
    ax.bar(range(3), vals, color=[GREY, BLUE, ORANGE], width=0.62,
           yerr=[los, his], error_kw=dict(lw=0.9, capsize=2.5, ecolor='#333'))
    ax.axhline(0.5, color='#999', lw=0.8, ls=':')
    ax.axhline(r['raw_expression_auroc'], color=AQUA, lw=1.4)
    ax.axhline(r['indicator_floor'], color=VIOLET, lw=1.4, ls='--')
    ax.set_xticks(range(3)); ax.set_xticklabels(names, fontsize=6.9)
    ax.set_ylim(0.30, 0.68)
    ax.set_title(f'{lab[k]}  {titles[tag]}', loc='left', fontweight='bold', pad=13)
    ax.text(0.0, 1.03, f"{r['n_patients']} patients, {MODEL_NAME.get(r['model'], r['model'])}",
            transform=ax.transAxes, fontsize=6.6, color=GREY, ha='left', va='bottom')
    for i, v in enumerate(vals):
        ax.text(i, v + his[i] + 0.010, f'{v:.3f}', ha='center', fontsize=7, zorder=6,
                bbox=dict(facecolor='white', edgecolor='none', pad=0.6))
for _r in range(_nrow): axes[_r * _ncol].set_ylabel('AUROC vs activity labels')
from matplotlib.lines import Line2D
fig.legend(handles=[Line2D([], [], color=AQUA, lw=1.4, label='raw expression, no model'),
                    Line2D([], [], color=VIOLET, lw=1.4, ls='--', label='availability baseline'),
                    Line2D([], [], color='#999', lw=0.8, ls=':', label='chance')],
           frameon=False, loc='lower center', ncol=3, fontsize=6.6, handlelength=1.6,
           bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(w_pad=1.2, rect=(0, 0.07 / _nrow, 1, 1)); fig.savefig(FIGDIR + '/fig_ladder.png'); plt.close(fig)
# the remaining multi-design panels keep the three main designs; the second backbone is compared
# with Design A in its own table
_keep = [(k, l) for k, l, _ in DES if k not in ('A_backbone2', 'A_backbone3', 'A_interleaved')]
tags = [k for k, _ in _keep]; lab = [l for _, l in _keep]

# ---------------------------------------------------------------- Figure 3: responsive vs informed
fig, axes = plt.subplots(1, 3, figsize=(W, 2.7), gridspec_kw=dict(width_ratios=[1.2, 1, 1]))
ax = axes[0]
x = np.arange(len(tags)); w = 0.36
# the patient-blind floor and the movement on the same reactions (expression-bearing), on a log axis so that
# the three floors and the movement can all be read; the two single-measurement floors are lines
det = [RUNS[t]['determinism'].get('mean_abs_diff_expr', RUNS[t]['determinism']['mean_abs_diff']) for t in tags]
mv = [RUNS[t]['patient_vs_shuffled']['mean_move_from_reaction_arm']['patient'] for t in tags]
ax.bar(x - w/2, det, w, color=LGREY, edgecolor=GREY, lw=0.8, label='patient-blind floor')
ax.bar(x + w/2, mv, w, color=BLUE, label='one value added')
# the two single-measurement floors were taken on Design A's reactions (the cross-session one on
# A's prompts re-answered in E; the replicate arm on its own stratified sample), so they are drawn
# over the designs that share those reactions, A and E, and not over Design B
_span = [(i - 0.45, i + 0.45) for i, t in enumerate(tags) if t in ('A_representative', 'E_representative40')]
xs_ = R.get('cross_session', {}).get('arms', {}).get('patient')
# the two floor values are written to the right of the last bar group, in the margin the extended
# x-range leaves, so that they do not sit on top of the Design E bars
_xr = len(tags) - 0.5 + 0.08
if xs_:
    for k, (x0, x1) in enumerate(_span):
        ax.hlines(xs_['mean_abs_diff_expr'], x0, x1, color=VIOLET, lw=1.1, ls='--',
                  label='cross-session floor' if k == 0 else None)
    ax.text(_xr, xs_['mean_abs_diff_expr'], f"{xs_['mean_abs_diff_expr']:.4f}",
            ha='left', va='center', fontsize=5.8, color=VIOLET)
fr_ = RUNS.get('F_replicate', {}).get('determinism_patient_prompt')
if fr_:
    for k, (x0, x1) in enumerate(_span):
        ax.hlines(fr_['mean_abs_diff'], x0, x1, color=AQUA, lw=1.1, ls=':',
                  label='within-session floor' if k == 0 else None)
    ax.text(_xr, fr_['mean_abs_diff'], f"{fr_['mean_abs_diff']:.4f}",
            ha='left', va='center', fontsize=5.8, color=AQUA)
ax.set_xlim(-0.6, len(tags) - 0.5 + 0.62)
for xi, m_ in enumerate(mv):
    ax.text(xi + w/2, m_ * 1.2, f'{m_:.2f}', ha='center', va='bottom', fontsize=6, color=BLUE)
ax.set_yscale('log'); ax.set_ylim(1e-4, 100)
from matplotlib.ticker import NullLocator
ax.yaxis.set_minor_locator(NullLocator())
ax.set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1]); ax.set_yticklabels(['0.0001', '0.001', '0.01', '0.1', '1'])
ax.set_xticks(x); ax.set_xticklabels(lab); ax.set_xlabel('design')
ax.set_ylabel('mean |change| in returned $p$ (log)')
ax.legend(frameon=False, loc='upper left', fontsize=5.9, handlelength=1.4, labelspacing=0.25, borderpad=0.1,
          bbox_to_anchor=(-0.02, 1.0))
ptitle(ax, 'A', 'Movement and floors')

ax = axes[1]
sr = [RUNS[t]['patient_vs_shuffled']['between_patient_sd_real'] for t in tags]
ss = [RUNS[t]['patient_vs_shuffled']['between_patient_sd_stranger'] for t in tags]
ax.bar(x - w/2, sr, w, color=BLUE, label="patient's own data")
ax.bar(x + w/2, ss, w, color=ORANGE, label="a stranger's data")
ax.set_xticks(x); ax.set_xticklabels(lab); ax.set_xlabel('design')
ax.set_ylabel('between-patient s.d. of scores')
ax.set_ylim(0, 0.42); ax.legend(frameon=False, loc='upper left', fontsize=6.6, handlelength=1.2)
ptitle(ax, 'B', 'Own vs. stranger')

ax = axes[2]
rr = [RUNS[t]['per_arm']['reaction']['median_inter_patient_r'] for t in tags]
pp = [RUNS[t]['per_arm']['patient']['median_inter_patient_r'] for t in tags]
sh = [RUNS[t]['per_arm']['shuffled']['median_inter_patient_r'] for t in tags]
for i, (a, b, c) in enumerate(zip(rr, pp, sh)):
    ax.plot([i - 0.16, i, i + 0.16], [a, b, c], color=LGREY, lw=1.0, zorder=1)
ax.scatter(x - 0.16, rr, s=26, color=GREY, zorder=3, label='reaction only')
ax.scatter(x, pp, s=26, color=BLUE, zorder=3, label='+ own')
ax.scatter(x + 0.16, sh, s=26, color=ORANGE, zorder=3, label="+ stranger's")
ax.set_xticks(x); ax.set_xticklabels(lab); ax.set_xlabel('design')
# the statistic is taken on the expression-bearing reactions, where the patient prompts differ, so the
# two data-bearing arms sit near 0.1-0.2 and the identical-prompt arm at its ceiling of 1
ax.set_ylabel('median inter-patient correlation'); ax.set_ylim(0, 1.22)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.legend(frameon=False, loc='upper left', fontsize=6.4, handlelength=0.7, handletextpad=0.4,
          labelspacing=0.25, borderpad=0.1, ncol=3, columnspacing=0.6,
          bbox_to_anchor=(-0.01, 1.03))
ptitle(ax, 'C', 'Inter-patient agreement')
fig.tight_layout(w_pad=2.0); fig.savefig(FIGDIR + '/fig_responsive.png'); plt.close(fig)

# ---------------------------------------------------------------- Figure 4: presence vs magnitude
D = np.load(PATHS.data('rxn_context.npz'), allow_pickle=True)
XD = D['X']; HASD = D['has']       # read once: indexing the archive re-reads the array every time
fig, axes = plt.subplots(1, 4, figsize=(W + 1.4, 2.3), gridspec_kw=dict(width_ratios=[1.15, 0.95, 1, 1]))
ax = axes[0]
# The two columns are collinear by construction, so the stacked shares depend on the entry
# order; the panel shows the whole-model R-squared and, beside it, the gain the magnitude buys
# within expression-bearing reactions, where presence is constant and the question is well posed.
base = [RUNS[t]['variance_decomposition']['r2_reaction_identity'] for t in tags]
full = [RUNS[t]['variance_decomposition']['r2_plus_magnitude'] for t in tags]
wb = [RUNS[t]['variance_decomposition']['expr_only_r2_reaction_identity'] for t in tags]
wm = [RUNS[t]['variance_decomposition']['expr_only_r2_plus_magnitude'] for t in tags]
wp = [RUNS[t]['variance_decomposition'].get('expr_only_r2_plus_percentile') for t in tags]
bw = 0.27
ax.bar(x - bw, base, bw, color=LGREY, edgecolor=GREY, lw=0.7, label='identity only')
ax.bar(x - bw, np.array(full) - np.array(base), bw, bottom=base, color=VIOLET, label='+ presence and value')
ax.bar(x, wb, bw, color='#e8e0f5', edgecolor=VIOLET, lw=0.7, label='identity, expression-bearing only')
ax.bar(x, np.array(wm) - np.array(wb), bw, bottom=wb, color=RED, label='+ value, presence fixed')
# the same restricted fit with the printed percentile in place of the value
if all(v is not None for v in wp):
    ax.bar(x + bw, wb, bw, color='#e8e0f5', edgecolor=VIOLET, lw=0.7)
    ax.bar(x + bw, np.array(wp) - np.array(wb), bw, bottom=wb, color='#c0392b', hatch='///', edgecolor='white', lw=0,
           label='+ percentile, presence fixed')
ax.set_xticks(x); ax.set_xticklabels(lab); ax.set_xlabel('design')
ax.set_ylabel('$R^2$ of the returned score'); ax.set_ylim(0, 1.32); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.legend(frameon=False, loc='upper left', fontsize=5.4, handlelength=1.1, ncol=1, labelspacing=0.2)
ptitle(ax, 'A', 'Variance explained')

ax = axes[1]
r = RUNS['A_representative']['strata']
gp = [r['reaction']['expression_bearing']['mean_p'], r['patient']['expression_bearing']['mean_p']]
np_ = [r['reaction']['no_expression']['mean_p'], r['patient']['no_expression']['mean_p']]
ax.plot([0, 1], gp, 'o-', color=BLUE, lw=1.6, ms=6, label='reaction receives a value')
ax.plot([0, 1], np_, 'o-', color=ORANGE, lw=1.6, ms=6, label='reaction receives none')
for _x, _v, _c in ((0, gp[0], BLUE), (1, gp[1], BLUE), (0, np_[0], ORANGE), (1, np_[1], ORANGE)):
    ax.text(_x, _v + 0.034, f'{_v:.3f}', ha='center', va='bottom', fontsize=7, color=_c)
ax.set_xticks([0, 1]); ax.set_xticklabels(['reaction\nonly', '+ patient\ncontext'])
ax.set_xlim(-0.35, 1.35); ax.set_ylim(0, 0.75)
ax.set_ylabel('mean returned $p$(active)')
ax.legend(frameon=False, loc='upper left', fontsize=6.4, handlelength=1.2, labelspacing=0.25)
ptitle(ax, 'B', 'Presence of a value')

raw = json.load(open(PATHS.run(RUN_A, 'raw.json'))); des = json.load(open(PATHS.run(RUN_A, 'design.json')))
IDX = np.array(des['reactions']); PSEL = np.array(des['patients']); hx = HASD[IDX]
# the cohort percentile printed beside the value: the argsort rank within the cohort, as the prompt builder computes it
import cohort_order as _CO
PCTF = _CO.percentile_as_collected(XD)
xs, ps, ys = [], [], []
for a, pj in enumerate(PSEL):
    for b, ri in enumerate(IDX):
        if not hx[b]: continue
        v = raw.get(f'patient|{pj}|{ri}')
        if v and v['prob'] is not None:
            xs.append(XD[pj, ri]); ps.append(PCTF[pj, ri]); ys.append(v['prob'])
xs, ps, ys = np.array(xs), np.array(ps), np.array(ys)
for ax, xv, xlabel, sp, letter, title in (
        (axes[2], xs, 'expression value shown, $\\log_2(\\mathrm{CPM}+1)$',
         RUNS['A_representative']['score_vs_shown_value']['spearman'], 'C', 'Level of the value'),
        (axes[3], ps, 'cohort percentile shown',
         RUNS['A_representative']['variance_decomposition'].get('expr_only_spearman_percentile'), 'D', 'Percentile of the value')):
    ax.scatter(xv, ys, s=3, alpha=0.13, color=BLUE, linewidths=0)
    bins = np.quantile(xv, np.linspace(0, 1, 11))
    ctr = [(bins[i] + bins[i+1]) / 2 for i in range(10)]
    med = [np.median(ys[(xv >= bins[i]) & (xv <= bins[i+1])]) for i in range(10)]
    ax.plot(ctr, med, 'o-', color=RED, lw=1.6, ms=4, label='decile median')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('returned $p$(active)')
    ax.set_ylim(-0.03, 1.38); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # the correlation and the series label are stacked in the empty band above the scatter, not on top
    # of the points
    if sp is not None:
        ax.text(0.03, 0.98, f'Spearman $\\rho$ = {sp:+.3f}', transform=ax.transAxes, ha='left',
                va='top', fontsize=7)
    ax.legend(frameon=False, loc='upper left', fontsize=6.8, handlelength=1.4, borderpad=0.1,
              handletextpad=0.5, bbox_to_anchor=(0.0, 0.88))
    ptitle(ax, letter, title)
fig.tight_layout(w_pad=1.4); fig.savefig(FIGDIR + '/fig_presence.png'); plt.close(fig)

# ---------------------------------------------------------------- Figure 5: MSI
M = R['msi']
raw = json.load(open(PATHS.run(RUN_MSI, 'raw.json'))); des = json.load(open(PATHS.run(RUN_MSI, 'design.json')))
keep = np.array(des['keep']); Y = np.array(des['y']); PANEL = np.array(des['panel'])
own = np.array([raw[f'own|{int(p)}']['val'] for p in keep], float)
swp = np.array([raw[f'swap|{int(p)}']['val'] for p in keep], float)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
F = XD[np.ix_(keep, PANEL)]
oof = np.zeros(len(Y)); pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000))
for tr, te in StratifiedKFold(5, shuffle=True, random_state=2024).split(F, Y):
    oof[te] = pipe.fit(F[tr], Y[tr]).predict_proba(F[te])[:, 1]
# the same reference fitted on the swapped panels the model was shown (archived donor of every swap
# record), the power argument of the text: the swap removes the signal a learner can use
_src = [raw.get(f'swap|{int(p)}', {}).get('src') for p in keep]
oofs = ooff = None
if all(v is not None for v in _src):
    Fs = XD[np.ix_([int(v) for v in _src], PANEL)]; oofs = np.zeros(len(Y)); ooff = np.zeros(len(Y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=2024).split(Fs, Y):
        oofs[te] = pipe.fit(Fs[tr], Y[tr]).predict_proba(Fs[te])[:, 1]
        # the frozen swap diagnostic: the reference fitted on the training patients' own panels and
        # applied unchanged to the held-out patients' donor panels (msi_stats.logistic_auroc_frozen_swap)
        ooff[te] = pipe.fit(F[tr], Y[tr]).predict_proba(Fs[te])[:, 1]

_bb = [(k, nm) for k, nm in (('msi', 'Qwen'), ('msi_backbone2', 'Gemma'), ('msi_backbone3', 'Mistral')) if R.get(k)]
fig, axes = plt.subplots(1, 4, figsize=(W + 2.4, 2.4), gridspec_kw=dict(width_ratios=[1.45, 1, 0.95, 1.1]))
ax = axes[0]
aucs = {}
series = [(oof, AQUA, 'logistic', '-'), (own, BLUE, 'own', '-'), (swp, ORANGE, 'swap', '-')]
if ooff is not None: series.insert(1, (ooff, AQUA, 'logistic_frozen', '--'))
for s, c, nm, ls_ in series:
    fpr, tpr, _ = roc_curve(Y, s)
    aucs[nm] = roc_auc_score(Y, s)
    lbl = {'logistic': f"LR, own panels {aucs[nm]:.3f}",
           'logistic_frozen': f"LR frozen, strangers' {aucs[nm]:.3f}",
           'own': f"LLM, own panel {aucs[nm]:.3f}",
           'swap': f"LLM, stranger's {aucs[nm]:.3f}"}[nm]
    ax.plot(fpr, tpr, color=c, lw=1.6, ls=ls_, label=lbl)
ax.plot([0, 1], [0, 1], ls=':', color='#999', lw=0.9)
ax.set_xlabel('false positive rate'); ax.set_ylabel('true positive rate')
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
ax.legend(frameon=False, loc='lower right', fontsize=5.6, handlelength=1.2, labelspacing=0.25,
          borderpad=0.1, handletextpad=0.4, bbox_to_anchor=(1.05, -0.03))
ptitle(ax, 'A', 'MSI status, ROC')

ax = axes[1]
c = collections.Counter(np.round(own, 3).tolist())
ks = sorted(c); vs = [c[k] for k in ks]
ax.bar([str(k) for k in ks], vs, color=BLUE, width=0.62)
ax.set_xlabel('probability returned'); ax.set_ylabel(f'patients (of {len(Y)})')
ax.set_ylim(0, len(Y) * 1.12)
ax.text(0.97, 0.99, f'{M["arms"]["own"]["n_distinct_values"]} distinct values\n'
                    f'{100*M["arms"]["own"]["modal_share"]:.1f}% take one value',
        transform=ax.transAxes, ha='right', va='top', fontsize=7.2)
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
ptitle(ax, 'B', 'Probabilities returned')

ax = axes[2]
# the patients who all received the modal probability, and how many different rationales
# accompanied that one number
n_r, n_d = M['reason_n'], M['reason_distinct']
n_c, n_cd = M['reason_complete_n'], M['reason_complete_distinct']
ax.bar([0], [n_r], color=BLUE, width=0.55)
ax.bar([1], [n_d], color=RED, width=0.55)
ax.set_xticks([0, 1])
ax.set_xticklabels([f'patients at\n$p$ = {M["arms"]["own"]["modal_value"]:.2f}', 'distinct\nrationales'])
for xi, v in zip([0, 1], [n_r, n_d]):
    ax.text(xi, v + n_r * 0.02, str(v), ha='center', fontsize=8, fontweight='bold')
ax.set_ylabel('count'); ax.set_ylim(0, n_r * 1.22)
if M.get('full_archive'):
    _note = f'all {M.get("own_finish_n", n_r)} replies archived in full;\n{M.get("own_finish_length_n", 0)} cut by the token cap'
else:
    _note = f'{n_c} complete within the archived prefix,\n{n_cd} of them distinct'
ax.text(0.5, -0.36, _note, transform=ax.transAxes, ha='center', va='top', fontsize=6.2, color=GREY)
ptitle(ax, 'C', 'Rationales')

# the own and stranger's arms of every backbone on this target, with patient-level bootstrap 95% intervals
ax = axes[3]
for i, (k, nm) in enumerate(_bb):
    for j, (arm, col) in enumerate((('own', BLUE), ('swap', ORANGE))):
        v = R[k]['arms'][arm]
        xi = i + (-0.17 if j == 0 else 0.17)
        ax.errorbar(xi, v['auroc'], yerr=[[v['auroc'] - v['auroc_ci'][0]], [v['auroc_ci'][1] - v['auroc']]],
                    fmt='o', color=col, ms=4.5, capsize=2.5, lw=1.1, label=('own panel' if j == 0 else "stranger's") if i == 0 else None)
ax.axhline(0.5, color='#999', lw=0.8, ls=':')
ax.axhline(M['logistic_auroc'], color=AQUA, lw=1.2)
ax.text(len(_bb) - 0.45, M['logistic_auroc'] - 0.035, 'LR, own panels', color=AQUA, fontsize=6.2, ha='right', va='top')
ax.set_xticks(range(len(_bb))); ax.set_xticklabels([nm for _, nm in _bb], fontsize=7)
ax.set_xlim(-0.6, len(_bb) - 0.4); ax.set_ylim(0.2, 1.2); ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_ylabel('AUROC, MSI-H vs MSS')
ax.legend(frameon=False, loc='upper left', fontsize=6.0, handlelength=1.0, labelspacing=0.25, ncol=2,
          bbox_to_anchor=(-0.03, 1.04), columnspacing=0.8, handletextpad=0.4, borderpad=0.0)
ptitle(ax, 'D', 'Backbones')
fig.tight_layout(w_pad=1.4); fig.savefig(FIGDIR + '/fig_msi.png'); plt.close(fig)

# ---------------------------------------------------------------- Supplementary figure: reliability
# Reliability diagrams of the returned probability against the assay call on every backbone: each
# distinct returned probability is one marker at (returned probability, observed MSI-H rate among
# the patients who received it), sized by that count, for the own-panel and stranger's-panel arms;
# the diagonal is perfect calibration and the horizontal line the prevalence (the constant predictor).
fig, axes = plt.subplots(1, len(_bb), figsize=(2.3 * len(_bb) + 0.6, 2.5), squeeze=False)
for i, (k, nm) in enumerate(_bb):
    ax = axes[0, i]; Mk = R[k]
    ax.plot([0, 1], [0, 1], color='#999', lw=0.8, ls=':', label='perfect calibration' if i == 0 else None)
    ax.axhline(Mk['prevalence'], color=GREY, lw=0.9, ls='--', label='prevalence (constant predictor)' if i == 0 else None)
    for arm, col, dx in (('own', BLUE, -0.008), ('swap', ORANGE, 0.008)):
        rel = Mk['arms'][arm].get('reliability') or []
        ps = np.array([r['p'] for r in rel]); ob = np.array([r['observed'] for r in rel]); ns = np.array([r['n'] for r in rel])
        ax.scatter(ps + dx, ob, s=6 + 60 * ns / max(ns.max(), 1), color=col, alpha=0.75, edgecolor='white', lw=0.4,
                   label=('own panel' if arm == 'own' else "stranger's panel") if i == 0 else None, zorder=3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xticks([0, 0.25, 0.5, 0.75, 1]); ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xlabel('returned probability'); ax.set_ylabel('observed MSI-H rate' if i == 0 else '')
    ptitle(ax, 'ABC'[i], nm)
axes[0, 0].legend(frameon=False, loc='upper left', fontsize=6.0, handlelength=1.4, labelspacing=0.3, borderpad=0.2,
                  scatterpoints=1, markerscale=0.9)
fig.tight_layout(w_pad=1.2); fig.savefig(FIGDIR + '/fig_reliability.png'); plt.close(fig)
# ---------------------------------------------------------------- Figure: the own-minus-donor contrast across designs
# The own-minus-donor contrast of every collection of the main backbone that has both arms, one row
# each: the mean paired difference in per-patient AUROC with its 90% Student interval, as
# stats.analyze computes it. The rows of the prespecified replication are read from
# results/replication_frozen.json, so the figure is drawn only where that file exists. The shaded
# band is the narrower equivalence margin (0.02) and the dotted lines mark the wider one (0.05).
_gp = _os.path.join(PATHS.ROOT, 'results', 'replication_frozen.json')
if _os.path.exists(_gp):
    _G = json.load(open(_gp)); _W1 = _G['wording1']; _rows = []
    for _k, _lab in (('A_representative', 'Design A, schedule 1'), ('A_donor2', 'Design A, schedule 2'),
                     ('B_balanced', 'Design B'), ('E_representative40', 'Design E, schedule 1'),
                     ('E_donor2', 'Design E, schedule 2')):
        _v = RUNS.get(_k)
        if _v and _v.get('patient_vs_shuffled'):
            _pv = _v['patient_vs_shuffled']
            _rows.append((f"{_lab} (n = {_v['n_patients']})", _pv['auroc_diff_mean'], _pv['auroc_diff_ci90'], 'earlier'))
    _n1 = _W1['n_patients']
    for _s, _lab in (('d1', 'Replication, schedule 1, primary'), ('d2', 'Replication, schedule 2'),
                     ('d3', 'Replication, schedule 3')):
        if _s in _W1['donor']:
            _d = _W1['donor'][_s]
            _rows.append((f"{_lab} (n = {_n1})", _d['diff_mean'], _d['diff_ci90'], 'primary' if _s == 'd1' else 'replication'))
    if 'pooled_donor' in _W1:
        _rows.append((f"Replication, mean of three donor arms (n = {_n1})", _W1['pooled_donor']['mean'],
                      _W1['pooled_donor']['ci90'], 'replication'))
    if 'wording2' in _G:
        _d = _G['wording2']['donor']['d1']
        _rows.append((f"Replication, second wording (n = {_G['wording2']['n_patients']})", _d['diff_mean'], _d['diff_ci90'], 'replication'))
    # the printed numbers of the replication ranked with no model (replication_posthoc.py; exploratory)
    _xp = _os.path.join(PATHS.ROOT, 'results', 'replication_posthoc.json')
    if _os.path.exists(_xp):
        _XR = json.load(open(_xp))['replication']
        for _k, _lab in (('printed_percentile', 'Printed percentile, no model, schedule 1'), ('printed_value', 'Printed value, no model, schedule 1')):
            _d = _XR[_k]['d1']['diff']
            _rows.append((f"{_lab} (n = {_d['n']})", _d['mean'], _d['ci90'], 'reference'))
    fig, ax = plt.subplots(figsize=(W, 0.27 * len(_rows) + 0.85))
    ax.axvspan(-0.02, 0.02, color=LGREY, alpha=0.6, lw=0, zorder=0)
    for _m in (-0.05, 0.05): ax.axvline(_m, color=GREY, lw=0.8, ls=':', zorder=1)
    ax.axvline(0, color='black', lw=0.8, zorder=1)
    for _i, (_lab, _mu, _ci, _kind) in enumerate(_rows):
        _y = len(_rows) - 1 - _i; _col = {'earlier': GREY, 'reference': ORANGE}.get(_kind, BLUE)
        ax.plot(_ci, [_y, _y], color=_col, lw=1.2 if _kind == 'earlier' else 1.7, solid_capstyle='butt', zorder=2)
        ax.plot([_mu], [_y], marker='D' if _kind == 'primary' else 'o', ms=5.2 if _kind == 'primary' else 4.2,
                color=_col, mec='white', mew=0.5, zorder=3)
    ax.set_yticks(range(len(_rows))); ax.set_yticklabels([_r[0] for _r in _rows][::-1])
    for _t, _r in zip(ax.get_yticklabels(), _rows[::-1]):
        if _r[3] == 'primary': _t.set_fontweight('bold')
    ax.axhline(len(_rows) - sum(1 for _r in _rows if _r[3] == 'earlier') - 0.5, color=LGREY, lw=0.8)
    if any(_r[3] == 'reference' for _r in _rows):
        ax.axhline(sum(1 for _r in _rows if _r[3] == 'reference') - 0.5, color=LGREY, lw=0.8)
    _lim = max(0.06, max(max(abs(_r[2][0]), abs(_r[2][1])) for _r in _rows) + 0.006)
    ax.set_xlim(-_lim, _lim); ax.set_ylim(-0.6, len(_rows) - 0.4); ax.tick_params(axis='y', length=0)
    ax.set_xlabel('own minus donor, AUROC (mean and 90% interval)')
    fig.tight_layout(); fig.savefig(FIGDIR + '/fig_replication.png'); plt.close(fig)

# ---------------------------------------------------------------- Graphical abstract
# The design and the patient-level result in one landscape image of fixed size (5:2, 2656 x 1062
# pixels), drawn from the same results file as the figures: the three prompt arms on the left, the
# own-panel and stranger's-panel AUROC of each backbone on the right.
fig = plt.figure(figsize=(13.28, 5.31))
ax = fig.add_axes([0.015, 0.03, 0.56, 0.94]); ax.axis('off'); ax.set_xlim(0, 10); ax.set_ylim(0, 4.6)
ax.text(0.15, 4.28, 'Three prompts that differ in one line', fontsize=19, fontweight='bold', va='center')
for i, (nm, txt, note, fc, ec) in enumerate(
        [('reaction', 'the reaction, with no patient data', 'patient-blind', LGREY, GREY),
         ('patient', 'plus this patient\'s own expression', 'own input', '#cfe0f6', BLUE),
         ('shuffled', 'plus a stranger\'s expression,\npresented as this patient\'s', 'donor input', '#fbd9c9', ORANGE)]):
    y = 3.35 - i * 1.06
    ax.add_patch(FancyBboxPatch((0.15, y - 0.41), 7.1, 0.82, boxstyle='round,pad=0.04', fc=fc, ec=ec, lw=2.0))
    ax.text(0.42, y, nm, fontsize=17, fontweight='bold', family='monospace', va='center')
    ax.text(2.55, y, txt, fontsize=14.5, va='center', linespacing=1.15)
    ax.text(7.5, y, note, fontsize=15, va='center', color=ec, style='italic', fontweight='bold')
ax.text(0.15, 0.30, 'Does the answer depend on whose values the model is shown?', fontsize=15.5, color='black', va='center')

ax = fig.add_axes([0.665, 0.17, 0.315, 0.60])
for i, (k, nm) in enumerate(_bb):
    for j, (arm, col) in enumerate((('own', BLUE), ('swap', ORANGE))):
        v = R[k]['arms'][arm]
        ax.errorbar(i + (-0.17 if j == 0 else 0.17), v['auroc'],
                    yerr=[[v['auroc'] - v['auroc_ci'][0]], [v['auroc_ci'][1] - v['auroc']]],
                    fmt='o', color=col, ms=11, capsize=6, lw=2.4, capthick=2.4,
                    label=('own panel' if j == 0 else "stranger's panel") if i == 0 else None)
ax.axhline(0.5, color='#999', lw=1.6, ls=':')
ax.axhline(M['logistic_auroc'], color=AQUA, lw=2.6)
ax.text(len(_bb) - 0.45, M['logistic_auroc'] - 0.04, 'supervised reference', color=AQUA, fontsize=13.5, ha='right', va='top')
ax.set_xticks(range(len(_bb))); ax.set_xticklabels([nm for _, nm in _bb], fontsize=15)
ax.set_xlim(-0.6, len(_bb) - 0.4); ax.set_ylim(0.2, 1.0); ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.tick_params(axis='y', labelsize=13.5, width=1.4, length=5); ax.tick_params(axis='x', width=1.4, length=5)
for sp in ('left', 'bottom'): ax.spines[sp].set_linewidth(1.4)
ax.set_ylabel('AUROC, MSI-H against MSS', fontsize=14.5)
ax.legend(frameon=False, loc='lower left', fontsize=13.5, handlelength=1.0, ncol=2, columnspacing=1.0,
          handletextpad=0.3, borderpad=0.0, bbox_to_anchor=(-0.02, 1.01))
fig.text(0.612, 0.93, 'Patient-level target: MSI status', fontsize=19, fontweight='bold', va='center')
fig.text(0.612, 0.045, 'Three language models, the same 40-reaction panel', fontsize=13, color=GREY, va='center')
with plt.rc_context({'savefig.bbox': 'standard'}):      # the canvas is the image: no tight cropping here
    fig.savefig(FIGDIR + '/graphical_abstract.png', dpi=200)
plt.close(fig)
print('figures written')
