#!/usr/bin/env python3
"""Emit numbers.tex: one LaTeX macro per quantity quoted in the manuscript.

No result from this study's own runs is typed into the manuscript by hand. Almost every macro below
is written from results_frozen.json, which stats.py computes from the raw model responses. A few
quantities that describe the free-text replies or the ground-truth mix of a control are read back
directly from the archived response files, which are the same artifacts stats.py reads. Rebuilding
the paper after a rerun therefore updates the text automatically, and a number that no longer has a
run behind it fails the build rather than surviving as stale prose.
"""
import json, os, re, sys

# Every optional macro family below is emitted only when its results file is present, and the
# manuscript's \ifdefined guards then silently drop the passages that quote it. A missing results
# file would therefore delete manuscript text instead of failing the build, which is the one failure
# mode the guard convention cannot catch by itself. The files the current manuscript depends on are
# named here and their absence is fatal, so a partial build has to be asked for on purpose.
REQUIRED = [
    ('results_frozen.json',           'code/stats.py and code/msi_stats.py'),
    ('results/msi2_frozen.json',      'code/msi2_stats.py'),
    ('results/crossed_boot_10k.json', 'code/crossed_boot_10k.py'),
    ('results/tie_audit.json',        'code/tie_audit.py'),
]
_absent = [(q, g) for q, g in REQUIRED
           if not (os.path.exists(q) or os.path.exists(os.path.basename(q)))]
if _absent and os.environ.get('P2_ALLOW_PARTIAL') != '1':
    for q, g in _absent:
        print(f'missing required results file: {q}  (regenerate with {g})', file=sys.stderr)
    print('These feed \\ifdefined-guarded passages that would be dropped from the PDF with no other '
          'warning. Set P2_ALLOW_PARTIAL=1 to build a deliberately partial draft.', file=sys.stderr)
    sys.exit(1)

R = json.load(open('results_frozen.json'))
L = []

# The archive keeps the per-run response files under results/ in the released tree and beside the
# scripts in a live working tree; accept either so the same driver serves both.
def rawpath(run, name='raw.json'):
    for p in (os.path.join('results', run, name), os.path.join(run, name)):
        if os.path.exists(p): return p
    return None

def rawload(run, name='raw.json'):
    p = rawpath(run, name)
    return json.load(open(p)) if p else None

# a checkout that carries the analysis without the manuscript sources has no such directory;
# the macro file is a result and is written either way
OUTFILE = 'ms/numbers.tex' if os.path.isdir('ms') else 'manuscript/numbers.tex'
os.makedirs(os.path.dirname(OUTFILE), exist_ok=True)
def cmd(name, val, fmt='{:.4f}'):
    import math
    if isinstance(val, str): s = val
    elif val is None or (isinstance(val, float) and math.isnan(val)): s = 'n/a'
    elif fmt == 'int': s = f'{val:,}'.replace(',', '{,}')
    elif fmt == 'pct': s = f'{100*val:.1f}'
    elif fmt == 'p': s = ('< 0.0001' if val < 1e-4 else f'= {val:.4f}')   # carries its own relation sign
    else: s = fmt.format(val)
    L.append(f'\\newcommand{{\\{name}}}{{{s}}}')

N = R['network']
cmd('netRxn', N['n_reactions'], 'int'); cmd('netActive', N['n_active'], 'int')
cmd('netBase', N['base_rate'], '{:.3f}'); cmd('netExpr', N['n_expression_bearing'], 'int')
cmd('netPat', N['n_patients'], 'int'); cmd('netPatOther', N['n_patients'] - 1, 'int')
cmd('netPrevExpr', N['prevalence_expression_bearing'], 'pct')
cmd('netPrevNo', N['prevalence_no_expression'], 'pct')
cmd('netIndicator', N['indicator_floor']); cmd('netRawExpr', N['raw_expression_auroc'])
cmd('netRawExprSD', N['raw_expression_auroc_sd'])

C = R.get('cohort')
if C:
    cmd('netGpr', C['n_gpr_rule'], 'int'); cmd('netNoGpr', C['n_no_gpr_rule'], 'int')
    cmd('msiL', C['msi_l'], 'int'); cmd('msiNE', C['not_evaluable'], 'int')

TAGS = {'A_representative': 'A', 'B_balanced': 'B', 'C_backbone2': 'C', 'D_reasoning': 'D',
        'E_representative40': 'E', 'F_replicate': 'F', 'A_backbone2': 'S',   # S: Design A on the second backbone
        'A_backbone3': 'T', 'A_interleaved': 'I',
        # the original blocked-order collections of the designs whose primary evidence is interleaved
        'A_blocked': 'Ao', 'E_blocked': 'Eo', 'A_backbone2_blocked': 'So', 'A_backbone3_blocked': 'To',
        # Design A on the main backbone under a second, independently drawn donor schedule
        'A_donor2': 'Ad',
        # Design E under the same second donor schedule
        'E_donor2': 'Ed'}
# the blocked collection of Design E, whichever tag carries it (E_blocked once the interleaved
# collection exists, otherwise the design itself): the endpoint-interruption section reads it as rEb*
if 'E_blocked' in R['runs']: R['runs']['E_blockedref'] = R['runs']['E_blocked']
elif 'E_representative40' in R['runs']: R['runs']['E_blockedref'] = R['runs']['E_representative40']
TAGS['E_blockedref'] = 'Eb'
# which collection is primary for each design: a flag macro per design whose primary is interleaved
for _tag, _out in (R.get('collections') or {}).items():
    if _tag in TAGS and _tag not in (R.get('collection_fallback') or {}) and (_out.endswith('il') or _out.endswith('il_d2')):
        cmd(f'r{TAGS[_tag]}Interleaved', 'yes')                            # T: third backbone; I: interleaved arms
if C:
    for k, n in (('n_gene_and_expression', 'netGeneExpr'), ('n_gene_no_expression', 'netGeneNoExpr'),
                 ('n_expression_no_gene', 'netExprNoGene'), ('n_aligned', 'netAligned'), ('n_unaligned', 'netUnaligned')):
        if k in C: cmd(n, C[k], 'int')
# lowest parse rate over the nine arms of the three main reaction-activity designs
_pm = [R['runs'][k]['per_arm'][a]['parsed'] for k in ('A_representative', 'B_balanced', 'E_representative40')
       for a in ('reaction', 'patient', 'shuffled') if k in R['runs'] and 'parsed' in R['runs'][k]['per_arm'].get(a, {})]
if _pm: cmd('rMainParsedMin', min(_pm), 'pct')
_rs = {k: R['runs'][k].get('resent_calls', 0) for k in ('A_representative', 'B_balanced', 'E_representative40') if k in R['runs']}
if _rs:
    cmd('mainResentPhrase', 'none of the three collections contains a re-sent call' if not any(_rs.values())
        else 'the re-sent calls of the three collections number ' + ', '.join(f"{v} in Design~{TAGS[k]}" for k, v in _rs.items()))
# the expression-bearing stratum of the three main designs: whether every arm sits below one half
# there, and the smallest raw-expression ranking on the same stratum
_eb = [(R['runs'][k]['per_arm'][a].get('auroc_expr'), R['runs'][k]['raw_expression_auroc_expr'], R['runs'][k]['per_arm'][a]['auroc'] - R['runs'][k]['raw_expression_auroc_expr'])
       for k in ('A_representative', 'B_balanced', 'E_representative40') for a in ('reaction', 'patient', 'shuffled')
       if k in R['runs'] and a in R['runs'][k]['per_arm'] and R['runs'][k]['per_arm'][a].get('auroc_expr') is not None]
if _eb:
    cmd('ebArmsBelowHalfPhrase', 'every arm falls below one half in each of the three designs on the main backbone' if all(v < 0.5 for v, _, _ in _eb)
        else 'not every arm falls below one half on the main backbone')
    # the same stratum on the further backbones, as one sentence per backbone chosen from the numbers
    _ebb = []
    for k, nm in (('A_backbone2', 'second'), ('A_backbone3', 'third')):
        r = R['runs'].get(k)
        if not r or any(r['per_arm'].get(a, {}).get('auroc_expr') is None for a in ('reaction', 'patient', 'shuffled')): continue
        b, p, sw = (r['per_arm'][a]['auroc_expr'] for a in ('reaction', 'patient', 'shuffled'))
        if max(b, p, sw) < 0.5:
            _ebb.append(f"on the {nm} backbone every arm sits below one half there as well ({b:.4f}, {p:.4f} and {sw:.4f})")
        elif b >= 0.5 and b > p and b > sw:
            _ebb.append(f"on the {nm} backbone the patient-blind arm reaches {b:.4f} on those reactions, above one half and above both "
                        f"patient arms ({p:.4f} own, {sw:.4f} stranger's), so there the added value lowers the expression-bearing ranking as well as the full-set one")
        else:
            _ebb.append(f"on the {nm} backbone the three arms sit at {b:.4f}, {p:.4f} and {sw:.4f} on those reactions, not all below one half")
    if _ebb: cmd('ebBackbonesSentence', 'The same stratum on the further backbones of Section~\\ref{sec:backbone2} reads as follows: ' + '; '.join(_ebb) + '.')
    cmd('ebRawMin', min(r for _, r, _ in _eb))
    _gap = [r - v for v, r, _ in [(x[0], x[1], x[2]) for x in _eb]]
    cmd('ebArmBelowRawMin', min(_gap), '{:.2f}'); cmd('ebArmBelowRawMax', max(_gap), '{:.2f}')
for tag, r in R['runs'].items():
    t = TAGS[tag]
    if r.get('model'): cmd(f'r{t}Model', str(r['model']).replace('_', '\\_'))
    cmd(f'r{t}Rxn', r['n_reactions'], 'int'); cmd(f'r{t}Expr', r['n_expression_bearing'], 'int')
    cmd(f'r{t}Pat', r['n_patients'], 'int'); cmd(f'r{t}Calls', r['n_calls'], 'int')
    cmd(f'r{t}Resent', r.get('resent_calls', 0), 'int')
    cmd(f'r{t}ResentPct', 100.0 * r.get('resent_calls', 0) / r['n_calls'], '{:.1f}')
    cmd(f'r{t}Base', r['label_base_rate'], '{:.3f}')
    if r.get('minutes') is not None: cmd(f'r{t}Minutes', r['minutes'], '{:.0f}')
    cmd(f'r{t}RawExpr', r['raw_expression_auroc']); cmd(f'r{t}Ind', r['indicator_floor'])
    if r.get('raw_expression_auprc') is not None:
        cmd(f'r{t}RawExprAuprc', r['raw_expression_auprc']); cmd(f'r{t}IndAuprc', r['indicator_auprc']); cmd(f'r{t}ConstAuprc', r['constant_auprc'])
    if r.get('raw_expression_auroc_expr') is not None: cmd(f'r{t}RawExprEb', r['raw_expression_auroc_expr'])
    if r.get('raw_percentile_auroc') is not None:
        cmd(f'r{t}RawPct', r['raw_percentile_auroc']); cmd(f'r{t}RawPctEb', r['raw_percentile_auroc_expr'])
        # where the percentile-only ranking sits against the availability baseline, as a word and a size
        _dpi = r['raw_percentile_auroc'] - r['indicator_floor']
        cmd(f'r{t}RawPctVsInd', 'above' if _dpi > 0 else 'below'); cmd(f'r{t}RawPctMinusIndAbs', abs(_dpi), '{:.4f}')
    for a in ('reaction', 'patient', 'shuffled'):
        if a not in r['per_arm']: continue
        v = r['per_arm'][a]; A_ = a.capitalize()
        cmd(f'r{t}{A_}', v['auroc']); cmd(f'r{t}{A_}SD', v['auroc_sd'])
        cmd(f'r{t}{A_}Lo', v['auroc_ci'][0]); cmd(f'r{t}{A_}Hi', v['auroc_ci'][1])
        if v.get('auroc_expr') is not None: cmd(f'r{t}{A_}AucEb', v['auroc_expr'])   # per-patient average on expression-bearing reactions
        if v.get('auprc') is not None: cmd(f'r{t}{A_}Auprc', v['auprc'])
        # position of each arm against the two references, as a signed difference and as a word
        cmd(f'r{t}{A_}MinusInd', v['auroc'] - r['indicator_floor'], '{:+.4f}')
        cmd(f'r{t}{A_}VsInd', 'above' if v['auroc'] > r['indicator_floor'] else 'below')
        cmd(f'r{t}{A_}MinusRaw', v['auroc'] - r['raw_expression_auroc'], '{:+.4f}')
        cmd(f'r{t}{A_}R', v['median_inter_patient_r']); cmd(f'r{t}{A_}P', v['mean_p'], '{:.3f}')
        if 'median_inter_patient_r_all' in v: cmd(f'r{t}{A_}RAll', v['median_inter_patient_r_all'])
        cmd(f'r{t}{A_}Nval', v['n_distinct_values'], 'int')
        if 'parsed' in v: cmd(f'r{t}{A_}Parsed', v['parsed'], 'pct')
    if 'determinism' in r:
        d = r['determinism']
        cmd(f'r{t}DetId', d['identical'], 'pct'); cmd(f'r{t}DetDiff', d['mean_abs_diff'], '{:.4f}')
        cmd(f'r{t}DetPairs', d['n_prompt_pairs'], 'int')
        if 'identical_expr' in d:
            cmd(f'r{t}DetIdExpr', d['identical_expr'], 'pct'); cmd(f'r{t}DetDiffExpr', d['mean_abs_diff_expr'], '{:.4f}')
    for a, v in (r.get('resent') or {}).items():
        A_ = a.capitalize()
        cmd(f'r{t}Resent{A_}N', v['n'], 'int'); cmd(f'r{t}Resent{A_}Share', 100.0 * v['share_of_arm'], '{:.1f}')
        cmd(f'r{t}Resent{A_}Pats', v['n_patients_affected'], 'int')
        cmd(f'r{t}Resent{A_}UnaffPats', r['n_patients'] - v['n_patients_affected'], 'int')
        cmd(f'r{t}Resent{A_}CellsList', ' and '.join(str(c) for c in v['cells_per_affected_patient'].values()))
        cmd(f'r{t}Resent{A_}All', v['auroc_all'])
        if v.get('auroc_unaffected') is not None: cmd(f'r{t}Resent{A_}Unaff', v['auroc_unaffected'])
        if v.get('auroc_affected') is not None: cmd(f'r{t}Resent{A_}Aff', v['auroc_affected'])
    for a, v in (r.get('surviving_per_patient') or {}).items():
        A_ = ''.join(part.capitalize() for part in a.split('_'))     # macro names cannot carry underscores
        for _d, _w in (('2', 'Two'), ('3', 'Three'), ('4', 'Four')): A_ = A_.replace(_d, _w)   # nor digits
        cmd(f'r{t}Surv{A_}Min', v['min'], 'int'); cmd(f'r{t}Surv{A_}Max', v['max'], 'int')
    if r.get('surviving_per_patient'):
        cmd(f'r{t}SurvMin', min(v['min'] for v in r['surviving_per_patient'].values()), 'int')
        cmd(f'r{t}SurvMax', max(v['max'] for v in r['surviving_per_patient'].values()), 'int')
    if 'determinism' in r and 'n_cell_pairs' in r['determinism']:
        cmd(f'r{t}DetCells', r['determinism']['n_cell_pairs'], 'int')
    for a, v in (r.get('arms_vs_blind') or {}).items():
        A_ = {'patient': 'Own', 'shuffled': 'Str'}[a]
        cmd(f'r{t}{A_}OverBlind', v['mean_diff'], '{:+.4f}'); cmd(f'r{t}{A_}OverBlindP', v['paired_p'], 'p')
        cmd(f'r{t}{A_}OverBlindLo', v['ci90'][0], '{:+.4f}'); cmd(f'r{t}{A_}OverBlindHi', v['ci90'][1], '{:+.4f}')
    _vb = r.get('arms_vs_blind') or {}
    if 'patient' in _vb and 'shuffled' in _vb:
        def _ex(v): return v['ci90'][0] > 0 or v['ci90'][1] < 0
        _eo, _es = _ex(_vb['patient']), _ex(_vb['shuffled'])
        if _eo and _es: _ph = "both 90\\% intervals excluding zero"
        elif not _eo and not _es: _ph = "neither 90\\% interval excluding zero"
        elif _es: _ph = f"the own difference's 90\\% interval including zero and the stranger's ({_vb['shuffled']['ci90'][0]:+.4f} to {_vb['shuffled']['ci90'][1]:+.4f}) excluding it"
        else: _ph = f"the stranger's difference's 90\\% interval including zero and the own ({_vb['patient']['ci90'][0]:+.4f} to {_vb['patient']['ci90'][1]:+.4f}) excluding it"
        cmd(f'r{t}VsBlindZeroPhrase', _ph)
    if r.get('between_patient_sd_diff'):
        v = r['between_patient_sd_diff']
        cmd(f'r{t}SdDiff', v['diff'], '{:+.3f}'); cmd(f'r{t}SdDiffLo', v['ci95'][0], '{:+.3f}'); cmd(f'r{t}SdDiffHi', v['ci95'][1], '{:+.3f}')
    for a, v in (r.get('raw_expression_binned') or {}).items():
        A_ = {'patient': 'Own', 'shuffled': 'Str'}[a]
        cmd(f'r{t}Binned{A_}', v['auroc']); cmd(f'r{t}Binned{A_}Levels', v['levels'], 'int')
    if r.get('reaction_bootstrap'):
        v = r['reaction_bootstrap']
        cmd(f'r{t}RbB', v['B'], 'int'); cmd(f'r{t}RbOwnGeInd', v['p_own_ge_indicator'], '{:.3f}')
        cmd(f'r{t}RbOwnGeRaw', v['p_own_ge_raw'], '{:.3f}')
        cmd(f'r{t}RbDiffLo', v['own_minus_stranger_ci95'][0], '{:+.4f}'); cmd(f'r{t}RbDiffHi', v['own_minus_stranger_ci95'][1], '{:+.4f}')
        if v.get('own_minus_blind_ci95'):
            cmd(f'r{t}RbOwnBlindLo', v['own_minus_blind_ci95'][0], '{:+.4f}'); cmd(f'r{t}RbOwnBlindHi', v['own_minus_blind_ci95'][1], '{:+.4f}')
            cmd(f'r{t}RbStrBlindLo', v['stranger_minus_blind_ci95'][0], '{:+.4f}'); cmd(f'r{t}RbStrBlindHi', v['stranger_minus_blind_ci95'][1], '{:+.4f}')
            cmd(f'r{t}RbOwnBlindLeZero', v['p_own_minus_blind_le0'], '{:.3f}')
        for a, ci in v['arm_ci95'].items():
            A_ = a.capitalize(); cmd(f'r{t}Rb{A_}Lo', ci[0]); cmd(f'r{t}Rb{A_}Hi', ci[1])
    if r.get('crossed_bootstrap'):
        v = r['crossed_bootstrap']
        cmd(f'r{t}CbB', v['B'], 'int')
        cmd(f'r{t}CbDiffLo', v['own_minus_stranger_ci90'][0], '{:+.4f}'); cmd(f'r{t}CbDiffHi', v['own_minus_stranger_ci90'][1], '{:+.4f}')
        cmd(f'r{t}CbDiffLoNF', v['own_minus_stranger_ci95'][0], '{:+.4f}'); cmd(f'r{t}CbDiffHiNF', v['own_minus_stranger_ci95'][1], '{:+.4f}')
        cmd(f'r{t}CbWithinTwo', 'lies inside' if v['own_minus_stranger_within_002'] else 'does not lie inside')
        cmd(f'r{t}CbWithinFive', 'lies inside' if v['own_minus_stranger_within_005'] else 'does not lie inside')
        _lo95, _hi95 = v['own_minus_stranger_ci95']
        cmd(f'r{t}CbWithinTwoNF', 'lies inside' if (_lo95 > -0.02 and _hi95 < 0.02) else 'does not lie inside')
        # how far each upper endpoint sits from the narrower margin, stated as a distance with the
        # side named, so a sentence cannot get the direction or the confidence level wrong
        _h90, _h95 = v['own_minus_stranger_ci90'][1], _hi95
        cmd(f'r{t}CbHiSlackNinety', f"{abs(0.02 - _h90):.4f} {'inside the margin' if _h90 < 0.02 else 'outside it'}")
        cmd(f'r{t}CbHiSlackNinetyFive', f"{abs(_h95 - 0.02):.4f} {'inside the margin' if _h95 < 0.02 else 'outside it'}")
        if v.get('own_minus_blind_ci90'):
            cmd(f'r{t}CbOwnBlindLo', v['own_minus_blind_ci90'][0], '{:+.4f}'); cmd(f'r{t}CbOwnBlindHi', v['own_minus_blind_ci90'][1], '{:+.4f}')
            cmd(f'r{t}CbStrBlindLo', v['stranger_minus_blind_ci90'][0], '{:+.4f}'); cmd(f'r{t}CbStrBlindHi', v['stranger_minus_blind_ci90'][1], '{:+.4f}')
    if 'determinism_patient_prompt' in r:
        d = r['determinism_patient_prompt']
        cmd(f'r{t}RepId', d['identical'], 'pct'); cmd(f'r{t}RepDiff', d['mean_abs_diff'], '{:.4f}')
        cmd(f'r{t}RepPairs', d['n_pairs'], 'int'); cmd(f'r{t}RepN', d['n_repeats'], 'int')
        if 'identical_all' in d:
            cmd(f'r{t}RepIdAll', d['identical_all'], 'pct'); cmd(f'r{t}RepDiffAll', d['mean_abs_diff_all'], '{:.4f}')
            cmd(f'r{t}RepPairsAll', d['n_pairs_all'], 'int')
        cmd(f'r{t}RepNval', d['n_distinct_values'], 'int')
    if 'patient_vs_shuffled' in r:
        p = r['patient_vs_shuffled']
        cmd(f'r{t}TostMargin', p['tost_margin'], '{:.2f}')
        cmd(f'r{t}TostP', max(p['tost_p_lower'], p['tost_p_upper']), '{:.4f}')
        cmd(f'r{t}TostPass', 'passes' if p['tost_equivalent'] else 'does not pass')
        if 'tost_margin_narrow' in p:
            cmd(f'r{t}TostNMargin', p['tost_margin_narrow'], '{:.2f}')
            cmd(f'r{t}TostNP', max(p['tost_narrow_p_lower'], p['tost_narrow_p_upper']), '{:.4f}')
            cmd(f'r{t}TostNPass', 'passes' if p['tost_narrow_equivalent'] else 'does not pass')
        if 'auroc_diff_ci90' in p:
            cmd(f'r{t}AucDiffLo', p['auroc_diff_ci90'][0], '{:+.4f}'); cmd(f'r{t}AucDiffHi', p['auroc_diff_ci90'][1], '{:+.4f}')
        cmd(f'r{t}Pairs', p['n_pairs'], 'int'); cmd(f'r{t}Ident', p['identical'], 'pct')
        cmd(f'r{t}Changed', 1 - p['identical'], 'pct')
        cmd(f'r{t}Diff', p['mean_abs_diff'], '{:.3f}'); cmd(f'r{t}ArmR', p['arm_correlation'], '{:.3f}')
        cmd(f'r{t}ResR', p['residual_correlation'], '{:+.3f}')
        cmd(f'r{t}ResLo', p['residual_correlation_ci'][0], '{:+.3f}')
        cmd(f'r{t}ResHi', p['residual_correlation_ci'][1], '{:+.3f}')
        cmd(f'r{t}SdReal', p['between_patient_sd_real'], '{:.3f}')
        cmd(f'r{t}SdStr', p['between_patient_sd_stranger'], '{:.3f}')
        # paired own-versus-swapped AUROC difference over patients, promised in Methods beside TOST
        cmd(f'r{t}AucDiff', p['auroc_diff_mean'], '{:+.4f}')
        cmd(f'r{t}AucDiffSD', p['auroc_diff_sd'], '{:.4f}')
        cmd(f'r{t}AucDiffP', p['paired_p'], '{:.4f}')
        cmd(f'r{t}AucDiffW', p.get('wilcoxon_p'), '{:.4f}')
        cmd(f'r{t}MoveP', p['mean_move_from_reaction_arm'].get('patient'), '{:.3f}')
        cmd(f'r{t}MoveS', p['mean_move_from_reaction_arm'].get('shuffled'), '{:.3f}')
        if 'determinism' in r and r['determinism']['mean_abs_diff'] > 0:
            cmd(f'r{t}MoveRatio',
                p['mean_move_from_reaction_arm']['patient'] / r['determinism']['mean_abs_diff'], '{:.0f}')
    if 'variance_decomposition' in r:
        d = r['variance_decomposition']
        cmd(f'r{t}VdN', d['n'], 'int')
        cmd(f'r{t}VdIdent', d['r2_reaction_identity'], '{:.3f}')
        cmd(f'r{t}VdPres', d['r2_plus_presence'], '{:.3f}')
        cmd(f'r{t}VdMag', d['r2_plus_magnitude'], '{:.3f}')
        cmd(f'r{t}VdDPres', d['r2_plus_presence'] - d['r2_reaction_identity'], '{:.3f}')
        cmd(f'r{t}VdDMag', d['r2_plus_magnitude'] - d['r2_plus_presence'], '{:.3f}')
        if 'r2_plus_magnitude_first' in d:
            cmd(f'r{t}VdMagFirst', d['r2_plus_magnitude_first'], '{:.3f}')
            cmd(f'r{t}VdDMagFirst', d['r2_plus_magnitude_first'] - d['r2_reaction_identity'], '{:.3f}')
            cmd(f'r{t}VdDPresLast', d['r2_plus_magnitude'] - d['r2_plus_magnitude_first'], '{:.3f}')
            cmd(f'r{t}VdCorr', d['r2_presence_magnitude_corr'], '{:.2f}')
            cmd(f'r{t}VdPShown', d['mean_p_shown'], '{:.3f}'); cmd(f'r{t}VdPNot', d['mean_p_not_shown'], '{:.3f}')
            cmd(f'r{t}VdBlindShown', d['mean_p_blind_shown'], '{:.3f}'); cmd(f'r{t}VdBlindNot', d['mean_p_blind_not_shown'], '{:.3f}')
        if 'expr_only_n' in d:
            cmd(f'r{t}VdExprN', d['expr_only_n'], 'int')
            cmd(f'r{t}VdExprIdent', d['expr_only_r2_reaction_identity'], '{:.3f}')
            cmd(f'r{t}VdExprMag', d['expr_only_r2_plus_magnitude'], '{:.3f}')
            if 'expr_only_r2_plus_percentile' in d:
                cmd(f'r{t}VdExprPct', d['expr_only_r2_plus_percentile'], '{:.3f}')
                cmd(f'r{t}VdExprBoth', d['expr_only_r2_plus_both'], '{:.3f}')
                cmd(f'r{t}SpearPct', d['expr_only_spearman_percentile'], '{:+.3f}')
    if 'score_vs_shown_value' in r:
        cmd(f'r{t}Pear', r['score_vs_shown_value']['pearson'], '{:+.3f}')
        cmd(f'r{t}Spear', r['score_vs_shown_value']['spearman'], '{:+.3f}')
    for a in ('reaction', 'patient', 'shuffled'):
        if a not in r.get('strata', {}): continue
        s = r['strata'][a]; A_ = a.capitalize()
        for nm, k in (('Ex', 'expression_bearing'), ('No', 'no_expression'), ('Tr', 'transport_exchange'), ('Met', 'metabolic')):
            if k in s:
                cmd(f'r{t}{A_}{nm}P', s[k]['mean_p'], '{:.3f}')
                cmd(f'r{t}{A_}{nm}Auc', s[k]['auroc'])
                cmd(f'r{t}{A_}{nm}N', s[k]['n'], 'int')
        cmd(f'r{t}{A_}PosP', s['mean_p_label_active'], '{:.3f}')
        cmd(f'r{t}{A_}NegP', s['mean_p_label_inactive'], '{:.3f}')

# how many tumors the probe was actually run on, per design: the reaction-activity designs and the
# prompt-format variants, which are the runs the abstract and introduction describe. The reasoning
# sensitivity run is smaller again and is quoted separately as thkPat.
_npat = [R['runs'][k]['n_patients'] for k in ('A_representative', 'B_balanced', 'C_backbone2', 'E_representative40', 'A_backbone2')
         if k in R['runs']] + [f['n_patients'] for f in R.get('formats', {}).values()]
if _npat:
    cmd('rPatMin', min(_npat), 'int'); cmd('rPatMax', max(_npat), 'int')
# the movement-to-noise ratio across the designs that have both quantities
_ratios = []
for k in ('A_representative', 'B_balanced', 'C_backbone2', 'E_representative40', 'A_backbone2'):
    r_ = R['runs'].get(k)
    if r_ and 'determinism' in r_ and 'patient_vs_shuffled' in r_ and r_['determinism']['mean_abs_diff'] > 0:
        _ratios.append(r_['patient_vs_shuffled']['mean_move_from_reaction_arm']['patient'] / r_['determinism']['mean_abs_diff'])
if _ratios:
    cmd('rMoveRatioMin', min(_ratios), '{:.0f}'); cmd('rMoveRatioMax', max(_ratios), '{:.0f}')
# the cross-session floor: Design A's patient and patient-blind prompts answered again in Design E
XS = R.get('cross_session')
if XS:
    cmd('xsPat', XS['n_patients'], 'int'); cmd('xsRxn', XS['n_reactions'], 'int')
    for a, v in XS['arms'].items():
        A_ = {'patient': 'Own', 'reaction': 'Blind'}[a]
        cmd(f'xs{A_}Cells', v['n_cells'], 'int'); cmd(f'xs{A_}CellsExpr', v['n_cells_expr'], 'int')
        cmd(f'xs{A_}Id', v['identical'], 'pct'); cmd(f'xs{A_}Diff', v['mean_abs_diff'], '{:.4f}')
        cmd(f'xs{A_}IdExpr', v['identical_expr'], 'pct'); cmd(f'xs{A_}DiffExpr', v['mean_abs_diff_expr'], '{:.4f}')
    if R['cross_session'].get('time_gap'):
        _g = R['cross_session']['time_gap']
        cmd('xsGapMedianHours', _g['median_hours'], '{:.1f}'); cmd('xsGapMinHours', _g['min_hours'], '{:.1f}'); cmd('xsGapMaxHours', _g['max_hours'], '{:.1f}')
    xfloor = XS['arms']['patient']['mean_abs_diff_expr']
    _xr = []
    for k in ('A_representative', 'B_balanced', 'E_representative40'):
        r_ = R['runs'].get(k)
        if r_ and 'patient_vs_shuffled' in r_ and xfloor > 0:
            ratio = r_['patient_vs_shuffled']['mean_move_from_reaction_arm']['patient'] / xfloor
            cmd(f'r{TAGS[k]}MoveRatioXs', ratio, '{:.0f}'); _xr.append(ratio)
    if _xr: cmd('rMoveRatioXsMin', min(_xr), '{:.0f}'); cmd('rMoveRatioXsMax', max(_xr), '{:.0f}')
    # every design's movement against its own patient-blind floor on the expression-bearing reactions
    for k, r_ in R['runs'].items():
        if k in TAGS and r_.get('determinism', {}).get('mean_abs_diff_expr') and 'patient_vs_shuffled' in r_:
            _fl = r_['determinism']['mean_abs_diff_expr']
            if _fl > 0: cmd(f'r{TAGS[k]}MoveRatioFree', r_['patient_vs_shuffled']['mean_move_from_reaction_arm']['patient'] / _fl, '{:.0f}')
    # Design B has no cross-session measurement: its movement against its own free floor on its own reactions
    rB = R['runs'].get('B_balanced')
    if rB and rB.get('determinism', {}).get('mean_abs_diff_expr'):
        cmd('rBMoveRatioOwnFree', rB['patient_vs_shuffled']['mean_move_from_reaction_arm']['patient'] / rB['determinism']['mean_abs_diff_expr'], '{:.0f}')
    # the free floor against the cross-session floor on the same reactions
    dA = R['runs'].get('A_representative', {}).get('determinism', {})
    if dA.get('mean_abs_diff_expr'):
        cmd('xsOverFreeExpr', xfloor / dA['mean_abs_diff_expr'], '{:.1f}')
    # the two prompts' cross-session drift on the same expression-bearing reactions, as a phrase
    _xb = (XS['arms'].get('reaction') or {}).get('mean_abs_diff_expr')
    if _xb:
        if abs(xfloor - _xb) < 0.0003:
            cmd('xsOwnVsBlindPhrase', 'the two prompts drift by the same amount')
        elif xfloor > _xb:
            cmd('xsOwnVsBlindPhrase', f'the data-bearing prompt drifts {xfloor / _xb:.1f} times as much as the patient-blind one')
        else:
            cmd('xsOwnVsBlindPhrase', f'the patient-blind prompt drifts {_xb / xfloor:.1f} times as much as the data-bearing one')
    fr = R['runs'].get('F_replicate', {}).get('determinism_patient_prompt', {})
    if fr.get('mean_abs_diff'):
        cmd('xsOverRep', xfloor / fr['mean_abs_diff'], '{:.1f}')
if 'A_representative' in R['runs'] and 'A_backbone2' in R['runs']:
    # how far the second backbone's patient-blind arm sits below the first's, on the same reactions
    _d = R['runs']['A_representative']['per_arm']['reaction']['auroc'] - R['runs']['A_backbone2']['per_arm']['reaction']['auroc']
    cmd('rSBlindBelowA', abs(_d), '{:.2f}'); cmd('rSBlindMinusA', -_d, '{:+.4f}')
    _d2 = R['runs']['A_representative']['per_arm']['patient']['auroc'] - R['runs']['A_backbone2']['per_arm']['patient']['auroc']
    cmd('rSOwnBelowA', abs(_d2), '{:.2f}')
if 'A_representative' in R['runs'] and 'E_representative40' in R['runs']:
    cmd('rEExtraPat', R['runs']['E_representative40']['n_patients'] - R['runs']['A_representative']['n_patients'], 'int')
    # the 90% t-interval half-width Design E's sample supports at the spread of the paired difference
    # observed in Design A: the sensitivity of the sample, stated as an interval width
    from scipy import stats as _st
    _nE = R['runs']['E_representative40']['n_patients']; _sdA = (R['runs']['A_representative'].get('patient_vs_shuffled') or {}).get('auroc_diff_sd')
    if _sdA is not None:
        cmd('rEHalfWidthFromA', _st.t.ppf(0.95, _nE - 1) * _sdA / (_nE ** 0.5), '{:.4f}')
# the narrower equivalence margin as a share of the distance from chance to the raw-expression ranking
_rA = R['runs'].get('A_representative')
if _rA and 'patient_vs_shuffled' in _rA and _rA['patient_vs_shuffled'].get('tost_margin_narrow'):
    cmd('rETostNMarginSharePct', 100.0 * _rA['patient_vs_shuffled']['tost_margin_narrow'] / (_rA['raw_expression_auroc'] - 0.5), '{:.0f}')

# whether the precision-recall areas keep every arm below both references in the three main designs
_apb = []
for _k in ('A_representative', 'B_balanced', 'E_representative40'):
    _r = R['runs'].get(_k)
    if _r and _r.get('raw_expression_auprc') is not None:
        _ok = all(v.get('auprc') is not None and v['auprc'] < min(_r['raw_expression_auprc'], _r['indicator_auprc']) for v in _r['per_arm'].values())
        _apb.append((TAGS[_k], _ok))
if _apb:
    _bad = [t for t, ok in _apb if not ok]
    cmd('auprcOrderPhrase', 'keeps every arm below both references in all three designs' if not _bad else
        'keeps every arm below both references except in Design~' + ' and Design~'.join(_bad))

# each interleaved re-collection against its blocked collection, arm by arm (interleaved minus blocked)
for _tag, x in (R.get('interleaved_vs_blocked') or {}).items():
    t = TAGS[_tag]
    for a, A_ in (('reaction', 'Blind'), ('patient', 'Own'), ('shuffled', 'Str')):
        if a in x['auroc_interleaved'] and a in x['auroc_blocked']:
            _d = x['auroc_interleaved'][a] - x['auroc_blocked'][a]
            cmd(f'il{t}{A_}MinusBlocked', _d, '{:+.4f}'); cmd(f'il{t}{A_}MinusBlockedAbs', abs(_d))
            cmd(f'il{t}{A_}Blocked', x['auroc_blocked'][a])
            # the blocked collection's arm against the availability baseline (shared by both collections)
            _ind = R['runs'][_tag]['indicator_floor']
            cmd(f'il{t}{A_}BlockedVsInd', 'above' if x['auroc_blocked'][a] > _ind else 'below')
        y = x['arms'].get(a)
        if y:
            cmd(f'il{t}{A_}Id', y['identical'], 'pct'); cmd(f'il{t}{A_}Diff', y['mean_abs_diff'])
            cmd(f'il{t}{A_}IdExpr', y['identical_expr'], 'pct'); cmd(f'il{t}{A_}DiffExpr', y['mean_abs_diff_expr'])
            cmd(f'il{t}{A_}Cells', y['n_cells'], 'int'); cmd(f'il{t}{A_}CellsExpr', y['n_cells_expr'], 'int')
    if x.get('own_minus_stranger_blocked') is not None:
        cmd(f'il{t}OwnStrBlocked', x['own_minus_stranger_blocked'], '{:+.4f}')
        cmd(f'il{t}OwnStrInterleaved', x['own_minus_stranger_interleaved'], '{:+.4f}')
    if x.get('time_gap'):
        cmd(f'il{t}GapMedianHours', x['time_gap']['median_hours'], '{:.1f}'); cmd(f'il{t}GapMinHours', x['time_gap']['min_hours'], '{:.1f}')
        cmd(f'il{t}GapMaxHours', x['time_gap']['max_hours'], '{:.1f}')
    _ids = [x['arms'][a]['identical'] for a in x['arms']]
    if _ids: cmd(f'il{t}IdMin', min(_ids), 'pct'); cmd(f'il{t}IdMax', max(_ids), 'pct')
    _ds = [abs(x['auroc_interleaved'][a] - x['auroc_blocked'][a]) for a in x['auroc_interleaved'] if a in x['auroc_blocked']]
    if _ds: cmd(f'il{t}AurocMaxAbs', max(_ds))
if R.get('interleaved_vs_blocked'):
    cmd('ilDesigns', len(R['interleaved_vs_blocked']), 'int')
    _dn = {'A_representative': 'Design~A on the main backbone', 'A_backbone2': 'Design~A on the second backbone',
           'A_backbone3': 'Design~A on the third backbone', 'E_representative40': 'Design~E'}
    _an = {'reaction': 'patient-blind', 'patient': 'own-data', 'shuffled': "stranger's-data"}
    # arms that change side of the availability baseline between the two collections
    _flips = []
    for _tag, x in R['interleaved_vs_blocked'].items():
        _ind = R['runs'][_tag]['indicator_floor']
        for a in x['auroc_interleaved']:
            if a in x['auroc_blocked'] and ((x['auroc_blocked'][a] > _ind) != (x['auroc_interleaved'][a] > _ind)):
                _flips.append((_tag, a, x['auroc_blocked'][a], x['auroc_interleaved'][a], _ind))
    if not _flips:
        cmd('ilIndFlipPhrase', 'no arm changes side of the availability baseline between the two collections')
    else:
        cmd('ilIndFlipPhrase', ('one arm changes' if len(_flips) == 1 else f'{len(_flips)} arms change') + ' side of the availability baseline between the two collections: ' +
            '; '.join(f"the {_an[a]} arm of {_dn.get(t, t)} ({b:.4f} blocked, {i:.4f} interleaved, against {f:.4f})" for t, a, b, i, f in _flips))
    # own-minus-stranger differences that change sign
    _sflips = [_dn.get(t, t) for t, x in R['interleaved_vs_blocked'].items()
               if x.get('own_minus_stranger_blocked') is not None and (x['own_minus_stranger_blocked'] > 0) != (x['own_minus_stranger_interleaved'] > 0)]
    cmd('ilOwnStrFlipPhrase', 'no paired own-minus-stranger difference changes sign' if not _sflips else
        'the paired own-minus-stranger difference changes sign in ' + ' and '.join(_sflips))
    _all_ids = [y['identical'] for x in R['interleaved_vs_blocked'].values() for y in x['arms'].values()]
    _all_ds = [abs(x['auroc_interleaved'][a] - x['auroc_blocked'][a]) for x in R['interleaved_vs_blocked'].values() for a in x['auroc_interleaved'] if a in x['auroc_blocked']]
    cmd('ilIdMinAll', min(_all_ids), 'pct'); cmd('ilAurocMaxAbsAll', max(_all_ds))
for _key, _dp in (('donor_schedules', 'ds'), ('donor_schedules_E', 'dsE')):
    if not R.get(_key): continue
    x = R[_key]
    cmd(_dp + 'OwnStrOne', x['own_minus_stranger'][0], '{:+.4f}'); cmd(_dp + 'OwnStrTwo', x['own_minus_stranger'][1], '{:+.4f}')
    cmd(_dp + 'OwnStrTwoLo', x['own_minus_stranger_ci90'][1][0], '{:+.4f}'); cmd(_dp + 'OwnStrTwoHi', x['own_minus_stranger_ci90'][1][1], '{:+.4f}')
    cmd(_dp + 'OwnStrOneLo', x['own_minus_stranger_ci90'][0][0], '{:+.4f}'); cmd(_dp + 'OwnStrOneHi', x['own_minus_stranger_ci90'][0][1], '{:+.4f}')
    cmd(_dp + 'StrOne', x['stranger_auroc'][0]); cmd(_dp + 'StrTwo', x['stranger_auroc'][1])
    cmd(_dp + 'SdStrOne', x['between_patient_sd_stranger'][0]); cmd(_dp + 'SdStrTwo', x['between_patient_sd_stranger'][1])
    if x.get('own_auroc'):
        cmd(_dp + 'OwnOne', x['own_auroc'][0]); cmd(_dp + 'OwnTwo', x['own_auroc'][1])
        cmd(_dp + 'BlindOne', x['blind_auroc'][0]); cmd(_dp + 'BlindTwo', x['blind_auroc'][1])
    if x.get('equivalent_002') and x['equivalent_002'][1] is not None:
        cmd(_dp + 'TwoEquivTwo', 'passes' if x['equivalent_002'][1] else 'does not pass')
        cmd(_dp + 'TwoEquivFive', 'passes' if x['equivalent_005'][1] else 'does not pass')
        # the sign of the difference under the two schedules, as a phrase
        _s1, _s2 = x['own_minus_stranger'][0] > 0, x['own_minus_stranger'][1] > 0
        cmd(_dp + 'SignPhrase', 'the same sign under both schedules' if _s1 == _s2 else 'opposite signs under the two schedules')
if 'A_representative' in R['runs'] and 'A_backbone3' in R['runs']:
    _d = R['runs']['A_representative']['per_arm']['reaction']['auroc'] - R['runs']['A_backbone3']['per_arm']['reaction']['auroc']
    cmd('rTBlindMinusA', -_d, '{:+.4f}'); cmd('rTBlindBelowA', abs(_d), '{:.2f}')
    _d2 = R['runs']['A_representative']['per_arm']['patient']['auroc'] - R['runs']['A_backbone3']['per_arm']['patient']['auroc']
    cmd('rTOwnMinusA', -_d2, '{:+.4f}'); cmd('rTOwnBelowA', abs(_d2), '{:.2f}')

for tag, f in R.get('formats', {}).items():
    T = tag.capitalize()[:3]          # Cat, Per
    cmd(f'fmt{T}Rxn', f['n_reactions'], 'int'); cmd(f'fmt{T}Pat', f['n_patients'], 'int')
    cmd(f'fmt{T}Calls', f['n_calls'], 'int')
    for a, A_ in (('patient', 'Own'), ('shuffled', 'Str')):
        if a not in f['per_arm']: continue
        v = f['per_arm'][a]
        cmd(f'fmt{T}{A_}', v['auroc'])
        cmd(f'fmt{T}{A_}Lo', v['auroc_ci'][0]); cmd(f'fmt{T}{A_}Hi', v['auroc_ci'][1])
        cmd(f'fmt{T}{A_}R', v['median_inter_patient_r']); cmd(f'fmt{T}{A_}P', v['mean_p'], '{:.3f}')
        cmd(f'fmt{T}{A_}Nval', v['n_distinct_values'], 'int')
        if 'mean_p_expression_bearing' in v:
            cmd(f'fmt{T}{A_}ExP', v['mean_p_expression_bearing'], '{:.3f}')
            cmd(f'fmt{T}{A_}NoP', v['mean_p_no_expression'], '{:.3f}')
    if 'patient_vs_shuffled' in f:
        p = f['patient_vs_shuffled']
        cmd(f'fmt{T}Ident', p['identical'], 'pct'); cmd(f'fmt{T}Diff', p['mean_abs_diff'], '{:.3f}')
        cmd(f'fmt{T}SdReal', p['between_patient_sd_real'], '{:.3f}')
        cmd(f'fmt{T}SdStr', p['between_patient_sd_stranger'], '{:.3f}')

D_ = R['runs'].get('D_reasoning')
if D_:
    cmd('thkRxn', D_['n_reactions'], 'int'); cmd('thkPat', D_['n_patients'], 'int')
    cmd('thkCalls', D_['n_calls'], 'int'); cmd('thkResent', D_.get('resent_calls', 0), 'int')
    cmd('thkMin', D_.get('minutes', 0), '{:.0f}')
    for a in ('reaction', 'patient', 'shuffled'):
        if a not in D_['per_arm']: continue
        v = D_['per_arm'][a]; A_ = a.capitalize()
        cmd(f'thk{A_}', v['auroc']); cmd(f'thk{A_}Parsed', v['parsed'], 'pct')
        cmd(f'thk{A_}Lo', v['auroc_ci'][0]); cmd(f'thk{A_}Hi', v['auroc_ci'][1])
        cmd(f'thk{A_}R', v['median_inter_patient_r']); cmd(f'thk{A_}P', v['mean_p'], '{:.3f}')
        cmd(f'thk{A_}Nval', v['n_distinct_values'], 'int')
    _p = [D_['per_arm'][a]['parsed'] for a in D_['per_arm']]
    cmd('thkParsedAll', sum(_p) / len(_p), 'pct')
    cmd('thkEmptyAll', 1 - sum(_p) / len(_p), 'pct')

# the positive-control sampling, read from the archived trials: distinct reactions and patients
_pcr = rawload('pc')
if _pcr:
    _rx = {k.split('|')[2] for k in _pcr if k.startswith('read|')}; _pt = {k.split('|')[1] for k in _pcr if k.startswith('read|')}
    cmd('pcRxn', len(_rx), 'int'); cmd('pcPat', len(_pt), 'int')
for bk, e in R.get('positive_control', {}).items():
    b = {'backbone1': 'One', 'backbone2': 'Two', 'backbone3': 'Three'}[bk]
    if 'echo_modal_share' in e: cmd(f'pc{b}EchoModal', e['echo_modal_share'], 'pct')
    if 'echoval_modal_share' in e:
        cmd(f'pc{b}EchovalModal', e['echoval_modal_share'], 'pct'); cmd(f'pc{b}EchovalWithinDigit', e['echoval_within_one_digit'], 'pct')
        # a parenthetical for near-misses, empty when every reply is exact
        _ex = e['echoval']['accuracy']; _wd = e['echoval_within_one_digit']
        cmd(f'pc{b}EchovalWithinClause', '' if abs(_wd - _ex) < 1e-9 else f" ({100 * _wd:.1f}\\% within one unit of the last printed digit)")
    for task, v in e.items():
        if not isinstance(v, dict): continue
        cmd(f'pc{b}{task.capitalize()}', v['accuracy'], 'pct')
        if 'modal_answer_share' in v: cmd(f'pc{b}{task.capitalize()}Modal', v['modal_answer_share'], 'pct')
        if 'n_errors' in v:
            cmd(f'pc{b}{task.capitalize()}Errors', v['n_errors'], 'int'); cmd(f'pc{b}{task.capitalize()}ErrorsFirst', v['errors_answering_first'], 'int')
            # "all ten of its errors", "8 of its 10 errors", or "none of its 10 errors" name the first record
            _ne, _nf = int(v['n_errors']), int(v['errors_answering_first'])
            _w = {0: 'no', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten'}
            if _ne == 0: _ph = 'it makes no errors'
            elif _nf == _ne: _ph = f'all {_w.get(_ne, _ne)} of its errors name the first record'
            elif _nf == 0: _ph = f'none of its {_w.get(_ne, _ne)} errors names the first record'
            else: _ph = f'{_nf} of its {_ne} errors name the first record'
            cmd(f'pc{b}{task.capitalize()}ErrorsPhrase', _ph)
        cmd(f'pc{b}{task.capitalize()}N', v['n'], 'int')
        if 'n_scored' in v: cmd(f'pc{b}{task.capitalize()}Scored', v['n_scored'], 'int'); cmd(f'pc{b}{task.capitalize()}Tied', v['n_tied'], 'int')
        if v.get('accuracy_hard') is not None:
            cmd(f'pc{b}{task.capitalize()}Hard', v['accuracy_hard'], 'pct')
            cmd(f'pc{b}{task.capitalize()}HardN', v['n_hard'], 'int')
            # where the errors fall: the count on the near-threshold trials against the count on the rest,
            # so that a sentence about legibility cannot localize the errors where they are not
            _nh = int(v['n_hard']); _n_all = int(v['n']); _e_all = round(_n_all * (1.0 - v['accuracy'])); _e_h = round(_nh * (1.0 - v['accuracy_hard']))
            _e_easy = max(_e_all - _e_h, 0); _n_easy = _n_all - _nh
            cmd(f'pc{b}{task.capitalize()}ErrorsEasy', _e_easy, 'int'); cmd(f'pc{b}{task.capitalize()}ErrorsHard', _e_h, 'int')
            if _e_all == 0: _lp = 'it makes no errors'
            elif _e_easy > _e_h: _lp = f'so most of its errors, about {_e_easy} of {_e_all}, fall on trials that are not close calls'
            elif _e_easy == 0: _lp = 'so every one of its errors is a close call'
            else: _lp = f'so most of its errors, about {_e_h} of {_e_all}, are close calls'
            if _n_easy > 0: cmd(f'pc{b}{task.capitalize()}EasyAcc', 100.0 * (1.0 - _e_easy / _n_easy), '{:.1f}')
            cmd(f'pc{b}{task.capitalize()}WherePhrase', _lp)

# which arms of the primary collections pass a reference under either metric, as one sentence
# built from the numbers, so that "no arm passes" and "the one arm that passes" are computed
_ARMN = {'reaction': 'patient-blind', 'patient': 'own-data', 'shuffled': "stranger's-data"}
_RUNN = {'A_representative': 'Design~A', 'B_balanced': 'Design~B', 'E_representative40': 'Design~E',
         'A_backbone2': 'Design~A on the second backbone', 'A_backbone3': 'Design~A on the third backbone'}
_pass = {'auroc_ind': [], 'auroc_raw': [], 'auprc_ind': [], 'auprc_raw': []}
for k, nm in _RUNN.items():
    r = R['runs'].get(k)
    if not r: continue
    for a in ('reaction', 'patient', 'shuffled'):
        v = r['per_arm'].get(a)
        if not v: continue
        if v.get('auroc') is not None:
            if v['auroc'] > r['indicator_floor']: _pass['auroc_ind'].append((nm, a, v['auroc'], r['indicator_floor']))
            if v['auroc'] > r['raw_expression_auroc']: _pass['auroc_raw'].append((nm, a, v['auroc'], r['raw_expression_auroc']))
        if v.get('auprc') is not None and r.get('indicator_auprc') is not None:
            if v['auprc'] > r['indicator_auprc']: _pass['auprc_ind'].append((nm, a, v['auprc'], r['indicator_auprc']))
            if v['auprc'] > r['raw_expression_auprc']: _pass['auprc_raw'].append((nm, a, v['auprc'], r['raw_expression_auprc']))
if any(R['runs'].get(k, {}).get('per_arm', {}).get('reaction', {}).get('auprc') is not None for k in _RUNN):
    def _lst(items):
        by = {}
        for nm, a, v, ref in items: by.setdefault((nm, ref), []).append((a, v))
        return '; '.join(f"the {' and '.join(_ARMN[a] for a, _ in arms)} arm{'s' if len(arms) > 1 else ''} of {nm} "
                         f"({' and '.join(f'{v:.4f}' for _, v in arms)} against {ref:.4f})" for (nm, ref), arms in by.items())
    _sent = []
    if not _pass['auroc_ind'] and not _pass['auroc_raw']: _sent.append('Under AUROC no arm of the primary collections passes either reference')
    else:
        if _pass['auroc_ind']: _sent.append('Under AUROC the availability baseline is passed by ' + _lst(_pass['auroc_ind']))
        if _pass['auroc_raw']: _sent.append('Under AUROC the raw ranking is passed by ' + _lst(_pass['auroc_raw']))
        else: _sent.append('under AUROC no arm passes the raw ranking')
    if not _pass['auprc_ind'] and not _pass['auprc_raw']: _sent.append('and under the precision--recall area none does either')
    else:
        if _pass['auprc_ind']: _sent.append('under the precision--recall area the availability baseline is passed by ' + _lst(_pass['auprc_ind']))
        else: _sent.append('under the precision--recall area no arm passes the availability baseline')
        if _pass['auprc_raw']: _sent.append('and the raw ranking by ' + _lst(_pass['auprc_raw']))
        else: _sent.append('and no arm passes the raw ranking under it')
    cmd('refPassSentence', '; '.join(_sent) + '.')

# the distance between the one-reaction rule and the full supervised panel on the MSI target, as a phrase
_M = R.get('msi') or {}
_bu, _lg = _M.get('best_univariate_auroc'), _M.get('logistic_auroc')
if _bu is not None and _lg is not None:
    _gap = abs(_lg - _bu)
    cmd('msiBestUniGapPhrase', 'within one hundredth of the full panel' if _gap < 0.01 else 'within two hundredths of the full panel' if _gap < 0.02
        else 'within a few hundredths of the full panel' if _gap < 0.05 else f'{_gap:.2f} below the full panel')

# the wider-margin equivalence verdicts of the three main designs as one phrase
_tp = {k: (R['runs'][k].get('patient_vs_shuffled') or {}).get('tost_equivalent') for k in ('A_representative', 'B_balanced', 'E_representative40') if k in R['runs']}
if len(_tp) == 3 and all(v is not None for v in _tp.values()):
    if all(_tp.values()): cmd('tostWidePhrase', 'passes in all three designs')
    elif not any(_tp.values()): cmd('tostWidePhrase', 'passes in none of the three designs')
    else: cmd('tostWidePhrase', ', '.join(f"{'passes' if v else 'does not pass'} in Design~{TAGS[k]}" for k, v in _tp.items()))

# the abstract's one clause on legibility, chosen from the three percentile controls of every backbone:
# a backbone is "at ceiling" when none of its three controls falls below 98% accuracy
_pcb = []
for bk, e in R.get('positive_control', {}).items():
    _acc = [e[t]['accuracy'] for t in ('echo', 'read', 'compare') if isinstance(e.get(t), dict)]
    if len(_acc) == 3: _pcb.append((bk, min(_acc) >= 0.98))
if len(_pcb) >= 2:
    _nc = sum(1 for _, ok in _pcb if ok); _ni = len(_pcb) - _nc
    _w = {1: 'one', 2: 'two', 3: 'three'}
    if _ni == 0: cmd('pcAbstractPhrase', f'at ceiling on all {_w.get(len(_pcb), len(_pcb))} backbones')
    elif _nc == 0: cmd('pcAbstractPhrase', 'imperfectly on every backbone')
    else:
        _ord = {'backbone1': 'first', 'backbone2': 'second', 'backbone3': 'third'}
        _imp = [_ord.get(bk, bk) for bk, ok in _pcb if not ok]
        cmd('pcAbstractPhrase', f"at ceiling on {_w.get(_nc, _nc)} backbone{'s' if _nc > 1 else ''} and imperfectly on the "
            + (_imp[0] if _ni == 1 else ' and '.join(_imp)))

# Majority-class rate of each control's ground truth, read back from the archived trials. Accuracy
# on a two-way question has to be read against this and not only against chance, since a model that
# answers by constant scores the larger class.
for b, outs in (('One', ('pc', 'pc_echo_b1')), ('Two', ('pc_b2', 'pc_echo_b2'))):
    _r = {}
    for o in outs:
        _d = rawload(o)
        if _d: _r.update(_d)
    for task in ('read', 'compare'):
        # the answer key is the printed integer percentile, as in stats.py; tied compare trials are excluded
        def _key(x):
            p1 = int(round(x['pct'])); p2 = int(round(x['pct2'])) if x.get('pct2') is not None else None
            if task == 'read': return p1 > 50
            return None if p1 == p2 else (1 if p1 > p2 else 2)
        _t = [_key(x) for k, x in _r.items() if k.startswith(task + '|') and x.get('pct') is not None]
        _t = [t for t in _t if t is not None]
        if not _t: continue
        _share = max(_t.count(v) for v in set(_t)) / len(_t)
        cmd(f'pc{b}{task.capitalize()}Majority', _share, 'pct')

def msi_macros(M, pre):
    cmd(pre + 'K', M['k'], 'int'); cmd(pre + 'N', M['n_patients'], 'int')
    cmd(pre + 'High', M['n_msi_high'], 'int'); cmd(pre + 'Stable', M['n_mss'], 'int')
    cmd(pre + 'Prev', M['prevalence'], 'pct')
    for a in ('own', 'swap'):
        if a not in M['arms']: continue
        v = M['arms'][a]; A_ = a.capitalize()
        cmd(f'{pre}{A_}', v['auroc']); cmd(f'{pre}{A_}Lo', v['auroc_ci'][0]); cmd(f'{pre}{A_}Hi', v['auroc_ci'][1])
        cmd(f'{pre}{A_}Nval', v['n_distinct_values'], 'int'); cmd(f'{pre}{A_}Mode', v['modal_share'], 'pct')
        cmd(f'{pre}{A_}Mean', v['mean_p'], '{:.3f}'); cmd(f'{pre}{A_}SD', v['sd_p'], '{:.3f}')
        cmd(f'{pre}{A_}Val', v['modal_value'], '{:.2f}')
        cmd(f'{pre}{A_}Msih', v['mean_p_msih'], '{:.3f}'); cmd(f'{pre}{A_}Mss', v['mean_p_mss'], '{:.3f}')
        # the reversed-orientation AUROC on the same observations, 1 - AUROC: an arithmetic
        # illustration that a score below one half is a reverse-direction association, not a performance
        cmd(f'{pre}{A_}Rev', 1.0 - v['auroc'])
        if 'auprc' in v:
            cmd(f'{pre}Auprc{A_}', v['auprc']); cmd(f'{pre}Brier{A_}', v['brier'])
            cmd(f'{pre}Cal{A_}', v['calibration_in_the_large'], '{:+.3f}')
    if 'constant' in M:
        cmd(pre + 'AuprcConst', M['constant']['auprc']); cmd(pre + 'BrierConst', M['constant']['brier'])
    cmd(pre + 'Logit', M['logistic_auroc']); cmd(pre + 'LogitLo', M['logistic_auroc_ci'][0])
    cmd(pre + 'LogitHi', M['logistic_auroc_ci'][1])
    if 'logistic_auprc' in M:
        cmd(pre + 'AuprcLogit', M['logistic_auprc']); cmd(pre + 'BrierLogit', M['logistic_brier'])
        cmd(pre + 'CalLogit', M['logistic_calibration_in_the_large'], '{:+.3f}')
    if 'logistic_auroc_inductive' in M:
        cmd(pre + 'LogitInd', M['logistic_auroc_inductive']); cmd(pre + 'LogitIndLo', M['logistic_auroc_inductive_ci'][0])
        cmd(pre + 'LogitIndHi', M['logistic_auroc_inductive_ci'][1]); cmd(pre + 'LogitIndOverlap', M['logistic_inductive_panel_overlap'], 'pct')
    if 'logistic_auroc_frozen_swap' in M:
        cmd(pre + 'LogitFrozenSwap', M['logistic_auroc_frozen_swap']); cmd(pre + 'LogitFrozenSwapLo', M['logistic_auroc_frozen_swap_ci'][0])
        cmd(pre + 'LogitFrozenSwapHi', M['logistic_auroc_frozen_swap_ci'][1])
    if 'own_finish_length_n' in M:
        cmd(pre + 'OwnFinishLength', M['own_finish_length_n'], 'int'); cmd(pre + 'OwnFinishN', M['own_finish_n'], 'int')
    if 'logistic_auroc_swapped' in M:
        cmd(pre + 'LogitSwap', M['logistic_auroc_swapped']); cmd(pre + 'LogitSwapLo', M['logistic_auroc_swapped_ci'][0]); cmd(pre + 'LogitSwapHi', M['logistic_auroc_swapped_ci'][1])
    cmd(pre + 'BestUni', M['best_univariate_auroc'])
    if 'best_univariate_id' in M:
        cmd(pre + 'BestUniId', M['best_univariate_id'].replace('_', '\\_')); cmd(pre + 'BestUniDir', M['best_univariate_direction'])
        cmd(pre + 'BestUniOof', M['best_univariate_auroc_oof'])
    if 'count' in M['arms']:
        c = M['arms']['count']
        cmd(pre + 'CountN', c['n'], 'int'); cmd(pre + 'CountExact', c['exact'], 'pct')
        cmd(pre + 'CountWithin', c['within_two'], 'pct'); cmd(pre + 'CountMae', c['mae'], '{:.2f}')
        cmd(pre + 'CountR', c['spearman'], '{:.3f}')
        cmd(pre + 'CountTruth', c['truth_mean'], '{:.1f}'); cmd(pre + 'CountPred', c['pred_mean'], '{:.1f}')
    if 'reason_distinct' in M:
        cmd(pre + 'ReasonDistinct', M['reason_distinct'], 'int'); cmd(pre + 'ReasonN', M['reason_n'], 'int')
    # How much of the panel the distinct rationales quote, counted on the complete rationales only
    # (the archive keeps the first 300 characters of each reply); computed in msi_stats.py.
    if 'reason_complete_n' in M:
        cmd(pre + 'ReasonComplete', M['reason_complete_n'], 'int'); cmd(pre + 'ReasonTrunc', M['reason_truncated_n'], 'int')
        cmd(pre + 'ReasonCompleteDistinct', M['reason_complete_distinct'], 'int')
        cmd(pre + 'ReasonDigit', M['reason_complete_digit'], 'int'); cmd(pre + 'ReasonPct', M['reason_complete_percentile'], 'int')
        cmd(pre + 'ReasonDigitPct', 100.0 * M['reason_complete_digit'] / max(M['reason_complete_distinct'], 1), '{:.0f}')
        cmd(pre + 'ReasonDigitAll', M['reason_any_digit_all'], 'int'); cmd(pre + 'ReasonPctAll', M['reason_any_percentile_all'], 'int')
        cmd(pre + 'ReasonDigitAllPct', 100.0 * M['reason_any_digit_all'] / max(M['reason_distinct'], 1), '{:.0f}')
        cmd(pre + 'ReasonLenComplete', M['reason_mean_len_complete'], '{:.0f}'); cmd(pre + 'ReasonLenTrunc', M['reason_mean_len_truncated'], '{:.0f}')
    ov = M.get('own_vs_swap')
    if ov:
        cmd(pre + 'Diff', ov['auroc_diff'], '{:+.4f}'); cmd(pre + 'DiffLo', ov['ci95'][0], '{:+.4f}'); cmd(pre + 'DiffHi', ov['ci95'][1], '{:+.4f}')
        cmd(pre + 'DiffLoNinety', ov['ci90'][0], '{:+.4f}'); cmd(pre + 'DiffHiNinety', ov['ci90'][1], '{:+.4f}')
        cmd(pre + 'DiffLoNF', ov['ci95'][0], '{:+.4f}'); cmd(pre + 'DiffHiNF', ov['ci95'][1], '{:+.4f}')
        cmd(pre + 'DiffP', ov['boot_p'], '{:.2f}'); cmd(pre + 'DiffN', ov['n'], 'int')
        # the bootstrap p is a share of resamples (B draws), so a value below the two-decimal floor is
        # reported as an inequality rather than as "= 0.00"
        _bp = ov['boot_p']
        cmd(pre + 'DiffPRel', ('< 0.001' if _bp < 0.001 else '< 0.01' if _bp < 0.01 else f"= {_bp:.3f}"))
        cmd(pre + 'DiffEquivFive', 'passes' if ov['equivalent_005'] else 'does not pass')
        cmd(pre + 'DiffEquivTwo', 'passes' if ov['equivalent_002'] else 'does not pass')
    if 'best_univariate_folds_agree' in M:
        cmd(pre + 'BestUniFoldsAgree', 'the same reaction with the same sign in all five folds' if M['best_univariate_folds_agree'] else 'not the same reaction in every fold')


# the label construction, from the companion's label builder (paper1/code/build_labels_ht29.py)
_lbl = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'paper1', 'data', 'labels_ht29.json')
if os.path.exists(_lbl):
    LB_ = json.load(open(_lbl)); _mm = [v['n_matched'] for v in LB_['models'].values()]
    cmd('lblModels', len(LB_['models']), 'int'); cmd('lblMatchMin', min(_mm), 'int'); cmd('lblMatchMax', max(_mm), 'int')
    cmd('lblUnionActive', LB_['union_active'], 'int'); cmd('lblHtActive', LB_['ht29_active'], 'int')
    cmd('lblUnionNotHt', LB_['union_active_not_ht29'], 'int'); cmd('lblHtPrev', LB_['ht29_prevalence'], 'pct')
    # the label-provenance table rows: model, tissue of origin in the NCI-60 panel, reactions, matched
    _tissue = {'786-O': 'renal', 'HOP-62': 'lung (non-small cell)', 'HOP-92': 'lung (non-small cell)',
               'HS-578T': 'breast', 'HT29': 'colon', 'MALME-3M': 'melanoma', 'MDA-MB-231/ATCC': 'breast',
               'NCI-H226': 'lung (non-small cell)', 'RPMI-8226': 'leukemia panel (myeloma)',
               'SR': 'leukemia panel (lymphoma)', 'UO-31': 'renal'}
    _rows = sorted(LB_['models'].values(), key=lambda v: v['model_id'].upper())
    L.append('\\newcommand{\\lblRows}{' + ' '.join(
        f"{v['model_id'].replace('/ATCC', '')} & {_tissue.get(v['model_id'], '')} & {v['n_reactions']:,} & {v['n_matched']:,} \\\\"
        for v in _rows) + '}')
# every reaction-activity run rescored against the tissue-matched label (HT29 alone)
HT = R.get('label_ht29')
if HT:
    cmd('htActive', HT['n_active'], 'int'); cmd('htPrev', HT['prevalence'], 'pct'); cmd('htIndNetwork', HT['indicator_floor_network'])
    for tag, v in HT['runs'].items():
        t = TAGS[tag]
        cmd(f'ht{t}Base', v['label_base_rate'], '{:.3f}'); cmd(f'ht{t}Raw', v['raw_expression_auroc']); cmd(f'ht{t}Ind', v['indicator_floor'])
        for a, A_ in (('reaction', 'Reaction'), ('patient', 'Patient'), ('shuffled', 'Shuffled')):
            if a in v['per_arm']:
                w = v['per_arm'][a]; cmd(f'ht{t}{A_}', w['auroc']); cmd(f'ht{t}{A_}Lo', w['auroc_ci'][0]); cmd(f'ht{t}{A_}Hi', w['auroc_ci'][1])
        for k, nm in (('own_minus_stranger', 'OwnStr'), ('own_minus_blind', 'OwnBlind'), ('stranger_minus_blind', 'StrBlind')):
            if k in v:
                w = v[k]; cmd(f'ht{t}{nm}', w['mean'], '{:+.4f}'); cmd(f'ht{t}{nm}Lo', w['ci90'][0], '{:+.4f}'); cmd(f'ht{t}{nm}Hi', w['ci90'][1], '{:+.4f}')
                cmd(f'ht{t}{nm}Pos', w['n_positive'], 'int'); cmd(f'ht{t}{nm}N', w['n'], 'int')
    # the largest own-minus-stranger reading and whether any arm reaches either reference, across runs
    _gaps = [v['own_minus_stranger']['mean'] for v in HT['runs'].values() if 'own_minus_stranger' in v]
    if _gaps: cmd('htOwnStrMax', max(_gaps), '{:+.4f}'); cmd('htOwnStrMin', min(_gaps), '{:+.4f}')
    _below = all(v['per_arm'][a]['auroc'] < min(v['raw_expression_auroc'], v['indicator_floor']) for v in HT['runs'].values() for a in v['per_arm'])
    cmd('htAllBelow', 'every arm in every design' if _below else 'not every arm')
    cmd('htRuns', len(HT['runs']), 'int')
    cmd('htRunsWord', {1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven'}.get(len(HT['runs']), str(len(HT['runs']))))
    # the arms that rise above the availability baseline under the tissue-matched label, named
    _armname = {'reaction': 'patient-blind', 'patient': 'own-data', 'shuffled': "stranger's-data"}
    _above = [(TAGS[tag], a, v['per_arm'][a]['auroc'] - v['indicator_floor'], v['per_arm'][a]['auroc'] < v['raw_expression_auroc']) for tag, v in HT['runs'].items() for a in v['per_arm'] if v['per_arm'][a]['auroc'] >= v['indicator_floor']]
    cmd('htAboveIndN', len(_above), 'int')
    _desc = {'A': 'Design~A on the main backbone', 'B': 'Design~B', 'E': 'Design~E', 'S': 'Design~A on the second backbone', 'T': 'Design~A on the third backbone'}
    cmd('htAboveIndList', '; '.join(f"the {_armname[a]} arm of {_desc.get(t, t)} (by {d:+.4f})" for t, a, d, _ in _above) if _above else 'none')
    # a clause for the sentence that says every arm stays below both references: empty when that is
    # true, otherwise the exceptions named (the raw ranking is checked separately below)
    if _above:
        _lead = 'the one exception is ' if len(_above) == 1 else 'the exceptions are '
        _verb = 'rises' if len(_above) == 1 else 'rise'
        cmd('htAboveIndClause', '; ' + _lead + ' and '.join(f"the {_armname[a]} arm of {_desc.get(t, t)}, which {_verb} above the availability baseline by {d:+.4f}" + (' while staying below the raw ranking' if br else ' and above the raw ranking as well') for t, a, d, br in _above))
    else:
        cmd('htAboveIndClause', '')
    _above_raw = [1 for v in HT['runs'].values() for a in v['per_arm'] if v['per_arm'][a]['auroc'] >= v['raw_expression_auroc']]
    cmd('htAboveRawN', len(_above_raw), 'int')
    _excl = [1 for v in HT['runs'].values() if 'own_minus_stranger' in v and (v['own_minus_stranger']['ci90'][0] > 0 or v['own_minus_stranger']['ci90'][1] < 0)]
    cmd('htOwnStrExclZeroN', len(_excl), 'int')
    _w = {0: 'none', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five'}.get(len(_excl), str(len(_excl)))
    _runs_w = {1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven'}.get(len(HT['runs']), str(len(HT['runs'])))
    cmd('htOwnStrExclZeroPhrase', f"{_w} of the {_runs_w} intervals {'excludes' if len(_excl) <= 1 else 'exclude'} zero")

for _k, _pre in (('msi', 'msi'), ('msi_backbone2', 'msiTwo'), ('msi_backbone3', 'msiThree'),
                 ('msi_first', 'msiFirst'), ('msi_backbone2_first', 'msiTwoFirst'), ('msi_backbone3_first', 'msiThreeFirst'),
                 ('msi_blocked', 'msiBlocked'), ('msi_backbone2_blocked', 'msiTwoBlocked'), ('msi_backbone3_blocked', 'msiThreeBlocked')):
    if R.get(_k):
        msi_macros(R[_k], _pre)
        if R[_k].get('full_archive'): cmd(_pre + 'FullArchive', 'yes')
        if R[_k].get('collection_kind') == 'interleaved': cmd(_pre + 'IlPrimary', 'yes')
# the best own-panel AUROC over the backbones' primary probe collections, for the conclusion
_own = [R[_k]['arms']['own']['auroc'] for _k in ('msi', 'msi_backbone2', 'msi_backbone3') if R.get(_k) and 'own' in R[_k].get('arms', {})]
if _own: cmd('msiBestOwn', max(_own))
# the earlier collections against the one reported (the first, prefix-archive collection and, where
# the interleaved collection is primary, the blocked complete-archive one): AUROC then and now
_il_moves, _il_dmoves, _il_flips = [], [], []
_bk_name = {'msi': 'the main backbone', 'msi_backbone2': 'the second backbone', 'msi_backbone3': 'the third backbone'}
for _k, _pre in (('msi', 'msi'), ('msi_backbone2', 'msiTwo'), ('msi_backbone3', 'msiThree')):
    _c = R.get(_k)
    for _suffix, _tag in (('_first', 'First'), ('_blocked', 'Blocked')):
        _f = R.get(_k + _suffix)
        if _f and _c and _f is not _c:
            for a in ('own', 'swap'):
                if a in _f['arms'] and a in _c['arms']:
                    cmd(f'{_pre}{a.capitalize()}{_tag}Minus', _c['arms'][a]['auroc'] - _f['arms'][a]['auroc'], '{:+.4f}')
            if _f.get('own_vs_swap') and _c.get('own_vs_swap'):
                cmd(f'{_pre}Diff{_tag}', _f['own_vs_swap']['auroc_diff'], '{:+.4f}')
                if _suffix == '_blocked':
                    _il_moves += [abs(_c['arms'][a]['auroc'] - _f['arms'][a]['auroc']) for a in ('own', 'swap') if a in _f['arms'] and a in _c['arms']]
                    _il_dmoves.append(abs(_c['own_vs_swap']['auroc_diff'] - _f['own_vs_swap']['auroc_diff']))
                    if (_c['own_vs_swap']['auroc_diff'] > 0) != (_f['own_vs_swap']['auroc_diff'] > 0): _il_flips.append(_bk_name[_k])
if R.get('msi_interleaved'):
    cmd('msiInterleaved', 'yes')
    if _il_moves:
        cmd('msiIlAurocMaxAbs', max(_il_moves)); cmd('msiIlOwnStrMaxAbsChange', max(_il_dmoves))
        cmd('msiIlFlipPhrase', "no backbone's own-minus-stranger difference changes sign between the two" if not _il_flips
            else 'the own-minus-stranger difference changes sign on ' + ' and '.join(_il_flips))

# release identifiers (repository root release.json): the Git tag the manuscript corresponds to and,
# once minted, the version DOIs; an empty field leaves its macro undefined
for _cand in (os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', 'release.json'), 'release.json', '../release.json'):
    if os.path.exists(_cand):
        _rel = json.load(open(_cand))
        for _k, _m in (('tag', 'releaseTag'), ('version_doi_code', 'releaseDoiCode'), ('version_doi_data', 'releaseDoiData'), ('commit', 'releaseCommit')):
            if _rel.get(_k):
                _v = str(_rel[_k])
                # twelve hex digits identify the commit without printing a wall of hash
                if _m == 'releaseCommit' and len(_v) > 12: _v = _v[:12]
                cmd(_m, _v.replace('_', '\\_'))
        break

# the crossed bootstrap at 10,000 draws under three recorded seeds (crossed_boot_10k.py): the
# Monte Carlo range of each endpoint and whether the 0.02 reading is the same under every seed
_cb = rawload('..', 'crossed_boot_10k.json') if False else None
for _cand in ('results/crossed_boot_10k.json', 'crossed_boot_10k.json'):
    if os.path.exists(_cand): _cb = json.load(open(_cand)); break
if _cb:
    cmd('cbTenB', _cb['B'], 'int'); cmd('cbTenSeeds', len(_cb['seeds']), 'int')
    for tag, d in _cb['designs'].items():
        t = TAGS.get(tag)
        if not t: continue
        cmd(f'r{t}CbTenHiMin', d['ci90_upper_range'][0], '{:+.4f}'); cmd(f'r{t}CbTenHiMax', d['ci90_upper_range'][1], '{:+.4f}')
        cmd(f'r{t}CbTenLoMin', d['ci90_lower_range'][0], '{:+.4f}'); cmd(f'r{t}CbTenLoMax', d['ci90_lower_range'][1], '{:+.4f}')
        cmd(f'r{t}CbTenHiMinNF', d['ci95_upper_range'][0], '{:+.4f}'); cmd(f'r{t}CbTenHiMaxNF', d['ci95_upper_range'][1], '{:+.4f}')
        cmd(f'r{t}CbTenLoMinNF', d['ci95_lower_range'][0], '{:+.4f}'); cmd(f'r{t}CbTenLoMaxNF', d['ci95_lower_range'][1], '{:+.4f}')
        cmd(f'r{t}CbTenWithinTwo', 'lies inside' if d['within_002_90_all_seeds'] else ('lies inside under some seeds and not others' if d['within_002_90_any_seed'] else 'does not lie inside'))
        cmd(f'r{t}CbTenWithinTwoNF', 'lies inside' if d['within_002_95_all_seeds'] else 'does not lie inside')
        cmd(f'r{t}CbTenSpread', max(d['ci90_upper_range'][1] - d['ci90_upper_range'][0], d['ci90_lower_range'][1] - d['ci90_lower_range'][0]), '{:.4f}')
        cmd(f'r{t}CbTenVerdictStable', 'the same' if (d['within_002_90_all_seeds'] or not d['within_002_90_any_seed']) else 'not the same')

# the tie audit (tie_audit.py): how many evaluated entries sit on exactly tied stored values
_ta = None
for _cand in ('results/tie_audit.json', 'tie_audit.json'):
    if os.path.exists(_cand): _ta = json.load(open(_cand)); break
if _ta:
    for key, pre in (('msi_panel', 'tieMsi'), ('A_representative', 'tieA'), ('E_representative40', 'tieE')):
        d = _ta['designs'].get(key)
        if not d: continue
        cmd(pre + 'Entries', d['n_entries'], 'int'); cmd(pre + 'Exact', d['tied_exact'], 'int'); cmd(pre + 'ExactPct', d['tied_exact_share'], '{:.2f}')
        cmd(pre + 'Round', d['tied_after_rounding'], 'int'); cmd(pre + 'RoundPct', d['tied_after_rounding_share'], '{:.2f}')
        cmd(pre + 'SpreadMax', d['printed_percentile_spread_within_tie_max'], '{:.0f}'); cmd(pre + 'SpreadMean', d['printed_percentile_spread_within_tie_mean'], '{:.1f}')
        cmd(pre + 'ShiftMean', d['midrank_shift_of_tied_entries_mean'], '{:.1f}'); cmd(pre + 'ShiftMax', d['midrank_shift_of_tied_entries_max'], '{:.0f}')
        cmd(pre + 'Rxn', d['n_reactions'], 'int'); cmd(pre + 'Pat', d['n_patients'], 'int')
    c = _ta['cohort_expression_bearing']
    cmd('tieCohortPct', c['tied_exact_share'], '{:.2f}')

# the immutable record of the served model artifacts (model_records.py): one digest per backbone,
# so a model swapped under the same served name would change the manuscript rather than pass
_mr = None
for _cand in ('results/model_records.json', 'model_records.json'):
    if os.path.exists(_cand): _mr = json.load(open(_cand)); break
if _mr:
    _MB = {'mistral-small-3.2-24b-instruct': 'Mi', 'gemma-4-31b-it': 'Ge', 'qwen3-8-27b': 'Qw'}
    _nrec = 0
    for _m in _mr['models']:
        _t = _MB.get(_m['served_name'])
        if not _t: continue
        pre = 'mr' + _t
        cmd(pre + 'Name', _m['served_name'].replace('_', '\\_'))
        if _m.get('artifact_digest'):
            _nrec += 1
            cmd(pre + 'Digest', _m['artifact_digest'][:16])
            cmd(pre + 'Files', _m['n_files'], 'int')
            cmd(pre + 'GiB', _m['total_bytes'] / (1024 ** 3), '{:.1f}')
            if _m.get('revision'): cmd(pre + 'Rev', _m['revision'][:12])
        else:
            cmd(pre + 'Digest', 'not recorded')
    cmd('mrRecorded', _nrec, 'int'); cmd('mrModels', len(_mr['models']), 'int')
    cmd('mrRecordedWord', ['no', 'one', 'two', 'three'][_nrec] if _nrec < 4 else str(_nrec))
    cmd('mrGeneratedDate', str(_mr.get('generated_utc', ''))[:10])

# what the archives of the external collections record about their deployments
# (deployment_records.py): per backbone, the served name, endpoint, launch identity, timestamps,
# decoding settings and call counts, and the list of what was not recorded
_dp = None
for _cand in ('results/msi2/deployment_records.json', 'msi2/deployment_records.json'):
    if os.path.exists(_cand): _dp = json.load(open(_cand)); break
if _dp:
    _DB = {'mistral-small-3.2-24b-instruct': 'Mi', 'gemma-4-31b-it': 'Ge', 'qwen3-8-27b': 'Qw'}
    _tex = lambda s: s.replace('_', '\\_').replace('%', '\\%').replace('#', '\\#')
    _ndp = 0
    for _b in _dp['backbones']:
        _t = _DB.get(_b['served_name'])
        if not _t: continue
        _ndp += 1; pre = 'xdp' + _t
        cmd(pre + 'Served', _tex(_b['served_name']))
        # breakable forms for narrow table columns: a break is allowed after every hyphen and slash
        _brk = lambda t: _tex(t).replace('-', '-\\allowbreak{}').replace('/', '/\\allowbreak{}')
        cmd(pre + 'ServedBrk', _brk(_b['served_name']))
        cmd(pre + 'Host', 'loopback' if _b['host'] == 'loopback' else 'remote')
        _u = _b['url'][0]
        _host = _u.split('//', 1)[-1].split('/', 1)[0]           # host[:port] only; the path names the model
        # a loopback address is printed; a remote host is left to the design records in the release
        cmd(pre + 'Endpoint', _tex(_host) if _b['host'] == 'loopback' else 'URL in the design records')
        _l = _b.get('launch') or {}
        cmd(pre + 'HubId', _tex(_l['hub_id']) if _l.get('hub_id') else 'not held by us')
        cmd(pre + 'HubIdBrk', _brk(_l['hub_id']) if _l.get('hub_id') else 'not held by us')
        cmd(pre + 'Dtype', _l['dtype'] if _l.get('dtype') else 'not recorded')
        cmd(pre + 'Template', _l.get('template') or 'n/a')
        cmd(pre + 'Coll', _b['n_collections'], 'int')
        cmd(pre + 'Calls', _b['calls_archived'], 'int')
        cmd(pre + 'Failed', _b['failed_calls'], 'int')
        cmd(pre + 'Retried', _b['calls_needing_retry'], 'int')
        cmd(pre + 'First', _b['first_started_utc'].replace(' UTC', ''))
        cmd(pre + 'Last', _b['last_started_utc'].replace(' UTC', ''))
        cmd(pre + 'Minutes', _b['minutes_total'], '{:.0f}')
        cmd(pre + 'Temp', _b['temperature'][0], '{:g}')
        cmd(pre + 'MaxTok', _b['max_tokens'][0], 'int')
        _len = _b['finish_reasons'].get('length', 0)
        cmd(pre + 'LenStops', _len, 'int')
        _echo = set(_b['echoed_names'].keys()) == {_b['served_name']}
        cmd(pre + 'Echo', 'every reply' if _echo else 'not every reply')
        cmd(pre + 'PromptTokMed', _b['prompt_tokens_median'], '{:.0f}')
        cmd(pre + 'PromptTokMin', _b['prompt_tokens_min'], 'int')
        cmd(pre + 'PromptTokMax', _b['prompt_tokens_max'], 'int')
        cmd(pre + 'LatencyMed', _b['latency_s_median'], '{:.1f}')
    cmd('xdpBackbones', _ndp, 'int')
    _dates = sorted({_b['first_started_utc'][:10] for _b in _dp['backbones']} |
                    {_b['last_started_utc'][:10] for _b in _dp['backbones']})
    cmd('xdpDate', _dates[0] if len(_dates) == 1 else f'{_dates[0]} to {_dates[-1]}')
    _allcalls = sum(_b['calls_archived'] for _b in _dp['backbones'])
    _allfail = sum(_b['failed_calls'] for _b in _dp['backbones'])
    _alllen = sum(_b['finish_reasons'].get('length', 0) for _b in _dp['backbones'])
    cmd('xdpCallsAll', _allcalls, 'int'); cmd('xdpFailedAll', _allfail, 'int'); cmd('xdpLenStopsAll', _alllen, 'int')

# the frozen percentile-only interface on the external cohorts (msi2_stats.py): the primary
# confirmatory contrast and every secondary row
_m2 = None
for _cand in ('results/msi2_frozen.json', 'msi2_frozen.json'):
    if os.path.exists(_cand): _m2 = json.load(open(_cand)); break
if _m2:
    _BB = {'mistral-small-3.2-24b-instruct': 'Mi', 'gemma-4-31b-it': 'Ge', 'qwen3-8-27b': 'Qw'}
    _CO = {'tcga': 'Dev', 'gse39582': 'ExtA', 'gse13294': 'ExtB'}
    _CF = {'zero_shot': 'Zero', 'evidence': 'Evid', 'tool': 'Tool'}
    for run, r in _m2['runs'].items():
        b = _BB.get(r.get('backbone')); c = _CO.get(r['cohort']); f = _CF.get(r['config'])
        if not (b and c and f): continue
        pre = 'x' + b + c + f
        cmd(pre + 'Own', r['own_auroc']); cmd(pre + 'OwnAuprc', r['own_auprc'])
        cmd(pre + 'Diff', r['own_minus_donor'], '{:+.4f}'); cmd(pre + 'DiffAbs', abs(r['own_minus_donor']))
        # the interval and the null must belong to the same estimate as the point: the collected
        # donor arm where it exists, the cached reconstruction only where it does not
        _ci = r.get('own_minus_donor_direct_ci95') or r['own_minus_donor_ci95']
        # the reported p is the Monte Carlo estimator (b + 1) / (B + 1) on the collected arm where it
        # exists; the plain proportion is kept beside it, and neither reports zero exceedances as zero
        _np_ = r.get('own_minus_donor_direct_null_p_mc')
        if _np_ is None: _np_ = r.get('own_minus_donor_direct_null_p')
        if _np_ is None: _np_ = r.get('exch_null_p_mc', r['exch_null_p'])
        _b = r.get('own_minus_donor_direct_null_exceedances')
        if _b is None: _b = r.get('exch_null_exceedances')
        if _b is not None:
            cmd(pre + 'NullExceed', int(_b), 'int')
            cmd(pre + 'NullDraws', int(r.get('own_minus_donor_direct_null_draws') or r.get('exch_null_draws') or 0), 'int')
        cmd(pre + 'DiffLo', _ci[0], '{:+.4f}'); cmd(pre + 'DiffHi', _ci[1], '{:+.4f}')
        cmd(pre + 'DiffCachedLo', r['own_minus_donor_ci95'][0], '{:+.4f}')
        cmd(pre + 'DiffCachedHi', r['own_minus_donor_ci95'][1], '{:+.4f}')
        cmd(pre + 'NullPCached', r['exch_null_p'], 'p')
        cmd(pre + 'NullP', _np_, 'p'); cmd(pre + 'N', r['n_eval'], 'int'); cmd(pre + 'NPos', r['n_pos'], 'int')
        cmd(pre + 'Brier', r['brier_own']); cmd(pre + 'SchedSD', r['own_minus_donor_schedule_sd'])
        # the donor arm's own AUROC, so the contrast can be read against the two levels that make it
        if r.get('donor_auroc') is not None: cmd(pre + 'Donor', r['donor_auroc'])
        if r.get('donor_auroc_cached') is not None: cmd(pre + 'DonorCached', r['donor_auroc_cached'])
        # the cached reconstruction beside the collected donor calls: the two differ only by serving
        # variation, and reporting both is what keeps the primary from resting on a determinism claim
        if r.get('own_minus_donor_cached') is not None:
            cmd(pre + 'DiffCached', r['own_minus_donor_cached'], '{:+.4f}')
            if r.get('donor_auroc') is not None:
                _gap = r['own_minus_donor'] - r['own_minus_donor_cached']
                cmd(pre + 'DiffCachedGap', _gap, '{:+.4f}'); cmd(pre + 'DiffCachedGapAbs', abs(_gap))
        if r.get('donor_served_vs_cached_n_changed') is not None:
            cmd(pre + 'DonorServChanged', r['donor_served_vs_cached_n_changed'], 'int')
            cmd(pre + 'DonorServN', r['donor_served_vs_cached_n'], 'int')
            cmd(pre + 'DonorServMax', r['donor_served_vs_cached_max_abs'])
        cmd(pre + 'Dir', 'above' if r['own_minus_donor'] > 0 else 'below')
        cmd(pre + 'Word', 'higher' if r['own_minus_donor'] > 0 else 'lower')
        # the reading the interval and the null support, generated so it follows the numbers
        _lo, _hi = _ci
        cmd(pre + 'Reading', 'a correctly directed advantage for the patient\'s own panel' if (_lo > 0 and _np_ < 0.05)
            else ('a reversal, the donor panel reading the phenotype better' if _hi < 0
                  else 'an inconclusive comparison, the interval spanning zero'))
        if r.get('serving_repeat_median_abs') is not None:
            cmd(pre + 'ServMed', r['serving_repeat_median_abs']); cmd(pre + 'ServMax', r['serving_repeat_max_abs'])
            if r.get('serving_repeat_n_changed') is not None:
                cmd(pre + 'ServChanged', r['serving_repeat_n_changed'], 'int')
                cmd(pre + 'ServRepeatN', r['serving_repeat_n'], 'int')
        # the Brier score of a constant predictor returning this cohort's own prevalence, so a
        # returned probability can be read against the trivial alternative rather than in isolation
        if r.get('brier_own') is not None and r.get('n_eval'):
            _p = r['n_pos'] / r['n_eval']
            cmd(pre + 'BrierConst', _p * (1.0 - _p))
            cmd(pre + 'BrierVsConst', 'better than' if r['brier_own'] < _p * (1.0 - _p) else 'worse than')
        if r.get('tool_only_auroc') is not None:
            cmd(pre + 'ToolOnly', r['tool_only_auroc']); cmd(pre + 'ToolFid', r['tool_fidelity_median_abs'])
            cmd(pre + 'ToolFidPct', 100.0 * r['tool_fidelity_within_0.05'], '{:.0f}')
        if r.get('calibrated'):
            cmd(pre + 'CalBrier', r['calibrated']['brier']); cmd(pre + 'CalAuroc', r['calibrated']['auroc'])
            cmd(pre + 'CalLogloss', r['calibrated']['logloss'])
            cmd(pre + 'CalFallback', 'yes' if r['calibrated'].get('constant_fallback') else 'no')
    # the calibration maps themselves (secondary; fitted on the development cohort, applied unchanged
    # externally; a slope at or below zero is replaced by the constant map, and both are recorded)
    _nfb = 0; _ncal = 0
    for _key, _c in (_m2.get('calibration') or {}).items():
        _mdl, _cfg, _ep = _key.split('|')
        b = _BB.get(_mdl); f = _CF.get(_cfg)
        if not (b and f) or _ep != 'nonmsih': continue
        pre = 'xCal' + b + f; _ncal += 1
        cmd(pre + 'FitA', _c.get('fitted_a', _c['a']), '{:+.3f}'); cmd(pre + 'FitB', _c.get('fitted_b', _c['b']), '{:+.3f}')
        cmd(pre + 'A', _c['a'], '{:+.3f}'); cmd(pre + 'B', _c['b'], '{:+.3f}')
        _fb = bool(_c.get('constant_fallback'))
        cmd(pre + 'Fallback', 'constant' if _fb else 'fitted'); _nfb += int(_fb)
    if _ncal:
        cmd('xCalMaps', _ncal, 'int'); cmd('xCalFallbackN', _nfb, 'int')
        cmd('xCalMapsWord', ['no', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'][_ncal] if _ncal < 10 else str(_ncal))
        cmd('xCalFallbackWord', ['no', 'one', 'two', 'three', 'four', 'five', 'six', 'seven'][_nfb] if _nfb < 8 else str(_nfb))
        _fbk = [k for k, c in _m2['calibration'].items() if c.get('constant_fallback') and k.endswith('|nonmsih')]
        _ord = {'qwen3-8-27b': 'first', 'gemma-4-31b-it': 'second', 'mistral-small-3.2-24b-instruct': 'third'}
        _cfn = {'zero_shot': 'zero-shot', 'evidence': 'evidence-assisted', 'tool': 'tool-assisted'}
        _parts = [f"the {_ord.get(k.split('|')[0], k.split('|')[0])} backbone's {_cfn.get(k.split('|')[1], k.split('|')[1])} map" for k in _fbk]
        cmd('xCalFallbackList', (', '.join(_parts[:-1]) + ' and ' + _parts[-1]) if len(_parts) > 1 else (_parts[0] if _parts else 'none'))
    # the fixed design constants of the external inference, emitted so the text can name them
    cmd('xNullDraws', 10000, 'int'); cmd('xBootDrawsExt', 2000, 'int')
    cmd('xNullSeed', 13, 'int'); cmd('xBootSeed', 7, 'int'); cmd('xBootSeedDirect', 17, 'int')
    cmd('xNullSeedDirect', 23, 'int')
    cmd('planDate', '2026-09-10')
    cmd('xSchedSeedList', '11, 22, 33, 44, 55'); cmd('xSchedPrimarySeed', 11, 'int')
    # the five-schedule spread per backbone and configuration across the three cohorts, so the text
    # names every cohort's value rather than quoting one of them as a bound
    _CN = {'tcga': 'the development cohort', 'gse39582': 'the first external cohort', 'gse13294': 'the second external cohort'}
    _sd = {}
    for run, r in _m2['runs'].items():
        b = _BB.get(r.get('backbone')); f = _CF.get(r['config'])
        if b and f and r.get('own_minus_donor_schedule_sd') is not None:
            _sd.setdefault((b, f), {})[r['cohort']] = r['own_minus_donor_schedule_sd']
    for (b, f), _v in _sd.items():
        pre = 'x' + b + f
        cmd(pre + 'SchedSDMax', max(_v.values()))
        _parts = [f"{_v[c]:.4f} on {_CN[c]}" for c in ('tcga', 'gse39582', 'gse13294') if c in _v]
        cmd(pre + 'SchedSDList', (', '.join(_parts[:-1]) + ' and ' + _parts[-1]) if len(_parts) > 1 else _parts[0])
    # the collection ledger: every configuration attempted, and what became of it
    if _m2.get('collections_attempted') is not None:
        cmd('xColAttempted', _m2['collections_attempted'], 'int')
        cmd('xColAnalyzed', _m2['collections_analyzed'], 'int')
        cmd('xColComplete', _m2['responses_in_a_complete_collection'], 'int')
        _inc = _m2.get('incomplete') or {}
        cmd('xColIncomplete', len(_inc), 'int')
        cmd('xColIncompleteWord', {0: 'none', 1: 'one', 2: 'two', 3: 'three', 4: 'four'}.get(len(_inc), str(len(_inc))))
        _pretty = {'qwen': 'the first backbone', 'gemma-4-31b-it': 'the second backbone',
                   'mistral-small-3.2-24b-instruct': 'the third backbone'}
        _names = []
        for _k, _v in sorted(_inc.items()):
            _parts = _k.split('_')
            _who = next((w for m, w in _pretty.items() if m in _k), _k)
            _names.append(f"the tool-assisted collection on the development cohort for {_who} "
                          f"({_v['responses']} of {_m2['responses_in_a_complete_collection']} responses)")
        cmd('xColIncompleteList',
            ('none' if not _names else _names[0] if len(_names) == 1
             else ' and '.join([', '.join(_names[:-1]), _names[-1]])))

    for coh, r in _m2.get('frozen_reference_transfer', {}).items():
        pre = 'xRef' + _CO.get(coh, coh)
        cmd(pre + 'Auroc', r['auroc']); cmd(pre + 'Auprc', r['auprc']); cmd(pre + 'N', r['n'], 'int'); cmd(pre + 'NPos', r['n_pos'], 'int')
        cmd(pre + 'Diff', r['own_minus_donor'], '{:+.4f}')
    # the coverage record of the two external cohorts, so the supplement can account for every
    # deposited sample rather than only the ones scored
    for _cv in ('results/external/coverage.json', 'external/coverage.json'):
        if not os.path.exists(_cv): continue
        _cov = json.load(open(_cv))
        for _coh, _nm in (('gse39582', 'ExtA'), ('gse13294', 'ExtB')):
            _c = _cov.get(_coh)
            if not _c: continue
            pre = 'xCov' + _nm
            cmd(pre + 'Deposited', _c['n_samples_deposited'], 'int'); cmd(pre + 'Tumor', _c['n_tumor'], 'int')
            cmd(pre + 'NonTumor', _c['n_samples_deposited'] - _c['n_tumor'], 'int')
            cmd(pre + 'PanelCovered', _c['panel_covered'], 'int'); cmd(pre + 'PanelSize', _c['panel_size'], 'int')
            _lc = _c.get('label_counts') or {}
            cmd(pre + 'NoCall', _lc.get('NA', 0), 'int')
            cmd(pre + 'Called', sum(v for k, v in _lc.items() if k != 'NA'), 'int')
        break

    _fr = _m2['frozen_reference']['msih_vs_nonmsih']
    cmd('xRefDevOof', _fr['auroc_oof']); cmd('xRefC', _fr['C'], '{:g}')
    # the out-of-fold AUPRC and Brier score on the same out-of-fold prediction vector, so the
    # development row of the external table describes one vector throughout; the in-sample refit
    # applied to the development patients keeps its own name and is not put beside out-of-fold values
    if isinstance(_fr.get('oof'), dict) and 'score' in _fr['oof']:
        import numpy as _np
        from sklearn.metrics import average_precision_score as _aps, brier_score_loss as _bsl
        _yo = _np.array(_fr['oof']['y']); _so = _np.array(_fr['oof']['score'])
        cmd('xRefDevOofAuprc', float(_aps(_yo, _so))); cmd('xRefDevOofBrier', float(_bsl(_yo, _np.clip(_so, 0, 1))))
        cmd('xRefDevOofN', int(len(_yo)), 'int'); cmd('xRefDevOofNPos', int(_yo.sum()), 'int')
    cmd('xRefSingleId', _fr['single_feature']['reaction_id'].replace('_', '\\_'))
    cmd('xRefSingleAuroc', _fr['single_feature']['auroc_dev'])
    cmd('xRefPrev', 100.0 * _fr['prevalence'], '{:.1f}')

# what each collection's design.json actually records (collection_manifest.py), so the methods can
# state its provenance coverage instead of asserting a uniform one
for _cm in ('results/collection_manifest.json', 'collection_manifest.json'):
    if not os.path.exists(_cm): continue
    _man = json.load(open(_cm))
    cmd('cmCollections', _man['n_collections'], 'int')
    for _f, _nm in (('url', 'Url'), ('model', 'Model'), ('max_tokens', 'Tokens'), ('temperature', 'Temp'),
                    ('seed', 'Seed'), ('donor_seed', 'Donor'), ('started_at', 'Start'),
                    ('interleave', 'Interleave'), ('full_archive', 'Archive')):
        cmd('cm' + _nm, _man['n_recording_field'][_f], 'int')
    _all = [f for f in ('url', 'model', 'max_tokens', 'temperature', 'seed', 'donor_seed', 'started_at')
            if _man['n_recording_field'][f] == _man['n_collections']]
    cmd('cmUniversal', ', '.join(_all) if _all else 'the model name alone')
    break

open(OUTFILE, 'w').write('% generated by make_numbers.py -- do not edit\n' + '\n'.join(L) + '\n')
print(f'wrote {OUTFILE} with {len(L)} macros')
