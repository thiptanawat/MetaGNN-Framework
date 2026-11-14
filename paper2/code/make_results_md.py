#!/usr/bin/env python3
"""Regenerate RESULTS.md from results_frozen.json, so the summary cannot drift from the runs."""
import json
import _paths as PATHS
R = json.load(open(PATHS.data('results_frozen.json')))
N, C = R['network'], R['cohort']
def arm(t, a):
    v = R['runs'][t]['per_arm'][a]
    return f"{v['auroc']:.4f} [{v['auroc_ci'][0]:.4f}, {v['auroc_ci'][1]:.4f}]"
L = []
w = L.append
w("# Language-Model Evidence results, final\n")
w("Title: *Language-model annotation of metabolic reaction activity responds to a patient's transcriptome "
  "without gaining patient-specific information from it*\n")
w("Every number below is emitted by `stats.py` into `results_frozen.json` and substituted into the "
  "manuscript as a LaTeX macro. Nothing is transcribed by hand.\n")
w("## Setting\n")
w(f"| quantity | value |\n|---|---|\n"
  f"| Recon3D reactions | {N['n_reactions']:,} |\n"
  f"| labeled active (independent reconstructions) | {N['n_active']:,} ({N['base_rate']:.1%}) |\n"
  f"| expression-bearing reactions | {N['n_expression_bearing']:,} |\n"
  f"| reactions carrying a GPR rule | {C['n_gpr_rule']:,} |\n"
  f"| patients | {N['n_patients']} |\n"
  f"| prevalence, expression-bearing vs rest | {N['prevalence_expression_bearing']:.1%} vs {N['prevalence_no_expression']:.1%} |\n"
  f"| raw-expression baseline, whole network | {N['raw_expression_auroc']:.4f} +/- {N['raw_expression_auroc_sd']:.4f} |\n"
  f"| information-free indicator floor | {N['indicator_floor']:.4f} |\n")
w("## Main experiment\n")
w("| design | reaction only | + own expression | + stranger's | raw expression | indicator floor |\n|---|---|---|---|---|---|")
DESIGNS = [(t, nm) for t, nm in [('A_representative', 'A, representative'),
                                 ('B_balanced', 'B, balanced'),
                                 ('C_backbone2', 'C, balanced, backbone two'),
                                 ('E_representative40', 'E, representative, 40 patients')]
           if t in R['runs'] and 'patient_vs_shuffled' in R['runs'][t]]
for t, nm in DESIGNS:
    r = R['runs'][t]
    w(f"| {nm} | {arm(t,'reaction')} | {arm(t,'patient')} | {arm(t,'shuffled')} | "
      f"{r['raw_expression_auroc']:.4f} | {r['indicator_floor']:.4f} |")
w("\nNo arm in any design reaches the raw-expression baseline. In the representative design every arm "
  "also sits below the information-free indicator floor.\n")
w("## The three controls\n")
w("| quantity | " + " | ".join("Design " + nm.split(",")[0] for _, nm in DESIGNS) + " |\n|---|" + "---|" * len(DESIGNS))
for k, f, lab, pct in [('determinism', 'identical', 'determinism floor, identical answers (%)', 1),
                       ('determinism', 'mean_abs_diff', 'determinism floor, mean |diff|', 0),
                       ('patient_vs_shuffled', 'between_patient_sd_real', 'between-patient sd, own data', 0),
                       ('patient_vs_shuffled', 'between_patient_sd_stranger', 'between-patient sd, stranger', 0),
                       ('patient_vs_shuffled', 'identical', 'patient vs stranger, identical answers (%)', 1),
                       ('patient_vs_shuffled', 'mean_abs_diff', 'patient vs stranger, mean |diff|', 0)]:
    v = [R['runs'][t][k][f] for t, _ in DESIGNS]
    g = (lambda x: f'{100*x:.1f}') if pct else (lambda x: f'{x:.4f}')
    w(f"| {lab} | " + " | ".join(g(x) for x in v) + " |")
w("\n## Presence versus magnitude (R2 of the returned score)\n")
w("| design | identity | + presence + value | value-bearing only: identity | + the value |\n|---|---|---|---|---|")
for t, nm in DESIGNS:
    d = R['runs'][t]['variance_decomposition']
    w(f"| {nm.split(',')[0]} | {d['r2_reaction_identity']:.3f} | {d['r2_plus_magnitude']:.3f} | "
      f"{d.get('expr_only_r2_reaction_identity', float('nan')):.3f} | {d.get('expr_only_r2_plus_magnitude', float('nan')):.3f} |")
w("\nThe presence indicator and the value are collinear by construction, so no sequential share is attributed to either; "
  "the last two columns ask the question where it is well posed.")
s = R['runs']['A_representative']['strata']
w(f"\nMean returned p on reactions with no mapped gene, Design A: {s['reaction']['no_expression']['mean_p']:.3f} "
  f"without patient context, {s['patient']['no_expression']['mean_p']:.3f} with it. The model rebuilds the "
  "information-free indicator internally.\n")
if R.get('formats'):
    w("## Prompt format\n")
    w("| evidence written as | own profile | a stranger's | mean p, expression-bearing | mean p, no gene |\n|---|---|---|---|---|")
    ref = R['runs']['B_balanced']
    w(f"| a number plus its percentile (Design B, the reference for the format arms) | {ref['per_arm']['patient']['auroc']:.4f} | "
      f"{ref['per_arm']['shuffled']['auroc']:.4f} | "
      f"{ref['strata']['patient']['expression_bearing']['mean_p']:.3f} | "
      f"{ref['strata']['patient']['no_expression']['mean_p']:.3f} |")
    for tag, f in R['formats'].items():
        p = f['per_arm'].get('patient', {}); q = f['per_arm'].get('shuffled', {})
        w(f"| {tag} | {p.get('auroc', float('nan')):.4f} | {q.get('auroc', float('nan')):.4f} | "
          f"{p.get('mean_p_expression_bearing', float('nan')):.3f} | {p.get('mean_p_no_expression', float('nan')):.3f} |")
    w("")
w("## Positive controls (identical record format)\n")
w("| model | echo the value | compare two values | threshold a value | modal answer share, threshold |\n|---|---|---|---|---|")
_model = R['runs'].get('A_representative', {}).get('model', 'the served backbone')
for bk, nm in [('backbone1', _model), ('backbone2', 'second backbone')]:
    if bk not in R['positive_control']: continue
    e = R['positive_control'][bk]
    w(f"| {nm} | {e['echo']['accuracy']:.1%} | {e['compare']['accuracy']:.1%} | "
      f"{e['read']['accuracy']:.1%} | {e['read']['modal_answer_share']:.1%} |")
w("")
M = R.get('msi')
if M:
    w("## Patient-level target: microsatellite instability\n")
    w(f"{M['n_patients']} patients ({M['n_msi_high']} MSI-H, {M['n_mss']} MSS), a {M['k']}-reaction panel "
      "chosen by across-patient variance without consulting the label.\n")
    w("| method | AUROC | 95% CI |\n|---|---|---|")
    w(f"| logistic regression, same features, 5-fold CV | {M['logistic_auroc']:.4f} | {M['logistic_auroc_ci'][0]:.4f}, {M['logistic_auroc_ci'][1]:.4f} |")
    for a, nm in [('own', "language model, patient's own profile"), ('swap', "language model, a stranger's profile")]:
        v = M['arms'][a]; w(f"| {nm} | {v['auroc']:.4f} | {v['auroc_ci'][0]:.4f}, {v['auroc_ci'][1]:.4f} |")
    w(f"| best single reaction, in sample (optimistic) | {M['best_univariate_auroc']:.4f} | |")
    v = M['arms']['own']
    w(f"\nThe model returns {v['n_distinct_values']} distinct probabilities across all {M['n_patients']} patients; "
      f"{v['modal_share']:.1%} receive the value {v['modal_value']}. Among the {M['reason_n']} patients who received "
      f"it, there are {M['reason_distinct']} distinct free-text rationales.\n")
    c = M['arms']['count']
    w(f"Aggregation control (how many of the {M['k']} reactions sit at or above the 75th percentile): Spearman "
      f"{c['spearman']:.3f} with the true count over {c['n']} patients, exact on {c['exact']:.1%}, mean absolute "
      f"error {c['mae']:.2f}, predicted mean {c['pred_mean']:.2f} against a true mean of {c['truth_mean']:.2f}.\n")
D = R['runs'].get('D_reasoning')
if D:
    w("## Reasoning-enabled arm\n")
    w("| arm | responses parsed | mean p of the parsed answers |\n|---|---|---|")
    for a in ('reaction', 'patient', 'shuffled'):
        if a not in D['per_arm']: continue
        v = D['per_arm'][a]
        w(f"| {a} | {v['parsed']:.1%} | {v['mean_p']:.3f} |")
    w("\nWith reasoning enabled most calls returned an empty message because the whole token budget "
      "went to reasoning tokens the API does not return, and the attrition differs between arms in a "
      "way that is confounded with an endpoint outage. No accuracy is reported for this run, in the "
      "manuscript or here; it is a cost, attrition and answer-form report.\n")
# the prespecified replication (REPLICATION_PLAN_2026-11-13.md), from results/replication_frozen.json
import os as _os
_gp = _os.path.join(PATHS.ROOT, 'results', 'replication_frozen.json')
if _os.path.exists(_gp):
    G = json.load(open(_gp)); W1 = G['wording1']; d1 = W1['donor']['d1']
    w("## Prespecified replication of the own-versus-donor contrast\n")
    w("Plan: `REPLICATION_PLAN_2026-11-13.md`, committed before the first call. Statistics: `code/replication_stats.py`, "
      "written to `results/replication_frozen.json`.\n")
    w(f"First wording: {W1['n_patients']} patients and {W1['n_reactions']} reactions that no earlier collection used, "
      f"{W1['blocks_completed']} blocks of ten patients completed, {W1['n_calls']:,} calls.\n")
    w("| quantity | value |\n|---|---|")
    w(f"| own arm, AUROC | {d1['own_auroc']:.4f} |")
    for s_ in sorted(W1['donor']):
        w(f"| donor arm, schedule {s_[1:]}, AUROC | {W1['donor'][s_]['donor_auroc']:.4f} |")
    w(f"| raw expression ranking | {W1['raw_expression_auroc']:.4f} |")
    w(f"| availability indicator | {W1['indicator_floor']:.4f} |")
    for s_ in sorted(W1['donor']):
        v = W1['donor'][s_]
        w(f"| own minus donor, schedule {s_[1:]}{' (primary)' if s_ == 'd1' else ''} | {v['diff_mean']:+.4f} "
          f"[{v['diff_ci90'][0]:+.4f}, {v['diff_ci90'][1]:+.4f}] (90% interval) |")
    if 'pooled_donor' in W1:
        v = W1['pooled_donor']
        w(f"| own minus the mean of the donor arms | {v['mean']:+.4f} [{v['ci90'][0]:+.4f}, {v['ci90'][1]:+.4f}] |")
    w(f"| primary contrast inside the 0.02 margin | {'yes' if d1['equivalent_002'] else 'no'} |")
    w(f"| primary contrast inside the 0.05 margin | {'yes' if d1['equivalent_005'] else 'no'} |")
    w(f"| 90% interval of the primary contrast excludes zero | {'yes' if G['decision']['interval_excludes_zero'] else 'no'} |")
    _xq = _os.path.join(PATHS.ROOT, 'results', 'replication_posthoc.json')
    if _os.path.exists(_xq):
        XR = json.load(open(_xq))['replication']
        w("\nOne analysis outside the plan (`code/replication_posthoc.py`, `results/replication_posthoc.json`), added after the "
          "numbers above had been read: the own-minus-donor contrast of the printed numbers with no model.\n")
        w("| quantity (first wording) | schedule 1 | schedule 2 | schedule 3 |\n|---|---|---|---|")
        def _c(v): return f"{v['mean']:+.4f} [{v['ci90'][0]:+.4f}, {v['ci90'][1]:+.4f}]"
        w("| language model | " + " | ".join(_c(XR['language_model'][s_]['diff']) for s_ in ('d1', 'd2', 'd3')) + " |")
        w("| printed percentile, no model | " + " | ".join(_c(XR['printed_percentile'][s_]['diff']) for s_ in ('d1', 'd2', 'd3')) + " |")
        w("| printed value, no model | " + " | ".join(_c(XR['printed_value'][s_]['diff']) for s_ in ('d1', 'd2', 'd3')) + " |")
        w("| model minus printed percentile | " + " | ".join(_c(XR['model_minus_printed_percentile'][s_]) for s_ in ('d1', 'd2', 'd3')) + " |")
        w("| correlation across patients, model with printed percentile | "
          + " | ".join(f"{XR['association'][s_]['model_with_printed_percentile']['pearson']:.3f}" for s_ in ('d1', 'd2', 'd3')) + " |")
    if 'wording2' in G:
        W2 = G['wording2']; e1 = W2['donor']['d1']; P = G['paraphrase']
        w(f"\nSecond wording: {W2['n_patients']} patients, {W2['n_calls']:,} calls.\n")
        w("| quantity | value |\n|---|---|")
        w(f"| own arm, AUROC | {e1['own_auroc']:.4f} |")
        w(f"| donor arm, AUROC | {e1['donor_auroc']:.4f} |")
        w(f"| own minus donor | {e1['diff_mean']:+.4f} [{e1['diff_ci90'][0]:+.4f}, {e1['diff_ci90'][1]:+.4f}] |")
        w(f"| mean absolute change of the own-arm answer between the wordings | {P['own_between_wordings']['mean_abs_diff']:.4f} |")
        w(f"| mean absolute change between the own and donor answers, first wording, same patients | {P['own_vs_donor_wording1']['mean_abs_diff']:.4f} |")
    w("")
# the same sample on the second and third backbones (REPLICATION_BACKBONES_PLAN_2026-11-13.md)
_bb = {}
for _t in ('b2', 'b3'):
    _f = _os.path.join(PATHS.ROOT, 'results', f'replication_frozen_{_t}.json'); _r = _os.path.join(PATHS.ROOT, 'results', f'replication_reference_{_t}.json')
    if _os.path.exists(_f) and _os.path.exists(_r): _bb[_t] = (json.load(open(_f)), json.load(open(_r))['replication'])
if len(_bb) == 2 and _os.path.exists(_gp):
    _M = json.load(open(_gp)); _XM = json.load(open(_os.path.join(PATHS.ROOT, 'results', 'replication_posthoc.json')))['replication']
    w("## The replication sample on the second and third backbones\n")
    w("Plan: `REPLICATION_BACKBONES_PLAN_2026-11-13.md`, committed before the first call to either backbone on the sample. Statistics: "
      "`code/replication_backbones.py` (`replication_stats.py` unchanged on each archive), written to `results/replication_frozen_b2.json` "
      "and `results/replication_frozen_b3.json`; the comparison with the printed percentile, which that plan names in advance, is in "
      "`results/replication_reference_b2.json` and `results/replication_reference_b3.json`. Each backbone is read separately and nothing "
      "is pooled. The first column repeats the main backbone, where the comparison with the printed percentile was added outside its plan.\n")
    w("| quantity (first wording, 100 patients) | main backbone | second backbone (" + _bb['b2'][0]['backbone'] + ") | third backbone (" + _bb['b3'][0]['backbone'] + ") |\n|---|---|---|---|")
    _cols = [(_M, _XM), _bb['b2'], _bb['b3']]
    def _c2(m, c): return f"{m:+.4f} [{c[0]:+.4f}, {c[1]:+.4f}]"
    def _row(lab, fn): w(f"| {lab} | " + " | ".join(fn(f_, r_) for f_, r_ in _cols) + " |")
    _row("own arm, AUROC", lambda f_, r_: f"{f_['wording1']['donor']['d1']['own_auroc']:.4f}")
    _row("donor arm, schedule 1, AUROC", lambda f_, r_: f"{f_['wording1']['donor']['d1']['donor_auroc']:.4f}")
    for _s in ('d1', 'd2', 'd3'):
        _row(f"own minus donor, schedule {_s[1:]}{' (primary)' if _s == 'd1' else ''}, 90% interval",
             lambda f_, r_, _s=_s: _c2(f_['wording1']['donor'][_s]['diff_mean'], f_['wording1']['donor'][_s]['diff_ci90']))
    _row("primary contrast inside the 0.02 margin", lambda f_, r_: 'yes' if f_['decision']['equivalent_002'] else 'no')
    _row("90% interval of the primary contrast excludes zero", lambda f_, r_: 'yes' if f_['decision']['interval_excludes_zero'] else 'no')
    _row("model minus printed percentile, schedule 1", lambda f_, r_: _c2(r_['model_minus_printed_percentile']['d1']['mean'], r_['model_minus_printed_percentile']['d1']['ci90']))
    _row("own minus donor, second wording (40 patients)", lambda f_, r_: _c2(f_['wording2']['donor']['d1']['diff_mean'], f_['wording2']['donor']['d1']['diff_ci90']))
    w("")
w("## References\n")
w("Every reference in the manuscript was resolved against PubMed E-utilities, the Crossref API or the "
  "arXiv API before use; the procedure and its outcome are in `docs/VERIFICATION.md` at the repository root.")
open(PATHS.out('RESULTS.md'), 'w').write('\n'.join(L) + '\n')
print(f'wrote RESULTS.md ({len(chr(10).join(L))} chars)')
