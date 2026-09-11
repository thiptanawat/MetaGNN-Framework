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
w("# Paper 2 results, final\n")
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
      f"error {c['mae']:.2f}, predicted mean {c['pred_mean']} against a true mean of {c['truth_mean']}.\n")
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
w("## References\n")
w("Every reference in the manuscript was resolved against PubMed E-utilities, the Crossref API or the "
  "arXiv API before use; the procedure and its outcome are in `docs/VERIFICATION.md` at the repository root.")
open(PATHS.out('RESULTS.md'), 'w').write('\n'.join(L) + '\n')
print(f'wrote RESULTS.md ({len(chr(10).join(L))} chars)')
