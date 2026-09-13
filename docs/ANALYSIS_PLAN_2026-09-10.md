# Prospective analysis plan for the new evidence (frozen 10 September 2026)

This plan fixes, before any of the new experiments were run, the units, questions,
metrics, baselines, splits, missingness rules, multiplicity policy, equivalence margins,
uncertainty procedures and the wording reserved for negative outcomes. Everything
reported in the manuscripts before this date is development evidence and is described as
already explored. Any departure from this plan is recorded in the progress log of
`docs/ROADMAP_RESPONSE_2026-09-10.md` with its reason, and the departed-from rule is
still reported.

## 1. The Reaction-Scoring Audit

### Units and populations

- Patients: the 624 TCGA-COAD/READ tumors of the released cohort; fixed.
- Reactions: the 10,600 Recon3D reactions; the label is the reconstruction-membership
  proxy shared by every patient.
- Reaction families: label-blind groups built from stoichiometry and gene-protein-reaction
  (GPR) rules, defined in Section 1.3; the family is the resampling block for the
  family-disjoint evaluation.
- Cells: one patient fold crossed with one reaction (or reaction-family) fold. Cells of the
  same grid are not independent experiments; their dispersion is reported descriptively.
- Cell lines and genes (external audit): DepMap models joined by `ModelID`, grouped by
  `PatientID` for every split; genes restricted to a metabolic panel fixed on development
  data.

### The one primary question

Whether a sample's own inputs add predictive information beyond training-only per-gene
means for an independently published predictor on a measured, sample-varying endpoint
(the external audit, P1-E2). The family-disjoint evaluation (P1-E1) and the diagnostic
calibration (P1-E3) are the two supporting studies; the existing 624-patient benchmark
remains the motivating case.

### P1-E1, family-disjoint evaluation

- Family relations, each documented separately: (i) exact stoichiometric equivalence
  (identical signed metabolite vectors with compartments); (ii) compartment-stripped
  equivalence with substrate and product sides preserved (a transported metabolite is not
  cancelled across compartments); (iii) nonempty GPR equivalence (identical gene sets after
  normalizing the rule). Families are the connected components of the union. Component
  sizes are reported; if the largest component exceeds 5% of the reactions, the two
  prespecified alternatives are (a) drop relation (iii) for components in which it alone
  joins more than 50 reactions, and (b) split on (i) and (ii) only, and both schemes are
  reported. Components are never edited after any model result has been seen.
- Folds: three family folds stratified on family-level prevalence and expression
  availability; patient folds unchanged; inner-validation families (20% of the training
  families, seeded) lie entirely inside the outer-training families; the loss mask is
  identical for every arm of a cell.
- Arms: the graph model with own, cohort-mean, zeroed, indicator and permuted expression;
  the raw-expression ranking; the cohort-mean ranking; the pooled frozen linear model on
  annotation and topology; and a nearest-family lookup that returns the training label of
  the most similar training family (Jaccard on metabolite sets, ties by GPR overlap).
- Pilot: cell (p0, f0) with three training seeds for the own and cohort-mean arms; the
  seed standard deviation of the contrast is reported before the remaining cells run.
- Primary estimand: the paired own-minus-cohort-mean AUROC on held-out families under the
  matched (inner-validation) rule; secondary: the change in each model's AUROC and AUPRC
  from the random reaction split to the family split; performance by similarity to
  training (exact twin, stripped twin, GPR twin, none), by prevalence and by availability.
- Uncertainty: the exact sign-flip test over cells for the paired contrast; a family-level
  bootstrap within each cell for the AUROC drop; fold dispersion descriptive.
- Negative outcomes: if accuracy collapses on family-disjoint reactions, the result is
  reported as evidence that the random split rewarded near-twin recovery; if it survives,
  as family-level proxy generalization. Neither outcome is patient-flux validation.

**Outcome (recorded 2026-09-11, after the runs).** The union scheme's largest family holds 319
reactions, 3.01% of the network, so the prespecified 5% escape clause was not reached and the union
scheme is the only one carried forward. The pilot at (p0, r0) ran at three training seeds; the
contrast there takes +0.0258, +0.0225 and +0.0138, a spread of the same order as the cell-to-cell
standard deviation. The arms listed above all ran except that the raw-expression and cohort-mean
rankings are the ranked linear rows of the reference suite rather than separate scripts. The primary
estimand is -0.0032 (sign-flip p = 0.60, not distinguishable from zero); the secondary drop from the
random split is 0.050 to 0.078 AUROC for the graph arms and 0.021 for the no-patient linear floor.
The similarity-stratum breakdown is not reported, because the family-disjoint split leaves no
exact-twin or stripped-twin stratum to report; the audit rerun against the new folds (3.7% residual
under the gene-table rule, rescored away) covers the same question.

### P1-E2, external audit of a published predictor

- Model: DeepMeta as published, with its released checkpoint; the authors' benchmark is
  reproduced first and the training roster is recorded. Any refit is labeled adapted.
- Data: DepMap 24Q4 v1 (`CRISPRGeneEffect.csv`, `OmicsExpressionProteinCodingGenesTPMLogp1.csv`,
  `Model.csv`); the join key is `ModelID`; every split groups by `PatientID`. Complete-case
  counts are reported after the join.
- Gene panel: metabolic genes present in DeepMeta's output space and in the gene-effect
  matrix; genes whose gene-effect variance across development cell lines lies in the lowest
  decile are excluded on development data; the panel is fixed before any test score is
  computed.
- Split: cell lines are the test units; when the checkpoint's training roster is known,
  test lines are those not in it; otherwise a documented held-out subset grouped by
  `PatientID`. A lineage-held-out evaluation is a separate sensitivity.
- Interventions: own profile; training-cohort mean profile; within-lineage donor;
  cross-lineage donor. Every sample-dependent input is recomputed for the substituted
  profile; expression-only and graph-only swaps are reported separately when the model
  takes both.
- Primary metric: macro within-gene concordance across held-out cell lines, ties in the
  prediction scored one half, averaged equally over the fixed panel; larger is better.
  Donor effect = mean over genes of (C_own − C_donor) on identical gene-cell pairs.
  Secondary: per-gene Spearman correlation, the number of constant-prediction genes,
  per-cell-line ranking across genes, held-out squared error against the per-gene mean.
- Baselines: training-only per-gene mean; lineage-conditioned per-gene mean;
  expression-only ridge (and elastic net) fitted on development lines; a tuned tree
  regressor; a no-sample gene-annotation model.
- Smallest effect of interest for the donor effect: 0.01 concordance (one pair in a
  hundred re-ordered); a 95% interval within (−0.01, 0.01) is reported as "no
  sample-specific advantage larger than 0.01"; an interval spanning zero and 0.01 is
  "inconclusive".
- Uncertainty: paired bootstrap over cell lines (grouped by `PatientID`) and, separately,
  over gene families, rebuilding all pairs inside each resample; three fixed donor
  schedules, the primary conditioning on the first.
- Negative outcomes: if the native reproduction fails, the exact blocker is documented
  and the fallback (FlowGAT on its native data, entity/GPR split and baselines only) is
  run; the manuscript then states that no human sample-specific external audit was
  achieved.

### P1-E3, diagnostic calibration

- Worlds: independent synthetic datasets with a reaction term, a sample term and a
  controlled sample-by-reaction interaction; factors varied on a fixed grid: interaction
  strength (0, small, medium, large), assay noise (two levels), near-duplicate rate (two
  levels), label prevalence (two levels). Targets both derived from the measured
  expression and unrelated to it are generated and labeled as such.
- Predictors: sample-blind; oracle with the true sample-specific signal;
  expression-sensitive but label-irrelevant; restricted nonlinear ((x−1)^2 on the shared
  label); and a learned linear scorer with tuning inside training.
- Procedure under calibration: the full audit as applied in the manuscript (double
  hold-out, own versus cohort-mean versus donor contrast, sign-flip test over cells,
  collapse check), including donor selection.
- Development and lock: statistics and thresholds are developed on world set A (seeds
  1–40), then locked; false claims of personalization under the null, interval coverage
  and power are assessed on world set B (seeds 101–300 or more until the binomial
  standard error of the false-positive rate is below 2 percentage points).
- Output: a decision table mapping observed combinations (sign and interval of the
  own-minus-mean contrast; own-minus-donor; target variation) to permitted claims,
  preserving "inconclusive" and "unidentifiable" rows.

## 2. The Language-Model Evidence

### Units and populations

- Patients: TCGA-COAD/READ (development; 92 MSI-H and 487 MSS under the original
  endpoint, 44 MSI-L added under the harmonized endpoint); GSE39582 CIT tumors (external
  test; dMMR versus pMMR); GSE13294 (backup external test).
- Reactions: the fixed 40-reaction MSI panel of the development collections, restricted
  to reactions coverable on the external platform by the rule in Section 2.3.
- Sessions: repeated identical prompts within and across sessions quantify serving
  variation; they never add patients.

### The one primary confirmatory contrast

The own-minus-donor AUROC difference of the locked percentile-only Mistral Small 3.2 24B
interface on the external cohort, endpoint MSI-H (dMMR) versus non-MSI-H (pMMR), donors
from a uniform derangement independent of MSI status. Larger is better; the sign is
reported with a 95% patient-level interval built by the procedure in Section 2.5. Every
other collection (the factorial, the other backbones, the tool and calibration arms) is
secondary and is described as such; no confirmatory claim is made from a secondary
analysis.

### 2.1 Frozen interface

- Prompt: the MSI system prompt of the development collections with the raw expression
  field removed; each panel row shows the reaction identifier, the subsystem and the
  within-cohort percentile only; the availability statement is unchanged. The exact text
  is stored in `paper2/code/msi_probe.py` under `--format percentile_only` and archived
  with every collection.
- Decoding: temperature 0, `max_tokens` 160, plain chat, JSON-only instruction, retry
  budget three per prompt, full request and response archive, interleaved sending.
- Percentile: within-cohort midrank percentile of the reaction-level expression, 0–100,
  rounded to the nearest integer as in the development collections; ties share the
  midrank.
- The original raw-plus-percentile TCGA configuration is reported beside it as a separate,
  already explored configuration; it is never called a replay.

### 2.2 Endpoint harmonization

- Development: MSI-H versus non-MSI-H (MSS plus MSI-L) for the harmonized analysis; the
  original MSI-H versus MSS analysis is kept as a separate endpoint.
- External: dMMR versus pMMR as annotated in GSE39582; if per-locus calls are available,
  MSI-H versus non-MSI-H is used instead; the mapping table is reported in the main text.
- Expression-derived subtypes are never used as ground truth.

### 2.3 Assay transfer

- Probe-to-gene mapping: GPL570 annotation as released by GEO; probe sets mapped to Entrez
  identifiers, the per-gene value is the maximum over probe sets (frozen).
- Reaction-level expression: the GPR evaluation of the development pipeline (min over
  AND, max over OR) on the log2 array intensities; a reaction is retained when every gene
  its rule needs is measured; reactions that fail this on the external platform are
  removed from the panel in both cohorts before any external outcome is inspected.
- Percentiles are computed within each cohort separately (transductive, label-free); no
  joint normalization or batch correction across cohorts.
- Coverage and distribution shift (per-reaction percentile spread, share of ties) are
  reported for both cohorts.

### 2.4 Comparators

- Frozen supervised reference: L2 logistic regression on the panel percentiles fitted on
  all development patients, applied unchanged externally; its own-minus-donor difference
  under the same derangement.
- Development-selected single-feature rule: the panel reaction with the highest
  development AUROC, direction fixed on development data.
- Development-prevalence constant.
- Secondary backbones: Qwen3.8-27B and Gemma 4 31B-it under the same frozen interface.

### 2.5 Uncertainty and nulls

- Donor control: uniform random derangements of the patients, drawn independently of MSI
  status; five schedules (seeds 11, 22, 33, 44, 55); the primary conditions on seed 11;
  the spread over schedules is reported.
- Population interval: bootstrap over patients that, in each replicate, redraws the
  derangement among the resampled patients and recomputes the own and donor AUROC on the
  same recipients; 2,000 replicates; percentile interval.
- Null: 10,000 full profile-label alignment permutations (fixed points allowed) within
  site or platform blocks where those exist; the observed own-minus-donor difference is
  compared with this distribution.
- Serving variation: a 40-patient identical-prompt repeat within the session and in a
  later session, reported as the median absolute probability change.

### 2.6 Missingness and failures

- Failed calls after three attempts and unparseable replies are counted by arm and
  reported. The primary analysis uses paired evaluable patients; the sensitivity assigns
  every failed output the development prevalence and recomputes the contrast.

### 2.7 Factorial (P2-E1), secondary

- Conditions on the identical reaction record and availability statement: record only;
  availability only; raw only; percentile only; both. Identifier intervention on a
  prespecified 20% subset: authentic identifier and name versus neutral labels, equations
  and GPR preserved.
- Estimands: paired percentile effect on the returned probability given raw absent
  (percentile only minus availability only) and given raw present (both minus raw only);
  their difference is the interaction; per-reaction direction; the change in per-patient
  AUROC and AUPRC against the membership label.
- Panel size: chosen from a simulation of the crossed interval width using development
  effect sizes, before collection; no post hoc enlargement.
- Backbones: Mistral and Qwen; the key contrast on Gemma if time permits.

### 2.8 Tool and calibration arms (P2-E4, P2-E5), secondary

- Configurations on the frozen Mistral interface: zero-shot; evidence-assisted with a
  development-selected, frozen set of reaction definitions; tool-assisted in which a
  deterministic validator, the percentile transformation and the frozen logistic scorer
  return a probability that the model must report with an explanation. Tool-only is
  reported beside all three.
- Calibration: a logistic map on the logit of the returned probability, fitted on
  out-of-fold development predictions per configuration; a fitted slope at or below zero
  is reported as a constant map. Metrics: Brier, log loss with probabilities clipped to
  [0.005, 0.995], calibration slope and intercept, reliability curves with bootstrap
  bands, AUROC and AUPRC; references are the development prevalence and the logistic
  scorer.

### 2.9 Equivalence margins

The 0.02 and 0.05 AUROC margins belong to the reaction-proxy own-versus-stranger estimand
of the development designs and are not reused. No equivalence margin is declared for the
MSI contrasts; intervals are reported and a nonsignificant difference is never called
equivalence.

### 2.10 Negative outcomes

- External interval spanning zero: "inconclusive replication".
- Reversed sign with an interval excluding zero: "reversed on the external cohort".
- Every predictor including the frozen reference losing signal: "transport failure",
  investigated as assay, coverage or endpoint mismatch before any statement about the
  language model.
- Tool-assisted gain equal to the tool's own performance: the gain is attributed to the
  tool.

## 3. Multiplicity

One primary contrast per paper. Secondary analyses are labeled secondary wherever a
number is printed and are not adjusted; they are never described as confirmatory.

## 4. Reporting

Every figure caption names the endpoint, the biological unit, the data source, the
configuration, the sample counts, the interval procedure and the prediction manifest
identifier. Original-protocol and revised-protocol outputs are never mixed in a figure.
