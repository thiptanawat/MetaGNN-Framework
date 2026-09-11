# Benchmark validity in metabolic reaction activity scoring

Code and results for two studies of what transcriptome-based metabolic reaction activity
benchmarks actually measure, plus `metabench`, a small package that implements the five
checks as reusable functions. The manuscripts themselves are not here; they are under
review, and this repository will carry them once they are published.

The scorer the first study audits is in the same repository, unchanged, under `framework/`
with its per-cancer pipelines under `pipelines/`. Nothing here reads it; these studies read
the saved predictions it produced.

Everything regenerates from the raw results committed here. Every result from either
study's own runs is emitted as a LaTeX macro from a results file that is itself derived
from the per-cell outputs by a committed script, so re-running the analysis updates the
papers and a result with no run behind it fails the build. Fixed design constants and
descriptions of the input data are written as stated.

## The problem, in one paragraph

Methods that decide which reactions of a reference network are active in a sample are
benchmarked against activity labels that are **shared across every sample** in a cohort,
and cross-validated by holding out **samples**. If the target does not vary between
samples, holding out samples hides no labels, and a model carrying a free 64-dimensional
embedding per reaction can reproduce the target exactly without ever reading a sample. We
show one that does: AUROC 1.0000 on 624 colorectal tumors, using no patient input.
Holding out reactions as well collapses it to 0.5189, which is 5.04 standard errors above
chance and far below every naive baseline in the table.

<!-- generated:start -->
## What we found (Paper 1)

Held-out-reaction AUROC on the 15 cells of the substitution basis (5 patient folds x 3 reaction folds), every row on the same cells, linear rows with the regularization tuned by inner cross-validation; the rewiring row is 9 cells over patient folds 0, 1, 2.

| predictor | AUROC | uses |
|---|---|---|
| patient-invariant availability baseline | 0.6085 | whether a reaction receives any expression |
| each patient's own expression, ranked | 0.6342 | the transcriptome, no model |
| cohort-mean expression, ranked | 0.6340 | one average transcriptome |
| each patient's own expression, fitted | 0.6385 | the transcriptome, fitted |
| cohort-mean expression, fitted | 0.6469 | the same, fitted, **no patient identity** |
| topology only, fitted (6 columns) | 0.7401 | degrees, metabolite count, single-metabolite flag; **no patient data** |
| annotation only, fitted (63 columns) | 0.8097 | gene rule, gene count, reversibility, subsystem; **no patient data** |
| network features, fitted (69 columns) | 0.8544 | topology + annotation, **no patient data at all** |
| network features + cohort-mean expression | 0.8588 | + aggregate expression |
| network features + own expression | 0.8597 | + patient identity |
| graph model, degree-preserving rewiring | 0.6292 | a random network of the same degrees |
| graph model, expression zeroed | 0.7605 | topology, learned |
| graph model, presence pattern only (3 cells, fold 0) | 0.8363 | which reactions have an expressed gene, no magnitudes |
| graph model, cohort-mean expression | 0.8679 | topology + aggregate expression |
| graph model, patient's own expression | 0.8535 | everything |

A matched comparison leaves three things standing.

**The reference network predicts these labels far better than the transcriptome does, and a linear model on it comes close to the graph model.** With both sides fitted the same way on the same cells, the network features lead the fitted expression baseline by +0.2075 (topology alone +0.0932, annotation alone +0.1628), in every cell. Against the linear model given the same expression column, the graph model leads it by 0.0092 with one cohort-average vector, trails it by 0.0061 with each patient's own transcriptome, and trails it by 0.0939 with no expression at all. The no-patient linear model sits 0.0135 below the graph model with a cohort average and 0.0009 above the graph model with each patient's own data.

**The graph model's structural signal is the metabolic network itself, not generic graph statistics.** Rewiring the network at fixed degree, so every node keeps its exact connectivity in every relation and nothing else survives, moves the graph model by -0.2236 to 0.6292 over 9 paired cells (exact sign-flip p = 0.0040, its floor at that count), below the raw-expression ranking, while its fit to training reactions stays at 0.9988: given a network carrying no usable information the model falls back on memorizing, and the double hold-out catches it.

A second permutation seed (11) on 3 of those cells gives 0.6341, against 0.6294 for seed 7 on the same cells (seed 7 minus seed 11: -0.0047), and 0.2188 below the unrewired model on those cells.

**Most of what the cohort-mean column adds is the presence pattern.** Shown only which reactions have an expressed gene, the graph model reaches 0.8363 on the 3 cells of the presence-only arm, +0.0817 over the zeroed column and 0.0309 short of the cohort mean on the same cells: 73 % of the aggregate-expression step is the pattern, the rest the magnitudes.

**Individual expression is worth at most +0.0034 AUROC over a cohort average here, and at worst -0.0144.** We measured the effect of giving each patient their own transcriptome instead of one average vector ten ways on one cohort and one fold structure; 6 of the ten estimates are negative by more than a thousandth. The collapse rows are reported net of the part that averaging dropout noise alone produces; the graph model's retraining is reported under the grid's selection rule (-0.0144) and, as the paper's primary contrast, under early stopping on reactions withheld from the loss (-0.0086, negative in 12 of 15 cells).

| measurement | effect of patient identity | n | exact p |
|---|---|---|---|
| expression only, ranked, no model | +0.0002 | 15 | |
| expression only, linear | -0.0084 | 15 |  |
| network features + expression, linear | +0.0009 | 15 |  |
| feature model, cohort-mean collapse, net of noise | -0.0107 | 15 | 0.000062 |
| memorization control, collapse, net of noise | +0.0000 | 15 | 0.9020 |
| graph model, cohort-mean collapse, net of noise | -0.0040 | 15 | 0.000062 |
| expression only, linear, pooled and frozen | -0.0104 | 15 | |
| network features + expression, linear, pooled and frozen | +0.0034 | 15 | |
| graph model, input substitution, paired, grid's selection rule | -0.0144 | 15 across 5 patient folds | **0.000062** (patient-fold means: 0.00001) |
| graph model, input substitution, paired, selected on withheld reactions (primary) | -0.0086 | 15 across 5 patient folds | **0.0036** (patient-fold means: 0.0208) |

The linear rows are patient-invariant in their fitted part, so no cell-level p is given for them.

**Under the upstream choices tried, no estimate makes individual expression worth more than a few thousandths of AUROC over a cohort average.** Three choices upstream of every result were varied once each: the label (rebuilt from the HT29 reconstruction alone instead of the union of eleven), the expression normalization (within-patient percentiles instead of log2(TPM + 1)) and, for the graph model, the training seed, the early-stopping criterion and a wrong-patient consistency check. Linear rows are rerun on every cell; graph arms on the fold-0 cells.

| check | expression only, linear | network features + expression, linear | graph model, input substitution |
|---|---|---|---|
| HT29-only label | -0.0078 (n=15) | -0.0004 (n=15) | -0.0187 (n=3, fold 0) |
| within-patient percentile normalization | +0.0029 (n=15) | +0.0012 (n=15) | -0.0165 (n=3, fold 0) |
| second training seed | n/a | n/a | -0.0166 (n=3, fold 0) |
| wrong-patient arm, a consistency check under the shared label (own minus permuted column) | n/a | n/a | +0.0003 (n=3, fold 0) |

## What the benchmark cannot see

Every result above is measured against a label vector shared by all patients, which cannot reward patient specificity. Against a phenotype that does vary between patients, microsatellite instability (92 MSI-H of 579 annotated), the graph model's out-of-fold score vector classifies unstable against stable tumors at AUROC **0.938** (mean over three reaction folds), against 0.958 for the model's own input and 0.951 for the transcriptome before GPR mapping. The arms that show every patient an identical input sit at chance (0.502 and 0.520). The output carries most of the patient signal it is given; the benchmark, whose labels do not vary with the patient, scores that same output 0.0144 below the cohort-mean arm.

The same probe on three further attributes the model never saw. Output is the mean over reaction folds; the permutation p is that of the least favorable reaction fold (200 label permutations through the full nested procedure, floor 0.005); the cohort-mean arm sees an identical input for every patient.

| attribute | n | positives | input | output | perm. p | cohort-mean arm |
|---|---|---|---|---|---|---|
| microsatellite instability | 579 | 92 | 0.958 | 0.938 | 0.0050 | 0.502 |
| chromosomal instability vs genomically stable | 386 | 328 | 0.907 | 0.817 | 0.0050 | 0.465 |
| sex | 621 | 330 | 0.747 | 0.563 | 0.1244 | 0.482 |
| anatomical site, colon vs rectum | 624 | 458 | 0.795 | 0.589 | 0.0150 | 0.448 |

The output follows its input in rank order and on no attribute adds to it.

## Paper 2 in brief

A language model (qwen3-8-27b) asked to score the same reactions from each patient's transcriptome. Three prompt arms: patient-blind, the patient's own value, and a stranger's value in the same slot (a new donor for every reaction).

| design | reactions | patients | patient-blind | + own | + stranger's | raw expression, no model | availability baseline |
|---|---|---|---|---|---|---|---|
| A | 300 | 10 | 0.5188 | 0.5498 | 0.5470 | 0.6708 | 0.6239 |
| B | 240 | 8 | 0.4358 | 0.4481 | 0.4463 | 0.5115 | 0.5000 |
| E | 300 | 40 | 0.5186 | 0.5590 | 0.5531 | 0.6724 | 0.6239 |

No arm reaches the raw-expression ranking or the availability baseline in any design. The paired own-minus-stranger's difference at 40 patients is +0.0059 (90 % interval -0.0028 to +0.0146), equivalent to zero within 0.02 AUROC. Between-patient spread of the scores is 0.278 with the patients' own data and 0.283 with strangers'.

Three determinism floors (mean absolute change in the returned probability when nothing but the session changes): repeating the patient-blind prompt 0.0018; re-sending the patient prompt within a session 0.0019; answering Design A's patient prompts again in Design E, hours later, 0.0065 over 3,000 cells (0.0041 on the reactions where a value is shown). One expression value moves the answer by 0.353 on those reactions in Design A, about 86 times the largest floor.

Positive controls on a record matched in the placement of the number: echo 100.0 %, compare 98.3 %, threshold 100.0 %, scored against the printed percentiles. On microsatellite instability, a target that varies between patients, logistic regression on the same 40-reaction panel reaches 0.9241; the model reaches 0.5178 with the patient's own panel and 0.4889 with a stranger's (paired difference +0.0289, 95 % interval -0.0280 to +0.0866), returning one probability for 86.5 % of patients under 418 distinct rationales.
<!-- generated:end -->

## metabench: the five checks

```python
from metabench import (entity_folds, memorization_gap, indicator_floor,
                       structure_floor, cohort_mean_gain, dispersion_ratio)

folds = entity_folds(y, has_input, n_folds=3)          # hold out entities, not just samples
memorization_gap(scores, y, train_mask)                # is it memorizing identity?
indicator_floor(y, has_input)                          # what is free from the partition?
structure_floor(topology_features, y, folds)           # what is free from the network?
cohort_mean_gain(scores_real, scores_cohort_mean, y)   # does sample identity help or hurt?
dispersion_ratio(scores, mc_std, n_mc)                 # does the output read the input?
```

Each takes numpy arrays any pipeline already produces. `docs/PROTOCOL.md` explains how to
wire them into an existing evaluation and what each verdict means.

## Layout

```
metabench/            the five checks, framework-agnostic
paper1/code/          experiment drivers, the inference script, and every generator that feeds the manuscript
paper1/results/       one JSON per (model, lambda, patient fold, reaction fold), plus rollups
                      and per_patient.json, one held-out AUROC per patient per arm
paper1/data/          results_p1.json, derived from results/ by code/build_results_p1.py;
                      recon3d_aligned.json.gz, the reference network in the canonical order of the
                      training arrays (see below); and families/, the label-blind reaction families
                      the family-disjoint split rests on, rebuilt by code/build_families.py
paper1/manuscript/    numbers.tex, the generated macro file, once reproduce.sh has written it
paper2/code/          language-model probes, positive controls, patient-level target
paper2/results/       the archived model responses and per-run designs
paper2/manuscript/    numbers.tex, the generated macro file, once reproduce.sh has written it
docs/PROTOCOL.md      how to apply the checks to your own benchmark
docs/VERIFICATION.md  how every citation in both papers was verified
reproduce.sh          regenerates every number and figure from the committed results
framework/, pipelines/,  the scorer these studies audit, and its per-cancer pipelines,
  tests/, validators/    unchanged; see README.md
```

## One reaction order

The training arrays (stoichiometry, edges, expression features, gene table, labels) follow the
reaction order of the Recon3D MATLAB distribution. The BiGG JSON export lists the same reactions
in a different order under partly different identifiers, and must never be indexed by position
against those arrays. `paper1/code/build_aligned_recon3d.py` aligns the two by identifier and by
stoichiometric signature (10,311 of 10,600 reactions) and writes `recon3d_aligned.json`; every
script in both papers that needs annotation reads that table. Nothing else in the repository reads
the BiGG JSON.

## Reproducing

```bash
pip install -r requirements.txt
bash reproduce.sh analysis     # rebuild all statistics, macros and figures, no GPU, no network
```

`reproduce.sh` ends by writing `paper1/manuscript/numbers.tex` and
`paper2/manuscript/numbers.tex`. Each is one LaTeX macro per quoted result, and between them
they are every number the two papers report, in a form a reader can diff against a rerun.
The `paper` stage typesets the manuscripts and does nothing here until the sources are
added.

Re-running the experiments themselves needs the deposited cohort and a GPU; see
`paper1/code/README.md`. Serving is not bit-deterministic even at temperature zero, which
the language-model study measures directly, so a fresh collection of that study reproduces
reported quantities to within that noise rather than exactly.

## Data

Processed inputs are deposited under CC-BY-4.0 at
<https://doi.org/10.5281/zenodo.21217578>: the curated 624-patient TCGA-COAD/READ cohort,
GPR-mapped reaction features, reconstruction-derived activity labels, fixed splits and the
parsed Recon3D network. That is the record's concept identifier, so it always resolves to
the current version. The archived framework is at
<https://doi.org/10.5281/zenodo.21217568> under MIT, likewise a concept identifier.

Per-cell prediction arrays (`*_preds.npz`, 375 MB) are too large for this repository and
are deposited with the data record. Every quoted result is derived from the
committed JSON files, so `reproduce.sh` regenerates every number and figure without them.
What the arrays are needed for is recomputing those committed JSON files from scratch, which
is more than one table: ten scripts read them, and between them they produce `per_patient.json`
(the per-patient AUROC vectors that are the unit of every sign-flip and block-level test, so
every p-value in Paper 1 traces back to them), `strata.json`, the phenotype probes and their
permutation nulls, `mask_scores.json`, `arm_noise.json`, `collapse_noise.json` and
`fam_clean_scores.json`. A reader who wants to re-derive those rollups rather than take them
from the repository needs the deposited arrays; a reader who wants to check that the reported
numbers follow from the rollups does not.

## License

Code MIT. Deposited data CC-BY-4.0.
