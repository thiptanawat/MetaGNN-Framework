# Analysis plan: the linear rungs of the decomposition on a second cohort

Date: 13 November 2026. This plan, the cohort builder and the summary script were committed to the public repository before any analysis script was run on the second cohort. Deviations, if any, are listed at the end under a dated heading.

## 1. Question

Every result of the study rests on one cohort. The decomposition has rungs that need no graph training: the ranked and fitted expression rows and the rows that add expression to the network features. This analysis asks whether those rungs show the same pattern on a second cohort, and in particular whether a patient's own expression adds anything over the cohort-mean expression there.

## 2. Cohort

| Item | Rule |
|---|---|
| Patients | the 1,095 patients of TCGA-BRCA with a primary-tumor STAR count file, one file per patient, as listed in the manifest of an earlier download; none is selected or removed by any outcome |
| Expression | built as the first cohort's was built: log2(1 + counts per million) of the unstranded counts, the library size taken over every gene of the count file; the patient filter of the first cohort's build (at least 5,000 detected genes and 1,000,000 reads) is applied |
| Genes and gene rules | the 21,263 genes of the first cohort's matrix and the resolved gene-protein-reaction groups of the first cohort (`data/gpr_resolved_cohort1.json`); no gene is selected on the second cohort |
| Check of the builder | `code/build_second_cohort.py --verify` rebuilds the first cohort's reaction features from its gene matrix; the largest absolute difference from the stored features is 0 |
| Label | the label vector of the study, unchanged; it is the same for every patient and does not depend on the cohort |
| Splits | five patient folds drawn by the same rule and seed on the second cohort's patients; the three reaction folds of the study (random) and its three family-disjoint reaction folds, unchanged |

`data/cohort2_expression_qc.json` records what the builder read and wrote. It was written before any analysis script ran.

## 3. Analyses

`code/naive_baselines_allfolds.py` and `code/pooled_baselines.py` are run on the second cohort without change, on the random reaction split and on the family-disjoint split, and `code/second_cohort_stats.py` summarizes them beside the first cohort's released results (`data/results_cohort2.json`). The tuned regime is reported, as in the decomposition table.

**Primary quantity.** On the family-disjoint split of the second cohort, the paired difference over the 15 cells between the row "network features + own expression" and the row "network features + cohort-mean expression".

**Secondary quantities.** The same contrast on the random split; the contrast between the fitted own-expression row and the fitted cohort-mean row, and between the two ranked rows, on both splits; the level of every linear row on the second cohort beside the first.

**Reading.** Cells share patients and reactions and are not independent. The mean, the standard deviation over cells, the range, the number of negative cells and the sign-flip p value over cells are reported as descriptive figures. A contrast inside 0.02 AUROC of zero is read as agreement with the first cohort, where the three linear contrasts lie between -0.019 and +0.004. The result is reported whatever it is.

## 4. What this analysis does not do

The graph attention scorer is not retrained on the second cohort: its cells need a graphics processor for about one hour each, and none was available. The rows that take no patient input (topology, annotation, network features) do not depend on the cohort and are recomputed only as a check that the two runs share labels and reaction folds.

## 5. Deviations

None at the time of the commit.

### Record after the analysis (13 November 2026)

No deviation. The first analysis script started 24 seconds after the plan was committed and 21 seconds after the commit was pushed (`data/cohort2_plan_record.json`; the push is in the public activity record of the repository). The builder kept all 1,095 patients, since the patient filter removed none, and found every one of the 21,263 genes in the count files. The patients are listed in `data/cohort2_patients.tsv`, in the order the analysis read them; their count files are the STAR gene-count files of the Genomic Data Commons, as `pipelines/brca/01_download_tcga_brca.py` retrieves them. The four result files are `results/cohort2/naive_allfolds.json`, `results/cohort2/naive_pooled.json` and their `_fam` counterparts, and the summary is `data/results_cohort2.json`.

| Quantity (second cohort) | Value |
|---|---|
| Primary: network + own minus network + cohort mean, family-disjoint split | +0.0014 (standard deviation over 15 cells 0.0052, range -0.0041 to +0.0093, negative in 10 cells) |
| The same contrast, random split | -0.0003 (0.0005) |
| Fitted own-expression row minus fitted cohort-mean row | -0.0015 (random), -0.0091 (family-disjoint) |
| Ranked own-expression row minus ranked cohort-mean row | +0.0007 (random), +0.0007 (family-disjoint) |
| Rows without patient input, largest difference from the first cohort | 0 |

All six contrasts lie within 0.02 AUROC of zero, which Section 3 reads as agreement with the first cohort.

## 6. Planned extension: the graph scorer on the second cohort (fixed on 13 November 2026, not run)

The graph attention scorer has not been trained on the second cohort. This section fixes the analysis before any such run. It was written after the linear results of Section 5 were known and before any graph-model result on the second cohort existed.

| Item | Rule |
|---|---|
| Driver | `code/double_holdout_ctrl.py`, unchanged, launched by `code/gpu_queue_cohort2.sh`: the family-disjoint reaction split, the matched selection rule (20% of each cell's training reactions withheld from the loss and used for stopping), the default training seed, every cell of the 5 x 3 grid |
| Arms | own expression and cohort-mean expression (feature modes `real` and `mean`), 30 runs |
| Primary quantity | the paired difference over the 15 cells between the own-expression arm and the cohort-mean arm, AUROC on held-out reactions of held-out patients, as on the first cohort, where it is -0.0062 (standard deviation over cells 0.0245) |
| Reading | descriptive, as for the first cohort: the mean, the standard deviation over cells, the range and the number of negative cells. A mean within 0.02 AUROC of zero is read as agreement with the first cohort |
| Reporting | the result is reported whatever it is, with any deviation listed under a dated heading below |
