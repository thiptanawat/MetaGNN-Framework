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
