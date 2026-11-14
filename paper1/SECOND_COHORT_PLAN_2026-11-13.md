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

### Clarification (13 November 2026)

The 1,095 tumors of the second cohort are the breast cohort of the preprint that accompanies this repository (https://doi.org/10.64898/2026.09.02.748315) and of its data record (`brca/` in the cohort archive), where an earlier run of the scorer was evaluated against labels thresholded from that cohort's own input. "Before any analysis script was run on the second cohort" at the head of this plan and "has not been trained on the second cohort" in Section 6 refer to the analyses of this plan: rows scored against the shared label vector, on features rebuilt by `code/build_second_cohort.py`. None of them had been computed on this cohort before the plan was committed.

### Record of the graph-scorer runs of Section 6 (13 and 14 November 2026)

The 30 runs of Section 6 started at 15:44 on 13 November 2026 and ended at 07:40 on 14 November, local time (UTC+7). Section 6 had been public since 03:05 on 13 November local time (20:05 UTC on 12 November; commit `36cb74db`, tag `benchmark-validity-2026-11c`), 12 hours and 40 minutes before the first run started. Four departures from it are recorded. None depends on a result.

| Departure | What happened |
|---|---|
| Workstation | The runs were made on a second workstation, because both accelerators of the first were in use by another service. The accelerator model (NVIDIA H100, 80 GB) and the versions of PyTorch (2.10.0, CUDA 12.8, cuDNN 9.10.2) and PyTorch Geometric (2.7.0) are the same; the Python interpreter is 3.12.3 in place of 3.10.12. Up to four runs shared one accelerator. |
| Code | The driver is the released `code/double_holdout_ctrl.py`, byte-identical to the file that ran the first cohort. The released `code/trainer.py` differs from the file that ran the first cohort in comments, docstrings and log messages only. |
| Priority | The jobs ran at a lowered scheduling priority, which enters no computation. |
| Check outside the plan | One cell of the first cohort (patient fold 0, reaction fold 0, own-expression arm, the released training seed) was rerun twice on the second workstation as a check of the environment. It scored 0.7745 and 0.7724, against 0.7903 in the released run, whose two training-seed replicates of that cell scored 0.7776 and 0.7854. These reruns enter no analysis. The contrast of Section 6 is taken between two arms trained on the same workstation. |

The record of the runs is `data/cohort2_graph_run_record.json`; the cells are in `results/cohort2/fam/`, and `code/second_cohort_graph.py` writes `data/results_cohort2_graph.json`.

### Outcome of Section 6

| Quantity | Second cohort | First cohort |
|---|---|---|
| Own-expression arm, AUROC, mean over cells | 0.7924 | 0.7952 |
| Cohort-mean arm, AUROC, mean over cells | 0.8062 | 0.8014 |
| Primary quantity: own minus cohort mean, mean over the 15 cells | -0.0138 | -0.0062 |
| Standard deviation over cells | 0.0153 | 0.0245 |
| Range over cells | -0.0477 to +0.0075 | -0.0515 to +0.0258 |
| Negative cells | 13 of 15 | 8 of 15 |

Reading fixed in Section 6: the mean of the second cohort lies within 0.02 AUROC of zero, which is read as agreement with the first cohort. Two things were not part of the reading and are reported as descriptions. The difference is negative in 13 of the 15 cells. An exact sign-flip test gives p = 0.0019 with the cells flipped singly (rounded up, as in the manuscript), and 0.0625 and 0.25 with whole patient folds and whole reaction folds flipped, the smallest values that five and three folds allow; the cells share patients, reactions and one label vector. The two fold-level values were added on 14 November, after the outcome had been read, so that the second cohort is described as the first is. Under the same stopping rule the cohort-mean arm trained for a median of 28 epochs (16 to 67) and the own-expression arm for a median of 16 (16 to 30); whether the longer training accounts for the difference between the arms was not tested.

### A check outside the plan: the training seed at one cell (14 November 2026)

Section 6 fixes one training seed. After its outcome was known, cell p0r0 of the second cohort was retrained at the two further training seeds of the first cohort's check (2025, 2026), own-expression arm and cohort-mean arm, with the folds and everything else as in Section 6 (`code/gpu_queue_cohort2_seed.sh`; four runs on the second workstation, 13:12 to 14:50 on 14 November, local time). The check is no part of the plan and enters none of its quantities: `code/second_cohort_graph.py` reads the default-seed files only. `code/second_cohort_seed.py`, written before any of the four runs had finished, reads the check and writes `data/results_cohort2_seed.json`.

| Training seed | Own-expression arm | Cohort-mean arm | Own minus cohort mean, second cohort | The same cell on the first cohort |
|---|---|---|---|---|
| 2024 (the seed of Section 6) | 0.7744 | 0.7994 | -0.0250 | +0.0258 |
| 2025 | 0.7902 | 0.7844 | +0.0058 | +0.0225 |
| 2026 | 0.7704 | 0.7884 | -0.0180 | +0.0138 |

At this cell the difference spans 0.0308 AUROC over the three seeds and changes sign; on the first cohort it spans 0.0120 and keeps one sign. One cell shows how far a single cell moves with the seed. It does not show how far the mean over the 15 cells would move, and the reading of Section 6 rests on that mean.

## 7. Planned extension: a second training seed on the full grid (fixed on 14 November 2026, not run)

Section 6 was run at one training seed, and the check above shows that the difference at a single cell can change sign with the seed. This section fixes a repetition of the whole grid at a second training seed before it is run. It was written after the outcome of Section 6 and the three values of the check were known. Of the 30 runs it names, the two of cell p0r0 already exist from that check (own minus cohort mean +0.0058 at that cell); the other 28 had not been started when this section was committed.

| Item | Rule |
|---|---|
| Driver | `code/double_holdout_ctrl.py`, unchanged, launched by `code/gpu_queue_cohort2_seed2025.sh`: everything as in Section 6, with the training seed 2025 (the seed of parameter initialization and dropout; the patient folds, the reaction folds and the inner-validation reactions stay as they are) |
| Arms | own expression and cohort-mean expression, every cell of the 5 x 3 grid: 30 runs, of which the two of cell p0r0 are reused from the check and 28 are new |
| Primary quantity | the paired difference over the 15 cells between the own-expression arm and the cohort-mean arm at the second seed, AUROC on held-out reactions of held-out patients |
| Reading | as in Section 6: the mean, the standard deviation over cells, the range and the number of negative cells; a mean within 0.02 AUROC of zero is read as agreement with the first cohort. No test is fixed; sign-flip p values with single cells, whole patient folds and whole reaction folds flipped are given as descriptions |
| Beside it | the same summary for the per-cell mean of the two seeds' differences; the difference between the two seeds' grid means; the mean absolute change of a cell's difference between the seeds; the number of cells whose difference keeps its sign; and the Pearson and Spearman correlations of the two seeds' differences across the 15 cells. Training length, which Section 6 left open: at each seed, the epochs of the cohort-mean arm minus those of the own-expression arm cell by cell, the correlation of the difference in AUROC with that gap across the cells, and the mean difference over the cells whose two arms stopped within two epochs of each other; for the seed of Section 6 these figures are computed after its outcome was known. All descriptive |
| Analysis | `code/second_cohort_graph_seed.py`, committed with this section; it writes `data/results_cohort2_graph_seed.json` |
| Reporting | the result is reported whatever it is, beside that of Section 6, with any deviation listed under a dated heading below |

The seed 2025 is the first of the two further seeds of the first cohort's check. A third seed is not part of this section.
