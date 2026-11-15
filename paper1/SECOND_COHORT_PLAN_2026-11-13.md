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

### Record of the runs of Section 7 (14 and 15 November 2026)

Section 7 was public from 20:57 on 14 November 2026, local time (13:57 UTC; commit `9d3b8a8f`, tag `benchmark-validity-2026-11d`). The first of its 28 new runs was issued 6 minutes later, at 21:03, and the last cell was written at 07:02 on 15 November, local time. The two runs of cell p0r0 were reused from the check above, as the section says. The analysis script was run once, after all 30 cells existed. Workstation, code and scheduling priority are those recorded for Section 6. Two further points are recorded; neither depends on a result.

| Point | What happened |
|---|---|
| Launcher | The plan's launcher (`code/gpu_queue_cohort2_seed2025.sh`) issued the first sixteen jobs of its list on one accelerator of the second workstation. It was stopped at 02:26 on 15 November, while it waited for a free slot, and the runs it had started ran to their end. The last fourteen jobs were issued on the second accelerator of that workstation by a copy of the launcher (`code/gpu_queue_cohort2_seed2025_gpu0.sh`), from 00:47 on 15 November. The copy builds the same job list with the same job arguments, issues it from its end, and skips a job whose result file exists or that is already running. This shortened the wall-clock time and, from 02:34, left the first accelerator to the collections of the other study. No job was run twice. |
| One value seen before the cells were complete | At 04:17 on 15 November the last lines of the log of one finished run were printed to learn its run time. They hold the result record of that run: the held-out AUROC of the own-expression arm of cell p2r0 at the second seed. The cohort-mean run of that cell had not ended, so no difference was known. The analysis script and the reading of this section were public, and the sentences that report the outcome in the manuscript had been written before. No other value was read before all 30 cells existed. |

The record of the runs is `data/cohort2_graph_seed_run_record.json`, which lists every job with its launcher and its launch time. The cells are in `results/cohort2/fam/` (the files whose names carry `_ts2025`), and `code/second_cohort_graph_seed.py` writes `data/results_cohort2_graph_seed.json`.

### Outcome of Section 7

| Quantity (second cohort) | Second training seed (2025) | Training seed of Section 6 |
|---|---|---|
| Own-expression arm, AUROC, mean over cells | 0.7917 | 0.7924 |
| Cohort-mean arm, AUROC, mean over cells | 0.8050 | 0.8062 |
| Primary quantity: own minus cohort mean, mean over the 15 cells | -0.0133 | -0.0138 |
| Standard deviation over cells | 0.0228 | 0.0153 |
| Range over cells | -0.0551 to +0.0269 | -0.0477 to +0.0075 |
| Negative cells | 11 of 15 | 13 of 15 |
| Sign-flip p with cells, patient folds and reaction folds flipped | 0.04, 0.125, 0.25 | 0.0019, 0.0625, 0.25 |
| Epochs of the own-expression arm, median (range) | 16 (16 to 35) | 16 (16 to 30) |
| Epochs of the cohort-mean arm, median (range) | 22 (16 to 68) | 28 (16 to 67) |

Reading fixed in Section 7: at the second seed the mean lies within 0.02 AUROC of zero, which the plan reads as agreement with the first cohort. One of the 15 values, that of cell p0r0 (+0.0058), was known when the section was fixed.

Beside it, as descriptions. Averaged over the two seeds cell by cell, the difference is -0.0135 (standard deviation over cells 0.0164, range -0.0469 to +0.0101, negative in 12 of 15 cells). Between the seeds the mean over the grid changed by +0.0005, the difference of a single cell by 0.0161 on average and by at most 0.0384, and 11 of the 15 cells kept their sign; across the cells the two seeds' differences correlate at +0.46 (Pearson) and +0.39 (Spearman). Training length: at the second seed the cohort-mean arm trained 11.2 epochs longer than the own-expression arm on average (17.5 epochs longer at the seed of Section 6), the difference in AUROC correlates with that gap across the cells at -0.36 (Pearson; -0.25 at the seed of Section 6), and over the cells whose two arms stopped within two epochs of each other the mean difference is -0.0092 over 6 cells (-0.0165 over 4 cells at the seed of Section 6). The figures for the seed of Section 6 were computed after its outcome was known. Whether training length causes the difference between the arms was not tested.

## 8. Planned extension: the two arms at equal training length (fixed on 15 November 2026, not run)

Sections 6 and 7 trained both arms under one stopping rule, and under that rule the cohort-mean arm was trained longer than the own-expression arm: a median of 28 epochs against 16 at the training seed of Section 6 and 22 against 16 at the second seed. With a patience of 15 epochs, the scored checkpoint was that of epoch 13 against that of epoch 1 at the median (7 against 1 at the second seed). The descriptions fixed in Section 7 relate the difference in AUROC to that gap across the cells, but no run tests whether the longer training accounts for the difference between the arms. This section fixes such runs before they are made. It was written after the outcomes of Sections 6 and 7 and the training lengths of all their runs were known. None of the 60 runs it names had been started when this section was committed. Its driver had been run only in tests on a subset of 30 patients, which compared files with each other and whose scores were not read.

| Item | Rule |
|---|---|
| Driver | `code/double_holdout_len.py`, a copy of `code/double_holdout_ctrl.py` that differs from it in the training loop only: there is no early stopping and no checkpoint selection. A cell is trained for 13 epochs, and the test patients are scored on the held-out reactions after 1, 7 and 13 epochs with the weights of that epoch, in the way the other driver scores its selected checkpoint (Monte Carlo dropout with 30 passes, the same result fields). The weights and the states of the random number generators are saved before each scoring and restored after it, so that a scoring does not change the training that follows. The folds, the family-disjoint split, the inner-validation reactions, the model and the optimizer are those of Sections 6 and 7. The AUROC on the inner-validation reactions is still computed after every epoch and recorded; it selects nothing. `code/double_holdout_ctrl.py` is unchanged |
| Lengths | 1, 7 and 13 epochs, the epochs of the scored checkpoints at the median under the stopping rule: 1 for the own-expression arm at both seeds, 13 and 7 for the cohort-mean arm at the first and at the second seed. The reference length of a seed is that of its cohort-mean arm, 13 epochs at the seed 2024 and 7 at the seed 2025 |
| Arms and runs | own expression and cohort-mean expression, every cell of the 5 x 3 grid, at the training seeds 2024 and 2025: 60 runs, launched by `code/gpu_queue_cohort2_len.sh` with one call for each seed, on the second workstation of Sections 6 and 7 |
| Primary quantity | at each seed and each length, the paired difference over the 15 cells between the own-expression arm and the cohort-mean arm trained for the same number of epochs, AUROC on held-out reactions of held-out patients |
| Reading of the margin | as in Sections 6 and 7: the mean, the standard deviation over cells, the range and the number of negative cells; a mean within 0.02 AUROC of zero is read as agreement with the first cohort, and the number of the six means that lie within it is reported. No test is fixed; the sign-flip p values are given as descriptions |
| Reading of the training length | at each seed, if the mean difference at the reference length has the sign of the planned difference of that seed (-0.0138 at the seed 2024, -0.0133 at the seed 2025) and at least half its size, the longer training of the cohort-mean arm is read as not accounting for most of the planned difference at that seed; otherwise it is read as accounting for most of it. If the two seeds differ in this, the text says so. The rule describes 15 dependent cells and is given beside the means, not in place of them |
| Beside it | each arm's mean AUROC at each length; the share of the planned difference that each equal-length difference amounts to; the mean validation AUROC of each arm after every epoch. All descriptive |
| What these runs are not | repetitions of the runs of Sections 6 and 7. Training on this hardware is not reproduced bit for bit from one run to the next (the check recorded for Section 6), and a checkpoint taken at a fixed epoch is not one selected on validation, so the AUROC of an arm at its own median length need not equal that arm's value in Section 6 or 7. The comparison of this section is between the two arms within these runs |
| Analysis | `code/second_cohort_graph_length.py`, committed with this section; it writes `data/results_cohort2_graph_length.json` and stops if a cell file is missing |
| Order of reading | the analysis script is run once, when all 60 runs exist. If the runs of the second seed cannot be completed, the first seed is analysed alone (`--seed 2024`) and the text says so |
| Reporting | the result is reported whatever it is, beside those of Sections 6 and 7, with any deviation listed under a dated heading below |

A further length, a third seed and the first cohort are not part of this section. On the first cohort the two arms stopped after a median of 16 epochs each at the seed of the primary grid.

### Record of the runs of Section 8 (15 and 16 November 2026)

Section 8 was public from 15:28 on 15 November 2026, local time (08:28 UTC; commit `709dccdd` on the main branch). The second workstation read that commit from the public repository at 08:28:46 UTC, and the first of the 60 runs was launched at 08:29:07 UTC, 37 seconds after the push. The last result file was written at 03:24 on 16 November, local time. The launcher that the section names was called once for each seed, at the same time: `SEED=2024 GPU=1` on accelerator 1 of the second workstation; `SEED=2025 GPU=0` on accelerator 0 of the second workstation, with at most four runs at a time on each. No job was issued twice. Workstation, environment and scheduling priority are those recorded for Sections 6 and 7, and the driver, the launcher and the analysis script are the files committed with the section (their SHA-256 values are in the record).

Nothing was read before the runs were complete: no result file and no job log of the section was opened before all 180 result files existed, and progress was followed by counting files and processes. The analysis script was run once, when all 60 runs existed.

Deviations from Section 8: none.

The record of the runs is `data/cohort2_graph_length_run_record.json`, which lists every job with its launch time and the time at which its result files were written. The cells are in `results/cohort2/fam_len/` (60 runs, three result files each), and `code/second_cohort_graph_length.py` writes `data/results_cohort2_graph_length.json`.

### Outcome of Section 8

| Training seed | Epochs | Own-expression arm | Cohort-mean arm | Primary quantity: own minus cohort mean, mean over the 15 cells | Standard deviation over cells | Range over cells | Negative cells | Sign-flip p (cells, patient folds, reaction folds) | Share of the planned difference |
|---|---|---|---|---|---|---|---|---|---|
| 2024 | 1 | 0.7912 | 0.7961 | -0.0049 | 0.0187 | -0.0397 to +0.0208 | 8 of 15 | 0.3256, 0.3125, 1 | 0.36 |
| 2024 | 7 | 0.7810 | 0.8020 | -0.0210 | 0.0162 | -0.0524 to +0.0096 | 14 of 15 | 0.00025, 0.0625, 0.25 | 1.53 |
| 2024 | 13 (reference length) | 0.7814 | 0.8058 | -0.0244 | 0.0157 | -0.0516 to -0.0019 | 15 of 15 | 0.000062, 0.0625, 0.25 | 1.77 |
| 2025 | 1 | 0.7909 | 0.7895 | +0.0014 | 0.0223 | -0.0551 to +0.0325 | 6 of 15 | 0.8237, 0.75, 1 | -0.10 |
| 2025 | 7 (reference length) | 0.7826 | 0.8005 | -0.0179 | 0.0233 | -0.0684 to +0.0223 | 11 of 15 | 0.0098, 0.0625, 0.25 | 1.34 |
| 2025 | 13 | 0.7817 | 0.8022 | -0.0206 | 0.0161 | -0.0536 to +0.0014 | 14 of 15 | 0.00019, 0.0625, 0.25 | 1.55 |

The planned differences, under the stopping rule, are -0.0138 at the seed 2024 and -0.0133 at the seed 2025.

Reading of the margin fixed in Section 8: 3 of the 6 equal-length means lie within 0.02 AUROC of zero.

Reading of the training length fixed in Section 8: at both seeds the difference at the reference length has the sign of the planned difference and at least half its size, so the longer training of the cohort-mean arm is not read as accounting for most of the planned difference.

Beside it, as descriptions. The mean validation AUROC of each arm after every epoch, over the 15 cells; it selected nothing in these runs:

| Training seed, arm | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2024, own expression | 0.7677 | 0.7466 | 0.7428 | 0.7354 | 0.7382 | 0.7384 | 0.7378 | 0.7350 | 0.7385 | 0.7395 | 0.7384 | 0.7378 | 0.7381 |
| 2024, cohort-mean expression | 0.7638 | 0.7496 | 0.7562 | 0.7526 | 0.7517 | 0.7542 | 0.7549 | 0.7567 | 0.7518 | 0.7579 | 0.7547 | 0.7585 | 0.7614 |
| 2025, own expression | 0.7648 | 0.7440 | 0.7399 | 0.7431 | 0.7412 | 0.7413 | 0.7428 | 0.7403 | 0.7375 | 0.7391 | 0.7426 | 0.7411 | 0.7434 |
| 2025, cohort-mean expression | 0.7551 | 0.7540 | 0.7527 | 0.7502 | 0.7492 | 0.7553 | 0.7507 | 0.7518 | 0.7524 | 0.7548 | 0.7579 | 0.7570 | 0.7581 |

As the section says, these runs do not repeat those of Sections 6 and 7: at the seed 2024 the own-expression arm scores 0.7912 after 1 epoch here against 0.7924 under the stopping rule, and the cohort-mean arm 0.8058 after 13 epochs here against 0.8062; at the seed 2025 the own-expression arm scores 0.7909 after 1 epoch here against 0.7917 under the stopping rule, and the cohort-mean arm 0.8005 after 7 epochs here against 0.8050. The comparison of this section is between the two arms within these runs. Every figure describes 15 dependent cells at each seed, and no test was fixed.
