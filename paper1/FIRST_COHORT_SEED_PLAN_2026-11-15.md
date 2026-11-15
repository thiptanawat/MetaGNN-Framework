# Plan: the graph scorer on the first cohort at a second training seed, on the full grid

Date: 15 November 2026. This plan, the launcher `code/gpu_queue_cohort1_seed2025.sh` and the analysis script `code/first_cohort_graph_seed.py` were fixed before any run of the plan was started, and the SHA-256 of the three files was recorded at that time. This file is not changed afterwards. Where and when the runs were made, whether the plan was public before them, and any deviation are recorded in `data/cohort1_graph_seed_run_record.json`.

## 1. Question

The primary contrast of the study is the paired difference, in the graph attention scorer, between each patient's own expression and the cohort-mean expression under the family-disjoint split and the matched selection rule. On the first cohort it was estimated at one training seed (2024): the own-expression arm scored 0.7952 and the cohort-mean arm 0.8014, a difference of -0.0062 over the 15 cells of the grid (standard deviation over cells 0.0245, range -0.0515 to +0.0258, negative in 8 cells, sign-flip p = 0.34).

A check at one cell had retrained cell p0r0 at two further training seeds and found differences of +0.0258, +0.0225 and +0.0138 across the three seeds, a spread of the order of the standard deviation over cells. How far the mean over the grid moves with the training seed is not known for this cohort. On the second cohort the whole grid is being repeated at a second training seed under Section 7 of `SECOND_COHORT_PLAN_2026-11-13.md`. This plan does the same on the first cohort, so that the contrast of each cohort is available at two training seeds.

## 2. What was known when this plan was fixed

The numbers of Section 1, the first seed's grid on the second cohort (-0.0138, standard deviation 0.0153, negative in 13 of 15 cells) and the one-cell checks of both cohorts were public. The runs of Section 7 of the second-cohort plan were under way; none of their results had been read, and they were not complete. No run of this plan had been started.

## 3. Runs

| Item | Rule |
|---|---|
| Driver | `code/double_holdout_ctrl.py`, unchanged, launched by `code/gpu_queue_cohort1_seed2025.sh`: everything as in the primary grid (`code/gpu_queue8.sh`: the family-disjoint split of the union family map, the matched selection rule with 20% of each cell's training reactions as inner validation, the graph attention scorer without the mass-balance term), with the training seed 2025 (the seed of parameter initialization and dropout; the patient folds, the reaction folds and the inner-validation reactions stay as they are) |
| Arms | own expression and cohort-mean expression, every cell of the 5 x 3 grid: 30 runs |
| Seed | 2025, the second seed of Section 7 of the second-cohort plan and the first of the two further seeds of the one-cell check |
| Where | the second workstation, all 30 runs. The two runs of cell p0r0 at this seed that the one-cell check made on the first workstation are not reused, so that both arms of every cell of this plan come from one workstation |
| Cells | `results/fam_seed2025/`, with the file names the driver gives them |

## 4. A confound that the plan cannot remove

The first seed's grid was run on the first workstation, whose accelerators are in use by another service and cannot be used for training now. The second seed's grid is run on the second workstation, which has the same accelerator model and the same versions of PyTorch, CUDA and PyTorch Geometric, under Python 3.12.3 in place of 3.10.12. Seed and workstation therefore change together. A check outside the plans reran the own-expression arm of cell p0r0 on the second workstation with the released seed: it scored 0.7745 and 0.7724, against 0.7903 in the released run, so that arm scored lower there by about 0.01, and two runs with one seed differed by 0.0021.

Within a seed, both arms of every cell are trained on one workstation, so the contrast at a seed does not mix workstations. The comparison between the two seeds does, and it is reported as a comparison between two seeds on two workstations.

## 5. Primary quantity and reading

The paired difference over the 15 cells between the own-expression arm and the cohort-mean arm at the second seed, AUROC on held-out reactions of held-out patients: its mean, its standard deviation over cells, its range and the number of negative cells. No test is fixed. Exact sign-flip p values with single cells, whole patient folds and whole reaction folds flipped are given as descriptions; the cells share patients, reactions and one label vector.

Reading: a mean within 0.02 AUROC of zero, the margin of the second-cohort plan, is read as agreement with the first seed, whose contrast the manuscript describes as small beside its spread across cells and not distinguishable from zero. A mean outside that margin is reported as a dependence of the primary contrast on the training seed or the workstation of that size, and the manuscript's statements about the contrast are then worded for both values.

The designated primary contrast of the manuscript stays the first seed's grid. The result of this plan is reported beside it in the main text, whatever it is.

## 6. Beside it, descriptive

- the same summary for the per-cell mean of the two seeds' differences;
- the difference between the two seeds' grid means, the mean and the largest absolute change of a cell's difference between the seeds, the number of cells whose difference keeps its sign, and the Pearson and Spearman correlations of the two seeds' differences across the 15 cells;
- training length at each seed: the epochs of the cohort-mean arm minus those of the own-expression arm cell by cell, the correlation of the difference in AUROC with that gap across the cells, and the mean difference over the cells whose two arms stopped within two epochs of each other;
- at cell p0r0 and the second seed, the two arms and their difference on the two workstations, from the files of the one-cell check and of this plan.

## 7. Analysis

`code/first_cohort_graph_seed.py` reads `results/fam/` (the first seed's grid and the one-cell check) and `results/fam_seed2025/` and writes `data/results_cohort1_graph_seed.json`. It stops if a cell of the second seed is missing, and it checks that the first seed's contrast equals the one already published in `data/results_cohort2_graph.json`.

No result of this plan is read before all 30 cells exist, and then only through that script.

## 8. What is not part of this plan

A third seed; the zeroed, presence-only and wrong-patient arms; the random reaction split; any change to the first seed's grid.
