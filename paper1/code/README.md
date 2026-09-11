# Paper 1 code

Two kinds of script live here. The first kind runs experiments and needs the deposited
cohort and a GPU. The second kind turns their outputs into the numbers and figures in the
manuscript, needs neither, and is what `reproduce.sh` calls.

## Experiments (GPU, deposited cohort required)

| script | what it does |
|---|---|
| `double_holdout.py` | the main grid: every (architecture, mass-balance weight, patient fold, reaction fold) cell, trained on train patients x train reactions and evaluated on test patients x held-out reactions with MC-dropout |
| `double_holdout_ctrl.py` | the same driver with `--feature_mode {real,mean,zero,indicator,permute}`, which substitutes the expression column with the training-cohort mean, with zeros, with the presence pattern of that mean (one where it is nonzero), or with another patient's real column (a seeded derangement within each split, the wrong-patient arm), while holding architecture, folds and seeds fixed; `--train_seed` and `--expr_transform rank` drive the seed and normalization checks |
| `double_holdout_rewire.py` | the same driver with `--rewire --rewire_seed S`: a degree-preserving rewiring of every relation (configuration model), everything else unchanged |
| `build_aligned_recon3d.py` | aligns the BiGG JSON export of Recon3D to the canonical reaction order of the training arrays and writes `recon3d_aligned.json`, which every annotation-reading script uses |
| `build_labels_ht29.py` | rebuilds the eleven-model union label from `11models.mat`, checks that it reproduces the deposited vector exactly, and writes the tissue-matched HT29-only label used by the robustness check of Section 3.8 |
| `arm_noise.py` | the per-score Monte Carlo error of each substitution arm, from the saved uncertainties |
| `naive_baselines_allfolds.py` | every linear reference point on every (patient fold, reaction fold) cell, with the network features split into topology and annotation and the regularization tuned by inner cross-validation (`naive_baselines.py` and `naive_baselines2.py` are the fold-0 suites that preceded it); average precision is written beside every AUROC under a parallel `auprc` key, with the held-out prevalence of every cell |
| `pooled_baselines.py` | the pooled, frozen counterpart of the two per-patient linear rows: one logistic regression per cell fitted on the stacked rows of a seeded subsample of training patients crossed with the training reactions, frozen, and scored on each test patient's own held-out rows (expression only, and network features plus expression); same environment variables, `--cells p0r0 ...` restricts a run, output `naive_pooled.json` |
| `structure_floor.py` | the standalone structure-only fit |
| `cohort_stats.py` | every cohort constant the manuscript quotes, computed from the data files |
| `count_params.py` | exact parameter counts, taken from the same construction path the experiments use |
| `per_patient_auc.py` | one held-out AUROC per (cell, arm, patient) from the saved predictions, the unit the inference script works on; the average precision of the same scores and the held-out prevalence are written beside it (`auprc`, `prevalence`) |
| `probe_preds.py` | reads the saved per-patient predictions and writes the prediction-collapse table |
| `collapse_noise.py` | the part of the collapse gain that averaging Monte Carlo dropout noise alone produces, from the saved uncertainties |
| `strata_eval.py` | held-out AUROC of every arm inside nine classes of reaction, from the saved predictions |
| `phenotype_probe.py`, `phenotype_probe_multi.py`, `phenotype_null.py` | the out-of-fold phenotype probes (microsatellite instability; chromosomal instability, sex, site) and their permutation nulls through the identical nested pipeline; the two probes now store the out-of-fold predictions of every fold (`oof`), from which `build_results_p1.py` forms an interval for the mean over reaction folds by resampling patients, in place of the envelope of the per-fold intervals |
| `trainer.py` | the model and dataset definitions, carried forward unchanged. The dataset key `'cohort_220'` is a historical name from an earlier cohort size and selects this project's cohort whatever its size; it does not mean 220 patients |
| `run_all.sh`, `run_ctrl.sh`, `run_ctrl2.sh`, `watchdog.sh`, `gpu_queue6.sh` | launchers, the supervisor that restarts a stalled grid, and the prioritized queue for the robustness cells (one GPU, at most five concurrent cells; the driver skips finished cells) |

Each cell writes its own result file and is skipped when that file is present, so the grid
is restart-safe and can be filled incrementally.

### The robustness checks

The two upstream variations are driven differently on the two sides of the ladder.

The linear rows are `naive_baselines_allfolds.py`, which reads four environment variables:

| variable | what it does |
|---|---|
| `P1_DATA` | the cohort directory (default `~/metagnn/work/crc_624`) |
| `P1_ALIGNED` | `recon3d_aligned.json` from `build_aligned_recon3d.py` |
| `P1_OUT` | the output file: `naive_allfolds.json` for the main grid, `naive_allfolds_ht29.json` and `naive_allfolds_rank.json` for the two checks, so a check is never pooled with the main grid (`naive_pooled.json` for `pooled_baselines.py`) |
| `P1_EXPR_TRANSFORM` | `none` (the main grid) or `rank`, which replaces every patient's expression column with its within-patient percentile among the reactions nonzero in that patient |

The graph arms are `double_holdout_ctrl.py`, which takes the same choices as flags: `--expr_transform
rank` for the second normalization, `--train_seed S` for a second training realization (output tag
`_ts<S>`), and `--feature_mode permute` for the wrong-patient arm, in which every patient is shown
another patient's real column through a derangement drawn separately within the train, validation and
test splits (output tag `_permute`). The tissue-matched label check needs no flag: point `--data` at a
cohort directory whose `activity_pseudolabels.pt` is the HT29-only vector that `build_labels_ht29.py`
writes. Write each check's cells to its own directory under `results/` (`ht29/`, `seedrep/`,
`rankgnn/`, `permute/`); `build_results_p1.py` picks each up automatically when the files are there
and leaves the corresponding macros undefined when they are not, which is what the `\ifdefined`
guards in the manuscript switch on.

## Analysis (no GPU, no network, no deposited data)

| script | what it does |
|---|---|
| `split_audit.py` | the grouped-split audit: for every reaction fold, how many held-out reactions share an exact stoichiometry, the same stoichiometry with compartments removed, or the same gene rule with at least one training reaction of that fold, with the label statistics of the matched reactions; runs on the aligned network and the label and availability vectors alone (the deposited cohort when present, `paper2/rxn_context.npz` otherwise) and writes `../results/split_audit.json` |
| `inference_p1.py` | the paired comparisons on the cell as the unit: exact sign-flip tests, block-level tests by patient fold and by reaction fold, fold-mean t-tests and a two-way decomposition, from `../results/per_patient.json`; the pooled rows of `naive_pooled.json` are paired against the per-patient and cohort-mean rows and the graph model when that file is present |
| `build_results_p1.py` | derives `../data/results_p1.json` from the per-cell files in `../results/`. Every average records its cell count and every comparison between feature modes is paired on (patient fold, reaction fold), so arms with different numbers of completed cells are never averaged against each other |
| `make_numbers_p1.py` | emits one LaTeX macro per quoted result into `../manuscript/numbers.tex` |
| `make_figures_p1.py` | draws the four figures from the same file the text reads |
| `check_macros_p1.py` | fails the build if the manuscript uses a macro the pipeline does not define, and if a sentence inside an `\ifdefined` guard reads a direction off an absolute value instead of a signed or `*Word` macro |
| `make_ledger.py` | writes `../results/LEDGER.json`: every result family with the script that produced it, the settings its files carry and a SHA-256 of every result and data file, plus the checksums of the external audit's prediction CSVs and of the released prediction arrays |
| `verify_ledger.py` | checks every hash the ledger records against the file it names, that every result file is listed, and any released prediction arrays present locally against their checksum list; exits nonzero on a mismatch |
| `build_families.py` | rebuilds the label-blind reaction families behind the family-disjoint split from the reference network alone |
| `diagnostic_sim.py` | the synthetic worlds in which the substitution audit's decision rule is fixed on development worlds and read on test worlds |

Run them in the order `reproduce.sh` gives, or run `bash ../../reproduce.sh analysis` from
the repository root, which does the same for both papers.

## The external audit of a published predictor

`deepmeta_audit/` holds the pipeline behind the manuscript's external audit as it ran, with
its own `RUN_FULL.md` (stages, costs, restart behavior) and `PROVENANCE.md` (inputs, fixed
choices, deviations, and the two corrections made after the first run). Its outputs are under
`../results/deepmeta/`, including the per-arm prediction CSVs that `deepmeta_audit/metrics.py`
scores. Its environment differs from the rest of this directory (`RUN_FULL.md`, section 0).
