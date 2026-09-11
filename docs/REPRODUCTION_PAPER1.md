# Reproducing Paper 1

Paper 1 is "Auditing patient-specific metabolic reaction scoring with joint holdouts and
input-substitution controls". Its manuscript sources are withheld from this repository until
publication; everything the manuscript's numbers rest on is here, and the map at the end of
this guide gives, for every table and figure of the last built documents, the result files
behind it and the script that produced them.

## Three levels of reproduction

The word "reproduce" covers three different things here, and this guide keeps them apart.

1. **Regenerating the reported numbers from the committed results.** `bash reproduce.sh
   analysis` reads the per-cell result files under `paper1/results/`, aggregates them into
   `paper1/data/results_p1.json`, writes every quoted number as a LaTeX macro into
   `paper1/manuscript/numbers.tex`, draws the figures, rebuilds the ledger of result-file
   hashes and verifies it. No GPU, no network and no deposited data are needed, and the run
   takes a few minutes on a laptop. This is the level a public checkout reproduces.
2. **Recomputing the committed results from the raw predictions.** The per-cell scripts that
   turn prediction arrays into result files (`per_patient_auc.py`, `mask_scores_p1.py`,
   `fam_clean_scores.py`, `probe_preds.py`, `collapse_noise.py`, `arm_noise.py`,
   `strata_eval.py`, `phenotype_probe.py`, `phenotype_probe_multi.py`, `phenotype_null.py`)
   read the per-cell prediction arrays, which are released as assets of the tagged release
   rather than committed (2.7 GB; see "The prediction arrays" below), and the cohort files of
   the data deposit. For the external audit of a published predictor, the same level is
   `paper1/code/deepmeta_audit/metrics.py` run on the committed per-arm prediction CSVs
   under `paper1/results/deepmeta/preds/`, which needs the DepMap inputs that
   `fetch_inputs.sh` retrieves.
3. **Regenerating the predictions themselves.** The training cells need the GPU host and the
   deposited cohort, and a retrained cell is a new realization of the same protocol, not a
   bit-identical copy. The external audit's predictions need the authors' released checkpoint
   and the DepMap inputs, and regenerate deterministically up to float32 reduction-order noise
   (`paper1/code/deepmeta_audit/PROVENANCE.md`, deviation 6).

The `paper` mode of `reproduce.sh` typesets the manuscripts after level 1. It needs the
manuscript sources, which are not part of the public checkout, and says so and stops when they
are absent; a public checkout runs the analysis mode only.

## Environment

One `requirements.txt` at the repository root covers both papers. For Paper 1:

- Python 3.10.12
- NumPy 2.2.6, SciPy 1.15.3, scikit-learn 1.7.2, h5py 3.16.0 (pinned; a scikit-learn minor
  version can move a four-decimal logistic-regression AUROC)
- PyTorch 2.10.0 (CUDA 12.8 build) and PyTorch Geometric 2.7.0, needed only for the GPU
  experiment scripts; the analysis scripts that regenerate numbers, figures and the map run
  on CPU
- pandas, joblib and matplotlib as lower bounds only (used by the analysis and figure
  scripts; no quoted number depends on their exact version)

The external audit pipeline has its own environment note in
`paper1/code/deepmeta_audit/RUN_FULL.md` (the versions it was run under, and two settings that
are not optional).

The Genomic Data Commons manifest behind the cohort was retrieved 9 March 2026; the eleven
Human Metabolic Atlas reconstructions behind the activity label are cited to Robinson et al.
in the manuscript's reference list.

## The one command

```
bash reproduce.sh analysis
```

runs, for Paper 1 and in this order,

```
python3 paper1/code/split_audit.py
python3 paper1/code/build_families.py
python3 paper1/code/split_audit.py --family_map paper1/data/families/families_union.json \
    --out paper1/results/split_audit_fam.json --matched_out paper1/results/fam_matched.json
python3 paper1/code/inference_p1.py
python3 paper1/code/build_results_p1.py
python3 paper1/code/make_ledger.py
python3 paper1/code/make_numbers_p1.py
python3 paper1/code/make_figures_p1.py
python3 paper1/code/check_macros_p1.py
python3 paper1/code/verify_ledger.py
```

and then the Paper 2 chain, and finally `tools/check_release.py`. The ledger runs after the
last script that writes a hashed file (`build_results_p1.py` writes `data/results_p1.json`)
and before the macros that quote its counts, so nothing it hashes changes after it is written;
`verify_ledger.py` then checks every hash against the file it names and fails the run on a
mismatch, which is what catches a ledger committed beside newer results.

## The ledger and the release check

`paper1/results/LEDGER.json` records, for every result family, the script that produced it,
the settings the result files themselves carry, the checkpoint policy and endpoint, and the
SHA-256 of every result and data file. It also records the checksums of the external audit's
prediction CSVs and of the released prediction arrays (below). `python3
paper1/code/verify_ledger.py` checks it; `python3 tools/check_release.py` checks the
versioned required-artifact manifest in `tools/release_manifest.json` (every file and file
set a public checkout must carry, with counts) and, when the manuscript sources are present,
every file path the manuscripts name. A checkout without the sources gets the manifest check
and an explicit note that the path check was skipped, not a pass.

## The prediction arrays

Each training cell wrote one array beside its result file (`<cell>_preds.npz`: scores and
uncertainties for the cell's test patients over all reactions, the held-out and fit masks and,
for the synthetic cells, the per-patient labels). There are 299 of them, 2.7 GB in all, too
large to commit. They are attached to the tagged release named in `release.json` as one
tarball per result family (`preds_dh.tar`, `preds_ctrl.tar`, `preds_rewire.tar`,
`preds_fam.tar`, `preds_ivr.tar`, `preds_synth.tar`, `preds_ht29.tar`, `preds_seedrep.tar`,
`preds_rankgnn.tar`, `preds_permute.tar`), and their checksums are committed:
`paper1/results/prediction_arrays.sha256` lists every array and
`paper1/results/prediction_array_tarballs.sha256` every tarball. To use them, download the
tarballs, check them against the tarball list, unpack them under
`paper1/results/prediction_arrays/` (so that, for instance,
`paper1/results/prediction_arrays/fam/gnn_B_mb0.0_p0_r0_ivr_fam_preds.npz` exists) and run
`verify_ledger.py`, which checks every array it finds there against the array list. Every
array was checked, before release, against the result file written beside it: the mean
per-patient held-out AUROC recomputed from the array equals the figure the result file reports
to the four decimals it carries, for all 299.

The scripts of level 2 read the arrays from the directory named by `P1_OUTD` (default
`~/metagnn/out`); point it at `paper1/results/prediction_arrays` after unpacking.

## The external audit of a published predictor

`paper1/code/deepmeta_audit/` is the pipeline behind the manuscript's external audit, as it
ran: `manifest.py` (eligible lines, the held-out set, the gene panel, the gene families and
the input checksums), `arms.py` (donor schedules), `build_arms.py` and `run_arms.py` (the
inputs and predictions of the six arms), `native_repro.py` (the authors' own benchmark
reproduced on the current inputs), `metrics.py` (every statistic) and `fetch_inputs.sh`
(the retrieval recipe for the two pinned repositories, the DepMap 24Q4 files and the
checkpoint, each verified by md5). `run_donor_arms.sh` is the launcher that produced the donor
arms the results carry, kept unedited; `RUN_FULL.md` gives the stage-by-stage commands, costs
and restart behavior, and `PROVENANCE.md` every fixed choice, every deviation from the
specification, and the two corrections made after the first run (the held-out population and
the donor constraint), with the superseded records kept beside the current ones.

Its outputs are committed under `paper1/results/deepmeta/`: `manifest.json`,
`schedules.json`, `native_repro.json`, `audit_results.json` and the two schedule-sensitivity
files, the superseded `manifest_138.json`, `schedules_138.json` and
`schedules_unconstrained.json`, `recompute_cells.json`, and the per-arm predictions themselves
under `preds/seed11`, `preds/seed22` and `preds/seed33` (one CSV per arm: cell, gene, raw
prediction). `metrics.py` recomputes the audit_results files from those CSVs and the DepMap
gene-effect and expression files (level 2); `run_arms.py` recomputes the CSVs from the
checkpoint (level 3).

## What needs the GPU host or the deposited cohort

- **The training cells.** `double_holdout.py`, `double_holdout_ctrl.py` and
  `double_holdout_rewire.py`, with `naive_baselines_allfolds.py`, `pooled_baselines.py`,
  `structure_floor.py` and the alignment and label builders `build_aligned_recon3d.py` and
  `build_labels_ht29.py`, are the experiment tier of `paper1/code/README.md`; every per-cell
  result file is their output, and they need the GPU host and the deposited cohort.
- **The deposited-only data files** named in the manuscript's methods: `activity_pseudolabels.pt`
  (the label input before alignment), `clinical_metadata.tsv` and `11models.mat` (the eleven raw
  reconstruction files the label union is built from). These are part of the Zenodo data deposit
  referenced in `STUDIES.md`, not this repository.
- **The prediction arrays**, released as tagged-release assets as described above, are needed
  for level 2 and not for level 1.

## Table and figure map

<!-- map:start -->
_Generated by `tools/make_repro_map.py` from the documents built on 2026-09-11; the manuscript sources are not in the public checkout, so this is the last built numbering._

### Main text

| Label (number in the built document) | Caption lead | Input result files (relative to `paper1/`) | Produced by |
|---|---|---|---|
| `fig:family` (Figure 1) | The family-disjoint reaction split. | data/families/families_union.json, results/fam/*.json, results/fam_clean_scores.json | build_families.py, double_holdout_ctrl.py, fam_clean_scores.py; drawn by make_figures_p1.py |
| `tab:ladder` (Table 1) | The decomposition. | results/*_mean.json, results/*_zero.json, results/ivr/*.json, results/naive_allfolds.json, results/naive_pooled.json | double_holdout_ctrl.py, naive_baselines_allfolds.py, pooled_baselines.py |
| `fig:personalization` (Figure 2) | <estNWordCap> related estimates of the individual-versus-cohort-mean contrast. | the substitution ladder (results/ivr/*.json, results/fam/*.json), results/naive_allfolds*.json, results/naive_pooled*.json, results/inference.json and the robustness cells | double_holdout_ctrl.py, naive_baselines_allfolds.py, pooled_baselines.py, inference_p1.py; drawn by make_figures_p1.py |
| `tab:deepmeta` (Table 2) | The same predictions read two ways. | results/deepmeta/audit_results.json, audit_results_seed22.json, audit_results_seed33.json, schedules.json, manifest.json | code/deepmeta_audit/metrics.py on results/deepmeta/preds/seed*/*.csv (run_donor_arms.sh is the launcher that ran) |

### Supplementary material

| Label (number in the built document) | Caption lead | Input result files (relative to `paper1/`) | Produced by |
|---|---|---|---|
| `fig:provenance` (Figure S1) | Where the inputs and the label come from. | data/labels_ht29.json, results/cohort.json | build_labels_ht29.py, cohort_stats.py; drawn by make_figures_p1.py |
| `tab:label` (Table S1) | The eleven reconstructions behind the label. | data/labels_ht29.json | build_labels_ht29.py |
| `fig:memorization` (Figure S2) | What the standard protocol rewards. | the main grid, as tab:main | double_holdout.py; drawn by make_figures_p1.py |
| `tab:strata` (Table S2) | Where the signal lives. | results/strata.json | strata_eval.py |
| `fig:ladder` (Figure S3) | The decomposition. | as tab:ladder | drawn by make_figures_p1.py |
| `tab:phenotypes` (Table S3) | Four patient attributes the model never saw. | results/phenotype_probe.json, results/phenotype_probe_multi.json, results/phenotype_null_*.json | phenotype_probe.py, phenotype_probe_multi.py, phenotype_null.py |
| `fig:phenotype` (Figure S4) | What the benchmark cannot see. | results/phenotype_probe.json, results/phenotype_null_*.json | phenotype_probe.py, phenotype_null.py; drawn by make_figures_p1.py |
| `tab:masks` (Table S4) | Scores on the three reaction masks, both selection rules. | results/mask_scores.json | mask_scores_p1.py, on the released prediction arrays |
| `tab:robust` (Table S5) | The individual-versus-cohort-mean contrast under the upstream changes tried. | results/ht29/*.json, results/seedrep/*.json, results/rankgnn/*.json, results/permute/*.json, results/naive_allfolds_ht29.json, results/naive_allfolds_rank.json | double_holdout_ctrl.py under each change, naive_baselines_allfolds.py |
| `tab:family` (Table S6) | The five arms on both reaction splits. | results/fam/*.json, results/naive_allfolds_fam.json, results/naive_pooled_fam.json, results/split_audit_fam.json, results/fam_matched.json, results/fam_clean_scores.json | double_holdout_ctrl.py, naive_baselines_allfolds.py, pooled_baselines.py, split_audit.py, fam_clean_scores.py |
| `tab:dsim` (Table S7) | The audit on the <dsimNTest> test worlds, by regime. | results/diagnostic_sim.json | diagnostic_sim.py |
| `tab:domains` (Table S8) | The donor-effect readings, domain by domain. | results/deepmeta/audit_results.json | code/deepmeta_audit/metrics.py |
| `tab:settings` (Table S9) | Target and deployment settings. | none: a conceptual table written into the methods | not generated |
| `tab:practice` (Table S10) | What is held out in representative studies of reaction- and gene-level metabolic scoring. | results/cohort.json, results/param_counts.json | cohort_stats.py, count_params.py |
| `tab:main` (Table S11) | The same protocol, two axes. | results/<model>_mb<lambda>_p<pf>_r<rf>.json (the main grid) | double_holdout.py |
| `tab:collapse` (Table S12) | Collapsing predictions to their cohort mean. | results/collapse.json, results/collapse_noise.json, results/arm_noise.json | compare_ctrl.py, collapse_noise.py, arm_noise.py |
| `tab:naive` (Table S13) | Naive predictors on held-out reactions. | results/naive_baselines.json, results/naive_baselines2.json | naive_baselines.py, naive_baselines2.py |
| `tab:synth` (Table S14) | The audit on a synthetic patient-varying target. | results/synth/*.json | double_holdout_ctrl.py with the synthetic target options |
<!-- map:end -->
