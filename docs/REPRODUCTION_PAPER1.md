# Reproducing Paper 1

Paper 1 is "Holding out reactions as well as patients changes what transcriptome-based
metabolic reaction scoring benchmarks measure". Its manuscript sources are withheld until
publication; the table and figure numbers below are the manuscript's, and everything they
rest on is here.

Start at `reproduce.sh`, which runs the whole chain from the committed per-cell outputs and
ends by writing `paper1/manuscript/numbers.tex`, the macro file that carries every number
the paper quotes. The sections below say which script produced which of those numbers, in
case you want to rerun one of them alone. Two things cannot be regenerated from this
repository: the training cells themselves, which need the GPU host, and anything that reads
the patient cohort, which is deposited separately.

Labels and captions below are read from the current `.tex` sources, and table and figure
numbers are the ones the current source produces in document order. The committed PDF and
`.aux` file are rebuilt from that same source, so their numbers should agree; if the two
ever diverge, the source order given here is the current one.

## Environment

One `requirements.txt` at the repository root covers both papers. For Paper 1:

- Python 3.10.12
- NumPy 2.2.6, SciPy 1.15.3, scikit-learn 1.7.2, h5py 3.16.0 (pinned; a scikit-learn minor
  version can move a four-decimal logistic-regression AUROC)
- PyTorch 2.10.0 (CUDA 12.8 build) and PyTorch Geometric 2.7.0, needed only for the GPU
  experiment scripts; the analysis scripts that regenerate numbers, figures and this table
  run on CPU
- pandas, joblib and matplotlib as lower bounds only (used by the analysis and figure
  scripts; no quoted number depends on their exact version)

The Genomic Data Commons manifest behind the cohort was retrieved 9 March 2026; the eleven
Human Metabolic Atlas reconstructions behind the activity label are cited to Robinson et al.
in the manuscript's reference list.

## The one command

```
bash reproduce.sh analysis
```

run from the repository root, regenerates `paper1/manuscript/numbers.tex` and all five
figures under `paper1/manuscript/figures/` from the per-cell result files already committed
under `paper1/results/`. It calls, in order:

```
python3 paper1/code/inference_p1.py
python3 paper1/code/build_results_p1.py
python3 paper1/code/make_numbers_p1.py
python3 paper1/code/make_figures_p1.py
python3 paper1/code/check_macros_p1.py
```

`bash reproduce.sh paper` additionally typesets both papers' PDFs with `pdflatex` (three
passes each). The last recorded end-to-end timing for `bash reproduce.sh paper` across both
papers, with no GPU and no network access, was 7 minutes 54 seconds; the analysis-only steps
above are a small fraction of that, dominated by loading and aggregating the per-cell JSON
files.

The `paper` mode writes three documents for paper1: `manuscript/paper1.pdf`, the combined
reading copy whose order is article, references, supplement; `manuscript/paper1_main.pdf`,
the article alone, which is the manuscript file a journal takes; and
`manuscript/paper1_supp.pdf`, the supplement alone, which is the supplementary file. The two
separate documents resolve their cross-references into each other through the `xr` package,
so section, table and figure numbers agree across all three; `tools/make_supp.py` generates
their wrappers and the supplement's own short reference list from the same bodies, so none of
the three can drift from the others.


**One script `paper1/code/README.md` lists is not on this call list.** `split_audit.py`
recomputes the near-duplicate held-out-reaction audit behind the split-audit macros and
writes `paper1/results/split_audit.json`; `reproduce.sh` does not invoke it, so those macros
are only regenerated if that file is already present (`build_results_p1.py` reads it if
found and leaves the corresponding macros undefined otherwise). The file is present in this
repository, so `bash reproduce.sh analysis` currently reproduces every split-audit number in
the text; run `python3 paper1/code/split_audit.py` by hand first if that file is ever
removed or needs to be regenerated.

### The path-existence check

```
python3 tools/check_release.py
```

scans both manuscripts for every file path named in a `\texttt{...}` or `\path{...}` span
and checks that it exists in the repository. As of this writing it reports every path found
except the deposited-data files listed below (which are named in the text but are not meant
to be in this repository) and one HuggingFace model identifier that contains a slash but is
not a filesystem path (`Qwen/Qwen3.8-27B`, in Paper 2's bibliography). See the end of this
document, and `docs/OVERLAP.md`, for the full current output.

## Table and figure map

Every row below is produced by the same pipeline: `build_results_p1.py` aggregates the
per-cell files under `paper1/results/` into `paper1/data/results_p1.json`,
`make_numbers_p1.py` turns that into `paper1/manuscript/numbers.tex`, and the manuscript's
tables are typeset directly from those macros (there is no separate generated file per
table). The five figures are the exception: each is its own script-drawn file. The "input
result files" column names the raw files a table or figure's numbers ultimately trace back
to, one level up from `results_p1.json`; the "produced by" column names the experiment
script that generated those raw files, for readers who want to regenerate the inputs
themselves (which needs the GPU host and the deposited cohort; see below).

The main text carries tables numbered 1 to 7 and figures numbered 1 to 4. Supplementary
material follows the main text, numbered S1 onward, and carries four more tables (S1 to
S4) and one more figure (S1).

### Main text

| Label (number) | Caption (lead) | Input result files | Produced by (experiment script) | Output | Macros file |
|---|---|---|---|---|---|
| `tab:practice` (Table 1) | What is held out in representative studies of reaction- and gene-level metabolic scoring | `results/cohort.json`, `results/param_counts.json`, and the manuscript's own description of cited prior work | `cohort_stats.py`, `count_params.py` | typeset from macros, no separate file | `numbers.tex` |
| `tab:settings` (Table 2) | Target and deployment settings | none; a conceptual table of the three ways to split this benchmark, written directly into `body_methods.tex` and not read from any result file | n/a, not generated | typeset directly in the manuscript body, no separate file | n/a |
| `tab:main` (Table 3) / `fig:memorization` (Fig. 1) | The same protocol, two axes / What the standard protocol rewards | `results/<model>_mb<lambda>_p<pfold>_r<rfold>.json` (the main grid, every architecture x patient-fold x reaction-fold cell) | `double_holdout.py` | table typeset from macros; figure at `manuscript/figures/fig_memorization.png` | `numbers.tex` |
| `tab:naive` (Table 4) | Naive predictors on held-out reactions | `results/naive_baselines.json`, `results/naive_baselines2.json` | the fold-0 naive-baseline suites (superseded for the all-fold rows by `naive_baselines_allfolds.py`, below) | typeset from macros | `numbers.tex` |
| `tab:ladder` (Table 5) / `fig:ladder` (Fig. 2) | The decomposition | `results/<model>_mb*_mean.json`, `results/<model>_mb*_zero.json` (input-substitution cells), `results/naive_allfolds.json` (reference rows), `results/ivr/*.json` (the same arms, and the indicator and permuted arms, selected on withheld reactions) | `double_holdout_ctrl.py --feature_mode {mean,zero,indicator,permute}`, `naive_baselines_allfolds.py` | table typeset from macros; figure at `manuscript/figures/fig_ladder.png` | `numbers.tex` |
| `tab:collapse` (Table 6) | Collapsing predictions to their cohort mean | `results/collapse.json`, `results/collapse_noise.json` | `probe_preds.py`, `collapse_noise.py`, reading the main grid's saved per-patient predictions and uncertainties | typeset from macros | `numbers.tex` |
| `fig:phenotype` (Fig. 3) | What the benchmark cannot see | `results/phenotype_probe.json`, `results/phenotype_probe_multi.json`, `results/phenotype_null_{msi,cin_vs_gs,sex,site}.json` | `phenotype_probe.py`, `phenotype_probe_multi.py`, `phenotype_null.py` | figure at `manuscript/figures/fig_phenotype.png` | `numbers.tex` |
| `fig:personalization` (Fig. 4) | Ten related estimates of the individual-versus-cohort-mean contrast | the substitution ladder, `naive_allfolds.json`, `naive_pooled.json` and `inference.json` together (this figure spans several of the sources above) | `double_holdout_ctrl.py`, `naive_baselines_allfolds.py`, `pooled_baselines.py`, `inference_p1.py` | `manuscript/figures/fig_personalization.png` | `numbers.tex` |
| `tab:synth` (Table 7) | The audit on a synthetic patient-varying target | `results/synth/*.json` (the graph model's unsubstituted and cohort-mean arms on the fold-0 cells, with all or half of the expression-bearing reactions given a patient-varying label, under both selection rules) | `double_holdout_ctrl.py --synth_target cohortmedian --synth_frac {1.0,0.5}` with and without `--inner_val_rxn 0.2` | typeset from macros (the section, its title and its closing sentence are guarded on the `syn*` macros and generated from the sign of the result) | `numbers.tex` |

| `sec:family` (Section 4.6) / `fig:family` (Fig. 3) | The same question with biochemically related reactions withheld together | `data/families/families_union.json` (the label-blind family map), `results/fam/*.json` (the five arms refitted on family-disjoint folds), `results/naive_allfolds_fam.json` and `results/naive_pooled_fam.json` (the linear rows on the same folds), `results/split_audit_fam.json` and `results/fam_matched.json` (the audit rerun against those folds), `results/fam_clean_scores.json` (the arms rescored with the audited residue dropped), `results/naive_allfolds_rerun.json` (the nearest-family lookup row and the reproduction check) | `build_families.py` (the map), `family_folds.py` (the folds), `double_holdout_ctrl.py --family_map ... --family_tag fam --inner_val_rxn 0.2`, `naive_baselines_allfolds.py` and `pooled_baselines.py` with `P1_FAMILY_MAP` set, `split_audit.py --family_map ... --matched_out ...`, `fam_clean_scores.py` | prose and figure typeset from macros, all behind `\ifdefined\famPat` | `numbers.tex` |

### Supplementary material

| Label (number) | Caption (lead) | Input result files | Produced by (experiment script) | Output | Macros file |
|---|---|---|---|---|---|
| `tab:label` (Table S1) / `fig:provenance` (Fig. S1) | The eleven reconstructions behind the label / Where the inputs and the label come from | `data/labels_ht29.json` (both); `results/cohort.json` (figure only, for the network, gene and expression-mask counts) | `build_labels_ht29.py` (writes `labels_ht29.json`); `cohort_stats.py` (writes `cohort.json`); drawn by `make_figures_p1.py` | table typeset from one macro (`\lblRows`); figure at `manuscript/figures/fig_provenance.png` | `numbers.tex` |
| `tab:strata` (Table S2) | Where the signal lives | `results/strata.json` | `strata_eval.py`, reading the main grid's saved per-reaction predictions | typeset from macros | `numbers.tex` |
| `tab:phenotypes` (Table S3) | Four patient attributes the model never saw | `results/phenotype_probe.json`, `results/phenotype_probe_multi.json`, `results/phenotype_null_{msi,cin_vs_gs,sex,site}.json` | `phenotype_probe.py`, `phenotype_probe_multi.py`, `phenotype_null.py` | typeset from macros | `numbers.tex` |
| `tab:robust` (Table S4) | The individual-versus-cohort-mean contrast under the upstream changes tried | `results/ht29/*.json`, `results/seedrep/*.json`, `results/rankgnn/*.json`, `results/permute/*.json`, `results/naive_allfolds_ht29.json`, `results/naive_allfolds_rank.json` | `double_holdout_ctrl.py --feature_mode permute`, `double_holdout_ctrl.py --train_seed S`, `double_holdout_ctrl.py --expr_transform rank`, runs against the HT29-only label, `naive_baselines_allfolds.py` with `P1_EXPR_TRANSFORM=rank` or the HT29 label | typeset from macros (rows guarded by `\ifdefined` are silently omitted when their cells are absent; all are present in the current build, see below) | `numbers.tex` |

`inference_p1.py` also feeds the sign-flip and block-level significance figures quoted
throughout the Results text (the `inf*` macros), from `results/per_patient.json` and
(when present) `results/naive_pooled.json`; it is not tied to a single table or figure and
is listed once here rather than repeated in every row that cites a p-value.

| `supp:family` (Supp. Section) | The family-disjoint reaction split: construction, audit and two stability checks | same as `sec:family` | same as `sec:family` | prose and one table typeset from macros | `numbers.tex` |
| the prediction ledger | What produced every result file, under which seed and policy | every file under `results/` and `data/` | `make_ledger.py` | `results/LEDGER.json`; the counts it reports are quoted in the back matter | `numbers.tex` |

## What needs the GPU host or the deposited cohort

None of the following can be regenerated from this repository alone; each needs either the
GPU host the original runs used, or the cohort deposited on Zenodo (`activity_pseudolabels.pt`,
`clinical_metadata.tsv`, `11models.mat`, the raw expression and per-cell prediction arrays),
or both. `tools/check_release.py` confirms these three data file names are absent from the
repository, which is expected: they are deposited data, not released code.

- **The entire "Experiments" tier** of `paper1/code/README.md`: `double_holdout.py`,
  `double_holdout_ctrl.py`, `double_holdout_rewire.py`, `naive_baselines_allfolds.py`,
  `pooled_baselines.py`, `structure_floor.py`, `per_patient_auc.py`, `probe_preds.py`,
  `collapse_noise.py`, `strata_eval.py`, `phenotype_probe.py`, `phenotype_probe_multi.py`,
  `phenotype_null.py`, and the alignment/label builders `build_aligned_recon3d.py` and
  `build_labels_ht29.py`. Every result file in the table above is this tier's output; the
  Analysis tier only reads what these scripts already wrote.
- **The robustness cells are now complete.** `paper1/results/rankgnn/` and
  `paper1/results/seedrep/` each hold six files, a real and a mean-substituted run on every
  reaction fold of patient fold 0, behind the rank-normalization-replicate and
  training-seed-replicate rows of `tab:robust`; `paper1/results/ht29/` holds nine files, all
  three input conditions (real, mean, zero) on the same three reaction folds, behind its
  tissue-matched-label rows. `paper1/results/permute/` still holds the wrong-patient
  (individual-versus-cohort-mean consistency check) cells for fold 0 only, three files,
  unchanged. Every macro these feed (`\robSeedPat`, `\robHtGnnPat`, `\robRkGnnPat`,
  `\robIvrPat`, and the pooled-and-frozen variants) is defined in the committed
  `numbers.tex`, so every row of `tab:robust` is present in the current build.
  `build_results_p1.py` only ever reports the cells that exist, and every count of cells
  behind an average is itself quoted in the manuscript.
- **The deposited-only data files** named in `paper1/manuscript/body_methods.tex` and
  cited by `paper1/code/README.md`: `activity_pseudolabels.pt` (the HT29-only label input
  before alignment), `clinical_metadata.tsv` and `11models.mat` (the eleven raw Human
  Metabolic Atlas reconstruction files the label union is built from). These are part of the
  Zenodo data deposit referenced in `README.md`, not this repository.
- **The per-cell prediction arrays** (`*_preds.npz`, roughly 375 MB total per `README.md`)
  that back the per-patient AUROC and collapse computations are deposited alongside the
  cohort and are not committed here; `per_patient.json`, `collapse.json` and the other
  already-aggregated result files this repository does carry are what let `reproduce.sh
  analysis` regenerate every quoted number without them.

## The path-existence check, in full

Running `python3 tools/check_release.py` against this repository currently reports 8 paths
found and 4 missing, all four of them the deposited-data files above plus one non-path
HuggingFace identifier: `11models.mat`, `activity_pseudolabels.pt` and `clinical_metadata.tsv`
(all named in `paper1/manuscript/body_methods.tex`), and `Qwen/Qwen3.8-27B` (named in Paper
2's bibliography, `paper2/manuscript/body_bib.tex`, and not a repository path at all).
Everything else `\texttt{...}` or `\path{...}` names in either manuscript, including
cross-references into the other paper's `code/` and `data/` directories, resolves to a file
that exists in this repository today.
