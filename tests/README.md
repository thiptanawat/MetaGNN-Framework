# Experiments test-suite

Pytest suite that gates the `experiments/` tree before each push.
Maps directly to the MetaGNN Preprint manuscript:
each test module covers a single experiment ID (E01..E14)
that anchors a specific table/figure of the paper, per
[`EXPERIMENTS_INDEX.md`](../../EXPERIMENTS_INDEX.md).

## What the suite checks

| Layer            | What it verifies                                       |
|------------------|--------------------------------------------------------|
| **Smoke**        | Every `.py` under each experiment's `code/` parses (ast) and byte-compiles. |
| **Structure**    | Each E01..E14 folder has the expected README, `code/`, `results/` (or the documented E06-style sub-folder layout). |
| **Inventory**    | `shared_data/` exposes the canonical sub-folders (`recon3d/`, `tcga_crc_220/`, …) and the in-repo payloads (`gpr_mask`, `training_logs`) are present. |
| **Path resolution** | The actual input files each script opens at runtime are reachable from `shared_data/`, are the right size, have the right schema, and contain real data — not placeholders. Also asserts no script ships a `/Users/...` or `/home/<user>/...` hardcoded path. |
| **Script execution** | Every safe-to-invoke script runs cleanly with `--help` AND its module-level imports execute without raising. Catches scripts that fail at import time on a minimal env (e.g. broken `from torch_geometric import …` outside an `if HAS_TORCH:` guard). |
| **Unit**         | The pure functions inside each experiment script. See per-module breakdown below. |
| **Integration (slow)** | E07 and E02 are run **end-to-end** on the wired real data. E07 ingests the real 624-cohort multi-seed predictions, computes per-patient F1 across 50 seed-fold pairs, runs Mann-Whitney U + Wilcoxon, and emits the manuscript-anchored TSVs / LaTeX snippet / boxplot. The tests then assert the output numbers (92 MSI-H, 487 MSS, MSI-invariant Δ) match the manuscript exactly. Tagged `slow` + `integration`; `pytest -m "not slow"` skips them. |

## Per-experiment coverage

| File | Experiment | What it tests |
|---|---|---|
| `test_E01_metagnn_model.py` | E01 — Headline 220 CRC (Table 1) | `apply_gpr_rules` AND/OR semantics; MetaGNN forward shape, score range, MC-Dropout uncertainty *(skips when torch_geometric is missing)* |
| `test_E02_aggregate_multiseed.py` | E02 — Multi-seed 220 HMA (Table 2) | bootstrap CI determinism; per-seed and per-fold builders; LaTeX booktabs emission |
| `test_E07_msi_stratified.py` | E07 — MSI stratified (Table 6) | `tcga_patient_barcode` parsing; per-patient F1; full MSI lookup integration test against synthetic seed-fold tree |
| `test_E08_clustering.py` | E08 — Pass F clustering pilot | KMeans blob recovery, hierarchical consensus, ARI vs MSI/CMS, Pass-F verdict logic |
| `test_E09_seed_ensemble.py` | E09 — Pass D seed ensemble (Table 7) | mean-of-sigmoid, mean-of-logit, majority-vote, ensemble-σ math; 624-B AUROC verdict branches |
| `test_E10_brca_pipeline.py` | E10 — BRCA cross-cancer (Table 8) | `parse_gpr_rule` string parser; `build_edge_indices` topology + currency-metabolite filter |
| `test_E14_fba_viability.py` | E14 — FBA viability (Table 13) | Reaction-id mapping; per-patient percentile; viability rule (`> 1e-6`); random-baseline scaling; optional COBRApy textbook-model round-trip |
| `test_data_inventory.py` | shared_data + manuscript | All 11 canonical shared_data sub-folders present; GPR mask + training-log payloads ship with the repo; EXPERIMENTS_INDEX lists every E*. |
| `test_path_resolution.py` | Every experiment | Verifies the actual data files each script opens at runtime: Recon3D.xml is real SBML, `clinical_metadata_msi.tsv` has 624 patients with ≥500 evaluable MSI calls, all 10 seeds × 5 folds of multiseed predictions exist with the expected schema, ≥100 BRCA STAR-counts files are present, every E07/E14 default path resolves to a real file, and no Python script ships a `/Users/...` or `/home/<user>/...` hardcoded path. |
| `test_script_reads.py` | **Every experiment, every raw input** | The single most comprehensive guarantee: enumerates every raw input file consumed by any script (Recon3D bipartite graph × 7 files, TCGA-CRC 220 cohort × 8 files, TCGA-CRC 624 cohort × 5 files, HMA labels × 2, multi-seed 624 ConfigA + ConfigB × 4 files each, multi-seed 220 A + B1 × 2 files, E14 fold-prediction CSVs × 2, TCGA-BRCA × 12 files including best_model.pt, TCGA-LUAD × 2 outputs, DepMap × 1, training logs × 2 = 47 entries), asserts every one resolves on disk, plus directory-population checks (data-processing pipeline 220 ≥ 200 reaction_features, BRCA raw ≥ 100 STAR counts, etc.), full multi-seed completeness check (10 seeds × 5 folds × 4 artefacts = 200 files), and per-script default-path resolution for E07/E14. |
| `test_script_execution.py` | E02, E07, E10, E11, E14 | `--help` runs cleanly for every argparse-driven script; `import`-ing each script as a module succeeds. Catches scripts that fail at top-level on a minimal env. |
| `test_integration_e07.py` | **E07 end-to-end** | Runs `msi_stratified_624_configB.py` against the real 10-seed × 5-fold tree. Asserts 92 MSI-H + 487 MSS, all 10 expected outputs produced, MSI-invariance holds, p-values are valid, LaTeX snippet has manuscript anchor. **~70 s.** |
| `test_integration_e02.py` | **E02/E03 end-to-end** | Runs `aggregate_multiseed.py` against the real 624-cohort multi-seed tree. Asserts 50 seed-fold rows, 10 seeds, summary CIs enclose the mean, booktabs LaTeX is emitted. **~10 s.** |
| `test_smoke_compile.py` | Every experiment | Per-file `ast.parse` plus a single-subprocess batch `py_compile` over all discovered files. |
| `test_structure.py` | Every experiment | README ≥ 50 chars, `code/` populated (or in the documented hardlink-only set), `results/` present (or E06-style sub-folders). |

## Running

```bash
cd Journals/experiments

# Install minimum runtime
pip install -r requirements-test.txt

# Full suite (auto-skips heavy/optional deps)
pytest

# Only unit tests
pytest -m unit

# Only smoke tests
pytest -m smoke

# Skip integration tests
pytest -m "not integration"

# Verbose
pytest -v

# Specific experiment
pytest tests/test_E09_seed_ensemble.py -v
```

Expected on a minimal env (numpy/pandas/scipy/sklearn/matplotlib, no
torch_geometric / cobra / rdkit) **with `shared_data/` properly wired
to the canonical project trees**:

```
# Fast path (skips end-to-end runs)
$ pytest -m "not slow"
279 passed, 11 skipped, 13 deselected in ~9 s

# Full suite (with end-to-end runs of E02 and E07 on real data)
$ pytest
292 passed, 11 skipped in ~80 s
```

The 11 skips are intentional and self-documenting at run-time:
- 4 skips when torch_geometric is missing (E01 model tests)
- 1 skip when cobra is missing (E14 textbook FBA round-trip)
- 6 skips for the documented hardlink-only experiments (E04, E05, E06, E12, E13)

If `shared_data/` is empty (a fresh checkout without the data drop),
the path-resolution + integration layers will FAIL with messages
identifying the missing files. Run
`bash scripts/wire_shared_data.sh` (or set up the symlinks per
`../../shared_data/README.md`) to fix.

## Optional packages (auto-detected)

| Package | Tests it unlocks |
|---------|-----------------|
| `torch` (+ `torch_geometric`) | E01 MetaGNN model tests |
| `cobra` (+ network) | E14 COBRApy textbook FBA round-trip |
| `rdkit` | (currently no tests gate on it; pipelines optionally use it) |
| `h5py` | (currently no tests gate on it; pipelines optionally use it) |

If installed, the corresponding tests will run automatically.

## How to add a new test for a new experiment

1. Create `tests/test_E15_yourexperiment.py`.
2. Import path constants from `_paths`: `from _paths import EXPERIMENTS_DIR`.
3. For pure helpers, prefer extraction-via-regex into an isolated namespace
   (see `test_E07_msi_stratified.py`) over importing the whole script —
   most experiment scripts have side-effects at module top-level
   (path creation, `OUTDIR.mkdir(...)`, hardcoded `ROOT = Path(...)`).
4. Mark slow / dependency-gated tests with the registered markers
   (`@pytest.mark.unit`, `@pytest.mark.requires_torch`, etc.).
5. Run `pytest tests/test_E15_*.py -v` until green, then run the full
   suite to make sure you didn't regress anything.

## CI

`.github/workflows/tests.yml` wires this suite to GitHub Actions on
every push and pull request, with a Python 3.10/3.11/3.12 matrix.
