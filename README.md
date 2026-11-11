# MetaGNN

[![Code DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21217568.svg)](https://doi.org/10.5281/zenodo.21217568)
[![Data DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21217578.svg)](https://doi.org/10.5281/zenodo.21217578)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A heterogeneous graph attention network that scores per-reaction metabolic activity across the full Recon3D network (10,600 reactions, 5,835 metabolites) from patient transcriptomes, with Monte Carlo dropout uncertainty and an optional LLM-based boundary-reaction reasoning module.

## Repository layout

```
framework/     the installable metagnn package (GATv2 model, trainer, graph builder,
               MC-dropout uncertainty, evaluator) + driver scripts and configs
pipelines/     per-cancer pipelines (TCGA-CRC, TCGA-BRCA, TCGA-LUAD)
tests/         18-module pytest suite
validators/    per-experiment result validators used by the tests
paper1/        the joint patient-and-reaction holdout study: code and results
paper2/        the language-model probe: code and the archived model responses
metabench/     the five benchmark-validity checks, framework-agnostic
docs/          reproduction notes, the frozen analysis plan, the protocol
reproduce.sh   regenerates every reported number from the committed results
```

## Two studies of what this benchmark measures

The scorer above is trained against reaction activity labels that are the same for every
patient in a cohort, and cross-validated by holding out patients. Two studies in this
repository ask what that arrangement can and cannot establish, and they do not flatter the
model: under a joint patient-and-reaction holdout a logistic regression on network features,
given no patient data at all, matches the graph scorer, and giving the scorer each patient's
own transcriptome in place of one cohort average does not reliably help. The same audit is
then applied unchanged to an independently published metabolic-dependency predictor scored
against measured CRISPR gene effect.

`STUDIES.md` is the entry point: what was found, how the repository is laid out, and the one
command that regenerates every number from the committed outputs. `metabench/` packages the
five checks so they can be pointed at another benchmark. The manuscripts themselves are not
part of this repository until they are published.

## Installation

The exact environment the published results were produced with is pinned in
`framework/requirements.txt` (PyTorch 2.2.0 / PyTorch-Geometric 2.5.0, CUDA 12.1,
cobra 0.29.0, rdkit 2023.09.5). A CPU-only stack re-verified to import and pass the
test suite is in `requirements-verified.txt`:

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements-verified.txt
```

## Running the tests

```bash
pip install pytest pytest-timeout
pytest tests/ -q
```

Heavy dependencies (`torch_geometric`, `cobra`, `rdkit`) are auto-detected; tests
that need them skip gracefully if they are absent, so the pure-Python unit tests run
on a minimal install. See `TEST_REPORT.md` for the full pass/skip breakdown.

## Reproducing the results

`bash reproduce.sh analysis` regenerates every number of the two studies from the committed
result files, with no GPU and no network. `docs/REPRODUCTION_PAPER1.md` and
`docs/REPRODUCTION_PAPER2.md` separate what a checkout regenerates from what needs the
prediction arrays, the served models or the accelerator.

The curated cohort is archived on Zenodo (see the Data DOI badge above): 624 colorectal tumors
with their reaction features, the reconstruction-derived labels, the clinical annotation and the
fixed splits, together with the per-cell prediction arrays of the first study.
`paper1/data/splits/splits.json` is the same split file as the one in the archive. Each
`pipelines/<cancer>/` directory is numbered in execution order, starting from
`00_build_recon3d_graph.py`, which rebuilds the heterogeneous graph from `Recon3D.json` in the
data archive.

Two earlier results of the scorer should not be read as accuracy. The near-ceiling score on the
breast cohort (AUROC 0.9864) is label leakage: those labels are thresholds of the model's own
input where a reaction carries expression and seeded random draws elsewhere, as the preprint
cited below shows and `diagnostics/05_label_replay.py` in the data archive replays. The
versions of the data record deposited in July and September 2026 also held a 220-patient
colorectal set that was later found to be synthetic, and colorectal feature tensors that were
all zero; the current version replaces both, and `VERSION_NOTE.md` in its cohort archive lists
what changed. Use the current version of the record.

## Data availability

All processed tensors derive from open-access TCGA expression obtained through the
NIH Genomic Data Commons and are deposited on Zenodo (CC-BY-4.0). The METABRIC and
CPTAC-BRCA cohorts used in the cross-consortium analysis are governed by their own
data-use terms and are **not** redistributed here; they can be obtained from
cBioPortal and the NIH Proteomic Data Commons by users with appropriate access.

## Citation

If you use this code or these checks, please cite the preprint and the archived records:

> Phongwattana, T. and Chan, J. H. Benchmark validity in graph neural network scoring of
> metabolic reaction activity on Recon3D: detecting label leakage, memorized noise and
> input-invariant models. bioRxiv (2026). https://doi.org/10.64898/2026.09.02.748315
>
> Code: https://doi.org/10.5281/zenodo.21217568 · Data: https://doi.org/10.5281/zenodo.21217578

The two identifiers above and the badges are the concept identifiers of the records, which
resolve to their current versions. The two benchmark-validity studies correspond to the tag named in
`release.json`; the version identifiers minted for that tag are recorded there and in
`CITATION.cff` once they exist.

## License

MIT (see `framework/LICENSE`).
