# MetaGNN

[![Code DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21217569.svg)](https://doi.org/10.5281/zenodo.21217569)
[![Data DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21217579.svg)](https://doi.org/10.5281/zenodo.21217579)
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
five checks so they can be pointed at another benchmark. The manuscripts themselves are under
review and are not in this repository yet.

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

The processed tensors, trained checkpoints, and per-reaction score outputs are
archived on Zenodo (see the Data DOI badge above). Each `pipelines/<cancer>/`
directory is numbered in execution order, starting from `00_build_recon3d_graph.py`,
which rebuilds the heterogeneous graph from `Recon3D.json` in the data archive.
`verify_reproducibility.py` in the data archive recomputes the BRCA headline metrics
(AUROC 0.9864, AUPRC 0.9937) directly from the shipped files.

## Data availability

All processed tensors derive from open-access TCGA expression obtained through the
NIH Genomic Data Commons and are deposited on Zenodo (CC-BY-4.0). The METABRIC and
CPTAC-BRCA cohorts used in the cross-consortium analysis are governed by their own
data-use terms and are **not** redistributed here; they can be obtained from
cBioPortal and the NIH Proteomic Data Commons by users with appropriate access.

## Citation

If you use this code, please cite the MetaGNN manuscript and the archived releases:

> Phongwattana, T. and Chan, J. H. MetaGNN: A Heterogeneous Graph Attention Network
> for Hub-Connectivity-Aware Reaction Activity Scoring Toward Metabolic Network
> Reconstruction. Code: https://doi.org/10.5281/zenodo.21217569 ·
> Data: https://doi.org/10.5281/zenodo.21217579

## License

MIT (see `framework/LICENSE`).
