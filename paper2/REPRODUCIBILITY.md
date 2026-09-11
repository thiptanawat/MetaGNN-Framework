# Reproducing the results

Two levels, both supported by what is in this archive.

## 1. From the archived model responses (no network, no GPU, about two minutes)

    bash reproduce.sh analysis

Reads every stored response and rebuilds `results_frozen.json`, `ms/numbers.tex` and
`ms/figures/*.png`. `bash reproduce.sh paper` also typesets the manuscript. Because every result in
the text is a LaTeX macro generated from `results_frozen.json`, any change in the analysis shows up
as a change in the paper.

## 2. From scratch, re-issuing every model call

    ENDPOINT=http://<host>/<model>/v1/chat/completions bash collect.sh

Requires an OpenAI-compatible chat-completions endpoint. Everything else is deterministic given the
seed: reaction and patient sampling use `numpy.random.default_rng(2024)`, the donor shown in the
`shuffled` arm is drawn once, before any call, from a second seeded generator and recorded in
`design.json` under `swap_source`, and each script writes its own `design.json` recording exactly
which reactions and patients it drew. Design E extends Design A's patient sample rather than
replacing it (`--patients_from repr/design.json`).

Every prompt is built from `recon3d_aligned.json.gz`, the reference network in the canonical order
of the expression and label arrays (see the repository README, "One reaction order"). The BiGG JSON
export of Recon3D lists the same reactions in a different order and is never read by these scripts.
The 289 reactions that could not be aligned are excluded from every sample.

Transport errors are stored, not hidden. The driver retries a request four times with exponential
back-off; if all four fail it stores `__ERROR__ ...` as the reply and the parsed probability is
`null`. Re-running the same command resumes and re-sends every cell whose probability is `null`;
`--repair` re-sends only the transport errors and leaves empty or unparseable answers as the
outcomes they are (this matters for the reasoning-enabled run, where an empty reply is a result).
The endpoint answered a block of consecutive calls with HTTP 502/503 during the Design E and
reasoning runs of the archived collection; the affected calls (588 and 357) were re-sent with
`--repair` once it returned. Each `design.json` records the count under `resent_calls` and the time
under `resent_log`, and `resent_keys.json` beside each of the two runs lists the affected
`arm|patient|reaction` keys and the error each first received, so the re-sent cells can be dropped
or examined; `stats.py` reports the stranger's arm of Design E with and without them (`resent`).

Two answer keys are computed at analysis time rather than taken from the collection scripts: the
positive controls are scored against the percentile as printed (an integer), with `compare` trials
whose two printed integers are equal excluded as having no correct answer, and the MSI counting
control is keyed on the printed integer percentiles at or above 75. The collection scripts stored
keys computed from unrounded values; those fields are ignored, and `positive_control.py` now
stores the printed-integer key.

Serving is not bit-deterministic even at temperature 0, and the manuscript measures that directly:
repeating the patient-blind prompt within a session reproduces the same parsed answer about 99 % of
the time, and answering Design A's patient prompts again in Design E, hours later, reproduces it
97 % of the time with a mean absolute change of 0.006 (`cross_session` in `results_frozen.json`). A
re-collection should reproduce every reported quantity to within that noise, not exactly.

`results/repr40/design.json` records `patients_from: repr_patients.json`, the file the collection
machine held Design A's patient list in; it is the `patients` list of `results/repr/design.json`,
which is what `collect.sh` passes.

## What each file does

| file | role |
|---|---|
| `llm_probe.py` | the three arms: `reaction`, `patient`, `shuffled`. Resumable; rerunning skips completed cells; `--repair` re-sends transport errors only |
| `positive_control.py` | the `echo`, `read` and `compare` controls, built by the same record code path |
| `msi_probe.py` | the patient-level target: `own`, `swap` and the `count` aggregation control |
| `stats.py` | every statistic in the paper except the MSI experiment; writes `results_frozen.json` |
| `msi_stats.py` | the MSI experiment and its supervised reference; appends to `results_frozen.json` |
| `make_numbers.py` | turns `results_frozen.json` into `ms/numbers.tex`, one macro per quoted result |
| `make_figures.py` | all five figures, drawn from `results_frozen.json` and the raw responses |
| `analyze.py` | a human-readable console summary of one run, used during development |

## Inputs

| file | contents |
|---|---|
| `rxn_context.npz` | `labels` (10,600 activity labels), `has` (5,635 expression-bearing), `X` (624 x 10,600 GPR-mapped log2(TPM+1)), `pids` |
| `recon3d_aligned.json.gz` | the reference network in the canonical order: names, subsystems, equations, gene rules and bounds attached by alignment |
| `clinical_metadata_msi.tsv` | microsatellite instability status and molecular subtype for all 624 patients |

The first and third come from the data deposit and the second is built from it by
`paper1/code/build_aligned_recon3d.py`; `reproduce.sh` does not modify them.

The external-cohort transport needs three further files, which are public and are not redistributed
here. `code/external_cohort.py --fetch` downloads them into `~/geo` (or `--geo_dir`) and refuses to
run without them:

| file | source |
|---|---|
| `GSE39582_series_matrix.txt.gz` | <https://ftp.ncbi.nlm.nih.gov/geo/series/GSE39nnn/GSE39582/matrix/> |
| `GSE13294_series_matrix.txt.gz` | <https://ftp.ncbi.nlm.nih.gov/geo/series/GSE13nnn/GSE13294/matrix/> |
| `GPL570.annot.gz` | <https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPLnnn/GPL570/annot/> |

The reaction panels it derives (`results/external/*_panel.npz`, `panel_frozen.json`,
`coverage.json`) are committed, so every number in the manuscript reproduces without the downloads;
they are needed only to rebuild the panels from the raw series matrices.

## Stored responses

Each run directory holds `raw.json`, keyed `arm|patient|reaction`, with the parsed probability, the
parsed decision and the first 400 characters of the model's reply, plus `design.json` recording the
sampled reactions, the sampled patients and the arms. Nothing is discarded, including the calls that
failed to parse.
