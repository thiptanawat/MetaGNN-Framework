# Reproducing the Language-Model Evidence

The Language-Model Evidence study is "Testing when language models use sample-specific molecular evidence: controlled
interventions and external cohort evaluation". Its manuscript sources are withheld from
this repository until publication; every model response the paper reads is archived here, so
its statistics regenerate offline, and the map at the end of this guide gives, for every
table and figure of the last built documents, the archives behind it and the scripts that
collected and scored them.

## Two levels of reproduction

1. **Regenerating the reported numbers from the archived responses.** `bash reproduce.sh
   analysis` reads every archive under `paper2/results/` (each collection is a `design.json`
   recording the prompts, seeds, donors, endpoint and settings, and a `raw.json` holding
   every reply with its serving metadata), recomputes every statistic, and writes
   `paper2/results_frozen.json`, `paper2/results/msi2_frozen.json`,
   `paper2/manuscript/numbers.tex` and the figures. It contacts no model and needs no GPU.
   The external-cohort block resamples 21 collections with 10,000-draw nulls, so the Language-Model Evidence
   chain takes about fifteen minutes.
2. **Re-collecting the responses.** `paper2/collect.sh` re-issues every collection the
   manuscript reports, with the seeds and donor seeds the archives record, against endpoints
   you supply (`ENDPOINT`/`MODEL` for the first backbone, `ENDPOINT2`/`MODEL2` and
   `ENDPOINT3`/`MODEL3` for the other two). Because every draw the drivers make comes from
   NumPy generators seeded with 2024, a re-collection sends the identical prompts. It will
   not reproduce the answers bit for bit: serving is not deterministic even at temperature
   zero, which the manuscript measures directly as its determinism floor, and the first
   backbone's endpoint was not under the authors' control. The scripts that actually ran on
   the study's host are kept unedited under `paper2/code/provenance/`, with a README saying
   what each collected; they hardcode that host's directories and endpoint, so they are the
   provenance record and not the reproduction path. What `collect.sh` does not do is serve
   the second and third backbones; the serving flags are in `run_msi2_local.sh` and
   `run_backbone3.sh` under `provenance/`, and the manuscript's serving section repeats them.

The `paper` mode of `reproduce.sh` typesets the manuscripts after level 1. It needs the
manuscript sources, which are not part of the public checkout, and says so and stops when
they are absent.

## Environment

The repository-root `requirements.txt` covers the packages the analysis scripts need (NumPy,
SciPy, scikit-learn, pandas, matplotlib); none of level 1 needs PyTorch or a GPU.

The committed result files of this study were written by `bash reproduce.sh analysis` with the
pinned NumPy, SciPy and scikit-learn on an Intel x86-64 processor with AVX-512. A fresh clone
regenerates them byte for byte in a clean installation of those versions on such a processor; this
was checked on two machines, one of them the host the collections ran on. The files hold ten
decimals, so two dependences show in them:

- **Library versions.** With other versions every value is the same except seven quantities of
  the supervised reference on the patient-level target (`msi_stats.py`: its Brier score, its
  calibration in the large, and the AUROCs and intervals of the reference applied to swapped
  panels), which move in the fifth to seventh decimal because the fitted coefficients stop at
  slightly different points. One printed value is among them: the reference fitted on swapped
  panels is 0.5309 with scikit-learn 1.7.2 and SciPy 1.15.3, and 0.5310 with scikit-learn 1.9.1
  and SciPy 1.18.1.
- **Processor.** The percentile printed in a prompt is the rank of the patient's value within the
  cohort, taken from NumPy's default `argsort`, which leaves the order of tied values to an
  implementation that differs with the processor's vector instruction set; 938 of the 5,635
  expression-bearing reactions have tied values, and within a tie group the order is not the order
  of the patients. The inductively selected panel of `msi_stats.py` is a top-k selection of the
  same kind. The statistics rebuild both as the collection scripts did, so they reproduce the
  prompts on processors of the kind the collections ran on, Intel x86-64 with AVX-512: the 11,400
  percentiles that the positive-control records stored at collection time equal the rebuilt ranks
  there, the 522 on tied values included, and those 522 differ when AVX-512 is disabled. An AMD
  processor with AVX-512 was not tested. Elsewhere three groups of stored values differ:
  - the five quantities computed from the percentile (its AUROC as a ranking over all reactions
    and over expression-bearing reactions, the two variance-decomposition terms that use it, and
    the Spearman correlation with it), by at most 0.002;
  - the overlap of the inductive panel, by one percentage point (0.75 where 0.76 is stored);
  - the two summaries of `tie_audit.py` of how far the midrank convention would move the printed
    percentile of a tied entry, its mean by up to one point and its maximum by up to two.

  That is 71 of the 5,383 generated values on x86-64 without AVX-512 (checked by disabling
  it in NumPy, the whole analysis) and 61 on aarch64 (checked on Apple Silicon for every
  script of the chain except `msi2_stats.py`), for example a raw-percentile AUROC of 0.6183 and
  0.6182 where the reported value is 0.6181. The values among them that the manuscript prints
  differ by at most 0.001. The reported values are those of the collection host.

`tools/check_environment.py`, which `reproduce.sh` runs first, names any installed version that
differs from the pins and says when NumPy runs without AVX-512 or on another architecture.

Producing new responses needs served models:

- The first backbone, Qwen3.8-27B, was reached through an OpenAI-compatible chat-completions
  endpoint not under the authors' control, whose serving precision and software version are
  not recorded.
- The second, Gemma 4 31B-it, and the third, Mistral Small 3.2 24B Instruct, were served
  locally with vLLM 0.19.0 in bfloat16 on one accelerator, from the weight repositories and
  with the flags the provenance launchers record.

## The one command

```
bash reproduce.sh analysis
```

runs, from inside `paper2/` and in this order,

```
python3 code/stats.py
python3 code/msi_stats.py
python3 code/collection_manifest.py
python3 code/tie_audit.py
python3 code/crossed_boot_10k.py
python3 code/msi2_stats.py
python3 code/deployment_records.py
python3 code/make_numbers.py
python3 code/check_macros.py
python3 code/make_results_md.py
python3 code/make_figures.py
```

after the Reaction-Scoring Audit chain, and then `tools/check_release.py`. `msi_stats.py` appends to the
`results_frozen.json` that `stats.py` writes, so the order matters. `stats.py` prefers each
design's interleaved, complete-archive collection where one exists and falls back to the
blocked one under the design's own tag, so a checkout missing a collection degrades to the
blocked figure rather than an inconsistent build; every collection the manuscript reports is
present, so that fallback is not exercised.

## The records behind the external test

The external-cohort test of the manuscript rests on records that are committed beside the
archives, and that this guide names so a reader can find each without the manuscript:

- **The analysis plan**, `docs/ANALYSIS_PLAN_2026-09-10.md`, written before the external
  collections were made. The manuscript's supplement records three departures from it and
  what the repository's history can and cannot show about its date: the plan first appears
  in the public repository in the release of 11 September, after the collections of 10
  September, so the public record does not by itself establish the freeze, and the
  manuscript does not call the test preregistered.
- **The frozen interface**: `results/external/panel_frozen.json` (the panel restricted to
  reactions coverable on the external platform), `results/external/frozen_scorer.json` (the
  supervised reference fitted on the development cohort and frozen before any external
  patient was scored), and the two external panels `results/external/gse39582_panel.npz`
  and `gse13294_panel.npz`, all built by `external_cohort.py` and `frozen_scorer.py`.
- **The deployment records**, `results/msi2/deployment_records.json`, written by
  `deployment_records.py` from the 21 collections' `design.json` and `raw.json`: per
  backbone, the served name, the endpoint, the timestamps, the decoding settings, the call
  counts, the finish reasons and what the server echoed, and a list of what the archives do
  not record.
- **The model records**, `results/model_records.json`, written by `model_records.py` on the
  serving host after the collections: for the two locally served backbones, the Hub revision
  of the cached snapshot the launcher loaded from, every configuration, tokenizer and weight
  file with its size and SHA-256, and one digest over that list; the first backbone is
  recorded as unavailable, since no artifact directory exists for an endpoint the authors did
  not run.
- **The statistics**, `results/msi2_frozen.json`, from `msi2_stats.py`: for every
  collection the own and donor AUROCs, the collected-arm contrast with its descriptive
  recipient-pair bootstrap range and the one-sided conditional permutation test (the
  exceedance count, the draw count, the plain proportion and the $(b+1)/(B+1)$ estimator),
  the cached reconstruction and the five-schedule spread beside it, the serving-repeat
  statistics, the calibration maps (with the fitted slope kept beside the applied one and the
  constant fallback flagged where the plan's rule replaced a nonpositive slope) and the
  frozen reference's transfer to each cohort.

## What is shared with the Reaction-Scoring Audit

`paper1/data/activity_labels_ht29.pt` and `recon3d_aligned.json.gz` (copied into `paper2/`
as well) are inputs of this paper's methods and are committed; `stats.py` stops with an error
rather than silently dropping the tissue-matched table if the label file is absent.
`rxn_context.npz` and `clinical_metadata_msi.tsv` are committed under `paper2/`.

## Table and figure map

<!-- map:start -->
_Generated by `tools/make_repro_map.py` from the documents built on 2026-11-11; the manuscript sources are not in the public checkout, so this is the last built numbering._

### Main text

| Label (number in the built document) | Caption lead | Input result files (relative to `paper2/`) | Produced by |
|---|---|---|---|
| `fig:design` (Figure 1) | The three arms. | none: a schematic | drawn by make_figures.py |
| `tab:main` (Table 1) | Main results. | results/repr_il/ (Design A), results/pilot/ (Design B), results/repr40_il/ (Design E) | llm_probe.py (collection), stats.py (statistics) |
| `fig:msi` (Figure 2) | A target that varies between patients. | results/msi_il/ | msi_stats.py; drawn by make_figures.py |
| `tab:msi` (Table 2) | Microsatellite instability, a patient-level target. | results/msi_il/, clinical_metadata_msi.tsv | msi_probe.py, msi_stats.py |
| `tab:external` (Table 3) | Every analyzed collection of the frozen percentile-only interface. | results/msi2/*/ (21 collections), results/external/, results/msi2_frozen.json | msi_probe2.py (collection), msi2_stats.py (statistics) |

### Supplementary material

| Label (number in the built document) | Caption lead | Input result files (relative to `paper2/`) | Produced by |
|---|---|---|---|
| `fig:reliability` (Figure S1) | Reliability of the returned probability against the assay call. | results/msi_il/, results/msi_b2il/, results/msi_b3il/ | msi_stats.py; drawn by make_figures.py |
| `tab:label` (Table S1) | The eleven reconstructions behind the label. | ../paper1/data/labels_ht29.json | ../paper1/code/build_labels_ht29.py |
| `fig:ladder` (Figure S2) | No arm improves on the raw-expression baseline. | as tab:main | stats.py; drawn by make_figures.py |
| `tab:format` (Table S2) | The same reactions and patients, the evidence written three ways. | results/fmt_cat/, results/fmt_pct/ | llm_probe.py with the format flags, stats.py |
| `fig:responsive` (Figure S3) | Responsive, and as responsive to a stranger's values. | results/repr_il/, results/repr40_il/ | stats.py; drawn by make_figures.py |
| `tab:htlabel` (Table S3) | Every reaction-activity run rescored against the HT29-only label. | the archived responses of tab:main and tab:backbone2, rescored against ../paper1/data/activity_labels_ht29.pt | stats.py |
| `fig:presence` (Figure S4) | What the added number is associated with. | results/repr_il/ | stats.py; drawn by make_figures.py |
| `tab:deployment` (Table S4) | The deployments of the external collections, as the archives record them. | results/msi2/*/design.json and raw.json, results/model_records.json | deployment_records.py, model_records.py |
| `tab:resample` (Table S5) | The primary contrast under every resampling scheme. | results/repr_il/, results/crossed_boot_10k.json | stats.py, crossed_boot_10k.py |
| `tab:pc` (Table S6) | Positive controls, on a record matched in the placement of the number. | results/pc/, results/pc_echoval/ | positive_control.py, stats.py |
| `tab:backbone2` (Table S7) | Design A on three backbones. | results/repr_b2il/, results/repr_b3il/ | llm_probe.py on the second and third backbones, stats.py |
| `tab:backbone2desc` (Table S8) | What the Design A answers look like on each backbone. | results/repr_b2il/, results/repr_b3il/ | stats.py |
| `tab:backbone2pc` (Table S9) | The positive controls on the same backbones. | results/pc_b2/, results/pc_b3il/ | positive_control.py, stats.py |
| `tab:backbone2msi` (Table S10) | The microsatellite-instability probe on the same backbones. | results/msi_b2il/, results/msi_b3il/ | msi_probe.py, msi_stats.py |
| `tab:protocol` (Table S11) | Each run collected twice: blocked order, then interleaved with the complete archive. | results/repr/ against results/repr_il/, and the same pairs for the other backbones and Design E | stats.py |
<!-- map:end -->
