# Reproducing Paper 2

Paper 2 is "Response sensitivity does not establish patient-specific validity in
language-model annotation of metabolic reactions". Its manuscript sources are withheld until
publication; the table and figure numbers below are the manuscript's.

Every model response the paper reads is archived here, so the statistics regenerate offline:
`reproduce.sh` takes the archives to `paper2/manuscript/numbers.tex` without contacting any
model. What cannot be regenerated is the collection itself, which needs a served model, and
the two shared inputs that live in the companion study's directory.

Labels and captions below are read from the current `.tex` sources, and table and figure
numbers are the ones the current source produces in document order. The committed
`paper2.pdf`, `paper2.aux` and `paper2.log` are rebuilt from that same source: every
`body_*.tex` file, including `body_protocol.tex` (the section that carries `tab:protocol`)
and `body_supp.tex` (the supplementary material, which now supersedes what was once a
separate appendix file), predates them, so their numbers agree with the source order given
here. If the two are ever allowed to diverge again, run `bash reproduce.sh paper` to
retypeset.

The `paper` mode writes three documents for paper2: `manuscript/paper2.pdf`, the combined
reading copy whose order is article, references, supplement; `manuscript/paper2_main.pdf`,
the article alone, which is the manuscript file a journal takes; and
`manuscript/paper2_supp.pdf`, the supplement alone, which is the supplementary file. The two
separate documents resolve their cross-references into each other through the `xr` package,
so section, table and figure numbers agree across all three; `tools/make_supp.py` generates
their wrappers and the supplement's own short reference list from the same bodies, so none of
the three can drift from the others.


## Environment

The same repository-root `requirements.txt` used by Paper 1 covers the packages the
analysis scripts need (NumPy, SciPy, scikit-learn, pandas, matplotlib); none of the
regeneration described below needs PyTorch, PyTorch Geometric or a GPU. Producing new model
responses is a different matter:

- The main backbone, Qwen3.8-27B, is served through an OpenAI-compatible chat-completions
  endpoint that the manuscript describes as not under the authors' control (serving
  precision and vLLM version are not recorded for it).
- The second backbone, Gemma 4 31B-it, and the third, Mistral Small 3.2 24B, are served
  locally with vLLM in bfloat16 on one H100 GPU, per `paper2/manuscript/body_methods.tex`
  (Section "Backbone and serving").
- Every draw the drivers make (reactions, probe patients, donors) comes from NumPy
  generators seeded with 2024, so a re-collection against the same or an equivalent endpoint
  sends the identical prompts; it will not reproduce the model's answers bit-for-bit, because
  serving is not deterministic even at temperature zero (the manuscript measures this
  directly as the determinism floor).

## The one command

```
bash reproduce.sh analysis
```

run from the repository root, reads every response archived under `paper2/results/` and
rewrites `paper2/results_frozen.json`, `paper2/manuscript/numbers.tex` and the six figures
under `paper2/manuscript/figures/`. No network access and no GPU are used. It calls, from
inside `paper2/`, in order:

```
python3 code/stats.py
python3 code/msi_stats.py
python3 code/make_numbers.py
python3 code/check_macros.py
python3 code/make_results_md.py
python3 code/make_figures.py
```

(`msi_stats.py` appends to the same `results_frozen.json` that `stats.py` writes, so it must
run second, not independently.) `paper2/REPRODUCIBILITY.md` records this step at about two
minutes; `bash reproduce.sh paper`, which additionally typesets both papers' PDFs, was last
timed end to end at 7 minutes 54 seconds with no GPU and no network.

Re-collecting responses from scratch instead of reading the archive needs a live endpoint:

```
ENDPOINT=http://<host>/<model>/v1/chat/completions bash collect.sh
```

as documented in `paper2/REPRODUCIBILITY.md`, which this guide does not repeat in full.

### The path-existence check

```
python3 tools/check_release.py
```

scans both manuscripts for every file path named in a `\texttt{...}` or `\path{...}` span
and checks that it exists in the repository. See `docs/REPRODUCTION_PAPER1.md` for the
current full output; the only paths it reports missing that Paper 2 names are one
HuggingFace model identifier in its own bibliography (`Qwen/Qwen3.8-27B`, which contains a
slash but is not a repository path) and three data files named in Paper 1's methods that
Paper 2 does not itself reference.

## Table and figure map

`stats.py` computes every reaction-activity statistic in the paper (Designs A, B and E, the
positive controls, the format-invariance arms, the reasoning-enabled run, the second- and
third-backbone repeats, the tissue-matched-label rescoring, and the blocked-versus-interleaved
comparison) from the archived responses under `paper2/results/`, and writes them all into
`results_frozen.json`; `msi_stats.py` computes the microsatellite-instability experiment
separately and appends to the same file. `make_numbers.py` turns the combined file into
`numbers.tex`; the manuscript's tables are typeset directly from those macros, with one
exception (`tab:label`, built from a single generated macro, `\lblRows`). `make_figures.py`
draws all six figures from the same combined file.

The main text carries tables numbered 1 to 5 and figures numbered 1 to 5. Supplementary
material follows the main text, numbered S1 onward, and carries three more tables (S1 to
S3) and one more figure (S1).

### Main text

| Label (number) | Caption (lead) | Primary result directories (`paper2/results/`) | Produced by | Output | Macros file |
|---|---|---|---|---|---|
| `fig:design` (Fig. 1) | The three arms | none (a schematic of the prompt design, not a data plot) | `make_figures.py` | `manuscript/figures/fig_design.png` | `numbers.tex` |
| `tab:msi` (Table 1) / `fig:msi` (Fig. 2) | Microsatellite instability, a patient-level target / A target that varies between patients | `msi_il/` (primary) or `msi/` (fallback, blocked order), plus `clinical_metadata_msi.tsv` and the supervised-reference fit inside `msi_stats.py` itself | `msi_probe.py` (collection), `msi_stats.py` (statistics) | table typeset from macros; figure at `manuscript/figures/fig_msi.png` | `numbers.tex` |
| `tab:main` (Table 2) / `fig:ladder` (Fig. 3) | Main results / No arm improves on the raw-expression baseline | `repr_il/` (primary) or `repr/` (fallback) for Design A; `pilot/` for Design B; `repr40_il/` or `repr40/` for Design E | `llm_probe.py` (collection), `stats.py` (statistics) | table typeset from macros; figure at `manuscript/figures/fig_ladder.png` | `numbers.tex` |
| `tab:pc` (Table 3) | Positive controls, on a record matched in the placement of the number | `pc/` (main backbone) | `positive_control.py`, `stats.py` | typeset from macros | `numbers.tex` |
| `tab:backbone2` (Table 4) | Design A, the positive controls and the MSI probe on three backbones | `repr_b2il/` or `repr_b2/`, `pc_b2/`, and `msi_b2il/`/`msi_b2/` (second backbone); `repr_b3il/` or `repr_b3/`, `pc_b3il/`/`pc_b3/`, and `msi_b3il/`/`msi_b3/` (third backbone) | `llm_probe.py`, `positive_control.py`, `msi_probe.py` on the second and third backbones; `stats.py`, `msi_stats.py` | typeset from macros; the third-backbone column (behind `\ifdefined\rTPatient`) is included in the current build | `numbers.tex` |
| `tab:protocol` (Table 5, new) | Each design collected twice: blocked order, then interleaved with the complete archive | `repr/` vs `repr_il/`, `repr_b2/` vs `repr_b2il/`, `repr_b3/` vs `repr_b3il/`, and `repr40/` vs `repr40_il/` (all four pairs now complete) | `llm_probe.py`, `stats.py`'s blocked-versus-interleaved comparison | typeset from macros; the third-backbone and Design E rows (behind `\ifdefined\ilTOwnStrBlocked` and `\ifdefined\ilEOwnStrBlocked`) are included in the current build | `numbers.tex` |
| `fig:responsive` (Fig. 4) | Responsive, and responsive to the wrong thing | same as `tab:main`, plus `det_patient/` (the within-session replicate) for the noise-floor bars | `llm_probe.py`, `stats.py` | `manuscript/figures/fig_responsive.png` | `numbers.tex` |
| `fig:presence` (Fig. 5) | What the added number is associated with | same as `tab:main`; the least-squares decomposition is computed inside `stats.py` from the same archived responses | `stats.py` | `manuscript/figures/fig_presence.png` | `numbers.tex` |

| `sec:external` (Section 4.3) | The frozen percentile-only interface transported to two independent cohorts | `results/msi2/` (23 runs: three cohorts by three configurations by three backbones, where each ran), `results/external/` (the two frozen panels), `results/frozen_scorer.json` | `external_cohort.py` (panels), `frozen_scorer.py` (the frozen supervised reference), `msi_probe2.py` (collection), `msi2_stats.py` (statistics, 10,000-draw exchangeability nulls) | prose typeset from macros, all behind `\ifdefined\xMiExtAZeroDiff` | `numbers.tex`, via `results/msi2_frozen.json` |

### Supplementary material

| Label (number) | Caption (lead) | Primary result directories (`paper2/results/`) | Produced by | Output | Macros file |
|---|---|---|---|---|---|
| `tab:label` (Table S1) | The eleven reconstructions behind the label | `paper1/data/labels_ht29.json` (not a Paper 2 result directory; shared with Paper 1) | `paper1/code/build_labels_ht29.py` (generates the source file); `make_numbers.py` (emits the table body as the single macro `\lblRows`) | typeset from one macro | `numbers.tex` |
| `tab:format` (Table S2) | The same reactions and patients, the evidence written three ways | `fmt_cat/`, `fmt_pct/` (Design B reactions and patients, evidence rewritten) | `llm_probe.py` with its format-variant flags, `stats.py` | typeset from macros | `numbers.tex` |
| `tab:htlabel` (Table S3) | Every reaction-activity run rescored against the HT29-only label | the same archived responses as `tab:main` and `tab:backbone2`, rescored (no new calls) against `paper1/data/activity_labels_ht29.pt` | `stats.py` | typeset from macros | `numbers.tex` |
| `supp:external` (Supp. Section) | The external cohorts: mapping, coverage and every frozen choice | same as `sec:external` | same as `sec:external` | prose and counts typeset from macros | `numbers.tex` |
| the tie audit (Supp.) | How often two patients' stored values tie, and under which rounding | every archived response file, reread | `tie_audit.py` | prose typeset from macros | `numbers.tex`, via `results/tie_audit.json` |
| the crossed bootstrap (Supp.) | The primary contrast resampled over patients and reactions together, 10,000 draws under three seeds | `repr_il/` (primary Design A archive) | `crossed_boot_10k.py` | prose typeset from macros | `numbers.tex`, via `results/crossed_boot_10k.json` |
| `fig:reliability` (Fig. S1) | Reliability of the returned probability against the assay call | same MSI-probe archives as `tab:msi` and the MSI rows of `tab:backbone2` (`msi_il`/`msi`, `msi_b2il`/`msi_b2`, `msi_b3il`/`msi_b3`, each preferring the newer interleaved collection when present; see `msi_stats.py`'s `_fold` helper) | `msi_probe.py` (collection), `msi_stats.py` (the per-arm `reliability` field) | `manuscript/figures/fig_reliability.png` | `numbers.tex` |

`make_results_md.py` additionally writes a plain-Markdown summary of `results_frozen.json`
(read by `make_readme.py` for the repository's `README.md` tables); it is not itself a
source for any manuscript table or figure.

## What needs a live endpoint or the GPU host

None of the following can be regenerated from this repository alone.

- **Every raw response archive** (`paper2/results/*/raw.json` and `design.json`) is the
  output of `llm_probe.py`, `positive_control.py` or `msi_probe.py` against a live
  OpenAI-compatible endpoint. The Analysis tier above (`stats.py`, `msi_stats.py`,
  `make_numbers.py`, `make_figures.py`) only reads what these already wrote; it issues no
  network calls itself.
- **The second and third backbones specifically** need the local GPU serving stack the
  manuscript describes (one H100, vLLM, bfloat16), not only an endpoint; the main backbone's
  endpoint is external and, per the manuscript, was not under the authors' control.
- **Every queued collection has now landed, and `collect.sh` re-issues all of them.**
  `paper2/collect.sh` issues every collection the manuscript reports as primary, with the
  seeds and donor seeds the archives record, parameterized by `ENDPOINT`/`MODEL` for the
  main backbone and `ENDPOINT2`/`ENDPOINT3` for the two further backbones; the external
  cohorts' frozen panels and the frozen supervised reference are built inside it. The
  scripts that actually ran on the study's host are kept unedited under
  `paper2/code/provenance/` with a README that says what each collected; they change into
  that host's directories and hardcode its endpoint, so they are a provenance record and
  not the reproduction path. What `collect.sh` does not do is serve the models: the second
  and third backbones need the local serving stack described below, and the serving flags
  are recorded in the manuscript. The collections themselves are already present:
  - The third backbone's interleaved, complete-archive re-collection (`repr_b3il/`,
    `pc_b3il/`, `msi_b3il/`) is present alongside its earlier blocked-order collection
    (`repr_b3/`, `pc_b3/`, `msi_b3/`). The manuscript's "three backbones" framing (the
    abstract, the contributions list, `body_results2.tex`, `tab:protocol` (Table 5) and the
    conclusion) is written behind `\ifdefined\rTPatient` and related guards, and those
    macros are now defined in the committed `numbers.tex`, so the current build reports
    three backbones throughout.
  - Design E's interleaved, complete-archive re-collection (`repr40_il/`) and the
    second-donor-schedule sensitivity run (`repr_il_d2/`) are likewise present alongside
    `repr40/` (blocked order). Table 5's Design E row and the `\ifdefined\rAdPatient`
    donor-schedule paragraph of `body_protocol.tex` are correspondingly active in the
    current build.
  - Design A and Design A on the second backbone already had both their blocked and
    interleaved collections (`repr/` and `repr_il/`; `repr_b2/` and `repr_b2il/`); the third
    backbone and Design E have now joined them, so all of Table 5's rows and every
    Design-A/backbone/Design-E number quoted in the text are the interleaved,
    complete-archive figures the manuscript treats as primary.
  - `stats.py`'s `pick()` helper (in its `__main__` block) always prefers the interleaved
    collection when one exists and falls back to the blocked one, under the design's own
    tag, when it does not, so a partially-collected robustness queue would have degraded
    gracefully rather than producing an inconsistent build; with every collection now
    complete, that fallback is no longer exercised for these designs.
- **The reasoning-enabled run** (Supplementary Section S12, `sec:think`) is complete
  (`results/think/`) and is not gated behind an `\ifdefined` guard; its high non-parse rate
  is a property of the run itself, discussed in the text, not a sign that data is missing.
- **The two inputs shared with Paper 1** that this paper's methods depend on,
  `paper1/data/activity_labels_ht29.pt` and `recon3d_aligned.json.gz`, are already present in
  this repository (the second copied into `paper2/` as well as under `paper1/data/`), so
  they are not a reproducibility gap; `paper2/manuscript/body_back.tex` states that
  `stats.py` stops with an error, rather than silently dropping `tab:htlabel`, if the label
  file is ever absent. `rxn_context.npz` and `clinical_metadata_msi.tsv`, described in
  `paper2/REPRODUCIBILITY.md` as coming from the data deposit, are likewise already
  committed under `paper2/`.
