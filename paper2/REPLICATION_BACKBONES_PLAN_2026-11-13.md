# Analysis plan: the replication of the own-versus-donor contrast on two further backbones

Date: 13 November 2026. This plan, `code/replication_backbones.py`, `code/replication_serving_record.py` and the launcher `code/provenance/run_replication_backbones.sh` were committed to the public repository in one commit before the first call to either backbone on the replication sample. Nothing below was changed after that commit; deviations, if any, are listed at the end under a dated heading.

The plan extends `REPLICATION_PLAN_2026-11-13.md` (below, the first plan). It was written after the outcome of the first plan on the main backbone was known. Neither of the two backbones named here had been sent a prompt of the replication sample when it was written.

## 1. Question

On the main backbone the replication found the paired difference between the own arm and the donor arm in per-patient AUROC inside the 0.02 margin, with a 90% interval that excluded zero (+0.0061, +0.0002 to +0.0120; 100 patients). An analysis added after that outcome had been read found that ranking the printed percentile with no model gives a difference of the same sign on the same prompts (+0.0035, -0.0002 to +0.0073).

This extension asks, on the same sample and for each of the two other backbones of the study:

1. whether the own-minus-donor difference lies within the margins of the first plan, and
2. whether it differs from the difference that the printed percentile gives with no model.

## 2. Sample

The sample of the first plan, unchanged: `results/replication/design_master.json` (SHA-256 `2f39d5fc87a1cce2fe210e457f35ad8120d464c04968c42d9be48bda32bc6869`). Nothing is redrawn. The 300 reactions, the order of the patient list, the blocks of 10 patients and the three donor schedules d1, d2 and d3 are those of the first plan.

## 3. Backbones and serving

| Tag | Backbone | Hugging Face repository and revision | Served name |
|---|---|---|---|
| b3 | Mistral Small 3.2 24B Instruct | `mistralai/Mistral-Small-3.2-24B-Instruct-2506`, `95a6d26c4bfb886c58daf9d3f7332c857cb27b43` | `mistral-small-3.2-24b-instruct` |
| b2 | Gemma 4 31B-it | `google/gemma-4-31B-it`, `439edf5652646a0d1bd8b46bfdc1d3645761a445` | `gemma-4-31b-it` |

These are the revisions that `results/model_records.json` records for the earlier collections of the two backbones. Before a backbone is served, every file of its downloaded snapshot is hashed and compared with that record. A backbone whose files differ from the record is not collected.

Each backbone is served by one local vLLM process on one accelerator (NVIDIA H100, 80 GB) of a second host, not the host of the earlier collections, one backbone at a time. The serving flags are those of the earlier collections of these backbones (`code/provenance/run_b2il.sh`, `code/provenance/run_b3il.sh`): weights in bfloat16, a context of 8,192 tokens, at most 32 sequences in a batch, prefix caching, a memory fraction of 0.90; for Gemma, reasoning disabled through the chat template; for Mistral, its own tokenizer, configuration and weight format, with image input disabled. The server loads the local snapshot of the revision named above and has no access to the model repository while it runs. The libraries are installed in the versions of the first host (vLLM 0.19.0, PyTorch 2.10.0, Transformers 5.5.4, Tokenizers 0.22.2, mistral-common 1.11.0); the Python interpreter is 3.12 in place of 3.10.

For each backbone a serving record is written before its first call, `results/replication_<tag>/serving_record.json`: the system of the host, the accelerator and its driver, the library versions, the command that started the server, the answer of its model list, and the comparison of the snapshot's files with `results/model_records.json` and, for the files that record does not cover (the chat template of Gemma among them), with `results/replication/first_host_snapshot_files.json`. The record is written by `code/replication_serving_record.py`.

## 4. Arms, wordings and requests

As in Section 4 of the first plan, for each backbone:

| Wording | Arms | Patients |
|---|---|---|
| 1 | own; donor under d1, d2 and d3; patient-blind in blocks 1 and 2 | blocks 1 to 10 (100 patients), 126,000 calls |
| 2 | own; donor under d1; patient-blind in blocks 1 and 2 | blocks 1 to 4 (the first 40 patients), 30,000 calls |

The collector is `code/replication_probe.py`, unchanged. Requests are as in the first plan: temperature 0, at most 160 output tokens, one system message and one user message, 24 concurrent requests. Gemma receives the keyword that disables reasoning, as the main backbone did; Mistral, whose chat template has no such switch, receives plain chat requests (`--plain_chat`), as in its earlier collections.

The archives are `results/replication_b3/w1`, `results/replication_b3/w2`, `results/replication_b2/w1` and `results/replication_b2/w2`.

## 5. Collection order, checks before the first call, and stopping

Mistral is collected first and Gemma second, so that one backbone is complete as early as possible. Within a backbone the order is that of the first plan: wording 1, blocks 1 to 4; wording 2, blocks 1 to 4; wording 1, blocks 5 to 10. Inside a block every call of every arm is sent in one seeded random order. A call is final once the server has answered it. Transport errors are sent again once at the end of the block. An answer that cannot be parsed is missing in its cell and is not imputed.

Before the first call of a backbone on the sample, 600 patient-blind prompts of an earlier design (`results/repr`, through the collector's `--bench` option) are sent to measure the rate and to see that answers come back and parse. Nothing of them is stored, and no prompt of the replication sample is among them. If fewer than 95% of these answers parse, the collection of that backbone does not start. Such a failure is repaired on the serving side only; the prompts, the parser and the sample stay as they are, and the repair is recorded under Deviations.

There is no limit by the clock. The second host is on loan for a time that is not known in advance. If it is withdrawn, or if the server stops, the collection stops where it is and is continued when that is possible; the collector keeps every answer already received. Only completed blocks are analyzed, as in the first plan. A backbone without a completed block of wording 1 is reported as not collected.

## 6. Analysis

Each backbone is analyzed separately, by the code of the first plan. `code/replication_backbones.py` places a copy of the backbone's archive under the directory names that `code/replication_stats.py` reads and runs that script unchanged; the output is stored as `results/replication_frozen_<tag>.json`. Run with `--selftest` on the archive of the main backbone, this route reproduces `results/replication_frozen.json`.

**Primary analysis, for each backbone.** That of Section 6 of the first plan: wording 1, donor schedule d1, all completed blocks; the mean over patients of the paired difference, own minus donor, in per-patient AUROC against the shared label; its two-sided 90% Student interval; and the two one-sided tests of equivalence at the margins 0.02 and 0.05.

**Decision rules, for each backbone.** Those of the first plan:

1. Equivalence within a margin is declared when both one-sided tests reject at 0.05, that is, when the 90% interval lies inside the margin.
2. If the 90% interval excludes zero, a difference is reported with its sign, whether or not it lies inside a margin.
3. The result is reported in the main text whatever it is.

The two backbones are two questions. No test is adjusted for the other, no estimate is pooled over backbones, and no claim is made about "at least one" backbone. The primary estimates of the three backbones are shown side by side, descriptively.

**The printed-number reference, named here in advance.** For the main backbone this analysis was added after the outcome had been read. For the two backbones of this plan it is fixed before the first call. `replication_posthoc.one_design`, unchanged, ranks each patient's reactions by the printed percentile, and by the printed value, with no model, and computes the same paired own-minus-donor difference in per-patient AUROC; these two differences depend on the sample only, and with all ten blocks complete they are the ones already reported for the main backbone. For each backbone it gives the paired difference between the model's own-minus-donor difference and that of the printed percentile, with its 90% interval, and the correlation of the two across patients. The output is `results/replication_reference_<tag>.json`. The reading is fixed as follows, on donor schedule d1:

- if the 90% interval of "model minus printed percentile" includes zero, the backbone's difference is described as not distinguishable from what ranking the printed percentile gives;
- if the interval lies above zero, as larger than that; if it lies below zero, as smaller.

The same quantity under d2 and d3 is reported beside it.

**Secondary analyses**, descriptive and without correction for multiplicity: those listed in Section 6 of the first plan (the contrast under d2 and d3 and the pooled contrast, the crossed bootstrap, equivalence at 0.01 as exploratory, the contrast under wording 2, the comparison of wordings, and the AUROC of every arm beside the raw-expression ranking and the availability indicator, with the share of identical answers and the residual correlation between the own and donor arms).

## 7. Deviations

None at the time of the commit.
