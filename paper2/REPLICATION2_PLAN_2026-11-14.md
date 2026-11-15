# Analysis plan: a second replication of the own-versus-donor contrast, with the printed-number reference named in advance

Date: 14 November 2026. This plan, the sample, `code/replication2_design.py`, `code/replication2.py`, `code/replication_service_record.py`, the launchers `code/provenance/run_replication2_service.sh` and `code/provenance/run_replication2_local.sh`, and the two one-option changes named in Section 5 were committed to the public repository in one commit before the first call on this sample was issued. Nothing below was changed after that commit; deviations, if any, are listed at the end under a dated heading.

The plan follows `REPLICATION_PLAN_2026-11-13.md` (below, the first plan) and `REPLICATION_BACKBONES_PLAN_2026-11-13.md` (its extension to two further backbones). It was written after the outcomes of both were known. No backbone had been sent a prompt of the sample of this plan when it was written.

## 1. Question

On the first replication sample the paired difference between the own arm and the donor arm in per-patient AUROC lay inside the 0.02 margin for all three backbones, and for the main backbone its 90% interval excluded zero (+0.0061, +0.0002 to +0.0120; 100 patients). Ranking the printed percentile with no model gave a difference of the same sign on the same prompts. For the main backbone that comparison was added after the outcome had been read. For the other two backbones the extension named it in advance, on one donor schedule, and neither difference was distinguishable from that of the printed percentile.

This replication asks, on reactions and patients that no earlier collection used, and for each of the three backbones:

1. whether the own-minus-donor difference again lies within the margins of the first plan, and
2. whether it is equivalent, within a margin fixed here, to the difference that the printed percentile gives with no model, when both are averaged over the donor schedules.

For the main backbone the second question is put under a plan for the first time.

## 2. What was known when this plan was written

The reading of Section 7 was applied to the first sample, whose collections were complete and public, by the script this plan fixes (`python3 code/replication2.py --first-sample`; output `results/replication2/first_sample_reference.json`).

| First sample, 100 patients, 90% interval in parentheses | Own minus donor, schedule d1 (as published) | Own minus donor, mean of the three schedules | Model minus printed percentile, mean of the three schedules | Standard deviation over patients of the last column |
|---|---|---|---|---|
| Main backbone (Qwen3.8-27B) | +0.0061 (+0.0002 to +0.0120) | +0.0056 (+0.0005 to +0.0108) | +0.0016 (-0.0012 to +0.0044) | 0.0170 |
| Second backbone (Gemma 4 31B-it) | +0.0015 (-0.0042 to +0.0071) | +0.0035 (-0.0014 to +0.0083) | -0.0006 (-0.0034 to +0.0022) | 0.0169 |
| Third backbone (Mistral Small 3.2 24B Instruct) | +0.0042 (-0.0041 to +0.0125) | +0.0060 (-0.0013 to +0.0134) | +0.0020 (-0.0029 to +0.0069) | 0.0298 |

Ranked with no model, the printed percentile gave +0.0040 (+0.0007 to +0.0073) on the mean of the three schedules; this number depends on the sample only and is the same for every backbone. On the first sample the reading of Section 7 declares equivalence within 0.01 for all three backbones, and no interval of the last contrast excludes zero.

These numbers come from data already collected. They motivated the pooling over schedules, the margin and the size of the sample. They are not evidence for the question of this plan: the reading was chosen with them in view, which is why it is put to a new sample here.

## 3. Sample

| Item | Rule |
|---|---|
| Reactions | 300, drawn at random (seed 20261115) from the 9,350 annotated reactions that no earlier collection used: the 661 reactions of the 15 archived designs and the 300 reactions of the first replication sample are excluded |
| Patients | an ordered list of 200, drawn with the same generator from the 449 patients that no earlier collection used: the 55 patients of the archived designs and the whole ordered list of 120 of the first replication sample are excluded, whether or not a patient of that list was called |
| Donor schedules | three (d1, d2, d3); each draws one donor per patient and reaction, uniformly from the 623 other patients of the cohort (seeds 20261116, 20261117, 20261118) |
| File | `results/replication2/design_master.json`, written by `code/replication2_design.py`; SHA-256 of the file `eb1df4423b5e16d2b4887ad0a5bab42e7e99a4123d5ea3ab5025b22bcae546db` |

The seed was fixed before the draw and the sample was drawn once. The collection goes through the patient list in blocks of 10, in the order of the draw.

The contrast of the printed percentile depends on the sample only and needs no model. It was not computed on this sample before this plan was committed. The analysis script was tested on a sample drawn with another seed and on invented answers.

The sample has 200 patients, twice the first replication. With the standard deviations of the last column of Section 2, the 90% interval of "model minus printed percentile" is expected to have a half-width of about 0.0020 for the main backbone, 0.0020 for the second and 0.0035 for the third, against a margin of 0.01.

## 4. Backbones and serving

| Tag | Backbone | Served as | Where |
|---|---|---|---|
| b1 | Qwen3.8-27B (main backbone) | `qwen3-8-27b`, the service that issued the first replication | first host |
| b3 | Mistral Small 3.2 24B Instruct, `mistralai/Mistral-Small-3.2-24B-Instruct-2506`, revision `95a6d26c4bfb886c58daf9d3f7332c857cb27b43` | `mistral-small-3.2-24b-instruct`, one local vLLM process | second host |
| b2 | Gemma 4 31B-it, `google/gemma-4-31B-it`, revision `439edf5652646a0d1bd8b46bfdc1d3645761a445` | `gemma-4-31b-it`, one local vLLM process | second host |

**Main backbone.** The service reports its own configuration. The record read before the first call is reduced by `code/replication_service_record.py` to the fields that describe the model and the engine and archived as `results/replication2_b1/serving_record.json`; run on the records saved before the first replication, that script reproduces `results/replication/serving_config.json`. If the record names a model source other than `Qwen/Qwen3.8-27B-FP8`, the main backbone is not collected. The collection starts only while the service reports two healthy replicas.

**Second and third backbones.** As in the extension of the first plan: the revisions that `results/model_records.json` records, every file of the downloaded snapshot hashed and compared with that record before the backbone is served, a backbone whose files differ not collected; one local vLLM process on one accelerator (NVIDIA H100, 80 GB) of the second host with the serving flags of the extension (`code/provenance/run_replication_backbones.sh`); a serving record written before the first call, `results/replication2_<tag>/serving_record.json`, by `code/replication_serving_record.py`.

Requests are as in the first plan: temperature 0, at most 160 output tokens, one system message and one user message, 24 concurrent requests. The main backbone and Gemma receive the keyword that disables reasoning; Mistral, whose chat template has no such switch, receives plain chat requests.

## 5. Arms, wording and collector

| Wording | Arms | Patients | Calls per backbone |
|---|---|---|---|
| 1, the wording of the earlier collections | own; donor under d1, d2 and d3; patient-blind in blocks 1 and 2 | blocks 1 to 20 (200 patients) | 246,000 |

The second wording of the first plan is not collected again. Its question, whether a paraphrase moves the answers, was put to all three backbones on the first sample and is not a question of this plan.

The collector is `code/replication_probe.py` with one option added, `--master`, which names the sample file. With its default the collector prints, for the first sample, the prompts that the published collector prints (compared on the 15,000 prompts of block 1 of each wording). `code/replication_serving_record.py` gained the option `--plan`, which sets the plan named in the record. Nothing else in the two scripts changed.

The archives are `results/replication2_b1/w1`, `results/replication2_b2/w1` and `results/replication2_b3/w1`.

## 6. Collection order, checks before the first call, and stopping

The three backbones are collected independently. The main backbone is collected on the first host. Mistral and then Gemma are collected on the second host, when its accelerator is free of the training jobs of the other study. The two hosts may collect at the same time.

Within a backbone the blocks are collected in order, 1 to 20. Inside a block every call of every arm is sent in one seeded random order. A call is final once the server has answered it. Transport errors are sent again once at the end of the block. An answer that cannot be parsed is missing in its cell and is not imputed.

Before the first call of a backbone on the sample, 600 patient-blind prompts of an earlier design (`results/repr`, through the collector's `--bench` option) are sent to measure the rate and to see that answers come back and parse. Nothing of them is stored, and no prompt of the sample is among them. If fewer than 95% of these answers parse, the collection of that backbone does not start. Such a failure is repaired on the serving side only; the prompts, the parser and the sample stay as they are, and the repair is recorded under Deviations.

**Night window of the service.** Other clients use the service of the main backbone. A block of the main backbone is started only between 22:00 and 06:00 local time (UTC+7). A block still running at 06:30 is left incomplete, and the collection goes on at 22:00 with the calls of that block still to be sent; the answers already received are kept. While the service does not report two healthy replicas, a watchdog pauses the collector and resumes it one minute after the service is healthy again. The watchdog reads the status of the service and changes nothing in it.

**Second host.** There is no limit by the clock. The host is on loan for a time that is not known in advance. If it is withdrawn, or if the server stops, the collection stops where it is and is continued when that is possible.

**Stopping.** The collection of a backbone ends when its 20 blocks are complete. If a host is withdrawn before that, the collection of its backbones ends where it is. The decision to end a collection depends on the clock and on the availability of a host, never on an answer. No statistic of Section 7 is computed on the archive of a backbone before the collection of that backbone has ended. Only completed blocks are analyzed, and their number is reported. A backbone without a completed block is reported as not collected.

## 7. Analysis

`code/replication2.py` computes everything, each backbone separately.

**Primary analysis, for each backbone.** That of the first plan, by its code: `replication_stats.main()` is run unchanged on a copy of the archive and of the sample placed under the directory names it reads, and its output is stored as `results/replication2_frozen_<tag>.json`. Wording 1, donor schedule d1, all completed blocks: the mean over patients of the paired difference, own minus donor, in per-patient AUROC against the shared label; its two-sided 90% Student interval; and the two one-sided tests of equivalence at the margins 0.02 and 0.05. Run on the three archives of the first sample, this route reproduces the published `results/replication_frozen.json`, `results/replication_frozen_b2.json` and `results/replication_frozen_b3.json` (`--selftest`).

Decision rules, those of the first plan:

1. Equivalence within a margin is declared when both one-sided tests reject at 0.05, that is, when the 90% interval lies inside the margin.
2. If the 90% interval excludes zero, a difference is reported with its sign, whether or not it lies inside a margin.
3. The result is reported in the main text whatever it is.

**The printed-number reference, for each backbone.** `replication_posthoc.one_design`, unchanged, ranks each patient's reactions by the printed percentile with no model and gives the same paired own-minus-donor difference in per-patient AUROC under each donor schedule. For every patient, the model's own-minus-donor difference is averaged over the three schedules, the difference of the printed percentile is averaged over the same three schedules, and the second is subtracted from the first. The analysis gives the mean of this quantity over patients, its two-sided 90% Student interval, and the two one-sided tests of equivalence at the margin 0.01. The output is `results/replication2_reference_<tag>.json`.

The reading is fixed as follows.

1. Equivalence of the model's difference and that of the printed percentile within 0.01 AUROC is declared when both one-sided tests reject at 0.05, that is, when the 90% interval of "model minus printed percentile" lies inside the margin.
2. If that interval includes zero, the two are described as not distinguishable. If it lies above zero, the model's difference is described as larger than that of the printed percentile; if it lies below zero, as smaller. This is reported whether or not equivalence is declared.
3. The result is reported in the main text whatever it is.

The margin 0.01 is half the narrow margin of the primary analysis; the first plan listed it as exploratory for the primary contrast. The mean of the three schedules replaces the single schedule of the extension because on the first sample the three schedules gave +0.0026, +0.0001 and +0.0022 for the main backbone, so one schedule is one draw of the donors.

The three backbones are three questions. No test is adjusted for the others, no estimate is pooled over backbones, and no claim is made about "at least one" backbone. The estimates of the three backbones are shown side by side, descriptively.

**Secondary analyses**, descriptive and without correction for multiplicity:

- "model minus printed percentile" under each single donor schedule, as the extension reported it, and equivalence at a margin of 0.005 on the mean of the three schedules, as exploratory;
- the own-minus-donor difference of the printed percentile and of the printed value with no model, with their intervals, and the correlation across patients between the model's difference and that of the printed percentile;
- those of the first plan that this collection supports: the primary contrast under d2 and d3, the range of the three means and the pooled contrast; the crossed bootstrap of the primary contrast over patients and reactions; equivalence of the primary contrast at 0.01, as exploratory; the AUROC of every arm beside the raw-expression ranking and the availability indicator, the share of identical answers, the residual correlation between the own and donor arms, and the determinism floor of repeated patient-blind prompts;
- for each backbone, the mean of "model minus printed percentile" over the patients of both replication samples together, as a description; no decision rests on it.

## 8. Deviations

None at the time of the commit.

### Record after the collection (15 November 2026)

The commit that holds this plan, the sample, the analysis script and the launchers is `e825337`. It was pushed at 14:55:49 UTC on 14 November 2026 (21:55:49 local time, UTC+7; the push is in the public activity record of the repository). The first call on the sample was sent at 15:00:53 UTC (22:00:53 local time), 5 minutes after the push, to the main backbone. All three collections are complete: for each backbone 20 of the 20 blocks, 200 patients and 246,000 calls, with the arms and the order of Sections 5 and 6. The sample file that the collectors read has the SHA-256 that Section 3 names. No statistic of Section 7 was computed on the archive of a backbone before the collection of that backbone had ended.

| | Qwen3.8-27B (b1) | Mistral Small 3.2 24B Instruct (b3) | Gemma 4 31B-it (b2) |
|---|---|---|---|
| Served by | the service of the first host; its record, read at 15:00:14 UTC, names the model source `Qwen/Qwen3.8-27B-FP8` and two healthy replicas | one local vLLM process on the second host; 17 files of the snapshot compared with `results/model_records.json`, all identical | one local vLLM process on the second host; 8 files of the snapshot compared with `results/model_records.json`, all identical |
| Serving record | read 15:00:14 UTC, before the first call | written 21:41:56 UTC, before the first call | written 23:18:49 UTC, before the first call |
| Check calls before the collection (600 patient-blind prompts of an earlier design) | 600 of 600 parsed | 600 of 600 parsed | 600 of 600 parsed |
| First call on the sample, last reply (local time) | 22:00:53 on 14 November, 02:15:00 on 15 November | 04:42:17 on 15 November, 06:15:56 on 15 November | 06:19:31 on 15 November, 10:53:36 on 15 November |
| Calls answered per second over that time | 16.1 | 43.8 | 15.0 |
| Calls sent a second time; calls left without an answer; answers that could not be parsed | 0; 0; 0 | 0; 0; 0 | 0; 0; 0 |

No departure from the plan is recorded. The following points are recorded beside it; none depends on an answer of a model.

| Point | What happened |
|---|---|
| Answer files | The answer file of every backbone (w1/raw.json) is committed compressed, as w1/raw.json.gz (gzip), because the file of the main backbone is larger than the 100 MB that the hosting service accepts for one file; reproduce.sh unpacks the three files before anything reads them, and the SHA-256 that code/collection_manifest.py records is that of the unpacked file. |
| Main backbone | 20 of 20 blocks, no call sent again, no transport error left and no answer that could not be parsed; the collector ended with status 0 at 2026-11-15T02:15:04+07:00. |
| Third backbone | 20 of 20 blocks, no call sent again, no transport error left and no answer that could not be parsed; the collection ended with status 0 at 2026-11-15T06:16:00+07:00. |
| Second backbone | 20 of 20 blocks, no call sent again, no transport error left and no answer that could not be parsed; the collection ended with status 0 at 2026-11-15T10:53:40+07:00. |
| Second and third backbones | The launcher issued on the second host is code/provenance/run_replication2_local_acc1.sh, which is code/provenance/run_replication2_local.sh with the wait for the training jobs of the other study written for the accelerator it serves on alone. The published launcher had been waiting since 14 November for the training jobs of the whole host and was replaced on 15 November at 02:35 local time, before it had issued anything, because the other study by then also ran training jobs on the other accelerator of that host. Section 6 of the plan names the condition as the accelerator being free of those jobs. Nothing after the wait differs between the two files, and the calls were issued from the checkout of the plan's commit. |

The course of the collection is in `results/replication2/record.json`, which `code/replication2_record.py` writes from the archives, the serving records and `results/replication2/course.json`.

### Outcome

| Quantity (200 patients; 90% interval in parentheses) | Main backbone (Qwen3.8-27B) | Second backbone (Gemma 4 31B-it) | Third backbone (Mistral Small 3.2 24B Instruct) |
|---|---|---|---|
| Primary: own minus donor, donor schedule d1 | +0.0001 (-0.0038 to +0.0039) | +0.0007 (-0.0032 to +0.0047) | -0.0027 (-0.0091 to +0.0037) |
| Own arm, donor arm under d1 (AUROC) | 0.6157, 0.6156 | 0.5717, 0.5709 | 0.6125, 0.6152 |
| Own minus donor, donor schedule d2 | -0.0019 (-0.0059 to +0.0020) | -0.0022 (-0.0064 to +0.0020) | -0.0050 (-0.0115 to +0.0015) |
| Own minus donor, donor schedule d3 | -0.0000 (-0.0041 to +0.0040) | -0.0003 (-0.0048 to +0.0043) | -0.0029 (-0.0093 to +0.0036) |
| Own minus donor, mean of the three schedules | -0.0006 (-0.0041 to +0.0029) | -0.0006 (-0.0044 to +0.0032) | -0.0035 (-0.0094 to +0.0023) |
| Patient-blind arm on the first two blocks (20 patients; AUROC) | 0.5453 | 0.4826 | 0.5987 |
| Raw-expression ranking, availability indicator (AUROC) | 0.6988, 0.6608 | 0.6988, 0.6608 | 0.6988, 0.6608 |
| Printed percentile ranked with no model: own minus donor, mean of the three schedules | -0.0015 (-0.0042 to +0.0012) | -0.0015 (-0.0042 to +0.0012) | -0.0015 (-0.0042 to +0.0012) |
| Reference of Section 7: model minus printed percentile, mean of the three schedules | +0.0009 (-0.0009 to +0.0026) | +0.0009 (-0.0012 to +0.0031) | -0.0020 (-0.0060 to +0.0020) |
| Model minus printed percentile, donor schedule d1 | +0.0008 (-0.0011 to +0.0028) | +0.0015 (-0.0008 to +0.0038) | -0.0019 (-0.0063 to +0.0024) |
| Model minus printed percentile, donor schedule d2 | +0.0003 (-0.0018 to +0.0024) | +0.0000 (-0.0023 to +0.0023) | -0.0027 (-0.0072 to +0.0017) |
| Model minus printed percentile, donor schedule d3 | +0.0015 (-0.0006 to +0.0035) | +0.0012 (-0.0014 to +0.0038) | -0.0014 (-0.0057 to +0.0029) |
| Correlation across patients between the two differences | 0.86 | 0.83 | 0.80 |
| Model minus printed percentile over both replication samples (300 patients) | +0.0011 (-0.0004 to +0.0026) | +0.0004 (-0.0013 to +0.0021) | -0.0007 (-0.0038 to +0.0025) |

The contrast of the printed percentile depends on the sample only and is the same in every column.

Decision rules of Section 7, for each backbone separately:

- **Main backbone (Qwen3.8-27B).** Primary analysis: (1) the 90% interval lies inside the margin 0.02 and inside the margin 0.05, so equivalence is declared at 0.02 and at 0.05; (2) the interval includes zero, so no difference is reported; (3) the result is reported in the main text of the manuscript. Printed-number reference: (1) the 90% interval of model minus printed percentile lies inside the margin 0.01, so equivalence within 0.01 AUROC is declared; (2) it includes zero, so the two differences are described as not distinguishable; (3) the result is reported in the main text. At the exploratory margin 0.005, equivalence would be declared.
- **Second backbone (Gemma 4 31B-it).** Primary analysis: (1) the 90% interval lies inside the margin 0.02 and inside the margin 0.05, so equivalence is declared at 0.02 and at 0.05; (2) the interval includes zero, so no difference is reported; (3) the result is reported in the main text of the manuscript. Printed-number reference: (1) the 90% interval of model minus printed percentile lies inside the margin 0.01, so equivalence within 0.01 AUROC is declared; (2) it includes zero, so the two differences are described as not distinguishable; (3) the result is reported in the main text. At the exploratory margin 0.005, equivalence would be declared.
- **Third backbone (Mistral Small 3.2 24B Instruct).** Primary analysis: (1) the 90% interval lies inside the margin 0.02 and inside the margin 0.05, so equivalence is declared at 0.02 and at 0.05; (2) the interval includes zero, so no difference is reported; (3) the result is reported in the main text of the manuscript. Printed-number reference: (1) the 90% interval of model minus printed percentile lies inside the margin 0.01, so equivalence within 0.01 AUROC is declared; (2) it includes zero, so the two differences are described as not distinguishable; (3) the result is reported in the main text. At the exploratory margin 0.005, equivalence would not be declared.

No estimate is pooled over backbones and no test is adjusted for the others. Every quantity of Section 7 is in `results/replication2_frozen_<tag>.json` and `results/replication2_reference_<tag>.json`.
