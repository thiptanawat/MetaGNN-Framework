# Analysis plan: replication of the own-versus-donor contrast

Date: 13 November 2026. This plan, the sample, the collector and the analysis code were committed to the public repository in one commit before the first call of the replication was issued. Nothing below was changed after that commit; deviations, if any, are listed at the end under a dated heading.

## 1. Question

In Design E (40 patients, 300 reactions, one backbone) the paired difference between the own arm and the donor arm in per-patient AUROC lay within 0.02 of zero. This replication asks whether that holds

1. on reactions and patients that no earlier collection of the study used,
2. under three donor schedules collected in one session,
3. under a second wording of the prompt, and
4. on a later deployment of the same backbone whose serving configuration is recorded.

## 2. Sample

| Item | Rule |
|---|---|
| Reactions | 300, drawn at random (seed 20261113) from the 9,650 annotated reactions that no archived reaction-activity collection used; the 661 reactions of the 15 archived designs are excluded |
| Patients | an ordered list of 120, drawn with the same generator from the 569 patients that no archived reaction-activity collection used; 55 are excluded |
| Donor schedules | three (d1, d2, d3); each draws one donor per patient and reaction, uniformly from the 623 other patients (seeds 20261114, 20261115, 20261116) |
| File | `results/replication/design_master.json`, written by `code/replication_design.py`; SHA-256 of the file `2f39d5fc87a1cce2fe210e457f35ad8120d464c04968c42d9be48bda32bc6869` |

The collection goes through the patient list in blocks of 10, in the order of the draw.

## 3. Backbone and serving

The main backbone of the study, `qwen3-8-27b`, as served on the collection host on 13 November 2026. The service reports its own configuration, and the record read before the first call is archived as `results/replication/serving_config.json`: model source `Qwen/Qwen3.8-27B-FP8`, two replicas, Ray 2.55.0, and the engine settings. The deployment was started on 25 October 2026, after every earlier collection, so it is a different deployment of the model. Requests are as in the earlier collections: temperature 0, at most 160 output tokens, reasoning disabled, one system message and one user message.

## 4. Arms and wordings

| Wording | Text | Arms | Patients |
|---|---|---|---|
| 1 | The wording of the earlier collections. The collector takes the system prompt, the prompt builder and the answer parser from the source of `code/llm_probe.py` | own; donor under d1, d2 and d3; patient-blind in blocks 1 and 2 | blocks 1 to 10 (100 patients) |
| 2 | A paraphrase that carries the same fields and the same two numbers in the same formats; fixed in `code/replication_probe.py` | own; donor under d1; patient-blind in blocks 1 and 2 | blocks 1 to 4 (the first 40 patients) |

The patient-blind arm repeats one prompt per patient, so two blocks (20 repeats of each prompt) are enough to measure the determinism floor.

## 5. Collection order and stopping rule

1. Wording 1, blocks 1 to 4.
2. Wording 2, blocks 1 to 4.
3. Wording 1, blocks 5 to 10.

Inside a block, every call of every arm is sent in one seeded random order, with 48 concurrent requests. A call is final once the server has answered it. Transport errors are sent again once at the end of the block. An answer that cannot be parsed is missing in its cell and is not imputed.

No block is started after 04:10 local time (UTC+7) on 13 November 2026, and a block still running at 04:40 is abandoned. Only completed blocks are analyzed. These limits depend on the clock and on no answer.

Before this plan was fixed, 1,920 patient-blind prompts of an earlier design (`results/repr`) were sent to the service to measure its rate (11 to 23 calls per second). No prompt of the replication sample was sent.

## 6. Analysis

`code/replication_stats.py` computes everything and writes `results/replication_frozen.json`. Each own-versus-donor contrast is computed by `stats.analyze`, the function that computed it for the earlier designs.

**Primary analysis.** Wording 1, donor schedule d1, all completed blocks: the mean over patients of the paired difference, own minus donor, in per-patient AUROC against the shared label; its two-sided 90% Student interval; and the two one-sided tests of equivalence at the margins 0.02 and 0.05. These are the margins of the earlier analysis. There they were chosen after an earlier collection; here they are fixed before the first call.

**Decision rules.**

1. Equivalence within a margin is declared when both one-sided tests reject at 0.05, that is, when the 90% interval lies inside the margin.
2. If the 90% interval excludes zero, a difference is reported with its sign, whether or not it lies inside a margin.
3. The result is reported in the main text whatever it is.

**Secondary analyses**, descriptive and without correction for multiplicity:

- the same contrast under d2 and d3, the range of the three means, and the pooled contrast (own minus the mean of the three donor AUROCs);
- a crossed bootstrap of the primary contrast over patients and reactions (10,000 draws under each of the seeds 29, 1029 and 2029);
- equivalence at a margin of 0.01, as exploratory;
- the contrast under wording 2 on its patients;
- the comparison of wordings on the patients who have both: the mean absolute change of the own-arm answer between wordings on expression-bearing reactions, beside the mean absolute own-versus-donor change within a wording and the determinism floor of repeated patient-blind prompts; the paired difference in own-arm AUROC between wordings;
- the AUROC of every arm beside the raw-expression ranking and the availability indicator on the same reactions, the share of identical answers, and the residual correlation between the own and donor arms.

## 7. Deviations

None at the time of the commit.

### Record after the collection (13 November 2026)

The collection ended at 02:06:55 on 13 November local time (UTC+7) with 10 of the 10 blocks of wording 1 and 4 of the 4 blocks of wording 2 complete: 100 patients under wording 1, 40 under wording 2, 156,000 calls. Four departures from the plan are recorded. None depends on an answer of the model.

| Departure | What happened |
|---|---|
| Dates | The plan is dated 13 November 2026, the day on which most of the collection was to run. Its commit (2026-11-12T16:19:14Z), the push of that commit (2026-11-12T16:19:16Z, from the public activity record of the repository) and the first call (2026-11-12T16:19:44Z, 28 seconds after the push) fall late on 12 November local time. The stopping times of Section 5 were kept as written. |
| A replica of the service failed | During block 1 of wording 1 one of the two replicas of the service failed its health check, and the service restarted it. For 274 seconds (23:23:43 to 23:28:17 on 12 November, local time) the requests in flight went unanswered and the collector could send 10 calls; 97 calls sent shortly afterwards returned a server error. The cause is not known to us; no log of the service was available. About two minutes before the failure a memory test of ours on one graphics processor of the host had ended with an out-of-memory error in its own process, and whether the two are connected is not known to us either. The calls that received a server error were sent again at the end of the block, as Section 5 provides. Over the whole collection 97 calls of wording 1 and 0 of wording 2 were sent a second time, and 0 remained without an answer. |
| Pause and concurrency | We stopped the collector after the failure and started it again once the service reported two healthy replicas, 109 seconds after its last call (23:29:49 to 23:31:38 on 12 November, local time), with 24 concurrent requests in place of 48, which it kept to the end. The answers already received were kept, and the block went on with the calls still to be sent, in the order of its seeded permutation. A watchdog, added at the restart, was to pause the collector whenever the service did not report two healthy replicas; it never had to act. The block record of block 1 gives the time of the restart as its start (see `block_times_note` in `results/replication/collection_record.json`). |
| Another client | Another client used the service during the collection. |

The course of the collection, block by block, is in `results/replication/collection_record.json`. Of the answers, 0 of wording 1 and 0 of wording 2 could not be parsed.

### Outcome

| Quantity | Value |
|---|---|
| Primary: own minus donor, wording 1, donor schedule d1 (100 patients) | +0.0061, 90% interval +0.0002 to +0.0120 |
| Own arm, donor arm (AUROC) | 0.5578, 0.5517 |
| Own minus donor, donor schedule d2 | +0.0052 (-0.0003 to +0.0107) |
| Own minus donor, donor schedule d3 | +0.0057 (-0.0001 to +0.0114) |
| Own minus the mean of the three donor arms | +0.0056 (+0.0005 to +0.0108) |
| Own minus donor, wording 2 (40 patients) | +0.0029 (-0.0042 to +0.0100) |

Decision rules of Section 6:

1. The 90% interval lies inside the margin 0.02 and lies inside the margin 0.05, so equivalence is declared at 0.02 and declared at 0.05.
2. The 90% interval excludes zero, so a difference is reported with its sign (own above donor).
3. The result is reported in the main text of the manuscript.

Every quantity of Section 6 is in `results/replication_frozen.json`.

### An analysis outside the plan (13 November 2026)

After the outcome above had been read, one analysis was added that this plan does not contain. It is exploratory, it changes nothing in `results/replication_frozen.json`, and the manuscript reports it as outside the plan. Script: `code/replication_posthoc.py`; output: `results/replication_posthoc.json`.

The donor of this design is redrawn at every reaction, so the donor arm prints a mosaic of patients and the own arm one patient's profile. The analysis ranks each patient's reactions by the printed percentile, or by the printed value, with no model, and computes the same paired own-minus-donor difference in per-patient AUROC.

| Quantity (wording 1, 90% interval) | Schedule d1 | Schedule d2 | Schedule d3 |
|---|---|---|---|
| Language model | +0.0061 (+0.0002 to +0.0120) | +0.0052 (-0.0003 to +0.0107) | +0.0057 (-0.0001 to +0.0114) |
| Printed percentile, no model | +0.0035 (-0.0002 to +0.0073) | +0.0050 (+0.0015 to +0.0086) | +0.0035 (-0.0004 to +0.0074) |
| Printed value, no model | +0.0012 (-0.0001 to +0.0024) | +0.0012 (+0.0001 to +0.0024) | +0.0011 (-0.0002 to +0.0024) |
| Model minus printed percentile | +0.0026 (-0.0007 to +0.0058) | +0.0001 (-0.0030 to +0.0033) | +0.0022 (-0.0010 to +0.0053) |
| Correlation across patients, model with printed percentile | 0.868 | 0.834 | 0.858 |
