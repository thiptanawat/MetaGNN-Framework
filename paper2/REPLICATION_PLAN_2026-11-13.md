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
