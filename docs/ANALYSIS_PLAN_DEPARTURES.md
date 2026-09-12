# Departures from the analysis plan, and the chronology

`ANALYSIS_PLAN_2026-09-10.md` is the plan the external test of the language-model study was run
under. This file records where the analysis as reported departs from the plan as written, what
changed after the collections for a reason the plan did not anticipate, which of the plan's
secondary studies were not run, and what the released record can and cannot show about when the
plan was fixed. It is the log the manuscript's supplement summarizes; the two say the same thing.

## Chronology

| event | when | evidence |
|---|---|---|
| plan written | 10 September 2026, by the authors' report; the date is the one in the file name | the local file; no registry entry; the file name dates neither the fixing of its contents nor their order relative to collections made the same day |
| external collections made | 10 September 2026, 13:35 to 16:04 UTC | `started_at` in each `paper2/results/msi2/*/design.json`; `paper2/results/msi2/deployment_records.json` |
| plan first independently inspectable | 11 September 2026, in the public release tagged `journal-2026-09` | the repository history |

The authors report that the plan was fixed before collection. Its first independently
inspectable public version postdates collection, so the timing cannot be independently verified
from the released record. The manuscript therefore does not call the external test preregistered
or confirmatory in the registered sense; the Mistral zero-shot own-minus-donor contrast on
GSE39582 is the designated primary external contrast, designated, by the authors' account, before
the external outcomes were read. This does not establish that the analysis was selected after
seeing results, and the absence of a public preregistration does not by itself invalidate the
study; it limits what a reader can check.

## Departures from the plan as written

| plan specification | what was done | why |
|---|---|---|
| Section 2.5: one population interval for the primary contrast, redrawing the donor derangement inside each resample | the primary contrast is the contrast on the donor responses actually served (the collected arm), whose recipient-pair bootstrap range fixes the assignment; the plan's redraw-inside-resampling procedure is applied to the cached reassignment, reported beside it as a sensitivity | the served and cached donor answers differ (20 of 519 calls on the first external cohort, by up to 0.37); what was actually served is the primary. Both ranges are descriptive resampling ranges with recipient-donor dependence uncorrected and carry no nominal coverage; the formal test is the conditional permutation test |
| Section 2.5: permute the outcome within site or platform blocks where those exist | the permutation is over the whole cohort | each external cohort is a single array platform, and the deposited metadata parsed carry no per-sample center of origin, so no block variable was available |
| Section 2.6: a sensitivity assigning every failed output the development prevalence | not computed | no call failed in any analyzed collection (`failed_calls` is zero throughout), so the sensitivity has nothing to assign |

## A change from the initial implemented analysis

The plan specifies the donor construction (a uniform derangement) and the resampling, but it does
not say whether the donor scores are the served replies to the donor prompts or the recipient's
own-prompt replies reassigned through the schedule. The analysis as first implemented used the
reassignment, on the argument that an identifier-free prompt is byte-identical to the donor's own
prompt and therefore returns the same answer. It does not always (serving is not
bit-deterministic), and once that was seen the served replies became the primary donor arm and the
reassignment became the sensitivity. This is a change from the initial implemented and reported
analysis, not a departure from an explicit plan instruction. It lowers the reported contrast
(+0.1125 served against +0.1162 reassigned).

## Secondary studies in the plan that were not run

| plan item | status |
|---|---|
| Section 2.7, the factorial (record only; availability only; raw only; percentile only; both) with the identifier intervention | not collected; an unrun secondary study, not an omission the manuscript reports on |
| Section 2.5, the identical-prompt repeat in a later session | not run; the within-session repeat was made in every collection (`own_repeat` in the archives) and is reported, so every serving-variation figure is within-session |

## Extensions beyond the plan

| item | status |
|---|---|
| the tool-assisted configuration on the first and second backbones | the plan specifies the tool-assisted configuration for the third backbone's interface, which was collected on all three cohorts; the same configuration was also attempted on the other two backbones as an optional extension, and those two collections were interrupted on the development cohort (900 and 800 of 1,586 responses) and never sent to the external cohorts. They are listed in the results ledger and excluded for that reason and no other |

## Followed as written, with a note

Section 2.8, calibration maps with a fitted slope at or below zero reported as a constant map: an
earlier version of the code applied a fitted negative slope; it was corrected before any
calibrated figure was reported, and the results file keeps the fitted slope beside the applied one.
Three of seven maps fell to the constant. No calibrated figure is promoted in the manuscript.

None of the departures or the change was made after reading an outcome in a direction that favored
the arm: the block permutation is a data limitation, and the interval departure and the primary-arm
change follow from one decision, to treat what was actually served as primary, which lowers the
reported contrast.

## What is unchanged from the plan

The interface (percentile-only panel), the panel restriction to reactions coverable on the
external platform, the endpoint harmonization (instability-high against everything else), the
donor construction (a uniform derangement of the evaluated patients drawn without reference to the
phenotype, seed 11 primary, 22, 33, 44, 55 for the schedule spread), the one-sided conditional
permutation test with 10,000 draws, the single designated primary contrast (the third backbone's
zero-shot own-minus-donor contrast on the first external cohort) with every other row secondary
and no multiplicity adjustment applied to the primary one, and the smallest effect of interest and
the wording reserved for each outcome.
