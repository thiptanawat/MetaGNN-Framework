# Departures from the analysis plan, and the chronology

`ANALYSIS_PLAN_2026-09-10.md` is the plan the external test of the language-model study was run
under. This file records where the analysis as reported departs from the plan as written, why,
and what the public record can and cannot show about when the plan was fixed. It is the
log the manuscript's supplement summarizes; the two say the same thing.

## Chronology

| event | when | evidence |
|---|---|---|
| plan written | 10 September 2026 (the date in its file name) | the local file; no registry entry |
| external collections made | 10 September 2026, 13:35 to 16:04 UTC | `started_at` in each `paper2/results/msi2/*/design.json`; `paper2/results/msi2/deployment_records.json` |
| plan first in the public repository | 11 September 2026, in the release tagged `journal-2026-09` | the repository history |

The public history therefore shows the plan after the collections it governs. The local record
shows it before them, and it is the only record there is. The manuscript does not call the
external test preregistered; "confirmatory" there means that one contrast was declared before
the external outcomes were read, as the local record shows, not that the declaration was lodged
where a third party could have seen it at the time.

## Departures

| plan specification | what was done | why |
|---|---|---|
| Section 2.5: one population interval for the primary contrast, redrawing the donor derangement inside each resample | the primary contrast is the contrast on the donor responses actually served (the collected arm), whose recipient-pair bootstrap range fixes the assignment; the plan's redraw-inside-resampling procedure is applied to the cached reconstruction, reported beside it as a sensitivity | the served and cached donor answers differ (20 of 519 calls on the first external cohort, by up to 0.37); what was actually served is the primary. Both ranges are descriptive resampling ranges with recipient-donor dependence uncorrected and carry no nominal coverage; the formal test is the conditional permutation test |
| Section 2.5: permute the outcome within site or platform blocks where those exist | the permutation is over the whole cohort | each external cohort is a single array platform, and the deposited metadata parsed carry no per-sample center of origin, so no block variable was available |
| Section 2.5: the reassigned reconstruction is the donor arm; the collected donor calls are a determinism check | the roles are exchanged: the collected calls are the donor arm, the reconstruction is the sensitivity | the collected calls were found to differ from the cached ones, which is the same finding as in the first row; treating what was served as primary lowers the reported contrast |
| Section 2.5: serving variation measured by a 40-patient identical-prompt repeat within the session and in a later session | the within-session repeat was made in every collection (`own_repeat` in the archives) and is reported; no later-session repeat was made | it was not run; the omission is recorded rather than filled in, and the manuscript's serving-variation figures are within-session only |
| Section 2.6: a sensitivity assigning every failed output the development prevalence | not computed | no call failed in any analyzed collection (`failed_calls` is zero throughout), so the sensitivity has nothing to assign |
| Section 2.8: the tool-assisted configuration on the frozen interface | completed on the third backbone only; its collections on the first and second backbones were interrupted on the development cohort (900 and 800 of 1,586 responses) and never sent to the external cohorts | the accelerator was reclaimed; the two incomplete collections are listed in the results ledger and excluded for that reason and no other |
| Section 2.8: calibration maps, with a fitted slope at or below zero reported as a constant map | followed; an earlier version of the code applied a fitted negative slope, which was corrected before any calibrated figure was reported, and the results file keeps the fitted slope beside the applied one. Three of seven maps fell to the constant. No calibrated figure is promoted in the manuscript | a negative slope reverses the ranking rather than repairing the probabilities |

None of these was made after reading an outcome in a direction that favored the arm: the
block-permutation departure is a data limitation, the interval and role departures follow from one
decision, to treat what was actually served as primary, which lowers the reported contrast, and the
two omissions (the later-session repeat and the tool-assisted configuration on two backbones)
are secondary evidence left unmade, not evidence added.

## What is unchanged from the plan

The interface (percentile-only panel), the panel restriction to reactions coverable on the
external platform, the endpoint harmonization (instability-high against everything else), the
donor construction (a uniform derangement of the evaluated patients drawn without reference to the
phenotype, seed 11 primary, 22, 33, 44, 55 for the schedule spread), the one-sided conditional
permutation test with 10,000 draws, the single confirmatory contrast (the third backbone's
zero-shot own-minus-donor contrast on the first external cohort) with every other row secondary
and no multiplicity adjustment applied to the confirmatory one, and the smallest effect of
interest and the wording reserved for each outcome.
