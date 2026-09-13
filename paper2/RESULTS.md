# Language-Model Evidence results, final

Title: *Language-model annotation of metabolic reaction activity responds to a patient's transcriptome without gaining patient-specific information from it*

Every number below is emitted by `stats.py` into `results_frozen.json` and substituted into the manuscript as a LaTeX macro. Nothing is transcribed by hand.

## Setting

| quantity | value |
|---|---|
| Recon3D reactions | 10,600 |
| labeled active (independent reconstructions) | 3,379 (31.9%) |
| expression-bearing reactions | 5,635 |
| reactions carrying a GPR rule | 5,938 |
| patients | 624 |
| prevalence, expression-bearing vs rest | 40.8% vs 21.8% |
| raw-expression baseline, whole network | 0.6342 +/- 0.0058 |
| information-free indicator floor | 0.6085 |

## Main experiment

| design | reaction only | + own expression | + stranger's | raw expression | indicator floor |
|---|---|---|---|---|---|
| A, representative | 0.5188 [0.5169, 0.5208] | 0.5498 [0.5274, 0.5733] | 0.5470 [0.5344, 0.5592] | 0.6708 | 0.6239 |
| B, balanced | 0.4358 [0.4340, 0.4376] | 0.4481 [0.4329, 0.4617] | 0.4463 [0.4318, 0.4614] | 0.5115 | 0.5000 |
| E, representative, 40 patients | 0.5186 [0.5176, 0.5197] | 0.5590 [0.5501, 0.5673] | 0.5531 [0.5481, 0.5588] | 0.6724 | 0.6239 |

No arm in any design reaches the raw-expression baseline. In the representative design every arm also sits below the information-free indicator floor.

## The three controls

| quantity | Design A | Design B | Design E |
|---|---|---|---|
| determinism floor, identical answers (%) | 98.9 | 99.1 | 98.7 |
| determinism floor, mean |diff| | 0.0018 | 0.0017 | 0.0022 |
| between-patient sd, own data | 0.2599 | 0.2640 | 0.2777 |
| between-patient sd, stranger | 0.2693 | 0.2599 | 0.2832 |
| patient vs stranger, identical answers (%) | 19.0 | 21.9 | 19.3 |
| patient vs stranger, mean |diff| | 0.3142 | 0.3101 | 0.3108 |

## Presence versus magnitude (R2 of the returned score)

| design | identity | + presence + value | value-bearing only: identity | + the value |
|---|---|---|---|---|
| A | 0.236 | 0.388 | 0.038 | 0.150 |
| B | 0.291 | 0.425 | 0.053 | 0.168 |
| E | 0.243 | 0.399 | 0.044 | 0.172 |

The presence indicator and the value are collinear by construction, so no sequential share is attributed to either; the last two columns ask the question where it is well posed.

Mean returned p on reactions with no mapped gene, Design A: 0.141 without patient context, 0.171 with it. The model rebuilds the information-free indicator internally.

## Prompt format

| evidence written as | own profile | a stranger's | mean p, expression-bearing | mean p, no gene |
|---|---|---|---|---|
| a number plus its percentile (Design B, the reference for the format arms) | 0.4481 | 0.4463 | 0.518 | 0.154 |
| categorical | 0.4381 | 0.4392 | 0.542 | 0.153 |
| percentile | 0.4489 | 0.4452 | 0.479 | 0.152 |

## Positive controls (identical record format)

| model | echo the value | compare two values | threshold a value | modal answer share, threshold |
|---|---|---|---|---|
| qwen3-8-27b | 100.0% | 98.3% | 100.0% | 53.3% |
| second backbone | 100.0% | 100.0% | 100.0% | 53.3% |

## Patient-level target: microsatellite instability

579 patients (92 MSI-H, 487 MSS), a 40-reaction panel chosen by across-patient variance without consulting the label.

| method | AUROC | 95% CI |
|---|---|---|
| logistic regression, same features, 5-fold CV | 0.9241 | 0.8879, 0.9544 |
| language model, patient's own profile | 0.5178 | 0.4795, 0.5566 |
| language model, a stranger's profile | 0.4889 | 0.4514, 0.5271 |
| best single reaction, in sample (optimistic) | 0.9084 | |

The model returns 7 distinct probabilities across all 579 patients; 86.5% receive the value 0.12. Among the 501 patients who received it, there are 418 distinct free-text rationales.

Aggregation control (how many of the 40 reactions sit at or above the 75th percentile): Spearman 0.879 with the true count over 300 patients, exact on 15.7%, mean absolute error 3.41, predicted mean 6.79 against a true mean of 9.92.

## Reasoning-enabled arm

| arm | responses parsed | mean p of the parsed answers |
|---|---|---|
| reaction | 23.8% | 0.122 |
| patient | 35.3% | 0.316 |
| shuffled | 38.8% | 0.311 |

With reasoning enabled most calls returned an empty message because the whole token budget went to reasoning tokens the API does not return, and the attrition differs between arms in a way that is confounded with an endpoint outage. No accuracy is reported for this run, in the manuscript or here; it is a cost, attrition and answer-form report.

## References

Every reference in the manuscript was resolved against PubMed E-utilities, the Crossref API or the arXiv API before use; the procedure and its outcome are in `docs/VERIFICATION.md` at the repository root.
