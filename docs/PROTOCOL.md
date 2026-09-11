# Applying the checks to your own benchmark

A short recipe. It assumes you already have a model, a label vector and a
sample-level cross-validation, which is where most pipelines start.

## Step 0: is your benchmark exposed?

Ask one question: **does the target vary between samples?**

If every sample is scored against the same label vector, it does not, and the rest
of this document applies. This is the normal case when activity labels come from a
reference reconstruction, from a curated database, or from thresholding a cohort
statistic. It is also the case when labels vary only weakly, though the argument
below is then approximate rather than exact.

If the target does vary per sample, these checks are still informative but
the ceiling argument does not apply, and a sample-only split may be adequate.

## Step 1: prove the ceiling to yourself

Before touching your model, train a deliberate memorizer: one free parameter per
entity, no sample features at all. Under a sample-only split it should reach AUROC
1.0. If it does, your benchmark cannot distinguish a good method from a lookup
table, and every number previously reported on it is uninterpretable in isolation.

This costs minutes and it is the single most convincing thing you can run.

## Step 2: hold out the entity axis

    from metabench import entity_folds
    folds = entity_folds(y, has_input, n_folds=3, seed=2024)

Then, for each (sample fold, entity fold) cell:

- compute the **loss on training entities only**;
- select the model on validation samples crossed with **training entities**;
- report **held-out entities on test samples**.

A held-out entity's label must never enter training or selection. This is joint
row-and-column exclusion, long established for relational prediction; we are only
applying it on a new pair of axes.

## Step 3: report the gap, not just the score

    memorization_gap(scores, y, train_mask)

Report AUROC on training entities and on held-out entities side by side. Their
difference is the diagnostic. In our data it is 0.0001 for a feature-only model and
0.4811 for a memorizer.

## Step 4: put floors under the score

    indicator_floor(y, has_input)
    structure_floor(topology_features, y, folds)

The indicator floor is what the partition alone gives you, in closed form. The
structure floor is what the network alone gives you. Both use no sample data. Report
your score against them, not against 0.5. On our benchmark they are 0.6085 and
0.7374 respectively, and the transcriptome baseline the field reports is 0.6344,
which sits below the structure floor.

## Step 5: substitute the cohort mean

Retrain with the same folds, seeds and architecture, showing **every sample the
training-cohort mean input** instead of its own.

    cohort_mean_gain(scores_real, scores_cohort_mean, y, heldout_mask)

Compute the mean over training samples only, so the substitution cannot leak. Verify
it applied by counting nonzero entries before you trust the result.

If accuracy does not fall, the method is not using sample identity. If it rises,
sample identity is costing you accuracy, and no amount of architecture work on the
sample axis will help until the benchmark can reward it.

## Step 6: check the output depends on the input

    dispersion_ratio(scores, mc_std, n_mc)

Between-sample spread against the model's own sampling noise. At the no-signal null
the output does not vary with the sample beyond resampling. Report it alongside
accuracy: a model can score well and be input-independent, and only this catches it.

## What to do with a bad result

A failed check is information about the benchmark at least as much as about the
method. In our case the benchmark cannot reward patient specificity, because its
target has none. The constructive responses are to find a target that varies per
sample, or to state plainly that the task being measured is entity-level rather than
sample-level, and to stop describing the output as personalized.
