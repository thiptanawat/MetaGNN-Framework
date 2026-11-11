# The fixed splits

`splits.json` holds the folds every run of the Reaction-Scoring Audit uses. It is written by
`paper1/code/export_splits.py` from the seed (2024), the cohort's patient list and the label and
expression-availability vectors, with the same lines the drivers use. The same file is deposited
with the cohort in the data record, under `crc_624/splits/`.

| Field | Contents |
|---|---|
| `patient_folds` | Five folds. Each gives the `train`, `val` and `test` barcodes; the test sets partition the 624 patients. |
| `reaction_folds_random` | Three folds. Each is the list of held-out reaction indices, 0-based, in the reaction order of the training arrays (the order of `recon3d_aligned.json`). Stratified on the label crossed with expression availability. |
| `reaction_folds_family` | Three folds in which every reaction family of `data/families/families_union.json` lies in one fold. |
| `reaction_folds_random_tissue_matched_label` | The random folds of the runs scored against the HT29-only label, which stratify on that label. |
| `checks` | Fold sizes and partition checks computed when the file was written. |

`python3 paper1/code/export_splits.py --check` rebuilds the folds in the running environment and
stops if they differ from this file; `bash reproduce.sh analysis` runs that check.

`array_check.json` records the comparison of these folds with the folds stored in the 299
prediction arrays of the release (`python3 paper1/code/export_splits.py --arrays DIR`): the test
patients and the held-out reactions of every array match.
