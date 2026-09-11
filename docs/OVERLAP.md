# Overlap between the two manuscripts and the companion preprint

Both manuscripts in this repository, and a third document neither one contains, come from
the same underlying data curation and the same authors. This note sets out what the two papers share, what
each restates for context and what is new in each, based only on what the two
manuscripts themselves say about the relationship, in their introductions ("Relation to our
companion preprint" in Paper 1, "Relation to the companion audit" in Paper 2) and in their
Methods sections. It does not draw on the companion document directly, since that document
is not part of this repository.

## Three documents, one word

"Companion" names three different things across these two manuscripts, and the table below
keeps them apart by column:

1. **The companion preprint.** An external document by the same authors, cited as
   `\cite{Companion}` in both manuscripts, that curated this colorectal cohort in the first
   place, fixed the graph neural network configuration Paper 1 studies, and separately
   audited a graph scorer on a second, breast cancer cohort. It is not part of this
   repository and carries no DOI in either bibliography as of this writing.
2. **Paper 1** ("Holding out reactions as well as patients changes what transcriptome-based
   metabolic reaction scoring benchmarks measure"), which Paper 2's bibliography cites as
   `\cite{CompanionDH}`, correctly, under its current title.
3. **Paper 2** ("Response sensitivity does not establish patient-specific validity in
   language-model annotation of metabolic reactions"), which Paper 1's bibliography cites as
   `\cite{CompanionLLM}`. **This citation is stale**: Paper 1's bibliography entry for
   `CompanionLLM` gives the title "Language-model annotation of metabolic reaction activity
   responds to a patient's transcriptome without gaining patient-specific information from
   it," which is an earlier title of Paper 2 and does not match Paper 2's current title
   above. The reverse citation, Paper 2 quoting Paper 1's title, is currently correct. This
   is a wording fix inside a `.tex` file (`paper1/manuscript/body_bib.tex`), so it is
   reported here rather than corrected by this document.

Paper 1 states the graph configuration is "carried forward unchanged from our companion
preprint" and describes the activity-label union as "our construction rather than a released
artifact, shared with the companion preprint." Paper 2 states that it "uses the cohort, the
labels and two reference points established in a companion preprint," and separately that
two of its input files "are shared with the companion audit and live under its directory of
the same repository," referring there to Paper 1, not to the external preprint. The word
"companion" is therefore doing two jobs even inside a single manuscript's Data availability
section; the table below always names which of the three documents it means.

## Numbers that appear in more than one manuscript

Both manuscripts compute the following from the same aligned reaction table and expression
matrix, independently, in separate scripts, and reach bit-identical values, which is the
strongest evidence that the two share their input data and not merely their description of
it:

- **624 patients** (`\nPat` in Paper 1, `\netPat` in Paper 2), **10,600 reactions**
  (`\nRxn` / `\netRxn`), **10,311 aligned and 289 unaligned** (`\nAligned`/`\nUnaligned` and
  `\netAligned`/`\netUnaligned`), **5,938 reactions with a gene rule** (`\nGpr`/`\netGpr`) of
  which **5,635 receive an expression value** (`\nExpr`/`\netExpr`), and **3,379 reactions
  active** under the label union (`\nActive`/`\netActive`, base rate 0.3188 in Paper 1's
  four-decimal form and 0.319 in Paper 2's three-decimal form: the same value, not a
  discrepancy).
- **The raw-expression baseline, 0.6342**, and **the patient-invariant availability
  baseline, 0.6085**: ranking reactions by the mapped expression value alone, and the
  informedness identity for whether a reaction receives any expression at all,
  respectively. Paper 1 names these `\rawExpr` and `\indicator`; Paper 2 names the same two
  quantities, computed the same way over the same network, `\netRawExpr` and
  `\netIndicator`. Both manuscripts also state the two reference points are baselines rather
  than a ceiling for their respective benchmarks, in nearly identical language.
- **One genuine minor inconsistency**: the share of expression-bearing reactions that are
  labeled active is quoted as 40.7% in Paper 1 (`\prevExpr`) and 40.8% in Paper 2
  (`\netPrevExpr`), a one-tenth-of-a-point difference in the same underlying quantity. It is
  small enough to be a rounding or fold-boundary artifact of the two papers' separate
  pipelines rather than a data mismatch, but the two figures are not identical and an editor
  comparing the manuscripts side by side would notice it.
- **The companion preprint's own numbers**, quoted rather than recomputed: an AUROC of
  0.9864 on its breast cancer cohort, splitting into 0.9291 on the partition of that cohort's
  labels it shows to be memorized noise, and 0.5800 on the colorectal cohort used by both
  manuscripts here, from a verified retraining after the companion preprint found the
  originally archived models had received no patient input. Paper 2's introduction and
  discussion both quote all three figures; Paper 1 does not restate them numerically but
  points to the same preprint for the diagnostics behind its own ceiling argument.

## Cohort, labels, model and what is new

| | Companion preprint (external, cited only) | Paper 1 | Paper 2 |
|---|---|---|---|
| **Cohort** | Curated the colorectal cohort used by both manuscripts here, and a separate breast cancer cohort used only in the companion preprint itself | TCGA-COAD and TCGA-READ primary tumors from the GDC, 624 patients after deduplication (the same cohort; Paper 1's Data availability section gives a GDC manifest retrieval date of 9 March 2026) | The identical 624-patient cohort, described in the same terms; Paper 2 additionally uses the patients' microsatellite instability status for its one patient-varying target |
| **Expression preprocessing and gene mapping** | Curated the expression release both manuscripts read | log2(TPM+1), gene-protein-reaction aggregation (minimum across AND-joined subunits, maximum across OR-joined isozymes), described without repeating the companion preprint's exact gene count at that step | The identical log2(TPM+1) and GPR aggregation rule, explicitly stated to be computed "over the 21,263 genes of the curated release," with that citation given to the companion preprint |
| **Activity labels** | Fixed the eleven-reconstruction union construction that both manuscripts use | States plainly that the label union "is our construction rather than a released artifact, shared with the companion preprint"; ships the label builder and the on-disk vector name (`activity_pseudolabels.pt`, deposited data, not in this repository) | Uses the identical label union as its main reaction-activity target, cites the same eleven-reconstruction construction to the companion preprint, and additionally rescores every one of its own runs against a tissue-matched, HT29-only relabeling that Paper 1's release builds (`paper1/data/activity_labels_ht29.pt`, `paper1/code/build_labels_ht29.py`) |
| **Graph model configuration** | Fixed the heterogeneous GATv2 architecture (hidden width 256, 3 message-passing layers, 8 attention heads, dropout 0.2, three typed relations, 519-column metabolite features) on its own two cohorts | States the same configuration "is carried forward unchanged from our companion preprint... and is stated here in full so that it can be reproduced without it," then studies that exact architecture's behavior under a joint patient-and-reaction hold-out | Not used. Paper 2's subject is a language model's zero-shot annotations, not the graph scorer; it reuses only the companion lineage's cohort, labels and the two reference points above, which need no trained model to compute |
| **Results reproduced for context** | (its own results, quoted by the other two) | Points to the companion preprint for the leakage and input-invariance diagnostics behind its own ceiling argument, without restating their numbers; recomputes the raw-expression and availability baselines independently and reaches the values above | Quotes the companion preprint's breast-cohort AUROC (0.9864, splitting to 0.9291) and colorectal-cohort AUROC (0.5800) directly in its introduction and discussion, as the "third failure mode" comparison; recomputes the same raw-expression and availability baselines independently and reaches the identical values above |
| **New experiments unique to each** | (not applicable) | The joint patient-by-reaction double hold-out itself; the naive-baseline suite; the structure-only floor; the input-substitution decomposition (real/mean/zero/indicator/permute feature modes) and its patient-identity finding; a degree-preserving rewiring null; out-of-fold phenotype probes (microsatellite instability, chromosomal instability, sex, tumor site) as targets that vary between patients; cohort-mean collapse analysis; the tissue-matched-label, rank-normalization and training-seed robustness checks | The three-arm prompting design (patient-blind, own-patient, stranger's-patient) and its determinism floors; the patient-swap vacuity proof for shared-label benchmarks; positive controls (echo, read, compare); the presence-versus-magnitude decomposition of the model's response; the microsatellite-instability patient-level probe (the one target in either manuscript that varies between patients, with a supervised reference shown to have power on it); repetition across a second and third backbone; format-invariance arms; the blocked-versus-interleaved collection-order check |

Everything in Paper 1's list is explicitly stated, in its own introduction, to be absent
from the companion preprint ("Everything this paper is about is absent from it"). Everything
in Paper 2's list follows the same pattern relative to both the companion preprint and Paper
1: Paper 2 does not repeat Paper 1's double hold-out, naive baselines or phenotype probes,
and Paper 1 does not repeat any of Paper 2's language-model experiments. The two manuscripts'
new-experiment lists do not overlap with each other at all; what they share is confined to
the cohort, the label construction, and the two reference points in the table above.
