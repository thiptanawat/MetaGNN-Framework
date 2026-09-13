# Comparison against the closest frameworks

Per the roadmap's Section 3. Columns are the capabilities that distinguish the two contributions;
a mark is entered only after reading the cited work's actual implementation, and a blank means the
capability is absent or not implemented there, not merely unmentioned. "Partial" is used where a
related but weaker form is present. This table is the source for the main-text related-work figure;
the wording in the manuscripts is "we provide and evaluate", not "the first".

Capabilities:
- **TP** target provenance: does the method distinguish a label shared across samples from a
  sample-varying measured outcome, and condition its claims on which it is?
- **ES** entity-similarity split: does it withhold biochemically related entities together
  (reaction families, GPR twins), not just random rows?
- **II** input interventions: does it substitute a sample's inputs (own / cohort-mean / donor) to
  measure what the model uses, recomputing every sample-dependent input?
- **ME** measured sample-varying endpoint: is there an external, sample-varying, measured outcome
  (gene effect, MSI phenotype), not only a reconstruction-membership proxy?
- **PC** useful-positive case: does the framework demonstrate it recognizes genuine sample-specific
  utility, not only failures (a calibrated positive control)?
- **XA** runnable external-model adapter: can it audit an independently published predictor as
  released, not only the authors' own model?

| Work | TP | ES | II | ME | PC | XA |
|---|---|---|---|---|---|---|
| DOME (Walsh 2021), reporting checklist | partial | | | | | |
| DataSAIL (Joeres 2025), similarity-aware splitting | | yes | | | | |
| Systema (Viñas Torné 2026), shared against perturbation-specific | yes | partial | | partial | | |
| GeneAgent / GeneGPT (2024 to 2025), tool-augmented language models | | | | | partial | partial |
| BioMaze (Zhao 2025), pathway-reasoning benchmark | | | | | | |
| TRIPOD-LLM (Gallifant 2025), reporting guideline for language models | partial | | | | | |
| FlowGAT (Hasibi 2024), graph model on a genome-scale model, essentiality | | partial | | yes (E. coli growth) | | own model only |
| DeepMeta (Wu 2025), metabolic dependency predictor | partial | | | yes (CRISPR) | | is the audited model |
| **Reaction-Scoring Audit (this work)** | yes | yes | yes | yes (DepMap gene effect) | yes (simulation) | yes (DeepMeta) |
| **Language-Model Evidence (this work)** | yes | partial | yes | yes (MSI, external cohort) | partial (tool arm) | partial (published workflow arm) |

Notes on the marks entered from the implementations:

- **DataSAIL** implements identity/similarity-aware splitting for arbitrary user-defined
  similarities; it does not itself distinguish label provenance or run input substitutions. The Reaction-Scoring Audit
  adds biologically justified reaction/GPR grouping and reports what the grouped split changes: all
  five substitution arms lose 0.050 to 0.078 AUROC, the no-patient linear floor loses 0.021, and the
  graph model's shortfall against that floor widens from 0.0083 to 0.0363. The grouped split is run
  and reported, not proposed.
- **Systema** separates systematic variation from perturbation-specific prediction, the neighbouring
  problem; it does not implement metabolic input substitutions or a reaction-family split. The Reaction-Scoring Audit
  positions its target-provenance and input-intervention axes against it.
- **GeneAgent / GeneGPT** are realistic tool-augmented biological LLM workflows; the Language-Model Evidence's tool arm
  tests one transparent stronger configuration in that spirit and adopts GeneAgent's practice of
  masking answer-exposing databases, rather than generalizing from zero-shot prompting alone.
- **FlowGAT** and **DeepMeta** are the external conventional models; DeepMeta is the audited model
  in Reaction-Scoring Audit-E2 and FlowGAT is the documented fallback. Neither ships a provenance/substitution
  audit; that is what the Reaction-Scoring Audit supplies as a reusable wrapper.
- The two "this work" rows record capabilities whose experiments have produced results. For the Reaction-Scoring Audit,
  ES rests on the family-disjoint split of Section 4.6, II on the five-arm substitution ladder, ME on
  the DepMap gene-effect audit, PC on the calibration simulation and XA on the DeepMeta wrapper. For
  the Language-Model Evidence, ES is marked partial because its entity grouping is at the level of reaction families
  inherited from the Reaction-Scoring Audit rather than rebuilt for its own units.
