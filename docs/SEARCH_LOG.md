# Dated literature-search log

Kept per the roadmap's Section 3. Records the searches run to position the two papers and to find
support for both successful and unsuccessful prediction, with the exact databases, queries, filters
and screening decisions. This is not a systematic review; it is a documented, reproducible search.

## 2026-09-10: targeted verification of the closest frameworks

Each of the following was resolved to a primary record and its metadata confirmed (Crossref,
NCBI E-utilities, arXiv, DataCite); the verified entries are in `docs/new_references.tex`.

- DOME reporting recommendations (Walsh et al., Nature Methods 2021, 10.1038/s41592-021-01205-4)
- DataSAIL similarity-aware splitting (Joeres et al., Nat Commun 2025, 10.1038/s41467-025-58606-8)
- Systema (Viñas Torné et al., Nat Biotechnol 2026, 10.1038/s41587-025-02777-8)
- GeneAgent (Wang et al., Nature Methods 2025, 10.1038/s41592-025-02748-6)
- GeneGPT (Jin et al., Bioinformatics 2024, 10.1093/bioinformatics/btae075)
- BioMaze (Zhao et al., arXiv:2502.16660 v5; no peer-reviewed venue as of this date)
- TRIPOD-LLM (Gallifant et al., Nature Medicine 2025, 10.1038/s41591-024-03425-5)
- Sclar et al. prompt-format sensitivity (ICLR 2024, arXiv:2310.11324)
- Tian et al. confidence elicitation (EMNLP 2023, 10.18653/v1/2023.emnlp-main.330)
- Riley et al. external-validation sample size (Stat Med 2021, 10.1002/sim.9025)
- Bengio & Grandvalet, K-fold variance (JMLR 5:1089–1105, 2004)
- DeepMeta (Wu et al., Cell Reports 2025, 10.1016/j.celrep.2025.115945)
- DepMap 24Q4 Public (Broad, Figshare+, 10.25452/figshare.plus.27993248.v1)
- Chronos (Dempster et al., Genome Biol 2021, 10.1186/s13059-021-02540-7)
- Tsherniak et al. dependency map (Cell 2017, 10.1016/j.cell.2017.06.010)
- Pacini et al. cross-study dependencies (Nat Commun 2021, 10.1038/s41467-021-21898-7)
- FlowGAT (Hasibi et al., npj Syst Biol Appl 2024, 10.1038/s41540-024-00348-2)
- Marisa et al. CIT/GSE39582 (PLOS Med 2013, 10.1371/journal.pmed.1001453)
- Jorissen et al. GSE13294 (Clin Cancer Res 2008, 10.1158/1078-0432.CCR-08-1431; note that the journal
  is *Clinical Cancer Research*, not *Cancer Research* as the roadmap wrote)
- NCBI GEO (Barrett et al., NAR 2013, 10.1093/nar/gks1193)
- TCGA gastrointestinal MSI calls (Liu et al., Cancer Cell 2018, 10.1016/j.ccell.2018.03.010)
- MSI across 18 cancer types (Hause et al., Nat Med 2016, 10.1038/nm.4191)
- Consensus molecular subtypes (Guinney et al., Nat Med 2015, 10.1038/nm.3967)
- MSI in colorectal cancer (Boland & Goel, Gastroenterology 2010, 10.1053/j.gastro.2009.12.064)
- Calibration (Van Calster et al., BMC Med 2019; Steyerberg & Vergouwe, Eur Heart J 2014)
- Mann-Whitney effect-size interval (Newcombe, Stat Med 2006, 10.1002/sim.2324)
- Jain et al. NCI-60 CORE (Science 2012, 10.1126/science.1218595)

## 2026-09-10: broad Crossref sweeps for neighbouring work (from 2024-01-01), 20 results each

Database: Crossref REST (`api.crossref.org/works`), relevance-ranked `query=`, `filter=from-pub-date:2024-01-01`,
`rows=20`. Crossref `query` is a full-text relevance match over the whole corpus, so most hits are
keyword collisions; the screening below keeps only records on the intended topic.

| # | Query | Total | Kept after screening |
|---|---|---|---|
| A | reaction similarity aware data splitting metabolic network machine learning | 2,066,659 | none new beyond DataSAIL (already cited) and GraphPart (already cited); no metabolic-reaction-similarity split found |
| B | large language model gene expression patient specific prediction benchmark control | 196,624 | none with an explicit input-substitution or negative-control design; the nearest (SSRN preprint 10.2139/ssrn.5392141, LLM on leukocyte RNA-seq) has no control framing |
| C | external validation microsatellite instability transcriptome prediction colorectal 2025 | 10,651,206 | 10.17816/gc642339 (transcriptome MSI detection, CRC); 10.35377/saucis...1638424 and 10.5220/0014184000004918 (deep-learning MSI from expression), adjacent transcriptome-MSI classifiers, none an external-cohort audit of an LLM |
| D | sample-specific metabolic model prediction gene dependency deep learning 2025 | 1,037,363 | outside the top 10: 10.1371/journal.pcbi.1013547 (sample-specific gene-combination effects) and an AACR abstract 10.1158/1538-7445.am2025-3645 (pathway-informed dependency prediction); DeepMeta, FlowGAT, Chronos, Tsherniak, Pacini remain the established references |

Screening decisions: excluded records outside biology/biomedicine (finance, hydrology, video coding,
etc.), excluded clinical reviews and epidemiology with no prediction model, and excluded records
without an accessible primary description. Kept records were checked to be about the intended topic
(similarity-aware splitting of biochemical entities; controlled evaluation of LLM use of a sample's
molecular data; external-cohort transcriptome MSI prediction; sample-specific metabolic dependency).

Conclusion for the novelty statement: no work was found that combines target-provenance auditing,
entity-similarity splitting, input-substitution controls and an external measured endpoint for
either the conventional-model case (Reaction-Scoring Audit) or the language-model case (Language-Model Evidence). The manuscripts
therefore say "we provide and evaluate", not "the first", and cite the neighbouring frameworks above
as the closest prior art.
