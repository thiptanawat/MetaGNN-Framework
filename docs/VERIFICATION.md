# Citation verification

Every reference in both manuscripts was resolved against an external authority
before use: PubMed E-utilities `esummary` for PMIDs, the Crossref REST API for DOIs,
and the arXiv API for preprints, with a token-overlap check between the resolved
title and the citation as written.

The checks fall into two kinds.

**The language-model literature set** (35 entries) resolved with a minimum
title-overlap score of 1.00 and no entry lacking an identifier. Two entries failed a
single route on one pass and were confirmed correct by a second, independent route;
both failures were transient API errors, not bad identifiers.

**The benchmark-validity and metabolic-modeling set** added during drafting was
resolved the same way. Of note, several claims in these manuscripts have prior art
that a first pass missed, and the citations were added as a result rather than being
discovered by a reviewer:

- joint row-and-column hold-out was formalized for drug-target prediction as
  "setting S4" a decade before this work;
- the qualitative observation that network topology predominantly determines what
  context-specific extraction algorithms output was published in 2023;
- the pattern of a naive summary-statistic predictor reaching near-ceiling under
  standard splits has been demonstrated in protein interaction and cancer
  drug-response benchmarks.

Each is cited and distinguished explicitly in the manuscripts rather than being
presented as new. The reusable scripts are in `paper2/code/verify_refs.py`.

## Independent re-verification, September 2026

Every entry in both bibliographies was re-resolved a second time, independently of
the first pass and of the scripts, by checking four fields per reference (title,
first-author surname, year, venue) against Crossref, PubMed or the arXiv API. The
two manuscripts share 13 keys, so 90 raw entries reduce to 78 distinct checks.

**Result: 76 verified, 1 metadata error, 1 unresolved, 0 fabricated.**

The metadata error and the unresolved entry are recorded here rather than quietly
fixed, because both matter to a reader.

**Metadata error, corrected.** Paper 2's entry for Gopalakrishnan et al. carried the
DOI `10.1016/j.ymben.2023.12.003`. That identifier resolves, but to a different
paper: Thalen et al., "Tuning of CHO secretional machinery improve activity of
secreted therapeutic sulfatase 150-fold," *Metabolic Engineering*, 2024. The correct
identifier is `10.1016/j.ymben.2022.12.003`, which Paper 1's bibliography already
used for the same reference. Paper 2 has been corrected. A DOI that resolves to the
wrong paper is the failure mode a title-overlap check cannot catch, which is why the
second pass checked identifiers by resolution rather than by string comparison.

**Unresolved: the authors' own companion manuscript.** The companion is cited by
both papers. It could not be located as a public preprint under the title given, and
the repository it points to suggests a different title for the same work. It is now
cited as a manuscript in preparation by the same authors, with no DOI claimed, and
Paper 1 states the graph model's hyperparameters inline so that the architecture is
reproducible without it.

Two further items were noted and left as they are. Cite keys `Gopalakrishnan2022`
and `Pahikkala2014` print 2023 and 2015 respectively, because both papers were
issued online the year before formal publication; the printed years are correct and
the keys are only labels. Powers (2011) has no Crossref record because its journal
is not a Crossref member; the entry names the journal and gives the author's own
2020 arXiv repost as the resolvable identifier, and says so.

## Re-verification of 6 September 2026

After the reviewer revision every reference of both manuscripts was re-resolved with
`tools/verify_refs.py` (Crossref for DOIs, the arXiv API for preprints), which compares the
resolved title, year and first author with the entry as written and writes
`docs/verify_refs_report.json`. Of 161 entries (75 in Paper 1, 86 in Paper 2), 150 carry a
resolvable identifier and every one of them resolved to the cited work with full title
agreement. Eleven entries carry no identifier by nature: the two companion manuscripts, cited
in both papers, two model cards, two classical statistics papers (Stein 1956; James and Stein
1961), Bengio and Grandvalet 2004, Pedregosa et al. 2011 and Ojala and Garriga 2010; these were
checked by hand against their publishers' pages. Year differences flagged by the script are
conference papers cited by their proceedings year with an arXiv identifier that predates the
proceedings, which is how the entries are written, and the one author flag is the
consortium-authored TCGA 2012 paper. Three details were corrected as a result: the Hamilton et
al. 2017 entry now carries the arXiv URL rather than an arXiv DOI that Crossref does not index;
the Schuh et al. 2026 entry no longer states a volume, since the article is an advance article;
and the Sclar et al. 2024 entry carries the paper's full title. Two references that had been
staged for a calibration analysis Paper 1 does not report (Sisk et al. 2023; Van Calster et al.
2019) were removed from Paper 1; the second remains in Paper 2, where it is cited. The
reviewer's concern that the Rahimi et al. 2026 DOI carries a non-bioRxiv prefix is unfounded:
`10.64898` is the prefix bioRxiv adopted in 2025, and the DOI resolves to the cited preprint.
