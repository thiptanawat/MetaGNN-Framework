# Provenance: inputs, fixed choices, and deviations

Everything below is fixed before any prediction was scored, and is recorded in
`manifest.json` (`inputs`, `fixed_choices`, `counts`) so the numbers can be traced back.
`fetch_inputs.sh` recreates the entire input directory below from source (two git clones at
pinned commits, five Figshare files, one Google Drive checkpoint), verifying every md5.

## 1. Input files

### DepMap Public 24Q4 (`../feas_deepmeta/depmap_24q4/`)

| file | md5 | bytes |
|---|---|---|
| `Model.csv` | `675210d17675f3517b0ce39a3c274f16` | 645,696 |
| `CRISPRGeneEffect.csv` | `6edf7ade09b9b34199210b559d4745d3` | 428,678,699 |
| `OmicsExpressionProteinCodingGenesTPMLogp1.csv` | `71794802b750ce77c422dad0720a40af` | 506,628,654 |
| `README.txt` | `54628c15a10b9ff6536db245cfed5231` | 43,103 |

### Published model and repository data (`../feas_deepmeta/`)

| file | md5 | bytes | role |
|---|---|---|---|
| `DeepMeta.pt` | `6bdd383581a53c7df72d066e0735e328` | 243,208,587 | the audited checkpoint |
| `DeepMeta/data/train_genes.csv` | `16b0a7cd7205e885353287afef6972c2` | 18,571 | the 928 node ids the model was trained to predict |
| `DeepMeta/data/all_cell_info.csv` | `dffa1c5d00db95a2237ebed7bf9ab620` | 11,306 | the authors' 760-line roster |
| `DeepMeta/data/test_cell_info.csv` | `8ee9756f556b901f886ceeaa5d43f35a` | 1,070 | the authors' 76 test lines |
| `DeepMeta/data/test_dtV2.csv` | `93d25d96dcaaf7752928807428ef1914` | 1,286,100 | the authors' test labels |
| `DeepMeta/data/test_preV2.csv` | `77f46f4b0288318bbb9c6dcc86cf28da` | 1,897,652 | the authors' own test predictions |
| `DeepMeta/data/enz_gene_mapping.rds` | `041bb5f289ed2b1bc10bdbfec187830a` | 395,902 | Ensembl / Entrez / symbol map |
| `DeepMeta/data/cpg_gene.rds` | `23df8b5f5abc7c2e7476cbfb2f4ffd6e` | 325,179 | 3247 CPG sets = node features |
| `DeepMeta/data/kegg_all_pathway.rds` | `7109e4d1d5910b16b401b8a26bc973d1` | 99,013 | pathway structure for gene families |
| `DeepMeta/data/HGM_all_gene.rds` | `ce5542670aab649e6b81cb47675b8220` | 672,904 | HMR enzyme-enzyme graph (family fallback) |
| `DeepMeta/data/GTEx_..._gene_median_tpm.gct` | `3d6ec49c6d524437362d70f2b6822367` | 17,780,477 | GTEx normal medians |
| `PreDeepMeta/data/model_gene_order.rda` | `b6e742c897c648877f01102c493d404e` | 25,574 | the 7993 expression-encoder genes, in order |
| `DeepMeta/data/meta_net/EnzGraphs/*_enzymes_based_graph.tsv` | (20 files, read as templates) | | per-lineage enzyme networks |

Repository commits: `DeepMeta` `44c62dc0d58605cf58e1dd56393789abdf5b5f9d` (2026-01-07),
`PreDeepMeta` `6ec0baee399547b38bf45937a247c8e019b54392` (2025-08-27).

Software: Python 3.11.15, torch 2.13.0+cpu, torch_geometric 2.8.0.post1, numpy 2.4.4,
pandas 3.0.2, scipy 1.17.1, scikit-learn 1.8.0, networkx 3.6.1.

### Every file the audit scripts read (and how `fetch_inputs.sh` provides it)

DepMap 24Q4, from Figshare 27993248 v1 (file id in parentheses): `Model.csv` (51065297),
`CRISPRGeneEffect.csv` (51064667), `OmicsExpressionProteinCodingGenesTPMLogp1.csv` (51065489),
`README.txt` (51065795). `CRISPRGeneDependency.csv` (51064631,
md5 `0d3bdadf0c59264e39f7fbadf232ccdb`) is fetched for completeness **but is not read by the
audit**: the audit's outcome is Chronos gene *effect*, not dependency probability.

From the `DeepMeta` clone: `DeepMeta.pt` (checkpoint, Google Drive `1ZQAaSeOgmgBy-dE23i5qu5pieaAdCCUd`),
`data/train_genes.csv`, `data/all_cell_info.csv`, `data/test_cell_info.csv`,
`data/test_dtV2.csv`, `data/test_preV2.csv`, `data/enz_gene_mapping.rds`, `data/cpg_gene.rds`,
`data/kegg_all_pathway.rds`, `data/HGM_all_gene.rds`,
`data/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct`,
`scripts/model/cell_net.py`, and the 19 tissue graphs
`data/meta_net/EnzGraphs/{adrenal_gland, bone_marrow, brain, iBreastCancer1746,
iCervicalCancer1611, iColorectalCancer1750, iEndometrialCancer1713, iHeadNeckCancer1628,
iLiverCancer1788, iLungCancer1490, iOvarianCancer1620, iPancreaticCancer1613, iProstateCancer1560,
iSkinCancer1386, iStomachCancer1511, iTestisCancer1483, iThyroidCancer1710, iUrothelialCancer1532,
kidney}_enzymes_based_graph.tsv` (the checkpoint itself lives at the feasibility-directory root,
not inside the clone). From the `PreDeepMeta` clone: `data/model_gene_order.rda`.
`fetch_inputs.sh` verifies all of these in its final pass (37 files); everything except the
five DepMap files and the checkpoint arrives with the two clones.

## 2. Fixed choices, and why

### Sample eligibility

| choice | value | reason |
|---|---|---|
| complete case | line must have 24Q4 expression **and** gene effect **and** a `Model.csv` row | all three are needed: expression builds both inputs, gene effect is the outcome, metadata gives the lineage |
| cancer filter | `OncotreeLineage != "Normal"` and `OncotreePrimaryDisease != "Non-Cancerous"` | exactly the filter DeepMeta applies (`README.md` lines 39-40, `scripts/train_data_preprocess.R` lines 3-4) |
| supported lineage | the 19 lineages with a tissue template | the model cannot be run without a GSM network and a GTEx normal tissue |
| lineage -> (network, GTEx tissue) | table in `config.LINEAGE_MAP` | built **by name**: `README.md` lines 43-59 give three positional vectors over `unique(OncotreeLineage)` of the authors' bundled `Model.csv`; resolving them positionally against 24Q4's `Model.csv` would silently mis-assign templates, because the two files have different lineage orders |
| status | `train` = in `all_cell_info.csv` minus `test_cell_info.csv` (684); `test` = `test_cell_info.csv` (76); `unseen` = neither | the checkpoint saw the 684 during training |
| patient exclusion | drop **any** candidate held-out line (authors' test or unseen) whose `PatientID` matches an authors' **train** line | a second line from the same patient is not independent of training, whichever roster it sits in; 8 of 142 candidates dropped (4 from the authors' test roster, 4 unseen). The first run applied this rule to the unseen lines only and kept a 138-line set; see section 4 |
| development set | the 669 authors' train lines present in 24Q4 | everything fitted or filtered (variance decile, all four baselines, lineage means) uses only these, so nothing is tuned on the held-out lines |
| held-out set | 71 test + 63 unseen = 134 | the evaluation set (`manifest.json`); the superseded 138-line set is kept as `manifest_138.json` |

### Gene panel

| choice | value | reason |
|---|---|---|
| candidate nodes | the 928 ids of `train_genes.csv` with no `" and "` (860) | a gene-complex node has no single gene-effect column; 68 complexes dropped |
| symbol mapping | `enz_gene_mapping.rds`, one symbol per Ensembl id | the mapping the model's own preprocessing uses |
| gene-effect column | match `SYMBOL (ENTREZ)` by symbol, else by Entrez id | 855 by symbol; 3 more by Entrez (`PRPF4B`->`PRP4K`, `METTL7A`->`TMT1A`, `METTL7B`->`TMT1B`), which are pure renames in 24Q4 |
| unmatched | `AKR1C1`, `ACOT1` dropped | no column in 24Q4 `CRISPRGeneEffect.csv` under either name |
| ambiguity | both `SLC35D2` nodes (`ENSG00000130958`, `ENSG00000285269`) dropped | two distinct network nodes share one gene-effect column; each appears in the same 4 tissue networks, so neither can be preferred |
| low-variance filter | drop the lowest decile of gene-effect variance **across development lines** (cut 0.0136, 86 genes) | a gene with no across-line spread cannot show sample-specificity in either direction; computing the cut on development lines keeps the held-out lines untouched |
| panel | 770 genes | |

### Gene families (block resampling)

Grouping by the first two characters of a symbol was ruled out by the specification, and
rightly: it groups paralogues by spelling, not by biology.

1. **KEGG (primary).** `DeepMeta/data/kegg_all_pathway.rds` (341 human pathways with class
   labels) ships with the repository and is the pathway structure the authors themselves use.
   A gene sits in many pathways, so the partition assigns each panel gene to the **smallest**
   pathway (fewest panel genes) containing it, ties broken by pathway name, the most specific
   context the gene has. 597 genes, 205 pathways.
2. **HMR enzyme graph (fallback)** for the 173 panel genes in no KEGG pathway: connected
   components of the HMR enzyme-enzyme graph after removing hub metabolites, i.e. metabolites
   linking more than 30 enzymes (98 of 1391 removed; without this, currency metabolites such as
   H2O/ATP collapse the graph into one component). 9 multi-gene components, 142 singletons.
3. **Cap.** No family may exceed 10% of the panel (77 genes); an oversized family would be split
   into alphabetical chunks. **No family needed splitting**: 356 families, largest 16 genes
   (2.1% of the panel).

### Inputs to the model

| choice | value | reason |
|---|---|---|
| expressed gene | `log2(TPM+1) > 1` | `PreEnzymeNet` default, and the definition in the paper |
| differential expression | `log2(TPM+1) / log2(GTEx median TPM + 1.01)` over `model_gene_order` | `PreDiffExp.R` plus `README.md` line 109; duplicated GTEx symbols averaged, as in `PreDiffExp` |
| templates | the **recipient's** lineage network and GTEx tissue, in every arm | the audit substitutes the *sample*, not the lineage; holding templates fixed is what isolates the sample-specific contribution |
| inference | `-d val -b 1`, `torch.set_grad_enabled(False)`, `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` | the authors' own script; the no-grad patch only prevents an out-of-memory kill, the arithmetic is unchanged |

### Arms and schedules

| choice | value | reason |
|---|---|---|
| seeds | 11 (primary), 22, 33 | three schedules show whether a conclusion depends on one draw |
| `within` | uniform derangement inside each lineage (rejection sampling from uniform permutations) | keeps lineage fixed, so only the sample identity changes |
| `within` singletons | Adrenal Gland and Prostate have one held-out line each: donor drawn from another lineage, flagged | a lineage of one cannot be deranged within itself |
| `cross` | uniform permutation with donor lineage != recipient lineage | the largest possible sample perturbation that still uses a real profile |
| `mean` | mean profile of the development lines of the recipient's lineage | asks whether a lineage-average sample is as good as the real one |
| `mean` fallback | development-wide mean when a lineage has < 5 development lines (Adrenal Gland 0, Testis 0, Prostate 4 -> 6 lines flagged) | a mean over 0-4 lines is not a lineage profile |
| `within_expr` / `within_graph` | donor's diff-exp with own SEN / donor's SEN with own diff-exp | separates the two input channels |
| shared construction | 4 SEN sets and 4 diff-exp files serve the 6 arms (`own`/`within_expr` share a SEN, `within`/`within_graph` share a SEN) | identical by construction, not an approximation: an arm is a pair (SEN profile, diff-exp profile), and files are always keyed by the recipient |

### Scoring

| choice | value | reason |
|---|---|---|
| orientation | every score is "higher = more dependent"; gene-effect-predicting baselines are negated | makes DeepMeta probabilities and baseline gene-effect predictions comparable |
| scoring domains (three, reported together) | `primary_template_domain`, `secondary_explicit_only`, `tertiary_zero_fill_all` | one domain choice buries the others; reporting all three shows how much the conclusion depends on it |
| primary: template domain | a pair counts only if **both** lines' lineage tissue templates carry the gene's node; a template-carried node the sample's SEN drops (unexpressed) keeps prediction 0; a line whose template lacks the node is not scored on that gene; same rule for per-line AUROC; per-gene domain size reported | the fair test: the model is only asked about genes its own tissue network contains; an unexpressed-but-present node is the model itself saying "not a dependency", so it is kept |
| secondary: explicit only | only pairs explicit in every arm | strictest common ground across arms |
| tertiary: zero-fill all | whole grid, absent nodes = 0 | widest but diluted: scores the model on genes absent from its tissue network, which drags per-line AUROC toward 0.5 |
| concordance | per gene, over held-out line pairs with **different** gene effect, share whose score ordering matches the ordering of negative gene effect, ties 1/2; macro-averaged equally over the panel | a within-gene, between-line statistic, which is exactly the claim "predictions are specific to the sample"; equal weighting stops a handful of high-coverage genes dominating |
| dependency threshold | gene effect `< -0.5` (prespecified, labelled in the output key) | the DepMap convention for calling a line dependent; note the authors' own labels use a *dependency probability* > 0.5, a different quantity, which is why the panel view and the authors' benchmark are reported separately |
| smallest effect of interest | 0.01 concordance | 1 pair in 100 changing order is the smallest difference worth acting on |
| bootstrap | 2000 replicates over held-out lines grouped by `PatientID`, rebuilding all pairs; 2000 over gene families; 95% percentile ranges | lines from one patient are not independent; genes in a pathway are not independent; both dependencies have to be priced in. The ranges are descriptive resampling ranges within this fixed audit and carry no nominal coverage claim |
| donor schedules | `within`: derangement inside the lineage under the constraint that the donor's `PatientID` differs from the recipient's; seeds 11 (primary), 22, 33 | a recipient whose own patient contributed another held-out line must not receive that line as its donor; the unconstrained schedules of the first run are kept as `schedules_unconstrained.json` |
| baselines | per-gene mean, lineage-conditioned per-gene mean, expression ridge (alpha by 5-fold CV over `1e1..1e7`, 7993 model genes), multi-output random forest (200 trees, `max_features=sqrt`, 2000 highest-variance model genes) | all fitted on development lines only; the per-gene mean is the reference for held-out squared error and scores concordance exactly 0.5 by construction (constant score -> all ties), which is a useful check on the concordance code |
| missing targets in baseline fits | filled with the development-line per-gene mean | multi-output fits need a complete Y; evaluation uses observed values only (0.29% of development targets are missing) |

## 3. Deviations from the specification, and why

1. **HMR graph file.** The specification names
   `data/meta_net/EnzGraphs/HMRdatabase_enzymes_based_graph.tsv`; that file does not exist in
   the repository (only per-tissue graphs, plus `GSM_xml/HMRdatabase.xml`). The equivalent
   tabular object *is* shipped as `data/HGM_all_gene.rds` (238,173 enzyme-enzyme edges over
   3,338 genes with a `Metabolite` column, the same schema the per-tissue `.tsv` files use), so
   that is what the fallback uses.
2. **KEGG was usable, so it is the primary grouping.** The specification asks for the pathway
   structure if one is usable in `data/`; `kegg_all_pathway.rds` covers 597 of 770 panel genes,
   so the HMR graph is only the fallback for the remaining 173.
3. **One extra panel exclusion.** Two node ids share a single gene-effect column (`SLC35D2`);
   both were dropped. The specification did not anticipate this case.
4. **`native_repro` runs 75 of the authors' 76 test lines.** `ACH-001211` is Soft Tissue /
   Rhabdoid Cancer in 24Q4, a lineage with no DeepMeta tissue template, so its inputs cannot be
   built. It is reported in `native_repro.json.excluded`.
5. **Smoke test scope.** `smoke.sh` uses the first six held-out **Lung** lines rather than the
   first six held-out lines overall, so that the `within` donors stay inside the lineage (as the
   real arm does) and the run fits the time budget on a small tissue network. It runs the `own`
   and `within` arms only, as specified; the `mean` arm is not exercised.
6. **Float precision.** The expression cache stores `float64`, so the rebuilt inputs reproduce
   the validated feasibility pipeline exactly: SEN feature files are byte-identical
   (md5 match on `ACH-000350`, `ACH-000219`, `ACH-000337`) and diff-exp matrices agree to 0.0.
   Predictions agree to 6.2e-6, which is float32 reduction-order noise in the GAT layers.

## 4. Corrections after the first run

Two things changed between the first full run and the run the results files carry, and both
are recorded rather than overwritten.

1. **The held-out population.** The first run excluded a line for sharing a patient with a
   training line only when that line was `unseen`, and kept the authors' 76-line test roster
   as it was; four of those test lines share a patient with a training line, so the 138-line
   set was not donor-independent of training. `manifest.py` now applies the exclusion to every
   candidate line and the held-out set is 134 lines (`counts.candidate_excluded_ids` lists the
   eight). `manifest_138.json` and `schedules_138.json` are the superseded records;
   `recompute_cells.json` lists the 134 lines whose arms were rebuilt and rescored.
2. **The donor constraint.** The first schedules were plain within-lineage derangements, so a
   recipient could receive as donor another held-out line from its own patient. `arms.py` now
   draws the within-lineage derangement under the constraint that the donor's patient differs
   from the recipient's (`donor_distinct_constraint` in `schedules.json`), and
   `run_donor_arms.sh` is the launcher that reran the four donor arms for all three seeds
   under it. `schedules_unconstrained.json` is the superseded assignment on the 134 lines, drawn
   between the two corrections; `schedules_138.json` is the first run's, on the 138 lines.

The `own` and `mean` arms do not depend on the schedule and were computed once on the 134
lines; the per-arm predictions under each seed are in `preds/seed11`, `preds/seed22` and
`preds/seed33` (the `own` and `mean` files there are copies of one computation).
