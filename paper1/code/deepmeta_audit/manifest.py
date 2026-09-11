"""Step 1 of the DeepMeta external audit: eligible-sample manifest, held-out set, gene panel,
gene families for block resampling.

Writes manifest.json plus two caches used by every later step:
  cache/expr_eligible.npz : log2(TPM+1), eligible lines x 19193 symbols (float64)
  cache/ge_candidates.npz : 24Q4 Chronos gene effect, eligible lines x candidate panel genes

Run:  python3 manifest.py            (~4 min, <2 GB RAM: the two big DepMap CSVs are
                                      parsed once and cached as .npz)
      python3 manifest.py --no-md5   (skip the ~1 GB of hashing when iterating)
"""
import argparse
import json
import os
import re
import time

import networkx as nx
import numpy as np
import pandas as pd
import pyreadr

import config as C
import jsonutil
import deepmeta_io as dio


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# --------------------------------------------------------------------------- caches
def load_expression(cells):
    """log2(TPM+1) for the given ModelIDs, columns = gene symbols (entrez suffix stripped).

    The cache only ever grows: a request for lines it does not hold parses those lines and adds
    them, so a later caller (e.g. native_repro, whose cells are not all eligible) can never
    shrink it.
    """
    path = os.path.join(C.CACHE, "expr_eligible.npz")
    cached = None
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        cached = pd.DataFrame(z["x"], index=list(z["cells"]), columns=list(z["genes"]))
        if set(cells) <= set(cached.index):
            return cached.loc[list(cells)]
    missing = set(cells) - (set(cached.index) if cached is not None else set())
    log(f"parsing expression matrix for {len(missing)} line(s) (cached afterwards)")
    rows = []
    for chunk in pd.read_csv(C.DEPMAP_FILES["OmicsExpressionProteinCodingGenesTPMLogp1.csv"],
                             index_col=0, chunksize=300):
        sel = chunk[chunk.index.isin(missing)]
        if len(sel):
            rows.append(sel.astype(np.float64))
    df = pd.concat(rows)
    df.columns = dio.strip_entrez(df.columns)
    assert df.columns.is_unique, "expression symbols are not unique after stripping Entrez ids"
    if cached is not None:
        df = pd.concat([cached, df.loc[:, cached.columns]])
    df = df[~df.index.duplicated()]
    os.makedirs(C.CACHE, exist_ok=True)
    np.savez_compressed(path, x=df.values.astype(np.float64),
                        cells=np.array(df.index, dtype=object),
                        genes=np.array(df.columns, dtype=object))
    return df.loc[[c for c in cells if c in df.index]]


def load_gene_effect(cells, columns):
    """24Q4 CRISPR (Chronos) gene effect for the given lines and "SYMBOL (ENTREZ)" columns.

    Like the expression cache, this cache only grows; the caller still gets exactly the lines
    and columns it asked for, in the order it asked for them.
    """
    path = os.path.join(C.CACHE, "ge_candidates.npz")
    req_cells, req_cols = list(cells), list(columns)
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        df = pd.DataFrame(z["x"], index=list(z["cells"]), columns=list(z["genes"]))
        if set(req_cells) <= set(df.index) and set(req_cols) <= set(df.columns):
            return df.loc[req_cells, req_cols]
        cells = sorted(set(req_cells) | set(df.index))       # never shrink the cache
        columns = sorted(set(req_cols) | set(df.columns))
    log("parsing CRISPRGeneEffect (once; cached afterwards)")
    wanted = set(columns)
    first = pd.read_csv(C.DEPMAP_FILES["CRISPRGeneEffect.csv"], nrows=0).columns[0]
    ge = pd.read_csv(C.DEPMAP_FILES["CRISPRGeneEffect.csv"], index_col=0,
                     usecols=lambda c: c in wanted or c == first)
    ge = ge.loc[[c for c in cells if c in ge.index], list(columns)].astype(np.float64)
    os.makedirs(C.CACHE, exist_ok=True)
    np.savez_compressed(path, x=ge.values.astype(np.float64),
                        cells=np.array(ge.index, dtype=object),
                        genes=np.array(ge.columns, dtype=object))
    return ge.loc[[c for c in req_cells if c in ge.index], req_cols]


# --------------------------------------------------------------------------- (a)-(c)
def build_samples(counts):
    model = pd.read_csv(C.DEPMAP_FILES["Model.csv"], dtype=str)
    counts["model_csv_rows"] = len(model)

    ge_ids = pd.read_csv(C.DEPMAP_FILES["CRISPRGeneEffect.csv"], usecols=[0]).iloc[:, 0]
    ex_ids = pd.read_csv(C.DEPMAP_FILES["OmicsExpressionProteinCodingGenesTPMLogp1.csv"],
                         usecols=[0]).iloc[:, 0]
    counts["lines_with_gene_effect"] = int(ge_ids.nunique())
    counts["lines_with_expression"] = int(ex_ids.nunique())

    inter = set(ge_ids) & set(ex_ids) & set(model.ModelID)
    counts["complete_case_expr_ge_model"] = len(inter)

    m = model[model.ModelID.isin(inter)].copy()
    cancer = m[(m.OncotreeLineage != "Normal") & (m.OncotreePrimaryDisease != "Non-Cancerous")]
    counts["after_deepmeta_cancer_filter"] = len(cancer)

    el = cancer[cancer.OncotreeLineage.isin(C.LINEAGE_MAP)].copy()
    counts["after_supported_lineage_filter_eligible"] = len(el)

    roster_all = set(pd.read_csv(C.DEEPMETA_FILES["all_cell_info.csv"]).cell)
    roster_test = set(pd.read_csv(C.DEEPMETA_FILES["test_cell_info.csv"]).cell)
    roster_train = roster_all - roster_test
    counts["authors_roster_all"] = len(roster_all)
    counts["authors_roster_train"] = len(roster_train)
    counts["authors_roster_test"] = len(roster_test)

    el["status"] = np.where(el.ModelID.isin(roster_train), "train",
                            np.where(el.ModelID.isin(roster_test), "test", "unseen"))
    el["net_template"] = el.OncotreeLineage.map(lambda l: C.LINEAGE_MAP[l][0])
    el["gtex_normal"] = el.OncotreeLineage.map(lambda l: C.LINEAGE_MAP[l][1])
    counts["eligible_by_status"] = el.status.value_counts().to_dict()
    counts["authors_train_lines_not_eligible"] = len(roster_train - set(el.ModelID))
    counts["authors_test_lines_not_eligible"] = len(roster_test - set(el.ModelID))
    counts["authors_test_lines_not_eligible_ids"] = sorted(roster_test - set(el.ModelID))

    # (b) an unseen line whose patient also contributed a training line is not really unseen.
    train_patients = set(model.PatientID[model.ModelID.isin(roster_train)])
    test_patients = set(model.PatientID[model.ModelID.isin(roster_test)])
    unseen = el.status == "unseen"
    el["patient_in_train"] = el.PatientID.isin(train_patients)
    el["patient_in_test"] = el.PatientID.isin(test_patients)
    excluded = unseen & el.patient_in_train
    counts["unseen_excluded_patient_shared_with_train"] = int(excluded.sum())
    counts["unseen_sharing_patient_with_test_not_excluded"] = int((unseen & el.patient_in_test).sum())

    # The training-patient exclusion applies to EVERY candidate test line, the authors' own test
    # roster included: a line whose patient also contributed a training line is not donor-independent
    # however the original split labeled it. Excluding it only from the newly unseen lines, which is
    # what an earlier version of this line did, left four such lines in the held-out set.
    candidate = (el.status == "test") | unseen
    el["heldout"] = candidate & ~el.patient_in_train
    counts["candidate_test_lines"] = int(candidate.sum())
    counts["candidate_excluded_patient_shared_with_train"] = int((candidate & el.patient_in_train).sum())
    counts["candidate_excluded_ids"] = sorted(el.ModelID[candidate & el.patient_in_train])
    counts["authors_test_excluded_patient_shared_with_train"] = int(((el.status == "test") & el.patient_in_train).sum())
    el["development"] = el.status == "train"
    counts["heldout_total"] = int(el.heldout.sum())
    counts["heldout_by_status"] = el.status[el.heldout].value_counts().to_dict()
    counts["development_total"] = int(el.development.sum())
    return el, counts


# --------------------------------------------------------------------------- (d) panel
def build_panel(el, counts):
    nodes = dio.train_node_ids()
    counts["train_node_ids"] = len(nodes)
    single = sorted(n for n in nodes if " and " not in n)
    counts["single_gene_nodes"] = len(single)
    counts["complex_nodes_dropped"] = len(nodes) - len(single)

    mapping = dio.gene_mapping()
    m1 = mapping[mapping.ensembl_id.isin(single)].drop_duplicates("ensembl_id").set_index("ensembl_id")
    counts["single_nodes_unmapped_to_symbol"] = len(set(single) - set(m1.index))

    ge_cols = pd.read_csv(C.DEPMAP_FILES["CRISPRGeneEffect.csv"], nrows=0).columns[1:]
    pat = re.compile(r"^(.+) \((\d+)\)$")
    sym2col, ent2col = {}, {}
    for c in ge_cols:
        mm = pat.match(c)
        sym2col[mm.group(1)] = c
        ent2col[mm.group(2)] = c
    counts["gene_effect_columns"] = len(ge_cols)

    rows = []
    for e in single:
        sym = m1.loc[e, "symbol"]
        ent = str(m1.loc[e, "gene_id"])
        col = sym2col.get(sym)
        how = "symbol" if col is not None else None
        if col is None:
            col = ent2col.get(ent)
            how = "entrez" if col is not None else None
        rows.append((e, sym, ent, col, how))
    cand = pd.DataFrame(rows, columns=["ensembl", "symbol", "entrez", "ge_col", "matched_by"])
    counts["matched_by_symbol"] = int((cand.matched_by == "symbol").sum())
    counts["matched_by_entrez"] = int((cand.matched_by == "entrez").sum())
    counts["unmatched_to_gene_effect"] = int(cand.ge_col.isna().sum())
    counts["unmatched_symbols"] = sorted(cand.symbol[cand.ge_col.isna()])
    cand = cand[cand.ge_col.notna()].copy()

    # two distinct network nodes may carry the same symbol (SLC35D2): a single gene-effect
    # column cannot be attributed to one of them, so both are dropped as ambiguous.
    dup = cand.ge_col.duplicated(keep=False)
    counts["dropped_ambiguous_shared_ge_column"] = int(dup.sum())
    counts["dropped_ambiguous_symbols"] = sorted(set(cand.symbol[dup]))
    cand = cand[~dup].copy()
    counts["candidate_panel"] = len(cand)

    dev = el.ModelID[el.development].tolist()
    ge = load_gene_effect(el.ModelID.tolist(), cand.ge_col.tolist())
    ge_dev = ge.loc[dev]
    n_obs = ge_dev.notna().sum(axis=0)
    var = ge_dev.var(axis=0, ddof=1)
    counts["development_lines_used_for_variance"] = len(dev)
    counts["candidate_genes_all_nan_in_development"] = int((n_obs == 0).sum())

    cand["n_obs_development"] = cand.ge_col.map(n_obs).values
    cand["var_development"] = cand.ge_col.map(var).values
    usable = cand.var_development.notna()
    cut = float(np.quantile(cand.var_development[usable].values, C.LOWVAR_DECILE))
    cand["dropped_low_variance"] = (~usable) | (cand.var_development <= cut)
    counts["low_variance_cut"] = cut
    counts["dropped_low_variance"] = int(cand.dropped_low_variance.sum())
    panel = cand[~cand.dropped_low_variance].copy().sort_values("ensembl").reset_index(drop=True)
    counts["panel_size"] = len(panel)
    counts["panel_gene_effect_nan_cells_max"] = int(
        (len(el) - ge[panel.ge_col].notna().sum(axis=0)).max())
    return panel, cand, counts


# --------------------------------------------------------------------------- (e) families
def build_families(panel, counts):
    """Partition the panel into resampling blocks.

    Primary structure: KEGG pathways (DeepMeta/data/kegg_all_pathway.rds, 341 human pathways
    with a class label; this file ships with the repository and is the pathway structure the
    authors themselves used). A gene belongs to many pathways, so the partition assigns each
    gene to the SMALLEST pathway (fewest panel genes) that contains it, ties broken by name;
    that makes the block the most specific pathway context the gene has.

    Fallback for panel genes absent from KEGG: connected components of the HMR enzyme graph
    (DeepMeta/data/HGM_all_gene.rds -- the enzyme-enzyme edge list of the whole HMR database,
    from/to/Metabolite, which is what the per-tissue *_enzymes_based_graph.tsv files are cut
    from; the repository ships no HMRdatabase_enzymes_based_graph.tsv) after removing hub
    metabolites, i.e. metabolites linking more than HMR_HUB_DEGREE enzymes. Genes still
    isolated form singleton families.

    Finally, no family may exceed FAMILY_CAP_FRAC of the panel; an oversized family is split
    into consecutive chunks of symbols sorted alphabetically.
    """
    syms = list(panel.symbol)
    sym_set = set(syms)
    kegg = pyreadr.read_r(C.DEEPMETA_FILES["kegg_all_pathway.rds"])[None]
    kp = kegg[kegg.genes.isin(sym_set)]
    sizes = kp.groupby("pathway").genes.nunique()
    assign = {}
    for g, d in kp.groupby("genes"):
        assign[g] = "KEGG:" + sorted(d.pathway.unique(), key=lambda p: (int(sizes[p]), p))[0]
    counts["family_genes_from_kegg"] = len(assign)
    counts["family_kegg_pathways_used"] = len(set(assign.values()))

    rest = sorted(sym_set - set(assign))
    counts["family_genes_needing_hmr_fallback"] = len(rest)
    if rest:
        hgm = pyreadr.read_r(C.DEEPMETA_FILES["HGM_all_gene.rds"])[None]
        deg = hgm.groupby("Metabolite").apply(
            lambda d: len(set(d["from"]) | set(d["to"])), include_groups=False)
        hubs = set(deg[deg > C.HMR_HUB_DEGREE].index)
        counts["hmr_hub_metabolites_removed"] = len(hubs)
        counts["hmr_metabolites_total"] = int(len(deg))
        ens_rest = set(panel.ensembl[panel.symbol.isin(rest)])
        sub = hgm[(~hgm.Metabolite.isin(hubs)) & hgm["from"].isin(ens_rest)
                  & hgm["to"].isin(ens_rest)]
        G = nx.Graph()
        G.add_nodes_from(ens_rest)
        G.add_edges_from(zip(sub["from"], sub["to"]))
        G.remove_edges_from(nx.selfloop_edges(G))
        e2s = dict(zip(panel.ensembl, panel.symbol))
        comps = sorted((sorted(c) for c in nx.connected_components(G)), key=lambda c: c[0])
        n_multi = 0
        for k, comp in enumerate(comps):
            name = f"HMR:{k:04d}" if len(comp) > 1 else f"SINGLETON:{e2s[comp[0]]}"
            n_multi += len(comp) > 1
            for e in comp:
                assign[e2s[e]] = name
        counts["family_hmr_components_multi_gene"] = n_multi
        counts["family_singletons"] = sum(1 for v in assign.values() if v.startswith("SINGLETON:"))

    fam = pd.Series(assign).reindex(syms)
    assert fam.notna().all()
    cap = max(1, int(np.floor(C.FAMILY_CAP_FRAC * len(syms))))
    counts["family_cap_genes"] = cap
    out, n_split = {}, 0
    for name, grp in fam.groupby(fam):
        members = sorted(grp.index)
        if len(members) <= cap:
            for g in members:
                out[g] = name
        else:
            n_split += 1
            for j in range(0, len(members), cap):
                for g in members[j:j + cap]:
                    out[g] = f"{name}|part{j // cap}"
    counts["families_split_by_cap"] = n_split
    fam = pd.Series(out).reindex(syms)
    counts["n_families"] = int(fam.nunique())
    counts["largest_family"] = int(fam.value_counts().max())
    counts["largest_family_share"] = float(fam.value_counts().max() / len(syms))
    return fam


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-md5", action="store_true", help="skip input hashing")
    args = ap.parse_args()
    C.ensure_dirs(C.OUT, C.CACHE)

    counts = {}
    t0 = time.time()
    el, counts = build_samples(counts)
    log("eligible lines:", counts["after_supported_lineage_filter_eligible"],
        "held-out:", counts["heldout_total"], "development:", counts["development_total"])

    panel, cand, counts = build_panel(el, counts)
    log("panel:", counts["panel_size"], "of", counts["candidate_panel"], "candidates")

    fam = build_families(panel, counts)
    panel["family"] = fam.values
    log("families:", counts["n_families"], "largest", counts["largest_family"])

    # expression cache (also verifies every eligible line really has an expression row)
    expr = load_expression(el.ModelID.tolist())
    counts["expression_rows_cached"] = int(expr.shape[0])
    counts["expression_genes"] = int(expr.shape[1])
    order = dio.model_gene_order()
    counts["model_gene_order"] = len(order)
    counts["model_genes_missing_from_expression"] = len(set(order) - set(expr.columns))
    counts["model_genes_missing_from_gtex"] = len(set(order) - set(dio.gtex_normal_matrix().index))

    ho = el[el.heldout]
    dev = el[el.development]
    manifest = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "fixed_choices": {
            "exp_cutoff_log2tpm1": C.EXP_CUTOFF,
            "gtex_pseudocount": C.NORMAL_PSEUDO,
            "dependency_threshold_gene_effect": C.DEP_THRESHOLD,
            "low_variance_decile": C.LOWVAR_DECILE,
            "family_cap_fraction": C.FAMILY_CAP_FRAC,
            "hmr_hub_degree": C.HMR_HUB_DEGREE,
            "seeds": C.SEEDS,
            "min_dev_lines_for_lineage_mean": C.MIN_DEV_FOR_LINEAGE_MEAN,
        },
        "lineage_map": {k: {"net_template": v[0], "gtex_normal": v[1]}
                        for k, v in C.LINEAGE_MAP.items()},
        "counts": counts,
        "heldout_by_lineage": {
            l: {"total": int((ho.OncotreeLineage == l).sum()),
                "test": int(((ho.OncotreeLineage == l) & (ho.status == "test")).sum()),
                "unseen": int(((ho.OncotreeLineage == l) & (ho.status == "unseen")).sum())}
            for l in sorted(ho.OncotreeLineage.unique())},
        "development_by_lineage": dev.OncotreeLineage.value_counts().sort_index().to_dict(),
        "heldout_patients_with_multiple_lines": int((ho.PatientID.value_counts() > 1).sum()),
        "samples": el[["ModelID", "PatientID", "OncotreeLineage", "OncotreePrimaryDisease",
                       "net_template", "gtex_normal", "status", "heldout", "development",
                       "patient_in_train", "patient_in_test"]]
        .sort_values("ModelID").to_dict("records"),
        "panel": panel[["ensembl", "symbol", "entrez", "ge_col", "matched_by",
                        "n_obs_development", "var_development", "family"]].to_dict("records"),
        "panel_dropped": cand[cand.dropped_low_variance][
            ["ensembl", "symbol", "ge_col", "var_development"]].to_dict("records"),
        "inputs": {},
    }
    if not args.no_md5:
        log("hashing inputs")
        for name, path in list(C.DEPMAP_FILES.items()) + list(C.DEEPMETA_FILES.items()):
            manifest["inputs"][name] = C.file_record(path)
    jsonutil.dump(manifest, C.MANIFEST_JSON)
    log("wrote", C.MANIFEST_JSON, f"in {time.time() - t0:.0f}s")

    print(json.dumps({k: counts[k] for k in [
        "complete_case_expr_ge_model", "after_deepmeta_cancer_filter",
        "after_supported_lineage_filter_eligible", "eligible_by_status",
        "unseen_excluded_patient_shared_with_train", "heldout_total", "heldout_by_status",
        "development_total", "candidate_panel", "dropped_low_variance", "panel_size",
        "n_families", "largest_family"]}, indent=1))


if __name__ == "__main__":
    main()
