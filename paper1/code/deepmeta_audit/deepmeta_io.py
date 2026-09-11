"""DeepMeta input construction, copied from the validated feasibility port
(feas_deepmeta/build_inputs.py, itself a line-by-line port of
PreDeepMeta::PreEnzymeNet / PreDiffExp / utils.R) and generalised so that the
expression profile fed to the model can be decoupled from the cell whose lineage
templates (tissue GSM enzyme network + GTEx normal tissue) are used.

Nothing here changes the authors' computation for the `own` arm: with
profile = the cell's own 24Q4 expression it reproduces build_inputs.py exactly.
"""
import os
import re

import numpy as np
import pandas as pd
import pyreadr

import config as C

_CACHE = {}


def _rds(path, key=None):
    if path not in _CACHE:
        obj = pyreadr.read_r(path)
        _CACHE[path] = obj[key] if key is not None else obj[None]
    return _CACHE[path]


def gene_mapping():
    """enz_gene_mapping.rds: gene_id (Entrez), ensembl_id, symbol."""
    return _rds(C.DEEPMETA_FILES["enz_gene_mapping.rds"])


def cpg_gene():
    """cpg_gene.rds: 3247 CPG sets x 3075 gene symbols (node feature source)."""
    return _rds(C.DEEPMETA_FILES["cpg_gene.rds"])


def model_gene_order():
    """The 7993 genes of the DeepMeta expression encoder, in input order."""
    return _rds(C.DEEPMETA_FILES["model_gene_order.rda"],
                "model_gene_order").iloc[:, 0].tolist()


def train_node_ids():
    return pd.read_csv(C.DEEPMETA_FILES["train_genes.csv"]).id.tolist()


def tissue_net(net_name):
    """Tissue/cancer GSM enzyme graph; read.table semantics (first col = row names)."""
    key = ("net", net_name)
    if key not in _CACHE:
        net = pd.read_csv(C.net_tsv(net_name), sep="\t", index_col=0)
        assert list(net.columns) == ["from", "to", "Metabolite"], net.columns
        _CACHE[key] = net
    return _CACHE[key]


def _ens2sym_tools():
    key = "ens2sym"
    if key not in _CACHE:
        mapping = gene_mapping()
        cpg = cpg_gene()
        _CACHE[key] = (mapping, cpg, set(cpg.columns))
    return _CACHE[key]


_NODE_CACHE = {}


def node_info(ensg):
    """Cell-independent half of PreDeepMeta/R/utils.R:ensg2name -- (symbol string, feature, symbols).

    The CPG feature vector of a node depends only on the node id, so it is memoised across
    cells; this is what makes rebuilding a SEN per (cell, arm) affordable.
    """
    hit = _NODE_CACHE.get(ensg)
    if hit is not None:
        return hit
    mapping, cpg, cpg_cols = _ens2sym_tools()
    split = ensg.split(" and ")
    sub = mapping[mapping.ensembl_id.isin(split)]
    syms = list(dict.fromkeys(sub.symbol.tolist()))     # unique(), in mapping row order
    sym = ",".join(syms)
    cols = [s for s in syms if s in cpg_cols]           # dplyr::select(any_of(.))
    if len(cols) == 0:
        fea = np.zeros(cpg.shape[0], dtype=int)
    else:
        fea = (cpg[cols].sum(axis=1).values != 0).astype(int)
    _NODE_CACHE[ensg] = (sym, fea, syms)
    return _NODE_CACHE[ensg]


def ensg2name(ensg, expressed):
    """PreDeepMeta/R/utils.R lines 37-65. Returns (symbol string, feature vector, is_exp)."""
    sym, fea, syms = node_info(ensg)
    is_exp = all(s in expressed for s in syms)          # all() of empty vector is TRUE in R
    return sym, fea, is_exp


def write_sen(profile, net_name, out_cell, sen_dir, exp_cutoff=C.EXP_CUTOFF,
              link_net=True):
    """Sample-specific enzyme network for one (expression profile, tissue template) pair.

    profile  : pd.Series of log2(TPM+1) indexed by gene symbol (the substituted profile)
    net_name : tissue GSM network of the RECIPIENT cell (never the donor's)
    out_cell : file stem, always the recipient ModelID
    """
    os.makedirs(sen_dir, exist_ok=True)
    net = tissue_net(net_name)
    expressed = set(profile.index[profile.values > exp_cutoff])
    ids = list(dict.fromkeys(list(net["from"]) + list(net["to"])))
    recs, feats = [], []
    for i in ids:
        sym, fea, is_exp = ensg2name(i, expressed)
        if len(sym) > 1 and is_exp:                     # PreEnzymeNet.R line 51
            recs.append((i, sym))
            feats.append(fea)
    feat = pd.DataFrame(recs, columns=["id", "gene"])
    fea_df = pd.DataFrame(np.vstack(feats),
                          columns=[f"fea-{k}" for k in range(1, cpg_gene().shape[0] + 1)])
    feat = pd.concat([feat, fea_df], axis=1)
    feat.insert(2, "is_exp", True)                      # column consumed+dropped by cell_net.py
    feat.to_csv(os.path.join(sen_dir, f"{out_cell}_feat.txt"), sep="\t", index=False)
    net_path = os.path.join(sen_dir, f"{out_cell}.txt")
    if os.path.lexists(net_path):
        os.remove(net_path)
    if link_net:
        # the edge list is the (large) tissue template, identical for every cell of a lineage;
        # symlink instead of copying so 6 arms x N cells do not multiply it on disk.
        # relative link, so the whole directory tree stays movable.
        os.symlink(os.path.basename(_materialise_net(net_name, sen_dir)), net_path)
    else:
        net.to_csv(net_path, sep="\t", index=False)
    return {"nodes_in_tissue_net": len(ids), "nodes_kept": len(feat),
            "tissue_net_edges": len(net), "network": net_name}


def _materialise_net(net_name, sen_dir):
    """One copy of each tissue edge list per SEN directory, in cell_net.py's expected format."""
    shared = os.path.join(sen_dir, "_net_%s.txt" % net_name)
    if not os.path.exists(shared):
        tissue_net(net_name).to_csv(shared, sep="\t", index=False)
    return shared


def gtex_normal_matrix():
    """GTEx median TPM -> log2(x + 1.01), gene symbol x tissue, restricted to model_gene_order."""
    key = "gtex"
    if key not in _CACHE:
        g = pd.read_csv(C.DEEPMETA_FILES["GTEx_median_tpm.gct"], sep="\t", skiprows=2)
        g = g[~g.Name.str.contains("PAR")].drop(columns=["Name"])
        tissues = [c for c in g.columns if c != "Description"]
        g[tissues] = np.log2(g[tissues].astype(float) + C.NORMAL_PSEUDO)
        order = model_gene_order()
        g = g[g.Description.isin(order)].groupby("Description")[tissues].mean()  # same-gene mean
        _CACHE[key] = g.reindex(order)
    return _CACHE[key]


def diff_exp_rows(profiles, normal_tissues):
    """log2(TPM+1) / log2(GTEx median TPM + 1.01) over model_gene_order.

    profiles       : dict cell -> pd.Series (log2(TPM+1) by symbol), the substituted profile
    normal_tissues : dict cell -> GTEx tissue of the RECIPIENT's lineage
    """
    order = model_gene_order()
    gtex = gtex_normal_matrix()
    cells = list(profiles)
    tumor = np.vstack([profiles[c].reindex(order).values.astype(float) for c in cells])
    normal = np.vstack([gtex[normal_tissues[c]].values.astype(float) for c in cells])
    diff = pd.DataFrame(tumor / normal, index=cells, columns=order)
    diff.insert(0, "cell", cells)
    return diff


def strip_entrez(columns):
    return [re.sub(r" \(.+", "", c) for c in columns]
