#!/usr/bin/env python3
"""Align the two Recon3D exports the project uses, and write one reaction table in the canonical order.

The training data (edge indices, stoichiometry, expression features, gene table, labels) are in the
order of the reactions in recon3d_stoich.h5, which descends from the Recon3D MATLAB distribution.
Recon3D.json is the BiGG export of the same model: the same 10,600 reactions and 5,835 metabolites,
in a different order, with identifiers that differ for about a fifth of the reactions. Any analysis
that indexed the JSON by position while reading arrays in the stoichiometry order was misaligned.

This script matches the two by identifier where the identifiers agree and by stoichiometric signature
(the set of metabolites with the sign of their coefficients, compartment tags normalized) where they
do not, and writes recon3d_aligned.json: one record per reaction in the canonical order, carrying the
canonical identifier, the aligned JSON record's name, subsystem, gene rule and bounds where a match
exists, the gene sets of the project's own gene table, and the metabolites read from the
stoichiometric matrix itself. Reactions with no match are flagged and carry the gene table and the
stoichiometry only.
"""
import os, re, ast, json, h5py, numpy as np, pandas as pd

D = os.path.expanduser("~/metagnn/metagnn-data-v2/crc_624")
RECON = os.path.expanduser("~/metagnn/metagnn-data-v2/shared/Recon3D.json")
OUT = os.path.expanduser("~/metagnn/out/recon3d_aligned.json")

h = h5py.File(os.path.join(D, "recon3d_stoich.h5"))
rid = [x.decode() for x in h["rxn_ids"][:]]; mid = [x.decode() for x in h["met_ids"][:]]; S = h["S"][:]
RX = json.load(open(RECON)); rx = RX["reactions"]; jpos = {r["id"]: i for i, r in enumerate(rx)}

def norm_met(m): return re.sub(r"\[(\w)\]$", r"_\1", m)
midn = [norm_met(m) for m in mid]
def jsig(r): return frozenset((k, 1 if v > 0 else -1) for k, v in r["metabolites"].items())
sig2j = {}
for i, r in enumerate(rx): sig2j.setdefault(jsig(r), []).append(i)
def hsig(i):
    col = S[:, i]; nz = np.nonzero(col)[0]
    return frozenset((midn[k], 1 if col[k] > 0 else -1) for k in nz)

g = pd.read_csv(os.path.join(D, "gpr_table.tsv"), sep="\t")
def table_sets(s):
    try: L = ast.literal_eval(s)
    except Exception: return []
    out = []
    for grp in L:
        genes = []
        for gsx in grp:
            for tok in str(gsx).split(","):
                tok = tok.strip().strip("[]' ")
                if tok and tok != "[]": genes.append(tok.split(".")[0])
        if genes: out.append(sorted(set(genes)))
    return out
GS = [table_sets(s) for s in g["gene_sets_str"].astype(str)]
assert len(GS) == len(rid)

recs = []; n_id = n_st = n_un = 0
for i, r in enumerate(rid):
    j = None; how = None
    if r in jpos: j = jpos[r]; how = "id"; n_id += 1
    else:
        c = sig2j.get(hsig(i), [])
        if len(c) == 1: j = c[0]; how = "stoichiometry"; n_st += 1
        else: n_un += 1
    col = S[:, i]; nz = np.nonzero(col)[0]
    mets = {midn[k]: float(col[k]) for k in nz}
    gene_sets = GS[i]
    rule_from_table = " or ".join(("(" + " and ".join(gs) + ")") if len(gs) > 1 else gs[0] for gs in gene_sets)
    rec = dict(idx=i, id=r, aligned=(j is not None), how=how, json_id=(rx[j]["id"] if j is not None else None),
               name=(rx[j].get("name") if j is not None else None),
               subsystem=(rx[j].get("subsystem") if j is not None else None),
               gene_reaction_rule=(rx[j].get("gene_reaction_rule") if j is not None else None),
               gene_rule_table=rule_from_table, gene_sets=gene_sets,
               n_genes=len({x for gs in gene_sets for x in gs}),
               lower_bound=(rx[j].get("lower_bound") if j is not None else None),
               upper_bound=(rx[j].get("upper_bound") if j is not None else None),
               metabolites=mets, is_exchange=r.startswith("EX_"))
    recs.append(rec)
json.dump(dict(reactions=recs, metabolite_ids=midn,
               note="canonical order = recon3d_stoich.h5 rxn_ids; JSON fields via alignment"),
          open(OUT, "w"))
print(f"aligned by id {n_id}, by stoichiometry {n_st}, unaligned {n_un}; wrote {OUT}")
# a self-check: the gene table should agree with the aligned JSON rule almost everywhere
agree = tot = 0
for rec in recs:
    if rec["aligned"] and rec["gene_sets"]:
        tot += 1
        jg = set(re.findall(r"(\d+)_AT\d+", rec["gene_reaction_rule"] or ""))
        tg = {x for gs in rec["gene_sets"] for x in gs}
        agree += (jg == tg)
print(f"gene table agrees with aligned JSON rule on {agree} of {tot} rule-bearing aligned reactions")
