"""The reference network in the canonical order of the training data.

Every array the probes read (labels, expression, the expression-bearing mask) is in the order of
the reactions in the project's stoichiometric matrix. The BiGG JSON export of Recon3D lists the same
reactions in a different order under partly different identifiers, so it must never be indexed by
position against those arrays. recon3d_aligned.json, built by paper1/code/build_aligned_recon3d.py,
carries one record per reaction in the canonical order with the JSON annotation attached by
identifier or by stoichiometric signature; 289 reactions have no match and are excluded from every
prompt. This module exposes that table in the shape the probes expect.
"""
import json, os, numpy as np

_path = os.environ.get("RECON3D_ALIGNED", "recon3d_aligned.json")
_A = json.load(open(_path))
RXN = []
for r in _A["reactions"]:
    RXN.append(dict(
        id=r["id"], name=r["name"], subsystem=r["subsystem"],
        gene_reaction_rule=(r["gene_reaction_rule"] if r["aligned"] else r["gene_rule_table"]),
        lower_bound=(r["lower_bound"] if r["lower_bound"] is not None else 0.0),
        upper_bound=r["upper_bound"], metabolites=r["metabolites"], aligned=bool(r["aligned"]),
        has_gene=bool(r["gene_sets"]), n_genes=int(r["n_genes"]), is_exchange=bool(r["is_exchange"])))
ALIGNED = np.array([r["aligned"] for r in RXN])
HAS_GENE = np.array([r["has_gene"] for r in RXN])
