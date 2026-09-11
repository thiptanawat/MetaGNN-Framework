"""Path resolution and slim data loading, shared by the analysis scripts.

The repository stores run outputs under results/<run>/ while the working tree they
were produced in used <run>/ directly. These helpers accept either, so the same
scripts run in both places. They also load a field-reduced copy of the reference
network, which is what the analysis needs and is small enough to ship.
"""
import os, json, gzip

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _first(*cands):
    for c in cands:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(cands[0])

def run(name, fn):
    """Path to a run artifact, e.g. run('repr', 'raw.json')."""
    return _first(os.path.join(ROOT, "results", name, fn),
                  os.path.join(ROOT, name, fn),
                  os.path.join(name, fn))

def data(fn):
    """Path to a top-level data file."""
    return _first(os.path.join(ROOT, fn), fn)

def reactions():
    """The reference network's reaction list, in the canonical order of every array the analysis reads.

    recon3d_aligned.json.gz is built by paper1/code/build_aligned_recon3d.py: one record per reaction
    in the order of the project's stoichiometric matrix, carrying the BiGG JSON annotation attached by
    identifier or stoichiometric signature (10,311 of 10,600 reactions) and the project's own gene
    table. The BiGG JSON itself must never be indexed by position against those arrays.
    """
    path = _first(os.path.join(ROOT, "recon3d_aligned.json.gz"), os.path.join(ROOT, "recon3d_aligned.json"),
                  "recon3d_aligned.json")
    fh = gzip.open(path, "rt") if path.endswith(".gz") else open(path)
    recs = json.load(fh)["reactions"]
    out = []
    for r in recs:
        out.append(dict(id=r["id"], name=r["name"], subsystem=r["subsystem"],
                        gene_reaction_rule=(r["gene_reaction_rule"] if r["aligned"] else r["gene_rule_table"]),
                        lower_bound=(r["lower_bound"] if r["lower_bound"] is not None else 0.0),
                        aligned=bool(r["aligned"]), has_gene=bool(r["gene_sets"]), n_genes=int(r["n_genes"]),
                        is_exchange=bool(r["is_exchange"])))
    return out

def out(fn):
    """Path to write a top-level output file."""
    return os.path.join(ROOT, fn)
