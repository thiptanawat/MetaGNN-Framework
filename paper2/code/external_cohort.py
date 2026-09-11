#!/usr/bin/env python3
"""Build the external cohorts' reaction-level panels for the microsatellite instability transport.

Frozen before any external outcome was inspected (docs/ANALYSIS_PLAN_2026-09-10.md, Section 2.3):
  probe sets -> Entrez identifiers from the GEO annotation of GPL570; the per-gene value is the
  maximum over the gene's probe sets of the deposited log2 intensity; reaction-level expression by
  the same gene-protein-reaction projection as the development cohort (gpr_map.py: minimum across
  a complex, maximum across alternatives); a panel reaction is retained when at least one of its
  complexes is fully measured on the array; percentiles are within-cohort midranks over the
  tumor samples of that cohort, 0 to 100, rounded to the nearest integer when printed.

Inputs: three files from the Gene Expression Omnibus, which are not redistributed here and are
fetched once into --geo_dir (default ~/geo):

  GSE39582_series_matrix.txt.gz   https://ftp.ncbi.nlm.nih.gov/geo/series/GSE39nnn/GSE39582/matrix/
  GSE13294_series_matrix.txt.gz   https://ftp.ncbi.nlm.nih.gov/geo/series/GSE13nnn/GSE13294/matrix/
  GPL570.annot.gz                 https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPLnnn/GPL570/annot/

Run --fetch to download exactly those three into --geo_dir, or place them there by hand. The script
also reads the development panel (results/msi_int/design.json) and the reference network. Outputs
under results/external/: <cohort>_panel.npz (patients x panel reactions, labels, identifiers), the
frozen panel (panel_frozen.json) and a coverage report (coverage.json). The outputs are committed,
so the analysis reproduces without the downloads; the downloads are needed only to rebuild them.
"""
import os, re, io, gzip, json, argparse
import numpy as np, pandas as pd
import _paths as PATHS
from gpr_map import project_matrix

ap = argparse.ArgumentParser()
ap.add_argument("--geo_dir", default=os.path.expanduser("~/geo"), help="directory with GSE39582_series_matrix.txt.gz, GSE13294_series_matrix.txt.gz, GPL570.annot.gz")
ap.add_argument("--out", default=os.path.join(PATHS.ROOT, "results", "external"))
ap.add_argument("--fetch", action="store_true",
                help="download the three GEO files named in the docstring into --geo_dir, then continue")
A = ap.parse_args(); os.makedirs(A.out, exist_ok=True)

GEO_FILES = {
    "GSE39582_series_matrix.txt.gz": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE39nnn/GSE39582/matrix/GSE39582_series_matrix.txt.gz",
    "GSE13294_series_matrix.txt.gz": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE13nnn/GSE13294/matrix/GSE13294_series_matrix.txt.gz",
    "GPL570.annot.gz":               "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPLnnn/GPL570/annot/GPL570.annot.gz",
}
if A.fetch:
    import urllib.request
    os.makedirs(A.geo_dir, exist_ok=True)
    for name, url in GEO_FILES.items():
        dest = os.path.join(A.geo_dir, name)
        if os.path.exists(dest):
            print("have", name); continue
        print("fetching", name, "from", url, flush=True)
        urllib.request.urlretrieve(url, dest + ".part"); os.replace(dest + ".part", dest)
_missing = [n for n in GEO_FILES if not os.path.exists(os.path.join(A.geo_dir, n))]
if _missing:
    raise SystemExit("missing GEO inputs in %s: %s\nDownload them with --fetch, or place them there "
                     "by hand from the URLs in this script's docstring." % (A.geo_dir, ", ".join(_missing)))

# ---- the development panel and the reference network --------------------------------------------
des = json.load(open(PATHS.run("msi_int", "design.json")))
PANEL = np.array(des["panel"]); PANEL_IDS = des["panel_ids"]
AL = json.load(gzip.open(os.path.join(PATHS.ROOT, "recon3d_aligned.json.gz")))["reactions"]
GENE_SETS = [r["gene_sets"] for r in AL]

# ---- GPL570: probe set -> Entrez ----------------------------------------------------------------
def read_annot(path):
    rows = []; inside = False
    with gzip.open(path, "rt", errors="replace") as f:
        for line in f:
            if line.startswith("!platform_table_begin"): inside = True; header = None; continue
            if line.startswith("!platform_table_end"): break
            if not inside: continue
            if header is None: header = line.rstrip("\n").split("\t"); continue
            rows.append(line.rstrip("\n").split("\t"))
    df = pd.DataFrame(rows, columns=header)
    return df
ann = read_annot(os.path.join(A.geo_dir, "GPL570.annot.gz"))
probe2entrez = {}
for pid, gid in zip(ann["ID"], ann["Gene ID"]):
    gid = (gid or "").strip()
    if not gid: continue
    probe2entrez[pid] = [g.strip() for g in gid.split("///") if g.strip()]

# ---- series matrices ----------------------------------------------------------------------------
def read_series(path):
    meta = {}; chars = []
    with gzip.open(path, "rt", errors="replace") as f:
        lines = f.readlines()
    tbl_start = next(i for i, l in enumerate(lines) if l.startswith("!series_matrix_table_begin"))
    tbl_end = next(i for i, l in enumerate(lines) if l.startswith("!series_matrix_table_end"))
    for l in lines[:tbl_start]:
        if l.startswith("!Sample_"):
            key, _, rest = l.partition("\t"); vals = [v.strip().strip('"') for v in rest.rstrip("\n").split("\t")]
            if key == "!Sample_characteristics_ch1": chars.append(vals)
            else: meta.setdefault(key[8:], vals)
    tbl = pd.read_csv(io.StringIO("".join(lines[tbl_start + 1:tbl_end])), sep="\t", index_col=0)
    tbl.columns = [c.strip('"') for c in tbl.columns]
    return meta, chars, tbl

def build(cohort, fname, label_fn, tumor_fn):
    meta, chars, tbl = read_series(os.path.join(A.geo_dir, fname))
    gsm = list(tbl.columns); n = len(gsm)
    charmap = [dict() for _ in range(n)]
    for row in chars:
        for i, v in enumerate(row):
            if ":" in v:
                k, _, val = v.partition(":"); charmap[i][k.strip()] = val.strip()
            elif v: charmap[i].setdefault("_bare", v)
    keep_tumor = np.array([tumor_fn(meta, charmap, i) for i in range(n)])
    y = np.array([label_fn(meta, charmap, i) for i in range(n)], dtype=object)
    # gene-level: max over probe sets per Entrez id, log2 scale as deposited
    M = tbl.values.astype(np.float64)
    gene_rows = {}
    for r, pid in enumerate(tbl.index):
        for g in probe2entrez.get(pid, []): gene_rows.setdefault(g, []).append(r)
    genes = sorted(gene_rows); G = np.stack([M[gene_rows[g]].max(axis=0) for g in genes], axis=1)  # samples x genes
    gene_index = {g: k for k, g in enumerate(genes)}
    X, cov = project_matrix([GENE_SETS[j] for j in PANEL], G, gene_index, fill=np.nan)
    Xt = X[keep_tumor]
    # within-cohort midrank percentiles over the tumor samples
    from scipy.stats import rankdata
    P = np.full_like(Xt, np.nan)
    for j in range(Xt.shape[1]):
        if cov[j]: P[:, j] = 100.0 * (rankdata(Xt[:, j]) - 1.0) / (Xt.shape[0] - 1)
    rec = dict(cohort=cohort, n_samples_deposited=n, n_tumor=int(keep_tumor.sum()), n_probe_sets=int(tbl.shape[0]),
               n_genes_with_entrez=len(genes), panel_covered=int(cov.sum()), panel_size=int(len(PANEL)),
               uncovered_panel_ids=[PANEL_IDS[j] for j in range(len(PANEL)) if not cov[j]],
               value_range=[float(np.nanmin(M)), float(np.nanmax(M))], processing=meta.get("data_processing", [""])[0],
               label_counts={str(k): int(v) for k, v in pd.Series(y[keep_tumor]).value_counts(dropna=False).items()})
    ids = [gsm[i] for i in range(n) if keep_tumor[i]]
    np.savez_compressed(os.path.join(A.out, f"{cohort}_panel.npz"), X=Xt.astype(np.float32), pct=P.astype(np.float32), ids=np.array(ids),
                        label=np.array([str(v) for v in y[keep_tumor]]), covered=cov, panel=PANEL, panel_ids=np.array(PANEL_IDS),
                        title=np.array([meta.get("title", [""] * n)[i] for i in range(n) if keep_tumor[i]]))
    return rec, cov

def lab39582(meta, cm, i):
    v = cm[i].get("mmr.status", "N/A"); return {"dMMR": "MSI-H", "pMMR": "non-MSI-H"}.get(v, "NA")
def tum39582(meta, cm, i): return cm[i].get("dataset", "") in ("discovery", "validation")
def lab13294(meta, cm, i):
    v = cm[i].get("_bare", ""); return {"MSI": "MSI-H", "MSS": "non-MSI-H"}.get(v, "NA")
def tum13294(meta, cm, i): return True

R = {}; covs = {}
for cohort, fname, lf, tf in (("gse39582", "GSE39582_series_matrix.txt.gz", lab39582, tum39582),
                              ("gse13294", "GSE13294_series_matrix.txt.gz", lab13294, tum13294)):
    if not os.path.exists(os.path.join(A.geo_dir, fname)): print("missing", fname); continue
    R[cohort], covs[cohort] = build(cohort, fname, lf, tf)
    print(cohort, json.dumps({k: v for k, v in R[cohort].items() if k != "processing"}))
covered_all = np.logical_and.reduce([c for c in covs.values()]) if covs else np.ones(len(PANEL), bool)
frozen = dict(note="development panel restricted to reactions coverable on GPL570 under the frozen projection rule; "
                   "fixed before any external outcome was inspected; used for every cohort including the development re-collection",
              panel=[int(PANEL[j]) for j in range(len(PANEL)) if covered_all[j]],
              panel_ids=[PANEL_IDS[j] for j in range(len(PANEL)) if covered_all[j]],
              dropped_ids=[PANEL_IDS[j] for j in range(len(PANEL)) if not covered_all[j]], development_panel_size=int(len(PANEL)))
json.dump(frozen, open(os.path.join(A.out, "panel_frozen.json"), "w"), indent=1)
json.dump(R, open(os.path.join(A.out, "coverage.json"), "w"), indent=1)
print("frozen panel:", len(frozen["panel"]), "of", len(PANEL), "; dropped", frozen["dropped_ids"])
