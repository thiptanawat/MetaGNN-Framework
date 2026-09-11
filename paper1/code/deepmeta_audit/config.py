"""Shared paths, constants and small helpers for the DeepMeta external audit.

Every fixed choice made anywhere in the pipeline is declared here so that it is
visible in one place (and mirrored in PROVENANCE.md).

The audit directory is self-contained apart from the *data* it reads, which lives in
the feasibility directory (DeepMeta clone, PreDeepMeta clone, checkpoint, DepMap 24Q4).
Set DEEPMETA_FEAS to move that data elsewhere; the two directories can be relocated
together as a unit.
"""
import hashlib
import os

AUDIT_DIR = os.path.dirname(os.path.abspath(__file__))
FEAS = os.environ.get(
    "DEEPMETA_FEAS",
    os.path.normpath(os.path.join(AUDIT_DIR, os.pardir, "feas_deepmeta")),
)
REPO = os.path.join(FEAS, "DeepMeta")            # authors' repository clone
PRE = os.path.join(FEAS, "PreDeepMeta")          # authors' preprocessing package clone
DEP = os.path.join(FEAS, "depmap_24q4")          # DepMap 24Q4 release files
CKPT = os.path.join(FEAS, "DeepMeta.pt")         # published checkpoint (md5 6bdd383581a53c7df72d066e0735e328)
CELL_NET_DIR = os.path.join(REPO, "scripts", "model")   # provides cell_net.py for the predictor

OUT = os.environ.get("DEEPMETA_AUDIT_OUT", AUDIT_DIR)
CACHE = os.path.join(OUT, "cache")
ARMS = os.path.join(OUT, "arms")
PREDS = os.path.join(OUT, "preds")
WORK = os.path.join(OUT, "work")                 # scratch PyG dirs, deleted between batches

MANIFEST_JSON = os.path.join(OUT, "manifest.json")
SCHEDULES_JSON = os.path.join(OUT, "schedules.json")
NATIVE_JSON = os.path.join(OUT, "native_repro.json")
RESULTS_JSON = os.path.join(OUT, "audit_results.json")

# ---------------------------------------------------------------- fixed choices
EXP_CUTOFF = 1.0            # expressed gene: log2(TPM+1) > 1 (PreEnzymeNet default, DeepMeta paper)
NORMAL_PSEUDO = 1.01        # log2(GTEx median TPM + 1.01) (DeepMeta README line 109)
DEP_THRESHOLD = -0.5        # prespecified binary dependency threshold on 24Q4 Chronos gene effect
LOWVAR_DECILE = 0.10        # drop panel genes in the lowest decile of development-line gene-effect variance
FAMILY_CAP_FRAC = 0.10      # no gene family may exceed this share of the panel
HMR_HUB_DEGREE = 30         # a metabolite touching >30 enzymes is a hub and its edges are dropped
SEEDS = [11, 22, 33]        # donor-schedule seeds; SEEDS[0] is the primary schedule
PRIMARY_SEED = 11
MIN_DEV_FOR_LINEAGE_MEAN = 5   # lineages with fewer development lines fall back to the development-wide mean
N_BOOT = 2000               # bootstrap replicates (line-level and family-level)
SESI = 0.01                 # smallest effect of interest, in concordance units
ARM_NAMES = ["own", "mean", "within", "cross", "within_expr", "within_graph"]

# lineage -> (tissue GSM enzyme network, GTEx normal tissue).
# Built by NAME from DeepMeta/README.md lines 43-59, whose positional vectors are resolved
# against unique(OncotreeLineage) of the authors' bundled data/Model.csv. 19 supported lineages.
LINEAGE_MAP = {
    "Ovary/Fallopian Tube": ("iOvarianCancer1620", "Ovary"),
    "Myeloid": ("bone_marrow", "Whole Blood"),
    "Bowel": ("iColorectalCancer1750", "Colon - Sigmoid"),
    "Skin": ("iSkinCancer1386", "Skin - Sun Exposed (Lower leg)"),
    "Bladder/Urinary Tract": ("iUrothelialCancer1532", "Bladder"),
    "Lung": ("iLungCancer1490", "Lung"),
    "Kidney": ("kidney", "Kidney - Cortex"),
    "Breast": ("iBreastCancer1746", "Breast - Mammary Tissue"),
    "Pancreas": ("iPancreaticCancer1613", "Pancreas"),
    "CNS/Brain": ("brain", "Brain - Amygdala"),
    "Esophagus/Stomach": ("iStomachCancer1511", "Stomach"),
    "Thyroid": ("iThyroidCancer1710", "Thyroid"),
    "Prostate": ("iProstateCancer1560", "Prostate"),
    "Head and Neck": ("iHeadNeckCancer1628", "Whole Blood"),
    "Uterus": ("iEndometrialCancer1713", "Uterus"),
    "Liver": ("iLiverCancer1788", "Liver"),
    "Cervix": ("iCervicalCancer1611", "Cervix - Endocervix"),
    "Adrenal Gland": ("adrenal_gland", "Adrenal Gland"),
    "Testis": ("iTestisCancer1483", "Testis"),
}

DEPMAP_FILES = {
    "Model.csv": os.path.join(DEP, "Model.csv"),
    "CRISPRGeneEffect.csv": os.path.join(DEP, "CRISPRGeneEffect.csv"),
    "OmicsExpressionProteinCodingGenesTPMLogp1.csv": os.path.join(
        DEP, "OmicsExpressionProteinCodingGenesTPMLogp1.csv"),
    "README.txt": os.path.join(DEP, "README.txt"),
}

DEEPMETA_FILES = {
    "DeepMeta.pt": CKPT,
    "train_genes.csv": os.path.join(REPO, "data", "train_genes.csv"),
    "all_cell_info.csv": os.path.join(REPO, "data", "all_cell_info.csv"),
    "test_cell_info.csv": os.path.join(REPO, "data", "test_cell_info.csv"),
    "test_dtV2.csv": os.path.join(REPO, "data", "test_dtV2.csv"),
    "test_preV2.csv": os.path.join(REPO, "data", "test_preV2.csv"),
    "enz_gene_mapping.rds": os.path.join(REPO, "data", "enz_gene_mapping.rds"),
    "cpg_gene.rds": os.path.join(REPO, "data", "cpg_gene.rds"),
    "kegg_all_pathway.rds": os.path.join(REPO, "data", "kegg_all_pathway.rds"),
    "HGM_all_gene.rds": os.path.join(REPO, "data", "HGM_all_gene.rds"),
    "GTEx_median_tpm.gct": os.path.join(
        REPO, "data", "GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct"),
    "model_gene_order.rda": os.path.join(PRE, "data", "model_gene_order.rda"),
}


def md5(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def file_record(path):
    return {"path": path, "bytes": os.path.getsize(path), "md5": md5(path)}


def net_tsv(net_name):
    return os.path.join(REPO, "data", "meta_net", "EnzGraphs",
                        f"{net_name}_enzymes_based_graph.tsv")


def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)
