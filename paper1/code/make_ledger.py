#!/usr/bin/env python3
"""A ledger of every result family behind Paper 1: what produced it, from what, under which seed.

A reader who wants to know where a number came from should not have to read the analysis code. This
walks results/ and data/, groups the files into the families the manuscript draws on, and records for
each family the producing script, the arguments and settings the result files themselves carry, the
seed, the data record, the checkpoint policy and the endpoint, together with a SHA-256 of every file
so that a later reader can tell whether the file they hold is the file the numbers were computed
from. Nothing here is typed from memory: every field is either read out of the result files or is a
constant of the repository recorded in SCRIPTS below, and any field a result file does not carry is
written as null rather than guessed.

Writes results/LEDGER.json. Run from anywhere: python3 paper1/code/make_ledger.py
"""
import os, re, json, glob, hashlib

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, DATA = os.path.join(HERE, "results"), os.path.join(HERE, "data")
OUT = os.path.join(RES, "LEDGER.json")

# What produced each family, and the policy facts that live in the code rather than in its output.
# "runner" is the script as invoked; "policy" records the choices a reader would otherwise have to
# infer. These are constants of this repository. Every script a runner names must exist, every long
# flag it passes must be one that script accepts, and every environment variable it names must be one
# that script reads: all three are checked against the scripts themselves below, so a runner line
# cannot drift into an interface the code does not have.
SCRIPTS = [
    dict(family="graph and feature model cells, grid's selection rule",
         pattern=r"^(gnn_B|gnn_A|mlp_emb|mlp)_mb[\d.]+_p\d_r\d(_(mean|zero|indicator|permute|rewire\d+))?(_rank)?(_ts\d+)?\.json$",
         where=".", runner="code/double_holdout.py (substitution arms via code/double_holdout_ctrl.py, rewiring via code/double_holdout_rewire.py)",
         policy=dict(selection="validation patients over all training reactions",
                     checkpoint="the epoch with the best validation AUROC is restored before scoring",
                     endpoint="local GPU, one CUDA device per run", target="reconstruction-membership proxy label")),
    dict(family="graph model cells, matched selection rule",
         pattern=r"^gnn_B_mb[\d.]+_p\d_r\d(_(mean|zero|indicator|permute))?(_ts\d+)?_ivr\.json$",
         where="ivr", runner="code/double_holdout_ctrl.py --inner_val_rxn 0.2",
         policy=dict(selection="an inner partition of the training reactions, withheld from the loss",
                     checkpoint="the epoch with the best inner-validation AUROC is restored before scoring",
                     endpoint="local GPU, one CUDA device per run", target="reconstruction-membership proxy label")),
    dict(family="graph model cells, family-disjoint reaction split",
         pattern=r"^gnn_B_mb[\d.]+_p\d_r\d(_(mean|zero|indicator|permute))?(_ts\d+)?_ivr_fam\.json$",
         where="fam", runner="code/double_holdout_ctrl.py --inner_val_rxn 0.2 --family_map data/families/families_union.json --family_tag fam",
         policy=dict(selection="an inner partition of the training reactions drawn family by family",
                     checkpoint="the epoch with the best inner-validation AUROC is restored before scoring",
                     endpoint="local GPU, one CUDA device per run", target="reconstruction-membership proxy label")),
    dict(family="graph model cells, synthetic patient-varying target",
         pattern=r"^gnn_B_mb[\d.]+_p\d_r\d(_(mean|zero|indicator|permute))?_synth[\w.]*(_ivr)?\.json$",
         where="synth", runner="code/double_holdout_ctrl.py with the synthetic target options",
         policy=dict(selection="as the file's inner_val_rxn field records",
                     checkpoint="the epoch with the best validation AUROC is restored before scoring",
                     endpoint="local GPU, one CUDA device per run",
                     target="a constructed target that varies between patients")),
    dict(family="linear reference rows, all folds",
         pattern=r"^naive_allfolds(_rank|_ht29|_fam|_rerun)?\.json$", where=".",
         runner="code/naive_baselines_allfolds.py (P1_FAMILY_MAP set for the _fam file)",
         policy=dict(selection="inner cross-validation over the training reactions for the tuned regime",
                     checkpoint="not applicable", endpoint="CPU", target="reconstruction-membership proxy label")),
    dict(family="pooled frozen linear rows",
         pattern=r"^naive_pooled(_120|_rank|_ht29|_fam)?\.json$", where=".",
         runner="code/pooled_baselines.py",
         policy=dict(selection="inner cross-validation over the training reactions for the tuned regime",
                     checkpoint="one model per cell, fitted on the training patients and applied unchanged to the test patients",
                     endpoint="CPU", target="reconstruction-membership proxy label")),
    dict(family="near-duplicate split audit", pattern=r"^split_audit(_fam)?\.json$|^fam_matched\.json$", where=".",
         runner="code/split_audit.py (--family_map for the _fam file)",
         policy=dict(selection="not applicable", checkpoint="not applicable", endpoint="CPU",
                     target="not applicable: the audit reads the network, not a model")),
    dict(family="scores on the three reaction masks", pattern=r"^mask_scores\.json$", where=".",
         runner="code/mask_scores_p1.py, on the saved per-patient prediction arrays",
         policy=dict(selection="not applicable", checkpoint="not applicable",
                     endpoint="CPU, from saved predictions", target="reconstruction-membership proxy label")),
    dict(family="family-disjoint cells rescored on the audited subset", pattern=r"^fam_clean_scores\.json$", where=".",
         runner="code/fam_clean_scores.py, on the saved per-patient prediction arrays",
         policy=dict(selection="not applicable", checkpoint="not applicable",
                     endpoint="CPU, from saved predictions", target="reconstruction-membership proxy label")),
    dict(family="phenotype probe", pattern=r"^phenotype_(probe|probe_ivr|null_\w+|sex|site|cin_vs_gs)\.json$", where=".",
         runner="code/phenotype_probe.py and code/phenotype_null.py (code/phenotype_probe_multi.py for the further attributes)",
         policy=dict(selection="not applicable", checkpoint="not applicable", endpoint="CPU, from saved predictions",
                     target="a clinical attribute the training label cannot express")),
    dict(family="collapse to the cohort mean", pattern=r"^collapse(_noise)?\.json$|^arm_noise\.json$", where=".",
         runner="code/compare_ctrl.py and code/collapse_noise.py",
         policy=dict(selection="not applicable", checkpoint="not applicable", endpoint="CPU, from saved predictions",
                     target="reconstruction-membership proxy label")),
    dict(family="diagnostic calibration simulation", pattern=r"^diagnostic_sim\.json$", where=".",
         runner="code/diagnostic_sim.py",
         policy=dict(selection="the decision rule is fixed on the development worlds and applied unchanged to the test worlds",
                     checkpoint="not applicable", endpoint="CPU",
                     target="synthetic targets whose sample-specific signal is known by construction")),
    dict(family="external audit of a published predictor", pattern=r".*\.json$", where="deepmeta",
         runner="the released model as published, driven by the audit wrapper; see Supplementary Section 16",
         policy=dict(selection="not applicable: the released checkpoint is used unchanged",
                     checkpoint="the authors' released checkpoint, unmodified", endpoint="CPU or GPU, single process",
                     target="measured CRISPR gene effect, DepMap 24Q4 Chronos")),
    dict(family="graph model cells, tissue-matched label", pattern=r".*\.json$", where="ht29",
         runner="code/double_holdout_ctrl.py against the label of code/build_labels_ht29.py",
         policy=dict(selection="validation patients over all training reactions",
                     checkpoint="the epoch with the best validation AUROC is restored before scoring",
                     endpoint="local GPU, one CUDA device per run",
                     target="a reconstruction-membership label built from the one colorectal reconstruction")),
    dict(family="graph model cells, wrong-patient arm", pattern=r".*\.json$", where="permute",
         runner="code/double_holdout_ctrl.py with the expression columns permuted between patients",
         policy=dict(selection="validation patients over all training reactions",
                     checkpoint="the epoch with the best validation AUROC is restored before scoring",
                     endpoint="local GPU, one CUDA device per run", target="reconstruction-membership proxy label")),
    dict(family="graph model cells, rank-normalized expression", pattern=r".*\.json$", where="rankgnn",
         runner="code/double_holdout_ctrl.py with the second expression normalization",
         policy=dict(selection="validation patients over all training reactions",
                     checkpoint="the epoch with the best validation AUROC is restored before scoring",
                     endpoint="local GPU, one CUDA device per run", target="reconstruction-membership proxy label")),
    dict(family="graph model cells, second training seed", pattern=r".*\.json$", where="seedrep",
         runner="code/double_holdout_ctrl.py at a second training seed",
         policy=dict(selection="validation patients over all training reactions",
                     checkpoint="the epoch with the best validation AUROC is restored before scoring",
                     endpoint="local GPU, one CUDA device per run", target="reconstruction-membership proxy label")),
    dict(family="cohort and label construction", pattern=r"^(cohort|strata|param_counts|per_patient|inference|results_p1)\.json$",
         where=".", runner="code/cohort_stats.py, code/count_params.py, code/per_patient_auc.py, code/inference_p1.py",
         policy=dict(selection="not applicable", checkpoint="not applicable", endpoint="CPU", target="not applicable")),
]

# fields a cell file carries that a reader would want in the ledger, and where they mean the same
CELL_FIELDS = ("train_seed", "data", "expr_transform", "feature_mode", "inner_val_rxn", "n_inner_val_rxn",
               "n_fit_rxn", "n_heldout_rxn", "n_active_labels", "family_map", "family_scheme", "n_heldout_families",
               "epochs", "n_train", "n_test")
TOP_FIELDS = ("seed", "_data", "_expr_transform", "_n_active_labels", "_family_map", "_columns", "_C_grid",
              "_n_heldout", "_heldout_prevalence", "rfolds", "n_reactions", "split", "family_scheme")

def sha256(path, cap=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def collect(values):
    """Distinct values of a field across a family, or a single value when they agree."""
    vs = [v for v in values if v is not None]
    if not vs: return None
    uniq = []
    for v in vs:
        if v not in uniq: uniq.append(v)
    if len(uniq) == 1: return uniq[0]
    try: return dict(min=min(uniq), max=max(uniq), n_distinct=len(uniq))
    except TypeError: return sorted(map(str, uniq))[:12]

assigned = set()
families = []
for spec in SCRIPTS:
    d = RES if spec["where"] == "." else os.path.join(RES, spec["where"])
    if not os.path.isdir(d): continue
    rx = re.compile(spec["pattern"])
    files = sorted(f for f in os.listdir(d) if f.endswith(".json") and rx.match(f)
                   and os.path.join(d, f) not in assigned)
    if not files: continue
    recs, fields = [], {k: [] for k in set(CELL_FIELDS) | set(TOP_FIELDS)}
    for f in files:
        p = os.path.join(d, f); assigned.add(p)
        recs.append(dict(file=os.path.relpath(p, HERE), bytes=os.path.getsize(p), sha256=sha256(p)))
        try: j = json.load(open(p))
        except Exception: continue
        if isinstance(j, dict):
            for k in fields:
                if k in j and not isinstance(j[k], (dict, list)): fields[k].append(j[k])
    families.append(dict(family=spec["family"], directory=os.path.relpath(d, HERE),
                         runner=spec["runner"], policy=spec["policy"], n_files=len(files),
                         settings={k: collect(v) for k, v in sorted(fields.items()) if collect(v) is not None},
                         files=recs))

# the data records the whole chain rests on
records = []
for p in sorted(glob.glob(os.path.join(DATA, "*")) + glob.glob(os.path.join(DATA, "families", "*"))):
    if os.path.isfile(p):
        records.append(dict(file=os.path.relpath(p, HERE), bytes=os.path.getsize(p), sha256=sha256(p)))

unassigned = sorted(os.path.relpath(p, HERE) for p in glob.glob(os.path.join(RES, "**", "*.json"), recursive=True)
                    if p not in assigned and os.path.basename(p) != "LEDGER.json")

L = dict(
    _generated_by="code/make_ledger.py",
    note=("One row per result family behind Paper 1: what produced it, the settings its own output "
          "records, the checkpoint policy and endpoint, and a SHA-256 of every file. Fields a result "
          "file does not carry are absent rather than guessed. The prediction arrays the mask and "
          "clean-subset rescorings read are not in this repository; they are deposited separately and "
          "the scripts that read them are named above."),
    n_families=len(families), n_result_files=sum(f["n_files"] for f in families),
    data_records=records, families=families,
    result_files_not_in_any_family=unassigned)
json.dump(L, open(OUT, "w"), indent=1)
assert not unassigned, ("every result file must belong to a family in the ledger; these do not: "
                        + ", ".join(unassigned))

# The runner strings are prose, so check them against the scripts. A ledger that documents a flag or
# an environment variable the code does not have sends a reader to the wrong split in silence, which
# is worse than no ledger at all.
import re as _re
def _read(name):
    p_ = os.path.join(HERE, "code", name)
    return open(p_).read() if os.path.exists(p_) else None
_bad = []
for spec in SCRIPTS:
    _named = _re.findall(r"code/([A-Za-z0-9_]+\.py)", spec["runner"])
    _srcs = {}
    for _s in _named:
        _src = _read(_s)
        if _src is None: _bad.append(f"{spec['family']}: names {_s}, which does not exist")
        else: _srcs[_s] = _src
    for _flag in set(_re.findall(r"(--[a-z0-9_]+)", spec["runner"])):
        if _srcs and not any(f'"{_flag}"' in v or f"'{_flag}'" in v for v in _srcs.values()):
            _bad.append(f"{spec['family']}: no script in its runner accepts {_flag}")
    for _env in set(_re.findall(r"\b(P1_[A-Z_]+)\b", spec["runner"])):
        if _srcs and not any(_env in v for v in _srcs.values()):
            _bad.append(f"{spec['family']}: no script in its runner reads {_env}")
if _bad:
    raise SystemExit("the ledger documents an interface the code does not have:\n  " + "\n  ".join(_bad))
print(f"wrote {os.path.relpath(OUT, HERE)}: {len(families)} families, "
      f"{sum(f['n_files'] for f in families)} result files, {len(records)} data records, "
      f"{len(unassigned)} unassigned")
for f in families:
    print(f"  {f['n_files']:4d}  {f['family']}")
if unassigned:
    print("  unassigned:", ", ".join(unassigned[:8]) + (" ..." if len(unassigned) > 8 else ""))
