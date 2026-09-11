#!/usr/bin/env python3
"""Derive data/results_p1.json from the raw per-cell outputs.

This is the only place experiment numbers enter the manuscript pipeline, and it
reads them rather than restating them:

    results/<model>_mb<lambda>_p<pfold>_r<rfold>[_mean|_zero].json   one training run
    results/naive_baselines.json    eight predictors with no neural network
    results/naive_baselines2.json   the matched fitted arms
    results/collapse.json           per-patient vs cohort-mean, from saved predictions
    results/collapse_noise.json     the same, net of what averaging dropout noise alone gives
    results/naive_allfolds.json     the linear reference points on every cell, both C regimes
    results/naive_pooled.json       the pooled, frozen linear rows on every cell (pooled_baselines.py)
    results/per_patient.json        one held-out AUROC (and average precision) per cell, arm, patient
    results/phenotype_null_*.json   permutation nulls through the identical nested pipeline
    results/split_audit.json        near-duplicate held-out reactions per reaction fold (split_audit.py)
    results/cohort.json             every cohort constant, computed from the data
    results/param_counts.json       parameter counts from the construction path used

Every average records how many cells it rests on, and every comparison between
feature modes is paired on (patient fold, reaction fold) so that arms with
different numbers of completed cells are never averaged against each other.

Run from the paper1 directory:  python3 code/build_results_p1.py
"""
import json, os, re, glob, math, statistics as st

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "data", "results_p1.json")

CELL = re.compile(r"^(?P<model>.+?)_mb(?P<lam>[\d.]+)_p(?P<pf>\d+)_r(?P<rf>\d+)"
                  r"(?:_(?P<mode>mean|zero|indicator|permute|rewire\d+))?(?:_(?P<xf>rank))?(?:_ts(?P<ts>\d+))?"
                  r"(?:_synth(?P<synthkind>own)?(?P<synthfrac>[\d.]+))?(?:_(?P<ivr>ivr))?(?:_(?P<fam>fam))?\.json$")

def load_cells(sub=None):
    out = []
    for f in sorted(glob.glob(os.path.join(RES if sub is None else os.path.join(RES, sub), "*.json"))):
        m = CELL.match(os.path.basename(f))
        if not m:
            continue
        d = json.load(open(f))
        d["_model"] = m.group("model"); d["_lam"] = float(m.group("lam"))
        d["_pf"] = int(m.group("pf")); d["_rf"] = int(m.group("rf"))
        d["_mode"] = m.group("mode") or "real"
        d["_ts"] = int(m.group("ts")) if m.group("ts") else None
        d["_xf"] = m.group("xf") or "none"
        d["_ivr"] = bool(m.group("ivr"))
        d["_fam"] = bool(m.group("fam"))
        d["_synth"] = (("ownrank" if m.group("synthkind") else "cohortmedian"), float(m.group("synthfrac"))) if m.group("synthfrac") else None
        d["_tag"] = os.path.basename(f)[:-5]
        out.append(d)
    return out

def mean(xs): return st.mean(xs) if xs else None
def sd(xs):  return st.stdev(xs) if len(xs) > 1 else None

def agg(cells, key):
    v = [c[key] for c in cells if key in c]
    return dict(n=len(v), mean=mean(v), sd=sd(v))

def opt(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None

C = [c for c in load_cells() if c["_ts"] is None and c["_xf"] == "none" and not c["_ivr"] and c["_synth"] is None]
# Where the saved per-patient AUROC vectors exist (the graph model's arms), use their
# full-precision means in place of the four-decimal values the cell files carry, so that
# a standard deviation of paired differences here and in inference_p1.py agree exactly.
PP = opt("per_patient.json") or {}
for c in C:
    if c["_model"] == "gnn_B" and c["_lam"] == 0.0:
        v = PP.get(f"p{c['_pf']}r{c['_rf']}", {}).get(c["_mode"])
        if v and v.get("auroc"):
            c["heldout_AUROC_perpatient"] = st.mean(v["auroc"])
real = [c for c in C if c["_mode"] == "real" and c["_lam"] == 0.0]
grid_real = [c for c in C if c["_mode"] == "real"]          # every configuration, every lambda
NB, NB2 = opt("naive_baselines.json"), opt("naive_baselines2.json")
NA, CN = opt("naive_allfolds.json"), opt("collapse_noise.json")
PHN = {a: opt(f"phenotype_null_{a}.json") for a in ("msi", "cin_vs_gs", "sex", "site")}
PHN = {a: v for a, v in PHN.items() if v}
COL, COH, PAR = opt("collapse.json"), opt("cohort.json"), opt("param_counts.json")
INF = opt("inference.json")
AN_ = opt("arm_noise.json")
STR, PHE = opt("strata.json"), opt("phenotype_probe.json")
PHM = {a: opt(f"phenotype_{a}.json") for a in ("cin_vs_gs", "sex", "site")}
PHM = {a: v for a, v in PHM.items() if v}
NPD, SAU = opt("naive_pooled.json"), opt("split_audit.json")

if COH and isinstance(COH, dict) and "msi_counts" in COH:
    # cohort_stats.py reads the base clinical file, which carries no MSI column; the MSI annotation
    # and its counts are documented under phenotype_annotation, from clinical_metadata_msi.tsv
    COH = {k: v for k, v in COH.items() if k != "msi_counts"}
R = {"_generated_by": "code/build_results_p1.py",
     "_inputs": sorted(os.path.basename(f) for f in glob.glob(os.path.join(RES, "*.json")))}

# ---- cohort constants -------------------------------------------------------
if COH:
    R["cohort"] = COH
else:
    raise SystemExit("results/cohort.json missing: run work/cohort_stats.py first")

# ---- protocol ---------------------------------------------------------------
any_cell = real[0]
# the fold sizes as the cells record them: the common size and, where one fold differs, its own
_tr = sorted({c["n_train"] for c in real}); _te = sorted({c["n_test"] for c in real})
_last = [c for c in real if c["_pf"] == max(c2["_pf"] for c2 in real)][0]
R["protocol"] = dict(pfolds=5, rfolds=3,
                     train_patients=min(_tr), test_patients=max(_te),
                     train_patients_last=_last["n_train"], test_patients_last=_last["n_test"],
                     train_sizes=_tr, test_sizes=_te,
                     val_patients=COH["n_val_patients"],
                     heldout_reactions=any_cell["n_heldout_rxn"])

# ---- main grid, one row per architecture ------------------------------------
R["grid"] = {}
for model in sorted({c["_model"] for c in real}):
    cs = sorted([c for c in real if c["_model"] == model], key=lambda c: (c["_pf"], c["_rf"]))
    gaps = [c["trainrxn_AUROC_perpatient"] - c["heldout_AUROC_perpatient"] for c in cs]
    R["grid"][model] = dict(
        cells=len(cs), cell_tags=[c["_tag"] for c in cs],
        pfolds_covered=sorted({c["_pf"] for c in cs}),
        train_rxn=mean([c["trainrxn_AUROC_perpatient"] for c in cs]),
        heldout_rxn=mean([c["heldout_AUROC_perpatient"] for c in cs]),
        heldout_rxn_sd=sd([c["heldout_AUROC_perpatient"] for c in cs]),
        gap=mean(gaps), gap_sd=sd(gaps), gap_abs=mean([abs(g) for g in gaps]), gap_abs_sd=sd([abs(g) for g in gaps]),
        gap_negative_cells=sum(1 for g in gaps if g < 0),
        rawexpr=mean([c["baseline_rawexpr"] for c in cs]),
        indicator=mean([c["baseline_indicator"] for c in cs]),
        rho=mean([c["rho"] for c in cs]),
        interpat_r=mean([c["interpatient_r_median"] for c in cs]))
    # per-patient-fold means of held-out AUROC, and a t-test against chance on those means,
    # because the cells of one patient fold share a training set and are not independent draws
    pf = {}
    for c in cs:
        pf.setdefault(c["_pf"], []).append(c["heldout_AUROC_perpatient"])
    fm = [st.mean(v) for _, v in sorted(pf.items())]
    R["grid"][model]["pfold_means"] = fm
    if len(fm) > 1 and st.pstdev(fm) > 0:
        from scipy import stats
        tt = stats.ttest_1samp(fm, 0.5)
        R["grid"][model]["pfold_t_vs_chance"] = dict(n=len(fm), t=float(tt.statistic), p=float(tt.pvalue),
                                                     min=min(fm), max=max(fm))

# ---- input substitution, paired on (pfold, rfold) ---------------------------
def by_key(model, mode, lam=0.0):
    return {(c["_pf"], c["_rf"]): c for c in C
            if c["_model"] == model and c["_mode"] == mode and c["_lam"] == lam}

RE_all = by_key("gnn_B", "real")
ME, ZE = by_key("gnn_B", "mean"), by_key("gnn_B", "zero")
ME_all, ZE_all = dict(ME), dict(ZE)
IND = by_key("gnn_B", "indicator")     # the presence-only arm: the same binary column for every patient
# The rewiring arm proper is permutation seed 7; any further seed is a replication on its own cells
# and is summarized separately, never pooled with seed 7.
RW_BY_SEED = {}
for c in C:
    if c["_model"] == "gnn_B" and c["_lam"] == 0.0 and (c["_mode"] or "").startswith("rewire"):
        RW_BY_SEED.setdefault(int(c["_mode"][6:]), {})[(c["_pf"], c["_rf"])] = c
RW = RW_BY_SEED.get(7, {})
# The ladder rests on the cells where the substitution arms have all run, so that
# every rung is computed on the same patients and reactions and every step between
# rungs is a paired difference rather than a difference of two separate averages.
BASIS = sorted(set(RE_all) & set(ME) & set(ZE))
RE = {k: RE_all[k] for k in BASIS}
ME = {k: ME[k] for k in BASIS}
ZE = {k: ZE[k] for k in BASIS}

def paired(a, b, key="heldout_AUROC_perpatient"):
    ks = sorted(set(a) & set(b))
    d = [a[k][key] - b[k][key] for k in ks]
    return dict(n=len(ks), keys=[f"p{p}r{r}" for p, r in ks],
                delta=mean(d), sd=sd(d),
                a_mean=mean([a[k][key] for k in ks]), b_mean=mean([b[k][key] for k in ks]))

S = dict(
    real=dict(n=len(RE), keys=[f"p{p}r{r}" for p, r in sorted(RE)],
              mean=mean([c["heldout_AUROC_perpatient"] for c in RE.values()])),
    cohort_mean=dict(n=len(ME), keys=[f"p{p}r{r}" for p, r in sorted(ME)],
                     mean=mean([c["heldout_AUROC_perpatient"] for c in ME.values()])),
    zero=dict(n=len(ZE), keys=[f"p{p}r{r}" for p, r in sorted(ZE)],
              mean=mean([c["heldout_AUROC_perpatient"] for c in ZE.values()])),
    patient_identity=paired(RE, ME),      # own expression minus cohort mean, paired
    aggregate_expression=paired(ME, ZE),  # cohort mean minus zeroed, paired
)
for m in ("real", "cohort_mean", "zero"):
    src = {"real": RE, "cohort_mean": ME, "zero": ZE}[m]
    S[m]["rho"] = mean([c["rho"] for c in src.values()])
    S[m]["interpat_r"] = mean([c["interpatient_r_median"] for c in src.values()])
S["expression_total"] = paired(RE, ZE)   # own expression minus zeroed, paired
# The no-signal null of the dispersion ratio, derived rather than asserted. If a model's output
# does not depend on the patient, the only across-patient variation in the saved scores is the
# residual of averaging T dropout passes. The numerator of rho is then the standard deviation of
# n such means and the denominator the mean of their estimated standard errors, so the expected
# ratio is sqrt((n-1)/n) * c4(n) / c4(T), where c4(m) = sqrt(2/(m-1)) * Gamma(m/2)/Gamma((m-1)/2)
# is the unbiased-standard-deviation constant, n is the number of test patients in a cell and T
# the number of passes. The zeroed arm, in which every patient receives an identical input, is
# the empirical check on it.
def _c4(m):
    return math.sqrt(2.0 / (m - 1)) * math.gamma(m / 2.0) / math.gamma((m - 1) / 2.0)
_npat = mean([c["n_test"] for c in RE.values()]) if RE else None
_T = (CN or {}).get("T")
if _npat and _T:
    _n = int(round(_npat))
    S["rho_null"] = math.sqrt((_n - 1) / _n) * _c4(_n) / _c4(_T)
    S["rho_null_terms"] = dict(n_test_patients=_n, dropout_passes=_T,
                               c4_n=_c4(_n), c4_T=_c4(_T),
                               empirical_check=S["zero"]["rho"])
else:
    raise SystemExit("cannot derive the dispersion-ratio null: need cell test sizes and "
                     "results/collapse_noise.json for the pass count T")
S["basis"] = [f"p{p}r{r}" for p, r in BASIS]
S["patient_folds"] = sorted({p for p, _ in BASIS})
S["real_all_pfolds"] = dict(n=len(RE_all),
                            mean=mean([c["heldout_AUROC_perpatient"] for c in RE_all.values()]))
# The degree-preserving rewiring null, paired against the real network on its own cells
if RW:
    S["rewired"] = dict(
        n=len(RW), keys=[f"p{p}r{r}" for p, r in sorted(RW)],
        mean=mean([c["heldout_AUROC_perpatient"] for c in RW.values()]),
        sd=sd([c["heldout_AUROC_perpatient"] for c in RW.values()]),
        trainrxn=mean([c["trainrxn_AUROC_perpatient"] for c in RW.values()]),
        gap=mean([c["trainrxn_AUROC_perpatient"] - c["heldout_AUROC_perpatient"]
                  for c in RW.values()]),
        exprbearing=mean([c["heldout_exprbearing_AUROC"] for c in RW.values()]),
        rho=mean([c["rho"] for c in RW.values()]),
        interpat_r=mean([c["interpatient_r_median"] for c in RW.values()]),
        seeds=[7],
        pfolds=sorted({p for p, _ in RW}),
        vs_real=paired(RE_all, RW),
        vs_mlp=paired(by_key("mlp", "real"), RW))   # the feature-only model on the rewiring's own cells
    S["rewired_other_seeds"] = {}
    for seed, cells in sorted(RW_BY_SEED.items()):
        if seed == 7:
            continue
        S["rewired_other_seeds"][str(seed)] = dict(
            n=len(cells), keys=[f"p{p}r{r}" for p, r in sorted(cells)],
            mean=mean([c["heldout_AUROC_perpatient"] for c in cells.values()]),
            sd=sd([c["heldout_AUROC_perpatient"] for c in cells.values()]),
            exprbearing=mean([c["heldout_exprbearing_AUROC"] for c in cells.values()]),
            trainrxn=mean([c["trainrxn_AUROC_perpatient"] for c in cells.values()]),
            vs_real=paired(RE_all, cells), vs_seed7=paired(RW, cells))
# The presence-only arm, paired on its own cells against the zeroed, cohort-mean and
# unsubstituted arms of the same name: what the presence pattern of the expression column is
# worth on its own, and what the magnitudes add on top of it.
if IND:
    S["indicator"] = dict(
        n=len(IND), keys=[f"p{p}r{r}" for p, r in sorted(IND)],
        mean=mean([c["heldout_AUROC_perpatient"] for c in IND.values()]),
        sd=sd([c["heldout_AUROC_perpatient"] for c in IND.values()]),
        trainrxn=mean([c["trainrxn_AUROC_perpatient"] for c in IND.values()]),
        exprbearing=mean([c["heldout_exprbearing_AUROC"] for c in IND.values()]),
        rho=mean([c["rho"] for c in IND.values()]),
        interpat_r=mean([c["interpatient_r_median"] for c in IND.values()]),
        pfolds=sorted({p for p, _ in IND}),
        presence_over_zero=paired(IND, ZE_all),       # indicator minus zeroed
        magnitude_over_presence=paired(ME_all, IND),  # cohort mean minus indicator
        own_over_presence=paired(RE_all, IND),        # own expression minus indicator
        cohort_mean_over_zero_same_cells=dict(
            n=len(set(IND) & set(ME_all) & set(ZE_all)),
            delta=mean([ME_all[k]["heldout_AUROC_perpatient"] - ZE_all[k]["heldout_AUROC_perpatient"]
                        for k in sorted(set(IND) & set(ME_all) & set(ZE_all))])))
# The wrong-patient arm: within each split a seeded derangement of the training-cohort patient
# index shows every patient another patient's real expression column, so the input keeps the shape
# and the between-patient variance of a real profile and only its owner changes. It is the sharper
# version of the cohort-mean substitution and it lands, if it lands, in results/permute/ with the
# same record format as the other arms. Paired against the unsubstituted cells of the same name,
# in the same direction as patient_identity: own column minus someone else's.
PM = {(c["_pf"], c["_rf"]): c for c in load_cells("permute")
      if c["_model"] == "gnn_B" and c["_lam"] == 0.0 and c["_mode"] == "permute" and c["_ts"] is None}
if PM:
    ks = sorted(set(PM) & set(RE_all))
    d = [RE_all[k]["heldout_AUROC_perpatient"] - PM[k]["heldout_AUROC_perpatient"] for k in ks]
    S["permute"] = dict(
        n=len(PM), keys=[f"p{p}r{r}" for p, r in sorted(PM)],
        pfolds=sorted({p for p, _ in PM}),
        mean=mean([c["heldout_AUROC_perpatient"] for c in PM.values()]),
        sd=sd([c["heldout_AUROC_perpatient"] for c in PM.values()]),
        trainrxn=mean([c["trainrxn_AUROC_perpatient"] for c in PM.values()]),
        exprbearing=mean([c["heldout_exprbearing_AUROC"] for c in PM.values()]),
        rho=mean([c["rho"] for c in PM.values()]),
        interpat_r=mean([c["interpatient_r_median"] for c in PM.values()]),
        patient_identity=dict(
            n=len(ks), keys=[f"p{p}r{r}" for p, r in ks], delta=mean(d), sd=sd(d),
            per_cell={f"p{p}r{r}": x for (p, r), x in zip(ks, d)},
            a_mean=mean([RE_all[k]["heldout_AUROC_perpatient"] for k in ks]),
            b_mean=mean([PM[k]["heldout_AUROC_perpatient"] for k in ks]),
            n_positive=sum(1 for x in d if x > 0), n_negative=sum(1 for x in d if x < 0)))
    # the cohort-mean arm on exactly those cells, so the two substitutions can be read side by side
    kc = sorted(set(PM) & set(ME_all))
    if kc:
        dc = [RE_all[k]["heldout_AUROC_perpatient"] - ME_all[k]["heldout_AUROC_perpatient"] for k in kc]
        S["permute"]["cohort_mean_same_cells"] = dict(
            n=len(kc), mean=mean([ME_all[k]["heldout_AUROC_perpatient"] for k in kc]),
            delta=mean(dc), sd=sd(dc),
            n_positive=sum(1 for x in dc if x > 0), n_negative=sum(1 for x in dc if x < 0))
R["substitution"] = S

# ---- naive baselines --------------------------------------------------------
# The unfitted reference rows come from naive_allfolds.json (every cell, canonical annotation).
# The two fold-0 suites that preceded it indexed the BiGG JSON by position and are retired.
def nb_block(d):
    return {k: dict(cells=v, mean=mean(v), sd=sd(v))
            for k, v in d.items() if isinstance(v, list)}
R["naive_baselines"] = nb_block(NB) if NB else {}
if NB2:
    R["naive_baselines"].update(nb_block(NB2))
    R["naive_baselines"]["_n_test_patients"] = NB2.get("_n_test_patients")
if NA:
    T_ = NA.get("tuned", {})
    for k, nk in (("indicator", "indicator"), ("expr_perpat_rank", "expr_perpat"),
                  ("expr_cohortmean_rank", "expr_cohortmean"), ("degree_shared", "degree_shared"),
                  ("subsystem_prevalence", "subsystem_prevalence")):
        if k in T_:
            v = [x for _, x in sorted(T_[k].items())]
            R["naive_baselines"][nk] = dict(cells=v, mean=mean(v), sd=sd(v))
    R["naive_baselines"]["chance"] = dict(cells=[0.5] * len(T_.get("indicator", {})), mean=0.5, sd=0.0)
    R["naive_baselines"]["_n_test_patients"] = (NA.get("_n_test_patients") or {}).get("p0")

# ---- the decomposition ladder --------------------------------------------------------
# The ladder proper is built below on the substitution basis from naive_allfolds.json. This block
# keeps the graph rungs and the unfitted reference rows for figures and text that predate it.
nb = R["naive_baselines"]
L = {}
if nb:
    for k in ("indicator", "expr_perpat", "expr_cohortmean"):
        if k in nb: L[k] = nb[k]["mean"]
L["gnn_zero"] = S["zero"]["mean"]
L["gnn_cohort_mean"] = S["cohort_mean"]["mean"]
L["gnn_real"] = S["real"]["mean"]
if L.get("expr_perpat") is not None and L.get("expr_cohortmean") is not None:
    L["nonparam_patient_effect"] = L["expr_perpat"] - L["expr_cohortmean"]
R["ladder"] = L

# ---- the linear reference points on the same cells as the graph arms ----------------
# naive_allfolds.json holds every linear predictor on every (patient fold, reaction fold)
# cell, in two regimes: the fixed C = 1 of the earlier suites, and C tuned by inner
# cross-validation on train reactions. The ladder proper is restated on the substitution
# basis so that every row, linear or graph, rests on the identical cells.
if NA:
    basis_keys = set(S["basis"])
    R["naive_allfolds"] = dict(regimes={}, columns=NA.get("_columns"), C_grid=NA.get("_C_grid"))
    for regime in ("fixed", "tuned"):
        G = NA.get(regime, {})
        block = {}
        for k, cells in G.items():
            allv = [v for _, v in sorted(cells.items())]
            bv = [v for c, v in sorted(cells.items()) if c in basis_keys]
            block[k] = dict(n_all=len(allv), mean_all=mean(allv), sd_all=sd(allv),
                            n_basis=len(bv), mean_basis=mean(bv), sd_basis=sd(bv),
                            cells=dict(sorted(cells.items())))
        R["naive_allfolds"]["regimes"][regime] = block
    # the ladder on the common basis, tuned regime for the fitted rows
    T = R["naive_allfolds"]["regimes"]["tuned"]; F = R["naive_allfolds"]["regimes"]["fixed"]
    LB = {}
    def mb(block, k): return block[k]["mean_basis"] if k in block and block[k]["n_basis"] else None
    LB["n_basis"] = T["structure_only"]["n_basis"] if "structure_only" in T else 0
    LB["indicator"] = mb(T, "indicator")
    LB["expr_perpat"] = mb(T, "expr_perpat_rank"); LB["expr_cohortmean"] = mb(T, "expr_cohortmean_rank")
    for k in ("expr_cohortmean_lr", "expr_perpat_lr", "topology_only", "annotation_only",
              "structure_only", "structure_plus_cohortmean", "structure_plus_perpat"):
        LB[k] = mb(T, k); LB[k + "_fixedC"] = mb(F, k)
    LB["gnn_zero"] = S["zero"]["mean"]; LB["gnn_cohort_mean"] = S["cohort_mean"]["mean"]
    LB["gnn_real"] = S["real"]["mean"]
    if LB.get("structure_only") is not None and LB.get("expr_cohortmean_lr") is not None:
        LB["structure_over_expression_fitted"] = LB["structure_only"] - LB["expr_cohortmean_lr"]
        LB["topology_over_expression_fitted"] = LB["topology_only"] - LB["expr_cohortmean_lr"]
        LB["annotation_over_expression_fitted"] = LB["annotation_only"] - LB["expr_cohortmean_lr"]
        LB["gnn_over_lr_no_expression"] = LB["gnn_zero"] - LB["structure_only"]
        LB["gnn_over_lr_with_expression"] = LB["gnn_cohort_mean"] - LB["structure_plus_cohortmean"]
        LB["gnn_over_lr_own_expression"] = LB["gnn_real"] - LB["structure_plus_perpat"]
        LB["gnn_mean_over_structure"] = LB["gnn_cohort_mean"] - LB["structure_only"]
        LB["gnn_real_over_structure"] = LB["gnn_real"] - LB["structure_only"]
        LB["lin_patient_expr_only"] = LB["expr_perpat_lr"] - LB["expr_cohortmean_lr"]
        LB["lin_patient_struct_expr"] = LB["structure_plus_perpat"] - LB["structure_plus_cohortmean"]
        LB["nonparam_patient_effect"] = LB["expr_perpat"] - LB["expr_cohortmean"]
        _a, _b = T.get("expr_perpat_rank", {}).get("cells", {}), T.get("expr_cohortmean_rank", {}).get("cells", {})
        LB["nonparam_positive_cells"] = sum(1 for c in basis_keys if c in _a and c in _b and _a[c] > _b[c])
    R["ladder_basis"] = LB

# ---- paired differences between two per-cell dictionaries, restricted to the basis ---------------
def _paired_dict(a, b, keys_subset=None):
    """Paired difference a minus b over the cells both carry (and the basis when given), with the
    sign counts by cell and by reaction fold, in the format of the linear-row deltas above."""
    ks = sorted(set(a) & set(b) & (set(keys_subset) if keys_subset is not None else set(a)))
    if not ks:
        return None
    dd = [a[c] - b[c] for c in ks]
    byr = {}
    for c, v in zip(ks, dd):
        byr.setdefault(c.split("r")[-1], []).append(v)
    rm = [mean(v) for _, v in sorted(byr.items())]
    return dict(n=len(ks), keys=ks, delta=mean(dd), sd=sd(dd), per_cell={c: v for c, v in zip(ks, dd)},
                a_mean=mean([a[c] for c in ks]), b_mean=mean([b[c] for c in ks]),
                n_positive=sum(1 for v in dd if v > 0), n_negative=sum(1 for v in dd if v < 0),
                n_rfolds=len(rm), n_rfolds_positive=sum(1 for v in rm if v > 0),
                n_rfolds_negative=sum(1 for v in rm if v < 0))

# the graph arms as per-cell dictionaries keyed like the linear rows (full-precision means where the
# per-patient vectors exist, the cell files otherwise)
def _gnn_cells(cells):
    return {f"p{p}r{r}": c["heldout_AUROC_perpatient"] for (p, r), c in cells.items()}
GNN_CELLS = dict(real=_gnn_cells(RE_all), cohort_mean=_gnn_cells(ME_all), zero=_gnn_cells(ZE_all))

# ---- the pooled, frozen linear rows on the same cells (pooled_baselines.py) -------------------
# One l2-penalized logistic regression per cell, fitted on the stacked rows of a seeded subsample of
# training patients crossed with the training reactions, frozen, and scored on each test patient's
# own held-out rows. The per-patient rows of naive_allfolds.json fit one model per test patient
# instead; the two are paired cell by cell here, on the substitution basis.
if NPD:
    basis_keys = set(S["basis"])
    NPB = dict(regimes={}, columns=NPD.get("_columns"), C_grid=NPD.get("_C_grid"),
               subsample=dict(rule=(NPD.get("_subsample") or {}).get("rule"),
                              max_train_patients=(NPD.get("_subsample") or {}).get("max_train_patients"),
                              n_used_by_pfold=NPD.get("_n_train_patients_used"),
                              n_used_min=(min(NPD["_n_train_patients_used"].values()) if NPD.get("_n_train_patients_used") else None),
                              n_used_max=(max(NPD["_n_train_patients_used"].values()) if NPD.get("_n_train_patients_used") else None)),
               n_train_rows=NPD.get("_n_train_rows"), inner_cv=NPD.get("_inner_cv"),
               C_selected=NPD.get("_C_selected"), expr_transform=NPD.get("_expr_transform"), data=NPD.get("_data"))
    for regime in ("fixed", "tuned"):
        block = {}
        for k, cells in (NPD.get(regime) or {}).items():
            allv = [v for _, v in sorted(cells.items())]
            bv = [v for c, v in sorted(cells.items()) if c in basis_keys]
            block[k] = dict(n_all=len(allv), mean_all=mean(allv), sd_all=sd(allv),
                            n_basis=len(bv), mean_basis=mean(bv), sd_basis=sd(bv), cells=dict(sorted(cells.items())))
        NPB["regimes"][regime] = block
    NPB["n_basis"] = min([v["n_basis"] for v in NPB["regimes"]["tuned"].values()] or [0])
    NPB["basis_complete"] = all(v["n_basis"] == len(basis_keys) for v in NPB["regimes"]["tuned"].values()) if NPB["regimes"]["tuned"] else False
    NPB["n_train_rows_mean"] = mean([v for c, v in (NPD.get("_n_train_rows") or {}).items() if c in basis_keys]) if NPD.get("_n_train_rows") else None
    NPB["deltas"] = {}
    for regime in ("fixed", "tuned"):
        P_ = NPD.get(regime) or {}; Q_ = (NA or {}).get(regime) or {}
        dd = {}
        for nm, pk, qk in (("pooled_vs_perpat_expr", "expr_pooled_lr", "expr_perpat_lr"),
                           ("pooled_vs_cohort_expr", "expr_pooled_lr", "expr_cohortmean_lr"),
                           ("pooled_vs_perpat_struct", "structure_plus_pooled", "structure_plus_perpat"),
                           ("pooled_vs_cohort_struct", "structure_plus_pooled", "structure_plus_cohortmean"),
                           ("pooled_struct_vs_structure_only", "structure_plus_pooled", "structure_only"),
                           ("pooled_struct_vs_pooled_expr", "structure_plus_pooled", "expr_pooled_lr")):
            if pk in P_ and qk in Q_:
                dd[nm] = _paired_dict(P_[pk], Q_[qk], basis_keys)
        # the graph arms against the frozen linear model given the same information
        for nm, gk, pk in (("gnn_real_vs_pooled_struct", "real", "structure_plus_pooled"),
                           ("gnn_cohort_mean_vs_pooled_struct", "cohort_mean", "structure_plus_pooled"),
                           ("gnn_real_vs_pooled_expr", "real", "expr_pooled_lr")):
            if pk in P_:
                dd[nm] = _paired_dict(GNN_CELLS[gk], P_[pk], basis_keys)
        NPB["deltas"][regime] = dd
    R["naive_pooled"] = NPB

# ---- average precision beside the AUROC, on the substitution basis ------------------------------
# The graph arms' per-patient average precision comes from per_patient.json (per_patient_auc.py),
# the linear rows' from the "auprc" block of naive_allfolds.json and naive_pooled.json. A row is
# summarized only where every basis cell carries it, so the AUPRC ladder rests on the same cells as
# the AUROC ladder; the held-out prevalence is the average precision of a random ranking.
AP = dict(basis=S["basis"], n_basis=len(S["basis"]), rows={}, rows_fixedC={}, deltas={}, prevalence=None)
basis_keys = set(S["basis"])
def _ap_row(cells, name, store):
    bv = {c: v for c, v in cells.items() if c in basis_keys}
    if bv and len(bv) == len(basis_keys):
        store[name] = dict(n=len(bv), mean=mean(bv.values()), sd=sd(list(bv.values())), cells=dict(sorted(bv.items())))
for arm, nm in (("real", "gnn_real"), ("mean", "gnn_cohort_mean"), ("zero", "gnn_zero")):
    cells = {c: st.mean(v[arm]["auprc"]) for c, v in PP.items() if arm in v and v[arm].get("auprc")}
    _ap_row(cells, nm, AP["rows"])
# the arms that run on their own cells: summarized on those cells, with the cell list
for arm, nm in (("indicator", "gnn_indicator"), ("rewire7", "gnn_rewire7")):
    cells = {c: st.mean(v[arm]["auprc"]) for c, v in PP.items() if arm in v and v[arm].get("auprc")}
    if cells:
        AP["rows"][nm] = dict(n=len(cells), mean=mean(cells.values()), sd=sd(list(cells.values())),
                              cells=dict(sorted(cells.items())), own_cells=True)
for src, store_t, store_f in ((NA, AP["rows"], AP["rows_fixedC"]), (NPD, AP["rows"], AP["rows_fixedC"])):
    for regime, store in (("tuned", store_t), ("fixed", store_f)):
        for k, cells in (((src or {}).get("auprc") or {}).get(regime) or {}).items():
            _ap_row(cells, k, store)
# the prevalence of the held-out reaction sets, from whichever file carries it
_prev = {}
for src in (NA, NPD):
    _prev.update((src or {}).get("_heldout_prevalence") or {})
for c, v in PP.items():
    for arm in v.values():
        if arm.get("prevalence") is not None:
            _prev.setdefault(c, arm["prevalence"])
_pb = {c: v for c, v in _prev.items() if c in basis_keys}
if _pb and len(_pb) == len(basis_keys):
    byr = {}
    for c, v in _pb.items():
        byr.setdefault("r" + c.split("r")[-1], []).append(v)
    AP["prevalence"] = dict(mean=mean(_pb.values()), by_rfold={k: mean(v) for k, v in sorted(byr.items())},
                            min=min(_pb.values()), max=max(_pb.values()), n=len(_pb))
_ap_rows = AP["rows"]
def _apd(nm, a, b):
    if a in _ap_rows and b in _ap_rows:
        AP["deltas"][nm] = _paired_dict(_ap_rows[a]["cells"], _ap_rows[b]["cells"], basis_keys)
_apd("patient_identity", "gnn_real", "gnn_cohort_mean"); _apd("aggregate_expression", "gnn_cohort_mean", "gnn_zero")
_apd("expression_total", "gnn_real", "gnn_zero")
_apd("lin_patient_expr_only", "expr_perpat_lr", "expr_cohortmean_lr")
_apd("lin_patient_struct_expr", "structure_plus_perpat", "structure_plus_cohortmean")
_apd("nonparam_patient_effect", "expr_perpat_rank", "expr_cohortmean_rank")
_apd("structure_over_expression_fitted", "structure_only", "expr_cohortmean_lr")
_apd("gnn_over_lr_no_expression", "gnn_zero", "structure_only")
_apd("gnn_over_lr_with_expression", "gnn_cohort_mean", "structure_plus_cohortmean")
_apd("gnn_over_lr_own_expression", "gnn_real", "structure_plus_perpat")
_apd("gnn_real_over_structure", "gnn_real", "structure_only")
_apd("pooled_vs_perpat_expr", "expr_pooled_lr", "expr_perpat_lr"); _apd("pooled_vs_cohort_expr", "expr_pooled_lr", "expr_cohortmean_lr")
_apd("pooled_vs_perpat_struct", "structure_plus_pooled", "structure_plus_perpat")
_apd("pooled_vs_cohort_struct", "structure_plus_pooled", "structure_plus_cohortmean")
_apd("gnn_real_vs_pooled_struct", "gnn_real", "structure_plus_pooled")
if AP["rows"] or AP["prevalence"]:
    R["auprc"] = AP

# ---- per-score Monte Carlo error in each arm, from the saved uncertainties (arm_noise.py) ----
if AN_:
    R["arm_noise"] = dict(T=AN_.get("T"), summary=AN_["summary"])

# ---- cohort-mean collapse ---------------------------------------------------
if COL:
    R["collapse"] = COL["summary"]
if CN:
    R["collapse_noise"] = dict(summary=CN["summary"], T=CN.get("T"), replicates=CN.get("replicates"))

# ---- inference -------------------------------------------------------------
if INF:
    R["inference"] = INF["comparisons"]
    R["inference_reproducibility"] = INF.get("reproducibility", {})

# ---- where the structural signal lives ------------------------------------
if STR:
    R["strata"] = dict(sizes=STR["strata_sizes"], prevalence=STR["strata_prevalence"],
                       rollup=STR["rollup"], paired=STR.get("paired"), heldout_mean=STR.get("strata_heldout_mean"),
                       basis_n=(len(STR["basis"]) if STR.get("basis") is not None else None))

# ---- a phenotype that varies between patients ------------------------------
# The probe is run once per reaction fold, each run with its own patient-level bootstrap interval.
# The "lo" and "hi" of the arm summaries below are the envelope of those per-fold intervals (the
# lowest lower bound and the highest upper bound), which is not an interval for the mean over folds.
# The per-fold records and the aggregate that follow are the reading that keeps the folds apart: one
# record per fold with its own interval and, where a permutation null was run at that fold, its p;
# and the mean of the fold AUROCs, with a bootstrap interval only where the probe file stores the
# out-of-fold predictions (a rerun of phenotype_probe.py writes them), formed by resampling
# patients once per replicate and applying the same resample to every fold.
def _strip_oof(v):
    return {k: x for k, x in v.items() if k != "oof"} if isinstance(v, dict) else v

def _auroc(y, s):
    """Mann-Whitney AUROC through midranks, so that ties are scored one half."""
    from scipy.stats import rankdata
    y = [int(v) for v in y]; n1 = sum(y); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return None
    r = rankdata(s)
    return (sum(ri for ri, yi in zip(r, y) if yi) - n1 * (n1 + 1) / 2.0) / (n1 * n0)

def _phe_folds(per, null_by_fold=None):
    """One record per reaction fold of an arm: fold, AUROC, its bootstrap bounds, patients, positives
    and the permutation p of the nested null at that fold where one was run."""
    recs = []
    for rk in ("r0", "r1", "r2"):
        v = per.get(rk)
        if not v:
            continue
        recs.append(dict(fold=int(rk[1]), auroc=v["auroc"], lo=v["ci_lo"], hi=v["ci_hi"], n=v["n"], n_pos=v["n_pos"],
                         perm_p=((null_by_fold or {}).get(rk) or {}).get("perm_p"),
                         perm_null_mean=((null_by_fold or {}).get(rk) or {}).get("perm_null_mean"),
                         perm_null_max=((null_by_fold or {}).get(rk) or {}).get("perm_null_max"),
                         oof_stored=bool(v.get("oof"))))
    return recs

def _phe_aggregate(per, recs, n_boot=1000, seed=2024):
    """The mean of the fold AUROCs, its spread, the envelope of the per-fold intervals for reference,
    and a patient-resampling bootstrap interval for the mean when the out-of-fold predictions of
    every fold are stored; otherwise a statement that none can be formed from the summaries."""
    if not recs:
        return None
    aurocs = [r["auroc"] for r in recs]
    agg = dict(n_rfolds=len(recs), mean=mean(aurocs), sd=sd(aurocs), min=min(aurocs), max=max(aurocs),
               envelope_lo=min(r["lo"] for r in recs), envelope_hi=max(r["hi"] for r in recs),
               n=recs[0]["n"], n_pos=recs[0]["n_pos"], lo=None, hi=None, interval=None, n_boot=None)
    oofs = [(per.get(f"r{r['fold']}") or {}).get("oof") for r in recs]
    if all(o and o.get("patient_ids") and o.get("score") for o in oofs):
        import numpy as np
        common = set(oofs[0]["patient_ids"])
        for o in oofs[1:]:
            common &= set(o["patient_ids"])
        common = sorted(common)
        if len(common) >= 40:
            Y = []; SC = []
            for o in oofs:
                pos = {p: i for i, p in enumerate(o["patient_ids"])}
                Y.append(np.array([o["y"][pos[p]] for p in common], int)); SC.append(np.array([o["score"][pos[p]] for p in common], float))
            if all((Y[0] == yy).all() for yy in Y[1:]):
                rng = np.random.default_rng(seed); boots = []
                for _ in range(n_boot):
                    i = rng.integers(0, len(common), len(common))
                    vals = [_auroc(Y[0][i], sc[i]) for sc in SC]
                    if all(v is not None for v in vals):
                        boots.append(float(np.mean(vals)))
                if boots:
                    # the stored per-fold AUROC recomputed from the stored predictions: the interval is
                    # kept only when the two agree, so it can never sit beside a mean it does not describe
                    full = [_auroc(Y[0], sc) for sc in SC]
                    m_oof = mean(full) if all(v is not None for v in full) else None
                    agg["mean_from_oof"] = m_oof; agg["n_common_patients"] = len(common)
                    if m_oof is not None and abs(m_oof - agg["mean"]) < 1e-3:
                        agg.update(lo=float(np.percentile(boots, 2.5)), hi=float(np.percentile(boots, 97.5)), n_boot=len(boots),
                                   interval="percentile interval of the mean over reaction folds from %d patient resamples of the "
                                            "stored out-of-fold predictions, the same resample applied to every fold, seed %d"
                                            % (len(boots), seed))
                    else:
                        agg["note"] = ("the stored out-of-fold predictions do not reproduce the stored per-fold AUROC "
                                       "(%.4f against %.4f), so no aggregate interval is formed from them" % (m_oof or float("nan"), agg["mean"]))
    if agg["lo"] is None and "note" not in agg:
        agg["note"] = ("no aggregate interval can be formed from the stored summaries: the probe file carries the "
                       "per-fold AUROC and the per-fold bootstrap bounds but not the out-of-fold predictions; "
                       "the per-fold intervals are reported instead")
    return agg

if PHE:
    arms = {}
    for mode, per in PHE["arms"].items():
        vals = [v for k, v in per.items() if k in ("r0", "r1", "r2")]
        if vals:
            arms[mode] = dict(
                n_rfolds=len(vals),
                auroc=mean([v["auroc"] for v in vals]),
                sd=sd([v["auroc"] for v in vals]),
                lo=min(v["ci_lo"] for v in vals), hi=max(v["ci_hi"] for v in vals),
                n_patients=vals[0]["n"], n_pos=vals[0]["n_pos"],
                per_rfold={k: v["auroc"] for k, v in per.items() if k in ("r0", "r1", "r2")})
        if "r0_with_permutation" in per:
            arms.setdefault(mode, {})["permutation"] = _strip_oof(per["r0_with_permutation"])
    R["phenotype"] = dict(target=PHE["phenotype"], arms=arms,
                          input=_strip_oof(PHE.get("input_reaction_expression")),
                          genes=_strip_oof(PHE.get("gene_level_reference")))
    # the per-fold reading, and the aggregate with its own interval where one can be formed
    _null = (PHN.get("msi") or {})
    R["phenotype"]["per_fold"] = {}; R["phenotype"]["aggregate"] = {}
    for mode, per in PHE["arms"].items():
        recs = _phe_folds(per, _null.get("output_by_fold") if mode == "real" else None)
        if recs:
            R["phenotype"]["per_fold"][mode] = recs
            R["phenotype"]["aggregate"][mode] = _phe_aggregate(per, recs)
    _in = PHE.get("input_reaction_expression")
    if _in:
        R["phenotype"]["per_fold"]["input"] = [dict(fold=None, auroc=_in["auroc"], lo=_in["ci_lo"], hi=_in["ci_hi"],
                                                    n=_in["n"], n_pos=_in["n_pos"],
                                                    perm_p=((_null.get("input") or {}).get("perm_p")),
                                                    note="one fit on the patients of the first fold's assembly; the input does not vary with the reaction fold")]
    R["phenotype"]["interval_note"] = ("arms[*].lo and .hi are the envelope of the per-fold bootstrap intervals; "
                                       "per_fold carries each fold's own interval and aggregate the mean over folds")

# the same probe on the matched-rule outputs (phenotype_probe_ivr.json). The grid's-rule figure above
# and this one come from different training runs and are never mixed; both are reported, each named
# by the protocol that produced it.
PHEIVR = opt("phenotype_probe_ivr.json")
if PHEIVR:
    _a = {}
    for mode, per in PHEIVR["arms"].items():
        vals = [v for k, v in per.items() if k in ("r0", "r1", "r2")]
        if not vals: continue
        _a[mode] = dict(n_rfolds=len(vals), auroc=mean([v["auroc"] for v in vals]), sd=sd([v["auroc"] for v in vals]),
                        lo=min(v["ci_lo"] for v in vals), hi=max(v["ci_hi"] for v in vals),
                        n_patients=vals[0]["n"], n_pos=vals[0]["n_pos"],
                        per_rfold={k: v["auroc"] for k, v in per.items() if k in ("r0", "r1", "r2")})
    R["phenotype_matched"] = dict(protocol=PHEIVR.get("protocol"), target=PHEIVR["phenotype"], arms=_a,
                                  input=_strip_oof(PHEIVR.get("input_reaction_expression")),
                                  genes=_strip_oof(PHEIVR.get("gene_level_reference")),
                                  note="the probe of phenotype_probe.py rerun on the matched-rule score matrices; "
                                       "arms and folds as in the grid's-rule block, never pooled with it")

# ---- permutation nulls through the identical nested pipeline ------------------------
if PHN:
    R["phenotype_null"] = {}
    for a, v in PHN.items():
        R["phenotype_null"][a] = dict(label=v["label"], arm=v["arm"],
            output={k: v["output"][k] for k in ("auroc", "n", "n_pos", "n_perm", "perm_null_mean",
                                                 "perm_null_sd", "perm_null_max", "perm_p", "perm_p_floor")
                    if k in v["output"]},
            # the null at every reaction fold; the headline p above is the least favorable fold's
            output_by_fold={rf: {k: w[k] for k in ("auroc", "perm_null_mean", "perm_null_sd", "perm_null_max", "perm_p")}
                            for rf, w in (v.get("output_by_fold") or {}).items()},
            input=({k: v["input"][k] for k in ("auroc", "n_perm", "perm_null_mean", "perm_null_sd",
                                                "perm_null_max", "perm_p", "perm_p_floor")}
                   if v.get("input") else None))

# ---- the same probe on further attributes -----------------------------------
if PHM:
    R["phenotype_multi"] = {}
    for a, v in PHM.items():
        real = [x["auroc"] for k, x in v["arms"].get("real", {}).items() if k.startswith("r")]
        rec = dict(label=v["label"], n_rfolds=len(real), real=(mean(real) if real else None),
                   real_sd=(sd(real) if len(real) > 1 else None),
                   real_lo=(min(x["ci_lo"] for x in v["arms"]["real"].values()) if real else None),
                   real_hi=(max(x["ci_hi"] for x in v["arms"]["real"].values()) if real else None))
        if real:
            any_r = next(iter(v["arms"]["real"].values()))
            rec.update(n=any_r["n"], n_pos=any_r["n_pos"])
        m = v["arms"].get("mean", {}).get("r0")
        if m: rec.update(cohort_mean=m["auroc"], cohort_mean_lo=m["ci_lo"], cohort_mean_hi=m["ci_hi"],
                         cohort_mean_n=m["n"], cohort_mean_pos=m["n_pos"])
        i = v.get("input_reaction_expression")
        if i: rec.update(input=i["auroc"], input_lo=i["ci_lo"], input_hi=i["ci_hi"])
        # the per-fold reading of the unsubstituted arm (three folds) and the cohort-mean arm (its
        # folds as run), with the fold-wise permutation p where the null exists
        _null = (PHN.get(a) or {})
        rec["per_fold"] = {}; rec["aggregate"] = {}
        for mode, per in v["arms"].items():
            recs = _phe_folds(per, _null.get("output_by_fold") if mode == "real" else None)
            if recs:
                rec["per_fold"][mode] = recs; rec["aggregate"][mode] = _phe_aggregate(per, recs)
        if i:
            rec["per_fold"]["input"] = [dict(fold=None, auroc=i["auroc"], lo=i["ci_lo"], hi=i["ci_hi"], n=i["n"], n_pos=i["n_pos"],
                                             perm_p=((_null.get("input") or {}).get("perm_p")),
                                             note="one fit on the patients of the first fold's assembly")]
        R["phenotype_multi"][a] = rec

# ---- the nearest-family label lookup, and a verification rerun of the linear suite ----------
# naive_allfolds_rerun.json is the linear suite rerun from the same code on a later day with no
# family map, for two purposes. It carries the nearest-family label lookup row, which the committed
# file predates, and every row the two files share is compared here cell by cell, so the claim that
# the linear suite reproduces exactly is checked at build time rather than asserted in prose.
NAR_ = opt("naive_allfolds_rerun.json")
if NAR_ and NA:
    def _cells_of(b): return b.get("cells") if isinstance(b, dict) and "cells" in b else (b if isinstance(b, dict) else None)
    _worst, _nrow, _ncell = 0.0, 0, 0
    for _reg in ("fixed", "tuned"):
        for _k in sorted(set(NA.get(_reg) or {}) & set(NAR_.get(_reg) or {})):
            _a, _b = _cells_of(NA[_reg][_k]), _cells_of(NAR_[_reg][_k])
            if not _a or not _b: continue
            _ks = sorted(set(_a) & set(_b))
            if not _ks: continue
            _nrow += 1; _ncell += len(_ks)
            _worst = max(_worst, max(abs(_a[c] - _b[c]) for c in _ks))
    _look = _cells_of((NAR_.get("tuned") or {}).get("nearest_lookup_k5") or {})
    R["linear_rerun"] = dict(
        note=("the linear reference suite rerun from the same code with no family map: every row the "
              "rerun shares with the committed file is compared cell by cell here"),
        rows_compared=_nrow, cells_compared=_ncell, max_abs_difference=_worst,
        reproduces_exactly=(_worst == 0.0),
        nearest_lookup=(dict(n=len(_look), mean=mean(list(_look.values())), sd=sd(list(_look.values())),
                             cells=dict(sorted(_look.items()))) if _look else None))
    # the same row on the family-disjoint folds, and the paired difference between the two splits
    _NAFe = opt("naive_allfolds_fam.json")
    if _look and _NAFe and (_NAFe.get("tuned") or {}).get("nearest_lookup_k5"):
        _lf = _cells_of(_NAFe["tuned"]["nearest_lookup_k5"])
        _ks = sorted(set(_look) & set(_lf))
        if len(_ks) > 1:
            _d = [_lf[c] - _look[c] for c in _ks]
            R["linear_rerun"]["nearest_lookup_family"] = dict(
                n=len(_ks), mean=mean([_lf[c] for c in _ks]), random_mean=mean([_look[c] for c in _ks]),
                delta=mean(_d), sd=sd(_d), n_negative=sum(1 for x in _d if x < 0),
                per_cell=dict(zip(_ks, _d)))

# ---- the split audit: near-duplicate held-out reactions (split_audit.py) ---------------------
if SAU:
    R["split_audit"] = dict(all=SAU["all"], folds=SAU["folds"], criteria=SAU.get("criteria"),
                            checks=SAU.get("checks"), source=SAU.get("source"), rfolds=SAU.get("rfolds"))
# the identical audit rerun against the family-disjoint folds (split_audit.py --family_map), so the
# two splits are priced by the same code on the same criteria and the residual is stated, not assumed
SAF = opt("split_audit_fam.json")
if SAF:
    R["split_audit_family"] = dict(all=SAF["all"], folds=SAF["folds"], rfolds=SAF.get("rfolds"),
                                   split=SAF.get("split"), family_scheme=SAF.get("family_scheme"),
                                   family_map=SAF.get("family_map"))

# ---- the phenotype annotation file, so the paper can say how the attributes overlap ---------
CLIN = os.path.join(HERE, "data", "clinical_metadata_msi.tsv")
if os.path.exists(CLIN):
    import csv
    rows = list(csv.DictReader(open(CLIN), delimiter="\t"))
    sub = lambda r: (r.get("SUBTYPE") or "").strip(); msi = lambda r: (r.get("msi_status") or "").strip()
    cin = [r for r in rows if sub(r) in ("COAD_CIN", "READ_CIN")]; gs = [r for r in rows if sub(r) in ("COAD_GS", "READ_GS")]
    R["phenotype_annotation"] = dict(
        n_rows=len(rows), n_cin=len(cin), n_gs=len(gs),
        cin_msih=sum(1 for r in cin if msi(r) == "MSI-H"), gs_msih=sum(1 for r in gs if msi(r) == "MSI-H"),
        cin_mss=sum(1 for r in cin if msi(r) == "MSS"), gs_mss=sum(1 for r in gs if msi(r) == "MSS"),
        n_msi_subtype=sum(1 for r in rows if sub(r) in ("COAD_MSI", "READ_MSI")),
        n_pole=sum(1 for r in rows if sub(r) in ("COAD_POLE", "READ_POLE")),
        n_no_subtype=sum(1 for r in rows if not sub(r)),
        msi_counts={k: sum(1 for r in rows if msi(r) == k) for k in ("MSI-H", "MSI-L", "MSS", "NOT EVALUABLE")})

# ---- parameter counts and status -------------------------------------------
# ---- robustness checks ----------------------------------------------------------------
# Three checks on the fold-0 cells, each kept apart from the main grid rather than pooled with it:
#   results/ht29/               gnn_B real / cohort-mean / zeroed arms, labels = the HT29 reconstruction
#                               alone (the one colorectal line of the eleven), with the linear rows on
#                               the same label in results/naive_allfolds_ht29.json
#   results/seedrep/            gnn_B real / cohort-mean arms on the same cells under a second training
#                               seed (folds unchanged), tag suffix _ts<seed>
#   results/naive_allfolds_rank.json   the linear rows with every patient's expression column replaced
#                               by its within-patient percentile (a second normalization), union label
ROB = {}
def _cellmap(cells):
    return {(c["_pf"], c["_rf"]): c for c in cells}
def _paired_cells(a, b, key="heldout_AUROC_perpatient"):
    ks = sorted(set(a) & set(b))
    d = [a[k][key] - b[k][key] for k in ks]
    return dict(n=len(ks), keys=[f"p{p}r{r}" for p, r in ks], delta=mean(d), sd=sd(d),
                per_cell={f"p{p}r{r}": x for (p, r), x in zip(ks, d)},
                a_mean=mean([a[k][key] for k in ks]), b_mean=mean([b[k][key] for k in ks]),
                n_positive=sum(1 for x in d if x > 0), n_negative=sum(1 for x in d if x < 0))
def _arm_block(cells):
    return dict(n=len(cells), keys=[f"p{c['_pf']}r{c['_rf']}" for c in sorted(cells, key=lambda c: (c["_pf"], c["_rf"]))],
                mean=mean([c["heldout_AUROC_perpatient"] for c in cells]),
                sd=sd([c["heldout_AUROC_perpatient"] for c in cells]),
                trainrxn=mean([c["trainrxn_AUROC_perpatient"] for c in cells]),
                exprbearing=mean([c["heldout_exprbearing_AUROC"] for c in cells]),
                rawexpr=mean([c["baseline_rawexpr"] for c in cells]),
                indicator=mean([c["baseline_indicator"] for c in cells]),
                rho=mean([c["rho"] for c in cells]), interpat_r=mean([c["interpatient_r_median"] for c in cells]),
                gap_abs=mean([abs(c["trainrxn_AUROC_perpatient"] - c["heldout_AUROC_perpatient"]) for c in cells]),
                gap_abs_sd=sd([abs(c["trainrxn_AUROC_perpatient"] - c["heldout_AUROC_perpatient"]) for c in cells]),
                gap_negative_cells=sum(1 for c in cells if c["trainrxn_AUROC_perpatient"] < c["heldout_AUROC_perpatient"]),
                n_active_labels=cells[0].get("n_active_labels"))
def _linear_rows(NAx, keys_subset=None):
    T = NAx.get("tuned", {}); out = {}
    for k, cells in T.items():
        sel = {c: v for c, v in cells.items() if keys_subset is None or c in keys_subset}
        if sel: out[k] = dict(n=len(sel), mean=mean(list(sel.values())), sd=sd(list(sel.values())), cells=dict(sorted(sel.items())))
    def d(x, y):
        if x in out and y in out:
            ks = sorted(set(out[x]["cells"]) & set(out[y]["cells"]))
            dd = [out[x]["cells"][c] - out[y]["cells"][c] for c in ks]
            # A row that uses no patient data carries one distinct value per reaction fold, so a
            # count over cells would count each reaction fold once per patient fold. Count the
            # reaction folds too, and let the text quote whichever unit the contrast allows.
            byr = {}
            for c, v in zip(ks, dd):
                byr.setdefault(c.split("r")[-1], []).append(v)
            rm = [mean(v) for _, v in sorted(byr.items())]
            return dict(n=len(ks), delta=mean(dd), sd=sd(dd),
                        n_positive=sum(1 for v in dd if v > 0), n_negative=sum(1 for v in dd if v < 0),
                        n_rfolds=len(rm), n_rfolds_positive=sum(1 for v in rm if v > 0),
                        n_rfolds_negative=sum(1 for v in rm if v < 0))
        return None
    out["_deltas"] = dict(lin_patient_expr_only=d("expr_perpat_lr", "expr_cohortmean_lr"),
                          lin_patient_struct_expr=d("structure_plus_perpat", "structure_plus_cohortmean"),
                          nonparam_patient_effect=d("expr_perpat_rank", "expr_cohortmean_rank"),
                          structure_over_expression_fitted=d("structure_only", "expr_cohortmean_lr"),
                          topology_over_expression_fitted=d("topology_only", "expr_cohortmean_lr"),
                          annotation_over_expression_fitted=d("annotation_only", "expr_cohortmean_lr"))
    out["_meta"] = dict(data=NAx.get("_data"), expr_transform=NAx.get("_expr_transform"), n_active_labels=NAx.get("_n_active_labels"),
                        columns=NAx.get("_columns"))
    return out

def _pooled_deltas(NPX, NAX, basis_keys, regime="tuned"):
    """The pooled, frozen rows of one pooled_baselines.py output against the cohort-mean and
    patient-adapted rows of the matching naive_allfolds file, paired on the basis cells."""
    if not NPX or not NAX: return None
    P_ = NPX.get(regime) or {}; Q_ = NAX.get(regime) or {}
    out = {}
    for nm, pk, qk in (("pooled_vs_cohort_expr", "expr_pooled_lr", "expr_cohortmean_lr"),
                       ("pooled_vs_cohort_struct", "structure_plus_pooled", "structure_plus_cohortmean"),
                       ("pooled_vs_perpat_expr", "expr_pooled_lr", "expr_perpat_lr"),
                       ("pooled_vs_perpat_struct", "structure_plus_pooled", "structure_plus_perpat")):
        if pk in P_ and qk in Q_:
            d = _paired_dict(P_[pk], Q_[qk], basis_keys)
            if d: out[nm] = d
    rows = {}
    for pk in ("expr_pooled_lr", "structure_plus_pooled"):
        if pk in P_:
            bv = [v for c, v in sorted(P_[pk].items()) if c in basis_keys]
            rows[pk] = dict(n=len(bv), mean=mean(bv), sd=sd(bv))
    return dict(rows=rows, deltas=out, max_train_patients=(NPX.get("_subsample") or {}).get("max_train_patients"),
                n_used=NPX.get("_n_train_patients_used"), expr_transform=NPX.get("_expr_transform"), data=NPX.get("_data")) if out else None

H = [c for c in load_cells("ht29") if c["_model"] == "gnn_B" and c["_lam"] == 0.0 and c["_ts"] is None]
NAH = opt("naive_allfolds_ht29.json")
if H or NAH:
    hr = _cellmap([c for c in H if c["_mode"] == "real"]); hm = _cellmap([c for c in H if c["_mode"] == "mean"]); hz = _cellmap([c for c in H if c["_mode"] == "zero"])
    # the paired real-versus-mean term needs those two arms; the zeroed arm may land later
    keys = sorted(set(hr) & set(hm)); keys3 = sorted(set(keys) & set(hz))
    blk = dict(basis=[f"p{p}r{r}" for p, r in keys], pfolds=sorted({p for p, _ in keys}),
               real=_arm_block([hr[k] for k in keys]) if keys else None,
               cohort_mean=_arm_block([hm[k] for k in keys]) if keys else None,
               zero=_arm_block([hz[k] for k in keys3]) if keys3 else None,
               patient_identity=_paired_cells(hr, hm) if keys else None,
               aggregate_expression=_paired_cells(hm, hz) if keys3 else None,
               expression_total=_paired_cells(hr, hz) if keys3 else None)
    # the same cells under the union label, for the side-by-side reading
    ur = _cellmap([c for c in C if c["_model"] == "gnn_B" and c["_lam"] == 0.0 and c["_mode"] == "real" and (c["_pf"], c["_rf"]) in set(keys)])
    um = _cellmap([c for c in C if c["_model"] == "gnn_B" and c["_lam"] == 0.0 and c["_mode"] == "mean" and (c["_pf"], c["_rf"]) in set(keys)])
    uz = _cellmap([c for c in C if c["_model"] == "gnn_B" and c["_lam"] == 0.0 and c["_mode"] == "zero" and (c["_pf"], c["_rf"]) in set(keys)])
    blk["union_same_cells"] = dict(real=_arm_block(list(ur.values())) if ur else None, cohort_mean=_arm_block(list(um.values())) if um else None,
                                   zero=_arm_block(list(uz.values())) if uz else None,
                                   patient_identity=_paired_cells(ur, um) if ur and um else None,
                                   aggregate_expression=_paired_cells(um, uz) if um and uz else None)
    if NAH:
        blk["linear_same_cells"] = _linear_rows(NAH, set(blk["basis"]) if blk["basis"] else {"p0r0", "p0r1", "p0r2"})
        blk["linear_all_cells"] = _linear_rows(NAH)
        LH = blk["linear_same_cells"]
        for nm, gk, lk in (("gnn_zero_over_structure", "zero", "structure_only"), ("gnn_mean_over_structure", "cohort_mean", "structure_only"),
                           ("gnn_real_over_structure", "real", "structure_only"),
                           ("gnn_mean_over_lr_with_expression", "cohort_mean", "structure_plus_cohortmean"),
                           ("gnn_real_over_lr_own_expression", "real", "structure_plus_perpat")):
            if blk.get(gk) and lk in LH: blk[nm] = blk[gk]["mean"] - LH[lk]["mean"]
        blk["pooled"] = _pooled_deltas(opt("naive_pooled_ht29.json"), NAH, set(S["basis"]))
    ROB["ht29"] = blk

SR = [c for c in load_cells("seedrep") if c["_model"] == "gnn_B" and c["_lam"] == 0.0 and c["_ts"] is not None]
if SR:
    seeds = sorted({c["_ts"] for c in SR}); ROB["seedrep"] = {}
    for ts in seeds:
        sr = _cellmap([c for c in SR if c["_ts"] == ts and c["_mode"] == "real"]); sm = _cellmap([c for c in SR if c["_ts"] == ts and c["_mode"] == "mean"])
        keys = sorted(set(sr) & set(sm))
        mr = _cellmap([c for c in C if c["_model"] == "gnn_B" and c["_lam"] == 0.0 and c["_mode"] == "real" and (c["_pf"], c["_rf"]) in set(keys)])
        mm = _cellmap([c for c in C if c["_model"] == "gnn_B" and c["_lam"] == 0.0 and c["_mode"] == "mean" and (c["_pf"], c["_rf"]) in set(keys)])
        ROB["seedrep"][str(ts)] = dict(
            basis=[f"p{p}r{r}" for p, r in keys], train_seed=ts, main_train_seed=2024,
            real=_arm_block([sr[k] for k in keys]) if keys else None, cohort_mean=_arm_block([sm[k] for k in keys]) if keys else None,
            patient_identity=_paired_cells(sr, sm),
            main_same_cells=dict(real=_arm_block([mr[k] for k in keys]) if keys and mr else None,
                                 cohort_mean=_arm_block([mm[k] for k in keys]) if keys and mm else None,
                                 patient_identity=_paired_cells(mr, mm)),
            real_vs_main=_paired_cells(sr, mr), cohort_mean_vs_main=_paired_cells(sm, mm))
        pi, pm = ROB["seedrep"][str(ts)]["patient_identity"], ROB["seedrep"][str(ts)]["main_same_cells"]["patient_identity"]
        if pi["n"] and pm["n"]:
            ROB["seedrep"][str(ts)]["patient_identity_seed_minus_main"] = dict(
                n=pi["n"], delta=pi["delta"] - pm["delta"],
                per_cell={k: pi["per_cell"][k] - pm["per_cell"][k] for k in pi["per_cell"] if k in pm["per_cell"]})

RG = [c for c in load_cells("rankgnn") if c["_model"] == "gnn_B" and c["_lam"] == 0.0 and c["_xf"] == "rank" and c["_ts"] is None]
ROB_RANKGNN = None
if RG:
    gr = _cellmap([c for c in RG if c["_mode"] == "real"]); gm = _cellmap([c for c in RG if c["_mode"] == "mean"])
    keys = sorted(set(gr) & set(gm))
    mr = _cellmap([c for c in C if c["_model"] == "gnn_B" and c["_lam"] == 0.0 and c["_mode"] == "real" and (c["_pf"], c["_rf"]) in set(keys)])
    mm = _cellmap([c for c in C if c["_model"] == "gnn_B" and c["_lam"] == 0.0 and c["_mode"] == "mean" and (c["_pf"], c["_rf"]) in set(keys)])
    ROB_RANKGNN = dict(basis=[f"p{p}r{r}" for p, r in keys],
                       real=_arm_block([gr[k] for k in keys]) if keys else None, cohort_mean=_arm_block([gm[k] for k in keys]) if keys else None,
                       patient_identity=_paired_cells(gr, gm),
                       main_same_cells=dict(real=_arm_block([mr[k] for k in keys]) if keys and mr else None,
                                            cohort_mean=_arm_block([mm[k] for k in keys]) if keys and mm else None,
                                            patient_identity=_paired_cells(mr, mm)),
                       real_vs_main=_paired_cells(gr, mr), cohort_mean_vs_main=_paired_cells(gm, mm))
NAR = opt("naive_allfolds_rank.json")
if NAR and NA:
    keys = set(S["basis"])
    rk = _linear_rows(NAR, keys); base = _linear_rows(NA, keys)
    comp = {}
    for k in ("expr_perpat_rank", "expr_cohortmean_rank", "expr_cohortmean_lr", "expr_perpat_lr",
              "structure_plus_cohortmean", "structure_plus_perpat", "structure_only"):
        if k in rk and k in base:
            ks = sorted(set(rk[k]["cells"]) & set(base[k]["cells"]))
            dd = [rk[k]["cells"][c] - base[k]["cells"][c] for c in ks]
            comp[k] = dict(rank=rk[k]["mean"], main=base[k]["mean"], delta=mean(dd), sd=sd(dd), n=len(ks))
    ROB["rank"] = dict(linear=rk, versus_main=comp, deltas_main=base["_deltas"], gnn=ROB_RANKGNN,
                       pooled=_pooled_deltas(opt("naive_pooled_rank.json"), NAR, keys))
# the pooled rows refitted on a larger training-patient subsample (pooled_baselines.py --max_train_patients)
NPL = opt("naive_pooled_120.json")
if NPL and NA and NPD:
    keys = set(S["basis"])
    large = _pooled_deltas(NPL, NA, keys)
    if large:
        P0 = NPD.get("tuned") or {}; P1 = NPL.get("tuned") or {}
        large["versus_default"] = {pk: _paired_dict(P1[pk], P0[pk], keys) for pk in ("expr_pooled_lr", "structure_plus_pooled") if pk in P0 and pk in P1}
        large["default_max_train_patients"] = (NPD.get("_subsample") or {}).get("max_train_patients")
        large["gnn_real_vs_pooled_struct"] = _paired_dict(GNN_CELLS["real"], P1["structure_plus_pooled"], keys) if "structure_plus_pooled" in P1 else None
        ROB["pooled_large"] = large
# early stopping on an inner held-out reaction partition (results/ivr/): the unsubstituted and
# cohort-mean arms refitted with the stopping criterion matched to the test question, paired with the
# main cells of the same name
IV = [c for c in load_cells("ivr") if c["_model"] == "gnn_B" and c["_lam"] == 0.0 and c["_ivr"] and c["_ts"] is None and c["_xf"] == "none" and c["_synth"] is None]
if IV:
    vr = _cellmap([c for c in IV if c["_mode"] == "real"]); vm = _cellmap([c for c in IV if c["_mode"] == "mean"])
    keys = sorted(set(vr) & set(vm))
    mr = _cellmap([c for c in C if c["_model"] == "gnn_B" and c["_lam"] == 0.0 and c["_mode"] == "real" and (c["_pf"], c["_rf"]) in set(keys)])
    mm = _cellmap([c for c in C if c["_model"] == "gnn_B" and c["_lam"] == 0.0 and c["_mode"] == "mean" and (c["_pf"], c["_rf"]) in set(keys)])
    ROB["ivr"] = dict(basis=[f"p{p}r{r}" for p, r in keys],
                      inner_val_fraction=(IV[0].get("inner_val_rxn")), n_inner_val_rxn=(IV[0].get("n_inner_val_rxn")),
                      n_fit_rxn=(IV[0].get("n_fit_rxn")),
                      real=_arm_block([vr[k] for k in keys]) if keys else None, cohort_mean=_arm_block([vm[k] for k in keys]) if keys else None,
                      patient_identity=_paired_cells(vr, vm),
                      main_same_cells=dict(real=_arm_block([mr[k] for k in keys]) if keys and mr else None,
                                           cohort_mean=_arm_block([mm[k] for k in keys]) if keys and mm else None,
                                           patient_identity=_paired_cells(mr, mm)),
                      real_vs_main=_paired_cells(vr, mr), cohort_mean_vs_main=_paired_cells(vm, mm))
    # the zeroed, presence-only and wrong-patient arms under the same matched rule, where they ran,
    # paired with the matched-rule unsubstituted and cohort-mean arms on the same cells and with the
    # grid's-rule arm of the same name
    vz = _cellmap([c for c in IV if c["_mode"] == "zero"]); vi = _cellmap([c for c in IV if c["_mode"] == "indicator"])
    vp = _cellmap([c for c in IV if c["_mode"] == "permute"])
    mz = _cellmap([c for c in C if c["_model"] == "gnn_B" and c["_lam"] == 0.0 and c["_mode"] == "zero"])
    mi = _cellmap([c for c in C if c["_model"] == "gnn_B" and c["_lam"] == 0.0 and c["_mode"] == "indicator"])
    for nm, arm, main_arm in (("zero", vz, mz), ("indicator", vi, mi), ("permute", vp, PM)):
        if not arm: continue
        ks = sorted(arm)
        ROB["ivr"][nm] = _arm_block([arm[k] for k in ks])
        ROB["ivr"][nm + "_vs_main"] = _paired_cells(arm, main_arm) if main_arm else None
    if vz:
        ROB["ivr"]["aggregate_expression"] = _paired_cells(vm, vz)      # cohort mean minus zeroed
        ROB["ivr"]["expression_total"] = _paired_cells(vr, vz)          # own minus zeroed
        ROB["ivr"]["main_same_cells"]["zero"] = _arm_block([mz[k] for k in sorted(set(vz) & set(mz))]) if set(vz) & set(mz) else None
        ROB["ivr"]["main_same_cells"]["aggregate_expression"] = _paired_cells(mm, mz)
    if vi:
        ROB["ivr"]["presence_over_zero"] = _paired_cells(vi, vz) if vz else None       # indicator minus zeroed
        ROB["ivr"]["magnitude_over_presence"] = _paired_cells(vm, vi)                  # cohort mean minus indicator
        ROB["ivr"]["own_over_presence"] = _paired_cells(vr, vi)                        # own minus indicator
    if vp:
        ROB["ivr"]["own_minus_permuted"] = _paired_cells(vr, vp)                       # own minus wrong-patient
        ROB["ivr"]["permuted_minus_mean"] = _paired_cells(vp, vm)                      # wrong-patient minus cohort mean
# the family-disjoint reaction split (results/fam/): the same five substitution arms refitted with
# biochemically related reactions withheld together, so that a held-out reaction has no near-twin in
# the training half. Every arm is paired with the random-split arm of the same name on the same cells,
# which prices the leakage the random split carries, and the own-minus-cohort-mean contrast is
# recomputed under the harder split, where it is the paper's primary value for that question.
FAM = [c for c in load_cells("fam") if c["_model"] == "gnn_B" and c["_lam"] == 0.0 and c["_fam"]
       and c["_ts"] is None and c["_xf"] == "none" and c["_synth"] is None]
if FAM:
    fr = _cellmap([c for c in FAM if c["_mode"] == "real"]); fm = _cellmap([c for c in FAM if c["_mode"] == "mean"])
    fz = _cellmap([c for c in FAM if c["_mode"] == "zero"]); fi = _cellmap([c for c in FAM if c["_mode"] == "indicator"])
    fp = _cellmap([c for c in FAM if c["_mode"] == "permute"])
    keys = sorted(set(fr) & set(fm))
    F = dict(basis=[f"p{p}r{r}" for p, r in keys],
             family_scheme=FAM[0].get("family_scheme"), family_map=FAM[0].get("family_map"),
             inner_val_fraction=FAM[0].get("inner_val_rxn"),
             n_heldout_rxn=mean([c["n_heldout_rxn"] for c in FAM]),
             n_heldout_rxn_min=min(c["n_heldout_rxn"] for c in FAM),
             n_heldout_rxn_max=max(c["n_heldout_rxn"] for c in FAM),
             n_heldout_families=mean([c["n_heldout_families"] for c in FAM if c.get("n_heldout_families") is not None]),
             n_fit_rxn=FAM[0].get("n_fit_rxn"), n_inner_val_rxn=FAM[0].get("n_inner_val_rxn"))
    for nm, arm in (("real", fr), ("cohort_mean", fm), ("zero", fz), ("indicator", fi), ("permute", fp)):
        if arm: F[nm] = _arm_block([arm[k] for k in sorted(arm)])
    # the contrasts, on the cells where both arms of the pair ran
    F["patient_identity"] = _paired_cells(fr, fm)                    # own minus cohort mean: primary
    if fp:
        F["own_minus_permuted"] = _paired_cells(fr, fp)              # own minus wrong patient
        F["permuted_minus_mean"] = _paired_cells(fp, fm)             # wrong patient minus cohort mean
    if fz:
        F["aggregate_expression"] = _paired_cells(fm, fz)            # cohort mean minus zeroed
        F["expression_total"] = _paired_cells(fr, fz)                # own minus zeroed
    if fi:
        F["presence_over_zero"] = _paired_cells(fi, fz) if fz else None
        F["magnitude_over_presence"] = _paired_cells(fm, fi)
        F["own_over_presence"] = _paired_cells(fr, fi)
    # the price of the random split: every arm against the matched-rule random-split arm of the same
    # name on the same cells, and the contrast-of-contrasts for the primary question
    if IV:
        _rand = dict(real=vr, cohort_mean=vm, zero=vz, indicator=vi, permute=vp)
        F["versus_random_split"] = {nm: _paired_cells(a, _rand[nm]) for nm, a in
                                    (("real", fr), ("cohort_mean", fm), ("zero", fz), ("indicator", fi), ("permute", fp))
                                    if a and _rand.get(nm)}
        _pi_rand = _paired_cells(vr, vm)
        _ks = sorted(set(F["patient_identity"]["keys"]) & set(_pi_rand["keys"]))
        if _ks:
            _dd = [F["patient_identity"]["per_cell"][k] - _pi_rand["per_cell"][k] for k in _ks]
            F["patient_identity_random_split"] = _pi_rand
            F["patient_identity_fam_minus_random"] = dict(n=len(_ks), keys=_ks, delta=mean(_dd), sd=sd(_dd),
                                                          per_cell=dict(zip(_ks, _dd)),
                                                          n_positive=sum(1 for x in _dd if x > 0),
                                                          n_negative=sum(1 for x in _dd if x < 0))
    # the family map's gene-rule relation reads the reference network's own rule string and the audit's
    # reads the project's gene table, so a few percent of held-out reactions still have a partner. The
    # saved predictions are scored again with those reactions dropped (fam_clean_scores.py), so the
    # paper can say whether the residue moves any contrast instead of assuming it does not.
    FC = opt("fam_clean_scores.json")
    if FC:
        _fc = {}
        for k, v in FC["cells"].items():
            if v["model"] != "gnn_B" or v["lam"] != 0.0 or "_ts20" in k: continue
            _fc.setdefault(v["variant"], {})[(v["pfold"], v["rfold"])] = v
        def _cmean(cs, key): return mean([c[key] for c in cs.values()]) if cs else None
        def _cpair(a, b, key):
            ks = sorted(set(a) & set(b))
            if len(ks) < 2: return None
            d = [a[k][key] - b[k][key] for k in ks]
            return dict(n=len(ks), delta=mean(d), sd=sd(d), n_positive=sum(1 for x in d if x > 0),
                        per_cell={f"p{q}r{r}": x for (q, r), x in zip(ks, d)})
        CL = dict(n_dropped=sorted({v["n_dropped"] for v in FC["cells"].values()}),
                  arms={}, contrasts={})
        for a, cs in _fc.items():
            CL["arms"][a] = dict(n=len(cs), full=_cmean(cs, "auroc_heldout"), clean=_cmean(cs, "auroc_heldout_clean"),
                                 n_heldout=_cmean(cs, "n_heldout"), n_heldout_clean=_cmean(cs, "n_heldout_clean"))
        for nm, a, b in (("patient_identity", "real", "mean"), ("own_minus_permuted", "real", "permute"),
                         ("aggregate_expression", "mean", "zero"), ("presence_over_zero", "indicator", "zero")):
            if a in _fc and b in _fc:
                full, cln = _cpair(_fc[a], _fc[b], "auroc_heldout"), _cpair(_fc[a], _fc[b], "auroc_heldout_clean")
                if full and cln:
                    CL["contrasts"][nm] = dict(full=full, clean=cln, shift=cln["delta"] - full["delta"])
        if CL["contrasts"]:
            CL["max_abs_shift"] = max(abs(v["shift"]) for v in CL["contrasts"].values())
            CL["signs_unchanged"] = all((v["full"]["delta"] > 0) == (v["clean"]["delta"] > 0) for v in CL["contrasts"].values())
        F["clean_subset"] = CL
    # the training-seed replicates of the family split, where they ran: the same cell at further seeds
    FS = [c for c in load_cells("fam") if c["_model"] == "gnn_B" and c["_lam"] == 0.0 and c["_fam"] and c["_ts"] is not None]
    if FS:
        _seeds = sorted({c["_ts"] for c in FS}); _blk = {}
        for ts in _seeds:
            _r = _cellmap([c for c in FS if c["_ts"] == ts and c["_mode"] == "real"])
            _m = _cellmap([c for c in FS if c["_ts"] == ts and c["_mode"] == "mean"])
            if _r and _m: _blk[str(ts)] = _paired_cells(_r, _m)
        if _blk:
            _base = {k: fr[k] for k in fr if k in fm}; _basem = {k: fm[k] for k in fm if k in fr}
            _cells = sorted(set().union(*[set(v["keys"]) for v in _blk.values()]))
            F["seed_replicates"] = dict(seeds=_seeds, cells=_cells, by_seed=_blk,
                                        default_seed=_paired_cells({k: v for k, v in fr.items() if f"p{k[0]}r{k[1]}" in _cells},
                                                                   {k: v for k, v in fm.items() if f"p{k[0]}r{k[1]}" in _cells}))
    # the linear reference rows refitted on the family-disjoint folds (run_family_linear.sh): the
    # no-patient floor, the expression rows and the nearest-family label lookup, each paired with the
    # same row on the random split, so the two splits are compared row by row on identical cells
    NAF, NPF_ = opt("naive_allfolds_fam.json"), opt("naive_pooled_fam.json")
    if NAF:
        _rand = (((R.get("naive_allfolds") or {}).get("regimes") or {}).get("tuned") or {})
        def _cellsof(blk):
            return blk.get("cells") if isinstance(blk, dict) and "cells" in blk else (blk if isinstance(blk, dict) else None)
        LIN = dict(rows={}, family_map=NAF.get("_family_map"))
        for k, blk in (NAF.get("tuned") or {}).items():
            fc = _cellsof(blk)
            if not fc: continue
            rec = dict(n=len(fc), mean=mean(list(fc.values())), sd=sd(list(fc.values())), cells=dict(sorted(fc.items())))
            rc = _cellsof(_rand.get(k) or {})
            if rc:
                ks = sorted(set(fc) & set(rc))
                if len(ks) > 1:
                    d = [fc[c] - rc[c] for c in ks]
                    rec["versus_random"] = dict(n=len(ks), delta=mean(d), sd=sd(d), random_mean=mean([rc[c] for c in ks]),
                                                per_cell=dict(zip(ks, d)), n_negative=sum(1 for x in d if x < 0))
            LIN["rows"][k] = rec
        if NPF_:
            for k, blk in (NPF_.get("tuned") or {}).items():
                fc = _cellsof(blk)
                if fc: LIN["rows"]["pooled_" + k] = dict(n=len(fc), mean=mean(list(fc.values())), sd=sd(list(fc.values())),
                                                        cells=dict(sorted(fc.items())))
        # the graph model against the no-patient linear floor, on each split, and the change in that gap
        _so = _cellsof((NAF.get("tuned") or {}).get("structure_only") or {})
        _sor = _cellsof(_rand.get("structure_only") or {})
        if _so:
            _fr_k = {f"p{p_}r{r_}": c["heldout_AUROC_perpatient"] for (p_, r_), c in fr.items()}
            ks = sorted(set(_fr_k) & set(_so))
            if len(ks) > 1:
                d = [_fr_k[c] - _so[c] for c in ks]
                LIN["gnn_real_minus_structure_only"] = dict(n=len(ks), delta=mean(d), sd=sd(d),
                                                            per_cell=dict(zip(ks, d)),
                                                            n_positive=sum(1 for x in d if x > 0))
            if _sor and IV:
                _vr_k = {f"p{p_}r{r_}": c["heldout_AUROC_perpatient"] for (p_, r_), c in vr.items()}
                ks2 = sorted(set(_vr_k) & set(_sor))
                if len(ks2) > 1:
                    d2 = [_vr_k[c] - _sor[c] for c in ks2]
                    LIN["gnn_real_minus_structure_only_random"] = dict(n=len(ks2), delta=mean(d2), sd=sd(d2),
                                                                       per_cell=dict(zip(ks2, d2)),
                                                                       n_positive=sum(1 for x in d2 if x > 0))
                    ks3 = sorted(set(ks) & set(ks2))
                    if len(ks3) > 1:
                        d3 = [(_fr_k[c] - _so[c]) - (_vr_k[c] - _sor[c]) for c in ks3]
                        LIN["gap_change"] = dict(n=len(ks3), delta=mean(d3), sd=sd(d3), per_cell=dict(zip(ks3, d3)),
                                                 n_negative=sum(1 for x in d3 if x < 0))
        F["linear"] = LIN
    # the family map's component sizes, the quantity the frozen analysis plan set a threshold on
    _fmp = os.path.join(HERE, "data", "families", "families_union.json")
    if os.path.exists(_fmp):
        # the seed the folds were drawn under, verified rather than asserted: rebuild the folds at the
        # module's default seed and check the held-out sizes against what the cells actually recorded
        try:
            import sys as _sys, inspect as _inspect, numpy as np
            _sys.path.insert(0, os.path.join(HERE, "code"))
            from family_folds import load_family_map as _lfm, family_rfolds as _frf
            _seed = _inspect.signature(_frf).parameters["seed"].default
            _z = np.load(os.path.join(os.path.dirname(HERE), "paper2", "rxn_context.npz"))
            _comp, _ = _lfm(_fmp)
            _sizes = sorted(len(te) for _, te in _frf(_comp, _z["labels"].astype(int), _z["has"].astype(bool),
                                                     n_folds=3, seed=_seed))
            _cell_sizes = sorted({c["n_heldout_rxn"] for c in FAM})
            F["fold_seed"] = int(_seed)
            F["fold_seed_verified"] = set(_cell_sizes).issubset(set(_sizes))
            F["fold_heldout_sizes"] = _sizes
        except Exception as _e:
            F["fold_seed"] = None; F["fold_seed_verified"] = False; F["fold_seed_error"] = str(_e)
        _fmj = json.load(open(_fmp)); _cid = _fmj["component_id"]
        _cnt = {}
        for c in _cid: _cnt[c] = _cnt.get(c, 0) + 1
        _sz = sorted(_cnt.values(), reverse=True)
        F["components"] = dict(scheme=_fmj.get("scheme"), n_reactions=_fmj["n_reactions"],
                               n_components=len(_sz), largest=_sz[0],
                               largest_share=_sz[0] / _fmj["n_reactions"],
                               top=_sz[:5], singletons=sum(1 for v in _sz if v == 1),
                               threshold=0.05, threshold_reached=(_sz[0] / _fmj["n_reactions"]) > 0.05)
    ROB["fam"] = F
# the three reaction masks scored separately (results/mask_scores.json, mask_scores_p1.py): every
# cell's saved predictions rescored on the loss-fit reactions, the inner-validation reactions and the
# outer held-out reactions, so that the memorization diagnostic under the matched rule compares
# reactions the model fitted with reactions it never saw, and never mixes the two; the grid's-rule
# arms are scored on the identical masks (their inner-validation reactions were fitted). AUPRC on
# the outer held-out reactions comes with it for every arm.
MS = opt("mask_scores.json")
if MS:
    def _ms_cells(dirname, variant, lam=0.0, model="gnn_B"):
        return {(c["pfold"], c["rfold"]): c for c in MS["cells"].values()
                if c["dir"] == dirname and c["variant"] == variant and c["model"] == model and c["lam"] == lam}
    def _ms_block(cells, keys):
        cs = [cells[k] for k in keys]
        def m(f): return mean([c[f] for c in cs])
        def s(f): return sd([c[f] for c in cs])
        return dict(n=len(cs), keys=[f"p{p}r{r}" for p, r in keys],
                    fit=m("auroc_fit"), inner_val=m("auroc_inner_val"), outer_test=m("auroc_outer_test"), train_all=m("auroc_train_all"),
                    outer_exprbearing=m("auroc_outer_exprbearing"),
                    gap_fit_minus_outer=mean([c["auroc_fit"] - c["auroc_outer_test"] for c in cs]),
                    gap_fit_minus_outer_sd=sd([c["auroc_fit"] - c["auroc_outer_test"] for c in cs]),
                    gap_inner_minus_outer=mean([c["auroc_inner_val"] - c["auroc_outer_test"] for c in cs]),
                    gap_inner_minus_outer_sd=sd([c["auroc_inner_val"] - c["auroc_outer_test"] for c in cs]),
                    gap_fit_minus_inner=mean([c["auroc_fit"] - c["auroc_inner_val"] for c in cs]),
                    inner_below_outer_cells=sum(1 for c in cs if c["auroc_inner_val"] < c["auroc_outer_test"]),
                    auprc_outer=m("auprc_outer_test"), auprc_outer_sd=s("auprc_outer_test"), auprc_outer_exprbearing=m("auprc_outer_exprbearing"),
                    auprc_fit=m("auprc_fit"), auprc_inner_val=m("auprc_inner_val"),
                    prevalence_outer=m("prevalence_outer_test"), prevalence_outer_exprbearing=m("prevalence_outer_exprbearing"),
                    n_fit=cs[0]["n_fit"], n_inner_val=cs[0]["n_inner_val"], n_outer_test=cs[0]["n_outer_test"])
    def _ms_paired(a, b, field):
        ks = sorted(set(a) & set(b)); d = [a[k][field] - b[k][field] for k in ks]
        return dict(n=len(ks), delta=mean(d), sd=sd(d), n_positive=sum(1 for x in d if x > 0), n_negative=sum(1 for x in d if x < 0),
                    per_cell={f"p{p}r{r}": x for (p, r), x in zip(ks, d)})
    pairs = {"real": (("ivr", "ivr"), ("dh", "real")), "cohort_mean": (("ivr", "mean_ivr"), ("ctrl", "mean")),
             "zero": (("ivr", "zero_ivr"), ("ctrl", "zero")), "indicator": (("ivr", "indicator_ivr"), ("ctrl", "indicator")),
             "permute": (("ivr", "permute_ivr"), ("permute", "permute"))}
    MB = dict(note=MS.get("note"), inner_val_fraction=MS.get("inner_val_rxn"), arms={})
    for nm, ((d1, v1), (d2, v2)) in pairs.items():
        a, b = _ms_cells(d1, v1), _ms_cells(d2, v2)
        if not a: continue
        ka = sorted(a); kb = sorted(set(a) & set(b))
        MB["arms"][nm] = dict(matched=_ms_block(a, ka), grid=(_ms_block(b, kb) if kb else None),
                              grid_same_cells_n=len(kb),
                              gap_change=(dict(fit_minus_outer=_ms_paired({k: dict(g=a[k]["auroc_fit"] - a[k]["auroc_outer_test"]) for k in kb},
                                                                            {k: dict(g=b[k]["auroc_fit"] - b[k]["auroc_outer_test"]) for k in kb}, "g"),
                                               fit=_ms_paired(a, b, "auroc_fit"), outer=_ms_paired(a, b, "auroc_outer_test"),
                                               inner_val=_ms_paired(a, b, "auroc_inner_val")) if kb else None))
    # the matched-rule ladder in AUPRC and the paired steps, on the cells every arm shares
    arms_m = {nm: _ms_cells(d1, v1) for nm, ((d1, v1), _) in pairs.items()}
    common = sorted(set.intersection(*[set(v) for v in arms_m.values() if v])) if any(arms_m.values()) else []
    if common:
        MB["auprc_ladder"] = dict(basis=[f"p{p}r{r}" for p, r in common],
                                  rungs={nm: mean([arms_m[nm][k]["auprc_outer_test"] for k in common]) for nm in arms_m if arms_m[nm]},
                                  rungs_exprbearing={nm: mean([arms_m[nm][k]["auprc_outer_exprbearing"] for k in common]) for nm in arms_m if arms_m[nm]},
                                  prevalence=mean([arms_m["real"][k]["prevalence_outer_test"] for k in common]),
                                  steps={f"{x}_minus_{y}": _ms_paired(arms_m[x], arms_m[y], "auprc_outer_test")
                                         for x, y in (("cohort_mean", "zero"), ("indicator", "zero"), ("cohort_mean", "indicator"),
                                                      ("real", "cohort_mean"), ("real", "permute"), ("permute", "cohort_mean"))
                                         if arms_m.get(x) and arms_m.get(y)})
    ROB["masks"] = MB
# the synthetic patient-varying positive control (results/synth/): the same driver on a label whose
# varying part is, by construction, a function of each patient's own column, under both selection
# rules and two fractions of varying reactions; the audit's contrast should turn positive here
SY = [c for c in load_cells("synth") if c["_model"] == "gnn_B" and c["_lam"] == 0.0 and c["_synth"] is not None and c["_ts"] is None and c["_xf"] == "none"]
if SY:
    def _mean_key(cells, key):
        v = [c[key] for c in cells if c.get(key) is not None]
        return mean(v) if v else None
    blocks = {}
    for kind, frac, ivr in sorted({(c["_synth"][0], c["_synth"][1], c["_ivr"]) for c in SY}):
        grp = [c for c in SY if c["_synth"] == (kind, frac) and c["_ivr"] == ivr]
        sr = _cellmap([c for c in grp if c["_mode"] == "real"]); sm = _cellmap([c for c in grp if c["_mode"] == "mean"])
        keys = sorted(set(sr) & set(sm))
        if not keys: continue
        rr = [sr[k] for k in keys]; mm = [sm[k] for k in keys]
        blocks[f"{kind}_{frac:g}_{'ivr' if ivr else 'main'}"] = dict(
            kind=kind, frac=frac, selection=("inner-validation reactions" if ivr else "main grid"), basis=[f"p{p}r{r}" for p, r in keys],
            n_vary_rxn=rr[0].get("n_vary_rxn"), n_heldout_vary_rxn=_mean_key(rr, "n_heldout_vary_rxn"),
            vary_positive_rate=_mean_key(rr, "heldout_vary_positive_rate"),
            real=dict(all=_mean_key(rr, "heldout_AUROC_perpatient"), vary=_mean_key(rr, "heldout_vary_AUROC"), shared=_mean_key(rr, "heldout_shared_AUROC"), trainrxn=_mean_key(rr, "trainrxn_AUROC_perpatient")),
            cohort_mean=dict(all=_mean_key(mm, "heldout_AUROC_perpatient"), vary=_mean_key(mm, "heldout_vary_AUROC"), shared=_mean_key(mm, "heldout_shared_AUROC"), trainrxn=_mean_key(mm, "trainrxn_AUROC_perpatient")),
            raw_own_column=dict(all=_mean_key(rr, "baseline_rawexpr"), vary=_mean_key(rr, "baseline_rawexpr_vary")),
            cohort_mean_column=dict(all=_mean_key(rr, "baseline_cohortmean"), vary=_mean_key(rr, "baseline_cohortmean_vary")),
            patient_identity=_paired_cells(sr, sm), patient_identity_vary=_paired_cells(sr, sm, key="heldout_vary_AUROC"),
            patient_identity_shared=_paired_cells(sr, sm, key="heldout_shared_AUROC") if all(c.get("heldout_shared_AUROC") is not None for c in rr + mm) else None)
    if blocks: ROB["synth"] = blocks
# the external audit of an independently published predictor (results/deepmeta/): DeepMeta on the
# DepMap 24Q4 Chronos gene effect. Only the summary blocks are carried into results_p1.json; the
# per-gene dictionaries stay in the audit file.
def _optsub(sub, name):
    p_ = os.path.join(RES, sub, name)
    return json.load(open(p_)) if os.path.exists(p_) else None
DMA, DMN, DMM = _optsub("deepmeta", "audit_results.json"), _optsub("deepmeta", "native_repro.json"), _optsub("deepmeta", "manifest.json")
if DMA and DMN and DMM:
    _pri = DMA["analyses"]["primary_template_domain"]
    R["external_audit"] = dict(
        model="DeepMeta", endpoint="DepMap 24Q4 Chronos gene effect",
        native=dict(cells=DMN["runnable_lines"], authors_cells=DMN["authors_test_lines"],
                    excluded=DMN["excluded"], pairs=DMN["ours_vs_authors_labels"]["n_pairs"],
                    nodes=DMN["ours_vs_authors_labels"]["n_nodes"],
                    auroc=DMN["ours_vs_authors_labels"]["auroc"], f1=DMN["ours_vs_authors_labels"]["f1_at_0.5"],
                    authors_auroc=DMN["authors_vs_authors_labels"]["auroc"], authors_f1=DMN["authors_vs_authors_labels"]["f1_at_0.5"],
                    pearson=DMN["ours_vs_authors_predictions"]["pearson_r"],
                    agreement=DMN["ours_vs_authors_predictions"]["binary_agreement_at_0.5"],
                    mean_abs_diff=DMN["ours_vs_authors_predictions"]["mean_abs_difference"]),
        counts=dict(heldout=DMM["counts"]["heldout_total"], heldout_test=DMM["counts"]["heldout_by_status"]["test"],
                    heldout_unseen=DMM["counts"]["heldout_by_status"]["unseen"], development=DMM["counts"]["development_total"],
                    patients=DMA["counts"]["patients_among_scored_lines"], lineages=DMA["counts"]["lineages_among_scored_lines"],
                    panel=DMA["counts"]["panel_genes"], complete_case=DMM["counts"]["complete_case_expr_ge_model"],
                    eligible=DMM["counts"]["after_supported_lineage_filter_eligible"],
                    # the exclusion applies to every candidate held-out line, the authors' own test
                    # roster included, so the count the text reports is the candidate-level one
                    patient_excluded=DMM["counts"]["candidate_excluded_patient_shared_with_train"],
                    patient_excluded_unseen=DMM["counts"]["unseen_excluded_patient_shared_with_train"],
                    candidate_lines=DMM["counts"]["candidate_test_lines"],
                    template_domain_fraction=DMA["counts"]["template_domain_fraction"],
                    dependency_threshold=DMA["counts"]["dependency_threshold"],
                    dependency_rate=DMA["counts"]["dependency_positive_fraction"]),
        concordance=_pri["concordance"], concordance_ci_lines=_pri["bootstrap_lines"]["concordance_ci"],
        concordance_ci_families=_pri["bootstrap_families"]["concordance_ci"],
        donor_effect=_pri["donor_effect"], donor_effect_vs_baselines=_pri["donor_effect_vs_baselines"],
        verdicts=_pri["verdicts_vs_sesi_0.01"],
        per_line_auroc={k: v["mean"] for k, v in _pri["per_line_auroc_dependency_lt_-0.5"].items()},
        spearman={k: dict(mean=v["mean"], frac_positive=v["frac_positive"], n=v["n_genes_evaluated"])
                  for k, v in _pri["spearman_vs_negative_gene_effect"].items()},
        constant_genes=_pri["genes_with_constant_scores"],
        secondary=dict(explicit_only=DMA["analyses"]["secondary_explicit_only"]["concordance"],
                       zero_fill=DMA["analyses"]["tertiary_zero_fill_all"]["concordance"],
                       # the smallest-effect verdicts of the two secondary domains, kept so the text can
                       # say where they agree with the primary domain and where they do not
                       explicit_only_verdicts=DMA["analyses"]["secondary_explicit_only"]["verdicts_vs_sesi_0.01"],
                       zero_fill_verdicts=DMA["analyses"]["tertiary_zero_fill_all"]["verdicts_vs_sesi_0.01"],
                       explicit_only_donor=DMA["analyses"]["secondary_explicit_only"]["donor_effect"],
                       zero_fill_donor=DMA["analyses"]["tertiary_zero_fill_all"]["donor_effect"],
                       explicit_only_per_line={k: v["mean"] for k, v in DMA["analyses"]["secondary_explicit_only"]["per_line_auroc_dependency_lt_-0.5"].items()},
                       zero_fill_per_line={k: v["mean"] for k, v in DMA["analyses"]["tertiary_zero_fill_all"]["per_line_auroc_dependency_lt_-0.5"].items()}),
        baselines_info=dict(development_lines=DMA["baselines_info"]["development_lines"],
                            ridge_alpha=DMA["baselines_info"]["ridge_alpha"],
                            ridge_n_features=DMA["baselines_info"]["ridge_n_features"],
                            rf_trees=DMA["baselines_info"]["rf_trees"],
                            rf_features=DMA["baselines_info"]["rf_features"],
                            lineage_mean_fallback_cells=len(DMA["baselines_info"]["lineage_mean_fallback_cells"])),
        concordance_range=dict(lo=min(_pri["concordance"].values()), hi=max(_pri["concordance"].values()),
                               n_predictors=len(_pri["concordance"])))
    # the donor schedules are a design fact of the substitution arms, not a result: which recipients
    # could not be deranged inside their own lineage, and which received a development-wide mean
    # because their lineage held too few development lines. They are reported in the text, so they
    # are read from the released schedule file rather than described from memory.
    DMS = _optsub("deepmeta", "schedules.json")
    if DMS:
        _pri_seed = str(DMS["primary_seed"])
        _sch = DMS["schedules"][_pri_seed]
        R["external_audit"]["schedule"] = dict(
            primary_seed=DMS["primary_seed"], seeds=sorted(int(k) for k in DMS["schedules"]),
            n_heldout=DMS["n_heldout"],
            mean_fallback_cells=len(_sch["mean_fallback_cells"]),
            within_cross_fallback_cells=len(_sch["within_flags"]),
            within_same_lineage_pairs=_sch["within_same_lineage_pairs"],
            donor_distinct_constraint=bool(_sch.get("donor_distinct_constraint", False)),
            heldout_lines_sharing_a_donor=len(_sch.get("heldout_lines_sharing_a_donor", [])),
            by_seed={k: dict(mean_fallback_cells=len(v["mean_fallback_cells"]),
                             within_cross_fallback_cells=len(v["within_flags"]),
                             within_same_lineage_pairs=v["within_same_lineage_pairs"])
                     for k, v in DMS["schedules"].items()})
    # the donor-schedule sensitivity: the same donor arms recomputed under the other two schedules.
    # A seed listed in a configuration is not evidence that its analysis ran, so only the seeds whose
    # result file is actually present are carried, and the count is reported with the range.
    _sens = {}
    for _sd in ("22", "33"):
        _f = _optsub("deepmeta", f"audit_results_seed{_sd}.json")
        if _f:
            _sens[_sd] = {k: v["donor_effect"]
                          for k, v in _f["analyses"]["primary_template_domain"]["verdicts_vs_sesi_0.01"].items()}
    if _sens:
        _sens[str(DMA["seed"])] = {k: v["donor_effect"] for k, v in _pri["verdicts_vs_sesi_0.01"].items()}
        _arms = sorted(set().union(*[set(v) for v in _sens.values()]))
        R["external_audit"]["schedule_sensitivity"] = dict(
            seeds=sorted(int(k) for k in _sens), by_seed=_sens,
            spread={a: dict(lo=min(_sens[k][a] for k in _sens if a in _sens[k]),
                            hi=max(_sens[k][a] for k in _sens if a in _sens[k]))
                    for a in _arms})

# the diagnostic calibration simulation (diagnostic_sim.py), folded in as its own block
DSIM = opt("diagnostic_sim.json")
if DSIM:
    R["diagnostic_sim"] = dict(params={k: DSIM["params"][k] for k in ("n_dev", "n_test", "P", "R", "pfolds", "rfolds",
                                                                          "inner_frac", "rounds", "dev_seeds", "test_seeds")
                                       if k in DSIM["params"]},
                               rule=DSIM["rule"], arms=DSIM.get("arms"),
                               test_confusion=DSIM["test_confusion"], dev_confusion=DSIM["dev_confusion"],
                               test_null_subsets=DSIM.get("test_null_subsets"), dev_null_subsets=DSIM.get("dev_null_subsets"),
                               counterexample=DSIM["counterexample"], decision_table=DSIM["decision_table"],
                               dev_decision_table=DSIM.get("dev_decision_table"),
                               # how many worlds of each configuration each set holds, so the text can say so
                               world_counts={tag: {"shared": sum(1 for w in DSIM[tag + "_worlds"] if w["label"] == "shared"),
                                                   "varying_alpha0": sum(1 for w in DSIM[tag + "_worlds"] if w["label"] == "varying" and w["alpha"] == 0),
                                                   "varying_weak": sum(1 for w in DSIM[tag + "_worlds"] if w["label"] == "varying" and 0 < w["alpha"] < 1.5),
                                                   "varying_strong": sum(1 for w in DSIM[tag + "_worlds"] if w["label"] == "varying" and w["alpha"] >= 1.5)}
                                             for tag in ("dev", "test") if tag + "_worlds" in DSIM},
                               # the learned model's chosen rounds, pooled over test worlds, per arm
                               rounds_chosen={arm: sorted({r for w in DSIM.get("test_worlds", []) for r in w["audit"]["learned"].get("rounds_chosen", {}).get(arm, [])})
                                              for arm in ("own", "mean", "donor")})

if ROB:
    R["robustness"] = ROB

R["parameters"] = PAR or {}
R["status"] = dict(
    grid_cells_done=len(grid_real),
    substitution_basis=S["basis"],
    rewired_cells=(S.get("rewired", {}) or {}).get("n", 0),
    grid_cells_total=75,
    substitution_cells=dict(real=S["real"]["n"], cohort_mean=S["cohort_mean"]["n"],
                            zero=S["zero"]["n"]),
    paired_cells=dict(patient_identity=S["patient_identity"]["n"],
                      aggregate_expression=S["aggregate_expression"]["n"]))

json.dump(R, open(OUT, "w"), indent=1)
print("wrote", os.path.relpath(OUT, HERE))
print("  grid cells      :", {k: v["cells"] for k, v in R["grid"].items()})
print("  substitution    :", R["status"]["substitution_cells"])
print("  paired patient  : n=%d  delta=%+.4f" % (S["patient_identity"]["n"], S["patient_identity"]["delta"]))
print("  paired aggexpr  : n=%d  delta=%+.4f" % (S["aggregate_expression"]["n"], S["aggregate_expression"]["delta"]))
print("  paired expr tot : n=%d  delta=%+.4f" % (S["expression_total"]["n"], S["expression_total"]["delta"]))
print("  ladder basis    : %d cells over patient folds %s" % (len(BASIS), S["patient_folds"]))
if S.get("rewired"):
    W = S["rewired"]
    print("  rewiring null   : n=%d  %.4f (real %.4f)  paired %+.4f  train-rxn %.4f"
          % (W["n"], W["mean"], W["vs_real"]["a_mean"], -W["vs_real"]["delta"], W["trainrxn"]))
if R.get("naive_pooled"):
    NPB = R["naive_pooled"]
    print("  pooled rows     : %d of %d basis cells%s" % (NPB["n_basis"], len(BASIS),
          "" if NPB["basis_complete"] else "  (INCOMPLETE: no pooled macro is emitted until every basis cell has run)"))
if R.get("auprc"):
    print("  AUPRC rows      : %s" % ", ".join(sorted(R["auprc"]["rows"])) if R["auprc"]["rows"] else "  AUPRC rows      : none")
if R.get("split_audit"):
    a = R["split_audit"]["all"]
    print("  split audit     : any-criterion duplicates %d of %d held-out reactions (%.1f%%)"
          % (a["any"]["n_matched"], a["n_heldout"], 100 * a["any"]["share_matched"]))
