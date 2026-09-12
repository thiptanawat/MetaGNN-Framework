#!/usr/bin/env python3
"""Emit manuscript/numbers.tex from data/results_p1.json: one macro per quoted result.

data/results_p1.json is itself derived by code/build_results_p1.py from the raw
per-cell outputs, so the chain from a training run to a number in the text has no
manual step in it. A quantity with no run behind it has no macro, and a sentence
that uses a macro with no definition fails the build.
"""
import json, os, math, statistics as st

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = json.load(open(os.path.join(HERE, "data", "results_p1.json")))
MS = "ms" if os.path.isdir(os.path.join(HERE, "ms")) else "manuscript"
L = []

def pfmt(p):
    """A p-value that never understates: four decimals rounded up, or two significant
    digits when the value is below 0.001 so a floor such as 2/2^12 prints as 0.00049."""
    if p is None: return "n/a"
    if p >= 0.001: return "{:.4f}".format(math.ceil(p * 1e4) / 1e4)
    if p < 1e-5: return "0.00001"          # the smallest value printed; a ceiling, never an understatement
    # two significant digits, rounded UP, so the printed value is never below the true one
    e = math.floor(math.log10(p)); scale = 10.0 ** (e - 1)
    v = math.ceil(p / scale - 1e-9) * scale
    return "{:.{d}f}".format(v, d=max(0, -e + 1))

def cmd(n, v, f="{:.4f}"):
    if isinstance(v, str): s = v
    elif v is None or (isinstance(v, float) and math.isnan(v)): s = "n/a"
    elif f == "int": s = f"{v:,}".replace(",", "{,}")
    elif f == "pct": s = f"{100*v:.1f}"
    elif f == "pct0": s = f"{100*v:.0f}"
    else: s = f.format(v)
    if s in ("+0.0000", "-0.0000"): s = "0.0000"     # a value that rounds to zero carries no sign
    L.append(f"\\newcommand{{\\{n}}}{{{s}}}")
    # every count also exists as a word (\\fooNWord), for prose in which the surrounding counts are spelled out
    if f == "int" and isinstance(v, int) and 0 <= v <= 12:
        L.append(f"\\newcommand{{\\{n}Word}}{{{_NUMWORDS[v]}}}")
_NUMWORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"]

def _word(v):
    """Spelled-out form for a small count used inside a generated phrase (cmd() already emits
    \\fooWord beside every int macro; this is for counts that appear only inside a sentence)."""
    return _NUMWORDS[v] if isinstance(v, int) and 0 <= v < len(_NUMWORDS) else f"{v:,}".replace(",", "{,}")

# ---- cohort ----------------------------------------------------------------
C = R["cohort"]
for k, n, f in [("n_patients","nPat","int"), ("n_reactions","nRxn","int"),
                ("n_metabolites","nMet","int"), ("n_active","nActive","int"),
                ("base_rate","baseRate","{:.4f}"),
                ("n_expression_bearing_probe20","nExpr","int"),
                ("n_gpr_rule","nGpr","int"), ("n_no_gpr_rule","nNoGpr","int"),
                ("n_gene_and_expression","nGeneExpr","int"), ("n_gene_no_expression","nGeneNoExpr","int"),
                ("n_expression_no_gene","nExprNoGene","int"),
                ("n_reactions_aligned_to_json","nAligned","int"), ("n_reactions_unaligned","nUnaligned","int"),
                ("n_cohortmean_nonzero","nMeanNonzero","int"),
                ("prevalence_expr","prevExpr","pct"),
                ("prevalence_no_expr","prevNoExpr","pct"),
                ("indicator_tpr","indTPR","{:.4f}"), ("indicator_fpr","indFPR","{:.4f}"),
                ("indicator_closed_form","indClosed","{:.4f}")]:
    cmd(n, C[k], f)

P = R["protocol"]
for k, n in [("pfolds","pFolds"), ("rfolds","rFolds"), ("train_patients","nTrainPat"),
             ("val_patients","nValPat"), ("test_patients","nTestPat"),
             ("train_patients_last","nTrainPatLast"), ("test_patients_last","nTestPatLast"),
             ("heldout_reactions","nHeldRxn")]:
    if k in P: cmd(n, P[k], "int")

# ---- parameter counts, from the construction path the experiments use -------
PAR = R.get("parameters", {})
for k, n in [("gnn_B","parGnn"), ("gnn_A","parGnnA"), ("mlp","parMlp"),
             ("mlp_emb","parEmb"), ("mlp_emb_embedding_only","parEmbTable"),
             ("mlp_emb_params_per_reaction","parEmbPerRxn")]:
    if k in PAR: cmd(n, PAR[k], "int")
if "gnn_B" in PAR: cmd("parGnnM", PAR["gnn_B"] / 1e6, "{:.2f}")

# ---- main grid --------------------------------------------------------------
NAMES = {"mlp": "Mlp", "mlp_emb": "Emb", "gnn_B": "Gnn"}
for m, r in R["grid"].items():
    t = NAMES[m]
    cmd(f"{t}Cells", r["cells"], "int")
    cmd(f"{t}Pfolds", len(r["pfolds_covered"]), "int")
    cmd(f"{t}Train", r["train_rxn"]); cmd(f"{t}Held", r["heldout_rxn"])
    if r["heldout_rxn_sd"] is not None: cmd(f"{t}HeldSD", r["heldout_rxn_sd"])
    cmd(f"{t}Gap", r["gap"]); cmd(f"{t}GapAbs", r["gap_abs"])
    if r["gap_sd"] is not None: cmd(f"{t}GapSD", r["gap_sd"])
    if r.get("gap_abs_sd") is not None: cmd(f"{t}GapAbsSD", r["gap_abs_sd"])
    cmd(f"{t}GapNeg", r["gap_negative_cells"], "int")
    cmd(f"{t}Rho", r["rho"], "{:.2f}"); cmd(f"{t}InterR", r["interpat_r"])
    cmd(f"{t}RawExpr", r["rawexpr"])
cmd("rawExpr", R["grid"]["mlp"]["rawexpr"])
cmd("indicator", R["grid"]["mlp"]["indicator"])
# the memorizer's held-out score against chance
G = R["grid"]["mlp_emb"]
if G["heldout_rxn_sd"]:
    se = G["heldout_rxn_sd"] / math.sqrt(G["cells"])
    cmd("EmbHeldT", (G["heldout_rxn"] - 0.5) / se, "{:.2f}")
    cmd("EmbHeldLo", G["heldout_rxn"] - 2.145 * se)
    cmd("EmbHeldHi", G["heldout_rxn"] + 2.145 * se)
pt = G.get("pfold_t_vs_chance")
if pt:
    cmd("EmbHeldPfoldN", pt["n"], "int"); cmd("EmbHeldPfoldT", pt["t"], "{:.1f}"); cmd("EmbHeldPfoldP", pfmt(pt["p"]))
    cmd("EmbHeldPfoldMin", pt["min"]); cmd("EmbHeldPfoldMax", pt["max"])
# gap separation, on absolute values so signed noise does not cancel
cmd("gapRatio", round(R["grid"]["mlp_emb"]["gap"] / R["grid"]["mlp"]["gap_abs"], -1), "{:.0f}")   # two significant figures, so the reader's arithmetic on the printed gaps agrees

# ---- input substitution -----------------------------------------------------
S = R["substitution"]
cmd("subCells", len(S["basis"]), "int")
cmd("subPfolds", len(S["patient_folds"]), "int")
cmd("subPfoldList", ", ".join(str(p) for p in S["patient_folds"]))
for k, n in [("real","cellsRealN"), ("cohort_mean","cellsMeanN"), ("zero","cellsZeroN")]:
    cmd(n, S[k]["n"], "int")
cmd("ladReal", S["real"]["mean"]); cmd("ladMean", S["cohort_mean"]["mean"])
cmd("ladZero", S["zero"]["mean"])
cmd("GnnHeldAll", S["real_all_pfolds"]["mean"])
cmd("GnnHeldAllN", S["real_all_pfolds"]["n"], "int")
for k, n in [("patient_identity","deltaPatient"), ("aggregate_expression","deltaAggExpr"),
             ("expression_total","deltaExpr")]:
    d = S[k]
    cmd(n, d["delta"], "{:+.4f}"); cmd(f"{n}N", d["n"], "int")
    if d["sd"] is not None: cmd(f"{n}SD", d["sd"], "{:.4f}")
for k, n in [("real","rhoReal"), ("zero","rhoZero"), ("cohort_mean","rhoMean")]:
    cmd(n, S[k]["rho"], "{:.2f}")
cmd("rhoNull", S["rho_null"], "{:.2f}")
RT = S.get("rho_null_terms")
if RT:
    cmd("rhoNullNpat", RT["n_test_patients"], "int"); cmd("rhoNullT", RT["dropout_passes"], "int")
    cmd("rhoNullEmp", RT["empirical_check"], "{:.2f}")
for k, n in [("real","irReal"), ("zero","irZero"), ("cohort_mean","irMean")]:
    cmd(n, S[k]["interpat_r"])

# ---- the wrong-patient arm ---------------------------------------------------
# Emitted only when the cells exist. Every macro here is either signed or a word derived from the
# sign, so a sentence built from them cannot assert a direction the run did not produce.
PMB = S.get("permute")
if PMB:
    cmd("robPermN", PMB["n"], "int"); cmd("robPermMean", PMB["mean"])
    if PMB["sd"] is not None: cmd("robPermSD", PMB["sd"])
    cmd("robPermCells", ", ".join(PMB["keys"]))
    cmd("robPermPfolds", len(PMB["pfolds"]), "int")
    cmd("robPermTrain", PMB["trainrxn"]); cmd("robPermExpr", PMB["exprbearing"])
    cmd("robPermRho", PMB["rho"], "{:.2f}"); cmd("robPermInterR", PMB["interpat_r"])
    if S.get("real"):
        cmd("robPermRhoDiffAbs", abs(PMB["rho"] - S["real"]["rho"]), "{:.2f}"); cmd("robPermInterRDiffAbs", abs(PMB["interpat_r"] - S["real"]["interpat_r"]), "{:.4f}")
    pi = PMB["patient_identity"]
    cmd("robPermPat", pi["delta"], "{:+.4f}"); cmd("robPermPatAbs", abs(pi["delta"]))
    cmd("robPermPatN", pi["n"], "int")
    if pi["sd"] is not None: cmd("robPermPatSD", pi["sd"])
    cmd("robPermPos", pi["n_positive"], "int"); cmd("robPermNeg", pi["n_negative"], "int")
    cmd("robPermPatWord", "benefit" if pi["delta"] > 0 else "cost")
    cmd("robPermPatCells", ", ".join(f"{v:+.4f}" for v in pi["per_cell"].values()))
    cmd("robPermRealSame", pi["a_mean"])
    cs = PMB.get("cohort_mean_same_cells")
    if cs:
        cmd("robPermMeanArm", cs["mean"]); cmd("robPermMeanPat", cs["delta"], "{:+.4f}")
        cmd("robPermMeanPatAbs", abs(cs["delta"]))
        cmd("robPermMeanPatWord", "benefit" if cs["delta"] > 0 else "cost")
        # the permuted column against the cohort mean on the same cells
        sm = pi["b_mean"] - cs["mean"]
        cmd("robPermStrMean", sm, "{:+.4f}"); cmd("robPermStrMeanAbs", abs(sm))
        cmd("robPermStrMeanWord", "above" if sm > 0 else "below")

# ---- the degree-preserving rewiring null -----------------------------------
W = S.get("rewired")
if W:
    cmd("rewN", W["n"], "int"); cmd("rewHeld", W["mean"])
    if W["sd"] is not None: cmd("rewHeldSD", W["sd"])
    cmd("rewTrain", W["trainrxn"]); cmd("rewGap", W["gap"])
    cmd("rewExpr", W["exprbearing"]); cmd("rewRho", W["rho"], "{:.2f}")
    cmd("rewInterR", W["interpat_r"])
    cmd("rewSeed", ", ".join(str(x) for x in W["seeds"]))
    cmd("rewPfoldList", ", ".join(str(x) for x in W["pfolds"])); cmd("rewPfolds", len(W["pfolds"]), "int")
    cmd("deltaRewire", -W["vs_real"]["delta"], "{:+.4f}")   # rewired minus real
    cmd("rewRealMean", W["vs_real"]["a_mean"])
    if W.get("vs_mlp") and W["vs_mlp"]["n"]: cmd("rewMlpSame", W["vs_mlp"]["a_mean"]); cmd("rewMlpSameN", W["vs_mlp"]["n"], "int")
    cmd("deltaRewireN", W["vs_real"]["n"], "int")
    for seed, W2 in (S.get("rewired_other_seeds") or {}).items():
        sn = {"11": "Eleven"}.get(seed, f"S{seed}")
        cmd(f"rew{sn}N", W2["n"], "int"); cmd(f"rew{sn}Held", W2["mean"])
        if W2["sd"] is not None: cmd(f"rew{sn}HeldSD", W2["sd"])
        cmd(f"rew{sn}Expr", W2["exprbearing"]); cmd(f"rew{sn}Train", W2["trainrxn"])
        cmd(f"rew{sn}CellList", ", ".join(W2["keys"]))
        cmd(f"deltaRew{sn}", -W2["vs_real"]["delta"], "{:+.4f}"); cmd(f"deltaRew{sn}N", W2["vs_real"]["n"], "int")
        cmd(f"rew{sn}RealSame", W2["vs_real"]["a_mean"])
        cmd(f"rew{sn}VsSevenDelta", W2["vs_seed7"]["delta"], "{:+.4f}"); cmd(f"rew{sn}VsSevenN", W2["vs_seed7"]["n"], "int")
        cmd(f"rew{sn}SevenSame", W2["vs_seed7"]["a_mean"])
        # the two seeds' disagreement as a share of the drop the second seed measures
        cmd(f"rew{sn}AgreeShare", 100.0 * abs(W2["vs_seed7"]["delta"]) / abs(W2["vs_real"]["delta"]), "{:.1f}")

# ---- the presence-only (indicator) arm ---------------------------------------
I = S.get("indicator")
if I:
    cmd("indN", I["n"], "int"); cmd("indHeld", I["mean"])
    if I["sd"] is not None: cmd("indHeldSD", I["sd"])
    cmd("indTrain", I["trainrxn"]); cmd("indExpr", I["exprbearing"])
    cmd("indRho", I["rho"], "{:.2f}"); cmd("indInterR", I["interpat_r"])
    cmd("indPfoldList", ", ".join(str(x) for x in I["pfolds"])); cmd("indPfolds", len(I["pfolds"]), "int")
    cmd("indCellList", ", ".join(I["keys"]))
    for k, n in (("presence_over_zero", "indDeltaPresence"), ("magnitude_over_presence", "indDeltaMagnitude"),
                 ("own_over_presence", "indDeltaOwn")):
        d = I[k]
        cmd(n, d["delta"], "{:+.4f}"); cmd(f"{n}N", d["n"], "int")
        if d["sd"] is not None: cmd(f"{n}SD", d["sd"])
        cmd(f"{n}A", d["a_mean"]); cmd(f"{n}B", d["b_mean"])
    cmd("indZeroSame", I["presence_over_zero"]["b_mean"]); cmd("indMeanSame", I["magnitude_over_presence"]["a_mean"])
    cmd("indRealSame", I["own_over_presence"]["a_mean"])
    cmd("indAggSame", I["cohort_mean_over_zero_same_cells"]["delta"], "{:+.4f}")
    if I["cohort_mean_over_zero_same_cells"]["delta"]:
        cmd("indPresenceShare", 100.0 * I["presence_over_zero"]["delta"] / I["cohort_mean_over_zero_same_cells"]["delta"], "{:.0f}")

# ---- naive baselines --------------------------------------------------------
NB = R["naive_baselines"]
NBN = {"chance":"nbChance", "indicator":"nbInd", "expr_perpat":"nbExprPer",
       "expr_cohortmean":"nbExprCoh", "degree_shared":"nbDegree",
       "subsystem_prevalence":"nbSubsys", "structure_only":"nbStruct",
       "structure_plus_cohortmean":"nbStructExpr",
       "expr_cohortmean_lr":"nbExprCohLr", "expr_perpat_lr":"nbExprPerLr",
       "structure_plus_perpat":"nbStructPer"}
for k, n in NBN.items():
    if k in NB:
        cmd(n, NB[k]["mean"])
        if NB[k].get("sd") is not None: cmd(f"{n}SD", NB[k]["sd"])
        cmd(f"{n}Cells", len(NB[k]["cells"]), "int")
if "_n_test_patients" in NB: cmd("nbTestPat", NB["_n_test_patients"], "int")

# ---- the decomposition ladder ----------------------------------------------
La = R["ladder"]
for k, n in [("indicator","ladInd"), ("expr_perpat","ladExprPer"),
             ("expr_cohortmean","ladExprCoh"), ("expr_cohortmean_lr","ladExprCohLr"),
             ("expr_perpat_lr","ladExprPerLr"), ("structure_only","ladStruct"),
             ("structure_plus_cohortmean","ladStructExpr"),
             ("structure_plus_perpat","ladStructPer")]:
    if La.get(k) is not None: cmd(n, La[k])
for k, n in [("gnn_over_lr_no_expression","deltaGnnLrNoExpr"),
             ("gnn_over_lr_with_expression","deltaGnnLrExpr"),
             ("structure_over_expression_fitted","deltaStructOverExpr"),
             ("lin_patient_expr_only","deltaLinPatExpr"),
             ("lin_patient_struct_expr","deltaLinPatStruct"),
             ("nonparam_patient_effect","deltaNonparPat")]:
    if La.get(k) is not None: cmd(n, La[k], "{:+.4f}")

# ---- cohort-mean collapse ---------------------------------------------------
K = R.get("collapse", {})
CN = {"mlp":"collMlp", "mlp_emb":"collEmb", "gnn_B":"collGnn"}
for m, n in CN.items():
    if m in K:
        s = K[m]
        cmd(f"{n}Per", s["perpat"]); cmd(f"{n}Coh", s["cohmean"])
        cmd(f"{n}Delta", s["paired_diff"], "{:+.4f}"); cmd(f"{n}N", s["n_cells"], "int")
        # the same quantity stated in the personalization direction, so a sentence about
        # what patient identity costs never has to mix the two sign conventions
        cmd(f"{n}Personal", -s["paired_diff"], "{:+.4f}")
        if s.get("paired_diff_sd"): cmd(f"{n}DeltaSD", s["paired_diff_sd"])
        cmd(f"{n}Shuf", s["shuffled"])

# ---- the collapse net of noise averaging ------------------------------------
KN = (R.get("collapse_noise") or {}).get("summary", {})
for m, n in CN.items():
    if m in KN:
        s = KN[m]
        cmd(f"{n}NoiseObs", s["observed_gain"], "{:+.4f}"); cmd(f"{n}NoiseOnly", s["noise_only_gain"], "{:+.4f}")
        cmd(f"{n}Ident", s["identity_component"], "{:+.4f}")
        cmd(f"{n}IdentPersonal", -s["identity_component"], "{:+.4f}")
        if s.get("identity_sd") is not None: cmd(f"{n}IdentSD", s["identity_sd"])
        cmd(f"{n}IdentPos", s["identity_n_positive"], "int"); cmd(f"{n}IdentN", s["n_cells"], "int")
        if s["observed_gain"]:
            cmd(f"{n}NoiseShare", s["noise_only_gain"] / s["observed_gain"], "pct")
if R.get("collapse_noise"):
    cmd("collNoiseT", R["collapse_noise"]["T"], "int"); cmd("collNoiseReps", R["collapse_noise"]["replicates"], "int")

# ---- the related estimates of the individual-versus-cohort-mean contrast, as one list -----------
# every estimate stated in the personalization direction: the unfitted ranking, the two linear
# comparisons fitted per patient, the two fitted once and frozen, the graph model's retraining, and
# the three collapses net of noise averaging; macros give their count, their extremes and the
# count of negatives, so that no sentence about the range has to name a particular estimator
_LBd = ((R.get("ladder_basis") or {}))
_PDt = (((R.get("naive_pooled") or {}).get("deltas") or {}).get("tuned") or {})
_est = []
for k in ("nonparam_patient_effect", "lin_patient_expr_only", "lin_patient_struct_expr"):
    if _LBd.get(k) is not None: _est.append((k, _LBd[k]))
for k in ("pooled_vs_cohort_expr", "pooled_vs_cohort_struct"):
    if _PDt.get(k): _est.append((k, _PDt[k]["delta"]))
if (R.get("substitution") or {}).get("patient_identity"): _est.append(("gnn_retrained", R["substitution"]["patient_identity"]["delta"]))
_IVe = ((R.get("robustness") or {}).get("ivr") or {})
if _IVe.get("patient_identity") and set(_IVe.get("basis") or []) == set((R.get("substitution") or {}).get("basis") or ["x"]):
    _est.append(("gnn_retrained_ivr", _IVe["patient_identity"]["delta"])); cmd("estHasIvr", "yes")
# the same retraining on the family-disjoint split, which is the same estimator on a harder split and
# therefore belongs in the ladder of estimates rather than beside it
_FAMe = ((R.get("robustness") or {}).get("fam") or {})
if _FAMe.get("patient_identity"):
    _est.append(("gnn_retrained_fam", _FAMe["patient_identity"]["delta"])); cmd("estHasFam", "yes")
for m in ("mlp", "mlp_emb", "gnn_B"):
    if m in KN: _est.append(("collapse_" + m, -KN[m]["identity_component"]))
if _est:
    _vals = [v for _, v in _est]
    # the largest contrast among the estimates that fit no architecture (the unfitted ranking and the
    # four linear rows), for the sentence that bounds the contrast before any architecture is chosen
    _pre = [v for k, v in _est if k in ("nonparam_patient_effect", "lin_patient_expr_only", "lin_patient_struct_expr", "pooled_vs_cohort_expr", "pooled_vs_cohort_struct")]
    if _pre: cmd("preArchMaxPat", max(_pre), "{:+.4f}"); cmd("preArchMinPat", min(_pre), "{:+.4f}")
    _w = {7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve"}.get(len(_est), str(len(_est)))
    cmd("estN", len(_est), "int"); cmd("estNWord", _w); cmd("estNWordCap", _w.capitalize())
    # the kinds of measurement among the estimates: a ranking with no model, comparisons fitted per
    # patient, comparisons fitted once and frozen, post-hoc collapses, and matched retrainings
    _kinds = {"nonparam_patient_effect": "ranking", "lin_patient_expr_only": "perpat", "lin_patient_struct_expr": "perpat",
              "pooled_vs_cohort_expr": "pooled", "pooled_vs_cohort_struct": "pooled", "gnn_retrained": "retrain",
              "gnn_retrained_ivr": "retrain", "gnn_retrained_fam": "retrain", "collapse_mlp": "collapse", "collapse_mlp_emb": "collapse", "collapse_gnn_B": "collapse"}
    _nk = len({_kinds.get(k, k) for k, _ in _est})
    cmd("estKinds", _nk, "int"); cmd("estKindsWord", {3: "three", 4: "four", 5: "five", 6: "six"}.get(_nk, str(_nk)))
    cmd("estMax", max(_vals), "{:+.4f}"); cmd("estMin", min(_vals), "{:+.4f}")
    cmd("estNeg", sum(1 for v in _vals if v < -0.001), "int"); cmd("estPos", sum(1 for v in _vals if v > 0.001), "int")
    cmd("estZero", sum(1 for v in _vals if abs(v) <= 0.001), "int")
    _nw = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}
    for _nm, _cnt in (("estNegWord", sum(1 for v in _vals if v < -0.001)), ("estPosWord", sum(1 for v in _vals if v > 0.001)), ("estZeroWord", sum(1 for v in _vals if abs(v) <= 0.001)),
                      ("estStrictPosWord", sum(1 for v in _vals if v > 0)), ("estStrictNegWord", sum(1 for v in _vals if v < 0)),
                      ("estAboveZeroWord", sum(1 for v in _vals if v >= 0.00005))):     # positive once rounded to four decimals
        cmd(_nm, _nw.get(_cnt, str(_cnt))); cmd(_nm + "Cap", _nw.get(_cnt, str(_cnt)).capitalize())

# ---- the ladder restated on the common basis, linear rows from every cell ------
LB = R.get("ladder_basis")
if LB:
    cmd("lbN", LB["n_basis"], "int")
    for k, n in [("indicator","lbInd"), ("expr_perpat","lbExprPer"), ("expr_cohortmean","lbExprCoh"),
                 ("expr_cohortmean_lr","lbExprCohLr"), ("expr_perpat_lr","lbExprPerLr"),
                 ("topology_only","lbTopo"), ("annotation_only","lbAnnot"), ("structure_only","lbStruct"),
                 ("structure_plus_cohortmean","lbStructExpr"), ("structure_plus_perpat","lbStructPer"),
                 ("expr_cohortmean_lr_fixedC","lbExprCohLrFixed"), ("topology_only_fixedC","lbTopoFixed"),
                 ("annotation_only_fixedC","lbAnnotFixed"), ("structure_only_fixedC","lbStructFixed"),
                 ("structure_plus_cohortmean_fixedC","lbStructExprFixed"),
                 ("structure_plus_perpat_fixedC","lbStructPerFixed"), ("expr_perpat_lr_fixedC","lbExprPerLrFixed")]:
        if LB.get(k) is not None: cmd(n, LB[k])
    for k, n in [("structure_over_expression_fitted","lbDeltaStructOverExpr"),
                 ("topology_over_expression_fitted","lbDeltaTopoOverExpr"),
                 ("annotation_over_expression_fitted","lbDeltaAnnotOverExpr"),
                 ("gnn_over_lr_no_expression","lbDeltaGnnLrNoExpr"),
                 ("gnn_over_lr_with_expression","lbDeltaGnnLrExpr"),
                 ("gnn_over_lr_own_expression","lbDeltaGnnLrOwn"),
                 ("gnn_mean_over_structure","lbDeltaGnnMeanOverStruct"),
                 ("gnn_real_over_structure","lbDeltaGnnRealOverStruct"),
                 ("lin_patient_expr_only","lbDeltaLinPatExpr"),
                 ("lin_patient_struct_expr","lbDeltaLinPatStruct"),
                 ("nonparam_patient_effect","lbDeltaNonparPat")]:
        if LB.get(k) is not None: cmd(n, LB[k], "{:+.4f}"); cmd(n + "Abs", abs(LB[k]), "{:.4f}")
    if LB.get("nonparam_positive_cells") is not None: cmd("lbNonparPos", LB["nonparam_positive_cells"], "int")
    if LB.get("gnn_over_lr_with_expression") is not None and LB.get("structure_over_expression_fitted"):
        cmd("lbGnnLrExprShareOfStructPct", 100.0 * LB["gnn_over_lr_with_expression"] / LB["structure_over_expression_fitted"], "{:.0f}")
    NAc = (R.get("naive_allfolds") or {}).get("columns") or {}
    for k, n in (("topology","lbColsTopo"), ("annotation","lbColsAnnot"), ("structure","lbColsStruct")):
        if k in NAc: cmd(n, NAc[k], "int")
    Cg = (R.get("naive_allfolds") or {}).get("C_grid")
    if Cg: cmd("lbCgrid", ", ".join(("{:g}".format(x)) for x in Cg))

# ---- a paired difference between two rows: signed value, magnitude, counts and the sign as a word -----
def delta_macros(pre, d, signed_only=False):
    if not d or not d.get("n"): return
    cmd(pre, d["delta"], "{:+.4f}")
    if signed_only: return
    cmd(pre + "Abs", abs(d["delta"])); cmd(pre + "N", d["n"], "int")
    if d.get("sd") is not None: cmd(pre + "SD", d["sd"])
    cmd(pre + "Pos", d["n_positive"], "int"); cmd(pre + "Neg", d["n_negative"], "int")
    cmd(pre + "Word", "benefit" if d["delta"] > 0 else "cost"); cmd(pre + "Dir", "above" if d["delta"] > 0 else "below")
    cmd(pre + "SignCount", "%d of %d" % (max(d["n_positive"], d["n_negative"]), d["n"]))
    if d.get("n_rfolds"):
        cmd(pre + "NRxn", d["n_rfolds"], "int"); cmd(pre + "PosRxn", d["n_rfolds_positive"], "int"); cmd(pre + "NegRxn", d["n_rfolds_negative"], "int")
    if d.get("a_mean") is not None: cmd(pre + "A", d["a_mean"]); cmd(pre + "B", d["b_mean"])

# ---- the pooled, frozen linear rows on the same basis (pooled_baselines.py) ---------------------
# One logistic regression per cell, fitted on the stacked rows of a seeded subsample of training
# patients and frozen before any test patient is scored; the per-patient rows fit one model per test
# patient. Emitted only when every basis cell has run, so that no row rests on fewer cells than the
# ladder it sits in.
NPB = R.get("naive_pooled")
if NPB and NPB.get("basis_complete"):
    T_, F_ = NPB["regimes"]["tuned"], NPB["regimes"]["fixed"]
    cmd("lbPooledN", NPB["n_basis"], "int")
    for k, n in (("expr_pooled_lr", "lbPooledExpr"), ("structure_plus_pooled", "lbPooledStruct")):
        if k in T_:
            cmd(n, T_[k]["mean_basis"])
            if T_[k].get("sd_basis") is not None: cmd(n + "SD", T_[k]["sd_basis"])
        if k in F_: cmd(n + "Fixed", F_[k]["mean_basis"])
    sub = NPB.get("subsample") or {}
    if sub.get("max_train_patients"): cmd("lbPooledMaxTrainPat", sub["max_train_patients"], "int")
    if sub.get("n_used_min"): cmd("lbPooledTrainPatMin", sub["n_used_min"], "int"); cmd("lbPooledTrainPatMax", sub["n_used_max"], "int")
    if NPB.get("n_train_rows_mean"): cmd("lbPooledRows", int(round(NPB["n_train_rows_mean"])), "int")
    cols = NPB.get("columns") or {}
    if "structure_plus_expression" in cols: cmd("lbPooledColsStruct", cols["structure_plus_expression"], "int")
    if "expression" in cols: cmd("lbPooledColsExpr", cols["expression"], "int")
    Cs = (NPB.get("C_selected") or {}).get("tuned") or {}
    for k, n in (("expr_pooled_lr", "lbPooledExprC"), ("structure_plus_pooled", "lbPooledStructC")):
        if k in Cs:
            vals = sorted({v for c, v in Cs[k].items() if c in set(S["basis"])})
            cmd(n, ", ".join("{:g}".format(v) for v in vals))
    PDN = (("pooled_vs_perpat_expr", "lbDeltaPooledVsPerExpr"), ("pooled_vs_cohort_expr", "lbDeltaPooledVsCohExpr"),
           ("pooled_vs_perpat_struct", "lbDeltaPooledVsPerStruct"), ("pooled_vs_cohort_struct", "lbDeltaPooledVsCohStruct"),
           ("pooled_struct_vs_structure_only", "lbDeltaPooledStructOverStruct"),
           ("pooled_struct_vs_pooled_expr", "lbDeltaPooledStructOverExpr"),
           ("gnn_real_vs_pooled_struct", "lbDeltaGnnRealOverPooledStruct"),
           ("gnn_cohort_mean_vs_pooled_struct", "lbDeltaGnnMeanOverPooledStruct"),
           ("gnn_real_vs_pooled_expr", "lbDeltaGnnRealOverPooledExpr"))
    for k, n in PDN:
        delta_macros(n, (NPB["deltas"].get("tuned") or {}).get(k))
        delta_macros(n + "Fixed", (NPB["deltas"].get("fixed") or {}).get(k), signed_only=True)

# ---- average precision beside the AUROC, on the same basis --------------------------------------
# Every AUPRC row is summarized on exactly the basis cells (the graph arms from per_patient.json, the
# linear rows from the "auprc" blocks of naive_allfolds.json and naive_pooled.json), or not at all;
# the presence-only and rewired arms are on their own cells, with their counts. The held-out
# prevalence is the average precision of a random ranking.
APR = R.get("auprc")
if APR:
    rows = APR.get("rows") or {}
    APN = {"gnn_real": "ladRealAuprc", "gnn_cohort_mean": "ladMeanAuprc", "gnn_zero": "ladZeroAuprc",
           "structure_only": "lbStructAuprc", "expr_cohortmean_lr": "lbExprCohLrAuprc", "expr_perpat_lr": "lbExprPerLrAuprc",
           "structure_plus_cohortmean": "lbStructExprAuprc", "structure_plus_perpat": "lbStructPerAuprc",
           "indicator": "lbIndAuprc", "expr_perpat_rank": "lbExprPerAuprc", "expr_cohortmean_rank": "lbExprCohAuprc",
           "topology_only": "lbTopoAuprc", "annotation_only": "lbAnnotAuprc",
           "degree_shared": "nbDegreeAuprc", "subsystem_prevalence": "nbSubsysAuprc",
           "expr_pooled_lr": "lbPooledExprAuprc", "structure_plus_pooled": "lbPooledStructAuprc",
           "gnn_indicator": "indHeldAuprc", "gnn_rewire7": "rewHeldAuprc"}
    for k, n in APN.items():
        v = rows.get(k)
        if v:
            cmd(n, v["mean"]); cmd(n + "N", v["n"], "int")
            if v.get("sd") is not None: cmd(n + "SD", v["sd"])
        f = (APR.get("rows_fixedC") or {}).get(k)
        if f: cmd(n + "Fixed", f["mean"])
    if rows: cmd("auprcN", APR["n_basis"], "int")
    pv = APR.get("prevalence")
    if pv:
        cmd("heldPrevalence", pv["mean"]); cmd("heldPrevalencePct", pv["mean"], "pct")
        cmd("heldPrevalenceMin", pv["min"]); cmd("heldPrevalenceMax", pv["max"])
        for rk, rv in (pv.get("by_rfold") or {}).items():
            cmd("heldPrevalence" + {"r0": "RZERO", "r1": "RONE", "r2": "RTWO"}[rk], rv)
    ADN = {"patient_identity": "deltaPatientAuprc", "aggregate_expression": "deltaAggExprAuprc",
           "expression_total": "deltaExprAuprc",
           "lin_patient_expr_only": "lbDeltaLinPatExprAuprc", "lin_patient_struct_expr": "lbDeltaLinPatStructAuprc",
           "nonparam_patient_effect": "lbDeltaNonparPatAuprc",
           "structure_over_expression_fitted": "lbDeltaStructOverExprAuprc",
           "gnn_over_lr_no_expression": "lbDeltaGnnLrNoExprAuprc", "gnn_over_lr_with_expression": "lbDeltaGnnLrExprAuprc",
           "gnn_over_lr_own_expression": "lbDeltaGnnLrOwnAuprc", "gnn_real_over_structure": "lbDeltaGnnRealOverStructAuprc",
           "pooled_vs_perpat_expr": "lbDeltaPooledVsPerExprAuprc", "pooled_vs_cohort_expr": "lbDeltaPooledVsCohExprAuprc",
           "pooled_vs_perpat_struct": "lbDeltaPooledVsPerStructAuprc", "pooled_vs_cohort_struct": "lbDeltaPooledVsCohStructAuprc",
           "gnn_real_vs_pooled_struct": "lbDeltaGnnRealOverPooledStructAuprc"}
    for k, n in ADN.items():
        delta_macros(n, (APR.get("deltas") or {}).get(k))
    # whether the AUROC and AUPRC versions of the contrast agree in sign, as a phrase
    _da = ((APR.get("deltas") or {}).get("patient_identity") or {}).get("delta")
    _dr = (R.get("substitution") or {}).get("patient_identity", {}).get("delta") if isinstance(R.get("substitution"), dict) else None
    if _da is not None and _dr is not None:
        cmd("deltaPatientAuprcSamePhrase", "the same sign as in AUROC" if (_da > 0) == (_dr > 0) else "the opposite sign to the AUROC contrast")
    # whether the fitted linear rows keep their order under AUPRC, and what the unfitted own- and
    # cohort-mean rankings do, as phrases
    _LBr = R.get("ladder_basis") or {}
    _fit = ["expr_perpat_lr", "expr_cohortmean_lr", "topology_only", "annotation_only", "structure_only", "structure_plus_cohortmean", "structure_plus_perpat"]
    if all(k in rows and _LBr.get(k) is not None for k in _fit):
        _o1 = sorted(_fit, key=lambda k: _LBr[k]); _o2 = sorted(_fit, key=lambda k: rows[k]["mean"])
        cmd("auprcFittedOrderPhrase", "the fitted linear rows keep their positions relative to one another under the second metric"
            if _o1 == _o2 else "the fitted linear rows do not all keep their positions relative to one another under the second metric")
    _np_a = ((APR.get("deltas") or {}).get("nonparam_patient_effect") or {}).get("delta"); _np_r = _LBr.get("nonparam_patient_effect")
    if _np_a is not None and _np_r is not None:
        if (_np_a > 0) != (_np_r > 0):
            _npd = (APR.get("deltas") or {}).get("nonparam_patient_effect")
            cmd("auprcNonparPhrase", "the unfitted own- and cohort-mean rankings exchange places, so the no-model estimate of the contrast is "
                f"{_np_a:+.4f} in AUPRC ({max(_npd['n_negative'], _npd['n_positive'])} of {_npd['n']} cells) against {_np_r:+.4f} in AUROC")
        else:
            cmd("auprcNonparPhrase", "the unfitted own- and cohort-mean rankings keep their order as well")
    # the presence-only indicator against reaction degree, the two shortcuts, under both metrics
    _ia, _da_ = (rows.get("indicator") or {}).get("mean"), (rows.get("degree_shared") or {}).get("mean")
    _ir, _dr_ = _LBr.get("indicator"), (R["naive_baselines"].get("degree_shared") or {}).get("mean")
    if None not in (_ia, _da_, _ir, _dr_):
        if (_ir > _dr_) and (_ia < _da_):
            cmd("auprcIndDegreePhrase", f"the presence-only indicator, above reaction degree in AUROC, falls below it in AUPRC ({_ia:.4f} against {_da_:.4f})")
        elif (_ir < _dr_) and (_ia > _da_):
            cmd("auprcIndDegreePhrase", f"the presence-only indicator, below reaction degree in AUROC, rises above it in AUPRC ({_ia:.4f} against {_da_:.4f})")
        else:
            cmd("auprcIndDegreePhrase", "the presence-only indicator and reaction degree keep their order")

# ---- permutation nulls through the identical pipeline -------------------------
PN = R.get("phenotype_null", {})
for a, an in (("msi","Msi"), ("cin_vs_gs","Cin"), ("sex","Sex"), ("site","Site")):
    v = PN.get(a)
    if not v: continue
    o = v["output"]
    cmd(f"pheNull{an}Obs", o["auroc"], "{:.3f}"); cmd(f"pheNull{an}Mean", o["perm_null_mean"], "{:.3f}")
    cmd(f"pheNull{an}SD", o["perm_null_sd"], "{:.3f}"); cmd(f"pheNull{an}Max", o["perm_null_max"], "{:.3f}")
    cmd(f"pheNull{an}P", pfmt(o["perm_p"])); cmd(f"pheNull{an}Floor", pfmt(o["perm_p_floor"]))
    cmd(f"pheNull{an}NPerm", o["n_perm"], "int")
    obf = v.get("output_by_fold") or {}
    if obf:
        cmd(f"pheNull{an}Nrf", len(obf), "int")
        for rf, w in obf.items():
            rn = {"r0": "RZERO", "r1": "RONE", "r2": "RTWO"}[rf]
            cmd(f"pheNull{an}P{rn}", pfmt(w["perm_p"])); cmd(f"pheNull{an}Obs{rn}", w["auroc"], "{:.3f}")
            cmd(f"pheNull{an}Max{rn}", w["perm_null_max"], "{:.3f}")
        cmd(f"pheNull{an}PMin", pfmt(min(w["perm_p"] for w in obf.values())))
        cmd(f"pheNull{an}FoldsAtFloor", sum(1 for w in obf.values() if w["perm_p"] <= 1.0 / (o["n_perm"] + 1) + 1e-12), "int")
        worst = max(obf.items(), key=lambda kv: kv[1]["perm_p"])[0]
        cmd(f"pheNull{an}WorstFold", {"r0": "first", "r1": "second", "r2": "third"}[worst])
    if v.get("input"):
        i = v["input"]
        cmd(f"pheNull{an}InObs", i["auroc"], "{:.3f}"); cmd(f"pheNull{an}InMean", i["perm_null_mean"], "{:.3f}")
        cmd(f"pheNull{an}InSD", i["perm_null_sd"], "{:.3f}"); cmd(f"pheNull{an}InP", pfmt(i["perm_p"]))

# ---- the Monte Carlo error of the noise term of the collapse correction --------
CNs = (R.get("collapse_noise") or {}).get("summary") or {}
for mk, mn in (("mlp","Mlp"),("mlp_emb","Emb"),("gnn_B","Gnn")):
    v = CNs.get(mk) or {}
    if v.get("noise_only_mc_se") is not None: cmd(f"coll{mn}NoiseSE", v["noise_only_mc_se"], "{:.5f}")
    if v.get("noise_only_gain") is not None: cmd(f"coll{mn}NoiseSigned", v["noise_only_gain"], "{:+.4f}")

# ---- per-score Monte Carlo error by arm --------------------------------------
ANs = (R.get("arm_noise") or {}).get("summary") or {}
for ak, an in (("real","Real"),("mean","Mean"),("zero","Zero"),("indicator","Ind"),("rewire7","Rew")):
    if ak in ANs: cmd(f"armSE{an}", ANs[ak]["mean_se"], "{:.4f}"); cmd(f"armSE{an}N", ANs[ak]["n_cells"], "int")

# ---- inference on the cell as the sampling unit -----------------------------
INF = R.get("inference", {})
IN = {"real_vs_mean": "infPat", "mean_vs_zero": "infAgg", "real_vs_zero": "infExpr",
      "real_vs_rewire7": "infRew", "real_vs_rewire11": "infRewEleven", "rewire7_vs_rewire11": "infRewSevenEleven",
      "indicator_vs_zero": "infIndPres",
      "mean_vs_indicator": "infIndMag", "real_vs_indicator": "infIndOwn", "collapse_mlp": "infCollMlp",
      "collapse_mlp_emb": "infCollEmb", "collapse_gnn_B": "infCollGnn",
      "collapse_identity_mlp": "infIdMlp", "collapse_identity_mlp_emb": "infIdEmb",
      "collapse_identity_gnn_B": "infIdGnn",
      "lin_tuned_structure_only_vs_expr_cohortmean_lr": "infLinStructExpr",
      "lin_tuned_topology_only_vs_expr_cohortmean_lr": "infLinTopoExpr",
      "lin_tuned_annotation_only_vs_expr_cohortmean_lr": "infLinAnnotExpr",
      "lin_tuned_structure_plus_perpat_vs_structure_plus_cohortmean": "infLinPatStruct",
      "lin_tuned_expr_perpat_lr_vs_expr_cohortmean_lr": "infLinPatExpr",
      "lin_tuned_structure_plus_cohortmean_vs_structure_only": "infLinAgg",
      "gnn_zero_vs_lin_tuned_structure_only": "infGnnLrNoExpr",
      "gnn_mean_vs_lin_tuned_structure_plus_cohortmean": "infGnnLrExpr",
      "gnn_real_vs_lin_tuned_structure_plus_perpat": "infGnnLrOwn",
      "gnn_mean_vs_lin_tuned_structure_only": "infGnnMeanStruct",
      "gnn_real_vs_lin_tuned_structure_only": "infGnnRealStruct",
      "gnn_real_vs_lin_fixed_structure_plus_perpat": "infGnnLrOwnFixed",
      "lin_fixed_structure_only_vs_expr_cohortmean_lr": "infLinStructExprFixed",
      "gnn_zero_vs_lin_fixed_structure_only": "infGnnLrNoExprFixed",
      "gnn_mean_vs_lin_fixed_structure_plus_cohortmean": "infGnnLrExprFixed",
      # the pooled, frozen linear rows (pooled_baselines.py), present only when that file has run
      "lin_tuned_expr_pooled_lr_vs_expr_perpat_lr": "infPooledPerExpr",
      "lin_tuned_expr_pooled_lr_vs_expr_cohortmean_lr": "infPooledCohExpr",
      "lin_tuned_structure_plus_pooled_vs_structure_plus_perpat": "infPooledPerStruct",
      "lin_tuned_structure_plus_pooled_vs_structure_plus_cohortmean": "infPooledCohStruct",
      "gnn_real_vs_lin_tuned_structure_plus_pooled": "infGnnLrPooled",
      "lin_fixed_expr_pooled_lr_vs_expr_perpat_lr": "infPooledPerExprFixed",
      "lin_fixed_expr_pooled_lr_vs_expr_cohortmean_lr": "infPooledCohExprFixed",
      "lin_fixed_structure_plus_pooled_vs_structure_plus_perpat": "infPooledPerStructFixed",
      "lin_fixed_structure_plus_pooled_vs_structure_plus_cohortmean": "infPooledCohStructFixed",
      "gnn_real_vs_lin_fixed_structure_plus_pooled": "infGnnLrPooledFixed",
      # the arms under inner-validation-reaction early stopping (results/ivr/)
      "ivr_real_vs_mean": "infIvrPat", "main_real_vs_mean_on_ivr_cells": "infIvrMainPat",
      "ivr_real_vs_main_real": "infIvrRealVsMain", "ivr_mean_vs_main_mean": "infIvrMeanVsMain",
      "ivr_real_vs_lin_tuned_structure_plus_perpat": "infIvrGnnLrOwn",
      "ivr_real_vs_lin_tuned_structure_plus_pooled": "infIvrGnnLrPooled",
      "ivr_mean_vs_lin_tuned_structure_plus_cohortmean": "infIvrGnnLrExpr",
      "ivr_real_vs_lin_tuned_structure_only": "infIvrGnnRealStruct",
      "ivr_mean_vs_lin_tuned_structure_only": "infIvrGnnMeanStruct",
      "ivr_mean_vs_zero": "infIvrAgg", "ivr_real_vs_zero": "infIvrExprTot", "ivr_indicator_vs_zero": "infIvrPresence",
      "ivr_mean_vs_indicator": "infIvrMagnitude", "ivr_real_vs_permute": "infIvrPerm", "ivr_permute_vs_mean": "infIvrPermMean",
      # the synthetic patient-varying positive control (results/synth/)
      "synth_full_ivr_real_vs_mean": "infSynIvrFullPat", "synth_full_ivr_real_vs_mean_vary": "infSynIvrFullPatVary",
      "synth_half_ivr_real_vs_mean": "infSynIvrHalfPat", "synth_half_ivr_real_vs_mean_vary": "infSynIvrHalfPatVary",
      "synth_full_main_real_vs_mean": "infSynMainFullPat", "synth_full_main_real_vs_mean_vary": "infSynMainFullPatVary",
      "synth_half_main_real_vs_mean": "infSynMainHalfPat", "synth_half_main_real_vs_mean_vary": "infSynMainHalfPatVary"}
for k, n in IN.items():
    c = INF.get(k)
    if not c:
        continue
    cmd(f"{n}N", c["n"], "int")
    cmd(f"{n}Mean", c["mean"], "{:+.4f}"); cmd(f"{n}MeanAbs", abs(c["mean"]), "{:.4f}")
    if c.get("sd") is not None:
        cmd(f"{n}SD", c["sd"])
        if c["n"] > 1:
            from scipy import stats as _st
            _h = _st.t.ppf(0.975, c["n"] - 1) * c["sd"] / (c["n"] ** 0.5)
            cmd(f"{n}Lo", c["mean"] - _h, "{:+.4f}"); cmd(f"{n}Hi", c["mean"] + _h, "{:+.4f}")
    if c.get("cohens_dz") is not None: cmd(f"{n}Dz", c["cohens_dz"], "{:+.1f}")
    # p-values never understate: rounded up at four decimals, two significant digits below 0.001
    cmd(f"{n}P", pfmt(c["signflip_p"])); cmd(f"{n}Floor", pfmt(c["signflip_floor"]))
    if "n_negative" in c:
        cmd(f"{n}Neg", c["n_negative"], "int"); cmd(f"{n}Pos", c["n_positive"], "int")
        # the sign counted rather than asserted: "15 of 15" together with the word for its direction
        cmd(f"{n}SignCount", "%d of %d" % (max(c["n_negative"], c["n_positive"]), c["n"]))
        cmd(f"{n}SignWord", "negative" if c["n_negative"] >= c["n_positive"] else "positive")
    cmd(f"{n}Word", "benefit" if c["mean"] > 0 else "cost")
    cmd(f"{n}Dir", "above" if c["mean"] > 0 else "below")
    for bk, bn in (("block_patient", "BlockPat"), ("block_reaction", "BlockRxn")):
        b = c.get(bk)
        if b: cmd(f"{n}{bn}P", pfmt(b["p"])); cmd(f"{n}{bn}Floor", pfmt(b["floor"])); cmd(f"{n}{bn}N", b["n_blocks"], "int")
    for tk, tn in (("t_patient_folds", "TPat"), ("t_reaction_folds", "TRxn")):
        b = c.get(tk)
        if b and b.get("p") is not None:
            cmd(f"{n}{tn}T", b["t"], "{:+.1f}"); cmd(f"{n}{tn}P", pfmt(b["p"])); cmd(f"{n}{tn}N", b["n"], "int")
            cmd(f"{n}{tn}Mean", b["mean"], "{:+.4f}"); cmd(f"{n}{tn}SD", b["sd"])
    v = c.get("variance") or {}
    for vk, vn in [("frac_patient_fold", "VarPat"), ("frac_reaction_fold", "VarRxn"),
                   ("frac_residual", "VarRes")]:
        if v.get(vk) is not None:
            cmd(f"{n}{vn}", v[vk], "pct"); cmd(f"{n}{vn}Whole", v[vk], "pct0")
    v_ = c.get("variance") or {}
    if v_.get("n_patient_folds"): cmd(f"{n}Pfolds", v_["n_patient_folds"], "int")
    if v_.get("n_reaction_folds"): cmd(f"{n}Rfolds", v_["n_reaction_folds"], "int")
    if (c.get("variance") or {}).get("balanced") is not None:
        cmd(f"{n}Balanced", "balanced" if c["variance"]["balanced"] else "unbalanced")
    w = c.get("within_cell_sd") or {}
    if w.get("a") is not None: cmd(f"{n}WithinA", w["a"])
    if w.get("b") is not None: cmd(f"{n}WithinB", w["b"])

# ---- does the per-patient ordering survive an independent retraining? -------
RP = R.get("inference_reproducibility") or {}
for m, n in (("real", "reproReal"), ("mean", "reproMean"), ("zero", "reproZero")):
    if m in RP and RP[m].get("r_mean") is not None:
        cmd(f"{n}R", RP[m]["r_mean"], "{:+.3f}")
        cmd(f"{n}Lo", RP[m]["r_min"], "{:+.3f}")
        cmd(f"{n}Hi", RP[m]["r_max"], "{:+.3f}")
        cmd(f"{n}N", RP[m]["n_pairs"], "int")

# ---- strata ----------------------------------------------------------------
ST = R.get("strata")
if ST:
    SN = {"gpr_rule":"Gpr","no_gpr_rule":"NoGpr","expression_bearing":"Expr","no_expression":"NoExpr",
          "transport_exchange":"Transp","metabolic":"Metab","degree_low":"DegLo","degree_mid":"DegMid",
          "degree_high":"DegHi"}
    AN = {"real":"Real","mean":"Mean","zero":"Zero","rewire7":"Rew","indicator":"Ind"}
    for sk, sn in SN.items():
        cmd(f"str{sn}N", ST["sizes"][sk], "int"); cmd(f"str{sn}Prev", ST["prevalence"][sk], "pct")
        if (ST.get("heldout_mean") or {}).get(sk) is not None: cmd(f"str{sn}Held", round(ST["heldout_mean"][sk]), "int")
        for ak, an in AN.items():
            v = ST["rollup"].get(ak, {}).get(sk)
            if v: cmd(f"str{sn}{an}", v["mean"]); cmd(f"str{sn}{an}Cells", v["n"], "int")
        for pk, pn in (("aggregate_expression", "Agg"), ("patient_identity", "Pat"), ("expression_total", "Tot"),
                       ("presence_over_zero", "Pres"), ("magnitude_over_presence", "Mag"), ("own_over_presence", "OwnPres")):
            v = (ST.get("paired") or {}).get(pk, {}).get(sk)
            if v: cmd(f"str{sn}Delta{pn}", v["mean"], "{:+.4f}"); cmd(f"str{sn}Delta{pn}Pos", v["n_positive"], "int"); cmd(f"str{sn}Delta{pn}N", v["n"], "int")
    if ST.get("basis_n") is not None: cmd("strBasisN", ST["basis_n"], "int")
    rw = ST["rollup"].get("rewire7", {})
    if rw: cmd("strRewCells", max(v["n"] for v in rw.values()), "int")
    iw = ST["rollup"].get("indicator", {})
    if iw: cmd("strIndCells", max(v["n"] for v in iw.values()), "int")
    if (ST.get("paired") or {}).get("patient_identity"):
        pv = ST["paired"]["patient_identity"]
        cmd("strMeanBeatsReal", sum(1 for v in pv.values() if v["mean"] < 0), "int"); cmd("strClasses", len(pv), "int")

# ---- phenotype -------------------------------------------------------------
PH = R.get("phenotype")
if PH:
    cmd("phePos", PH["arms"]["real"]["n_pos"], "int")
    cmd("pheN", PH["arms"]["real"]["n_patients"], "int")
    for ak, an in (("real","Real"),("mean","Mean"),("zero","Zero"),("rewire7","Rew"),("real_on_rewired_patients","RealOnRew")):
        a = PH["arms"].get(ak)
        if not a or "auroc" not in a: continue
        cmd(f"phe{an}", a["auroc"], "{:.3f}"); cmd(f"phe{an}Lo", a["lo"], "{:.3f}"); cmd(f"phe{an}Hi", a["hi"], "{:.3f}")
        if a.get("sd") is not None: cmd(f"phe{an}SD", a["sd"], "{:.3f}")
        cmd(f"phe{an}Nrf", a["n_rfolds"], "int"); cmd(f"phe{an}Npat", a["n_patients"], "int")
        for rk, rv in a["per_rfold"].items():
            cmd(f"phe{an}R{ {'0':'ZERO','1':'ONE','2':'TWO'}[rk[1]] }", rv, "{:.3f}")
    if PH.get("input"):
        cmd("pheInput", PH["input"]["auroc"], "{:.3f}"); cmd("pheInputLo", PH["input"]["ci_lo"], "{:.3f}"); cmd("pheInputHi", PH["input"]["ci_hi"], "{:.3f}")
    if PH.get("genes"):
        cmd("pheGenes", PH["genes"]["auroc"], "{:.3f}"); cmd("pheGenesLo", PH["genes"]["ci_lo"], "{:.3f}"); cmd("pheGenesHi", PH["genes"]["ci_hi"], "{:.3f}")
    pm = PH["arms"].get("real", {}).get("permutation")
    if pm:
        cmd("phePermNull", pm["perm_null_mean"], "{:.3f}"); cmd("phePermSD", pm["perm_null_sd"], "{:.3f}")
        cmd("phePermP", pfmt(pm["perm_p"]))
    if PH.get("input") and PH["arms"].get("real"):
        cmd("pheRealMinusInput", PH["arms"]["real"]["auroc"] - PH["input"]["auroc"], "{:+.3f}")
        cmd("pheRealMinusInputAbs", abs(PH["arms"]["real"]["auroc"] - PH["input"]["auroc"]), "{:.3f}")

# ---- further attributes -----------------------------------------------------
PM = R.get("phenotype_multi", {})
for a, an in (("cin_vs_gs", "Cin"), ("sex", "Sex"), ("site", "Site")):
    v = PM.get(a)
    if not v: continue
    if v.get("n"): cmd(f"phe{an}N", v["n"], "int"); cmd(f"phe{an}Pos", v["n_pos"], "int")
    if v.get("real") is not None:
        cmd(f"phe{an}Real", v["real"], "{:.3f}"); cmd(f"phe{an}RealLo", v["real_lo"], "{:.3f}"); cmd(f"phe{an}RealHi", v["real_hi"], "{:.3f}")
    if v.get("input") is not None:
        cmd(f"phe{an}Input", v["input"], "{:.3f}"); cmd(f"phe{an}InputLo", v["input_lo"], "{:.3f}"); cmd(f"phe{an}InputHi", v["input_hi"], "{:.3f}")
    if v.get("cohort_mean") is not None:
        cmd(f"phe{an}Mean", v["cohort_mean"], "{:.3f}")
        if v.get("cohort_mean_n"): cmd(f"phe{an}MeanN", v["cohort_mean_n"], "int"); cmd(f"phe{an}MeanPos", v["cohort_mean_pos"], "int")
    if v.get("real") is not None and v.get("input") is not None:
        cmd(f"phe{an}Gap", v["real"] - v["input"], "{:+.3f}")

# ---- the phenotype probe fold by fold ----------------------------------------------------------
# The probe runs once per reaction fold with its own patient-level bootstrap interval; the phe* macros
# above carry the mean over folds with the envelope of the three intervals. These give each fold its
# own value and interval (Zero, One, Two), the mean over folds with its spread, and, where the probe
# file stores the out-of-fold predictions, an interval for that mean from resampling patients across
# folds (FoldMeanLo, FoldMeanHi); the envelope is kept under FoldEnvLo and FoldEnvHi for reference.
# FoldRow is one table row: an "AUROC [lo, hi]" cell per reaction fold, n/a where a fold was not run,
# then the mean (with its interval when one exists); the input has no folds and its row spans them.
FOLDNAME = {0: "Zero", 1: "One", 2: "Two"}
def fold_macros(pre, recs, agg):
    for r in recs:
        if r.get("fold") is None: continue
        fn = FOLDNAME[r["fold"]]
        cmd(f"{pre}Fold{fn}", r["auroc"], "{:.3f}"); cmd(f"{pre}Fold{fn}Lo", r["lo"], "{:.3f}"); cmd(f"{pre}Fold{fn}Hi", r["hi"], "{:.3f}")
        cmd(f"{pre}Fold{fn}N", r["n"], "int"); cmd(f"{pre}Fold{fn}Pos", r["n_pos"], "int")
        if r.get("perm_p") is not None: cmd(f"{pre}Fold{fn}P", pfmt(r["perm_p"]))
    if not agg: return
    cmd(f"{pre}FoldMean", agg["mean"], "{:.3f}"); cmd(f"{pre}FoldNrf", agg["n_rfolds"], "int")
    cmd(f"{pre}FoldMin", agg["min"], "{:.3f}"); cmd(f"{pre}FoldMax", agg["max"], "{:.3f}")
    if agg.get("sd") is not None: cmd(f"{pre}FoldSD", agg["sd"], "{:.3f}")
    cmd(f"{pre}FoldEnvLo", agg["envelope_lo"], "{:.3f}"); cmd(f"{pre}FoldEnvHi", agg["envelope_hi"], "{:.3f}")
    if agg.get("lo") is not None:
        cmd(f"{pre}FoldMeanLo", agg["lo"], "{:.3f}"); cmd(f"{pre}FoldMeanHi", agg["hi"], "{:.3f}")
        cmd(f"{pre}FoldMeanBoot", agg["n_boot"], "int")
    by = {r["fold"]: r for r in recs if r.get("fold") is not None}
    cells = [("%.3f [%.3f, %.3f]" % (by[f]["auroc"], by[f]["lo"], by[f]["hi"])) if f in by else "n/a" for f in (0, 1, 2)]
    mean_cell = "%.3f" % agg["mean"] + (" [%.3f, %.3f]" % (agg["lo"], agg["hi"]) if agg.get("lo") is not None else "")
    cmd(f"{pre}FoldRow", " & ".join(cells + [mean_cell]))
def input_row(pre, rec):
    cmd(f"{pre}FoldRow", "\\multicolumn{3}{c}{%.3f [%.3f, %.3f]} & %.3f" % (rec["auroc"], rec["lo"], rec["hi"], rec["auroc"]))
    if rec.get("perm_p") is not None: cmd(f"{pre}FoldP", pfmt(rec["perm_p"]))
PH = R.get("phenotype")
if PH and PH.get("per_fold"):
    for ak, an in (("real", "Real"), ("mean", "Mean"), ("zero", "Zero"), ("rewire7", "Rew"), ("real_on_rewired_patients", "RealOnRew")):
        if ak in PH["per_fold"]: fold_macros("phe" + an, PH["per_fold"][ak], (PH.get("aggregate") or {}).get(ak))
    if PH["per_fold"].get("input"): input_row("pheInput", PH["per_fold"]["input"][0])
_mean_devs = []
if PH and isinstance((PH.get("aggregate") or {}).get("mean"), dict) and PH["aggregate"]["mean"].get("mean") is not None:
    _mean_devs.append(abs(PH["aggregate"]["mean"]["mean"] - 0.5))
elif PH and PH["arms"].get("mean", {}).get("auroc") is not None:
    _mean_devs.append(abs(PH["arms"]["mean"]["auroc"] - 0.5))
for a, an in (("cin_vs_gs", "Cin"), ("sex", "Sex"), ("site", "Site")):
    v = R.get("phenotype_multi", {}).get(a)
    if v and isinstance((v.get("aggregate") or {}).get("mean"), dict) and v["aggregate"]["mean"].get("mean") is not None:
        _mean_devs.append(abs(v["aggregate"]["mean"]["mean"] - 0.5))
    if not v or not v.get("per_fold"): continue
    for ak, arm_n in (("real", "Real"), ("mean", "Mean")):
        if ak in v["per_fold"]: fold_macros(f"phe{an}{arm_n}", v["per_fold"][ak], (v.get("aggregate") or {}).get(ak))
    if v["per_fold"].get("input"): input_row(f"phe{an}Input", v["per_fold"]["input"][0])
if _mean_devs: cmd("pheMeanMaxDev", max(_mean_devs), "{:.3f}")

# ---- the split audit: near-duplicate held-out reactions (split_audit.py) -------------------------
# Shares are percentages of the held-out reactions (all three folds together, every reaction held out
# once), with the smallest and largest fold beside them.
SA = R.get("split_audit")
if SA:
    a = SA["all"]
    cmd("auditHeldN", a["n_heldout"], "int"); cmd("auditRfolds", SA.get("rfolds") or len(SA["folds"]), "int")
    cmd("auditHeldPrev", a["prevalence_heldout"], "pct")
    cmd("auditHeldNFoldMin", min(a["heldout_per_fold"]), "int"); cmd("auditHeldNFoldMax", max(a["heldout_per_fold"]), "int")
    for c, nm in (("stoich", "Stoich"), ("stoich_comp", "StoichComp"), ("gpr", "Gpr"), ("any", "Any"),
                  ("stoich_set", "StoichSet"), ("stoich_reversed", "StoichRev"), ("gpr_bigg", "GprBigg")):
        v = a.get(c)
        if not v: continue
        cmd(f"auditDup{nm}N", v["n_matched"], "int"); cmd(f"auditDup{nm}Share", v["share_matched"], "pct")
        cmd(f"auditDup{nm}ShareMin", v["share_min"], "pct"); cmd(f"auditDup{nm}ShareMax", v["share_max"], "pct")
        for rf, sh in enumerate(v.get("share_by_fold") or []):
            cmd(f"auditDup{nm}Share" + ["RZERO", "RONE", "RTWO"][rf], sh, "pct")
        if v.get("active_share_matched") is not None:
            cmd(f"auditDup{nm}Active", v["active_share_matched"], "pct"); cmd(f"auditDup{nm}ActiveN", v["n_matched_active"], "int")
        if v.get("active_share_unmatched") is not None: cmd(f"auditDup{nm}ActiveUnmatched", v["active_share_unmatched"], "pct")
        if v.get("label_agreement") is not None:
            cmd(f"auditDup{nm}Agree", v["label_agreement"], "pct"); cmd(f"auditDup{nm}AgreeMajority", v["label_agreement_majority_share"], "pct")
        if v.get("partners_per_matched") is not None: cmd(f"auditDup{nm}Partners", v["partners_per_matched"], "{:.1f}")

# ---- the same audit on the family-disjoint folds: what the grouping removed and what it left ----
SAF = R.get("split_audit_family")
if SAF and SA:
    fa, ra = SAF["all"], SA["all"]
    for c, nm in (("stoich", "Stoich"), ("stoich_comp", "StoichComp"), ("gpr", "Gpr"), ("any", "Any"),
                  ("gpr_bigg", "GprBigg"), ("stoich_set", "StoichSet")):
        v = fa.get(c)
        if not v: continue
        cmd(f"auditFamDup{nm}N", v["n_matched"], "int"); cmd(f"auditFamDup{nm}Share", v["share_matched"], "pct")
        cmd(f"auditFamDup{nm}ShareMin", v["share_min"], "pct"); cmd(f"auditFamDup{nm}ShareMax", v["share_max"], "pct")
        if v.get("label_agreement") is not None: cmd(f"auditFamDup{nm}Agree", v["label_agreement"], "pct")
    # a generated reading of what the grouping achieved, so no direction is typed by hand
    _cleared = [nm for c, nm in (("stoich", "identical stoichiometry"), ("stoich_comp", "compartment-stripped stoichiometry"),
                                 ("gpr_bigg", "the reference network's own gene rule"))
                if (fa.get(c) or {}).get("n_matched") == 0]
    cmd("auditFamClearedList", ", ".join(_cleared) if _cleared else "none of the relations")
    cmd("auditFamClearedN", len(_cleared), "int")
    _r, _f = ra["any"]["share_matched"], fa["any"]["share_matched"]
    cmd("auditFamAnyWord", "falls" if _f < _r else ("rises" if _f > _r else "is unchanged"))
    cmd("auditFamResidualWord", "none" if fa["any"]["n_matched"] == 0 else "a residue")

# ---- the phenotype annotation overlap ----------------------------------------
PA = R.get("phenotype_annotation")
if PA:
    for k, n in (("n_cin","pheAnnotCin"), ("n_gs","pheAnnotGs"), ("cin_msih","pheAnnotCinMsiH"), ("gs_msih","pheAnnotGsMsiH"),
                 ("cin_mss","pheAnnotCinMss"), ("gs_mss","pheAnnotGsMss"), ("n_msi_subtype","pheAnnotMsiSubtype"),
                 ("n_pole","pheAnnotPole"), ("n_no_subtype","pheAnnotNoSubtype")):
        cmd(n, PA[k], "int")

# ---- the nearest-family label lookup, and the verification rerun of the linear suite ---------
LR = R.get("linear_rerun")
if LR:
    cmd("rerunRows", LR["rows_compared"], "int"); cmd("rerunCells", LR["cells_compared"], "int")
    cmd("rerunMaxDiff", LR["max_abs_difference"], "{:.0e}")
    cmd("rerunPhrase", ("every value agrees to the last stored digit"
                        if LR["reproduces_exactly"] else
                        f"the largest disagreement is {LR['max_abs_difference']:.2e}"))
    if LR.get("nearest_lookup"):
        cmd("lookupRand", LR["nearest_lookup"]["mean"]); cmd("lookupRandN", LR["nearest_lookup"]["n"], "int")
        if LR["nearest_lookup"].get("sd") is not None: cmd("lookupRandSD", LR["nearest_lookup"]["sd"])
    _lf = LR.get("nearest_lookup_family")
    if _lf:
        cmd("lookupFam", _lf["mean"]); cmd("lookupFamN", _lf["n"], "int")
        cmd("lookupDrop", _lf["delta"], "{:+.4f}"); cmd("lookupDropAbs", abs(_lf["delta"]))
        cmd("lookupDropSD", _lf["sd"]); cmd("lookupDropNeg", _lf["n_negative"], "int")
        cmd("lookupDropWord", "falls" if _lf["delta"] < 0 else "rises")
        cmd("lookupDropSignCount", "%d of %d" % (max(_lf["n_negative"], _lf["n"] - _lf["n_negative"]), _lf["n"]))
        # how the lookup on the random split stands against the fitted rows, which is the point:
        # a table lookup that reaches the fitted models' level is a statement about the benchmark
        _st = (((R.get("naive_allfolds") or {}).get("regimes") or {}).get("tuned") or {}).get("structure_only")
        _stm = (_st or {}).get("mean_all") if _st else None
        if _stm is not None:
            _d = LR["nearest_lookup"]["mean"] - _stm
            cmd("lookupVsStructRand", _d, "{:+.4f}"); cmd("lookupVsStructRandAbs", abs(_d))
            cmd("lookupVsStructRandDir", "above" if _d > 0 else "below")
        _iv2 = ((R.get("robustness") or {}).get("ivr") or {}).get("real")
        if _iv2 and _iv2.get("mean") is not None:
            _d = LR["nearest_lookup"]["mean"] - _iv2["mean"]
            cmd("lookupVsGnnRand", _d, "{:+.4f}"); cmd("lookupVsGnnRandAbs", abs(_d))
            cmd("lookupVsGnnRandDir", "above" if _d > 0 else "below")

# ---- the prediction ledger, so the back matter can say what it covers -----------
_LG = os.path.join(HERE, "results", "LEDGER.json")
if os.path.exists(_LG):
    _lg = json.load(open(_LG))
    cmd("ledgerFamilies", _lg["n_families"], "int"); cmd("ledgerFiles", _lg["n_result_files"], "int")
    cmd("ledgerRecords", len(_lg["data_records"]), "int")
    _pa = _lg.get("prediction_arrays")
    if _pa:
        cmd("ledgerArrays", _pa["n_arrays"], "int"); cmd("ledgerTarballs", len(_pa.get("tarballs", [])), "int")
        # the tarball sizes are not in the ledger; the arrays' total is read from the checksum list's
        # companion when present, else from the released tarballs' recorded total
        _gib = _pa.get("total_bytes")
        if _gib: cmd("ledgerArraysGB", _gib / 1e9, "{:.1f}")

# ---- status -----------------------------------------------------------------
St = R["status"]
cmd("gridDone", St["grid_cells_done"], "int"); cmd("gridTotal", St["grid_cells_total"], "int")

# ---- the label construction, read from data/labels_ht29.json (build_labels_ht29.py) -----------
LBJ = os.path.join(HERE, "data", "labels_ht29.json")
if os.path.exists(LBJ):
    LB_ = json.load(open(LBJ)); mm = [v["n_matched"] for v in LB_["models"].values()]
    cmd("lblModels", len(LB_["models"]), "int"); cmd("lblMatchMin", min(mm), "int"); cmd("lblMatchMax", max(mm), "int")
    cmd("lblUnionActive", LB_["union_active"], "int"); cmd("lblHtActive", LB_["ht29_active"], "int")
    cmd("lblUnionNotHt", LB_["union_active_not_ht29"], "int")
    cmd("lblHtPrev", LB_["ht29_prevalence"], "pct"); cmd("lblUnionPrev", LB_["union_prevalence"], "pct")
    cmd("lblHtMatched", LB_["models"]["HT29"]["n_matched"], "int")
    # the label-provenance table rows: model, tissue of origin in the NCI-60 panel, reactions, matched
    _tissue = {"786-O": "renal", "HOP-62": "lung (non-small cell)", "HOP-92": "lung (non-small cell)",
               "HS-578T": "breast", "HT29": "colon", "MALME-3M": "melanoma", "MDA-MB-231/ATCC": "breast",
               "NCI-H226": "lung (non-small cell)", "RPMI-8226": "leukemia panel (myeloma)",
               "SR": "leukemia panel (lymphoma)", "UO-31": "renal"}
    _rows = sorted(LB_["models"].values(), key=lambda v: v["model_id"].upper())
    L.append("\\newcommand{\\lblRows}{" + " ".join(
        f"{v['model_id'].replace('/ATCC', '')} & {_tissue.get(v['model_id'], '')} & {v['n_reactions']:,} & {v['n_matched']:,} \\\\"
        for v in _rows) + "}")

# ---- robustness checks -------------------------------------------------------------------------
# the phenotype probe rerun on the matched-rule outputs: the same question asked of the protocol the
# paper treats as primary, reported beside the grid's-rule figure and never pooled with it
PM = R.get("phenotype_matched")
if PM:
    _NM = {"real": "Real", "mean": "Mean", "zero": "Zero", "indicator": "Ind", "permute": "Perm"}
    for k, nm in _NM.items():
        v = PM["arms"].get(k)
        if not v: continue
        cmd("pheIvr" + nm, v["auroc"], "{:.3f}"); cmd("pheIvr" + nm + "SD", v["sd"], "{:.3f}")
        cmd("pheIvr" + nm + "Lo", v["lo"], "{:.3f}"); cmd("pheIvr" + nm + "Hi", v["hi"], "{:.3f}")
        cmd("pheIvr" + nm + "N", v["n_rfolds"], "int")
    if PM.get("input"): cmd("pheIvrInput", PM["input"]["auroc"], "{:.3f}")
    if PM.get("genes"): cmd("pheIvrGenes", PM["genes"]["auroc"], "{:.3f}")
    cmd("pheIvrNPat", PM["arms"]["real"]["n_patients"], "int"); cmd("pheIvrNPos", PM["arms"]["real"]["n_pos"], "int")
    # how far the wrong-patient arm falls from the unsubstituted one under this protocol
    if PM["arms"].get("permute"):
        _d = PM["arms"]["real"]["auroc"] - PM["arms"]["permute"]["auroc"]
        cmd("pheIvrRealMinusPerm", _d, "{:+.3f}"); cmd("pheIvrRealMinusPermAbs", abs(_d))
        cmd("pheIvrRealMinusPermWord", "below" if _d > 0 else "above")

# the external audit of an independently published predictor (P1-E2): DeepMeta on DepMap gene effect
EA = R.get("external_audit")
if EA:
    N = EA["native"]; C = EA["counts"]
    cmd("dmModel", EA["model"])
    cmd("dmNativeCells", N["cells"], "int"); cmd("dmNativeAuthorsCells", N["authors_cells"], "int")
    cmd("dmNativePairs", N["pairs"], "int"); cmd("dmNativeNodes", N["nodes"], "int")
    cmd("dmNativeAuroc", N["auroc"]); cmd("dmNativeAuthorsAuroc", N["authors_auroc"])
    cmd("dmNativeFone", N["f1"]); cmd("dmNativeAuthorsFone", N["authors_f1"])
    cmd("dmNativeR", N["pearson"], "{:.5f}"); cmd("dmNativeAgreePct", 100.0 * N["agreement"], "{:.3f}")
    cmd("dmNativeMaxDiff", N["mean_abs_diff"], "{:.5f}")
    for k, nm in (("heldout", "Heldout"), ("heldout_test", "HeldoutTest"), ("heldout_unseen", "HeldoutUnseen"),
                  ("development", "Dev"), ("patients", "Patients"), ("lineages", "Lineages"), ("panel", "Panel"),
                  ("complete_case", "CompleteCase"), ("eligible", "Eligible"), ("patient_excluded", "PatientExcluded"),
                  ("patient_excluded_unseen", "PatientExcludedUnseen"), ("candidate_lines", "CandidateLines")):
        cmd("dm" + nm, C[k], "int")
    cmd("dmTemplatePct", 100.0 * C["template_domain_fraction"], "{:.0f}")
    cmd("dmDepThresh", C["dependency_threshold"], "{:+.1f}"); cmd("dmDepRatePct", 100.0 * C["dependency_rate"], "{:.0f}")
    _AR = {"own": "Own", "mean": "Mean", "within": "Within", "cross": "Cross", "within_expr": "ExprSwap",
           "within_graph": "GraphSwap", "baseline_gene_mean": "GeneMean", "baseline_lineage_mean": "LineageMean",
           "baseline_ridge_expression": "Ridge", "baseline_random_forest": "Forest"}
    for k, nm in _AR.items():
        if k in EA["concordance"]: cmd("dmC" + nm, EA["concordance"][k])
        ci = EA["concordance_ci_lines"].get(k)
        if ci: cmd("dmC" + nm + "Lo", ci[0]); cmd("dmC" + nm + "Hi", ci[1])
        cif = EA["concordance_ci_families"].get(k)
        if cif: cmd("dmC" + nm + "FamLo", cif[0]); cmd("dmC" + nm + "FamHi", cif[1])
        if k in EA["per_line_auroc"]: cmd("dmLine" + nm, EA["per_line_auroc"][k])
        if k in EA["spearman"] and EA["spearman"][k].get("mean") is not None:
            cmd("dmRho" + nm, EA["spearman"][k]["mean"], "{:+.4f}")
            if EA["spearman"][k].get("frac_positive") is not None:
                cmd("dmRho" + nm + "PosPct", 100.0 * EA["spearman"][k]["frac_positive"], "{:.0f}")
        if k in EA["constant_genes"]: cmd("dmConst" + nm, EA["constant_genes"][k], "int")
    for k, nm in (("mean", "Mean"), ("within", "Within"), ("cross", "Cross"), ("within_expr", "ExprSwap"), ("within_graph", "GraphSwap")):
        v = EA["verdicts"].get(k)
        if not v: continue
        cmd("dmDe" + nm, v["donor_effect"], "{:+.4f}"); cmd("dmDe" + nm + "Abs", abs(v["donor_effect"]))
        cmd("dmDe" + nm + "Word", "costs" if v["donor_effect"] > 0 else "gains")
        cmd("dmDe" + nm + "WordPast", "cost" if v["donor_effect"] > 0 else "gained")
        cmd("dmDe" + nm + "Lo", v["ci_lines"][0], "{:+.4f}"); cmd("dmDe" + nm + "Hi", v["ci_lines"][1], "{:+.4f}")
        cmd("dmDe" + nm + "FamLo", v["ci_families"][0], "{:+.4f}"); cmd("dmDe" + nm + "FamHi", v["ci_families"][1], "{:+.4f}")
        cmd("dmDe" + nm + "Verdict", v["label_lines"]); cmd("dmDe" + nm + "VerdictFam", v["label_families"])
        # whether the interval excludes zero is a separate question from whether it excludes the
        # smallest effect of interest, and an arm can be inconclusive about the second while settling
        # the first. Both readings are emitted so the text never has to conflate them.
        cmd("dmDe" + nm + "ExclZero", "yes" if (v["ci_lines"][0] > 0 or v["ci_lines"][1] < 0) else "no")
    for k, nm in (("baseline_gene_mean", "GeneMean"), ("baseline_lineage_mean", "LineageMean"),
                  ("baseline_ridge_expression", "Ridge"), ("baseline_random_forest", "Forest")):
        if k in EA["donor_effect_vs_baselines"]:
            _v = EA["donor_effect_vs_baselines"][k]
            cmd("dmOwnVs" + nm, _v, "{:+.4f}"); cmd("dmOwnVs" + nm + "Abs", abs(_v))
            cmd("dmOwnVs" + nm + "Dir", "above" if _v > 0 else "below")
    # whether the within-gene metric resolves differences on these data at all: the strongest
    # sample-blind learner's family-resampled interval against the model's. A reviewer who sees
    # values near one half should be able to read off whether that is the metric or the model.
    _co, _cr = EA["concordance"].get("own"), EA["concordance"].get("baseline_ridge_expression")
    _io = EA["concordance_ci_families"].get("own"); _ir = EA["concordance_ci_families"].get("baseline_ridge_expression")
    if _co is not None and _cr is not None and _io and _ir:
        _sep = (_ir[0] > _io[1]) or (_io[0] > _ir[1])
        cmd("dmSensGapAbs", abs(_cr - _co))
        cmd("dmSensPhrase",
            ("the expression-only ridge's interval lies clear of the model's, so a difference of "
             f"{abs(_cr - _co):.4f} on this metric is resolvable on these data"
             if _sep else
             "the two intervals overlap, so this comparison does not by itself establish the metric's resolution"))
    # a generated reading of the verdicts: how many arms clear the smallest effect of interest
    _dl = [v for v in EA["verdicts"].values()]
    _none = sum(1 for v in _dl if v["label_lines"].startswith("no sample-specific"))
    _inc = sum(1 for v in _dl if v["label_lines"] == "inconclusive")
    _adv = sum(1 for v in _dl if v["label_lines"] == "advantage")
    cmd("dmDeMeanAbsThresh", 0.01, "{:.2f}")
    cmd("dmVerdictPhrase", (f"none of the {len(_dl)} substitutions produces an advantage the interval places above "
                            f"{0.01:.2f}: {_none} are read as no advantage larger than that and {_inc} as inconclusive")
        if _adv == 0 else f"{_adv} of the {len(_dl)} substitutions clears the smallest effect of interest")
    # the per-cell view against the sample-blind per-gene mean, the artifact the audit is built to catch
    _pl = EA["per_line_auroc"]
    if "own" in _pl and "baseline_gene_mean" in _pl:
        _d = _pl["own"] - _pl["baseline_gene_mean"]
        cmd("dmLineOwnVsGeneMean", _d, "{:+.4f}"); cmd("dmLineOwnVsGeneMeanAbs", abs(_d))
        cmd("dmLineOwnVsGeneMeanDir", "above" if _d > 0 else "below")
    _sec = EA["secondary"]
    # the smallest-effect verdicts domain by domain. The three domains do not give the same split of
    # below-threshold and inconclusive readings, and the text must report the split each one gives
    # rather than assert that they agree.
    _DOMS = (("primary", EA["verdicts"]), ("Sec", _sec.get("explicit_only_verdicts")), ("Ter", _sec.get("zero_fill_verdicts")))
    _AN = (("mean", "Mean"), ("within", "Within"), ("cross", "Cross"), ("within_expr", "ExprSwap"), ("within_graph", "GraphSwap"))
    _dom_counts = {}
    for _dnm, _dv in _DOMS:
        if not _dv: continue
        _b = [k for k, v in _dv.items() if v["label_lines"].startswith("no sample-specific")]
        _i = [k for k, v in _dv.items() if v["label_lines"] == "inconclusive"]
        _dom_counts[_dnm] = (len(_b), len(_i))
        if _dnm != "primary":
            for k, nm in _AN:
                v = _dv.get(k)
                if not v: continue
                cmd("dm" + _dnm + "De" + nm, v["donor_effect"], "{:+.4f}")
                cmd("dm" + _dnm + "De" + nm + "Lo", v["ci_lines"][0], "{:+.4f}"); cmd("dm" + _dnm + "De" + nm + "Hi", v["ci_lines"][1], "{:+.4f}")
                cmd("dm" + _dnm + "De" + nm + "Read", "below" if v["label_lines"].startswith("no sample-specific") else "spans")
            cmd("dm" + _dnm + "BelowN", len(_b), "int"); cmd("dm" + _dnm + "InconN", len(_i), "int")
    for k, nm in _AN:
        v = EA["verdicts"].get(k)
        if v: cmd("dmDe" + nm + "Read", "below" if v["label_lines"].startswith("no sample-specific") else "spans")
    if len(_dom_counts) == 3:
        _same = len({v for v in _dom_counts.values()}) == 1
        cmd("dmDomainsAgree", "yes" if _same else "no")
        cmd("dmDomainSplitPhrase",
            ("the three domains give the same split" if _same else
             f"the split is not the same in every domain: {_word(_dom_counts['primary'][0])} below the threshold and "
             f"{_word(_dom_counts['primary'][1])} inconclusive on the primary domain, "
             f"{_word(_dom_counts['Sec'][0])} and {_word(_dom_counts['Sec'][1])} with the pairs explicit in every arm, "
             f"{_word(_dom_counts['Ter'][0])} and {_word(_dom_counts['Ter'][1])} on the whole grid with absent nodes scored as not dependent"))
        # what does hold in every domain: no lower endpoint above the threshold
        _any_adv = any(v["label_lines"] == "advantage" for _, dv in _DOMS if dv for v in dv.values())
        _min_lo = min(v["ci_lines"][0] for _, dv in _DOMS if dv for v in dv.values())
        _max_lo = max(v["ci_lines"][0] for _, dv in _DOMS if dv for v in dv.values())
        cmd("dmDomainsMaxLower", _max_lo, "{:+.4f}")
        cmd("dmDomainsAnyAdvantage", "yes" if _any_adv else "no")
    # the extremes over every domain, both resampling blocks and every schedule present, so that the
    # sentence about the smallest effect of interest says which quantity it is about
    _rx = EA.get("range_extremes")
    if _rx:
        _nice = {"mean": "the lineage-conditioned mean", "within": "the same-lineage donor",
                 "cross": "the cross-lineage donor", "within_expr": "the expression branch only",
                 "within_graph": "the sample-derived graph only"}
        _DN = {"primary_template_domain": "the template domain", "secondary_explicit_only": "the explicit-pairs domain",
               "tertiary_zero_fill_all": "the whole grid"}
        cmd("dmExtMaxPoint", _rx["max_point"]["value"], "{:+.4f}")
        cmd("dmExtMaxPointWhere", f"{_nice.get(_rx['max_point']['arm'], _rx['max_point']['arm'])} on {_DN[_rx['max_point']['domain']]} under schedule {_rx['max_point']['seed']}")
        cmd("dmExtMaxLowerLines", _rx["max_lower_lines"]["value"], "{:+.4f}")
        cmd("dmExtMaxLowerFam", _rx["max_lower_families"]["value"], "{:+.4f}")
        cmd("dmExtNSeeds", _rx["n_seeds"], "int")
        cmd("dmExtNPointsAbove", _rx["n_point_estimates_above_threshold"], "int")
        cmd("dmExtAnyLowerAbove", "yes" if _rx["any_lower_endpoint_above_threshold"] else "no")
    cmd("dmSecOwn", _sec["explicit_only"]["own"]); cmd("dmSecRidge", _sec["explicit_only"]["baseline_ridge_expression"])
    cmd("dmTerOwn", _sec["zero_fill"]["own"]); cmd("dmTerLineOwn", _sec["zero_fill_per_line"]["own"])
    cmd("dmSecLineOwn", _sec["explicit_only_per_line"]["own"])
    cmd("dmRidgeAlpha", EA["baselines_info"]["ridge_alpha"], "{:.0f}")
    cmd("dmRidgeFeatures", EA["baselines_info"]["ridge_n_features"], "int")
    cmd("dmForestTrees", EA["baselines_info"]["rf_trees"], "int")
    cmd("dmForestGenes", EA["baselines_info"]["rf_features"], "int")
    cmd("dmLineageMeanFallbackCells", EA["baselines_info"]["lineage_mean_fallback_cells"], "int")
    # the observed spread of the within-gene metric across every predictor tried, which is what a
    # reader should be given instead of an argument from variance components about where the metric
    # can sit. It is the range of the column, not a bound.
    _cr = EA.get("concordance_range")
    if _cr:
        cmd("dmCRangeLo", _cr["lo"]); cmd("dmCRangeHi", _cr["hi"])
        cmd("dmCRangeSpan", _cr["hi"] - _cr["lo"]); cmd("dmCRangeN", _cr["n_predictors"], "int")
    # the donor schedules' design facts. The mean arm is a lineage-conditioned training mean, not one
    # profile shared by everybody, and a few recipients fall back; the within-lineage arm cannot
    # derange a lineage that holds a single held-out line.
    _sc = EA.get("schedule")
    if _sc:
        cmd("dmSchedSeed", _sc["primary_seed"], "int")
        cmd("dmSchedSeeds", ", ".join(str(x) for x in _sc["seeds"]))
        cmd("dmSchedNSeeds", len(_sc["seeds"]), "int")
        cmd("dmMeanFallback", _sc["mean_fallback_cells"], "int")
        cmd("dmWithinCrossFallback", _sc["within_cross_fallback_cells"], "int")
        cmd("dmWithinSameLineagePairs", _sc["within_same_lineage_pairs"], "int")
        cmd("dmSharedDonorLines", _sc["heldout_lines_sharing_a_donor"], "int")
        cmd("dmDonorDistinct", "yes" if _sc["donor_distinct_constraint"] else "no")
        _sd = _sc["by_seed"]
        cmd("dmSchedStable", "yes" if len({(v["mean_fallback_cells"], v["within_cross_fallback_cells"])
                                           for v in _sd.values()}) == 1 else "no")
    # the donor-schedule sensitivity, emitted only for the schedules whose analysis actually ran
    _ss = EA.get("schedule_sensitivity")
    if _ss:
        cmd("dmSchedRunN", len(_ss["seeds"]), "int")
        cmd("dmSchedRunList", ", ".join(str(x) for x in _ss["seeds"]))
        for k, nm in (("mean", "Mean"), ("within", "Within"), ("cross", "Cross"),
                      ("within_expr", "ExprSwap"), ("within_graph", "GraphSwap")):
            sp = _ss["spread"].get(k)
            if sp:
                cmd("dmDe" + nm + "SeedLo", sp["lo"], "{:+.4f}"); cmd("dmDe" + nm + "SeedHi", sp["hi"], "{:+.4f}")
                cmd("dmDe" + nm + "SeedSpan", sp["hi"] - sp["lo"])
        _w = _ss["spread"].get("within"); _c = _ss["spread"].get("cross")
        if _w and _c:
            cmd("dmSchedSensPhrase",
                (f"across the {_word(len(_ss['seeds']))} schedules the same-lineage donor effect runs from "
                 f"{_w['lo']:+.4f} to {_w['hi']:+.4f} and the cross-lineage one from {_c['lo']:+.4f} to "
                 f"{_c['hi']:+.4f}"))
    # the two readings of the smallest-effect comparison, kept apart: an interval that excludes
    # effects larger than the threshold is not the same finding as one that spans it.
    _dl = EA["verdicts"]
    _below = sorted(k for k, v in _dl.items() if v["label_lines"].startswith("no sample-specific"))
    _incon = sorted(k for k, v in _dl.items() if v["label_lines"] == "inconclusive")
    _nice = {"mean": "lineage-conditioned mean", "within": "same-lineage donor",
             "cross": "cross-lineage donor", "within_expr": "expression branch only",
             "within_graph": "sample-derived graph only"}
    cmd("dmBelowN", len(_below), "int")
    cmd("dmInconN", len(_incon), "int")
    cmd("dmBelowList", ", ".join(_nice.get(k, k) for k in _below) or "none")
    cmd("dmInconList", ", ".join(_nice.get(k, k) for k in _incon) or "none")
    _excl = sorted(k for k, v in _dl.items() if v["ci_lines"][0] > 0 or v["ci_lines"][1] < 0)
    cmd("dmExclZeroN", len(_excl), "int")
    cmd("dmExclZeroList", ", ".join(_nice.get(k, k) for k in _excl) or "none")
    cmd("dmExclZeroPhrase",
        ("no substitution's interval excludes zero" if not _excl else
         ("one substitution's interval excludes zero, the " + _nice.get(_excl[0], _excl[0])
          if len(_excl) == 1 else
          f"{_word(len(_excl))} substitutions have intervals that exclude zero: "
          + ", ".join(_nice.get(k, k) for k in _excl))))
    cmd("dmVerdictSplitPhrase",
        (f"{_word(len(_below))} of the {_word(len(_dl))} substitutions ({', '.join(_nice.get(k, k) for k in _below)}) "
         f"have intervals that exclude an advantage larger than 0.01, and {_word(len(_incon))} "
         f"({', '.join(_nice.get(k, k) for k in _incon)}) are inconclusive, with intervals that span it")
        if _below and _incon else
        (f"all {_word(len(_dl))} substitutions have intervals that exclude an advantage larger than 0.01"
         if not _incon else
         f"all {_word(len(_dl))} substitutions are inconclusive, with intervals that span 0.01"))

# the diagnostic calibration simulation (P1-E3): input-substitution arms with the predictor refitted
# on the substituted inputs, the matched selection rule for the learned model, exact sign-flip tests,
# and the decision rule fixed on the development worlds and read on the disjoint test worlds
DS = R.get("diagnostic_sim")
def _bound(k, n, upper):
    """exact one-sided 95% Clopper-Pearson bound on a proportion k/n (the beta quantile), so that
    0 of n gives 1 - 0.05^(1/n) rather than the rule-of-three approximation"""
    from scipy.stats import beta as _beta
    if n == 0: return None
    if upper:
        return 1.0 if k >= n else float(_beta.ppf(0.95, k + 1, n - k))
    return 0.0 if k <= 0 else float(_beta.ppf(0.05, k, n - k + 1))
if DS:
    pr = DS["params"]
    cmd("dsimNDev", pr["n_dev"], "int"); cmd("dsimNTest", pr["n_test"], "int")
    cmd("dsimP", pr["P"], "int"); cmd("dsimR", pr["R"], "int")
    cmd("dsimPFolds", pr["pfolds"], "int"); cmd("dsimRFolds", pr["rfolds"], "int")
    cmd("dsimCells", pr["pfolds"] * pr["rfolds"], "int")
    if "inner_frac" in pr: cmd("dsimInnerFrac", 100.0 * pr["inner_frac"], "{:.0f}")
    if "rounds" in pr:
        _r = [str(x) for x in pr["rounds"]]
        cmd("dsimRounds", (", ".join(_r[:-1]) + " and " + _r[-1]) if len(_r) > 1 else _r[0])
    if "dev_seeds" in pr: cmd("dsimDevSeedLo", pr["dev_seeds"][0], "int"); cmd("dsimDevSeedHi", pr["dev_seeds"][1], "int")
    if "test_seeds" in pr: cmd("dsimTestSeedLo", pr["test_seeds"][0], "int"); cmd("dsimTestSeedHi", pr["test_seeds"][1], "int")
    # error control of the fixed rule on the test worlds, for the learned model and the references
    for kind, pre in (("learned", "dsim"), ("oracle", "dsimOracle"), ("label_irrelevant", "dsimLirr"), ("blind", "dsimBlind")):
        tc = DS["test_confusion"].get(kind)
        if not tc: continue
        cmd(pre + "FP", tc["fp"], "int"); cmd(pre + "TP", tc["tp"], "int")
        cmd(pre + "NNull", tc["n_null"], "int"); cmd(pre + "NPos", tc["n_positive"], "int")
        cmd(pre + "FPR", 100.0 * tc["false_positive_rate"], "{:.1f}")
        _ub = _bound(tc["fp"], tc["n_null"], True)
        if _ub is not None: cmd(pre + "FPRUpper", 100.0 * _ub, "{:.1f}")
        if tc.get("power") is not None:
            cmd(pre + "Power", 100.0 * tc["power"], "{:.0f}")
            _lb = _bound(tc["tp"], tc["n_positive"], False)
            if _lb is not None: cmd(pre + "PowerLower", 100.0 * _lb, "{:.1f}")
    # the same rule on the development worlds, reported separately and never pooled with the test set
    dc = DS["dev_confusion"]["learned"]
    cmd("dsimDevFP", dc["fp"], "int"); cmd("dsimDevNNull", dc["n_null"], "int")
    cmd("dsimDevTP", dc["tp"], "int"); cmd("dsimDevNPos", dc["n_positive"], "int")
    if dc.get("power") is not None: cmd("dsimDevPower", 100.0 * dc["power"], "{:.0f}")
    # the two kinds of null world, separately: the donor contrast is structurally zero under a shared
    # label, so the informative null for it is the varying label with no signal
    ns = (DS.get("test_null_subsets") or {}).get("learned") or {}
    if "shared_label" in ns:
        q = ns["shared_label"]
        cmd("dsimSharedN", q["n"], "int"); cmd("dsimSharedFP", q["false_positives"], "int")
        cmd("dsimSharedDonorZeroWorlds", q["worlds_with_all_donor_contrasts_zero"], "int")
        # a floating-point residue is printed as such rather than rounded to a zero it is not
        _m = q["max_abs_donor_contrast"]
        cmd("dsimSharedMaxAbsDonor", ("$%s\\times10^{%d}$" % (("%.0e" % _m).split("e")[0], int(("%.0e" % _m).split("e")[1])))
            if 0 < _m < 1e-6 else "%.4f" % _m)
        cmd("dsimSharedMeanSig", q["mean_contrast_positive_sig"], "int")
    if "varying_label_no_signal" in ns:
        q = ns["varying_label_no_signal"]
        cmd("dsimVaryN", q["n"], "int"); cmd("dsimVaryFP", q["false_positives"], "int")
        cmd("dsimVaryMaxAbsDonor", q["max_abs_donor_contrast"], "{:.4f}")
        cmd("dsimVaryMeanSig", q["mean_contrast_positive_sig"], "int")
        _ub = _bound(q["false_positives"], q["n"], True)
        if _ub is not None: cmd("dsimVaryFPRUpper", 100.0 * _ub, "{:.1f}")
    for kind, pre in (("oracle", "dsimOracle"), ("label_irrelevant", "dsimLirr")):
        q = (DS.get("test_null_subsets") or {}).get(kind) or {}
        if "varying_label_no_signal" in q: cmd(pre + "VaryFP", q["varying_label_no_signal"]["false_positives"], "int")
        if "shared_label" in q: cmd(pre + "SharedFP", q["shared_label"]["false_positives"], "int")
    # world counts per configuration in each set
    for tag, pre in (("test", "dsimTest"), ("dev", "dsimDev")):
        wc = (DS.get("world_counts") or {}).get(tag)
        if wc:
            cmd(pre + "Shared", wc["shared"], "int"); cmd(pre + "VaryZero", wc["varying_alpha0"], "int")
            cmd(pre + "VaryWeak", wc["varying_weak"], "int"); cmd(pre + "VaryStrong", wc["varying_strong"], "int")
    # the boosting rounds the matched rule chose, pooled over the test worlds
    rc = DS.get("rounds_chosen") or {}
    if rc.get("own"):
        _all = sorted({r for v in rc.values() for r in v})
        cmd("dsimRoundsChosen", (", ".join(str(x) for x in _all[:-1]) + " and " + str(_all[-1])) if len(_all) > 1 else str(_all[0]))
    ce = DS["counterexample"]["constructed"]
    cmd("dsimCeOwn", ce["own_auroc"]); cmd("dsimCeMean", ce["cohort_mean_auroc"]); cmd("dsimCeDelta", ce["own_minus_mean"], "{:+.2f}")
    cmd("dsimCePairs", ce["n_pairs"], "int")
    if "generic_restricted_nonlinear_own_minus_mean_mean_test" in DS["counterexample"]:
        cmd("dsimGenericRestrOwnMean", DS["counterexample"]["generic_restricted_nonlinear_own_minus_mean_mean_test"], "{:+.4f}")
        cmd("dsimGenericBlindOwnMean", DS["counterexample"]["generic_blind_own_minus_mean_mean_test"], "{:+.4f}")
    # the decision table on the test worlds: one row per regime, every predictor
    _REG = {"shared label": "Shared", "alpha=0 (threshold": "VaryZero", "alpha=0.5": "Weak", "alpha>=1.5": "Strong"}
    _KN = {"learned": "", "oracle": "Oracle", "label_irrelevant": "Lirr", "blind": "Blind", "restricted_nonlinear": "Restr"}
    rows = []
    for row in DS["decision_table"]:
        reg = next((v for k, v in _REG.items() if k in row["regime"]), None)
        if not reg: continue
        cmd(f"dsim{reg}Worlds", row["n_worlds"], "int")
        for kind, kp in _KN.items():
            if kind not in row: continue
            cmd(f"dsim{reg}{kp}OwnMean", row[kind]["own_minus_mean"], "{:+.4f}")
            cmd(f"dsim{reg}{kp}OwnDonor", row[kind]["own_minus_donor"], "{:+.4f}")
            cmd(f"dsim{reg}{kp}Verdict", 100.0 * row[kind]["verdict_rate"], "{:.0f}")
        _lab = {"Shared": "shared label, no signal", "VaryZero": "varying label, no signal ($\\alpha=0$)",
                "Weak": "varying label, weak signal ($\\alpha=0.5$)",
                "Strong": "varying label, strong signal ($\\alpha\\geq 1.5$)"}[reg]
        rows.append(f"{_lab} & {row['n_worlds']} & ${row['learned']['own_minus_mean']:+.4f}$ & ${row['learned']['own_minus_donor']:+.4f}$ & "
                    f"{100*row['learned']['verdict_rate']:.0f} & {100*row['oracle']['verdict_rate']:.0f} & "
                    f"{100*row['label_irrelevant']['verdict_rate']:.0f} & ${row['blind']['own_minus_mean']:+.4f}$ \\\\")
    if rows: cmd("dsimRows", "\n".join(rows))
    # and on the development worlds, kept apart
    for row in DS.get("dev_decision_table") or []:
        reg = next((v for k, v in _REG.items() if k in row["regime"]), None)
        if reg: cmd(f"dsimDev{reg}Verdict", 100.0 * row["learned"]["verdict_rate"], "{:.0f}"); cmd(f"dsimDev{reg}Worlds", row["n_worlds"], "int")

ROB = R.get("robustness", {})
def arm_macros(pre, blk):
    if not blk: return
    cmd(pre, blk["mean"]); cmd(pre + "N", blk["n"], "int")
    if blk.get("sd") is not None: cmd(pre + "SD", blk["sd"])
    # the cell files' trainrxn field is scored on ALL outer-training reactions, which under the
    # matched rule includes the inner-validation reactions the loss never saw. It is the training
    # pool's score, not the fitted mask's, and only mask_scores separates the two.
    cmd(pre + "Train", blk["trainrxn"]); cmd(pre + "TrainPool", blk["trainrxn"])
    cmd(pre + "Expr", blk["exprbearing"])
    cmd(pre + "Raw", blk["rawexpr"]); cmd(pre + "Ind", blk["indicator"]); cmd(pre + "Rho", blk["rho"], "{:.2f}")
    # the memorization diagnostic of the arm: the absolute train-minus-held-out gap, per cell
    if blk.get("gap_abs") is not None:
        cmd(pre + "GapAbs", blk["gap_abs"]); cmd(pre + "GapNeg", blk["gap_negative_cells"], "int")
        if blk.get("gap_abs_sd") is not None: cmd(pre + "GapAbsSD", blk["gap_abs_sd"])
def pair_macros(pre, pc):
    if not pc or not pc["n"]: return
    cmd(pre, pc["delta"], "{:+.4f}"); cmd(pre + "Abs", abs(pc["delta"])); cmd(pre + "N", pc["n"], "int")
    if pc.get("sd") is not None: cmd(pre + "SD", pc["sd"])
    cmd(pre + "Pos", pc["n_positive"], "int"); cmd(pre + "Neg", pc["n_negative"], "int")
    cmd(pre + "Cells", ", ".join(f"{v:+.4f}" for v in pc["per_cell"].values()))
    cmd(pre + "Min", min(pc["per_cell"].values()), "{:+.4f}"); cmd(pre + "Max", max(pc["per_cell"].values()), "{:+.4f}")
    # a word carrying the sign, so that prose built on an absolute value cannot assert a direction
    # the run did not produce (the check script refuses a guarded sentence that does)
    cmd(pre + "Word", "benefit" if pc["delta"] > 0 else "cost")
    cmd(pre + "Dir", "above" if pc["delta"] > 0 else "below")
    cmd(pre + "SignCount", "%d of %d" % (max(pc["n_positive"], pc["n_negative"]), pc["n"]))
    cmd(pre + "SignWord", "positive" if pc["n_positive"] > pc["n_negative"] else "negative")
    # a sign summary that does not hide a tie behind a single direction word
    cmd(pre + "SignPhrase",
        ("positive in %d of %d cells and negative in the other %d" % (pc["n_positive"], pc["n"], pc["n_negative"]))
        if pc["n_positive"] == pc["n_negative"] else
        ("%s in %d of %d cells" % ("positive" if pc["n_positive"] > pc["n_negative"] else "negative",
                                   max(pc["n_positive"], pc["n_negative"]), pc["n"])))
LINKEYS = {"indicator": "Ind", "expr_perpat_rank": "ExprPer", "expr_cohortmean_rank": "ExprCoh",
           "expr_cohortmean_lr": "ExprCohLr", "expr_perpat_lr": "ExprPerLr", "topology_only": "Topo",
           "annotation_only": "Annot", "structure_only": "Struct", "structure_plus_cohortmean": "StructCoh",
           "structure_plus_perpat": "StructPer"}
DELKEYS = {"lin_patient_expr_only": "PatExpr", "lin_patient_struct_expr": "PatStruct", "nonparam_patient_effect": "Nonpar",
           "structure_over_expression_fitted": "StructOverExpr", "topology_over_expression_fitted": "TopoOverExpr",
           "annotation_over_expression_fitted": "AnnotOverExpr"}
def cells_phrase(pre, basis):
    """'the three fold-0 cells', 'the one fold-0 cell (p0r0)', or 'the N cells (...)': a phrase that
    agrees in number with the basis, so no sentence prints 'the 1 cells'."""
    basis = list(basis or []); n = len(basis)
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 9: "nine", 12: "twelve", 15: "fifteen"}
    w = words.get(n, str(n))
    if n == 0: return
    if all(b.startswith("p0") for b in basis):
        ph = f"the {w} fold-0 cell{'s' if n > 1 else ''}" + (f" ({basis[0]})" if n == 1 else (" (patient fold 0, every reaction fold)" if n == 3 else f" ({', '.join(basis)})"))
    else:
        ph = f"the {w} cell{'s' if n > 1 else ''} ({', '.join(basis)})" if n < 15 else f"the {w} cells of the full grid"
    cmd(pre + "CellsPhrase", ph)
    # the same phrase without the parenthetical gloss, for uses after the first
    cmd(pre + "CellsShort", ph.split(" (")[0])

def lin_macros(pre, LR):
    if not LR: return
    for k, nm in LINKEYS.items():
        if k in LR: cmd(pre + nm, LR[k]["mean"]); cmd(pre + nm + "N", LR[k]["n"], "int")
    for k, nm in DELKEYS.items():
        d = LR["_deltas"].get(k)
        if d:
            cmd(pre + nm, d["delta"], "{:+.4f}"); cmd(pre + nm + "Abs", abs(d["delta"])); cmd(pre + nm + "SD", d["sd"])
            cmd(pre + nm + "Pos", d["n_positive"], "int"); cmd(pre + nm + "Neg", d["n_negative"], "int")
            cmd(pre + nm + "N", d["n"], "int")
            cmd(pre + nm + "Word", "benefit" if d["delta"] > 0 else "cost")
            if d.get("n_rfolds"):
                cmd(pre + nm + "NRxn", d["n_rfolds"], "int")
                cmd(pre + nm + "PosRxn", d["n_rfolds_positive"], "int")
                cmd(pre + nm + "NegRxn", d["n_rfolds_negative"], "int")
H = ROB.get("ht29")
if H:
    cmd("robHtCells", ", ".join(H["basis"])); cmd("robHtN", len(H["basis"]), "int"); cells_phrase("robHt", H["basis"])
    for k, nm in (("real", "Real"), ("cohort_mean", "Mean"), ("zero", "Zero")): arm_macros("robHtGnn" + nm, H.get(k))
    pair_macros("robHtGnnPat", H.get("patient_identity")); pair_macros("robHtGnnAgg", H.get("aggregate_expression"))
    pair_macros("robHtGnnExprTot", H.get("expression_total"))
    U = H.get("union_same_cells", {})
    for k, nm in (("real", "Real"), ("cohort_mean", "Mean"), ("zero", "Zero")): arm_macros("robHtUnion" + nm, U.get(k))
    pair_macros("robHtUnionPat", U.get("patient_identity")); pair_macros("robHtUnionAgg", U.get("aggregate_expression"))
    lin_macros("robHtLin", H.get("linear_all_cells")); lin_macros("robHtLinSame", H.get("linear_same_cells"))
    for k in ("gnn_zero_over_structure", "gnn_mean_over_structure", "gnn_real_over_structure",
              "gnn_mean_over_lr_with_expression", "gnn_real_over_lr_own_expression"):
        if k in H:
            nm = "".join(w.capitalize() for w in k.split("_"))
            cmd("robHt" + nm, H[k], "{:+.4f}"); cmd("robHt" + nm + "Abs", abs(H[k]))
def pooled_check_macros(pre, PB):
    if not PB: return
    for k, nm in (("pooled_vs_cohort_expr", "PatExpr"), ("pooled_vs_cohort_struct", "PatStruct"),
                  ("pooled_vs_perpat_expr", "VsPerExpr"), ("pooled_vs_perpat_struct", "VsPerStruct")):
        d = (PB.get("deltas") or {}).get(k)
        if d: pair_macros(pre + nm, d)
    for k, nm in (("expr_pooled_lr", "Expr"), ("structure_plus_pooled", "Struct")):
        r = (PB.get("rows") or {}).get(k)
        if r: cmd(pre + nm, r["mean"]); cmd(pre + nm + "N", r["n"], "int")
    if PB.get("max_train_patients"): cmd(pre + "MaxTrainPat", PB["max_train_patients"], "int")
pooled_check_macros("robHtPooled", (ROB.get("ht29") or {}).get("pooled"))
_hp = (((ROB.get("ht29") or {}).get("pooled") or {}).get("deltas") or {}).get("pooled_vs_cohort_struct")
if _hp:
    _v = _hp["delta"]
    cmd("robHtPooledUpperPhrase", ("keeps a benefit of %+.4f under the tissue-matched one" % _v) if _v > 0.001
        else ("falls within a thousandth of zero under the tissue-matched one" if abs(_v) <= 0.001
              else "turns into a cost of %+.4f under the tissue-matched one" % _v))
pooled_check_macros("robRkPooled", (ROB.get("rank") or {}).get("pooled"))
PL = ROB.get("pooled_large")
if PL:
    pooled_check_macros("lbPooledLarge", PL)
    cmd("lbPooledLargeN", (PL.get("rows") or {}).get("structure_plus_pooled", {}).get("n", 0), "int")
    for k, nm in (("expr_pooled_lr", "Expr"), ("structure_plus_pooled", "Struct")):
        d = (PL.get("versus_default") or {}).get(k)
        if d: pair_macros("lbPooledLargeVsDefault" + nm, d)
    if PL.get("gnn_real_vs_pooled_struct"): pair_macros("lbPooledLargeGnnRealOverStruct", PL["gnn_real_vs_pooled_struct"])
    if PL.get("default_max_train_patients"): cmd("lbPooledDefaultMaxTrainPat", PL["default_max_train_patients"], "int")
# the largest linear estimate of the contrast under each upstream check, over the per-patient, the
# unfitted and (once run) the pooled rows, so that "every linear estimate" sentences quantify over
# the rows that were actually rerun
def _lin_check_vals(blk, lin_key):
    vals = []
    L = ((blk or {}).get(lin_key) or {}).get("_deltas") or {}
    for k in ("lin_patient_expr_only", "lin_patient_struct_expr", "nonparam_patient_effect"):
        if L.get(k): vals.append(L[k]["delta"])
    for k in ("pooled_vs_cohort_expr", "pooled_vs_cohort_struct"):
        d = (((blk or {}).get("pooled") or {}).get("deltas") or {}).get(k)
        if d: vals.append(d["delta"])
    return vals
_htv = _lin_check_vals(ROB.get("ht29"), "linear_all_cells"); _rkv = _lin_check_vals(ROB.get("rank"), "linear")
if _htv:
    cmd("robHtLinMaxPat", max(_htv), "{:+.4f}"); cmd("robHtLinN", len(_htv), "int")
    cmd("robHtLinPhrase", "every linear estimate of the contrast at or below zero under the tissue-matched label" if max(_htv) <= 0
        else f"every linear estimate of the contrast at or below {max(_htv):+.4f} under the tissue-matched label")
    # the same statement for a sentence that has already named the label
    cmd("robHtLinPhraseBare", "every linear estimate of the contrast at or below zero" if max(_htv) <= 0
        else f"every linear estimate of the contrast at or below {max(_htv):+.4f}")
    # the sentence that opens the linear-row reading of the label check; it quantifies over every
    # linear row rerun under the label, the pooled rows included once they have been rerun
    cmd("robHtLinTurnSentence", "The individual-versus-cohort-mean terms of the linear rows do not turn positive." if max(_htv) <= 0
        else f"No individual-versus-cohort-mean term of the linear rows rises above {max(_htv):+.4f}.")
    cmd("robHtPooledRerunPhrase", "including the two pooled rows" if len(_htv) >= 5 else "the pooled rows not rerun")
if _htv or _rkv:
    _m = max(_htv + _rkv); cmd("robLinMaxPat", _m, "{:+.4f}")
    cmd("robLinMaxPhrase", "no linear estimate makes individual expression worth more than a few thousandths of AUROC over a cohort average"
        if _m < 0.005 else f"no linear estimate makes individual expression worth more than {_m:+.4f} AUROC over a cohort average")
SRB = ROB.get("seedrep", {})
for ts, B in SRB.items():
    pre = "robSeed"    # one replicate seed is run; a further seed would need its own prefix
    cmd(pre + "Seed", B["train_seed"], "int"); cmd(pre + "MainSeed", B["main_train_seed"], "int")
    cmd(pre + "Cells", ", ".join(B["basis"])); cmd(pre + "N", len(B["basis"]), "int"); cells_phrase(pre, B["basis"])
    arm_macros(pre + "Real", B.get("real")); arm_macros(pre + "Mean", B.get("cohort_mean"))
    pair_macros(pre + "Pat", B.get("patient_identity"))
    Mn = B.get("main_same_cells", {})
    arm_macros(pre + "MainReal", Mn.get("real")); arm_macros(pre + "MainMean", Mn.get("cohort_mean")); pair_macros(pre + "MainPat", Mn.get("patient_identity"))
    pair_macros(pre + "RealVsMain", B.get("real_vs_main")); pair_macros(pre + "MeanVsMain", B.get("cohort_mean_vs_main"))
    d = B.get("patient_identity_seed_minus_main")
    if d:
        cmd(pre + "PatDiff", d["delta"], "{:+.4f}"); cmd(pre + "PatDiffAbs", abs(d["delta"]))
        cmd(pre + "PatDiffCells", ", ".join(f"{v:+.4f}" for v in d["per_cell"].values()))
        cmd(pre + "PatDiffMaxAbs", max(abs(v) for v in d["per_cell"].values()))
    break
RK = ROB.get("rank")
if RK:
    lin_macros("robRkLin", RK.get("linear"))
    # whether the second normalization keeps every linear estimate of the contrast inside the range
    # of the ten estimates under the main normalization
    _rkd = [d["delta"] for k, d in (RK.get("linear") or {}).get("_deltas", {}).items()
            if d and k in ("lin_patient_expr_only", "lin_patient_struct_expr", "nonparam_patient_effect")]   # the contrast estimates only
    _rkd += [d["delta"] for k, d in ((RK.get("pooled") or {}).get("deltas") or {}).items() if d and k in ("pooled_vs_cohort_expr", "pooled_vs_cohort_struct")]
    if _rkd and _est:
        _vals_all = [v for _, v in _est]
        _inside = all(min(_vals_all) <= v <= max(_vals_all) for v in _rkd)
        cmd("rkInsideRangePhrase", "keeps every linear estimate inside that range" if _inside
            else "moves a linear estimate outside that range, to " + ("%+.4f" % (max(_rkd, key=abs))))
        # where one rerun estimate lands relative to the range of the ten, for the sentences that
        # name a particular estimator: inside the range, just outside it, or outside it
        def _range_word(v):
            lo, hi = min(_vals_all), max(_vals_all)
            if lo <= v <= hi: return "inside that range"
            _gap = (v - hi) if v > hi else (lo - v)
            return "just outside it" if _gap <= 0.001 else f"outside it by {_gap:.4f}"
        _rkL = (RK.get("linear") or {}).get("_deltas", {})
        if _rkL.get("lin_patient_expr_only"): cmd("rkExprRangeWord", _range_word(_rkL["lin_patient_expr_only"]["delta"]))
        _rkPs = ((RK.get("pooled") or {}).get("deltas") or {}).get("pooled_vs_cohort_struct")
        if _rkPs: cmd("rkPooledStructRangeWord", _range_word(_rkPs["delta"]))
    # how many of the linear estimates change sign under the second normalization, and which, so
    # that the sentence introducing the check cannot undercount them once the pooled rows are rerun
    _rk_names = {"lin_patient_expr_only": "the expression-only row fitted per patient",
                 "lin_patient_struct_expr": "the network-plus-expression row fitted per patient",
                 "nonparam_patient_effect": "the no-model ranking",
                 "pooled_vs_cohort_expr": "the pooled expression-only row",
                 "pooled_vs_cohort_struct": "the pooled network-plus-expression row"}
    _rk_pairs = []
    for k in ("lin_patient_expr_only", "lin_patient_struct_expr", "nonparam_patient_effect"):
        a = (RK.get("linear") or {}).get("_deltas", {}).get(k); b = RK.get("deltas_main", {}).get(k)
        if a and b: _rk_pairs.append((k, b["delta"], a["delta"]))
    for k in ("pooled_vs_cohort_expr", "pooled_vs_cohort_struct"):
        a = ((RK.get("pooled") or {}).get("deltas") or {}).get(k); b = _PDt.get(k)
        if a and b: _rk_pairs.append((k, b["delta"], a["delta"]))
    if _rk_pairs:
        _chg = [k for k, m, r in _rk_pairs if (m < 0) != (r < 0)]
        _nw = {0: "no", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
        cmd("rkSignChangeN", len(_chg), "int"); cmd("rkLinN", len(_rk_pairs), "int")
        if not _chg: _ph = f"none of the {_nw.get(len(_rk_pairs), len(_rk_pairs))} linear estimates changes sign"
        else:
            _who = ", ".join(_rk_names[k] for k in _chg[:-1]) + (" and " if len(_chg) > 1 else "") + _rk_names[_chg[-1]]
            _ph = (f"{_nw.get(len(_chg), len(_chg))} of the {_nw.get(len(_rk_pairs), len(_rk_pairs))} linear estimates "
                   f"change{'s' if len(_chg) == 1 else ''} sign, {_who}")
        cmd("rkSignChangePhrase", _ph)
    for k, v in RK.get("versus_main", {}).items():
        nm = LINKEYS.get(k)
        if nm: cmd("robRkVs" + nm, v["delta"], "{:+.4f}"); cmd("robRkVs" + nm + "Abs", abs(v["delta"])); cmd("robRkMain" + nm, v["main"])
    for k, nm in DELKEYS.items():
        d = RK.get("deltas_main", {}).get(k)
        if d: cmd("robRkMainD" + nm, d["delta"], "{:+.4f}")
    G = RK.get("gnn")
    if G:
        cmd("robRkGnnCells", ", ".join(G["basis"])); cmd("robRkGnnN", len(G["basis"]), "int"); cells_phrase("robRkGnn", G["basis"])
        arm_macros("robRkGnnReal", G.get("real")); arm_macros("robRkGnnMean", G.get("cohort_mean")); pair_macros("robRkGnnPat", G.get("patient_identity"))
        Mn = G.get("main_same_cells", {})
        arm_macros("robRkGnnMainReal", Mn.get("real")); arm_macros("robRkGnnMainMean", Mn.get("cohort_mean")); pair_macros("robRkGnnMainPat", Mn.get("patient_identity"))
        pair_macros("robRkGnnRealVsMain", G.get("real_vs_main")); pair_macros("robRkGnnMeanVsMain", G.get("cohort_mean_vs_main"))

# early stopping on an inner held-out reaction partition (results/ivr/): the stopping criterion
# matched to the test question, on the fold-0 cells, paired with the main cells of the same name
IV = ROB.get("ivr")
if IV and IV.get("real") and IV.get("cohort_mean"):
    cmd("robIvrCells", ", ".join(IV["basis"])); cmd("robIvrN", len(IV["basis"]), "int")
    if IV.get("inner_val_fraction") is not None: cmd("robIvrFrac", 100.0 * IV["inner_val_fraction"], "{:.0f}")
    if IV.get("n_inner_val_rxn") is not None: cmd("robIvrNInner", IV["n_inner_val_rxn"], "int")
    if IV.get("n_fit_rxn") is not None: cmd("robIvrNFit", IV["n_fit_rxn"], "int")
    arm_macros("robIvrReal", IV.get("real")); arm_macros("robIvrMean", IV.get("cohort_mean")); pair_macros("robIvrPat", IV.get("patient_identity"))
    Mn = IV.get("main_same_cells", {})
    arm_macros("robIvrMainReal", Mn.get("real")); arm_macros("robIvrMainMean", Mn.get("cohort_mean")); pair_macros("robIvrMainPat", Mn.get("patient_identity"))
    pair_macros("robIvrRealVsMain", IV.get("real_vs_main")); pair_macros("robIvrMeanVsMain", IV.get("cohort_mean_vs_main"))
    # the zeroed, presence-only and wrong-patient arms under the matched rule, where they ran
    for _arm, _nm in (("zero", "Zero"), ("indicator", "Ind"), ("permute", "Perm")):
        if IV.get(_arm):
            arm_macros("robIvr" + _nm, IV[_arm]); pair_macros("robIvr" + _nm + "VsMain", IV.get(_arm + "_vs_main"))
    if IV.get("main_same_cells", {}).get("zero"): arm_macros("robIvrMainZero", IV["main_same_cells"]["zero"])
    for _k, _nm in (("aggregate_expression", "Agg"), ("expression_total", "ExprTot"), ("presence_over_zero", "Presence"),
                    ("magnitude_over_presence", "Magnitude"), ("own_over_presence", "OwnOverPresence"),
                    ("own_minus_permuted", "PermPat"), ("permuted_minus_mean", "PermMean")):
        if IV.get(_k): pair_macros("robIvr" + _nm, IV[_k])
    if IV.get("main_same_cells", {}).get("aggregate_expression"): pair_macros("robIvrMainAgg", IV["main_same_cells"]["aggregate_expression"])
    # a reading of the matched-rule wrong-patient check, chosen from its sign-flip p and its size
    _pp = INF.get("ivr_real_vs_permute")
    if _pp and _pp.get("n", 0) > 1:
        if _pp["signflip_p"] >= 0.05: _ph = f"not distinguishable from zero at {_pp['n']} cells"
        elif abs(_pp["mean"]) < 0.005: _ph = "small beside the movement of either arm under the change of rule, though stable in sign"
        else: _ph = "a difference the invariance does not predict"
        cmd("ivrPermReadingPhrase", _ph)
    # the share of the matched-rule aggregate-expression step that the presence pattern alone recovers
    if IV.get("presence_over_zero") and IV.get("aggregate_expression") and IV["aggregate_expression"]["delta"]:
        cmd("robIvrPresenceShare", 100.0 * IV["presence_over_zero"]["delta"] / IV["aggregate_expression"]["delta"], "{:.0f}")
    # the family-disjoint reaction split (results/fam/): the same five arms refitted with
    # biochemically related reactions withheld together. Every direction word is generated from the
    # numbers, and the reading of the primary contrast is assembled from its sign-flip test so that
    # the manuscript cannot assert a direction the run did not produce.
    FM = ROB.get("fam")
    if FM and FM.get("real") and FM.get("cohort_mean"):
        cmd("famCells", ", ".join(FM["basis"])); cmd("famN", len(FM["basis"]), "int")
        cmd("famScheme", FM.get("family_scheme") or "union")
        if FM.get("n_heldout_rxn") is not None:
            cmd("famNHeldout", int(round(FM["n_heldout_rxn"])), "int")
            cmd("famNHeldoutMin", FM["n_heldout_rxn_min"], "int"); cmd("famNHeldoutMax", FM["n_heldout_rxn_max"], "int")
        if FM.get("n_heldout_families") is not None: cmd("famNFamilies", int(round(FM["n_heldout_families"])), "int")
        for _k, _nm in (("real", "Real"), ("cohort_mean", "Mean"), ("zero", "Zero"),
                        ("indicator", "Ind"), ("permute", "Perm")):
            arm_macros("fam" + _nm, FM.get(_k))
        for _k, _nm in (("patient_identity", "Pat"), ("own_minus_permuted", "PermPat"),
                        ("permuted_minus_mean", "PermMean"), ("aggregate_expression", "Agg"),
                        ("expression_total", "ExprTot"), ("presence_over_zero", "Presence"),
                        ("magnitude_over_presence", "Magnitude"), ("own_over_presence", "OwnOverPresence"),
                        ("patient_identity_random_split", "PatRand"),
                        ("patient_identity_fam_minus_random", "PatMinusRand")):
            if FM.get(_k): pair_macros("fam" + _nm, FM[_k])
        # the price of the random split, arm by arm, and the largest of those prices
        VS = FM.get("versus_random_split") or {}
        for _k, _nm in (("real", "Real"), ("cohort_mean", "Mean"), ("zero", "Zero"),
                        ("indicator", "Ind"), ("permute", "Perm")):
            if VS.get(_k): pair_macros("famVsRand" + _nm, VS[_k])
        if VS:
            _ds = {k: v["delta"] for k, v in VS.items() if v and v.get("delta") is not None}
            if _ds:
                cmd("famVsRandMinAbs", min(abs(v) for v in _ds.values()))
                cmd("famVsRandMaxAbs", max(abs(v) for v in _ds.values()))
                cmd("famVsRandAllSameSign", "yes" if len({v > 0 for v in _ds.values()}) == 1 else "no")
                cmd("famVsRandDir", "below" if all(v < 0 for v in _ds.values()) else
                                    ("above" if all(v > 0 for v in _ds.values()) else "mixed"))
                # the direction as a word, so prose built on the absolute sizes cannot assert one
                _allneg, _allpos = all(v < 0 for v in _ds.values()), all(v > 0 for v in _ds.values())
                cmd("famVsRandWord", "loses" if _allneg else ("gains" if _allpos else "moves"))
                cmd("famVsRandWordPl", "lose" if _allneg else ("gain" if _allpos else "move"))
                cmd("famVsRandNoun", "drop" if _allneg else ("gain" if _allpos else "change"))
                cmd("famVsRandNounPl", "drops" if _allneg else ("gains" if _allpos else "changes"))
                cmd("famVsRandCostWord", "costs" if _allneg else ("buys" if _allpos else "changes"))
        # the reading of the primary contrast on the harder split, chosen from its sign-flip test
        _fp = INF.get("fam_real_vs_mean")
        if _fp and _fp.get("n", 0) > 1:
            _sig = _fp["signflip_p"] < 0.05
            if not _sig:
                cmd("famPatReadingPhrase",
                    f"small beside its own spread across cells and not distinguishable from zero at "
                    f"{_fp['n']} cells (sign-flip $p = {_fp['signflip_p']:.2f}$)")
                cmd("famPatReadingShort", "is smaller still and no longer distinguishable from zero")
                cmd("famPatVerdict", "not distinguishable from zero")
            elif _fp["mean"] > 0:
                cmd("famPatReadingPhrase", f"a gain the sign-flip test places away from zero at {_fp['n']} cells")
                cmd("famPatReadingShort", "helps"); cmd("famPatVerdict", "positive")
            else:
                cmd("famPatReadingPhrase", f"a loss the sign-flip test places away from zero at {_fp['n']} cells")
                cmd("famPatReadingShort", "costs"); cmd("famPatVerdict", "negative")
        # the matched-rule random-split contrast read the same way, so the two splits can be
        # compared in one sentence without a direction word being typed by hand
        _rp = INF.get("ivr_real_vs_mean")
        if _rp and _rp.get("n", 0) > 1:
            cmd("famRandShort", "not distinguishable from zero at %d cells" % _rp["n"] if _rp["signflip_p"] >= 0.05
                else ("a small gain" if _rp["mean"] > 0 else "a small loss"))
        # the same sign-flip sensitivity under the two block exchangeabilities, since the cells are
        # crossed and no single one of the three is the right unit on its own
        if _fp:
            cmd("famPatFlipCell", _fp["signflip_p"], "{:.2f}")
            for _b, _nm in (("block_patient", "Pfold"), ("block_reaction", "Rfold")):
                _v = _fp.get(_b)
                if _v:
                    cmd("famPatFlip" + _nm, _v["p"], "{:.2f}")
                    cmd("famPatFlip" + _nm + "Floor", _v["floor"], "{:.2f}")
                    cmd("famPatFlip" + _nm + "N", _v["n_blocks"], "int")
        _fd = INF.get("fam_pat_minus_rand_pat")
        if _fd and _fd.get("n", 0) > 1:
            cmd("famPatMinusRandPhrase",
                ("the two splits do not give distinguishable answers to this question at "
                 f"{_fd['n']} cells" if _fd["signflip_p"] >= 0.05 else
                 f"the harder split moves the contrast {'up' if _fd['mean'] > 0 else 'down'} by "
                 f"{abs(_fd['mean']):.4f}, a shift the sign-flip test places away from zero"))
        if FM.get("fold_seed") is not None:
            cmd("famSeed", str(FM["fold_seed"]))
            cmd("famSeedVerifiedPhrase",
                "rebuilding the folds at that seed reproduces the held-out sizes the cells recorded"
                if FM.get("fold_seed_verified") else
                "the folds could not be rebuilt from the committed map, and the seed is taken from the code")
        CP = FM.get("components")
        if CP:
            cmd("famCompN", CP["n_components"], "int"); cmd("famCompMax", CP["largest"], "int")
            cmd("famCompMaxPct", CP["largest_share"], "pct"); cmd("famCompSingletons", CP["singletons"], "int")
            cmd("famCompThreshPct", CP["threshold"], "pct")
            cmd("famCompThreshWord", "is reached" if CP["threshold_reached"] else "is not reached")
        # the linear reference rows on the family-disjoint folds, and the graph model's standing
        # against the no-patient floor on each split
        FL = FM.get("linear")
        if FL and FL.get("rows"):
            _RN = {"structure_only": "Struct", "annotation_only": "Annot", "topology_only": "Topo",
                   "structure_plus_cohortmean": "StructCoh", "structure_plus_perpat": "StructPer",
                   "expr_cohortmean_lr": "ExprCohLr", "expr_perpat_lr": "ExprPerLr",
                   "expr_cohortmean_rank": "ExprCoh", "expr_perpat_rank": "ExprPer",
                   "indicator": "Ind", "subsystem_prevalence": "Subsys", "degree_shared": "Degree",
                   "nearest_lookup_k5": "Lookup", "pooled_structure_plus_pooled": "Pooled",
                   "pooled_expr_pooled_lr": "PooledExpr"}
            for _k, _nm in _RN.items():
                b = FL["rows"].get(_k)
                if not b: continue
                cmd("famLin" + _nm, b["mean"]); cmd("famLin" + _nm + "N", b["n"], "int")
                if b.get("sd") is not None: cmd("famLin" + _nm + "SD", b["sd"])
                vr_ = b.get("versus_random")
                if vr_:
                    cmd("famLin" + _nm + "Rand", vr_["random_mean"])
                    cmd("famLin" + _nm + "VsRand", vr_["delta"], "{:+.4f}")
                    cmd("famLin" + _nm + "VsRandAbs", abs(vr_["delta"]))
                    cmd("famLin" + _nm + "VsRandN", vr_["n"], "int")
                    cmd("famLin" + _nm + "VsRandNeg", vr_["n_negative"], "int")
                    cmd("famLin" + _nm + "VsRandWord", "loses" if vr_["delta"] < 0 else "gains")
                    cmd("famLin" + _nm + "VsRandDir", "below" if vr_["delta"] < 0 else "above")
            for _k, _nm in (("gnn_real_minus_structure_only", "GnnOverStruct"),
                            ("gnn_real_minus_structure_only_random", "GnnOverStructRand"),
                            ("gap_change", "GapChange")):
                v = FL.get(_k)
                if not v: continue
                cmd("fam" + _nm, v["delta"], "{:+.4f}"); cmd("fam" + _nm + "Abs", abs(v["delta"]))
                cmd("fam" + _nm + "N", v["n"], "int"); cmd("fam" + _nm + "SD", v["sd"])
                cmd("fam" + _nm + "Dir", "above" if v["delta"] > 0 else "below")
                cmd("fam" + _nm + "Word", "above" if v["delta"] > 0 else "below")
            # a generated reading of what the harder split does to the deep model's standing
            _g, _gr = FL.get("gap_change"), FL.get("gnn_real_minus_structure_only_random")
            _gf = FL.get("gnn_real_minus_structure_only")
            _gc = INF.get("fam_gnn_lin_gap_change")
            if _g and _gr and _gf and _gc and _gc.get("n", 0) > 1:
                _wider = abs(_gf["delta"]) > abs(_gr["delta"]) and _gf["delta"] < 0 and _gr["delta"] < 0
                cmd("famGapPhrase",
                    (("the graph model's shortfall against the no-patient linear model widens from "
                      f"{abs(_gr['delta']):.4f} to {abs(_gf['delta']):.4f} AUROC")
                     if _wider else
                     ("the graph model's standing against the no-patient linear model moves by "
                      f"{abs(_g['delta']):.4f} AUROC"))
                    + (f", a change the sign-flip test places away from zero at {_gc['n']} cells"
                       if _gc["signflip_p"] < 0.05 else
                       f", a change not distinguishable from zero at {_gc['n']} cells"))
                cmd("famGapRatio", abs(_gf["delta"]) / abs(_gr["delta"]), "{:.1f}") if _gr["delta"] else None
        # the residue the audit still finds on the family split, and what dropping it does
        CL = FM.get("clean_subset")
        if CL and CL.get("contrasts"):
            cmd("famCleanDropMin", min(CL["n_dropped"]), "int"); cmd("famCleanDropMax", max(CL["n_dropped"]), "int")
            for _k, _nm in (("patient_identity", "Pat"), ("own_minus_permuted", "PermPat"),
                            ("aggregate_expression", "Agg"), ("presence_over_zero", "Presence")):
                v = CL["contrasts"].get(_k)
                if not v: continue
                cmd("famClean" + _nm, v["clean"]["delta"], "{:+.4f}")
                cmd("famClean" + _nm + "N", v["clean"]["n"], "int")
                cmd("famClean" + _nm + "SD", v["clean"]["sd"])
                cmd("famClean" + _nm + "Shift", v["shift"], "{:+.4f}")
                cmd("famClean" + _nm + "ShiftAbs", abs(v["shift"]))
            for _a, _nm in (("real", "Real"), ("mean", "Mean"), ("zero", "Zero"), ("indicator", "Ind"), ("permute", "Perm")):
                b = (CL.get("arms") or {}).get(_a)
                if b and b.get("clean") is not None:
                    cmd("famCleanArm" + _nm, b["clean"]); cmd("famCleanArm" + _nm + "N", b["n"], "int")
                    if b.get("n_heldout_clean") is not None:
                        cmd("famCleanArm" + _nm + "NRxn", int(round(b["n_heldout_clean"])), "int")
            cmd("famCleanMaxShiftAbs", CL["max_abs_shift"])
            cmd("famCleanSignsPhrase", "every contrast keeps its sign and its ordering"
                if CL.get("signs_unchanged") else "at least one contrast changes sign")
        # the training-seed replicates of the family split, where they ran
        FSR = FM.get("seed_replicates")
        if FSR and FSR.get("by_seed"):
            cmd("famSeedCells", ", ".join(FSR["cells"])); cmd("famSeedN", len(FSR["seeds"]), "int")
            cmd("famSeedList", ", ".join(str(x) for x in FSR["seeds"]))
            _vals = [FSR["default_seed"]["delta"]] + [v["delta"] for v in FSR["by_seed"].values()]
            cmd("famSeedDefault", FSR["default_seed"]["delta"], "{:+.4f}")
            cmd("famSeedValues", ", ".join(f"{v:+.4f}" for v in _vals))
            cmd("famSeedMin", min(_vals), "{:+.4f}"); cmd("famSeedMax", max(_vals), "{:+.4f}")
            cmd("famSeedSpreadAbs", max(_vals) - min(_vals))
            _NW = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
            cmd("famSeedSignPhrase", "all %s carry the same sign" % _NW.get(len(_vals), "%d values" % len(_vals))
                if len({v > 0 for v in _vals}) == 1 else "the values do not agree in sign")
        # the share of the family-split aggregate-expression step the presence pattern alone recovers
        if FM.get("presence_over_zero") and FM.get("aggregate_expression") and FM["aggregate_expression"]["delta"]:
            cmd("famPresenceShare", 100.0 * FM["presence_over_zero"]["delta"] / FM["aggregate_expression"]["delta"], "{:.0f}")
    # what the matched rule does to the memorization diagnostic, with the three reaction masks scored
    # separately (results/mask_scores.json): the loss-fit reactions, the inner-validation reactions
    # withheld from the loss and read for stopping, and the outer held-out reactions. The sentence is
    # generated in full so that its direction words follow the numbers.
    # what the matched rule does to the memorization gap, on the separated loss-fit mask: the
    # generated reduction, so no percentage is typed and none can go stale
    _mk = ROB.get("masks") or {}
    _ma_ = _mk.get("arms") or {}
    _rg, _rm = (_ma_.get("real") or {}).get("grid"), (_ma_.get("real") or {}).get("matched")
    if _rg and _rm and _rg.get("gap_fit_minus_outer"):
        cmd("mskGapReductionPct",
            100.0 * (1.0 - _rm["gap_fit_minus_outer"] / _rg["gap_fit_minus_outer"]), "{:.0f}")
    MK = ROB.get("masks") or {}
    _ma = MK.get("arms") or {}
    if _ma.get("real") and _ma["real"].get("grid"):
        _r, _g = _ma["real"]["matched"], _ma["real"]["grid"]; _gc = _ma["real"]["gap_change"]["fit_minus_outer"]
        cmd("mskN", _r["n"], "int"); cmd("mskNFit", _r["n_fit"], "int"); cmd("mskNInner", _r["n_inner_val"], "int"); cmd("mskNOuter", _r["n_outer_test"], "int")
        for _nm, _blk in _ma.items():
            for _rule, _b in (("Matched", _blk["matched"]), ("Grid", _blk.get("grid"))):
                if not _b: continue
                _p = "msk" + {"real": "Real", "cohort_mean": "Mean", "zero": "Zero", "indicator": "Ind", "permute": "Perm"}[_nm] + _rule
                cmd(_p + "Fit", _b["fit"]); cmd(_p + "Inner", _b["inner_val"]); cmd(_p + "Outer", _b["outer_test"]); cmd(_p + "TrainAll", _b["train_all"])
                cmd(_p + "Gap", _b["gap_fit_minus_outer"]); cmd(_p + "GapSD", _b["gap_fit_minus_outer_sd"] or 0.0)
                cmd(_p + "InnerGap", _b["gap_inner_minus_outer"], "{:+.4f}"); cmd(_p + "InnerGapAbs", abs(_b["gap_inner_minus_outer"]))
                cmd(_p + "Auprc", _b["auprc_outer"]); cmd(_p + "AuprcExpr", _b["auprc_outer_exprbearing"]); cmd(_p + "N", _b["n"], "int")
            _c = _blk.get("gap_change")
            if _c:
                _q = "msk" + {"real": "Real", "cohort_mean": "Mean", "zero": "Zero", "indicator": "Ind", "permute": "Perm"}[_nm]
                cmd(_q + "GapChange", _c["fit_minus_outer"]["delta"], "{:+.4f}"); cmd(_q + "GapChangeAbs", abs(_c["fit_minus_outer"]["delta"]))
                cmd(_q + "GapChangeNeg", _c["fit_minus_outer"]["n_negative"], "int"); cmd(_q + "GapChangeN", _c["fit_minus_outer"]["n"], "int")
                if _blk["grid"]["gap_fit_minus_outer"]:
                    cmd(_q + "GapChangePct", 100.0 * abs(_c["fit_minus_outer"]["delta"]) / _blk["grid"]["gap_fit_minus_outer"], "{:.0f}")
        _w = lambda a, b: "falls" if a < b else "rises"
        _pct = 100.0 * abs(_gc["delta"]) / _g["gap_fit_minus_outer"]
        _m = _ma.get("cohort_mean"); _mc = _m["gap_change"]["fit_minus_outer"] if _m and _m.get("gap_change") else None
        _agree = (f"within {abs(_r['gap_inner_minus_outer']):.4f} of" if abs(_r["gap_inner_minus_outer"]) < 0.005
                  else ("above" if _r["gap_inner_minus_outer"] > 0 else "below"))
        _mean_cl = ""
        if _mc:
            _cells = ("in every cell" if _mc["n_negative"] == _mc["n"] else f"though in only {_mc['n_negative']} of {_mc['n']} cells") if _mc["delta"] < 0 else \
                     ("in every cell" if _mc["n_positive"] == _mc["n"] else f"though in only {_mc['n_positive']} of {_mc['n']} cells")
            _cmp = ("narrows both arms' diagnostics" if (_mc["delta"] < 0 and _mc["n_negative"] > _mc["n"] // 2)
                    else "narrows the unsubstituted arm's diagnostic more than the cohort-mean arm's")
            _mean_cl = (f" The cohort-mean arm's gap {_w(_m['matched']['gap_fit_minus_outer'], _m['grid']['gap_fit_minus_outer'])} on average from "
                        f"{_m['grid']['gap_fit_minus_outer']:.4f} to {_m['matched']['gap_fit_minus_outer']:.4f}, {_cells}, so the change of rule {_cmp}.")
        _gcells = "in every cell" if _gc["n_negative"] == _gc["n"] else f"in {_gc['n_negative']} of {_gc['n']} cells"
        cmd("robIvrGapSentence",
            f"The criterion also changes the memorization diagnostic of Section~\\ref{{sec:memorization}}, read here on "
            f"the three reaction masks separately: on the same {_r['n']} cells the unsubstituted arm's AUROC on the "
            f"\\mskNFit{{}} reactions in the loss {_w(_r['fit'], _g['fit'])} from {_g['fit']:.4f} to {_r['fit']:.4f}, on the "
            f"\\mskNInner{{}} inner-validation reactions, which the matched rule withholds from the loss and the grid's "
            f"rule fits, {_w(_r['inner_val'], _g['inner_val'])} from {_g['inner_val']:.4f} to {_r['inner_val']:.4f}, and on the outer held-out reactions "
            f"{_w(_r['outer_test'], _g['outer_test'])} from {_g['outer_test']:.4f} to {_r['outer_test']:.4f}; the fit-minus-held-out gap "
            f"{_w(_r['gap_fit_minus_outer'], _g['gap_fit_minus_outer'])} from {_g['gap_fit_minus_outer']:.4f} to {_r['gap_fit_minus_outer']:.4f}, "
            f"a {'drop' if _gc['delta'] < 0 else 'rise'} of {_pct:.0f}\\% {_gcells}, and the inner-validation reactions, which the model never fitted, score "
            f"{_agree} the outer held-out reactions, so stopping on them did not fit them.{_mean_cl}")
        # the matched-rule AUPRC ladder on the shared cells
        _L = MK.get("auprc_ladder")
        if _L:
            cmd("mskLadderN", len(_L["basis"]), "int"); cmd("mskLadderPrev", _L["prevalence"])
            for _nm, _v in _L["rungs"].items():
                cmd("mskLad" + {"real": "Real", "cohort_mean": "Mean", "zero": "Zero", "indicator": "Ind", "permute": "Perm"}[_nm], _v)
            for _nm, _v in _L["rungs_exprbearing"].items():
                cmd("mskLadExpr" + {"real": "Real", "cohort_mean": "Mean", "zero": "Zero", "indicator": "Ind", "permute": "Perm"}[_nm], _v)
            for _k, _v in _L["steps"].items():
                _q = "mskStep" + "".join(w.capitalize() for w in _k.split("_"))
                cmd(_q, _v["delta"], "{:+.4f}"); cmd(_q + "Abs", abs(_v["delta"])); cmd(_q + "Pos", _v["n_positive"], "int"); cmd(_q + "N", _v["n"], "int")
                cmd(_q + "Word", "gain" if _v["delta"] > 0 else "loss")

# the synthetic patient-varying positive control: macros syn{Ivr|Main}{Full|Half}... per block
SYN = ROB.get("synth") or {}
_fracword = {1.0: "Full", 0.5: "Half", 0.25: "Quarter"}
for _k, B in SYN.items():
    if B["kind"] != "cohortmedian": continue
    pre = "syn" + ("Ivr" if B["selection"].startswith("inner") else "Main") + _fracword.get(B["frac"], "F" + str(B["frac"]).replace(".", ""))
    cmd(pre + "N", len(B["basis"]), "int"); cmd(pre + "Cells", ", ".join(B["basis"])); cells_phrase(pre, B["basis"])
    cmd(pre + "FracPct", 100.0 * B["frac"], "{:.0f}")
    if B.get("n_vary_rxn") is not None: cmd(pre + "NVary", B["n_vary_rxn"], "int")
    if B.get("n_heldout_vary_rxn") is not None: cmd(pre + "NHeldVary", B["n_heldout_vary_rxn"], "{:.0f}")
    if B.get("vary_positive_rate") is not None: cmd(pre + "VaryPrevPct", 100.0 * B["vary_positive_rate"], "{:.1f}")
    for arm, nm in (("real", "Real"), ("cohort_mean", "Mean"), ("raw_own_column", "RawOwn"), ("cohort_mean_column", "RawMean")):
        for part, pn in (("all", "All"), ("vary", "Vary"), ("shared", "Shared"), ("trainrxn", "Train")):
            v = (B.get(arm) or {}).get(part)
            if v is not None: cmd(pre + nm + pn, v)
    pair_macros(pre + "Pat", B.get("patient_identity")); pair_macros(pre + "PatVary", B.get("patient_identity_vary"))
    if B.get("patient_identity_shared"): pair_macros(pre + "PatShared", B["patient_identity_shared"])
if SYN: cmd("synBlocks", len([1 for B in SYN.values() if B["kind"] == "cohortmedian"]), "int")
# the section's title, its contrast clause and its closing sentence, chosen from the sign of the
# fully varying, matched-rule contrast on the varying reactions, so that none of them can outrun it
_syb = SYN.get("cohortmedian_1_ivr") or SYN.get("cohortmedian_1_main")
if _syb and _syb.get("patient_identity_vary"):
    _pv = _syb["patient_identity_vary"]; _d = _pv["delta"]; _allpos = _pv["n_positive"] == _pv["n"]
    if _d > 0.005 and _allpos:
        cmd("synTitlePhrase", "The audit's contrast turns positive on a target that carries patient-specific signal")
        cmd("synContrastClause", "a benefit where the main grid's proxy gives a cost")
        cmd("synClosingSentence", "The contrast therefore measures what the audit says it measures: it is positive where the "
            "target rewards the patient's own data and, for the graph model, at or below zero where the target cannot, and its sign on the proxy is a property of the proxy.")
    elif _d > 0:
        cmd("synTitlePhrase", "The audit's contrast on a target that carries patient-specific signal")
        cmd("synContrastClause", "a benefit in the mean, though not in every cell, where the main grid's proxy gives a cost")
        cmd("synClosingSentence", "The contrast moves in the direction the construction predicts, without turning positive in every cell, "
            "so the audit's sign on the proxy is read as a property of the proxy with that qualification.")
    else:
        cmd("synTitlePhrase", "The audit's contrast on a target that carries patient-specific signal")
        cmd("synContrastClause", "a cost, as on the main grid's proxy")
        cmd("synClosingSentence", "The contrast does not turn positive on this target, so this control does not show that the audit's "
            "sign on the proxy is a property of the proxy; Section~\\ref{sec:limitations} says what that leaves open.")

# a reading of the contrast under the matched selection rule against the main grid's, as a phrase
# chosen from the numbers, so that no sentence about the selection rule can outrun the result
_ivc = INF.get("ivr_real_vs_mean"); _mvc = INF.get("main_real_vs_mean_on_ivr_cells")
if _ivc and _mvc and _ivc["n"] >= 3:
    _m, _pm = _ivc["mean"], _mvc["mean"]; _p = _ivc["signflip_p"]; _npos, _nneg = _ivc["n_positive"], _ivc["n_negative"]
    if abs(_m) <= 0.005:
        _ph = ("is within half a hundredth of zero under the matched rule, so the penalty seen under the "
               "standard rule is largely a property of model selection")
        _short = "largely a property of model selection"
    elif _p >= 0.10:
        _ph = ("is not distinguishable from zero under the matched rule at this cell count, so its size "
               "under the matched rule is not established")
        _short = "not established under the matched rule"
    elif _m < 0 and abs(_m) < 0.5 * abs(_pm):
        _ph = ("shrinks by more than half under the matched rule but stays negative, so part of the penalty "
               "is a property of model selection and part survives it")
        _short = "partly a property of model selection"
    elif _m < 0 and abs(_m) < 0.9 * abs(_pm):
        _ph = ("shrinks by less than half under the matched rule and stays negative, so most of the penalty "
               "survives the change of selection rule")
        _short = "reduced but not removed by the selection rule"
    elif _m < 0:
        _ph = ("is as large under the matched rule, so the penalty does not depend on the selection rule")
        _short = "not a property of the selection rule"
    else:
        _ph = ("reverses under the matched rule, so the sign of the contrast is a property of model selection")
        _short = "a property of model selection in its sign"
    cmd("ivrReadingPhrase", _ph); cmd("ivrReadingShort", _short)
    # the same reading with the rule named inside the phrase, for sentences that have not named it
    _self = {"is within half a hundredth of zero under the matched rule": "is within half a hundredth of zero when the model is selected on reactions withheld from the loss",
             "is not distinguishable from zero under the matched rule at this cell count": "is not distinguishable from zero at this cell count when the model is selected on reactions withheld from the loss",
             "shrinks by more than half under the matched rule but stays negative": "shrinks by more than half but stays negative when the model is selected on reactions withheld from the loss",
             "shrinks by less than half under the matched rule and stays negative": "shrinks by less than half and stays negative when the model is selected on reactions withheld from the loss",
             "is as large under the matched rule": "is as large when the model is selected on reactions withheld from the loss",
             "reverses under the matched rule": "reverses when the model is selected on reactions withheld from the loss"}
    _ps = _ph
    for _k, _v in _self.items():
        if _ph.startswith(_k): _ps = _v + _ph[len(_k):]; break
    cmd("ivrReadingPhraseSelf", _ps.replace("the standard rule", "the grid's rule"))
    _rv = INF.get("ivr_real_vs_main_real"); _mv = INF.get("ivr_mean_vs_main_mean")
    if _rv and _mv:
        _dr, _dm = _rv["mean"], _mv["mean"]
        if _dm < _dr - 0.002:
            cmd("ivrArmPhrase", "so the grid's rule favors the arm whose input is one shared vector, which is the arm on which validation patients and training patients receive the identical column, though this experiment does not test that as the mechanism")
        elif _dr < _dm - 0.002:
            cmd("ivrArmPhrase", "so the grid's rule favors the own-expression arm")
        else:
            cmd("ivrArmPhrase", "so the two arms move alike under the change of rule")
    cmd("ivrSignEstablished", "yes" if (_p < 0.05) else "no")
    cmd("ivrSignStabilityPhrase", f"the sign is stable at {_ivc['n']} cells" if _p < 0.05 else f"the sign is not established at {_ivc['n']} cells")
    # the graph model against the no-patient linear floor under the matched rule: the better graph
    # arm's margin over the floor, and a phrase for the abstract and the discussion
    _gs_r, _gs_m = INF.get("ivr_real_vs_lin_tuned_structure_only"), INF.get("ivr_mean_vs_lin_tuned_structure_only")
    if _gs_r and _gs_m:
        _best = max(_gs_r["mean"], _gs_m["mean"])
        cmd("ivrGnnBestVsStruct", _best, "{:+.4f}"); cmd("ivrGnnBestVsStructAbs", abs(_best), "{:.4f}")
        # the phrase names no rule; the sentences that use it say "under the matched rule" themselves
        if _best <= 0: _gp = "at or above both of the graph model's arms"
        elif _best <= 0.001: _gp = f"within {abs(_best):.4f} of the graph model's better arm"
        else: _gp = f"{abs(_best):.4f} below the graph model's better arm"
        cmd("ivrGnnVsStructPhrase", _gp)
    # the clause of the robustness summary that says what the selection check does to the sign
    if _p < 0.05 and _m < 0: _sc = "the contrast keeps its sign under it"
    elif _p < 0.05 and _m > 0: _sc = "the contrast reverses its sign under it"
    else: _sc = "it is the one check under which the contrast's sign is not established"
    cmd("ivrSignClause", _sc)
    cmd("ivrSignPhrase", ("negative in every cell" if _nneg == _ivc["n"] else "positive in every cell" if _npos == _ivc["n"]
                          else f"negative in {_nneg} of {_ivc['n']} cells"))

# clauses that say when two different quantities print as the same number, so that no sentence
# asserts a coincidence the numbers do not show (or misses one they do)
_gp = ((((R.get("naive_pooled") or {}).get("deltas") or {}).get("tuned") or {}).get("gnn_real_vs_pooled_struct") or {})
_primary = (_IVe.get("patient_identity") or {}).get("delta") if _IVe.get("patient_identity") and "estHasIvr" in "".join(L) else ((R.get("substitution") or {}).get("patient_identity") or {}).get("delta")
if _gp.get("delta") is not None and _primary is not None:
    _same = f"{abs(_gp['delta']):.4f}" == f"{abs(_primary):.4f}"
    cmd("sameSizeGnnPooledClause", (", which happens to be of the same size as its matched-rule contrast" if "estHasIvr" in "".join(L) else ", which happens to be of the same size") if _same else "")
    cmd("sameSizeGnnPooledClauseB", " that happens to be of the same size as the contrast above" if f"{abs(_gp['delta']):.4f}" == f"{abs(_primary):.4f}" else "")
_pl = ((ROB.get("pooled_large") or {}).get("gnn_real_vs_pooled_struct") or {})
if _pl.get("delta") is not None and _gp.get("delta") is not None:
    _same = f"{abs(_pl['delta']):.4f}" == f"{abs(_gp['delta']):.4f}"
    cmd("lbPooledLargeGapVsDefaultClause", ", the same gap as with the default subsample" if _same
        else f", against {abs(_gp['delta']):.4f} with the default subsample")

# release identifiers (repository root release.json): the Git tag the manuscript corresponds to and,
# once minted, the version DOIs; an empty field leaves its macro undefined
for _cand in (os.path.join(HERE, "..", "release.json"), os.path.join(HERE, "..", "..", "release.json"), "release.json"):
    if os.path.exists(_cand):
        _rel = json.load(open(_cand))
        for _k, _m in (("tag", "releaseTag"), ("version_doi_code", "releaseDoiCode"), ("version_doi_data", "releaseDoiData"), ("commit", "releaseCommit")):
            if _rel.get(_k):
                _v = str(_rel[_k])
                # a commit is quoted to identify a state, not to be retyped; twelve hex digits
                # are unambiguous in a repository of this size and read as a word rather than a wall
                if _m == "releaseCommit" and len(_v) > 12: _v = _v[:12]
                cmd(_m, _v.replace("_", "\\_"))
        break
out = os.path.join(HERE, MS, "numbers.tex")
# a checkout that carries the analysis without the manuscript sources has no such directory;
# the macro file is a result and is written either way
os.makedirs(os.path.dirname(out), exist_ok=True)
_byname = {}
for _line in L:                      # the last definition of a name wins, so an explicit word macro overrides an automatic one
    _byname[_line.split("}")[0]] = _line
L = list(_byname.values())
open(out, "w").write("% generated by code/make_numbers_p1.py -- do not edit\n" + "\n".join(L) + "\n")
print(f"wrote {MS}/numbers.tex with {len(L)} macros")
