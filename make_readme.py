#!/usr/bin/env python3
"""Regenerate the result tables of README.md from the two results files.

The README quotes numbers from both studies. So that none of them is typed by hand, this
script rewrites everything between the markers

    <!-- generated:start -->  ...  <!-- generated:end -->

from paper1/data/results_p1.json and paper2/results_frozen.json. Run it after
`bash reproduce.sh analysis`; it is part of that script.
"""
import json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
P1 = json.load(open(os.path.join(HERE, "paper1", "data", "results_p1.json")))
P2 = json.load(open(os.path.join(HERE, "paper2", "results_frozen.json")))

def f4(x): return f"{x:.4f}"
def s4(x): return f"{x:+.4f}"
def f3(x): return f"{x:.3f}"
def a4(x): return f"{abs(x):.4f}"
def lead(x): return ("leads it by " if x >= 0 else "trails it by ") + a4(x)
def above(x): return a4(x) + (" above" if x >= 0 else " below")
def pfmt(p):
    """The manuscript's convention (make_numbers_p1.py): a p-value that never understates."""
    if p is None: return ""
    import math
    if p >= 0.001: return "{:.4f}".format(math.ceil(p * 1e4) / 1e4)
    if p < 1e-5: return "0.00001"
    e = math.floor(math.log10(p)); scale = 10.0 ** (e - 1)
    v = math.ceil(p / scale - 1e-9) * scale
    return "{:.{d}f}".format(v, d=max(0, -e + 1))

L = []
S, LB, INF = P1["substitution"], P1["ladder_basis"], P1["inference"]
cols = P1["naive_allfolds"]["columns"]
RW = S.get("rewired", {})
L += ["## What we found: the Reaction-Scoring Audit", "",
      f"Held-out-reaction AUROC on the {LB['n_basis']} cells of the substitution basis "
      f"({len(S['patient_folds'])} patient folds x 3 reaction folds), every row on the same cells, linear rows "
      f"with the regularization tuned by inner cross-validation; the rewiring row is {RW.get('n', 0)} cells over "
      f"patient folds {', '.join(str(p) for p in RW.get('pfolds', []))}.", "",
      "| predictor | AUROC | uses |", "|---|---|---|",
      f"| patient-invariant availability baseline | {f4(LB['indicator'])} | whether a reaction receives any expression |",
      f"| each patient's own expression, ranked | {f4(LB['expr_perpat'])} | the transcriptome, no model |",
      f"| cohort-mean expression, ranked | {f4(LB['expr_cohortmean'])} | one average transcriptome |",
      f"| each patient's own expression, fitted | {f4(LB['expr_perpat_lr'])} | the transcriptome, fitted |",
      f"| cohort-mean expression, fitted | {f4(LB['expr_cohortmean_lr'])} | the same, fitted, **no patient identity** |",
      f"| topology only, fitted ({cols['topology']} columns) | {f4(LB['topology_only'])} | degrees, metabolite count, single-metabolite flag; **no patient data** |",
      f"| annotation only, fitted ({cols['annotation']} columns) | {f4(LB['annotation_only'])} | gene rule, gene count, reversibility, subsystem; **no patient data** |",
      f"| network features, fitted ({cols['structure']} columns) | {f4(LB['structure_only'])} | topology + annotation, **no patient data at all** |",
      f"| network features + cohort-mean expression | {f4(LB['structure_plus_cohortmean'])} | + aggregate expression |",
      f"| network features + own expression | {f4(LB['structure_plus_perpat'])} | + patient identity |"]
if RW:
    L.append(f"| graph model, degree-preserving rewiring | {f4(RW['mean'])} | a random network of the same degrees |")
L += [f"| graph model, expression zeroed | {f4(LB['gnn_zero'])} | topology, learned |"]
if S.get("indicator"):
    I = S["indicator"]
    L.append(f"| graph model, presence pattern only ({I['n']} cells, fold {', '.join(str(p) for p in I['pfolds'])}) | {f4(I['mean'])} | which reactions have an expressed gene, no magnitudes |")
L += [f"| graph model, cohort-mean expression | {f4(LB['gnn_cohort_mean'])} | topology + aggregate expression |",
      f"| graph model, patient's own expression | {f4(LB['gnn_real'])} | everything |", ""]

lin = INF["lin_tuned_structure_only_vs_expr_cohortmean_lr"]
L += ["A matched comparison leaves three things standing.", "",
      "**The reference network predicts these labels far better than the transcriptome does, and a linear "
      "model on it comes close to the graph model.** With both sides fitted the same way on the same cells, the "
      f"network features lead the fitted expression baseline by {s4(LB['structure_over_expression_fitted'])} (topology alone "
      f"{s4(LB['topology_over_expression_fitted'])}, annotation alone {s4(LB['annotation_over_expression_fitted'])}), in every cell. "
      f"Against the linear model given the same expression column, the graph model {lead(LB['gnn_over_lr_with_expression'])} "
      f"with one cohort-average vector, {lead(LB['gnn_over_lr_own_expression'])} with each patient's own transcriptome, and "
      f"{lead(LB['gnn_over_lr_no_expression'])} with no expression at all. The no-patient linear model sits "
      f"{above(-LB['gnn_mean_over_structure'])} the graph model with a cohort average and {above(-LB['gnn_real_over_structure'])} the "
      "graph model with each patient's own data.", ""]
if RW:
    rw = INF.get("real_vs_rewire7", {})
    L += ["**The graph model's structural signal is the metabolic network itself, not generic graph statistics.** "
          "Rewiring the network at fixed degree, so every node keeps its exact connectivity in every relation and "
          f"nothing else survives, moves the graph model by {s4(-RW['vs_real']['delta'])} to {f4(RW['mean'])} over "
          f"{RW['vs_real']['n']} paired cells (exact sign-flip p = {pfmt(rw.get('signflip_p'))}, its floor at that count), "
          f"below the raw-expression ranking, while its fit to training reactions stays at {f4(RW['trainrxn'])}: given a network "
          "carrying no usable information the model falls back on memorizing, and the double hold-out catches it.", ""]
    for seed, W2 in (S.get("rewired_other_seeds") or {}).items():
        L += [f"A second permutation seed ({seed}) on {W2['n']} of those cells gives {f4(W2['mean'])}, against "
              f"{f4(W2['vs_seed7']['a_mean'])} for seed 7 on the same cells (seed 7 minus seed {seed}: "
              f"{s4(W2['vs_seed7']['delta'])}), and {a4(W2['vs_real']['delta'])} below the unrewired model on "
              f"those cells.", ""]
if S.get("indicator"):
    I = S["indicator"]
    share = 100.0 * I["presence_over_zero"]["delta"] / I["cohort_mean_over_zero_same_cells"]["delta"]
    L += ["**Most of what the cohort-mean column adds is the presence pattern.** Shown only which reactions have an "
          f"expressed gene, the graph model reaches {f4(I['mean'])} on the {I['n']} cells of the presence-only arm, "
          f"{s4(I['presence_over_zero']['delta'])} over the zeroed column and {a4(I['magnitude_over_presence']['delta'])} "
          f"short of the cohort mean on the same cells: {share:.0f} % of the aggregate-expression step is the pattern, "
          "the rest the magnitudes.", ""]
CN = P1.get("collapse_noise", {}).get("summary", {})
PDt = (((P1.get("naive_pooled") or {}).get("deltas") or {}).get("tuned") or {})
IVR = ((P1.get("robustness") or {}).get("ivr") or {}).get("patient_identity")
ests = [LB["nonparam_patient_effect"],
        INF["lin_tuned_expr_perpat_lr_vs_expr_cohortmean_lr"]["mean"],
        INF["lin_tuned_structure_plus_perpat_vs_structure_plus_cohortmean"]["mean"],
        -INF["collapse_identity_mlp"]["mean"], -INF["collapse_identity_mlp_emb"]["mean"],
        -INF["collapse_identity_gnn_B"]["mean"], INF["real_vs_mean"]["mean"]]
ests += [PDt[k]["delta"] for k in ("pooled_vs_cohort_expr", "pooled_vs_cohort_struct") if PDt.get(k)]
if IVR and IVR.get("n") == INF["real_vs_mean"]["n"]: ests.append(IVR["delta"])
n_neg = sum(1 for v in ests if v < -0.001); n_est = len(ests)
words = {7: "seven", 8: "eight", 9: "nine", 10: "ten"}.get(n_est, str(n_est))
L += [f"**Individual expression is worth at most {s4(max(ests))} AUROC over a cohort average here, and at worst "
      f"{s4(min(ests))}.** We measured the effect of giving each patient their own transcriptome instead of one "
      f"average vector {words} ways on one cohort and one fold structure; {n_neg} of the {words} estimates are negative by more "
      "than a thousandth. The collapse rows are reported net of the part that averaging dropout noise alone produces"
      + (f"; the graph model's retraining is reported under the grid's selection rule ({s4(INF['real_vs_mean']['mean'])}) and, as the "
         f"paper's primary contrast, under early stopping on reactions withheld from the loss ({s4(IVR['delta'])}, negative in "
         f"{IVR['n_negative']} of {IVR['n']} cells)" if IVR and IVR.get("n") == INF["real_vs_mean"]["n"] else "") + ".", "",
      "| measurement | effect of patient identity | n | exact p |", "|---|---|---|---|"]
def row(label, c, sign=1.0, p=True):
    if not c: return
    L.append(f"| {label} | {s4(sign * c['mean'])} | {c['n']} | {pfmt(c['signflip_p']) if p else ''} |")
L.append(f"| expression only, ranked, no model | {s4(LB['nonparam_patient_effect'])} | {LB['n_basis']} | |")
row("expression only, linear", INF.get("lin_tuned_expr_perpat_lr_vs_expr_cohortmean_lr"), p=False)
row("network features + expression, linear", INF.get("lin_tuned_structure_plus_perpat_vs_structure_plus_cohortmean"), p=False)
row("feature model, cohort-mean collapse, net of noise", INF.get("collapse_identity_mlp"), sign=-1.0)
row("memorization control, collapse, net of noise", INF.get("collapse_identity_mlp_emb"), sign=-1.0)
row("graph model, cohort-mean collapse, net of noise", INF.get("collapse_identity_gnn_B"), sign=-1.0)
for k, lab in (("pooled_vs_cohort_expr", "expression only, linear, pooled and frozen"), ("pooled_vs_cohort_struct", "network features + expression, linear, pooled and frozen")):
    if PDt.get(k): L.append(f"| {lab} | {s4(PDt[k]['delta'])} | {PDt[k]['n']} | |")
c = INF["real_vs_mean"]
L.append(f"| graph model, input substitution, paired, grid's selection rule | {s4(c['mean'])} | {c['n']} across {c['block_patient']['n_blocks']} patient folds | "
         f"**{pfmt(c['signflip_p'])}** (patient-fold means: {pfmt(c['t_patient_folds'].get('p'))}) |")
ci = INF.get("ivr_real_vs_mean")
if ci and ci.get("n") == c["n"]:
    L.append(f"| graph model, input substitution, paired, selected on withheld reactions (primary) | {s4(ci['mean'])} | {ci['n']} across {ci['block_patient']['n_blocks']} patient folds | "
             f"**{pfmt(ci['signflip_p'])}** (patient-fold means: {pfmt(ci['t_patient_folds'].get('p'))}) |")
L.append("")
L.append("The linear rows are patient-invariant in their fitted part, so no cell-level p is given for them.")
L.append("")

# ---- robustness checks (fill in as cells land; the pipeline guards every sentence on what exists)
RB = P1.get("robustness", {})
if RB:
    L += ["**Under the upstream choices tried, no estimate makes individual expression worth more than a few thousandths of AUROC over a cohort average.** Three choices "
          "upstream of every result were varied once each: the label (rebuilt from the HT29 reconstruction "
          "alone instead of the union of eleven), the expression normalization (within-patient percentiles "
          "instead of log2(CPM + 1)) and, for the graph model, the training seed, the early-stopping criterion and a "
          "wrong-patient consistency check. Linear rows are rerun on every cell; graph arms on the fold-0 cells.", "",
          "| check | expression only, linear | network features + expression, linear | graph model, input substitution |",
          "|---|---|---|---|"]
    def _d(block, key):
        d = (block or {}).get(key); return f"{d['delta']:+.4f} (n={d['n']})" if d else "not run"
    H = RB.get("ht29", {}); HL = (H.get("linear_all_cells") or {}).get("_deltas", {})
    hg = H.get("patient_identity"); hgs = f"{hg['delta']:+.4f} (n={hg['n']}, fold 0)" if hg else "in progress"
    L.append(f"| HT29-only label | {_d(HL, 'lin_patient_expr_only')} | {_d(HL, 'lin_patient_struct_expr')} | {hgs} |")
    K = RB.get("rank", {}); KL = (K.get("linear") or {}).get("_deltas", {}) if isinstance(K.get("linear"), dict) else {}
    kg = (K.get("gnn") or {}).get("patient_identity") if isinstance(K.get("gnn"), dict) else None
    kgs = f"{kg['delta']:+.4f} (n={kg['n']}, fold 0)" if kg else "in progress"
    L.append(f"| within-patient percentile normalization | {_d(KL, 'lin_patient_expr_only')} | {_d(KL, 'lin_patient_struct_expr')} | {kgs} |")
    SR = RB.get("seedrep") or {}
    SR1 = SR.get("1") if isinstance(SR, dict) else None
    srg = (SR1 or {}).get("patient_identity") if isinstance(SR1, dict) else None
    if srg and not srg.get("n"): srg = None
    srs = f"{srg['delta']:+.4f} (n={srg['n']}, fold 0)" if srg else "in progress"
    L.append(f"| second training seed | n/a | n/a | {srs} |")
    PM_ = P1.get("substitution", {}).get("permute")
    pmg = PM_.get("patient_identity") if isinstance(PM_, dict) else None
    pms = f"{pmg['delta']:+.4f} (n={pmg['n']}, fold 0)" if pmg else "in progress"
    L.append(f"| wrong-patient arm, a consistency check under the shared label (own minus permuted column) | n/a | n/a | {pms} |")
    L.append("")

# ---- phenotypes
PH, PN, PM = P1["phenotype"], P1.get("phenotype_null", {}), P1.get("phenotype_multi", {})
real, inp, gen = PH["arms"]["real"], PH["input"], PH["genes"]
mean_arm, zero_arm = PH["arms"].get("mean"), PH["arms"].get("zero")
L += ["## What the benchmark cannot see", "",
      "Every result above is measured against a label vector shared by all patients, which cannot reward "
      "patient specificity. Against a phenotype that does vary between patients, microsatellite "
      f"instability ({real['n_pos']} MSI-H of {real['n_patients']} annotated), the graph model's out-of-fold score vector "
      f"classifies unstable against stable tumors at AUROC **{f3(real['auroc'])}** (mean over three reaction folds), against "
      f"{f3(inp['auroc'])} for the model's own input and {f3(gen['auroc'])} for the transcriptome before GPR mapping. "
      + (f"The arms that show every patient an identical input sit at chance ({f3(mean_arm['auroc'])} and {f3(zero_arm['auroc'])}). "
         if mean_arm and zero_arm else "")
      + "The output carries most of the patient signal it is given; the benchmark, whose labels do not "
      f"vary with the patient, scores that same output {above(c['mean'])} the cohort-mean arm.", "",
      "The same probe on three further attributes the model never saw. Output is the mean over reaction "
      "folds; the permutation p is that of the least favorable reaction fold (200 label permutations through "
      "the full nested procedure, floor 0.005); the cohort-mean arm sees an identical input for every patient.", "",
      "| attribute | n | positives | input | output | perm. p | cohort-mean arm |", "|---|---|---|---|---|---|---|",
      f"| microsatellite instability | {real['n_patients']} | {real['n_pos']} | {f3(inp['auroc'])} | {f3(real['auroc'])} | "
      f"{pfmt(PN.get('msi', {}).get('output', {}).get('perm_p'))} | {f3(mean_arm['auroc']) if mean_arm else ''} |"]
for k, lab in (("cin_vs_gs", "chromosomal instability vs genomically stable"), ("sex", "sex"), ("site", "anatomical site, colon vs rectum")):
    v = PM.get(k)
    if not v: continue
    L.append(f"| {lab} | {v['n']} | {v['n_pos']} | {f3(v['input'])} | {f3(v['real'])} | "
             f"{pfmt(PN.get(k, {}).get('output', {}).get('perm_p'))} | {f3(v['cohort_mean']) if v.get('cohort_mean') is not None else ''} |")
L += ["", "The output follows its input in rank order and on no attribute adds to it.", ""]

# ---- Language-Model Evidence in brief
# the linear rungs on a second cohort (paper1/SECOND_COHORT_PLAN_2026-11-13.md)
C2 = json.load(open(os.path.join(HERE, "paper1", "data", "results_cohort2.json")))
_c2 = C2["cohorts"]["cohort2"]; _c1 = C2["cohorts"]["cohort1"]
_all2 = [_c2[sp]["contrasts"][k]["mean"] for sp in ("random", "family") for k in ("ranked", "fitted", "network")]
_all1 = [_c1[sp]["contrasts"][k]["mean"] for sp in ("random", "family") for k in ("ranked", "fitted", "network")]
_n2 = C2["second_cohort_expression_qc"]["patients_kept"]
L += [f"**The linear rungs on a second cohort.** The rows that need no graph training were repeated on {_n2:,} breast tumors "
      f"(TCGA-BRCA), built as the first cohort was, under a plan committed before the analysis ran "
      f"(`paper1/SECOND_COHORT_PLAN_2026-11-13.md`). On the family-disjoint split, the network features with each patient's own "
      f"expression against the same with the cohort-mean expression: {s4(_c2['family']['contrasts']['network']['mean'])} "
      f"(first cohort {s4(_c1['family']['contrasts']['network']['mean'])}). Over the three linear contrasts and both splits the "
      f"second cohort's values run from {s4(min(_all2))} to {s4(max(_all2))} and the first cohort's from {s4(min(_all1))} to "
      f"{s4(max(_all1))}.", ""]
# the graph scorer on the second cohort (Section 6 of the same plan)
CG = json.load(open(os.path.join(HERE, "paper1", "data", "results_cohort2_graph.json")))
_g2 = CG["cohorts"]["cohort2"]; _g1 = CG["cohorts"]["cohort1"]; _gc = _g2["contrast"]
L += [f"**The graph scorer on the second cohort.** Section 6 of that plan, fixed before any such run, names the driver of the first "
      f"cohort unchanged, the family-disjoint split, the matched selection rule and the default training seed, with the "
      f"own-expression arm and the cohort-mean arm on each of the {_gc['n_cells']} cells ({2 * _gc['n_cells']} runs). Own expression "
      f"scores {f4(_g2['arms']['own']['mean'])} and cohort-mean expression {f4(_g2['arms']['cohort_mean']['mean'])}, a paired "
      f"difference of {s4(_gc['mean'])} over cells (standard deviation {f4(_gc['sd'])}, range {s4(_gc['min'])} to {s4(_gc['max'])}, "
      f"negative in {_gc['n_negative']} of {_gc['n_cells']}), against {s4(_g1['contrast']['mean'])} on the first cohort "
      f"(standard deviation {f4(_g1['contrast']['sd'])}, negative in {_g1['contrast']['n_negative']} of {_g1['contrast']['n_cells']}). "
      f"The plan reads a mean within {CG['margin']:.2f} AUROC of zero as agreement with the first cohort, and this mean "
      f"{'lies' if _gc['within_margin'] else 'does not lie'} within that range. The cells share patients, reactions and one label "
      f"vector and the plan fixed no test, so these are descriptions. The runs were made on a second workstation "
      f"(`paper1/data/cohort2_graph_run_record.json`).", ""]
# outside that plan: one cell at two further training seeds
_sp = os.path.join(HERE, "paper1", "data", "results_cohort2_seed.json")
if os.path.exists(_sp):
    _SD = json.load(open(_sp)); _s2 = _SD["cohorts"]["cohort2"]; _s1 = _SD["cohorts"]["cohort1"]
    L[-2] += (f" Outside the plan, cell {_SD['cell']} was retrained at two further training seeds "
              f"({', '.join(str(x) for x in _SD['further_seeds'])}), both arms: the difference at that cell is "
              f"{', '.join(s4(v) for v in _s2['differences'])} across the three seeds, against {', '.join(s4(v) for v in _s1['differences'])} "
              f"for the same cell of the first cohort (`paper1/data/results_cohort2_seed.json`).")

R2 = P2["runs"]; A, E, B = R2["A_representative"], R2["E_representative40"], R2["B_balanced"]
XS = P2.get("cross_session", {}).get("arms", {}).get("patient", {})
pc = P2["positive_control"]["backbone1"]; M = P2["msi"]
def arm(r, a): return r["per_arm"][a]["auroc"]
L += ["## Language-Model Evidence in brief", "",
      f"A language model ({A.get('model', 'the served backbone')}) asked to score the same reactions "
      "from each patient's transcriptome. Three prompt arms: patient-blind, the patient's own value, and a stranger's "
      "value in the same slot (a new donor for every reaction).", "",
      "| design | reactions | patients | patient-blind | + own | + stranger's | raw expression, no model | availability baseline |",
      "|---|---|---|---|---|---|---|---|"]
for tag, r in (("A", A), ("B", B), ("E", E)):
    L.append(f"| {tag} | {r['n_reactions']} | {r['n_patients']} | {f4(arm(r, 'reaction'))} | {f4(arm(r, 'patient'))} | "
             f"{f4(arm(r, 'shuffled'))} | {f4(r['raw_expression_auroc'])} | {f4(r['indicator_floor'])} |")
pe = E["patient_vs_shuffled"]
L += ["",
      f"No arm reaches the raw-expression ranking or the availability baseline in any design. The paired own-minus-stranger's "
      f"difference at {E['n_patients']} patients is {s4(pe['auroc_diff_mean'])} (90 % interval {s4(pe['auroc_diff_ci90'][0])} to "
      f"{s4(pe['auroc_diff_ci90'][1])}), equivalent to zero within 0.02 AUROC. Between-patient spread of the scores is "
      f"{f3(pe['between_patient_sd_real'])} with the patients' own data and {f3(pe['between_patient_sd_stranger'])} with strangers'.",
      "",
      f"Three determinism floors (mean absolute change in the returned probability when nothing but the session changes): "
      f"repeating the patient-blind prompt {A['determinism']['mean_abs_diff']:.4f}; re-sending the patient prompt within a session "
      f"{R2['F_replicate']['determinism_patient_prompt']['mean_abs_diff']:.4f}; answering Design A's patient prompts again in Design E, "
      f"hours later, {XS.get('mean_abs_diff', float('nan')):.4f} over {XS.get('n_cells', 0):,} cells ({XS.get('mean_abs_diff_expr', float('nan')):.4f} on the "
      f"reactions where a value is shown). One expression value moves the answer by "
      f"{A['patient_vs_shuffled']['mean_move_from_reaction_arm']['patient']:.3f} on those reactions in Design A, about "
      f"{A['patient_vs_shuffled']['mean_move_from_reaction_arm']['patient'] / XS.get('mean_abs_diff_expr', 1):.0f} times the largest floor.",
      "",
      f"Positive controls on a record matched in the placement of the number: echo {100 * pc['echo']['accuracy']:.1f} %, compare {100 * pc['compare']['accuracy']:.1f} %, "
      f"threshold {100 * pc['read']['accuracy']:.1f} %, scored against the printed percentiles. On microsatellite instability, a target that "
      f"varies between patients, logistic regression on the same {M['k']}-reaction panel reaches {f4(M['logistic_auroc'])}; the model reaches "
      f"{f4(M['arms']['own']['auroc'])} with the patient's own panel and {f4(M['arms']['swap']['auroc'])} with a stranger's "
      f"(paired difference {s4(M['own_vs_swap']['auroc_diff'])}, 95 % interval {s4(M['own_vs_swap']['ci95'][0])} to {s4(M['own_vs_swap']['ci95'][1])}), "
      f"returning one probability for {100 * M['arms']['own']['modal_share']:.1f} % of patients under {M['reason_distinct']} distinct rationales.",
      ""]

# the prespecified replication of the own-versus-donor contrast (paper2/REPLICATION_PLAN_2026-11-13.md)
G = json.load(open(os.path.join(HERE, "paper2", "results", "replication_frozen.json")))
_g1 = G["wording1"]; _gd = _g1["donor"]["d1"]
_line = (f"**A prespecified replication.** The contrast was repeated on {_g1['n_patients']} patients and {_g1['n_reactions']} reactions "
         f"that no earlier collection used, under a plan committed before the first call (`paper2/REPLICATION_PLAN_2026-11-13.md`): "
         f"own {f4(_gd['own_auroc'])}, donor {f4(_gd['donor_auroc'])}, paired difference {s4(_gd['diff_mean'])} "
         f"(90 % interval {s4(_gd['diff_ci90'][0])} to {s4(_gd['diff_ci90'][1])}).")
if "schedule_spread" in _g1:
    _line += (f" Over {len(_g1['schedule_spread']['schedules'])} donor schedules the difference runs from "
              f"{s4(_g1['schedule_spread']['min'])} to {s4(_g1['schedule_spread']['max'])}.")
if "wording2" in G:
    _gw = G["wording2"]["donor"]["d1"]; _gp = G["paraphrase"]
    _line += (f" Under a second wording of the prompt, on {G['wording2']['n_patients']} of the patients, it is {s4(_gw['diff_mean'])} "
              f"({s4(_gw['diff_ci90'][0])} to {s4(_gw['diff_ci90'][1])}); rewording moves the own-arm answer by "
              f"{_gp['own_between_wordings']['mean_abs_diff']:.3f} on average and swapping the patient by "
              f"{_gp['own_vs_donor_wording1']['mean_abs_diff']:.3f}.")
# the one analysis outside the plan: the contrast of the printed numbers with no model
_xq = os.path.join(HERE, "paper2", "results", "replication_posthoc.json")
if os.path.exists(_xq):
    _xr = json.load(open(_xq))["replication"]; _xp = _xr["printed_percentile"]["d1"]["diff"]
    _xa = _xr["association"]["d1"]["model_with_printed_percentile"]["pearson"]; _xe = _xr["model_minus_printed_percentile"]["d1"]
    _line += (f" One analysis outside the plan, added after these numbers were read (`paper2/code/replication_posthoc.py`): ranking the "
              f"printed percentile with no model gives an own-minus-donor difference of {s4(_xp['mean'])} ({s4(_xp['ci90'][0])} to "
              f"{s4(_xp['ci90'][1])}) on the same patients and reactions, it correlates with the model's difference across patients at "
              f"{_xa:.2f}, and the model's difference exceeds it by {s4(_xe['mean'])} ({s4(_xe['ci90'][0])} to {s4(_xe['ci90'][1])}).")
L += [_line, ""]
# the same sample on the second and third backbones (paper2/REPLICATION_BACKBONES_PLAN_2026-11-13.md)
_bf = {t: os.path.join(HERE, "paper2", "results", f"replication_frozen_{t}.json") for t in ("b2", "b3")}
if all(os.path.exists(v) for v in _bf.values()):
    _B = {t: json.load(open(v)) for t, v in _bf.items()}
    _RF = {t: json.load(open(os.path.join(HERE, "paper2", "results", f"replication_reference_{t}.json")))["replication"]["model_minus_printed_percentile"]["d1"] for t in _B}
    _BRC = json.load(open(os.path.join(HERE, "paper2", "results", "replication", "backbones_record.json")))
    def _one(t):
        d = _B[t]["decision"]; m = _RF[t]
        return (f"{s4(d['diff_mean'])} ({s4(d['ci90'][0])} to {s4(d['ci90'][1])}; model minus printed percentile "
                f"{s4(m['mean'])}, {s4(m['ci90'][0])} to {s4(m['ci90'][1])})")
    _nz = sum(1 for t in _B if _B[t]["decision"]["interval_excludes_zero"]); _nd = sum(1 for t in _B if _RF[t]["excludes_zero"])
    L += [f"**The same sample on the second and third backbones.** After that outcome was known, the sample was put to the two other "
          f"backbones, each served locally on a second host from weights whose files match the record of their earlier collections, under a "
          f"second plan (`paper2/REPLICATION_BACKBONES_PLAN_2026-11-13.md`) that was pushed {_BRC['hours_between_push_and_first_call']:.1f} hours "
          f"before the first call and names the comparison with the printed percentile in advance. Own minus donor, primary contrast with its "
          f"90% interval: {_B['b2']['backbone']} {_one('b2')}; {_B['b3']['backbone']} {_one('b3')}. "
          f"{['Neither interval excludes zero', 'One of the two intervals excludes zero', 'Both intervals exclude zero'][_nz]}, and "
          f"{['neither difference is', 'one of the two differences is', 'both differences are'][_nd]} "
          f"distinguishable from what ranking the printed percentile gives. Each backbone is read separately and nothing is pooled.", ""]

# the generated tables live in whichever front-page file carries the markers: README.md in a
# checkout of the studies alone, STUDIES.md where the studies share a repository with the scorer
readme = next((q for q in (os.path.join(HERE, "README.md"), os.path.join(HERE, "STUDIES.md"))
               if os.path.exists(q) and "<!-- generated:start -->" in open(q).read()), None)
if readme is None:
    raise SystemExit("no front-page file carries the generated-section markers "
                     "(looked in README.md and STUDIES.md)")
txt = open(readme).read()
start, end = "<!-- generated:start -->", "<!-- generated:end -->"
if start not in txt:
    raise SystemExit("README.md lacks the generated-section markers")
new = txt[: txt.index(start) + len(start)] + "\n" + "\n".join(L) + txt[txt.index(end):]
open(readme, "w").write(new)
print(os.path.basename(readme), "regenerated:", len(L), "lines")
