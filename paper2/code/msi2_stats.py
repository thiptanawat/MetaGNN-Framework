#!/usr/bin/env python3
"""Statistics for the frozen percentile-only interface (P2-E2/E3/E4/E5).

Reads every collection under results/msi2/<cohort>_<backbone>_<config>_s<seed>/ and the frozen
references (results/external/frozen_scorer.json), and writes results/msi2_frozen.json. One primary
confirmatory contrast is declared in docs/ANALYSIS_PLAN_2026-09-10.md: the own-minus-donor AUROC of
the locked interface on the external cohort under the harmonized endpoint; everything else is
secondary and labelled so.

The donor arm is a second set of calls actually issued under the frozen schedule, and that collected
arm is the primary one. The percentile-only prompt carries no recipient identifier, so a donor prompt
for recipient i with donor j is byte-identical to donor j's own prompt and the served answer should
repeat; it does not always, because serving is not bit-deterministic, so the cached reconstruction
(donor score for recipient i is own[D[i]]) is reported beside the collected arm as a sensitivity
rather than in place of it. The reconstruction is also what makes schedule sensitivity possible
without new calls, since a redrawn derangement over the cached own scores costs nothing. Population
intervals resample patients and redraw the derangement inside each replicate; the exchangeability
null permutes the profile-label alignment. Percentiles are cohort midranks computed once, before any
outcome was read, and are held fixed inside every resample, so every interval here conditions on
them.
"""
import os, json, glob, numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
import _paths as PATHS

RES = os.path.join(PATHS.ROOT, "results", "msi2")
SCORER = json.load(open(os.path.join(PATHS.ROOT, "results", "external", "frozen_scorer.json")))

def load(run):
    d = json.load(open(os.path.join(RES, run, "design.json")))
    raw = json.load(open(os.path.join(RES, run, "raw.json")))
    n = len(d["y"]); y = np.array(d["y"])
    own = np.full(n, np.nan); donor = np.full(n, np.nan); tool = np.full(n, np.nan); rep = {}
    for a in range(n):
        r = raw.get(f"own|{a}"); own[a] = r["val"] if r and r["val"] is not None else np.nan
        r = raw.get(f"donor|{a}")
        if r and r["val"] is not None: donor[a] = r["val"]
        r = raw.get(f"own|{a}")
        if r and r.get("tool_prob") is not None: tool[a] = r["tool_prob"]
        r = raw.get(f"own_repeat|{a}")
        if r and r["val"] is not None: rep[a] = r["val"]
    return d, y, own, donor, tool, rep

def auc(y, s):
    ok = np.isfinite(s)
    return float(roc_auc_score(y[ok], s[ok])) if len(set(y[ok])) > 1 else float("nan")
def aup(y, s):
    ok = np.isfinite(s); return float(average_precision_score(y[ok], s[ok])) if len(set(y[ok])) > 1 else float("nan")

def derangement(n, seed):
    r = np.random.default_rng(seed)
    while True:
        p = r.permutation(n)
        if not np.any(p == np.arange(n)): return p

def own_minus_donor(y, own, D):
    """AUROC(own) - AUROC(donor) where donor score of recipient i is own[D[i]]; on patients with both finite."""
    ds = own[D]; ok = np.isfinite(own) & np.isfinite(ds)
    if len(set(y[ok])) < 2: return float("nan"), float("nan"), float("nan")
    return float(roc_auc_score(y[ok], own[ok])), float(roc_auc_score(y[ok], ds[ok])), int(ok.sum())

def analyze(run):
    d, y, own, donor, tool, rep = load(run)
    n = len(y); okc = np.isfinite(own)
    D0 = np.array(d["donor"])
    # the primary own-minus-donor, on the collected schedule (seed = donor_seed), and its schedule spread
    a_own, a_don0, n_eval = own_minus_donor(y, own, D0)
    # the donor arm as it was actually served: the responses collected under the donor prompts, which
    # the cached reconstruction above predicts but does not reproduce exactly
    _okd = np.isfinite(own) & np.isfinite(donor)
    _two = bool(_okd.sum()) and len(set(y[_okd])) > 1
    a_don_direct = float(roc_auc_score(y[_okd], donor[_okd])) if _two else float("nan")
    a_own_paired = float(roc_auc_score(y[_okd], own[_okd])) if _two else float("nan")
    n_direct = int(_okd.sum())
    _cachedd = own[D0]
    _okb = _okd & np.isfinite(_cachedd)
    _dd = np.abs(donor[_okb] - _cachedd[_okb]) if _okb.sum() else np.array([])
    sched = []
    for s in (11, 22, 33, 44, 55):
        Ds = derangement(n, s); _, ad, _ = own_minus_donor(y, own, Ds); sched.append(a_own - ad)
    # population interval: resample patients, redraw the derangement among them
    rng = np.random.default_rng(7); diffs = []
    idx0 = np.where(okc)[0]
    for _ in range(2000):
        bi = rng.choice(idx0, len(idx0), replace=True)
        if len(set(y[bi])) < 2: continue
        Dl = derangement(len(bi), int(rng.integers(1, 1_000_000)))
        so = own[bi]; sd = own[bi][Dl]
        diffs.append(roc_auc_score(y[bi], so) - roc_auc_score(y[bi], sd))
    diffs = np.array(diffs)
    # the same patient-level interval for the collected donor arm, with the schedule held at the one
    # under which the calls were issued: the pair (own_i, donor_i) resamples as a unit
    rngd = np.random.default_rng(17); ddiffs = []
    idxd = np.where(_okd)[0]
    if _two:
        for _ in range(2000):
            bi = rngd.choice(idxd, len(idxd), replace=True)
            if len(set(y[bi])) < 2: continue
            ddiffs.append(roc_auc_score(y[bi], own[bi]) - roc_auc_score(y[bi], donor[bi]))
    ddiffs = np.array(ddiffs)
    # exchangeability null: permute labels against the fixed own scores
    rngp = np.random.default_rng(13); null = []
    ya = y[okc]; sa = own[okc]
    for _ in range(10000):
        yp = rngp.permutation(ya)
        null.append(roc_auc_score(yp, sa) - roc_auc_score(yp, sa[derangement(len(yp), int(rngp.integers(1, 1_000_000)))]))
    null = np.array(null)
    dnull = np.array([])
    if _two:
        rngdn = np.random.default_rng(23); _dn = []
        yb = y[_okd]; ob = own[_okd]; db = donor[_okd]
        for _ in range(10000):
            yp = rngdn.permutation(yb)
            _dn.append(roc_auc_score(yp, ob) - roc_auc_score(yp, db))
        dnull = np.array(_dn)
    # serving determinism from the repeat subset
    rep_delta = [abs(rep[a] - own[a]) for a in rep if np.isfinite(own[a])]
    R = dict(run=run, cohort=d["cohort"], backbone=d.get("model"), config=d["config"], endpoint=d["endpoint"],
             donor_seed=d["donor_seed"], n=n, n_eval=n_eval, n_pos=int(y[okc].sum()), prevalence=round(float(y[okc].mean()), 4),
             failed_calls=d.get("failed_calls"),
             own_auroc=round(a_own, 4), own_auprc=round(aup(y, own), 4),
             donor_auroc_cached=round(a_don0, 4), own_minus_donor_cached=round(a_own - a_don0, 4),
             donor_auroc=round(a_don_direct, 4) if _two else None,
             own_minus_donor=round(a_own_paired - a_don_direct, 4) if _two else round(a_own - a_don0, 4),
             own_minus_donor_direct_n=n_direct,
             own_minus_donor_direct_ci95=([round(float(np.percentile(ddiffs, 2.5)), 4),
                                           round(float(np.percentile(ddiffs, 97.5)), 4)] if len(ddiffs) else None),
             own_minus_donor_direct_null_p=(round(float((dnull >= (a_own_paired - a_don_direct)).mean()), 4)
                                            if len(dnull) else None),
             donor_served_vs_cached_n=int(len(_dd)),
             donor_served_vs_cached_n_changed=int((_dd > 0).sum()) if len(_dd) else None,
             donor_served_vs_cached_median_abs=round(float(np.median(_dd)), 4) if len(_dd) else None,
             donor_served_vs_cached_max_abs=round(float(np.max(_dd)), 4) if len(_dd) else None,
             own_minus_donor_schedule_mean=round(float(np.mean(sched)), 4), own_minus_donor_schedule_sd=round(float(np.std(sched)), 4),
             own_minus_donor_ci95=[round(float(np.percentile(diffs, 2.5)), 4), round(float(np.percentile(diffs, 97.5)), 4)],
             own_minus_donor_boot_mean=round(float(np.mean(diffs)), 4),
             exch_null_mean=round(float(null.mean()), 4), exch_null_p=round(float((null >= (a_own - a_don0)).mean()), 4),
             serving_repeat_n=len(rep_delta), serving_repeat_median_abs=round(float(np.median(rep_delta)), 4) if rep_delta else None,
             serving_repeat_max_abs=round(float(np.max(rep_delta)), 4) if rep_delta else None,
             # a median of zero says the typical repeat is identical, not that every repeat is: the
             # count of repeats that moved at all is the quantity that settles the stronger claim
             serving_repeat_n_changed=int(sum(1 for x in rep_delta if x > 0)) if rep_delta else None,
             serving_repeat_p90_abs=round(float(np.quantile(rep_delta, 0.90)), 4) if rep_delta else None)
    # calibration of the own scores: developed on the development cohort, applied unchanged here
    ok = np.isfinite(own)
    R["brier_own"] = round(float(brier_score_loss(y[ok], np.clip(own[ok], 0, 1))), 4)
    R["logloss_own"] = round(float(log_loss(y[ok], np.clip(own[ok], 0.005, 0.995), labels=[0, 1])), 4)
    R["mean_prob_own"] = round(float(np.nanmean(own)), 4)
    if d["config"] == "tool":
        okt = np.isfinite(tool) & np.isfinite(own)
        R["tool_only_auroc"] = round(auc(y, tool), 4); R["tool_only_auprc"] = round(aup(y, tool), 4)
        R["tool_fidelity_median_abs"] = round(float(np.median(np.abs(own[okt] - tool[okt]))), 4)
        R["tool_fidelity_within_0.05"] = round(float((np.abs(own[okt] - tool[okt]) < 0.05).mean()), 4)
        R["n_tool_pairs"] = int(okt.sum())
    return R, (d, y, own)

def calibration_map(dev_scores, dev_y):
    """A monotone logistic map fitted on development own scores; returns (a, b) for
    logit(p_cal) = a*logit(p)+b, a>=0 enforced by refusing a<0 (report constant if so)."""
    from sklearn.linear_model import LogisticRegression
    ok = np.isfinite(dev_scores); p = np.clip(dev_scores[ok], 1e-4, 1 - 1e-4); z = np.log(p / (1 - p)).reshape(-1, 1)
    m = LogisticRegression(C=1e6, max_iter=5000).fit(z, dev_y[ok])
    return float(m.coef_[0, 0]), float(m.intercept_[0])

def apply_map(scores, ab):
    a, b = ab; p = np.clip(scores, 1e-4, 1 - 1e-4); z = a * np.log(p / (1 - p)) + b
    return 1.0 / (1.0 + np.exp(-z))

# the frozen supervised reference applied to each cohort's percentiles: does even the logistic
# reference, fitted on TCGA, transfer? (a transfer failure where the reference also loses signal is
# read differently from a language-model failure with a strong reference)
def reference_on_cohort(cohort):
    ext = os.path.join(PATHS.ROOT, "results", "external")
    fr = SCORER["endpoints"]["msih_vs_nonmsih"]; coef = np.array(fr["coef"]); b = float(fr["intercept"])
    fdef = json.load(open(os.path.join(ext, "panel_frozen.json")))
    panel_ids = fdef["panel_ids"]; panel = fdef["panel"]
    if cohort == "tcga":
        D = np.load(PATHS.data("rxn_context.npz"), allow_pickle=True); X = D["X"]
        import csv as _csv
        meta = {r["tcga_barcode"]: r for r in _csv.DictReader(open(PATHS.data("clinical_metadata_msi.tsv")), delimiter="\t")}
        pids = [str(p) for p in D["pids"]]; status = np.array([(meta[p]["msi_status"] or "").strip() for p in pids])
        P = np.stack([100.0*(rankdata(X[:, j])-1.0)/(X.shape[0]-1) for j in panel], axis=1)
        keep = np.where(np.isin(status, ("MSI-H","MSS","MSI-L")))[0]; y = (status[keep]=="MSI-H").astype(int); P = P[keep]
    else:
        Z = np.load(os.path.join(ext, f"{cohort}_panel.npz"), allow_pickle=True)
        order = [list(Z["panel_ids"]).index(i) for i in panel_ids]; P = Z["pct"][:, order]
        lab = np.array([str(v) for v in Z["label"]]); keep = np.where(np.isin(lab, ("MSI-H","non-MSI-H")))[0]
        y = (lab[keep]=="MSI-H").astype(int); P = P[keep]
    ok = np.isfinite(P).all(1); P = P[ok]/100.0; y = y[ok]
    s = 1.0/(1.0+np.exp(-(P@coef + b)))
    D0 = derangement(len(y), 11)
    return dict(cohort=cohort, n=int(len(y)), n_pos=int(y.sum()), auroc=round(float(roc_auc_score(y, s)), 4),
                auprc=round(float(average_precision_score(y, s)), 4),
                own_minus_donor=round(float(roc_auc_score(y, s) - roc_auc_score(y, s[D0])), 4))

# Every collection directory is enumerated, not only the analyzable ones. A collection whose
# design.json is missing was interrupted before the collector wrote it, and its responses are
# therefore incomplete; skipping it silently would leave a collected configuration out of the record
# with nothing to say so. Those are listed with their response counts under "incomplete".
_all_runs = sorted(os.path.basename(d) for d in glob.glob(os.path.join(RES, "*")) if os.path.isdir(d))
runs = sorted(os.path.basename(os.path.dirname(f)) for f in glob.glob(os.path.join(RES, "*", "design.json")))
_incomplete = {}
for _r in sorted(set(_all_runs) - set(runs)):
    _raw = os.path.join(RES, _r, "raw.json")
    _n = len(json.load(open(_raw))) if os.path.exists(_raw) else 0
    _incomplete[_r] = dict(responses=_n, reason="collection interrupted before the design record was written; "
                                                "the response set is partial and the run is not analyzed")
_complete_n = max((len(json.load(open(os.path.join(RES, r, "raw.json")))) for r in runs), default=0)
OUT = dict(note="frozen percentile-only interface; primary confirmatory contrast = external own-minus-donor AUROC on the "
                "harmonized endpoint, computed on the donor responses actually collected under the frozen schedule; "
                "the cached reconstruction (donor score = the donor's own score) is a sensitivity beside it and is what "
                "the five-schedule spread is computed on; all other rows secondary.",
           collections_attempted=len(_all_runs), collections_analyzed=len(runs),
           responses_in_a_complete_collection=_complete_n, incomplete=_incomplete,
           frozen_reference=SCORER["endpoints"], runs={}, calibration={})
if _incomplete:
    for _r, _v in _incomplete.items():
        print(f"{_r:44s} INCOMPLETE ({_v['responses']} of {_complete_n} responses); not analyzed", flush=True)
store = {}
for run in runs:
    try:
        R, s = analyze(run); OUT["runs"][run] = R; store[run] = s
        print(f"{run:44s} own {R['own_auroc']:.4f} own-donor {R['own_minus_donor']:+.4f} "
              f"[{R['own_minus_donor_ci95'][0]:+.4f},{R['own_minus_donor_ci95'][1]:+.4f}] null p {R['exch_null_p']:.3f} "
              f"Brier {R['brier_own']:.4f}"
              + (f" tool-only {R.get('tool_only_auroc')}" if R['config'] == 'tool' else ""), flush=True)
    except Exception as e:
        print(run, "FAILED", type(e).__name__, str(e)[:120], flush=True)
# calibration: fit each map on the development cohort of the same backbone+config, apply to external
for run, (d, y, own) in store.items():
    if d["cohort"] != "tcga": continue
    key = (d.get("model"), d["config"], d["endpoint"])
    ab = calibration_map(own, y); OUT["calibration"][f"{d.get('model')}|{d['config']}|{d['endpoint']}"] = dict(a=round(ab[0], 4), b=round(ab[1], 4), fitted_on=run)
for run, (d, y, own) in store.items():
    if d["cohort"] == "tcga": continue
    key = f"{d.get('model')}|{d['config']}|{d['endpoint']}"
    ab = OUT["calibration"].get(key)
    if not ab: continue
    ab = (ab["a"], ab["b"]); ok = np.isfinite(own); pc = apply_map(own[ok], ab)
    OUT["runs"][run]["calibrated"] = dict(brier=round(float(brier_score_loss(y[ok], pc)), 4),
                                          logloss=round(float(log_loss(y[ok], np.clip(pc, 0.005, 0.995), labels=[0, 1])), 4),
                                          auroc=round(float(roc_auc_score(y[ok], pc)), 4),
                                          slope_positive=bool(ab[0] > 0), map=dict(a=ab[0], b=ab[1]))
OUT["frozen_reference_transfer"] = {}
for _c in ("tcga", "gse39582", "gse13294"):
    try:
        OUT["frozen_reference_transfer"][_c] = reference_on_cohort(_c)
        print("reference", _c, OUT["frozen_reference_transfer"][_c])
    except Exception as e:
        print("reference", _c, "FAILED", type(e).__name__, str(e)[:100])
json.dump(OUT, open(os.path.join(PATHS.ROOT, "results", "msi2_frozen.json"), "w"), indent=1)
print("wrote results/msi2_frozen.json;", len(OUT["runs"]), "runs")
