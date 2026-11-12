#!/usr/bin/env python3
"""Statistics of the prespecified replication of the own-versus-donor contrast (plan of 13 November 2026).

Reads results/replication/w1 (wording 1) and, when present, results/replication/w2 (wording 2), and
writes results/replication_frozen.json. Every contrast between the own arm and a donor arm is computed
by stats.analyze, the function that computed the same contrast for the earlier designs: this script
only lays each donor schedule out as a run with the arm keys that function reads. The quantities
that the earlier designs did not have (three donor schedules in one session, the pooled donor
contrast, the comparison of two wordings) are computed here from the same per-patient AUROCs.

The analysis set is the set of patients in design.json, that is, the blocks that were completed.
Nothing is imputed: an unparsed answer is missing in its cell, as in stats.py.
"""
import os, sys, json, hashlib, tempfile
import numpy as np
from scipy import stats as st

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import stats as S
import _paths as PATHS

RUNS = {1: 'replication/w1', 2: 'replication/w2'}
DONOR_ARMS = {'d1': 'shuffled', 'd2': 'shuffled_d2', 'd3': 'shuffled_d3'}
R10 = lambda x: None if x is None or not np.isfinite(x) else round(float(x), 10)


def read_run(w):
    try:
        raw = json.load(open(PATHS.run(RUNS[w], 'raw.json'))); des = json.load(open(PATHS.run(RUNS[w], 'design.json')))
    except FileNotFoundError:
        return None, None
    return raw, des


def matrix(raw, arm, patients, reactions):
    M = np.full((len(patients), len(reactions)), np.nan)
    for a, pj in enumerate(patients):
        for b, ri in enumerate(reactions):
            v = raw.get(f'{arm}|{pj}|{ri}')
            if v and v['prob'] is not None: M[a, b] = v['prob']
    return M


def write_view(tmp, name, raw, des, patients, arms_map):
    """A run directory with the given patients, whose arms carry the keys stats.analyze reads."""
    out = {}
    for view_key, src_key in arms_map.items():
        for pj in patients:
            for ri in des['reactions']:
                rec = raw.get(f'{src_key}|{pj}|{ri}')
                if rec is not None: out[f'{view_key}|{pj}|{ri}'] = rec
    d = {k: v for k, v in des.items() if k not in ('swap_source', 'blocks')}
    d.update(patients=list(patients), arms=list(arms_map), n=len(out))
    os.makedirs(os.path.join(tmp, name))
    json.dump(out, open(os.path.join(tmp, name, 'raw.json'), 'w')); json.dump(d, open(os.path.join(tmp, name, 'design.json'), 'w'))


def paired(dv):
    """Mean paired difference over patients with the 90% Student interval and the two one-sided tests
    at the margins 0.05 and 0.02, as stats.analyze computes them."""
    dv = np.asarray(dv, float); ok = np.isfinite(dv); n = int(ok.sum()); m = float(dv[ok].mean())
    se = float(dv[ok].std(ddof=1) / np.sqrt(n)); q = float(st.t.ppf(0.95, n - 1))
    def tost(marg):
        return float(st.t.sf((m + marg) / se, n - 1)), float(st.t.cdf((m - marg) / se, n - 1))
    out = dict(n=n, mean=R10(m), sd=R10(dv[ok].std(ddof=1)), ci90=[R10(m - q * se), R10(m + q * se)],
               paired_p=float(f'{st.ttest_1samp(dv[ok], 0.0).pvalue:.3g}'))
    for marg, tag in ((0.05, '005'), (0.02, '002'), (0.01, '001')):
        lo, hi = tost(marg)
        out[f'tost_{tag}_p'] = [float(f'{lo:.3g}'), float(f'{hi:.3g}')]; out[f'equivalent_{tag}'] = bool(max(lo, hi) < 0.05)
    return out


def crossed(Mo, Md, y, B=10000, seeds=(29, 1029, 2029)):
    """Patients and reactions resampled together, arms paired within cells (crossed_boot_10k.py)."""
    per = {}
    for seed in seeds:
        rng = np.random.default_rng(seed); cd = []
        for _ in range(B):
            ix = rng.integers(0, Mo.shape[0], Mo.shape[0]); jx = rng.integers(0, Mo.shape[1], Mo.shape[1]); yb = y[jx]
            if len(set(yb.tolist())) < 2: continue
            cd.append(float(np.nanmean(S.auc_matrix(Mo[np.ix_(ix, jx)], yb) - S.auc_matrix(Md[np.ix_(ix, jx)], yb))))
        cd = np.array(cd)
        per[str(seed)] = dict(n_draws=int(len(cd)), mean=R10(cd.mean()),
                              ci90=[R10(np.percentile(cd, 5)), R10(np.percentile(cd, 95))],
                              ci95=[R10(np.percentile(cd, 2.5)), R10(np.percentile(cd, 97.5))])
    lo = [v['ci90'][0] for v in per.values()]; hi = [v['ci90'][1] for v in per.values()]
    return dict(B=B, seeds=list(seeds), per_seed=per, ci90_lower_range=[min(lo), max(lo)], ci90_upper_range=[min(hi), max(hi)],
                within_002_90_all_seeds=bool(min(lo) > -0.02 and max(hi) < 0.02),
                within_005_90_all_seeds=bool(min(lo) > -0.05 and max(hi) < 0.05))


def keep(R, schedule):
    """The fields of stats.analyze that the replication reports, for one donor schedule."""
    pv = R['patient_vs_shuffled']
    return dict(schedule=schedule, n_patients=R['n_patients'], own_auroc=R['per_arm']['patient']['auroc'],
                own_auroc_ci=R['per_arm']['patient']['auroc_ci'], donor_auroc=R['per_arm']['shuffled']['auroc'],
                donor_auroc_ci=R['per_arm']['shuffled']['auroc_ci'], parsed_own=R['per_arm']['patient']['parsed'],
                parsed_donor=R['per_arm']['shuffled']['parsed'],
                diff_mean=pv['auroc_diff_mean'], diff_sd=pv['auroc_diff_sd'], diff_ci90=pv['auroc_diff_ci90'],
                paired_p=pv['paired_p'], wilcoxon_p=pv['wilcoxon_p'],
                tost_005_p=[pv['tost_p_lower'], pv['tost_p_upper']], equivalent_005=pv['tost_equivalent'],
                tost_002_p=[pv['tost_narrow_p_lower'], pv['tost_narrow_p_upper']], equivalent_002=pv['tost_narrow_equivalent'],
                n_pairs=pv['n_pairs'], identical=pv['identical'], mean_abs_diff=pv['mean_abs_diff'],
                arm_correlation=pv['arm_correlation'], residual_correlation=pv['residual_correlation'],
                residual_correlation_ci=pv['residual_correlation_ci'],
                between_patient_sd_own=pv['between_patient_sd_real'], between_patient_sd_donor=pv['between_patient_sd_stranger'])


def wording_block(w, tmp):
    raw, des = read_run(w)
    if raw is None: return None, None, None
    pats = [int(x) for x in des['patients']]; rx = [int(x) for x in des['reactions']]
    y = S.LAB[np.array(rx)].astype(int); hx = S.HAS[np.array(rx)]
    out = dict(run=RUNS[w], wording=w, model=des.get('model'), n_patients=len(pats), n_reactions=len(rx),
               n_expression_bearing=int(hx.sum()), label_base_rate=R10(y.mean()), blocks_completed=len(des.get('blocks', [])),
               n_calls=int(des.get('n', 0)), resent_calls=int(des.get('resent_calls', 0)),
               transport_errors_left=int(sum(b.get('transport_errors_left', 0) for b in des.get('blocks', []))),
               unparsed=int(sum(b.get('unparsed', 0) for b in des.get('blocks', []))),
               minutes=des.get('minutes'), first_call=min((b['started_at'] for b in des.get('blocks', [])), default=None),
               last_block_finished=max((b['finished_at'] for b in des.get('blocks', [])), default=None),
               patients=pats, design_master_sha256=des.get('design_master_sha256'), code_sha256=des.get('code_sha256'))
    arms = des['arms']; scheds = [s for s, a in DONOR_ARMS.items() if a in arms]
    M = {a: matrix(raw, a, pats, rx) for a in arms}
    auc = {a: S.auc_matrix(M[a], y) for a in arms}
    out['donor'] = {}
    for s in scheds:
        name = f'_view_w{w}_{s}'
        write_view(tmp, name, raw, des, pats, {'patient': 'patient', 'shuffled': DONOR_ARMS[s]})
        Rv = S.analyze(name, f'G_w{w}_{s}')
        out['donor'][s] = keep(Rv, s)
        if s == 'd1':
            out['raw_expression_auroc'] = Rv['raw_expression_auroc']; out['indicator_floor'] = Rv['indicator_floor']
            out['raw_percentile_auroc'] = Rv['raw_percentile_auroc']
    if len(scheds) > 1:
        means = [out['donor'][s]['diff_mean'] for s in scheds]
        out['schedule_spread'] = dict(schedules=scheds, diff_means=means, min=min(means), max=max(means))
        pooled = auc['patient'] - np.mean([auc[DONOR_ARMS[s]] for s in scheds], axis=0)
        out['pooled_donor'] = paired(pooled)
    out['crossed_bootstrap_d1'] = crossed(M['patient'], M['shuffled'], y)
    # the patients for whom the patient-blind arm was collected: the determinism floor, and what a
    # number is worth against no number
    if 'reaction' in arms:
        bl = [pj for a, pj in enumerate(pats) if np.isfinite(M['reaction'][a]).any()]
        name = f'_view_w{w}_blind'
        write_view(tmp, name, raw, des, bl, {'reaction': 'reaction', 'patient': 'patient', 'shuffled': 'shuffled'})
        Rb = S.analyze(name, f'G_w{w}_blind')
        out['blind_subset'] = dict(n_patients=len(bl), blind_auroc=Rb['per_arm']['reaction']['auroc'],
                                   own_auroc=Rb['per_arm']['patient']['auroc'], donor_auroc=Rb['per_arm']['shuffled']['auroc'],
                                   determinism=Rb.get('determinism'), arms_vs_blind=Rb.get('arms_vs_blind'),
                                   mean_move_from_blind=Rb['patient_vs_shuffled']['mean_move_from_reaction_arm'])
    return out, (raw, des, pats, rx, y, hx, M, auc), scheds


def main():
    OUT = dict(note='prespecified replication of the own-versus-donor contrast on reactions and patients that no earlier '
                    'collection used; see REPLICATION_PLAN_2026-11-13.md',
               margins=dict(narrow=0.02, wide=0.05), primary='wording 1, donor schedule d1, all completed blocks')
    master = PATHS.run('replication', 'design_master.json')
    OUT['design_master_sha256'] = hashlib.sha256(open(master, 'rb').read()).hexdigest()
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            W1, K1, _ = wording_block(1, tmp)
            if W1 is None: sys.exit('results/replication/w1 not found')
            OUT['wording1'] = W1
            W2, K2, _ = wording_block(2, tmp)
            if W2 is not None:
                OUT['wording2'] = W2
                raw1, des1, p1, rx, y, hx, M1, auc1 = K1; raw2, des2, p2, _, _, _, M2, auc2 = K2
                common = [pj for pj in p2 if pj in set(p1)]
                i1 = [p1.index(pj) for pj in common]; i2 = [p2.index(pj) for pj in common]
                def move(A_, B_, mask):
                    m = np.isfinite(A_) & np.isfinite(B_) & mask[None, :]
                    return dict(n_pairs=int(m.sum()), mean_abs_diff=R10(np.abs(A_[m] - B_[m]).mean()),
                                identical=R10((A_[m] == B_[m]).mean()), correlation=R10(np.corrcoef(A_[m], B_[m])[0, 1]))
                allr = np.ones(len(rx), bool)
                P = dict(n_patients=len(common),
                         own_between_wordings=move(M1['patient'][i1], M2['patient'][i2], hx),
                         donor_between_wordings=move(M1['shuffled'][i1], M2['shuffled'][i2], hx),
                         own_vs_donor_wording1=move(M1['patient'][i1], M1['shuffled'][i1], hx),
                         own_vs_donor_wording2=move(M2['patient'][i2], M2['shuffled'][i2], hx),
                         own_auroc_wording1=R10(np.nanmean(auc1['patient'][i1])), own_auroc_wording2=R10(np.nanmean(auc2['patient'][i2])),
                         own_auroc_wording2_minus_wording1=paired(auc2['patient'][i2] - auc1['patient'][i1]),
                         own_minus_donor_wording1_same_patients=paired(auc1['patient'][i1] - auc1['shuffled'][i1]),
                         own_minus_donor_wording2=paired(auc2['patient'][i2] - auc2['shuffled'][i2]))
                if 'reaction' in M1 and 'reaction' in M2:
                    b = [k for k in range(len(common)) if np.isfinite(M1['reaction'][i1[k]]).any() and np.isfinite(M2['reaction'][i2[k]]).any()]
                    if b:
                        P['blind_between_wordings'] = move(M1['reaction'][[i1[k] for k in b]], M2['reaction'][[i2[k] for k in b]], allr)
                        P['blind_between_wordings']['n_patients'] = len(b)
                sw = P['own_vs_donor_wording1']['mean_abs_diff']; pw = P['own_between_wordings']['mean_abs_diff']
                P['paraphrase_over_swap'] = R10(pw / sw) if sw else None
                OUT['paraphrase'] = P
        finally:
            os.chdir(cwd)
    d1 = OUT['wording1']['donor']['d1']
    OUT['decision'] = dict(diff_mean=d1['diff_mean'], ci90=d1['diff_ci90'], equivalent_002=d1['equivalent_002'],
                           equivalent_005=d1['equivalent_005'],
                           interval_excludes_zero=bool(d1['diff_ci90'][0] > 0 or d1['diff_ci90'][1] < 0),
                           n_patients=d1['n_patients'])
    dst = os.path.join(PATHS.ROOT, 'results', 'replication_frozen.json')
    json.dump(OUT, open(dst, 'w'), indent=1)
    print('wrote', dst); print(json.dumps(OUT['decision'], indent=1))


if __name__ == '__main__':
    main()
