#!/usr/bin/env python3
"""The second prespecified replication of the own-versus-donor contrast, on three backbones.

Plan: paper2/REPLICATION2_PLAN_2026-11-14.md, committed with this script and with the sample
(results/replication2/design_master.json) before the first call on that sample.

For each backbone whose archive results/replication2_<tag>/w1 is present, two things are computed.

1. The statistics of the first plan. replication_stats.main() is run unchanged on a temporary study
   root that holds copies of the archive, of the second sample and of the reaction list under the
   directory names that script reads; what it writes is stored as results/replication2_frozen_<tag>.json
   with four fields added at the top (backbone, archive, plan, route).

2. The printed-number reference, which this plan names in advance for all three backbones.
   replication_posthoc.one_design, unchanged, ranks each patient's reactions by the printed percentile
   and by the printed value with no model and gives the own-minus-donor difference of each, under each
   donor schedule. New here, and fixed in the plan, is the contrast pooled over the donor schedules:
   for every patient the mean over the schedules of the model's own-minus-donor difference in AUROC,
   minus the mean over the schedules of the printed percentile's. Its mean over patients, standard
   deviation, 90% Student interval and the two one-sided tests of equivalence at the margins 0.01 (the
   margin of the reading) and 0.005 (exploratory) are stored, with the reading the plan fixes, as
   results/replication2_reference_<tag>.json.

    python3 code/replication2.py                 # every backbone whose archive is present
    python3 code/replication2.py --only b1       # one backbone
    python3 code/replication2.py --first-sample  # the same quantities on the FIRST replication sample, already
                                                 # collected: results/replication2/first_sample_reference.json
    python3 code/replication2.py --selftest      # the route of point 1 and the per-schedule part of point 2 on the
                                                 # first sample must reproduce the files already published

Tags: b1 is the main backbone (Qwen3.8-27B, served as qwen3-8-27b), b2 the second (Gemma 4 31B-it) and
b3 the third (Mistral Small 3.2 24B Instruct), as in the names of the earlier archives.
"""
import argparse, json, os, shutil, sys, tempfile
import numpy as np
from scipy import stats as st

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import _paths as PATHS
import stats as S
import replication_stats as RS
import replication_posthoc as RP

REAL_ROOT = PATHS.ROOT
PLAN = 'REPLICATION2_PLAN_2026-11-14.md'
BACKBONES = dict(b1='qwen3-8-27b', b2='gemma-4-31b-it', b3='mistral-small-3.2-24b-instruct')
MASTER2 = os.path.join(REAL_ROOT, 'results', 'replication2', 'design_master.json')
MASTER1 = os.path.join(REAL_ROOT, 'results', 'replication', 'design_master.json')
FIRST = dict(b1='replication', b2='replication_b2', b3='replication_b3')       # the archives of the first sample
MARGIN, MARGIN_EXPLORATORY = 0.01, 0.005
SCHEDULES = dict(d1='shuffled', d2='shuffled_d2', d3='shuffled_d3')
R10 = RP.R10


def frozen_statistics(archive, master):
    """replication_stats.main(), unchanged, on the archive under results/<archive>/ and the given sample."""
    src = os.path.join(REAL_ROOT, 'results', archive)
    with tempfile.TemporaryDirectory() as view:
        dst = os.path.join(view, 'results', 'replication'); os.makedirs(dst)
        shutil.copy(master, os.path.join(dst, 'design_master.json'))
        for fn in ('recon3d_aligned.json.gz', 'recon3d_aligned.json'):      # the reaction list stats.analyze reads
            if os.path.exists(os.path.join(REAL_ROOT, fn)): shutil.copy(os.path.join(REAL_ROOT, fn), os.path.join(view, fn))
        for w in ('w1', 'w2'):
            if os.path.exists(os.path.join(src, w, 'raw.json')) and os.path.exists(os.path.join(src, w, 'design.json')):
                os.makedirs(os.path.join(dst, w))
                for fn in ('raw.json', 'design.json'): shutil.copy(os.path.join(src, w, fn), os.path.join(dst, w, fn))
        PATHS.ROOT = view
        try:
            RS.main()
        finally:
            PATHS.ROOT = REAL_ROOT
        return json.load(open(os.path.join(view, 'results', 'replication_frozen.json')))


def equivalence(dv):
    """Mean over patients of a paired quantity, its standard deviation and 90% Student interval, and the two
    one-sided tests at the margin of the reading and at the exploratory margin."""
    dv = np.asarray(dv, float); dv = dv[np.isfinite(dv)]; n = len(dv)
    m = float(dv.mean()); sd = float(dv.std(ddof=1)); se = sd / np.sqrt(n); q = float(st.t.ppf(0.95, n - 1))
    out = dict(n=int(n), mean=R10(m), sd=R10(sd), ci90=[R10(m - q * se), R10(m + q * se)],
               paired_p=R10(float(2 * st.t.sf(abs(m / se), n - 1))) if se > 0 else None,
               excludes_zero=bool(m - q * se > 0 or m + q * se < 0))
    for marg, tag in ((MARGIN, '001'), (MARGIN_EXPLORATORY, '0005')):
        lo = float(st.t.sf((m + marg) / se, n - 1)); hi = float(st.t.cdf((m - marg) / se, n - 1))
        out[f'tost_{tag}_p'] = [float(f'{lo:.3g}'), float(f'{hi:.3g}')]; out[f'equivalent_{tag}'] = bool(max(lo, hi) < 0.05)
    return out


def reading(e):
    """The reading that the plan fixes for the pooled contrast 'model minus printed percentile'."""
    a = ('equivalent within %.2f AUROC to what ranking the printed percentile gives' % MARGIN) if e['equivalent_001'] else \
        ('equivalence within %.2f AUROC to what ranking the printed percentile gives is not declared' % MARGIN)
    if not e['excludes_zero']:
        b = 'the 90% interval includes zero, so the two are described as not distinguishable'
    elif e['mean'] > 0:
        b = 'the 90% interval lies above zero, so the model\'s difference is described as larger than that of the printed percentile'
    else:
        b = 'the 90% interval lies below zero, so the model\'s difference is described as smaller than that of the printed percentile'
    return dict(equivalent_within_margin=bool(e['equivalent_001']), margin=MARGIN, excludes_zero=bool(e['excludes_zero']),
                sign=(0 if not e['excludes_zero'] else (1 if e['mean'] > 0 else -1)), sentence=a + '; ' + b)


def reference(archive, master_path):
    """Per schedule: replication_posthoc.one_design, unchanged. Pooled over the schedules: computed here from the same
    per-patient AUROCs (stats.auc_matrix), as the plan fixes it."""
    raw = json.load(open(os.path.join(REAL_ROOT, 'results', archive, 'w1', 'raw.json')))
    des = json.load(open(os.path.join(REAL_ROOT, 'results', archive, 'w1', 'design.json')))
    master = json.load(open(master_path))
    assert master['reactions'] == des['reactions'], 'the archive was not collected on this sample'
    pos = {p: i for i, p in enumerate(master['patients'])}; rows = [pos[p] for p in des['patients']]
    arms = {s: a for s, a in SCHEDULES.items() if a in des['arms']}
    donors = {s: np.asarray(master['donors'][s])[rows] for s in arms}
    chk = np.array([[des['swap_source'][f'{p}|{r}'] for r in des['reactions']] for p in des['patients']])
    assert (chk == donors['d1']).all(), 'donor schedule d1 of the archive differs from the sample file'
    per_schedule = RP.one_design(des['patients'], des['reactions'], donors, raw, arms)
    # pooled over the schedules
    patients = np.asarray(des['patients']); reactions = np.asarray(des['reactions'])
    y = S.LAB[reactions].astype(int); hx = S.HAS[reactions]
    Po = np.round(S.PCT[np.ix_(patients, reactions)].astype(float)); Po[:, ~hx] = 0.0
    a_pct_own = S.auc_matrix(Po, y); a_llm_own = S.auc_matrix(RP.model_matrix(raw, 'patient', patients, reactions), y)
    d_pct, d_llm = [], []
    for s in arms:
        D = donors[s]
        Pd = np.round(S.PCT[D, reactions[None, :].repeat(len(patients), 0)].astype(float)); Pd[:, ~hx] = 0.0
        d_pct.append(a_pct_own - S.auc_matrix(Pd, y))
        d_llm.append(a_llm_own - S.auc_matrix(RP.model_matrix(raw, arms[s], patients, reactions), y))
    pl = np.mean(d_llm, axis=0); pp = np.mean(d_pct, axis=0)
    # the per-schedule arrays used here must be those of one_design
    for k, s in enumerate(arms):
        assert abs(float(np.nanmean(d_llm[k])) - per_schedule['language_model'][s]['diff']['mean']) < 1e-9
        assert abs(float(np.nanmean(d_pct[k])) - per_schedule['printed_percentile'][s]['diff']['mean']) < 1e-9
    e = equivalence(pl - pp)
    ok = np.isfinite(pl) & np.isfinite(pp)
    pooled = dict(schedules=list(arms), n_patients=int(len(patients)),
                  language_model=equivalence(pl), printed_percentile=equivalence(pp), model_minus_printed_percentile=e,
                  correlation_across_patients=R10(float(np.corrcoef(pl[ok], pp[ok])[0, 1])), reading=reading(e))
    return per_schedule, pooled, dict(patients=[int(x) for x in patients], model=pl, percentile=pp)


def same(a, b, path=''):
    """Exact equality of two JSON values; returns the paths that differ."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = [f'{path}/{k}' for k in set(a) ^ set(b)]
        for k in set(a) & set(b): out += same(a[k], b[k], f'{path}/{k}')
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b): return [path]
        out = []
        for i, (x, y) in enumerate(zip(a, b)): out += same(x, y, f'{path}[{i}]')
        return out
    return [] if a == b else [path]


def fmt(d):
    return f"{d['mean']:+.4f} [{d['ci90'][0]:+.4f}, {d['ci90'][1]:+.4f}]"


def first_sample():
    """The quantities of point 2 on the first replication sample, whose collections were complete and public before the
    plan of the second replication was fixed. The plan quotes them."""
    OUT = dict(note='the printed-number reference pooled over the three donor schedules, computed on the FIRST replication '
                    'sample (results/replication, results/replication_b2, results/replication_b3) by the code that the '
                    'plan of the second replication fixes; these numbers were known when that plan was written',
               plan=PLAN, margin=MARGIN, margin_exploratory=MARGIN_EXPLORATORY, backbones={})
    for tag, arch in FIRST.items():
        _, pooled, _ = reference(arch, MASTER1)
        OUT['backbones'][tag] = dict(backbone=BACKBONES[tag], archive=f'results/{arch}', pooled=pooled)
        print(f"first sample, {tag} ({BACKBONES[tag]}): model {fmt(pooled['language_model'])}, printed percentile "
              f"{fmt(pooled['printed_percentile'])}, model minus printed percentile {fmt(pooled['model_minus_printed_percentile'])}, "
              f"sd {pooled['model_minus_printed_percentile']['sd']:.4f}")
    os.makedirs(os.path.join(REAL_ROOT, 'results', 'replication2'), exist_ok=True)
    json.dump(OUT, open(os.path.join(REAL_ROOT, 'results', 'replication2', 'first_sample_reference.json'), 'w'), indent=1)


def selftest():
    bad = 0
    for tag, arch in FIRST.items():
        got = frozen_statistics(arch, MASTER1)
        want = json.load(open(os.path.join(REAL_ROOT, 'results', 'replication_frozen.json' if tag == 'b1' else f'replication_frozen_{tag}.json')))
        want = {k: v for k, v in want.items() if k not in ('backbone', 'archive', 'plan', 'route')}
        d1 = same(got, want)
        per, _, _ = reference(arch, MASTER1)
        ref = json.load(open(os.path.join(REAL_ROOT, 'results', 'replication_posthoc.json' if tag == 'b1' else f'replication_reference_{tag}.json')))['replication']
        d2 = same(json.loads(json.dumps(per)), ref)
        print(f'selftest {tag}: frozen statistics', 'identical to the published file' if not d1 else f'DIFFERENT at {d1[:6]}',
              '| per-schedule reference', 'identical to the published file' if not d2 else f'DIFFERENT at {d2[:6]}')
        bad += bool(d1) + bool(d2)
    sys.exit(1 if bad else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true'); ap.add_argument('--first-sample', action='store_true')
    ap.add_argument('--only', choices=sorted(BACKBONES), default=None)
    a = ap.parse_args()
    if a.selftest: selftest()
    if a.first_sample: first_sample(); return
    done = 0
    for tag, name in BACKBONES.items():
        if a.only and tag != a.only: continue
        archive = f'replication2_{tag}'
        if not os.path.exists(os.path.join(REAL_ROOT, 'results', archive, 'w1', 'raw.json')):
            print(f'{tag}: no archive under results/{archive}/w1, skipped'); continue
        F = frozen_statistics(archive, MASTER2)
        if F['wording1'].get('model') != name: sys.exit(f"{tag}: the archive names the model {F['wording1'].get('model')!r}, expected {name!r}")
        head = dict(backbone=name, archive=f'results/{archive}', plan=PLAN,
                    route='replication_stats.main(), unchanged, on a copy of the archive and of the second sample placed under the '
                          'directory names it reads; "replication/w1" below stands for this archive')
        json.dump({**head, **F}, open(os.path.join(REAL_ROOT, 'results', f'replication2_frozen_{tag}.json'), 'w'), indent=1)
        per, pooled, arr = reference(archive, MASTER2)
        R = dict(backbone=name, archive=f'results/{archive}', plan=PLAN, margin=MARGIN, margin_exploratory=MARGIN_EXPLORATORY,
                 note='the own-versus-donor contrast of the printed percentile and of the printed value with no model, beside the '
                      'contrast of the language model: per donor schedule (replication_posthoc.one_design, unchanged) and pooled '
                      'over the donor schedules; named in the plan before the first call on this sample',
                 pooled=pooled, per_schedule=per)
        # descriptive: the patients of both replication samples together (the two samples share no patient)
        if os.path.exists(os.path.join(REAL_ROOT, 'results', FIRST[tag], 'w1', 'raw.json')):
            _, _, arr1 = reference(FIRST[tag], MASTER1)
            assert not set(arr1['patients']) & set(arr['patients'])
            both = np.concatenate([arr1['model'] - arr1['percentile'], arr['model'] - arr['percentile']])
            R['both_samples'] = dict(note='mean of "model minus printed percentile" (mean of the three schedules) over the patients of '
                                          'the first and of the second replication sample together; descriptive, no decision rests on it',
                                     n_first=int(len(arr1['patients'])), n_second=int(len(arr['patients'])),
                                     model_minus_printed_percentile=equivalence(both))
        json.dump(R, open(os.path.join(REAL_ROOT, 'results', f'replication2_reference_{tag}.json'), 'w'), indent=1)
        d = F['decision']
        print(f"{tag} ({name}): own minus donor (d1) {d['diff_mean']:+.4f} [{d['ci90'][0]:+.4f}, {d['ci90'][1]:+.4f}], n {d['n_patients']}, "
              f"within 0.02: {d['equivalent_002']}, within 0.05: {d['equivalent_005']}, excludes zero: {d['interval_excludes_zero']}; "
              f"pooled model minus printed percentile {fmt(pooled['model_minus_printed_percentile'])}: {pooled['reading']['sentence']}")
        done += 1
    print(f'{done} backbone(s) written')


if __name__ == '__main__':
    main()
