#!/usr/bin/env python3
"""The replication of the own-versus-donor contrast on the two further backbones.

Plan: paper2/REPLICATION_BACKBONES_PLAN_2026-11-13.md, committed before the first call to either
backbone on the replication sample.

Nothing is computed here that the replication on the main backbone did not compute. For each backbone
the archive results/replication_<tag>/{w1,w2} is presented to replication_stats.py under the directory
names that script reads (a temporary study root that holds copies of the archive, of the sample and of
the reaction list), and replication_stats.main() is run unchanged; what it writes is stored as
results/replication_frozen_<tag>.json with three fields added at the top (backbone, archive, plan).
The printed-number reference is computed by replication_posthoc.one_design, unchanged, on the same
archive and stored as results/replication_reference_<tag>.json. For the main backbone that analysis
was added after its statistics had been read; for the two further backbones the plan names it in
advance.

    python3 code/replication_backbones.py            # every backbone whose archive is present
    python3 code/replication_backbones.py --selftest # the same route on the main backbone's archive;
                                                     # must reproduce results/replication_frozen.json and
                                                     # the replication block of results/replication_posthoc.json

Tags: b2 is the second backbone (Gemma 4 31B-it), b3 the third (Mistral Small 3.2 24B Instruct), as in
the names of the earlier archives (repr_b2il, repr_b3il).
"""
import argparse, json, os, shutil, sys, tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import _paths as PATHS
import replication_stats as RS          # imports stats, which loads the cohort arrays from the study root
import replication_posthoc as RP

REAL_ROOT = PATHS.ROOT
PLAN = 'REPLICATION_BACKBONES_PLAN_2026-11-13.md'
BACKBONES = dict(b2='gemma-4-31b-it', b3='mistral-small-3.2-24b-instruct')
MASTER = os.path.join(REAL_ROOT, 'results', 'replication', 'design_master.json')


def frozen_statistics(archive):
    """replication_stats.main(), unchanged, on the archive under results/<archive>/."""
    src = os.path.join(REAL_ROOT, 'results', archive)
    with tempfile.TemporaryDirectory() as view:
        dst = os.path.join(view, 'results', 'replication'); os.makedirs(dst)
        shutil.copy(MASTER, os.path.join(dst, 'design_master.json'))
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


def reference(archive):
    """The contrast of the printed percentile and of the printed value with no model, beside the model's
    (replication_posthoc.one_design, unchanged), on the first wording of the archive."""
    raw = json.load(open(os.path.join(REAL_ROOT, 'results', archive, 'w1', 'raw.json')))
    des = json.load(open(os.path.join(REAL_ROOT, 'results', archive, 'w1', 'design.json')))
    master = json.load(open(MASTER))
    assert master['reactions'] == des['reactions'], 'the archive was not collected on the replication sample'
    pos = {p: i for i, p in enumerate(master['patients'])}; rows = [pos[p] for p in des['patients']]
    arms = {s: a for s, a in (('d1', 'shuffled'), ('d2', 'shuffled_d2'), ('d3', 'shuffled_d3')) if a in des['arms']}
    donors = {s: np.asarray(master['donors'][s])[rows] for s in arms}
    chk = np.array([[des['swap_source'][f'{p}|{r}'] for r in des['reactions']] for p in des['patients']])
    assert (chk == donors['d1']).all(), 'donor schedule d1 of the archive differs from design_master.json'
    return RP.one_design(des['patients'], des['reactions'], donors, raw, arms)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--only', choices=sorted(BACKBONES), default=None)
    a = ap.parse_args()
    if a.selftest:
        got = frozen_statistics('replication')
        want = json.load(open(os.path.join(REAL_ROOT, 'results', 'replication_frozen.json')))
        d1 = same(got, want)
        ref = json.load(open(os.path.join(REAL_ROOT, 'results', 'replication_posthoc.json')))['replication']
        d2 = same(json.loads(json.dumps(reference('replication'))), ref)
        print('selftest, frozen statistics:', 'identical to results/replication_frozen.json' if not d1 else f'DIFFERENT at {d1[:8]}')
        print('selftest, printed-number reference:', 'identical to results/replication_posthoc.json (replication)' if not d2 else f'DIFFERENT at {d2[:8]}')
        sys.exit(1 if (d1 or d2) else 0)
    done = 0
    for tag, name in BACKBONES.items():
        if a.only and tag != a.only: continue
        archive = f'replication_{tag}'
        if not os.path.exists(os.path.join(REAL_ROOT, 'results', archive, 'w1', 'raw.json')):
            print(f'{tag}: no archive under results/{archive}/w1, skipped'); continue
        F = frozen_statistics(archive)
        if F['wording1'].get('model') != name: sys.exit(f"{tag}: the archive names the model {F['wording1'].get('model')!r}, expected {name!r}")
        head = dict(backbone=name, archive=f'results/{archive}', plan=PLAN,
                    route='replication_stats.main(), unchanged, on a copy of the archive placed under the directory names it '
                          'reads; "replication/w1" and "replication/w2" below stand for this archive')
        json.dump({**head, **F}, open(os.path.join(REAL_ROOT, 'results', f'replication_frozen_{tag}.json'), 'w'), indent=1)
        R = dict(backbone=name, archive=f'results/{archive}', plan=PLAN,
                 note='the own-versus-donor contrast of the printed percentile and of the printed value with no model, beside '
                      'the contrast of the language model; replication_posthoc.one_design, unchanged; named in the plan before '
                      'the first call to this backbone', replication=reference(archive))
        json.dump(R, open(os.path.join(REAL_ROOT, 'results', f'replication_reference_{tag}.json'), 'w'), indent=1)
        d = F['decision']; m = R['replication']['model_minus_printed_percentile']['d1']
        print(f"{tag} ({name}): own minus donor {d['diff_mean']:+.4f} [{d['ci90'][0]:+.4f}, {d['ci90'][1]:+.4f}], "
              f"n {d['n_patients']}, within 0.02: {d['equivalent_002']}, within 0.05: {d['equivalent_005']}, "
              f"excludes zero: {d['interval_excludes_zero']}; model minus printed percentile {m['mean']:+.4f} "
              f"[{m['ci90'][0]:+.4f}, {m['ci90'][1]:+.4f}]")
        done += 1
    print(f'{done} backbone(s) written')


if __name__ == '__main__':
    main()
