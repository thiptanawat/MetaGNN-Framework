#!/usr/bin/env python3
"""The frozen percentile-only interface for the microsatellite instability task, on any cohort.

This is the collection driver of the prospective analysis plan (docs/ANALYSIS_PLAN_2026-09-10.md,
Section 2): the development interface with the raw expression field removed, midrank percentiles,
an explicit uniform-derangement donor arm, the harmonized endpoint, complete request/response
archives, interleaved dispatch, a same-session repeat subset, and three configurations on one
backbone (zero-shot, evidence-assisted, tool-assisted). msi_probe.py, the development collector,
is left untouched so the original configuration remains reproducible.

Cohorts: tcga (development; percentiles over the 624 patients), gse39582 and gse13294 (external;
results/external/<cohort>_panel.npz from external_cohort.py). The panel is results/external/
panel_frozen.json for every cohort. Endpoints: nonmsih = MSI-H against MSS and MSI-L together
(the external cohorts' dMMR/pMMR); mss = MSI-H against MSS (development only).

Arms: own (the patient's own percentiles); donor (another patient's percentiles from a uniform
derangement of the evaluated patients, drawn with --donor_seed independently of the label, the
recipient's label kept); count (a reading control: how many panel percentiles are at or above
75). --repeat_n re-sends the own prompt of the first N evaluated patients in the same session.

Configurations: zero_shot; evidence (a frozen block of reaction names and a general background
note follows the profile); tool (the frozen logistic reference's probability for the shown panel
is stated and the model is asked to report it with a one-sentence explanation).
"""
import argparse, json, os, re, csv, time, threading
import numpy as np, urllib.request
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import rankdata
import _paths as PATHS

AP = argparse.ArgumentParser()
AP.add_argument('--url', default='http://ray-serve.203.156.3.39.nip.io/qwen3-8-27b/v1/chat/completions')
AP.add_argument('--model', default='qwen3-8-27b')
AP.add_argument('--cohort', choices=['tcga', 'gse39582', 'gse13294'], default='tcga')
AP.add_argument('--endpoint', choices=['nonmsih', 'mss'], default='nonmsih')
AP.add_argument('--config', choices=['zero_shot', 'evidence', 'tool'], default='zero_shot')
AP.add_argument('--arms', nargs='+', default=['own', 'donor', 'count'])
AP.add_argument('--donor_seed', type=int, default=11)
AP.add_argument('--count_n', type=int, default=300); AP.add_argument('--repeat_n', type=int, default=40)
AP.add_argument('--out', required=True); AP.add_argument('--seed', type=int, default=2024)
AP.add_argument('--workers', type=int, default=10); AP.add_argument('--max_tokens', type=int, default=160)
AP.add_argument('--plain_chat', action='store_true'); AP.add_argument('--retries', type=int, default=3)
AP.add_argument('--limit', type=int, default=0, help='smoke test: only the first N evaluated patients')
A = AP.parse_args(); os.makedirs(A.out, exist_ok=True)

EXT = os.path.join(PATHS.ROOT, 'results', 'external')
FROZEN = json.load(open(os.path.join(EXT, 'panel_frozen.json'))); PANEL = np.array(FROZEN['panel']); PANEL_IDS = FROZEN['panel_ids']
SCORER = json.load(open(os.path.join(EXT, 'frozen_scorer.json')))
RXN = PATHS.reactions()
assert [RXN[i]['id'] for i in PANEL] == PANEL_IDS

# ---- the cohort: percentiles, identifiers, labels ------------------------------------------------
if A.cohort == 'tcga':
    D = np.load(PATHS.data('rxn_context.npz'), allow_pickle=True); X = D['X']; IDS = [str(p) for p in D['pids']]
    meta = {r['tcga_barcode']: r for r in csv.DictReader(open(PATHS.data('clinical_metadata_msi.tsv')), delimiter='\t')}
    status = np.array([(meta[p]['msi_status'] or '').strip() for p in IDS])
    PCT = np.stack([100.0 * (rankdata(X[:, j]) - 1.0) / (X.shape[0] - 1) for j in PANEL], axis=1)
    keep_status = ('MSI-H', 'MSS', 'MSI-L') if A.endpoint == 'nonmsih' else ('MSI-H', 'MSS')
    keep = np.where(np.isin(status, keep_status))[0]; Y = (status[keep] == 'MSI-H').astype(int)
else:
    assert A.endpoint == 'nonmsih', 'the external cohorts carry the harmonized endpoint only'
    Z = np.load(os.path.join(EXT, f'{A.cohort}_panel.npz'), allow_pickle=True)
    assert [str(x) for x in Z['panel_ids']] == json.load(open(PATHS.run('msi_int', 'design.json')))['panel_ids']
    pos = [list(Z['panel_ids']).index(i) for i in PANEL_IDS]; PCT = Z['pct'][:, pos].astype(np.float64); IDS = [str(p) for p in Z['ids']]
    lab = np.array([str(v) for v in Z['label']])
    keep = np.where(np.isin(lab, ('MSI-H', 'non-MSI-H')))[0]; Y = (lab[keep] == 'MSI-H').astype(int)
if A.limit: keep = keep[:A.limit]; Y = Y[:A.limit]
rng = np.random.default_rng(A.seed)

# ---- the donor schedule: a uniform derangement of the evaluated patients, label-blind -------------
def derangement(n, seed):
    r = np.random.default_rng(seed)
    while True:
        p = r.permutation(n)
        if not np.any(p == np.arange(n)): return p
DONOR = derangement(len(keep), A.donor_seed)   # position -> position within keep

# ---- the frozen tool -----------------------------------------------------------------------------
EP = SCORER['endpoints']['msih_vs_nonmsih' if A.endpoint == 'nonmsih' else 'msih_vs_mss']
COEF = np.array(EP['coef']); INTERCEPT = float(EP['intercept'])
def tool_prob(pct_row): return float(1.0 / (1.0 + np.exp(-(COEF @ (pct_row / 100.0) + INTERCEPT))))

# ---- prompts (frozen text) -------------------------------------------------------------------------
SYS_MAIN = ('You are a molecular pathologist. You are shown a metabolic expression profile from one '
            'colorectal tumor: for each reaction of a fixed panel, the percentile of its RNA expression, '
            'mapped through its gene-protein-reaction rule, within a colorectal tumor cohort. Estimate the '
            'probability that this tumor is microsatellite instability-high (MSI-H) rather than microsatellite '
            'stable or low (MSS or MSI-L). Answer with a JSON object only, no prose:\n'
            '{"p_msi_high": <number between 0 and 1>, "reason": "<one short sentence>"}\n'
            'Use the full range; do not default to round numbers.')
SYS_COUNT = ('You are reading a table of numbers. Answer the counting question exactly. '
             'Reply with a JSON object only: {"answer": <integer>}')
EVIDENCE = ('REFERENCE NOTES (fixed for every tumor)\n'
            + '\n'.join(f"{RXN[i]['id']:<12} {(RXN[i].get('name') or '')[:60]}; {RXN[i].get('subsystem') or '(none)'}; {RXN[i].get('n_genes', 0)} gene(s) in its rule" for i in PANEL)
            + '\nBackground: microsatellite instability-high colorectal tumors arise from deficient DNA mismatch repair, most often '
              'through MLH1 promoter methylation in sporadic cases or germline mismatch-repair variants; they are hypermutated, '
              'enriched in the right colon, immune-infiltrated, and form a transcriptional group distinct from most '
              'microsatellite-stable tumors. The panel above is a fixed set of metabolic reactions chosen for expression '
              'variability, not for known association with this phenotype.')
def profile(pct_row):
    return '\n'.join(f"{RXN[i]['id']:<12} {(RXN[i].get('subsystem') or '(none)')[:34]:<34} percentile {pct_row[k]:3.0f}" for k, i in enumerate(PANEL))
def user_msg(arm, pct_row):
    if arm == 'count':
        return (f"PANEL\n{profile(pct_row)}\n\nQuestion: how many of the {len(PANEL)} reactions listed have a "
                f"percentile of 75 or greater? Count them and report the integer.")
    msg = f"METABOLIC PROFILE OF THIS TUMOR ({len(PANEL)} reactions)\n{profile(pct_row)}"
    if A.config == 'evidence': msg += '\n\n' + EVIDENCE
    if A.config == 'tool':
        msg += (f"\n\nTOOL OUTPUT: a logistic regression fitted on a separate reference cohort of colorectal tumors, applied to "
                f"the panel above, returns p(MSI-H) = {tool_prob(pct_row):.3f}. Report this probability as p_msi_high unless the "
                f"panel is malformed, and in the reason name the reactions whose percentiles contribute most.")
    return msg

LAST_META = {}
def call(sys_msg, msg):
    payload = {"model": A.model, "temperature": 0.0, "max_tokens": A.max_tokens,
               "messages": [{"role": "system", "content": sys_msg}, {"role": "user", "content": msg}]}
    if not A.plain_chat: payload["chat_template_kwargs"] = {"enable_thinking": False}
    body = json.dumps(payload).encode()
    for k in range(A.retries):
        t_sent = time.time()
        try:
            req = urllib.request.Request(A.url, data=body, headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=240) as r: rj = json.loads(r.read())
            ch = rj['choices'][0]
            LAST_META[threading.get_ident()] = dict(finish_reason=ch.get('finish_reason'), usage=rj.get('usage'), model=rj.get('model'),
                                                   id=rj.get('id'), sent_at=round(t_sent, 3), received_at=round(time.time(), 3), attempts=k + 1)
            return ch['message']['content']
        except Exception as e:
            if k == A.retries - 1:
                LAST_META[threading.get_ident()] = dict(error=f'{type(e).__name__}: {e}', attempts=k + 1, sent_at=round(t_sent, 3))
                return f'__ERROR__ {type(e).__name__}: {e}'
            time.sleep(2 ** k)
def parse(txt, arm):
    if not txt or txt.startswith('__ERROR__'): return None
    key = 'answer' if arm == 'count' else 'p_msi_high'
    m = re.search(r'"?' + key + r'"?\s*:\s*(-?[0-9]*\.?[0-9]+)', txt)
    if not m: return None
    try: return float(m.group(1))
    except ValueError: return None

# ---- the jobs ----------------------------------------------------------------------------------------
jobs = []
cnt_sel = rng.choice(len(keep), size=min(A.count_n, len(keep)), replace=False)
for arm in A.arms:
    for a in (cnt_sel if arm == 'count' else range(len(keep))):
        src = int(DONOR[a]) if arm == 'donor' else int(a)
        jobs.append((arm, int(a), src))
for a in range(min(A.repeat_n, len(keep))): jobs.append(('own_repeat', a, a))
RAWF = os.path.join(A.out, 'raw.json')
res = json.load(open(RAWF)) if os.path.exists(RAWF) else {}
jobs = [j for j in jobs if f'{j[0]}|{j[1]}' not in res or res[f'{j[0]}|{j[1]}']['val'] is None]
order_rng = np.random.default_rng(A.seed + 4241); jobs = [jobs[i] for i in order_rng.permutation(len(jobs))]
print(f'{A.cohort} {A.endpoint} {A.config}: {len(keep)} patients ({int(Y.sum())} MSI-H), {len(jobs)} calls to go ({len(res)} done), interleaved', flush=True)
t0 = time.time()
def work(j):
    arm, a, src = j
    txt = call(SYS_COUNT if arm == 'count' else SYS_MAIN, user_msg(arm, PCT[keep[src]]))
    return j, parse(txt, arm), txt, LAST_META.pop(threading.get_ident(), None)
with ThreadPoolExecutor(max_workers=A.workers) as ex:
    for n, (j, v, txt, meta) in enumerate(ex.map(work, jobs), 1):
        arm, a, src = j
        rec = dict(val=v, src=src, raw=txt, meta=meta)
        if A.config == 'tool' and arm in ('own', 'donor', 'own_repeat'): rec['tool_prob'] = round(tool_prob(PCT[keep[src]]), 6)
        res[f'{arm}|{a}'] = rec
        if n % 100 == 0:
            el = time.time() - t0; print(f'  {n}/{len(jobs)} {el/60:.1f} min ETA {(el/n)*(len(jobs)-n)/60:.1f} min', flush=True)
            json.dump(res, open(RAWF, 'w'))
json.dump(res, open(RAWF, 'w'))
failed = {arm: sum(1 for k, r in res.items() if k.startswith(arm + '|') and r['val'] is None) for arm in A.arms + ['own_repeat']}
json.dump(dict(cohort=A.cohort, endpoint=A.endpoint, config=A.config, arms=A.arms, panel=PANEL.tolist(), panel_ids=PANEL_IDS,
               keep=[int(k) for k in keep], ids=[IDS[k] for k in keep], y=Y.tolist(), donor_seed=A.donor_seed, donor=DONOR.tolist(),
               count_patients=[int(a) for a in cnt_sel], repeat_n=min(A.repeat_n, len(keep)), model=A.model, url=A.url,
               max_tokens=A.max_tokens, temperature=0.0, plain_chat=bool(A.plain_chat), retries=A.retries, seed=A.seed,
               interleave=True, full_archive=True, failed_calls=failed, minutes=round((time.time() - t0) / 60, 1),
               started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z', time.localtime(t0)),
               system_prompt=SYS_MAIN, count_prompt=SYS_COUNT, example_user_prompt=user_msg('own', PCT[keep[0]]),
               evidence_block=(EVIDENCE if A.config == 'evidence' else None),
               tool=(dict(kind='frozen logistic reference (frozen_scorer.json)', endpoint=EP['keep_status'], C=EP['C']) if A.config == 'tool' else None),
               percentile='within-cohort midrank over the tumor samples of the cohort, 0-100, printed as an integer'),
          open(os.path.join(A.out, 'design.json'), 'w'), indent=1)
print(f'done in {(time.time() - t0) / 60:.1f} min; failed calls {failed}', flush=True)
