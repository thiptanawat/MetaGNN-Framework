#!/usr/bin/env python3
"""E4: a patient-level task with patient-varying ground truth.

Motivation. Reaction activity labels in this benchmark are shared by every patient, so the mean
per-patient AUROC of a scorer that is given patient i's data and of one given patient j's data have
the same distribution by construction: the swap control cannot move that statistic whatever the model
does. To ask whether a language model uses the individual profile it is shown, the target itself has
to vary between patients. Microsatellite instability status does.

Design, fixed before the run:
  cohort   : the 92 MSI-H and 487 MSS patients of the 624-patient TCGA-COAD/READ cohort. The 44 MSI-L
             and 1 not-evaluable patients are excluded; MSI-L is an intermediate call with poor
             cross-assay reproducibility.
  panel    : the K reactions with the highest across-patient variance among expression-bearing
             reactions. Selection is unsupervised: the MSI label is never consulted.
  arms     : own   - the patient's own values for the panel
             swap  - a different, randomly chosen patient's values, presented as this patient's
  control  : count - "how many of the listed reactions are at or above the 75th cohort percentile?"
             The answer is determined by the printed numbers and requires aggregating across the
             whole profile, so it separates reading the profile from reasoning about it.
  scoring  : AUROC of the returned p(MSI-H) against the true status, over patients.
  reference: L2 logistic regression on the identical K features, 5-fold stratified patient-level CV.
"""
import argparse, json, os, re, time
import numpy as np, urllib.request
from concurrent.futures import ThreadPoolExecutor

AP = argparse.ArgumentParser()
AP.add_argument('--url', default='http://ray-serve.203.156.3.39.nip.io/qwen3-8-27b/v1/chat/completions')
AP.add_argument('--model', default='qwen3-8-27b')
AP.add_argument('--k', type=int, default=40, help='reactions in the profile')
AP.add_argument('--arms', nargs='+', default=['own', 'swap', 'count'])
AP.add_argument('--count_n', type=int, default=300, help='patients used for the aggregation control')
AP.add_argument('--out', default='msi'); AP.add_argument('--seed', type=int, default=2024)
AP.add_argument('--workers', type=int, default=10); AP.add_argument('--max_tokens', type=int, default=160)
AP.add_argument('--think', action='store_true')
AP.add_argument('--plain_chat', action='store_true', help='send no chat_template_kwargs (templates without a thinking switch)')
AP.add_argument('--full_archive', action='store_true', help='store every reply in full with finish reason, token usage, server model id, timestamps and attempts')
AP.add_argument('--interleave', action='store_true', help='send the calls of all arms in one seeded random order across arms and patients (the default sends arm by arm), so that no arm is confounded with the minutes at which it was sent; the donors and every prompt are unchanged')
A = AP.parse_args(); os.makedirs(A.out, exist_ok=True)
rng = np.random.default_rng(A.seed)

D = np.load('rxn_context.npz', allow_pickle=True)
HAS, X, PIDS = D['has'], D['X'], list(D['pids'])
from aligned_rxn import RXN, ALIGNED
assert len(RXN) == len(HAS)
import csv
meta = {r['tcga_barcode']: r for r in csv.DictReader(open('clinical_metadata_msi.tsv'), delimiter='\t')}
status = np.array([(meta[p]['msi_status'] or '').strip() for p in PIDS])
keep = np.where((status == 'MSI-H') | (status == 'MSS'))[0]
Y = (status[keep] == 'MSI-H').astype(int)

order = np.argsort(X, axis=0); ranks = np.empty_like(order)
for j in range(X.shape[1]): ranks[order[:, j], j] = np.arange(X.shape[0])
PCT = 100.0 * ranks / (X.shape[0] - 1)

expr = np.where(HAS & ALIGNED)[0]
PANEL = expr[np.argsort(-X[:, expr].std(axis=0))[:A.k]]        # unsupervised selection, aligned reactions only
PANEL = np.array(sorted(PANEL.tolist()))
P75 = np.percentile(X[:, PANEL], 75, axis=0)

def profile(src):
    L = []
    for ri in PANEL:
        r = RXN[ri]
        L.append(f"{r['id']:<12} {(r.get('subsystem') or '(none)')[:34]:<34} "
                 f"expr {X[src, ri]:6.3f}  percentile {PCT[src, ri]:3.0f}")
    return "\n".join(L)

SYS = {'own': ('You are a molecular pathologist. You are shown a metabolic expression profile from one '
               'colorectal tumor: for each reaction of a fixed panel, the RNA-seq expression mapped '
               'through its gene-protein-reaction rule and its percentile within a colorectal tumor '
               'cohort. Estimate the probability that this tumor is microsatellite instability-high '
               '(MSI-H) rather than microsatellite stable (MSS). Answer with a JSON object only, no '
               'prose:\n{"p_msi_high": <number between 0 and 1>, "reason": "<one short sentence>"}\n'
               'Use the full range; do not default to round numbers.'),
       'count': ('You are reading a table of numbers. Answer the counting question exactly. '
                 'Reply with a JSON object only: {"answer": <integer>}')}
SYS['swap'] = SYS['own']

def build(arm, src):
    if arm == 'count':
        return (f"PANEL\n{profile(src)}\n\nQuestion: how many of the {A.k} reactions listed have a "
                f"percentile of 75 or greater? Count them and report the integer.")
    return f"METABOLIC PROFILE OF THIS TUMOR ({A.k} reactions)\n{profile(src)}"

def call(sys_msg, msg, retries=4):
    payload = {"model": A.model, "temperature": 0.0, "max_tokens": A.max_tokens,
               "messages": [{"role": "system", "content": sys_msg}, {"role": "user", "content": msg}]}
    if not A.think and not A.plain_chat: payload["chat_template_kwargs"] = {"enable_thinking": False}
    body = json.dumps(payload).encode()
    for k in range(retries):
        try:
            t_sent = time.time()
            req = urllib.request.Request(A.url, data=body, headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=240) as r:
                rj = json.loads(r.read())
            if A.full_archive:
                ch = rj['choices'][0]
                LAST_META[threading.get_ident()] = dict(
                    finish_reason=ch.get('finish_reason'), usage=rj.get('usage'), model=rj.get('model'),
                    id=rj.get('id'), sent_at=round(t_sent, 3), received_at=round(time.time(), 3), attempts=k + 1)
            return rj['choices'][0]['message']['content']
        except Exception as e:
            if k == retries - 1:
                if A.full_archive: LAST_META[threading.get_ident()] = dict(error=f'{type(e).__name__}: {e}', attempts=k + 1, sent_at=round(t_sent, 3))
                return f'__ERROR__ {type(e).__name__}: {e}'
            time.sleep(2 ** k)
LAST_META = {}
import threading

def parse(txt, arm):
    if not txt or txt.startswith('__ERROR__'): return None
    key = 'answer' if arm == 'count' else 'p_msi_high'
    m = re.search(r'"?' + key + r'"?\s*:\s*(-?[0-9]*\.?[0-9]+)', txt)
    if m:
        try: return float(m.group(1))
        except ValueError: return None
    return None

jobs = []
cnt_sel = rng.choice(len(keep), size=min(A.count_n, len(keep)), replace=False)
for arm in A.arms:
    idx = cnt_sel if arm == 'count' else range(len(keep))
    for a in idx:
        pj = int(keep[a])
        src = pj if arm != 'swap' else int(rng.choice([k for k in keep.tolist() if k != pj]))
        jobs.append((arm, pj, src))
RAWF = os.path.join(A.out, 'raw.json')
res = json.load(open(RAWF)) if os.path.exists(RAWF) else {}
jobs = [j for j in jobs if f'{j[0]}|{j[1]}' not in res or res[f'{j[0]}|{j[1]}']['val'] is None]
if A.interleave:
    # the donors were drawn above in the fixed arm-by-arm order, so the prompts are identical to the
    # blocked collection's; only the dispatch order changes, from a separate generator
    order_rng = np.random.default_rng(A.seed + 4241)
    jobs = [jobs[i] for i in order_rng.permutation(len(jobs))]
print(f'{len(jobs)} calls to go ({len(res)} already done){"; arms interleaved" if A.interleave else ""}', flush=True)
t0 = time.time()
def work(j):
    arm, pj, src = j
    txt = call(SYS[arm], build(arm, src))
    meta = LAST_META.pop(threading.get_ident(), None) if A.full_archive else None
    return j, parse(txt, arm), txt, meta
with ThreadPoolExecutor(max_workers=A.workers) as ex:
    for n, (j, v, txt, meta) in enumerate(ex.map(work, jobs), 1):
        arm, pj, src = j
        rec = dict(val=v, src=src, raw=((txt if A.full_archive else txt[:300]) if txt else None))
        if meta is not None: rec['meta'] = meta
        res[f'{arm}|{pj}'] = rec
        if n % 50 == 0:
            el = time.time() - t0
            print(f'  {n}/{len(jobs)} {el/60:.1f} min ETA {(el/n)*(len(jobs)-n)/60:.1f} min', flush=True)
            json.dump(res, open(RAWF, 'w'))
json.dump(res, open(RAWF, 'w'))
json.dump(dict(panel=PANEL.tolist(), panel_ids=[RXN[i]['id'] for i in PANEL], keep=keep.tolist(),
               y=Y.tolist(), count_patients=[int(keep[a]) for a in cnt_sel], k=A.k, arms=A.arms,
               p75=P75.tolist(), model=A.model, minutes=round((time.time()-t0)/60, 1),
               full_archive=bool(A.full_archive), max_tokens=A.max_tokens, plain_chat=bool(A.plain_chat), url=A.url,
               interleave=bool(A.interleave),
               started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z', time.localtime(t0))),
          open(os.path.join(A.out, 'design.json'), 'w'))
print(f'done in {(time.time()-t0)/60:.1f} min', flush=True)
