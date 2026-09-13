#!/usr/bin/env python3
"""Positive control for Language-Model Evidence: can the model USE the expression value it is shown?

Identical prompt format to the `patient` arm of llm_probe.py, but the question has a numerically
determined answer that does not depend on any biology:
  read    : "Is the cohort percentile shown above 50?"  (answer is printed in the prompt)
  compare : two reactions are shown; "which has the higher percentile?"  (needs a comparison, still no biology)
  echo    : "What is the cohort percentile shown in this record?"  (copy the printed integer)
  echoval : "What is the normalized expression value shown in this record?"  (copy the printed three-decimal value)
If the model passes these and still fails to gain accuracy on the activity task, the null result is about
the biology, not about parsing numbers.

Usage: python3 positive_control.py --n 150 --patients 4 --out pc
"""
import argparse, json, os, re, time
import numpy as np, urllib.request
from concurrent.futures import ThreadPoolExecutor

AP = argparse.ArgumentParser()
AP.add_argument('--url', default='http://ray-serve.203.156.3.39.nip.io/qwen3-8-27b/v1/chat/completions')
AP.add_argument('--model', default='qwen3-8-27b'); AP.add_argument('--n', type=int, default=150)
AP.add_argument('--patients', type=int, default=4); AP.add_argument('--out', default='pc')
AP.add_argument('--seed', type=int, default=2024); AP.add_argument('--workers', type=int, default=12)
AP.add_argument('--tasks', nargs='+', default=['read', 'compare', 'echo'], help='any of read, compare, echo, echoval')
AP.add_argument('--plain_chat', action='store_true', help='send no chat_template_kwargs (templates without a thinking switch)')
AP.add_argument('--full_archive', action='store_true', help='store every reply in full with finish reason, token usage, server model id, timestamps and attempts')
A = AP.parse_args(); os.makedirs(A.out, exist_ok=True); rng = np.random.default_rng(A.seed)

D = np.load('rxn_context.npz', allow_pickle=True)
LAB, HAS, X, PIDS = D['labels'], D['has'], D['X'], D['pids']
from aligned_rxn import RXN, ALIGNED
assert len(RXN) == len(LAB)
order = np.argsort(X, axis=0); ranks = np.empty_like(order)
for j in range(X.shape[1]): ranks[order[:, j], j] = np.arange(X.shape[0])
PCT = 100.0 * ranks / max(X.shape[0] - 1, 1)

expr = np.where(HAS & ALIGNED)[0]
IDX = np.array(sorted(rng.choice(expr, size=min(A.n, len(expr)), replace=False).tolist()))
PSEL = rng.choice(len(PIDS), size=A.patients, replace=False)

def eq(r):
    sub = [f"{abs(v):g} {k}" for k, v in r['metabolites'].items() if v < 0]
    pro = [f"{v:g} {k}" for k, v in r['metabolites'].items() if v > 0]
    return ' + '.join(sub) + (' <=> ' if r.get('lower_bound', 0) < 0 else ' --> ') + ' + '.join(pro)

def block(ri, pj):
    r = RXN[ri]
    return (f"Reaction ID: {r['id']}\nName: {r.get('name') or '(unnamed)'}\n"
            f"Subsystem: {r.get('subsystem') or '(none)'}\nEquation: {eq(r)}\n"
            f"Gene-protein-reaction rule: {r.get('gene_reaction_rule') or '(none)'}\n"
            f"Tumor RNA-seq for this patient, mapped through the GPR rule: normalized expression "
            f"{X[pj, ri]:.3f} (cohort percentile {PCT[pj, ri]:.0f} of {len(PIDS)} colorectal tumors).")

SYS = {'read': ('You are reading an annotation record. Answer the question about the numbers in the record. '
                'Reply with a JSON object only: {"answer": <true|false>}'),
       'compare': ('You are reading two annotation records. Answer the comparison question. '
                   'Reply with a JSON object only: {"answer": <1|2>}'),
       'echo': ('You are reading an annotation record. Copy the requested number out of the record. '
                'Reply with a JSON object only: {"answer": <number>}')}
SYS['echoval'] = SYS['echo']

def build(task, pj, ri, ri2=None):
    if task == 'read':
        return block(ri, pj) + "\n\nQuestion: is the cohort percentile shown above 50?"
    if task == 'echo':
        return block(ri, pj) + "\n\nQuestion: what is the cohort percentile shown in this record?"
    if task == 'echoval':
        return block(ri, pj) + "\n\nQuestion: what is the normalized expression value shown in this record?"
    return (f"RECORD 1\n{block(ri, pj)}\n\nRECORD 2\n{block(ri2, pj)}\n\n"
            "Question: which record has the higher cohort percentile? Answer 1 or 2.")

def call(sys_msg, msg, retries=4):
    payload = {"model": A.model, "temperature": 0.0, "max_tokens": 60,
               "messages": [{"role": "system", "content": sys_msg}, {"role": "user", "content": msg}]}
    if not A.plain_chat: payload["chat_template_kwargs"] = {"enable_thinking": False}
    body = json.dumps(payload).encode()
    for k in range(retries):
        try:
            t_sent = time.time()
            req = urllib.request.Request(A.url, data=body, headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=120) as r:
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
                return f'__ERROR__ {e}'
            time.sleep(2 ** k)
LAST_META = {}
import threading

def parse(txt, task):
    if not txt or txt.startswith('__ERROR__'): return None
    if task in ('echo', 'echoval'):
        m = re.search(r'"answer"\s*:\s*(-?[0-9]*\.?[0-9]+)', txt) or re.search(r'(-?[0-9]*\.?[0-9]+)', txt)
        try: return float(m.group(1)) if m else None
        except ValueError: return None
    m = re.search(r'\{.*?\}', txt, re.S)
    try:
        v = json.loads(m.group(0))['answer'] if m else None
    except Exception:
        v = None
    if v is None:
        m2 = re.search(r'\b(true|false|1|2)\b', txt, re.I); v = m2.group(1).lower() if m2 else None
    if task == 'read': return {'true': True, 'false': False, True: True, False: False}.get(v, None)
    try: return int(v)
    except (TypeError, ValueError): return None

jobs = []
for task in A.tasks:
    for pj in PSEL:
        for ri in IDX:
            ri2 = int(rng.choice([k for k in IDX if k != ri])) if task == 'compare' else None
            jobs.append((task, int(pj), int(ri), ri2))
print(f'{len(jobs)} calls', flush=True)
res = {}; t0 = time.time()
def work(j):
    task, pj, ri, ri2 = j
    txt = call(SYS[task], build(task, pj, ri, ri2))
    meta = LAST_META.pop(threading.get_ident(), None) if A.full_archive else None
    return j, parse(txt, task), txt, meta
with ThreadPoolExecutor(max_workers=A.workers) as ex:
    for n, (j, ans, txt, meta) in enumerate(ex.map(work, jobs), 1):
        task, pj, ri, ri2 = j
        # the answer key is the printed integer percentile (the prompt formats it with :.0f); a compare
        # trial whose two printed integers are equal has no correct answer and is stored as 0
        p1, p2 = int(round(float(PCT[pj, ri]))), (int(round(float(PCT[pj, ri2]))) if ri2 is not None else None)
        if task == 'read':   truth = bool(p1 > 50)
        elif task == 'echo': truth = p1
        elif task == 'echoval': truth = round(float(X[pj, ri]), 3)      # the value as printed, three decimals
        else:                truth = 0 if p1 == p2 else (1 if p1 > p2 else 2)
        res[f'{task}|{pj}|{ri}|{ri2}'] = dict(ans=ans, truth=truth, raw=((txt if A.full_archive else txt[:200]) if txt else None),
                                              pct=float(PCT[pj, ri]), pct2=(float(PCT[pj, ri2]) if ri2 is not None else None),
                                              val=float(X[pj, ri]))
        if meta is not None: res[f'{task}|{pj}|{ri}|{ri2}']['meta'] = meta
        if n % 200 == 0: print(f'  {n}/{len(jobs)} {(time.time()-t0)/60:.1f} min', flush=True)
json.dump(res, open(os.path.join(A.out, 'raw.json'), 'w'))
print(f'done in {(time.time()-t0)/60:.1f} min', flush=True)
for task in A.tasks:
    v = [(r['ans'], r['truth']) for k, r in res.items() if k.startswith(task + '|') and r['ans'] is not None]
    tot = len([k for k in res if k.startswith(task + '|')])
    if v:
        acc = np.mean([(abs(a - t) < 1.0 if task == 'echo' else abs(a - t) < 0.0005 if task == 'echoval' else a == t) for a, t in v])
        print(f'{task:8s} parsed {len(v)}/{tot}  accuracy {acc:.3f}  (chance 0.500)', flush=True)
