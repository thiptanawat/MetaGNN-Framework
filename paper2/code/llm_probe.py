#!/usr/bin/env python3
"""Paper 2 pilot: does a language model add PATIENT-SPECIFIC information to metabolic reaction
activity annotation, or does it reproduce reaction-level textbook knowledge?

Three arms, identical reactions and identical scoring against the input-independent HMA labels:
  reaction   : reaction identity only (name, subsystem, GPR genes, equation). Patient-blind by construction.
  patient    : the same, plus THIS patient's GPR-mapped expression value and its cohort percentile.
  shuffled   : the same, plus a DIFFERENT random patient's expression, presented as if it were this one.

If patient ~ shuffled, the model is not using the patient data it was given. That is the same
input-invariance test Paper 1 applies to the graph model, transferred to a language model.

Usage: python3 llm_probe.py --n 120 --patients 4 --arms reaction patient shuffled --out pilot
"""
import argparse, json, os, random, re, sys, time
import numpy as np, urllib.request

AP = argparse.ArgumentParser()
AP.add_argument('--url', default='http://ray-serve.203.156.3.39.nip.io/qwen3-8-27b/v1/chat/completions')
AP.add_argument('--model', default='qwen3-8-27b')
AP.add_argument('--n', type=int, default=120, help='reactions sampled (stratified by label x expression)')
AP.add_argument('--patients', type=int, default=4)
AP.add_argument('--arms', nargs='+', default=['reaction', 'patient', 'shuffled'])
AP.add_argument('--out', default='pilot'); AP.add_argument('--seed', type=int, default=2024)
AP.add_argument('--workers', type=int, default=12); AP.add_argument('--max_tokens', type=int, default=160)
AP.add_argument('--sampling', choices=['stratified', 'random'], default='stratified',
                help='stratified = balanced on label x expression (removes prevalence structure); random = representative of the network')
AP.add_argument('--think', action='store_true', help='leave Qwen3 reasoning enabled (35x slower; sensitivity check)')
AP.add_argument('--repair', action='store_true',
                help='re-send only the calls whose stored answer is a transport error (__ERROR__), keeping '
                     'every parsed and every empty answer as final; the design is regenerated from the '
                     'same seed and arguments, so the prompts are the ones originally sent')
AP.add_argument('--donor_seed', type=int, default=None,
                help='seed of the generator that draws the stranger at every swapped cell (default: seed + 1); '
                     'a second value gives an independently chosen donor schedule on the same reactions and patients')
AP.add_argument('--patients_from', default=None,
                help='design.json of an earlier run whose patients are kept and extended to --patients, '
                     'so that a replication contains the original sample')
AP.add_argument('--plain_chat', action='store_true',
                help='send no chat_template_kwargs at all, for backbones whose chat template has no '
                     'thinking switch (the Qwen and Gemma templates take enable_thinking=false)')
AP.add_argument('--full_archive', action='store_true',
                help='store every reply in full together with its finish reason, token usage, the model '
                     'identifier the server reports, timestamps and the number of attempts (the original '
                     'runs stored a 400-character prefix and no metadata)')
AP.add_argument('--interleave', action='store_true',
                help='send the calls in a seeded random order across arms, patients and reactions instead '
                     'of arm by arm, so that no arm is confounded with the minutes at which it was sent')
AP.add_argument('--format', choices=['numeric', 'categorical', 'percentile'], default='numeric',
                help='how the expression evidence is written: the raw value plus percentile (numeric), '
                     'a verbal quartile label with no number at all (categorical), or the percentile '
                     'alone (percentile). Rules out numeric formatting as the cause of a null result.')
A = AP.parse_args()
os.makedirs(A.out, exist_ok=True)
rng = np.random.default_rng(A.seed); random.seed(A.seed)

D = np.load('rxn_context.npz', allow_pickle=True)
LAB, HAS, X, PIDS = D['labels'], D['has'], D['X'], D['pids']
from aligned_rxn import RXN, ALIGNED            # the network in the order of LAB, HAS and X
assert len(RXN) == len(LAB)
# per-reaction percentile of each patient's value, across the cohort
order = np.argsort(X, axis=0)
ranks = np.empty_like(order)
for j in range(X.shape[1]):
    ranks[order[:, j], j] = np.arange(X.shape[0])
PCT = 100.0 * ranks / max(X.shape[0] - 1, 1)

# Only reactions whose annotation is aligned to the canonical order enter a prompt.
if A.sampling == 'stratified':
    strata = 2 * LAB + HAS.astype(int); idx = []
    for s in range(4):
        pool = np.where((strata == s) & ALIGNED)[0]
        if len(pool): idx.extend(rng.choice(pool, size=min(len(pool), A.n // 4), replace=False).tolist())
    IDX = np.array(sorted(idx))
else:
    IDX = np.array(sorted(rng.choice(np.where(ALIGNED)[0], size=A.n, replace=False).tolist()))
if A.patients_from:
    base = json.load(open(A.patients_from))['patients']
    rest = [k for k in range(len(PIDS)) if k not in set(base)]
    extra = rng.choice(rest, size=max(0, A.patients - len(base)), replace=False).tolist()
    PSEL = np.array(base[:A.patients] + extra)
else:
    PSEL = rng.choice(len(PIDS), size=A.patients, replace=False)
# The stranger shown in the `shuffled` arm is drawn here, once, from a dedicated generator, and is
# written into design.json. Drawing it inside the worker threads instead would make the assignment
# depend on thread scheduling and leave it unrecorded.
_srng = np.random.default_rng(A.seed + 1 if A.donor_seed is None else A.donor_seed)
SWAP = {(int(pj), int(ri)): int(_srng.choice([k for k in range(len(PIDS)) if k != int(pj)]))
        for pj in PSEL for ri in IDX}

SYS = ("You are a metabolic biochemist annotating a genome-scale metabolic model of human colorectal "
       "tumor tissue. For the reaction described, estimate the probability that it carries flux (is "
       "active) in this tissue. Answer with a JSON object only, no prose:\n"
       '{"p_active": <number between 0 and 1>, "reason": "<one short sentence>"}\n'
       "p_active = 0 means certainly inactive, 0.5 means no information either way, 1 means certainly "
       "active. It is NOT your confidence in your own answer. Use the full range; do not default to "
       "round numbers.")

def eq(r):
    sub = [f"{abs(v):g} {k}" for k, v in r['metabolites'].items() if v < 0]
    pro = [f"{v:g} {k}" for k, v in r['metabolites'].items() if v > 0]
    arrow = ' <=> ' if r.get('lower_bound', 0) < 0 else ' --> '
    return ' + '.join(sub) + arrow + ' + '.join(pro)

def prompt(ri, pj, arm):
    r = RXN[ri]
    if arm.startswith('patient_r'): arm = 'patient'      # exact repeats of the patient prompt
    p = [f"Reaction ID: {r['id']}", f"Name: {r.get('name') or '(unnamed)'}",
         f"Subsystem: {r.get('subsystem') or '(none)'}", f"Equation: {eq(r)}",
         f"Reversible: {'yes' if r.get('lower_bound', 0) < 0 else 'no'}",
         f"Gene-protein-reaction rule: {r.get('gene_reaction_rule') or '(none: no gene assigned)'}"]
    if arm in ('patient', 'shuffled'):
        src = pj if arm == 'patient' else SWAP[(int(pj), int(ri))]
        if HAS[ri]:
            if A.format == 'numeric':
                ev = (f"normalized expression {X[src, ri]:.3f} (cohort percentile {PCT[src, ri]:.0f} "
                      f"of {len(PIDS)} colorectal tumors)")
            elif A.format == 'percentile':
                ev = (f"cohort percentile {PCT[src, ri]:.0f} of {len(PIDS)} colorectal tumors")
            else:
                q = PCT[src, ri]
                band = ('in the top quartile of the cohort (high)' if q >= 75 else
                        'above the cohort median but below the top quartile (moderately high)' if q >= 50 else
                        'below the cohort median but above the bottom quartile (moderately low)' if q >= 25 else
                        'in the bottom quartile of the cohort (low)')
                ev = f"expression is {band} for this reaction"
            p.append(f"Tumor RNA-seq for this patient, mapped to the reaction through its GPR rule: {ev}.")
        elif r['has_gene']:
            p.append("Tumor RNA-seq for this patient: no expression value is available for this reaction.")
        else:
            p.append("Tumor RNA-seq for this patient: no gene maps to this reaction, so no expression value is available.")
    return "\n".join(p)

def call(msg, retries=4):
    payload = {"model": A.model, "temperature": 0.0, "max_tokens": A.max_tokens,
               "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": msg}]}
    if not A.think and not A.plain_chat: payload["chat_template_kwargs"] = {"enable_thinking": False}
    body = json.dumps(payload).encode()
    for k in range(retries):
        try:
            t_sent = time.time()
            req = urllib.request.Request(A.url, data=body, headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=180) as r:
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

def parse(txt):
    if txt is None or txt.startswith('__ERROR__'): return None, None
    m = re.search(r'\{.*?\}', txt, re.S)
    if not m:
        m2 = re.search(r'"?p_active"?\s*:\s*([0-9]*\.?[0-9]+)', txt)   # truncated JSON
        if m2:
            try:
                v = float(m2.group(1)); return v, v >= 0.5
            except ValueError: pass
        return None, None
    try:
        o = json.loads(m.group(0))
        pr = o.get('p_active', o.get('probability'))
        try: pr = float(pr)
        except (TypeError, ValueError): pr = None
        if pr is None and isinstance(o.get('active'), bool): pr = 1.0 if o['active'] else 0.0
        return pr, (pr >= 0.5 if pr is not None else None)
    except Exception:
        m2 = re.search(r'"?p_active"?\s*:\s*([0-9]*\.?[0-9]+)', txt)
        if m2:
            try:
                v = float(m2.group(1)); return v, v >= 0.5
            except ValueError: pass
        return None, None

from concurrent.futures import ThreadPoolExecutor
jobs_all = [(arm, int(pj), int(ri)) for arm in A.arms for pj in PSEL for ri in IDX]
print(f'{len(jobs_all)} calls: {len(A.arms)} arms x {A.patients} patients x {len(IDX)} reactions', flush=True)
RAWF = os.path.join(A.out, 'raw.json'); DESF = os.path.join(A.out, 'design.json')
res = json.load(open(RAWF)) if os.path.exists(RAWF) else {}
prior = json.load(open(DESF)) if os.path.exists(DESF) else {}
done0 = len(res)
def _pending(j):
    k = f'{j[0]}|{j[1]}|{j[2]}'
    if k not in res: return True
    r = res[k]
    if A.repair:   # only transport errors are re-sent; an empty or unparseable answer is an outcome
        return isinstance(r.get('raw'), str) and r['raw'].startswith('__ERROR__')
    return r['prob'] is None
jobs = [j for j in jobs_all if _pending(j)]
if A.interleave:
    # a fixed permutation of the whole job list from its own generator, so the order is replayable
    # and independent of which cells happen to be pending on a resumed run
    _rank = np.empty(len(jobs_all), dtype=int)
    _rank[np.random.default_rng(A.seed + 101).permutation(len(jobs_all))] = np.arange(len(jobs_all))
    _pos = {j: k for k, j in enumerate(jobs_all)}
    jobs = sorted(jobs, key=lambda j: int(_rank[_pos[j]]))
    print('interleaved order: arms, patients and reactions shuffled together', flush=True)
n_resent = sum(1 for j in jobs if f'{j[0]}|{j[1]}|{j[2]}' in res)
print(f'resuming: {done0} already done, {len(jobs)} to go ({n_resent} re-sent)', flush=True)
t0 = time.time()
def work(j):
    arm, pj, ri = j
    txt = call(prompt(ri, pj, arm)); pr, act = parse(txt)
    meta = LAST_META.pop(threading.get_ident(), None) if A.full_archive else None
    return j, pr, act, txt, meta
with ThreadPoolExecutor(max_workers=A.workers) as ex:
    for n, (j, pr, act, txt, meta) in enumerate(ex.map(work, jobs), 1):
        rec = dict(prob=pr, active=act, raw=(txt if A.full_archive else txt[:400]) if txt else None)
        if meta is not None: rec['meta'] = meta
        res[f'{j[0]}|{j[1]}|{j[2]}'] = rec
        if n % 50 == 0:
            el = time.time() - t0
            print(f'  {n}/{len(jobs)}  {el/60:.1f} min  ETA {(el/n)*(len(jobs)-n)/60:.1f} min', flush=True)
            json.dump(res, open(RAWF, 'w'))
json.dump(res, open(RAWF, 'w'))
json.dump(dict(reactions=IDX.tolist(), reaction_ids=[RXN[i]['id'] for i in IDX], patients=PSEL.tolist(),
               arms=A.arms, n=len(jobs_all), sampling=A.sampling, aligned_pool=int(ALIGNED.sum()),
               # a re-sent call is one whose key already held a stored answer when this invocation began;
               # cells that were simply never reached by an interrupted invocation are not re-sent
               resent_calls=prior.get('resent_calls', 0) + n_resent,
               resent_log=prior.get('resent_log', []) + ([dict(calls=n_resent, repair=bool(A.repair),
                                                          at=time.strftime('%Y-%m-%dT%H:%M:%S'))] if n_resent else []),
               patients_from=A.patients_from,
               annotation='recon3d_aligned.json (canonical order)',
               format=A.format, think=bool(A.think), model=A.model, interleave=bool(A.interleave),
               full_archive=bool(A.full_archive), max_tokens=A.max_tokens, temperature=0.0,
               plain_chat=bool(A.plain_chat), url=A.url, seed=A.seed,
               donor_seed=(A.seed + 1 if A.donor_seed is None else A.donor_seed), started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z', time.localtime(t0)),
               swap_source={f'{k[0]}|{k[1]}': v for k, v in SWAP.items()},
               minutes=round(float(prior.get('minutes', 0.0)) + (time.time() - t0) / 60, 1)),
          open(DESF, 'w'), indent=1)
print(f'done in {(time.time()-t0)/60:.1f} min -> {A.out}/raw.json', flush=True)
