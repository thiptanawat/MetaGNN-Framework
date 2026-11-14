#!/usr/bin/env python3
"""Collector of the prespecified replication of the own-versus-donor contrast (plan of 13 November 2026).

The sample (reactions, the ordered patient list, three donor schedules) is read from
results/replication/design_master.json, which replication_design.py wrote before the first call.
The second replication (plan of 14 November 2026) reads its own sample, written by replication2_design.py,
through --master results/replication2/design_master.json; nothing else differs.

Arms, stored under the keys the earlier archives use:
  reaction       patient-blind prompt
  patient        this patient's value and percentile (own)
  shuffled       a donor's value presented as this patient's, donor schedule d1 (the primary donor arm)
  shuffled_d2    the same under donor schedule d2
  shuffled_d3    the same under donor schedule d3

Wording 1 is the wording of the earlier collections: its system prompt, its prompt builder and its
answer parser are taken from the source of llm_probe.py itself, so the two cannot drift apart.
Wording 2 is a paraphrase that carries the same fields and the same two numbers in the same formats.

The patients are collected in blocks, in the order of the master list. Inside a block every call of
every arm is sent in one seeded random order. A block counts only when all of its calls have an
answer or a recorded transport error; design.json lists the patients of the completed blocks and is
rewritten after each one, so a run that is stopped in the middle of a block leaves a design that
describes complete blocks only. A call is final once the server has answered it, parsed or not;
only transport errors are sent again, once, at the end of the block.

    python3 code/replication_probe.py --url URL --model qwen3-8-27b --wording 1 \
        --blocks 1-3 --arms reaction patient shuffled shuffled_d2 --out results/replication/w1

Runs from the study directory (paper2/), like the other collectors.
"""
import argparse, ast, hashlib, json, os, re, sys, threading, time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AP = argparse.ArgumentParser()
AP.add_argument('--url', required=True); AP.add_argument('--model', default='qwen3-8-27b')
AP.add_argument('--wording', type=int, choices=[1, 2], default=1)
AP.add_argument('--blocks', default='1', help='blocks of the master patient list to collect, e.g. 1-3 or 4')
AP.add_argument('--block_size', type=int, default=10)
AP.add_argument('--arms', nargs='+', default=['reaction', 'patient', 'shuffled', 'shuffled_d2'])
AP.add_argument('--blind_blocks', type=int, default=10**6,
                help='the patient-blind arm is collected only in blocks up to this number (it repeats one '
                     'prompt per patient, so a few blocks measure the determinism floor)')
AP.add_argument('--out', required=True)
AP.add_argument('--workers', type=int, default=16); AP.add_argument('--max_tokens', type=int, default=160)
AP.add_argument('--no_start_after', default=None,
                help='local time YYYY-MM-DDTHH:MM after which no new block is started')
AP.add_argument('--abandon_after', default=None,
                help='local time YYYY-MM-DDTHH:MM after which a block in progress is abandoned')
AP.add_argument('--plain_chat', action='store_true',
                help='send no chat_template_kwargs (backbones whose chat template has no thinking switch)')
AP.add_argument('--bench', type=int, default=0,
                help='send this many patient-blind prompts of the EARLIER design results/repr (not of the '
                     'replication sample), print the rate, store nothing; used to size the replication')
AP.add_argument('--print_prompts', type=int, default=0, help='print the first N prompts of block 1 and exit (no call)')
AP.add_argument('--master', default=os.path.join('results', 'replication', 'design_master.json'),
                help='the sample file; the default is the sample of the first replication, and the second '
                     'replication passes results/replication2/design_master.json')
A = AP.parse_args()

# ---- data, exactly as the earlier collectors load it -------------------------------------------
D = np.load('rxn_context.npz', allow_pickle=True)
LAB, HAS, X, PIDS = D['labels'], D['has'], D['X'], D['pids']
from aligned_rxn import RXN, ALIGNED
import cohort_order as CO
PCT = CO.percentile_as_collected(X)
MASTER = json.load(open(A.master))
IDX = [int(x) for x in MASTER['reactions']]
PALL = [int(x) for x in MASTER['patients']]
SCHED = {'shuffled': 'd1', 'shuffled_d2': 'd2', 'shuffled_d3': 'd3'}
DONOR = {s: {(PALL[a], IDX[b]): int(MASTER['donors'][s][a][b]) for a in range(len(PALL)) for b in range(len(IDX))}
         for s in MASTER['schedules']}
assert all(ALIGNED[i] for i in IDX)

# ---- wording 1: the earlier study's own system prompt, prompt builder and parser ---------------
_SRC = open(os.path.join(HERE, 'llm_probe.py')).read()
_tree = ast.parse(_SRC)
_keep = [n for n in _tree.body
         if (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'SYS' for t in n.targets))
         or (isinstance(n, ast.FunctionDef) and n.name in ('eq', 'prompt', 'parse'))]
assert [getattr(n, 'name', 'SYS') for n in _keep] == ['SYS', 'eq', 'prompt', 'parse'], 'llm_probe.py changed'


class _Fmt:            # the earlier collections' default evidence format
    format = 'numeric'


_NS = dict(RXN=RXN, HAS=HAS, X=X, PCT=PCT, PIDS=PIDS, A=_Fmt, SWAP={}, json=json, re=re, np=np)
exec(compile(ast.Module(body=_keep, type_ignores=[]), 'llm_probe.py', 'exec'), _NS)
SYS1, parse = _NS['SYS'], _NS['parse']


def prompt1(ri, pj, arm):
    if arm in SCHED:
        _NS['SWAP'] = {(int(pj), int(ri)): DONOR[SCHED[arm]][(int(pj), int(ri))]}
        return _NS['prompt'](ri, pj, 'shuffled')
    return _NS['prompt'](ri, pj, arm)


# ---- wording 2: a paraphrase with the same fields and the same two numbers ---------------------
SYS2 = ("As an expert in human metabolism, you are curating a genome-scale metabolic network for "
        "colorectal tumor tissue. Given the reaction below, state how likely it is that the reaction is "
        "active, meaning that it carries flux, in this tissue. Reply with one JSON object and nothing else:\n"
        '{"p_active": <number between 0 and 1>, "reason": "<one short sentence>"}\n'
        "A p_active of 0 says the reaction is certainly inactive, 1 says it is certainly active, and 0.5 "
        "says the available information does not favor either. The number is the probability of activity, "
        "not how sure you are of your answer. Use the whole scale and avoid defaulting to round values.")


def prompt2(ri, pj, arm):
    r = RXN[ri]; eq = _NS['eq']
    p = [f"Reaction identifier: {r['id']}", f"Reaction name: {r.get('name') or '(unnamed)'}",
         f"Pathway (subsystem): {r.get('subsystem') or '(none)'}", f"Stoichiometry: {eq(r)}",
         f"Reversibility: {'reversible' if r.get('lower_bound', 0) < 0 else 'irreversible'}",
         f"Genes (gene-protein-reaction rule): {r.get('gene_reaction_rule') or 'none (no gene is assigned)'}"]
    if arm == 'patient' or arm in SCHED:
        src = pj if arm == 'patient' else DONOR[SCHED[arm]][(int(pj), int(ri))]
        if HAS[ri]:
            p.append(f"RNA-seq of this patient's tumor gives this reaction, through its gene rule, a normalized "
                     f"expression of {X[src, ri]:.3f}, which is percentile {PCT[src, ri]:.0f} among the "
                     f"{len(PIDS)} colorectal tumors of the cohort.")
        elif r['has_gene']:
            p.append("RNA-seq of this patient's tumor provides no expression value for this reaction.")
        else:
            p.append("No gene is mapped to this reaction, so RNA-seq of this patient's tumor provides no "
                     "expression value for it.")
    return "\n".join(p)


SYS = SYS1 if A.wording == 1 else SYS2
prompt = prompt1 if A.wording == 1 else prompt2
LAST_META = {}


def call(msg, retries=4):
    payload = {"model": A.model, "temperature": 0.0, "max_tokens": A.max_tokens,
               "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": msg}]}
    if not A.plain_chat: payload["chat_template_kwargs"] = {"enable_thinking": False}
    body = json.dumps(payload).encode(); t_sent = time.time()
    for k in range(retries):
        try:
            t_sent = time.time()
            req = urllib.request.Request(A.url, data=body, headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=180) as r:
                rj = json.loads(r.read())
            ch = rj['choices'][0]
            LAST_META[threading.get_ident()] = dict(
                finish_reason=ch.get('finish_reason'), usage=rj.get('usage'), model=rj.get('model'),
                id=rj.get('id'), sent_at=round(t_sent, 3), received_at=round(time.time(), 3), attempts=k + 1)
            return ch['message']['content']
        except Exception as e:
            if k == retries - 1:
                LAST_META[threading.get_ident()] = dict(error=f'{type(e).__name__}: {e}', attempts=k + 1, sent_at=round(t_sent, 3))
                return f'__ERROR__ {type(e).__name__}: {e}'
            time.sleep(2 ** k)


def work(j):
    arm, pj, ri, msg = j
    txt = call(msg); pr, act = parse(txt)
    return (arm, pj, ri), pr, act, txt, LAST_META.pop(threading.get_ident(), None)


def _stamp(s):
    return None if s is None else time.mktime(time.strptime(s, '%Y-%m-%dT%H:%M'))


def block_patients(b):
    return PALL[(b - 1) * A.block_size: b * A.block_size]


def block_jobs(b):
    arms = [a for a in A.arms if a != 'reaction' or b <= A.blind_blocks]
    cells = [(arm, pj, ri) for arm in arms for pj in block_patients(b) for ri in IDX]
    order = np.random.default_rng(MASTER['seed'] + 1000 * A.wording + b).permutation(len(cells))
    return arms, [cells[k] for k in order]


if A.print_prompts:
    arms, cells = block_jobs(1)
    for arm, pj, ri in cells[:A.print_prompts]:
        print(f'--- {arm} | patient {pj} | reaction {ri}\n[system]\n{SYS}\n[user]\n{prompt(ri, pj, arm)}\n')
    sys.exit(0)

if A.bench:
    old = json.load(open(os.path.join('results', 'repr', 'design.json')))
    cells = [('reaction', int(old['patients'][0]), int(ri)) for ri in old['reactions']]
    cells = (cells * (A.bench // len(cells) + 1))[:A.bench]
    t0 = time.time(); ok = 0
    with ThreadPoolExecutor(max_workers=A.workers) as ex:
        for key, pr, act, txt, meta in ex.map(work, [(a, p, r, prompt(r, p, a)) for a, p, r in cells]):
            ok += pr is not None
    el = time.time() - t0
    print(f'bench: {A.bench} calls, {A.workers} workers, {el:.1f} s, {A.bench / el:.2f} calls/s, parsed {ok}', flush=True)
    sys.exit(0)

# ---- collection -----------------------------------------------------------------------------------
os.makedirs(A.out, exist_ok=True)
RAWL = os.path.join(A.out, 'raw.jsonl'); RAWF = os.path.join(A.out, 'raw.json'); DESF = os.path.join(A.out, 'design.json')
res = json.load(open(RAWF)) if os.path.exists(RAWF) else {}
if os.path.exists(RAWL):
    for line in open(RAWL):
        line = line.strip()
        if line:
            try:
                k, rec = json.loads(line)
            except ValueError:
                continue                      # a line cut by an interruption
            res[k] = rec
des = json.load(open(DESF)) if os.path.exists(DESF) else dict(blocks=[], resent_calls=0, minutes=0.0)
lo, _, hi = A.blocks.partition('-'); BLOCKS = list(range(int(lo), int(hi or lo) + 1))
t_no_start, t_abandon = _stamp(A.no_start_after), _stamp(A.abandon_after)
_sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()


def is_error(rec):
    return isinstance(rec.get('raw'), str) and rec['raw'].startswith('__ERROR__')


def write_design():
    done = sorted(des['blocks'], key=lambda x: x['block'])
    pats = [pj for blk in done for pj in blk['patients']]
    arms = [a for a in ('reaction', 'patient', 'shuffled', 'shuffled_d2', 'shuffled_d3')
            if any(a in blk['arms'] for blk in done)]
    out = dict(reactions=IDX, reaction_ids=[RXN[i]['id'] for i in IDX], patients=pats, arms=arms,
               n=int(sum(blk['calls'] for blk in done)), sampling='random, disjoint from the earlier designs (design_master.json)',
               aligned_pool=int(ALIGNED.sum()), annotation='recon3d_aligned.json (canonical order)', format='numeric',
               think=False, model=A.model, interleave=True, full_archive=True, max_tokens=A.max_tokens, temperature=0.0,
               plain_chat=bool(A.plain_chat), url=A.url, seed=MASTER['seed'], wording=A.wording,
               block_size=A.block_size, blocks=done, resent_calls=int(des.get('resent_calls', 0)),
               minutes=round(float(des.get('minutes', 0.0)), 1),
               donor_schedules={a: SCHED[a] for a in arms if a in SCHED},
               swap_source={f'{pj}|{ri}': DONOR['d1'][(pj, ri)] for pj in pats for ri in IDX},
               code_sha256=dict(replication_probe=_sha(os.path.abspath(__file__)), llm_probe=_sha(os.path.join(HERE, 'llm_probe.py'))),
               design_master_sha256=_sha(A.master))
    json.dump(out, open(DESF, 'w'), indent=1)


for b in BLOCKS:
    if any(blk['block'] == b for blk in des['blocks']):
        print(f'block {b}: already complete', flush=True); continue
    if not block_patients(b):
        print(f'block {b}: beyond the master list', flush=True); break
    if t_no_start is not None and time.time() > t_no_start:
        print(f'block {b}: not started, the start limit has passed', flush=True); break
    arms, cells = block_jobs(b)
    key = lambda c: f'{c[0]}|{c[1]}|{c[2]}'
    pending = [c for c in cells if key(c) not in res]
    print(f'block {b}: {len(cells)} calls ({len(arms)} arms x {len(block_patients(b))} patients x {len(IDX)} reactions), '
          f'{len(pending)} to go', flush=True)
    t0 = time.time(); started = time.strftime('%Y-%m-%dT%H:%M:%S%z', time.localtime(t0)); abandoned = False
    with open(RAWL, 'a') as fl, ThreadPoolExecutor(max_workers=A.workers) as ex:
        def run(batch):
            for n, (k3, pr, act, txt, meta) in enumerate(ex.map(work, [(a, p, r, prompt(r, p, a)) for a, p, r in batch]), 1):
                rec = dict(prob=pr, active=act, raw=txt if txt else None, meta=meta)
                res[f'{k3[0]}|{k3[1]}|{k3[2]}'] = rec
                fl.write(json.dumps([f'{k3[0]}|{k3[1]}|{k3[2]}', rec]) + '\n')
            fl.flush(); os.fsync(fl.fileno())
        for s in range(0, len(pending), 400):
            if t_abandon is not None and time.time() > t_abandon:
                abandoned = True; break
            run(pending[s:s + 400])
            el = time.time() - t0; n = min(s + 400, len(pending))
            print(f'  block {b}: {n}/{len(pending)}  {el / 60:.1f} min  {n / el:.2f} calls/s  '
                  f'ETA {(el / n) * (len(pending) - n) / 60:.1f} min', flush=True)
        n_resent = 0
        if not abandoned:
            bad = [c for c in cells if is_error(res[key(c)])]
            if bad:
                time.sleep(20); n_resent = len(bad); run(bad)        # the one repair pass
    if abandoned:
        print(f'block {b}: abandoned at the time limit; it does not enter the design', flush=True); break
    still = sum(1 for c in cells if is_error(res[key(c)]))
    des['blocks'].append(dict(block=b, patients=block_patients(b), arms=arms, calls=len(cells), started_at=started,
                              finished_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), minutes=round((time.time() - t0) / 60, 2),
                              workers=A.workers, resent_calls=n_resent, transport_errors_left=still,
                              unparsed=sum(1 for c in cells if res[key(c)]['prob'] is None and not is_error(res[key(c)]))))
    des['resent_calls'] = int(des.get('resent_calls', 0)) + n_resent
    des['minutes'] = float(des.get('minutes', 0.0)) + (time.time() - t0) / 60
    json.dump(res, open(RAWF, 'w')); write_design()
    print(f'block {b}: complete in {(time.time() - t0) / 60:.1f} min, {n_resent} re-sent, {still} transport errors left', flush=True)
print('done', flush=True)
