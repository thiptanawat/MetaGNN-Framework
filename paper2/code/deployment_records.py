#!/usr/bin/env python3
"""What the archives of the external collections record about their deployments.

Reads every results/msi2/*/design.json and raw.json (the frozen percentile-only interface on the
development cohort and the two external cohorts, three backbones by three configurations) and
writes results/msi2/deployment_records.json: one record per backbone with the served name, the
endpoint the drivers called, the launch identity taken from the retained launcher, the collection
timestamps, the decoding settings, the call counts and what the replies themselves carry (the name
the server echoed, finish reasons, prompt-token sizes, attempts). Fields the archives do not hold
are listed under "not_recorded" rather than filled in from memory.
"""
import glob, json, os, statistics
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = sorted(glob.glob(os.path.join(ROOT, 'results', 'msi2', '*', 'design.json')))
OUT = os.path.join(ROOT, 'results', 'msi2', 'deployment_records.json')

# the launch identity of the locally served backbones, as the retained launcher issued it
# (code/provenance/run_msi2_local.sh); the first backbone was served by an endpoint we did not run
LAUNCH = {
    'mistral-small-3.2-24b-instruct': {
        'launcher': 'code/provenance/run_msi2_local.sh',
        'hub_id': 'mistralai/Mistral-Small-3.2-24B-Instruct-2506',
        'dtype': 'bfloat16', 'max_model_len': 8192,
        'template': 'plain chat (no template flag; the chat template has no reasoning switch)',
        'extra': 'Mistral tokenizer, configuration and load format; image inputs disabled',
        'served_by': 'vLLM on one accelerator, loopback host'},
    'gemma-4-31b-it': {
        'launcher': 'code/provenance/run_msi2_local.sh',
        'hub_id': 'google/gemma-4-31B-it',
        'dtype': 'bfloat16', 'max_model_len': 8192,
        'template': 'thinking disabled through the chat-template flag',
        'extra': '',
        'served_by': 'vLLM on one accelerator, loopback host'},
    'qwen3-8-27b': {
        'launcher': 'code/provenance/run_msi2_qwen.sh',
        'hub_id': None,
        'dtype': None, 'max_model_len': None,
        'template': 'thinking disabled through the chat-template flag',
        'extra': '',
        'served_by': 'a vLLM/Ray Serve deployment not under our control'},
}
ORDER = ['qwen3-8-27b', 'gemma-4-31b-it', 'mistral-small-3.2-24b-instruct']   # first, second, third


def utc(s):
    return datetime.fromisoformat(s.replace('+0700', '+07:00').replace('+0000', '+00:00')).astimezone(timezone.utc)


recs = {}
for p in RUNS:
    d = json.load(open(p))
    run = os.path.basename(os.path.dirname(p))
    m = d['model']
    r = recs.setdefault(m, {
        'served_name': m, 'urls': set(), 'collections': [], 'started_at_utc': [],
        'temperature': set(), 'max_tokens': set(), 'plain_chat': set(), 'retries': set(), 'seed': set(),
        'donor_seed': set(), 'minutes': 0.0, 'failed_calls': 0, 'calls': 0, 'echoed_names': {},
        'finish_reasons': {}, 'attempts_gt1': 0, 'prompt_tokens': [], 'latency_s': []})
    r['urls'].add(d['url']); r['collections'].append(run)
    r['started_at_utc'].append(utc(d['started_at']).strftime('%Y-%m-%d %H:%M UTC'))
    r['temperature'].add(d['temperature']); r['max_tokens'].add(d['max_tokens'])
    r['plain_chat'].add(bool(d.get('plain_chat'))); r['retries'].add(d.get('retries'))
    r['seed'].add(d.get('seed')); r['donor_seed'].add(d.get('donor_seed'))
    r['minutes'] += float(d.get('minutes') or 0)
    r['failed_calls'] += sum((d.get('failed_calls') or {}).values())
    rp = os.path.join(os.path.dirname(p), 'raw.json')
    if os.path.exists(rp):
        raw = json.load(open(rp))
        for e in raw.values():
            meta = e.get('meta') or {}
            r['calls'] += 1
            n = meta.get('model'); r['echoed_names'][n] = r['echoed_names'].get(n, 0) + 1
            f = meta.get('finish_reason'); r['finish_reasons'][f] = r['finish_reasons'].get(f, 0) + 1
            if (meta.get('attempts') or 1) > 1: r['attempts_gt1'] += 1
            u = meta.get('usage') or {}
            if u.get('prompt_tokens'): r['prompt_tokens'].append(u['prompt_tokens'])
            if meta.get('sent_at') and meta.get('received_at'):
                r['latency_s'].append(meta['received_at'] - meta['sent_at'])

out = {'source': 'results/msi2/*/design.json and raw.json', 'backbones': []}
for i, m in enumerate(ORDER):
    if m not in recs: continue
    r = recs[m]
    url = sorted(r['urls'])
    host = 'loopback' if all(u.startswith('http://127.0.0.1') for u in url) else 'remote'
    pt = sorted(r['prompt_tokens'])
    rec = {
        'ordinal': i + 1, 'served_name': m, 'url': url, 'host': host,
        'launch': LAUNCH.get(m, {}),
        'n_collections': len(r['collections']), 'collections': r['collections'],
        'first_started_utc': min(r['started_at_utc']), 'last_started_utc': max(r['started_at_utc']),
        'temperature': sorted(r['temperature']), 'max_tokens': sorted(r['max_tokens']),
        'plain_chat': sorted(r['plain_chat']), 'retries': sorted(r['retries']),
        'seed': sorted(r['seed']), 'donor_seed': sorted(r['donor_seed']),
        'minutes_total': round(r['minutes'], 1), 'calls_archived': r['calls'],
        'failed_calls': r['failed_calls'], 'calls_needing_retry': r['attempts_gt1'],
        'echoed_names': r['echoed_names'], 'finish_reasons': r['finish_reasons'],
        'prompt_tokens_median': statistics.median(pt) if pt else None,
        'prompt_tokens_min': pt[0] if pt else None, 'prompt_tokens_max': pt[-1] if pt else None,
        'latency_s_median': round(statistics.median(r['latency_s']), 2) if r['latency_s'] else None,
        'not_recorded': ['weight revision or artifact digest', 'tokenizer file identity',
                         'serving software version (known for the local backbones only from the '
                         'serving environment, not from the archives)'] +
                        (['serving precision', 'hardware'] if host == 'remote' else []),
    }
    out['backbones'].append(rec)

json.dump(out, open(OUT, 'w'), indent=1)
for b in out['backbones']:
    print(b['ordinal'], b['served_name'], b['host'], b['n_collections'], 'collections', b['calls_archived'],
          'calls', b['first_started_utc'], '..', b['last_started_utc'], b['echoed_names'], b['finish_reasons'])
print('wrote', OUT)
