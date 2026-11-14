#!/usr/bin/env python3
"""The record of the plan and of the collection of the replication on the two further backbones.

Plan: paper2/REPLICATION_BACKBONES_PLAN_2026-11-13.md.

Writes results/replication/backbones_record.json. Three kinds of facts go into it.

  1. What this script reads from the archives results/replication_<tag>/{w1,w2}: the time every call was
     sent and answered (the collector's clock, stored with each reply), the blocks, the calls sent again
     and the answers that could not be parsed. From these it derives the first call, the last reply, the
     hours between them, the calls answered per second over that time, and whether the blocks were
     collected in the order the plan fixes.
  2. What it reads from the serving records results/replication_<tag>/serving_record.json: the comparison
     of the snapshot's files with results/model_records.json, the libraries, the commit of the checkout
     that issued the calls and the time the record was written.
  3. What no file of the collection holds and is entered by hand: the commit of the plan and the time of
     its push as the public activity record of the repository gives it (PLAN_COMMIT, PLAN_PUSHED_UTC
     below), and the file results/replication/backbones_course.json with the check calls sent before
     each collection, the check of the tokenization on the two hosts and the departures from the plan.

Nothing in the record depends on an answer of a model. The script stops if a statement of the plan that
it can test does not hold: a backbone whose files differ from the record, a serving record written after
the first call, a checkout other than the plan's commit, blocks out of the planned order, or a first call
that precedes the push while the course file says the plan was public first.
"""
import calendar, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
PLAN = 'REPLICATION_BACKBONES_PLAN_2026-11-13.md'
PLAN_COMMIT = 'b041137acece3841153118609b46a03895cd40e2'
PLAN_PUSHED_UTC = '2026-11-13T14:59:22Z'     # the push of that commit to the public repository (its activity record)
BACKBONES = dict(b3='mistral-small-3.2-24b-instruct', b2='gemma-4-31b-it')     # in the order the plan collects them
ORDER = [('w1', [1, 2, 3, 4]), ('w2', [1, 2, 3, 4]), ('w1', [5, 6, 7, 8, 9, 10])]   # Section 5 of the plan
LIBRARIES = dict(vllm='0.19.0', torch='2.10.0', transformers='5.5.4', tokenizers='0.22.2', mistral_common='1.11.0')  # Section 3


def utc(ts): return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(ts))
def epoch(s): return calendar.timegm(time.strptime(s, '%Y-%m-%dT%H:%M:%SZ'))
def local(s): return calendar.timegm(time.strptime(s[:19], '%Y-%m-%dT%H:%M:%S')) - (int(s[19:22]) * 3600 + int(s[19] + s[22:24]) * 60)


def backbone(tag, name):
    base = os.path.join(ROOT, 'results', f'replication_{tag}')
    if not os.path.exists(os.path.join(base, 'w1', 'raw.json')): return None
    out = dict(served_name=name, wordings={}); sent, received, blocks = [], [], []
    for w in ('w1', 'w2'):
        if not os.path.exists(os.path.join(base, w, 'raw.json')): continue
        raw = json.load(open(os.path.join(base, w, 'raw.json'))); des = json.load(open(os.path.join(base, w, 'design.json')))
        if des['model'] != name: sys.exit(f'{tag} {w}: the archive names the model {des["model"]!r}')
        s = [r['meta']['sent_at'] for r in raw.values()]; rcv = [r['meta']['received_at'] for r in raw.values()]
        sent += s; received += rcv
        done = [b for b in des['blocks'] if b.get('finished_at')]
        out['wordings'][w] = dict(
            calls=len(raw), blocks_complete=len(done), patients=sum(len(b['patients']) for b in done),
            first_call_utc=utc(min(s)), last_reply_utc=utc(max(rcv)),
            calls_sent_again=sum(int(b.get('resent_calls', 0)) for b in des['blocks']),
            transport_errors_left=sum(int(b.get('transport_errors_left', 0)) for b in des['blocks']),
            unparsed=sum(int(b.get('unparsed', 0)) for b in des['blocks']),
            answers_without_a_probability=sum(1 for r in raw.values() if r.get('prob') is None),
            answers_cut_at_the_token_limit=sum(1 for r in raw.values() if r['meta'].get('finish_reason') != 'stop'),
            calls_with_more_than_one_attempt=sum(1 for r in raw.values() if r['meta'].get('attempts', 1) != 1),
            concurrent_requests=sorted({int(b['workers']) for b in des['blocks']}),
            temperature=des['temperature'], max_tokens=des['max_tokens'], plain_chat=bool(des.get('plain_chat')),
            design_master_sha256=des['design_master_sha256'], collector_sha256=des['code_sha256'])
        blocks += [(local(b['started_at']), w, b['block']) for b in done]
        o = out['wordings'][w]
        if o['answers_without_a_probability'] != o['unparsed'] + o['transport_errors_left']:
            sys.exit(f'{tag} {w}: the block records and the answers disagree on the number of missing answers')
    # the order of the plan: wording 1 blocks 1 to 4, wording 2 blocks 1 to 4, wording 1 blocks 5 to 10
    got = [(w, b) for _t, w, b in sorted(blocks)]; want = [(w, b) for w, bs in ORDER for b in bs]
    out['block_order'] = [f'{w} block {b}' for w, b in got]
    out['block_order_as_planned'] = (got == want[:len(got)])
    first, last = min(sent), max(received); n = len(sent)
    out.update(first_call_utc=utc(first), last_reply_utc=utc(last), calls=n, hours=round((last - first) / 3600, 4),
               calls_per_second=round(n / (last - first), 3),
               rate_note='calls answered, divided by the time from the first call sent to the last reply received, '
                         'the gaps between the three collection steps included')
    S = json.load(open(os.path.join(base, 'serving_record.json'))); W = S['weights']
    out.update(weights_identical_to_record=bool(W['identical_to_record']), weight_files=int(W['n_files']),
               weight_bytes=int(W['total_bytes']), artifact_digest=W['artifact_digest'], revision=S['revision'],
               other_files={f['name']: f['first_host'] for f in W.get('other_files', [])},
               serving_record_written_utc=S['written_utc'], serving_record_before_first_call=bool(epoch(S['written_utc']) < first),
               study_commit=S['study_commit'], libraries=S['libraries'], python=S['host']['python'],
               accelerator=S['host']['accelerator'], host_label=S['host']['label'])
    problems = []
    if not out['weights_identical_to_record']: problems.append('the snapshot differs from results/model_records.json')
    if not out['serving_record_before_first_call']: problems.append('the serving record was written after the first call')
    if out['study_commit'] != PLAN_COMMIT: problems.append(f'the calls were issued from commit {out["study_commit"]}, not from the commit of the plan')
    if not out['block_order_as_planned']: problems.append(f'blocks out of the planned order: {out["block_order"]}')
    for lib, v in LIBRARIES.items():
        if S['libraries'].get(lib) != v: problems.append(f'{lib} {S["libraries"].get(lib)} in place of {v}')
    if problems: sys.exit(f'{tag}: ' + '; '.join(problems))
    return out


def main():
    course = json.load(open(os.path.join(ROOT, 'results', 'replication', 'backbones_course.json')))
    B = {tag: backbone(tag, name) for tag, name in BACKBONES.items()}
    B = {t: b for t, b in B.items() if b is not None}
    if not B: sys.exit('no archive of either backbone under results/replication_b2 or results/replication_b3')
    first = min(epoch(b['first_call_utc']) for b in B.values()); pushed = epoch(PLAN_PUSHED_UTC)
    tags = sorted(B, key=lambda t: epoch(B[t]['first_call_utc']))
    T = course['tokenization_check']
    for name, hosts in T['backbones'].items():
        if len({json.dumps(h, sort_keys=True) for h in hosts.values()}) != 1: sys.exit(f'tokenization check: the two hosts differ for {name}')
    for tag, b in B.items():
        b['check_calls_before_the_collection'] = course['check_calls'][tag]
        if course['check_calls'][tag]['parsed'] < 0.95 * course['check_calls'][tag]['calls']:
            sys.exit(f'{tag}: fewer than 95% of the check answers parsed; the plan does not allow the collection')
    rec = dict(
        note='the plan of the replication on the two further backbones and the course of its collection; written by '
             'code/replication_backbones_record.py from the archives, the serving records and results/replication/backbones_course.json; '
             'times of calls are those of the collector\'s clock',
        plan=PLAN, plan_commit=PLAN_COMMIT, plan_pushed_utc=PLAN_PUSHED_UTC,
        plan_pushed_source='the public activity record of the repository',
        first_call_utc=utc(first), first_call_backbone=tags[0],
        plan_public_before_first_call=bool(pushed < first),
        hours_between_push_and_first_call=round((first - pushed) / 3600, 2),
        collection_order=tags, collection_order_as_planned=(tags == [t for t in BACKBONES if t in B]),
        backbones=B, tokenization_check=T, departures=course['departures'], departures_sentence=course['departures_sentence'])
    if not rec['collection_order_as_planned']: sys.exit(f'the backbones were collected in the order {tags}')
    if course.get('plan_public_before_first_call') is not rec['plan_public_before_first_call']:
        sys.exit('results/replication/backbones_course.json and the recorded times disagree on whether the plan was public before the first call')
    json.dump(rec, open(os.path.join(ROOT, 'results', 'replication', 'backbones_record.json'), 'w'), indent=1)
    print(f"plan {PLAN_COMMIT[:12]} pushed {PLAN_PUSHED_UTC}; first call {rec['first_call_utc']} ({tags[0]}), "
          f"{rec['hours_between_push_and_first_call']} h after the push; public before the first call: {rec['plan_public_before_first_call']}")
    for t in tags:
        b = B[t]
        print(f"{t} ({b['served_name']}): {b['calls']} calls, {b['first_call_utc']} to {b['last_reply_utc']}, {b['hours']} h, "
              f"{b['calls_per_second']} calls/s, order as planned: {b['block_order_as_planned']}, weight files {b['weight_files']} identical: "
              f"{b['weights_identical_to_record']}, sent again {sum(w['calls_sent_again'] for w in b['wordings'].values())}, "
              f"unparsed {sum(w['unparsed'] for w in b['wordings'].values())}")


if __name__ == '__main__':
    main()
