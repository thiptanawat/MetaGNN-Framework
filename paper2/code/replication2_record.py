#!/usr/bin/env python3
"""The record of the plan and of the collection of the second replication of the own-versus-donor contrast.

Plan: paper2/REPLICATION2_PLAN_2026-11-14.md.

Writes results/replication2/record.json. Three kinds of facts go into it.

  1. What this script reads from the archives results/replication2_<tag>/w1 (tags b1, b2, b3): the time every
     call was sent and answered (the collector's clock, stored with each reply), the blocks, the arms of each
     block, the patients of each block, the calls sent again and the answers that could not be parsed. From
     these it derives the first call, the last reply, the hours between them, the calls answered per second
     over that time, and whether the blocks were collected in the order the plan fixes. Times are given in
     UTC and in the local time of the plan (UTC+7).
  2. What it reads from the serving records results/replication2_<tag>/serving_record.json: for the main
     backbone (b1) the service's own record reduced by replication_service_record.py (model source,
     replicas, time read); for the second and third backbones (b2, b3) the comparison of the snapshot's
     files with results/model_records.json, the libraries, the commit of the checkout that issued the
     calls and the time the record was written.
  3. What no file of the collection holds and is entered by hand: the commit of the plan and the time of its
     push as the public activity record of the repository gives it (PLAN_COMMIT, PLAN_PUSHED_UTC below), and
     the file results/replication2/course.json with the check calls sent before each collection, the
     departures from the plan and notes on the course of the collection.

Nothing in the record depends on an answer of a model. The script stops, with a message and a nonzero
status, if a statement of the plan that it can test does not hold: a first call that precedes the push of
the plan, a design_master_sha256 in an archive that differs from the SHA-256 of the sample file, blocks out
of order (they are 1 to 20, one after the other), an arm set that is not the one of the plan (the
patient-blind arm "reaction" in blocks 1 and 2 only, the four other arms in every block), patients that
are not the first ten per block of the ordered list of the sample, a serving record written after the first
call, a model source of the main backbone other than the one the plan names, a snapshot of the second or
third backbone that differs from the record, or fewer than 95% of the check answers parsed. With no
archive present it prints one line and ends with status 0.

    python3 code/replication2_record.py
"""
import calendar, hashlib, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
PLAN = 'REPLICATION2_PLAN_2026-11-14.md'
PLAN_COMMIT = 'e82533756fa19c38079e8ac29ccd4be9b1c1515d'
PLAN_PUSHED_UTC = '2026-11-14T14:55:49Z'     # the push of that commit to the public repository (its activity record)
PLAN_SAMPLE_SHA256 = 'eb1df4423b5e16d2b4887ad0a5bab42e7e99a4123d5ea3ab5025b22bcae546db'   # Section 3 of the plan
BACKBONES = dict(b1='qwen3-8-27b', b3='mistral-small-3.2-24b-instruct', b2='gemma-4-31b-it')   # the order of Section 4 of the plan
MODEL_SOURCE_B1 = 'Qwen/Qwen3.8-27B-FP8'      # Section 4: any other source, the main backbone is not collected
N_BLOCKS, BLOCK_SIZE, BLIND_BLOCKS = 20, 10, 2
ARMS_BLIND = ['reaction', 'patient', 'shuffled', 'shuffled_d2', 'shuffled_d3']
ARMS_OTHER = ['patient', 'shuffled', 'shuffled_d2', 'shuffled_d3']
LOCAL_HOURS = 7                                # the local time of the plan is UTC+7
EARLIER_LIBRARIES = dict(vllm='0.19.0', torch='2.10.0', transformers='5.5.4', tokenizers='0.22.2', mistral_common='1.11.0')


def utc(ts): return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(ts))
def loc(ts): return time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(ts + LOCAL_HOURS * 3600)) + '+%02d:00' % LOCAL_HOURS
def epoch(s): return calendar.timegm(time.strptime(s, '%Y-%m-%dT%H:%M:%SZ'))
def epoch_tz(s):
    """A time with a numeric offset, '2026-11-14T23:26:56+0900' or '...+09:00', as an epoch."""
    sign = -1 if s[19] == '-' else 1; hh = int(s[20:22]); mm = int(s[-2:])
    return calendar.timegm(time.strptime(s[:19], '%Y-%m-%dT%H:%M:%S')) - sign * (hh * 3600 + mm * 60)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''): h.update(chunk)
    return h.hexdigest()


def backbone(tag, name, master, master_sha, pushed):
    base = os.path.join(ROOT, 'results', f'replication2_{tag}')
    if not os.path.exists(os.path.join(base, 'w1', 'raw.json')): return None
    problems = []
    raw = json.load(open(os.path.join(base, 'w1', 'raw.json'))); des = json.load(open(os.path.join(base, 'w1', 'design.json')))
    if des['model'] != name: sys.exit(f'{tag}: the archive names the model {des["model"]!r}, not {name!r}')
    sent = [r['meta']['sent_at'] for r in raw.values()]; rcv = [r['meta']['received_at'] for r in raw.values()]
    first, last, n = min(sent), max(rcv), len(sent)
    done = [b for b in des['blocks'] if b.get('finished_at')]
    # blocks in order: numbered 1, 2, ..., one started after the other
    by_start = sorted(done, key=lambda b: (epoch_tz(b['started_at']), b['block']))
    numbers = [b['block'] for b in by_start]
    order_ok = numbers == list(range(1, len(done) + 1)) and len(done) <= N_BLOCKS
    if not order_ok: problems.append(f'blocks out of order: {numbers}')
    # the arms of each block
    arms_per_block = {str(b['block']): list(b['arms']) for b in by_start}
    arms_ok = all(sorted(b['arms']) == sorted(ARMS_BLIND if b['block'] <= BLIND_BLOCKS else ARMS_OTHER) for b in done)
    if not arms_ok: problems.append('an arm set that is not the one of the plan (the arm "reaction" in blocks 1 and 2 only, the four other arms in every block)')
    # patients: the first ten per block of the ordered list of the sample, in order
    want = [p for b in range(len(done)) for p in master['patients'][b * BLOCK_SIZE:(b + 1) * BLOCK_SIZE]]
    got = [p for b in by_start for p in b['patients']]
    patients_ok = (got == want)
    if not patients_ok: problems.append(f'the patients of the {len(done)} completed blocks are not the first {len(done) * BLOCK_SIZE} of the ordered list of the sample, in order')
    reactions_ok = (des['reactions'] == master['reactions'])
    if not reactions_ok: problems.append('the reactions of the archive are not those of the sample')
    sha_ok = (des['design_master_sha256'] == master_sha)
    if not sha_ok: problems.append('design_master_sha256 of the archive differs from the SHA-256 of results/replication2/design_master.json')
    if first < pushed: problems.append(f'the first call ({utc(first)}) precedes the push of the plan ({PLAN_PUSHED_UTC})')
    o = dict(served_name=name, model_in_archive=des['model'], blocks_complete=len(done), patients=len(got), reactions=len(des['reactions']),
             calls=n, first_call_utc=utc(first), first_call_local=loc(first), last_reply_utc=utc(last), last_reply_local=loc(last),
             hours=round((last - first) / 3600, 4), calls_per_second=round(n / (last - first), 3),
             rate_note='calls answered, divided by the time from the first call sent to the last reply received, the pauses between blocks '
                       '(for the main backbone, the days between the night windows) included',
             hours_between_push_and_first_call=round((first - pushed) / 3600, 2), plan_public_before_first_call=bool(pushed < first),
             calls_sent_again=sum(int(b.get('resent_calls', 0)) for b in des['blocks']),
             transport_errors_left=sum(int(b.get('transport_errors_left', 0)) for b in des['blocks']),
             unparsed=sum(int(b.get('unparsed', 0)) for b in des['blocks']),
             answers_without_a_probability=sum(1 for r in raw.values() if r.get('prob') is None),
             answers_cut_at_the_token_limit=sum(1 for r in raw.values() if r['meta'].get('finish_reason') != 'stop'),
             calls_with_more_than_one_attempt=sum(1 for r in raw.values() if r['meta'].get('attempts', 1) != 1),
             concurrent_requests=sorted({int(b['workers']) for b in des['blocks']}),
             temperature=des['temperature'], max_tokens=des['max_tokens'], plain_chat=bool(des.get('plain_chat')),
             block_order=[f'w1 block {b}' for b in numbers], block_order_as_planned=bool(order_ok),
             arms_per_block=arms_per_block, arms_as_planned=bool(arms_ok),
             patients_are_the_first_of_the_sample_in_order=bool(patients_ok), reactions_are_those_of_the_sample=bool(reactions_ok),
             design_master_sha256_in_archive=des['design_master_sha256'], design_master_sha256_of_file=master_sha,
             design_master_sha256_matches=bool(sha_ok), collector_sha256=des['code_sha256'])
    if o['answers_without_a_probability'] != o['unparsed'] + o['transport_errors_left']:
        problems.append('the block records and the answers disagree on the number of missing answers')
    S = json.load(open(os.path.join(base, 'serving_record.json')))
    if tag == 'b1':
        read_at = epoch_tz(S['read_at'])
        o.update(serving_record='the service\'s own record, reduced by code/replication_service_record.py',
                 model_source=S['model_source'], replicas=S['replicas'], read_at=S['read_at'], read_at_utc=utc(read_at),
                 application_status=S.get('application_status'), deployment_status=S.get('deployment_status'),
                 model_id=S.get('model_id'), ray_version=S.get('ray_version'), full_record_sha256=S.get('full_record_sha256'),
                 serving_record_before_first_call=bool(read_at < first))
        if S['model_source'] != MODEL_SOURCE_B1: problems.append(f'the service names the model source {S["model_source"]!r}, not {MODEL_SOURCE_B1!r}')
        if S['replicas'] != 2: problems.append(f'the service reports {S["replicas"]} replicas, not two')
    else:
        W = S['weights']; written = epoch(S['written_utc'])
        o.update(serving_record='code/replication_serving_record.py',
                 weights_identical_to_record=bool(W['identical_to_record']), weight_files=int(W['n_files']),
                 weight_bytes=int(W['total_bytes']), artifact_digest=W['artifact_digest'], revision=S['revision'],
                 other_files={f['name']: f['first_host'] for f in W.get('other_files', [])},
                 serving_record_written_utc=S['written_utc'], serving_record_before_first_call=bool(written < first),
                 serving_record_plan=S.get('plan'), study_commit=S['study_commit'], libraries=S['libraries'],
                 libraries_as_in_the_extension=all(S['libraries'].get(k) == v for k, v in EARLIER_LIBRARIES.items()),
                 python=S['host']['python'], accelerator=S['host']['accelerator'], host_label=S['host']['label'])
        if not o['weights_identical_to_record']: problems.append('the snapshot differs from results/model_records.json')
        if S.get('plan') != PLAN: problems.append(f'the serving record names the plan {S.get("plan")!r}')
        if S['study_commit'] != PLAN_COMMIT: problems.append(f'the calls were issued from commit {S["study_commit"]}, not from the commit of the plan')
    if not o['serving_record_before_first_call']: problems.append('the serving record was written after the first call')
    if problems: sys.exit(f'{tag}: ' + '; '.join(problems))
    return o


def main():
    B = {}
    master_path = os.path.join(ROOT, 'results', 'replication2', 'design_master.json')
    if not any(os.path.exists(os.path.join(ROOT, 'results', f'replication2_{t}', 'w1', 'raw.json')) for t in BACKBONES):
        print('no archive of the second replication under results/replication2_b1, _b2 or _b3; no record written'); return
    master = json.load(open(master_path)); master_sha = sha256_file(master_path); pushed = epoch(PLAN_PUSHED_UTC)
    course = json.load(open(os.path.join(ROOT, 'results', 'replication2', 'course.json')))
    for tag, name in BACKBONES.items():
        b = backbone(tag, name, master, master_sha, pushed)
        if b is None: continue
        c = course['check_calls'][tag]; b['check_calls_before_the_collection'] = c
        if c['parsed'] < 0.95 * c['calls']: sys.exit(f'{tag}: fewer than 95% of the check answers parsed; the plan does not allow the collection')
        B[tag] = b
    first = min(epoch(b['first_call_utc']) for b in B.values())
    tags = sorted(B, key=lambda t: epoch(B[t]['first_call_utc']))
    rec = dict(
        note='the plan of the second replication and the course of its collection; written by code/replication2_record.py from the archives, '
             'the serving records and results/replication2/course.json; times of calls are those of the collector\'s clock; '
             'local time is UTC+7, the local time of the plan',
        plan=PLAN, plan_commit=PLAN_COMMIT, plan_pushed_utc=PLAN_PUSHED_UTC,
        plan_pushed_source='the public activity record of the repository',
        sample_file='results/replication2/design_master.json', sample_sha256=master_sha, sample_sha256_in_plan=PLAN_SAMPLE_SHA256,
        sample_file_is_the_one_of_the_plan=bool(master_sha == PLAN_SAMPLE_SHA256),
        first_call_utc=utc(first), first_call_local=loc(first), first_call_backbone=tags[0],
        plan_public_before_first_call=bool(pushed < first and all(b['plan_public_before_first_call'] for b in B.values())),
        hours_between_push_and_first_call=round((first - pushed) / 3600, 2),
        hours_between_push_and_first_call_by_backbone={t: B[t]['hours_between_push_and_first_call'] for t in tags},
        backbones_with_an_archive=tags, backbones_without_an_archive=[t for t in BACKBONES if t not in B],
        collection_order=tags, backbones=B,
        notes=course.get('notes', []), departures=course.get('departures', []),
        departures_sentence=course.get('departures_sentence', 'No departure from the plan is recorded.'))
    json.dump(rec, open(os.path.join(ROOT, 'results', 'replication2', 'record.json'), 'w'), indent=1)
    print(f"plan {PLAN_COMMIT[:12]} pushed {PLAN_PUSHED_UTC}; first call {rec['first_call_utc']} ({tags[0]}), "
          f"{rec['hours_between_push_and_first_call']} h after the push; public before the first call of every backbone: {rec['plan_public_before_first_call']}")
    for t in tags:
        b = B[t]
        print(f"{t} ({b['served_name']}): {b['blocks_complete']} blocks, {b['patients']} patients, {b['calls']} calls, {b['first_call_utc']} to "
              f"{b['last_reply_utc']}, {b['hours']} h, {b['calls_per_second']} calls/s, sample hash matches: {b['design_master_sha256_matches']}, "
              f"sent again {b['calls_sent_again']}, unparsed {b['unparsed']}, transport errors left {b['transport_errors_left']}")


if __name__ == '__main__':
    main()
