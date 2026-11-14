#!/usr/bin/env python3
"""Serving record of the main backbone as its service reports it, reduced to the fields that describe
the model and the engine.

The service that serves the main backbone publishes its own configuration (the list of applications of
the serving framework) and its version. This script reads the two records, as saved to files before the
first call of a collection, and keeps the fields that describe the model and the engine: host
addresses, node names and process identifiers are left out, and the SHA-256 of the full applications
record is stored so that the reduced record can be tied to it.

    python3 code/replication_service_record.py --applications APPLICATIONS.json --version VERSION.json \
        --read-at 2026-11-12T23:18:25+0700 --out results/replication2_b1/serving_record.json

Run on the two records saved before the first call of the first replication, it reproduces
results/replication/serving_config.json (the note apart, which names the collection).
"""
import argparse, hashlib, json, time

AP = argparse.ArgumentParser()
AP.add_argument('--applications', required=True); AP.add_argument('--version', required=True)
AP.add_argument('--read-at', required=True, help='local time at which the two records were read, as YYYY-MM-DDTHH:MM:SS+ZZZZ')
AP.add_argument('--application', default='qwen3-8-27b'); AP.add_argument('--out', required=True)
AP.add_argument('--note', default='serving configuration of the backbone as the service reported it before the first call of the '
                                  'second replication; reduced to the fields that describe the model and the engine (host addresses '
                                  'and identifiers are left out)')
A = AP.parse_args()

raw = open(A.applications, 'rb').read(); D = json.loads(raw); V = json.load(open(A.version))
app = D['applications'][A.application]; cfg = app['deployed_app_config']
llm = cfg['args']['llm_configs']
assert len(llm) == 1, 'more than one model in the application'
llm = llm[0]
dep = app['deployments'][f'LLMServer:{A.application}']
auto = dep['deployment_config'].get('autoscaling_config') or {}
local = lambda s: time.strftime('%Y-%m-%dT%H:%M:%S%z', time.localtime(s))
out = dict(
    note=A.note, read_at=A.read_at, application=app['name'], route_prefix=app['route_prefix'], import_path=cfg['import_path'],
    model_id=llm['model_loading_config']['model_id'], model_source=llm['model_loading_config']['model_source'],
    accelerator_type=llm.get('accelerator_type'), engine_kwargs=llm.get('engine_kwargs'),
    replicas=sum(1 for r in dep.get('replicas', []) if r.get('state') == 'RUNNING'),
    autoscaling=dict(min_replicas=auto.get('min_replicas'), max_replicas=auto.get('max_replicas')),
    max_ongoing_requests=dep['deployment_config'].get('max_ongoing_requests'),
    application_status=app.get('status'), deployment_status=dep.get('status'),
    application_deployed_at=local(app['last_deployed_time_s']),
    replica_started_at=sorted(local(r['start_time_s']) for r in dep.get('replicas', [])),
    ray_version=V.get('ray_version'), ray_commit=V.get('ray_commit'),
    full_record_sha256=hashlib.sha256(raw).hexdigest())
json.dump(out, open(A.out, 'w'), indent=1); open(A.out, 'a').write('\n')
print('wrote', A.out, '|', out['model_source'], '| replicas', out['replicas'], '|', out['application_status'], out['deployment_status'])
