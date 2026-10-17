"""Explicit one-shot CPU-owner handoff; preserve the running 24-way screen."""
from pathlib import Path
import fcntl
import hashlib
import json
import os
import subprocess
import time
import uuid

from scripts.analysis.fm_transfer import endpoint_handoff as h
from starVLA.rl.flow_grpo.transactions import atomic_json

root = Path('/mnt/project/fm-transfer-20261016')
source = root / 'source/d8e4c6c7cdf11eed2a1f1ef8c3b84230b7e46999'
old = root / 'endpoint_handoff_B_v2'
new = root / 'endpoint_handoff_B_v3'
proof = root / 'qualification/B_endpoint_analysis_route_v3'
read = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert Path(h.__file__).resolve().is_relative_to(source)
assert not new.exists(), 'Existing migration attempt requires inspection, not replay'
prior = read(old / 'plan.json')
policy = read('/mnt/project/server_dispatch_policy.json')
assert set(prior['screen_hosts'] + [prior['training_host']]) <= set(policy['allowed_hosts'])
owner = read(old / 'owner.json')
stat = Path('/proc') / str(owner['pid']) / 'stat'
assert not stat.exists() or int(stat.read_text().split(') ', 1)[1].split()[19]) != owner['start_ticks'], \
    'Previous controller has not exited'
stopped = read(old / 'status.json')
assert stopped['status'] == 'STOPPED_CONTROLLER_ONLY' and stopped['existing_children_preserved']
assert not (old / 'failure.json').exists()
assert read(old / 'STOP_AFTER_TICK')['replacement_source'] == str(source)
assert not Path(prior['hard_manifest']).exists()
assert not Path(prior['gate']).exists() and not Path(prior['run']).exists()

value = h.plan(root, source, prior['screen_hosts'], prior['training_host'])
h.validate_plan(value)
h.validate_entrypoints(value)
allowed = {'schema', 'controller_source', 'screen_dispatch_source', 'sha256'}
assert all(value[k] == prior[k] for k in prior if k not in allowed)
assert set(value) - set(prior) == {'screen_analysis_source'}
assert value['stages'] == prior['stages']
assert read(proof / 'cli_validation.json')['no_model_or_scoring_execution']
assert read(proof / 'immutable_entrypoints.json')['plan_sha256'] == value['sha256']

screen = Path(value['screen'])
dispatch_paths = sorted(screen.glob('dispatches/*_rank*.json'))
dispatches = [read(p) for p in dispatch_paths]
assert len(dispatches) == 24 and sorted(x['rank'] for x in dispatches) == list(range(24))
assert all(x['source'] == value['worker_source'] and x['global_shards'] == 24 for x in dispatches)
screen_probes = {}
for host in sorted({x['host'] for x in dispatches}):
    records = [x for x in dispatches if x['host'] == host]
    alive = h.processes(host, records, 'scripts.analysis.fm_transfer.screen_train')
    assert all(alive), 'A screen worker needs inspection; no implicit retry'
    screen_probes[host] = dict(ranks=[x['rank'] for x in records], same_live_processes=alive)

adopt_names = ('sft_summary', 'screen_0', 'screen_1', 'screen_2')
parents = {}
for name in adopt_names:
    stage = old / 'stages' / name
    intent, done = read(stage / 'intent.json'), read(stage / 'status.json')
    assert intent['plan_sha256'] == done['plan_sha256'] == prior['sha256']
    assert intent['attempt'] == done['attempt'] and intent['command'] == value['stages'][name]
    assert done['state'] == 'COMPLETE' and done['exit_code'] == 0
    assert not h.processes(intent['host'], [done], 'scripts.analysis.fm_transfer.endpoint_handoff')[0]
    if 'child' in done:
        assert not h.processes(intent['host'], [done['child']], intent['command'][2])[0]
    parents[name] = (stage, intent, done)
assert all(not (old / 'stages' / x).exists() for x in
           ('hard_manifest', 'screen_summary', 'endpoint_gate', 'formal_RL'))

with (root / 'B_endpoint_handoff.owner.lock').open('a') as guard:
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    new.mkdir()
    atomic_json(new / 'plan.json', value)
    atomic_json(new / 'physical_hosts.json', read(old / 'physical_hosts.json'))
    adoptions = {}
    for name, (parent, intent, done) in parents.items():
        adoption = dict(parent=str(parent), intent_sha256=sha(parent / 'intent.json'),
                        status_sha256=sha(parent / 'status.json'), adopted_unix=time.time(),
                        physical_command_reexecuted=False)
        attempt = 'adopted-' + uuid.uuid4().hex
        stage = new / 'stages' / name
        stage.mkdir(parents=True)
        atomic_json(stage / 'intent.json', dict(intent, attempt=attempt,
                    plan_sha256=value['sha256'], adoption=adoption))
        atomic_json(stage / 'status.json', dict(done, attempt=attempt,
                    plan_sha256=value['sha256'], adoption=adoption))
        adoptions[name] = adoption
    receipt = dict(schema='fm_transfer_B_live_screen_analysis_route_migration_v1',
        old_control=str(old), new_control=str(new), old_plan_sha256=prior['sha256'],
        new_plan_sha256=value['sha256'], old_owner=owner,
        old_stopped_status_sha256=sha(old / 'status.json'),
        stop_request_sha256=sha(old / 'STOP_AFTER_TICK'),
        preflight_sha256=sha(proof / 'immutable_entrypoints.json'),
        adopted_completed_stages=adoptions, original_files_preserved=True,
        no_sampling_or_scoring_replayed=True, live_screen_processes_preserved=screen_probes,
        preserved_screen_manifest_sha256=sha(screen / 'manifest.json'),
        preserved_dispatch_sha256={str(p):sha(p) for p in dispatch_paths},
        existing_GPU_workers_signalled=0, physical_stage_commands_reexecuted=0,
        scientific_configuration_unchanged=True, stage_commands_unchanged=True,
        numerical_worker_source=value['worker_source'],
        screen_analysis_source=value['screen_analysis_source'],
        script_sha256=sha(Path(__file__)), observed_unix=time.time())
    atomic_json(new / 'migration_receipt.json', receipt)

env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(source),
           OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
command = [h.PYTHON, '-m', 'scripts.analysis.fm_transfer.endpoint_handoff',
           '--control', str(new), '--screen-hosts', *value['screen_hosts'],
           '--training-host', value['training_host']]
with (new / 'controller.log').open('xb') as log:
    child = subprocess.Popen(command, cwd=source, env=env, stdin=subprocess.DEVNULL,
                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
launch = dict(pid=child.pid,
    start_ticks=int((Path('/proc') / str(child.pid) / 'stat').read_text().split(') ', 1)[1].split()[19]),
    command=command, source=str(source), start_unix=time.time())
atomic_json(new / 'controller.launch.json', launch)
print(json.dumps(dict(status='EXPLICIT_V3_CONTROLLER_LAUNCHED', launch=launch,
                     completed_stages_adopted=list(adoptions), preserved_live_screen_workers=24)))
