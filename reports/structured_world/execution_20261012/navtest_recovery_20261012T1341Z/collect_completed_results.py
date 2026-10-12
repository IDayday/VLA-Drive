"""Export aggregate, complete Navtest evidence without scenes, images or weights."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path('/mnt/project/structured-world-fgtr-round1-artifacts/20261008')
QUEUE = ROOT / 'navsim_navtest_dispatch_20261011T0225'
OUT = ROOT / 'navtest_recovery_20261012T1341Z'
TARGETS = ['G1_FULL_UNIFORM_0060000', 'G3_EVENT_LOCAL_0090000']

def read(path):
    return json.loads(path.read_text())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

for key in TARGETS:
    proof = OUT / ('G1_60K_COMPARISONS_COMMAND.json' if key.startswith('G1') else 'G3_90K_COMPARISONS_COMMAND.json')
    assert read(proof)['full_population_validated'] is True
    assert read(QUEUE / 'completed' / (key + '.json'))['valid'] is True

curves, banks, costs = [], {}, {}
for request_path in sorted((QUEUE / 'requests').glob('*.json')):
    req = read(request_path)
    job = QUEUE / 'jobs' / req['key']
    scores = job / 'scores/COMPLETE.json'
    if not scores.exists():
        continue
    s = read(scores)
    identity = read(job / 'identity.json')
    for variant in ['q0', 'q_final']:
        value = s[variant]
        assert value['scenes'] == 12146 and value['valid'] is True and value['failed'] == 0
        assert sha(job / 'scores' / variant / 'scenes.csv') == value['scene_csv_sha256']
        curves.append(dict(key=req['key'], group=req['group'], updates=req['updates'],
            variant=variant, scenes=12146, failed=0,
            zero_scenes=round(value['zero_fraction'] * 12146),
            **{name: metric * 100 for name, metric in value['metrics'].items()}))
    banks[req['key']] = dict(request=req, prediction_identity=identity,
        canonical_scores=s, scores_complete_sha256=sha(scores))
    if req['key'] in TARGETS:
        reports = [read(p) for p in job.glob('attempts/*/shards/*.json')]
        completed = [r for r in reports if r['status'] == 'COMPLETE']
        assert completed
        load_s = sum((datetime.fromisoformat(r['loaded_utc']) -
            datetime.fromisoformat(r['started_utc'])).total_seconds() for r in completed)
        process_s = sum((datetime.fromisoformat(r['completed_utc']) -
            datetime.fromisoformat(r['started_utc'])).total_seconds() for r in completed)
        costs[req['key']] = dict(
            hosts=sorted({r['host'] for r in completed}),
            unique_physical_gpus=len({r['visible_gpu'] for r in completed}),
            completed_worker_count=len(completed),
            preserved_failed_worker_reports=sum(r['status'] == 'FAILED' for r in reports),
            successful_worker_resident_hours=process_s / 3600,
            successful_worker_load_hours=load_s / 3600,
            successful_inference_process_hours=sum(r['inference_seconds'] for r in completed) / 3600,
            deliberate_between_scene_idle_process_hours=sum(r.get('scheduling_idle_seconds', 0) for r in completed) / 3600,
            peak_reserved_GiB=max(r['peak_reserved_bytes'] for r in completed) / 2**30,
            peak_allocated_GiB=max(r['peak_allocated_bytes'] for r in completed) / 2**30,
            coverage=read(job / 'COMPLETE.json'),
            accounting_limits='Process wall-hours on shared GPUs, not exclusive GPU-hours or kernel-active time. Failed pre-load attempt, qualification and CPU scoring costs are separate; successful inference includes host gaps and pacing.')

with (OUT / 'LEARNING_CURVES.csv').open('x', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(curves[0]))
    writer.writeheader()
    writer.writerows(curves)
with (OUT / 'COMPLETE_RESULTS.json').open('x') as stream:
    json.dump(dict(utc=datetime.now(timezone.utc).isoformat(), banks=banks, costs=costs,
        caveats=['One training seed per group', 'Intermediate Navtest observations were explicitly requested',
                 'q0 is an internal proposal, not an independent baseline',
                 'Different updates do not isolate group effects']), stream, indent=2)
    stream.write('\n')
print(json.dumps({'complete_banks': len(banks), 'curve_rows': len(curves), 'new_targets': TARGETS}))
