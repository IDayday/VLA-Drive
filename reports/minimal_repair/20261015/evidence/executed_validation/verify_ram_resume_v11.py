import json, os, subprocess, sys, time
from pathlib import Path
import torch

w = Path('/mnt/project/VLA-Drive-minimal-repair-current-vector-v10-20261015')
sys.path.insert(0, str(w))
from tools.decision_local.common import atomic_json, check_host, digest
from tools.decision_local.verify_resume import equal

r = Path('/mnt/project/minimal-repair-artifacts/20261015')
o = Path('/mnt/project/decision-local-consequence-artifacts/20261014')
ram = Path('/dev/shm/decision-local-minimal-repair-v5-20261015')
resources = check_host()
receipt = json.loads((r/'RAM_CACHE_training-vla-zt_v10.json').read_text())
assert receipt['complete'] and receipt['all_chunk_bytes_and_metadata_exact']
out = r/'resume_RAM_v11'
out.mkdir(exist_ok=False)
base = [str(o/'envs/qwen/bin/python'), '-u', str(w/'tools/decision_local/train_small_model.py'),
        '--protocol', 'minimal_repair_v1', '--features', str(o/'anchor_train_full_v1'),
        '--labels', str(r/'targets_train64_v5'), '--kind', 'MIN_FUT', '--scope', 'smoke',
        '--epochs', '12', '--batch', '16', '--microbatch', '8', '--data-reader', 'mmap_tensors',
        '--max-seconds', '300']
jobs = []
for name, mode in [('continuous_A','shared'), ('continuous_B','shared'),
                   ('interrupted','stop7'), ('interrupted','resume_ram')]:
    argv = base + ['--output', str(out/name)]
    if mode == 'stop7': argv += ['--max-updates','7']
    if mode == 'resume_ram':
        argv[argv.index('--features')+1] = str(ram/'anchor_train_full_v1')
        argv += ['--resume']
    started = time.time()
    with (out/(name+'_'+mode+'.log')).open('x') as log:
        p = subprocess.run(argv, cwd=w, stdout=log, stderr=subprocess.STDOUT, timeout=330)
    jobs.append(dict(name=name, mode=mode, argv=argv, returncode=p.returncode,
                     started_unix=started, ended_unix=time.time()))
    atomic_json(out/'JOBS.json', jobs)
    p.check_returncode()
a = torch.load(out/'continuous_A/latest.pt', map_location='cpu', weights_only=False)
keys = ['identity','model','optimizer','scheduler','rng','completed','epoch','offset','exposure',
        'order','effective_optimizer_updates','exposure_counts']
results = {}
for name in ('continuous_B','interrupted'):
    b = torch.load(out/name/'latest.pt', map_location='cpu', weights_only=False)
    for key in keys: equal(a[key], b[key], key)
    results[name] = dict(exact=True, keys=keys, completed=a['completed'])
assert a['completed'] == 48
atomic_json(out/'ACCEPTANCE.json', dict(resources=resources, source_sha=a['identity']['source_sha'],
    script_sha256=digest(__file__), shared_features=str(o/'anchor_train_full_v1'),
    RAM_features=str(ram/'anchor_train_full_v1'), logical_feature_identity=a['identity']['features'],
    results=results, scope='Real64 scenes;48 updates;interruption7;unchanged frozen trainer;shared-to-RAM exact native state',
    canonical_data_changed=False, numerical_mode_changed=False))
print(json.dumps(results), flush=True)
