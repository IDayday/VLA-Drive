"""Read-only postprocessing of the two already registered full Navtest jobs."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/mnt/project/structured-world-fgtr-round1-artifacts/20261008')
OUT = ROOT / 'navtest_recovery_20261012T1341Z'
QUEUE = ROOT / 'navsim_navtest_dispatch_20261011T0225'
SOURCE = Path('/mnt/project/VLA-Drive-structured-world-navtest-oom-recovery-frozen-20261012/tools/structured_world')
PYTHON = '/root/miniconda3/envs/navsim/bin/python'
spec = importlib.util.spec_from_file_location('frozen_dispatch', SOURCE / 'run_navsim_navtest.py')
dispatcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dispatcher)

targets = [
    ('G1_FULL_UNIFORM_0060000', 'G1_60K_COMPARISONS.json', {
        'same_G1_50k': 'G1_FULL_UNIFORM_0050000',
        'same_G1_25k': 'G1_FULL_UNIFORM_0025000'}),
    ('G3_EVENT_LOCAL_0090000', 'G3_90K_COMPARISONS.json', {
        'same_G3_80k': 'G3_EVENT_LOCAL_0080000',
        'same_G3_70k': 'G3_EVENT_LOCAL_0070000',
        'same_G3_50k': 'G3_EVENT_LOCAL_0050000'})]

for key, name, references in targets:
    job = QUEUE / 'jobs' / key
    deadline = time.monotonic() + 3 * 3600
    while not (QUEUE / 'completed' / (key + '.json')).exists():
        if (job / 'INFERENCE_FAILED.json').exists():
            raise RuntimeError('Evaluation failed; original failure retained: ' + key)
        if time.monotonic() > deadline:
            raise RuntimeError('Postprocessing wait expired; no evaluation worker was changed')
        time.sleep(15)
    req = json.loads((QUEUE / 'requests' / (key + '.json')).read_text())
    assert dispatcher.valid_score_completion(job, req)
    cmd = [PYTHON, '-B', str(SOURCE / 'analyze_navsim_navtest.py'),
           '--scores', str(job / 'scores'), '--output', str(OUT / name)]
    for label, reference in references.items():
        cmd.extend(['--reference', label + '=' + str(QUEUE / 'jobs' / reference / 'scores/q_final/scenes.csv')])
    if (OUT / name).exists():
        raise RuntimeError('Analysis already exists; inspect it instead of overwriting: ' + name)
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    subprocess.run(cmd, env=env, check=True)
    dispatcher.atomic(OUT / name.replace('.json', '_COMMAND.json'), {
        'argv': cmd, 'validation_source_sha256': dispatcher.EXECUTED_TOOL_SHA256,
        'full_population_validated': True, 'finished_utc': dispatcher.now()})
    print(json.dumps({'finished': key, 'output': str(OUT / name)}), flush=True)
