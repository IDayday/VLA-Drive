"""Recheck only the already-authorized boundary-3584 feedback payload once."""
import importlib.util
import json
import os
from pathlib import Path
import resource
import tempfile
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
spec = importlib.util.spec_from_file_location('feedback_support', ROOT / 'tools/st_world_policy/analyze_full_feedback_support.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
KEY = 'a53a189d6603e316f5eaf2254fe85d3903e82ec27dce9167b143e6eddfff33c0'


def main():
    start = time.monotonic()
    target = HERE / 'AUTHORIZED_LEDGER_VALIDATION.json'
    m.require(not target.exists(), 'Validation receipt already exists; do not reread automatically')
    plan = m.create_plan(m.RUN, HERE.parent / 'full_train_feedback_capacity_v1/LEDGER_METADATA.json',
                         '/mnt/project/st-world-policy-20261016/audits/consequence/full_train_feedback_support_v1')
    item = next(i for s in plan['shards'] for i in s['items'] if i['ledger'] == KEY)
    m.require(item['payload_bytes'] == 3441888 and item['payload_sha256_declared'] == '8019dd29b17efa79c1fd1a7211bdf09a0b5be93931db75cd4db95ad9a3fb122f', 'Only original authorized payload admitted')
    m.configure_cpu(min(os.sched_getaffinity(0)))
    with tempfile.TemporaryDirectory(prefix='authorized-validation-', dir=HERE) as tmp:
        budget = m.Budget(tmp, plan['plan_id'], 'authorized_single_ledger', time.time()+110, 16 << 20, 1 << 20, False)
        with m.wall_guard(budget.deadline):
            analysis = m.verify_and_analyze(plan, item, budget)
        legacy_path = Path('/mnt/project/VLA-Drive-st-world-policy-20261016/reports/st_world_policy/20261018/consequence_next_hypothesis_v1/RESULTS.json')
        legacy_raw = budget.read(legacy_path); legacy = json.loads(legacy_raw)
        signal = legacy['signal']; nonzero = analysis['nonzero_advantage_scene_presentations']
        m.require([nonzero[k] for k in ('C1', 'D', 'C', 'T')] == [signal['nonzero_advantage_scenes'][k] for k in ('C1', 'C2', 'C3', 'C4')], 'Signal counts differ')
        m.require(analysis['reward_value_decomposition'] == legacy['advantage_value_decomposition'], 'Covariance algebra differs')
        m.require(analysis['EP_only_scene_presentations'] == signal['EP_only_nonzero_advantage_scenes'], 'EP-only differs')
        m.require(analysis['centered_A_float32'] == [s['C1_advantages_float32'] for s in legacy['per_scene']], 'Actual float32 A differs')
        for key, row in analysis['terminal_ties'].items():
            m.require({k:v for k,v in row.items() if k != 'empty_event_pairs'} == legacy['terminal_ties'][key], 'Terminal/time tie counts differ')
        out = {'status': 'AUTHOR_VALIDATION_PASSED_NOT_INDEPENDENT_APPROVAL', 'analyzer_sha256': m.digest(Path(m.__file__).read_bytes()),
               'ledger': KEY, 'payload_sha256': item['payload_sha256_declared'], 'payload_bytes': item['payload_bytes'],
               'payload_reads': 1, 'legacy_report_sha256': m.digest(legacy_raw), 'legacy_mathematics_exact_match': True,
               'analysis': analysis, 'pooled_amplitude': m.amplitude_statistics([analysis]), 'budget': budget.state,
               'wall_seconds': time.monotonic()-start, 'cpu_seconds': resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime,
               'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
               'cuda_initialized': m.torch.cuda.is_initialized(), 'threads': m.torch.get_num_threads(),
               'new_scores': 0, 'model_loads': 0, 'full_manifest_reads': 0, 'optimizer_updates': 0}
        m.atomic(target, m.encoded(out), immutable=True)
    print(json.dumps({k:out[k] for k in ('status','wall_seconds','cpu_seconds','peak_rss_bytes')}))


if __name__ == '__main__':
    main()
