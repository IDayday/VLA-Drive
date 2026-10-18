"""CPU fixtures only. No training assets or scoring are opened by these tests."""
import ast
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('feedback_support', ROOT / 'tools/st_world_policy/analyze_full_feedback_support.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = m.encoded(value)
    path.write_bytes(raw)
    return raw


def event_record(scene, candidate):
    # Equal terminal failures at different physical times, plus empty safe ties.
    d = (2 + 10 * candidate,) if scene % 8 == 0 and candidate in (1, 2) else ()
    c = ((3 + 5 * candidate, .5),) if scene % 8 == 1 and candidate in (1, 2) else ()
    t = ((5, 3 * candidate),) if scene % 8 == 2 and candidate in (1, 2) else ()
    flags = m.np.zeros(41, dtype=bool)
    flags[list(d)] = True
    db, cb, tb = [False] * 8, [1.] * 8, [False] * 8
    for x in d: db[m.bin_of(x)] = True
    for x, severity in c: cb[m.bin_of(x)] = min(cb[m.bin_of(x)], severity)
    for x, _ in t: tb[m.bin_of(x)] = True
    metrics = {'drivable_area_compliance': 1. - bool(d), 'no_at_fault_collisions': min(cb),
               'time_to_collision_within_bound': 1. - bool(t), 'ego_progress': .8 + (.02 * candidate if scene % 2 else 0.),
               'comfort': 1., 'driving_direction_compliance': 1.}
    metrics['score'] = metrics['no_at_fault_collisions'] * metrics['drivable_area_compliance'] * (5 * metrics['ego_progress'] + 5 * metrics['time_to_collision_within_bound'] + 2) / 12
    return {'metrics': metrics, 'consequences': {
        'road': {'valid': True, 'native_flag': flags, 'segment_any': db, 'any': bool(d), 'event': bool(d), 'first_time_seconds': min(d) * .1 if d else None},
        'collision_summary': {'valid': True, 'segment_nc_min': cb, 'nc_min': min(cb), 'event': bool(c), 'first_at_fault_time_seconds': min(x for x, _ in c) * .1 if c else None},
        'ttc_summary': {'valid': True, 'segment_any': tb, 'any': bool(t), 'event': bool(t), 'first_state_time_seconds': min(x for x, _ in t) * .1 if t else None},
        'collision_events': [{'at_fault': True, 'state_index': x, 'state_time_seconds': x * .1, 'nc_contribution': s} for x, s in c],
        'ttc_events': [{'violation': True, 'state_index': x, 'state_time_seconds': x * .1, 'projection_offset_steps': y, 'observation_index': x+y, 'observation_time_seconds': (x+y)*.1} for x, y in t]}}


class Fixture:
    def __init__(self, root, sizes=(32, 24)):
        self.run = root / 'run'; self.output = root / 'out'; self.rows = []
        self.identity = {'identity': 'fixture', 'base_binding': 'base', 'numeric_execution_hash': 'numeric',
                         'config': {'cell': 'C1', 'execution': {'world_size': 4, 'global_batch': 32},
                                    'feedback': {'campaign_id': 'campaign', 'score_protocol_hash': 'protocol'}}}
        write(self.run / 'identity.json', self.identity)
        write(self.run / 'official_feedback/IDENTITY.json', {'identity': 'fixture'})
        write(self.run / 'RESULT.json', {'status': 'COMPLETE', 'completed': len(sizes), 'policy_version': len(sizes),
                                       'identity': 'fixture', 'exposure': sum(sizes),
                                       'counts': {'original_GT_scenes': sum(sizes), 'actual_rollouts': 4 * sum(sizes)}})
        for version, scenes in enumerate(sizes): self.add_ledger(version, scenes)
        self.metadata = root / 'metadata.json'; write(self.metadata, self.rows)
        self.plan = m.create_plan(self.run, self.metadata, self.output)
        self.path = root / 'plan.json'; write(self.path, self.plan)

    def add_ledger(self, version, scenes):
        records, requests = [], []
        slots = scenes // 4
        for rank in range(4):
            rewards, contexts = [], []
            rollout = 'rollout-%d-%d' % (version, rank)
            for slot in range(slots):
                scene = rank * slots + slot; cache = 'cache-%d' % scene; contexts.append(cache)
                reference = m.np.zeros((41, 3), dtype=m.np.float64)
                inp = {'full_cache_sha256': cache, 'score_protocol_hash': 'protocol', 'context_hash': cache,
                       'reference_interpolated_states_sha256': m.array_hash(reference)}
                group = []
                for candidate in range(4):
                    r = event_record(scene, candidate)
                    trajectory = m.np.full((2, 3), candidate, dtype=m.np.float64)
                    origin = {'kind': 'in_memory_parameter_snapshot', 'policy_version': str(version), 'optimizer_step': version,
                              'state_sha256': 'state-%d' % version, 'numeric_execution_hash': 'numeric'}
                    q = {'candidate_id': '%s:%d:%d' % (rollout, slot, candidate), 'scene_id': 'scene-%d' % scene,
                         'source_log': 'log-%d' % (scene // 2), 'generator_identity': origin, 'generator_kind': 'policy_snapshot',
                         'generator_source_hash': m.identity(origin), 'policy_version': str(version), 'inference_runtime_hash': 'numeric',
                         'score_protocol_hash': 'protocol', 'campaign_id': 'campaign', 'repetition': 0,
                         'metric_cache_sha256': cache, 'trajectory_sha256': m.array_hash(trajectory)}
                    execution = {'candidate_executed_states_sha256': m.array_hash(trajectory)}
                    r.update(identity=q, status='VALID_SCORED', valid=True, score_units='0..1', context_unchanged=True,
                             metrics_valid={k: True for k in m.METRICS}, requested_trajectory=trajectory,
                             reference_interpolated_states=reference, executed_states=trajectory, context_hash=cache,
                             reference_contract={'schema': 'st_world_reference_input_and_execution_v2', 'input': inp, 'input_sha256': m.identity(inp),
                                                 'execution': execution, 'execution_sha256': m.identity(execution)})
                    requests.append(q); records.append(r); group.append(float(m.np.float32(r['metrics']['score'])))
                rewards.append(group)
            path = self.run / 'rollouts' / ('boundary_%06d' % version) / ('rank_%d_COMPLETE.json' % rank)
            write(path, {'identity': {'old_policy': {'version': version, 'optimizer_step': version, 'state_hash': 'state-%d' % version, 'base_binding_hash': 'base'},
                                     'numeric_execution_hash': 'numeric', 'rollout_id': rollout,
                                     'feedback': {'rewards': rewards, 'contexts': contexts, 'statuses': ['valid_scored'] * (slots*4), 'protocol': 'protocol', 'reward_kind': 'C1-PDMS'}},
                         'payload': {'path': str(path.with_name('rank_%d.pt' % rank)), 'sha256': 'not-read'}})
        key = m.identity(requests); folder = self.run / 'official_feedback' / key; attempt = folder / 'attempt_00'
        write(folder / 'REQUESTS.json', {'policy_version': str(version), 'requests': requests})
        admitted = write(attempt / 'ADMITTED.json', {'request_hash': key, 'policy_version': str(version), 'calls': len(requests), 'indices': list(range(len(requests)))})
        stream = io.BytesIO(); m.torch.save(records, stream); raw = stream.getvalue(); (attempt / 'RESULTS.pt').write_bytes(raw)
        complete = write(attempt / 'COMPLETE.json', {'result': {'path': str(attempt / 'RESULTS.pt'), 'sha256': m.digest(raw)}, 'admission_sha256': m.digest(admitted)})
        self.rows.append({'ledger': key, 'attempt_count': 1, 'complete_attempts': [{'attempt': 'attempt_00', 'payload_bytes': len(raw), 'payload_sha256_declared': m.digest(raw), 'complete_sha256': m.digest(complete)}]})


class FullFeedbackTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'): os.environ[k] = '1'
        m.load_numeric()
        cls.legacy_source = subprocess.check_output(['git', 'show', 'cafc25a6096703b78a83b1b0d803a480545111af:reports/st_world_policy/20261018/consequence_next_hypothesis_v1/analyze_existing.py'], cwd=ROOT).decode()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.affinity = os.sched_getaffinity(0); self.cpu = min(self.affinity)

    def tearDown(self):
        os.sched_setaffinity(0, self.affinity); self.temp.cleanup()

    def budget(self):
        return m.Budget(self.root / 'budget', 'fixture', 'test', time.time() + 110, 32 << 20, 2 << 20, False)

    def test_pure_math_ast_and_numeric_alignment(self):
        names = ('require', 'digest', 'identity', 'array_hash', 'bin_of', 'process_profile', 'advantages', 'decompose', 'tie_counts')
        old = {n.name: n for n in ast.parse(self.legacy_source).body if isinstance(n, ast.FunctionDef)}
        new = {n.name: n for n in ast.parse(Path(m.__file__).read_text()).body if isinstance(n, ast.FunctionDef)}
        for name in names:
            a = copy.deepcopy(old[name]); b = copy.deepcopy(new[name])
            if name == 'tie_counts':
                loops = [x for x in ast.walk(a) if isinstance(x, ast.For) and isinstance(x.target, ast.Name) and x.target.id == 's']
                loops[0].iter = copy.deepcopy(next(x for x in ast.walk(b) if isinstance(x, ast.For) and isinstance(x.target, ast.Name) and x.target.id == 's').iter)
            self.assertEqual(ast.dump(a), ast.dump(b), name)
        namespace = {'np': m.np, 'torch': m.torch, 'itertools': m.itertools, 'hashlib': m.hashlib, 'json': json}
        unit = ast.Module(body=[old[x] for x in names] + [old['synthetic_checks']], type_ignores=[])
        exec(compile(unit, '<pure legacy AST>', 'exec'), namespace)
        self.assertEqual(namespace['synthetic_checks']()['control_count'], 9)
        records = [event_record(s, c) for s in range(32) for c in range(4)]
        profiles = [m.process_profile(r) for r in records]
        self.assertEqual(namespace['tie_counts'](records, profiles), m.tie_counts(records, profiles))
        metrics = {k: m.np.array([r['metrics'][k] for r in records]).reshape(32, 4) for k in m.METRICS}
        self.assertEqual(namespace['decompose'](metrics)[2], m.decompose(metrics)[2])
        self.assertEqual(m.tie_counts(records[:96], profiles[:96])['safety_vector_tie']['eligible_scenes'], 24)

    def test_plan_no_payload_reads_and_inventory_rejection(self):
        f = Fixture(self.root)
        real_open = Path.open
        def guarded(path, *a, **kw):
            self.assertNotEqual(path.name, 'RESULTS.pt')
            return real_open(path, *a, **kw)
        with mock.patch.object(Path, 'open', guarded):
            plan = m.create_plan(f.run, f.metadata, f.output)
        self.assertEqual(plan['expected']['scene_presentations'], 56)
        rows = copy.deepcopy(f.rows); rows[0]['complete_attempts'] = []
        write(f.metadata, rows)
        with self.assertRaisesRegex(ValueError, 'one complete'): m.create_plan(f.run, f.metadata, f.output)

    def test_single_read_integrity_and_tail(self):
        f = Fixture(self.root); b = self.budget(); items = [i for s in f.plan['shards'] for i in s['items']]
        counts = {}
        real_open = Path.open
        def counted(path, *a, **kw):
            if path.name == 'RESULTS.pt': counts[str(path)] = counts.get(str(path), 0) + 1
            return real_open(path, *a, **kw)
        with mock.patch.object(Path, 'open', counted):
            analyses = [m.verify_and_analyze(f.plan, i, b) for i in items]
        self.assertEqual(sorted(a['scene_presentations'] for a in analyses), [24, 32])
        self.assertEqual(list(counts.values()), [1, 1])
        self.assertEqual(b.state['nominal_load_bytes'], sum(i['payload_bytes'] for i in items))
        result = m.aggregate(analyses)
        self.assertEqual(result['unique_scenes'], 32)
        self.assertEqual(result['counts']['scene_presentations'], 56)
        self.assertEqual(result['scene_exposure_distribution']['histogram_scene_count_by_presentations'], {1: 8, 2: 24})
        for analysis in analyses:
            for condition, ties in analysis['terminal_ties'].items():
                self.assertEqual(len(analysis['support_scene_slots']['nonempty_time_or_severity_diff_' + condition]), ties['physical_time_or_severity_diff_scenes'])
        for label, support in result['support_unique_population'].items():
            self.assertLessEqual(support['unique_scenes'], support['scene_presentations'])
            self.assertLessEqual(support['unique_logs'], support['unique_scenes'])
        self.assertGreater(result['terminal_ties_over_scene_presentations']['D_terminal_tie']['physical_time_or_severity_diff_pairs'], 0)
        self.assertGreater(result['terminal_ties_over_scene_presentations']['D_terminal_tie']['empty_event_pairs'], 0)

    def test_hash_failure_preserved_and_explicit_resume(self):
        f = Fixture(self.root); shard = next(b for b in f.plan['shards'] if b['items'])
        item = shard['items'][0]; path = f.run / 'official_feedback' / item['ledger'] / 'attempt_00/RESULTS.pt'
        good = path.read_bytes(); path.write_bytes(good[:-1] + bytes([good[-1] ^ 1]))
        deadline = time.time() + 110
        with self.assertRaisesRegex(ValueError, 'hash mismatch'): m.worker(f.path, shard['shard'], self.cpu, deadline)
        folder = f.output / ('shard_%02d' % shard['shard'])
        self.assertTrue((folder / 'ERROR_0001.json').exists()); self.assertFalse((folder / 'SHARD_COMPLETE.json').exists())
        spent = json.loads((folder / 'BUDGET.json').read_bytes())['logical_read_bytes']
        path.write_bytes(good)
        with self.assertRaisesRegex(ValueError, 'explicit --resume'): m.worker(f.path, shard['shard'], self.cpu, deadline)
        m.worker(f.path, shard['shard'], self.cpu, deadline, resume=True)
        self.assertGreater(json.loads((folder / 'BUDGET.json').read_bytes())['logical_read_bytes'], spent)
        self.assertEqual(m.worker(f.path, shard['shard'], self.cpu, deadline)['status'], 'SHARD_COMPLETE')
        with self.assertRaisesRegex(ValueError, 'Immutable/stale'): m.worker(f.path, shard['shard'], self.cpu, deadline+1, resume=True)

    def test_precharge_rejects_before_payload_open(self):
        path = self.root / 'payload'; path.write_bytes(b'0123456789')
        b = self.budget(); b.read_limit = 19
        with mock.patch.object(Path, 'open', side_effect=AssertionError('must not open')):
            with self.assertRaisesRegex(ValueError, 'before read/load'): b.read(path, payload=True)
        self.assertEqual(b.state['actual_read_bytes'], 0)

    def test_bandwidth_tokens_persist_across_resume(self):
        b = self.budget(); now = [time.time()]
        b.state['io_updated_unix'] = now[0]; b.save(); sleeps = []
        def sleep(duration):
            sleeps.append(duration); now[0] += duration
        with mock.patch.object(m.time, 'time', side_effect=lambda: now[0]), mock.patch.object(m.time, 'sleep', side_effect=sleep):
            b.pace(m.READ_BURST)
            b.pace(3 << 20)
            restored = m.Budget(b.folder, 'fixture', 'test', b.deadline, b.read_limit, b.output_limit, True)
            restored.pace(1 << 20)
        self.assertEqual(sleeps, [.75, .25])
        self.assertEqual(restored.state['io_tokens'], 0.)

    def test_resume_does_not_reread_completed_ledger(self):
        f = Fixture(self.root, (24,) * 9); shard = next(s for s in f.plan['shards'] if len(s['items']) == 2)
        original = m.verify_and_analyze; completed = shard['items'][0]['ledger']; calls = []
        def fail_second(plan, item, budget):
            calls.append(item['ledger'])
            if item['ledger'] != completed: raise RuntimeError('synthetic interruption')
            return original(plan, item, budget)
        deadline = time.time()+110
        with mock.patch.object(m, 'verify_and_analyze', side_effect=fail_second):
            with self.assertRaisesRegex(RuntimeError, 'synthetic interruption'): m.worker(f.path, shard['shard'], self.cpu, deadline)
        calls.clear()
        def observed(plan, item, budget):
            calls.append(item['ledger']); return original(plan, item, budget)
        with mock.patch.object(m, 'verify_and_analyze', side_effect=observed):
            m.worker(f.path, shard['shard'], self.cpu, deadline, resume=True)
        self.assertNotIn(completed, calls); self.assertEqual(len(calls), 1)

    def test_merge_requires_all_shards_and_all_boundaries(self):
        f = Fixture(self.root); deadline = time.time() + 110
        self.assertEqual(m.merge(f.path)['status'], 'INCOMPLETE'); self.assertFalse((f.output / 'RESULT.json').exists())
        for s in range(8): m.worker(f.path, s, self.cpu, deadline)
        result = m.merge(f.path, self.cpu)
        self.assertEqual(result['status'], 'COMPLETE'); self.assertEqual(result['counts']['candidate_presentations'], 224)
        self.assertEqual(m.merge(f.path, self.cpu), result)
        # Re-sealed but duplicated policy boundary must still fail full coverage.
        item_shard = next(s for s in f.plan['shards'] if s['items'])
        folder = f.output / ('shard_%02d' % item_shard['shard']); key = item_shard['items'][0]['ledger']
        path = folder / (key + '.json'); doc = m.checked_document(path.read_bytes())
        doc['analysis']['policy_version'] = 1 - doc['analysis']['policy_version']; raw = m.sealed(doc); path.write_bytes(raw)
        summary = m.checked_document((folder / 'SHARD_COMPLETE.json').read_bytes())
        summary['receipts'][0]['sha256'] = m.digest(raw); (folder / 'SHARD_COMPLETE.json').write_bytes(m.sealed(summary))
        with self.assertRaisesRegex(ValueError, 'policy boundary'): m.merge(f.path, self.cpu)

    def test_request_admission_and_run_identity_fail_closed(self):
        f = Fixture(self.root); item = next(s['items'][0] for s in f.plan['shards'] if s['items'])
        folder = f.run / 'official_feedback' / item['ledger']
        request = json.loads((folder / 'REQUESTS.json').read_bytes()); request['requests'][0]['scene_id'] = 'foreign'
        write(folder / 'REQUESTS.json', request)
        with self.assertRaisesRegex(ValueError, 'REQUESTS identity'): m.verify_and_analyze(f.plan, item, self.budget())
        f2 = Fixture(self.root / 'other'); item2 = next(s['items'][0] for s in f2.plan['shards'] if s['items'])
        changed = copy.deepcopy(f2.plan); changed['run_identity']['numeric_execution_hash'] = 'foreign'
        with self.assertRaisesRegex(ValueError, 'Numeric/reward'): m.verify_and_analyze(changed, item2, self.budget_for(self.root / 'second'))

    def budget_for(self, folder):
        return m.Budget(folder, 'fixture', 'test', time.time()+110, 32 << 20, 2 << 20, False)

    def test_exact_quantiles_zero_groups_and_windows(self):
        self.assertEqual(m.scalar_summary([0, 0, 0, 10])['quantiles']['0.5'], 0.)
        rows = [{'scene_presentations': 2, 'centered_A_float32': [[0., 0., 0., 0.], [-3., -1., 1., 3.]], 'within_group_A_range_float32': [0., 6.]}]
        a = m.amplitude_statistics(rows)
        self.assertEqual(a['all_scene_presentations']['abs_A_per_candidate']['count'], 8)
        self.assertEqual(a['nonzero_advantage_scene_presentations']['abs_A_per_candidate']['mean'], 2.)
        self.assertEqual(a['all_scene_presentations']['within_group_A_range']['quantiles']['0.5'], 3.)
        f = Fixture(self.root); b = self.budget(); item = next(s['items'][0] for s in f.plan['shards'] if s['items'])
        one = m.verify_and_analyze(f.plan, item, b)
        population = [dict(one, policy_version=v) for v in (0, 1023, 1024, 2047, 2048, 3071, 3072, 4095)]
        result = m.aggregate(population)
        self.assertEqual([r['counts']['ledgers'] for r in result['fixed_policy_version_windows'].values()], [2, 2, 2, 2])


if __name__ == '__main__':
    unittest.main()
