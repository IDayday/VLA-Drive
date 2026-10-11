"""Owned synthetic CPU records; no scientific binding, worker or GPU dispatch."""
import argparse
import copy
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_p0_fullgrad_probes as coord
oldtests = coord.full.source_module(coord.COLD/'tools/structured_world/test_run_p0_forward_probes.py',
    coord.bench.digest(coord.COLD/'tools/structured_world/test_run_p0_forward_probes.py'), '_owned_original_forward_fixtures')


def put(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(value))


def budget():
    return {'limit_kind': 'internal_finite_test_registration', 'user_GPU_hours_limit': None,
        'phase_limit': 30, 'total_internal_limit': 30, 'already_spent_external': 0, 'reserved_elsewhere': 0,
        'per_job_reserves': {'fullgradient': 2.5}, 'accounting_scope': 'fresh_fullgradient_phase_only',
        'old_zt3_cost_state': 'UNKNOWN_NOT_ZERO', 'accounting_sources': []}


def fixture_command(job, binding, bref):
    return ['/OWNED_CPU_private_python', '-B', '-u', '-m', 'torch.distributed.run', '--standalone',
        '--nnodes=1', '--nproc_per_node=8', '--max_restarts=0', '--', str(coord.PRODUCER/coord.REL), 'probe',
        '--model-source-root', binding['model']['root'], '--forward-source-root', binding['forward_producer']['root'],
        '--source-binding', bref['path'], '--source-binding-sha256', bref['sha256'],
        '--run', job['origin']['run'], '--tag', job['origin']['tag'], '--cache', '/OWNED_CPU_cache',
        '--dino-root', '/OWNED_CPU_dino', '--output', job['observed'], '--microbatch', '4']


def new_binding(root):
    original = coord.load(oldtests.BINDING)
    # Producer order differs from JSON's alphabetic order; zero-error ties must
    # retain the original group/parameter order after a sorted JSON round trip.
    groups = [{'names': ['head.bias'], 'elements': 1}, {'names': ['encoder.weight'], 'elements': 2}]
    gp = root/'OWNED_origin/optimizer_groups.json'; put(gp, groups)
    inv = [['head.bias', [1], 'torch.float32'], ['encoder.weight', [2], 'torch.float32']]
    cases = copy.deepcopy(original['cases'])
    for case in cases.values():
        case['gradient_inventory_sha256'] = coord.full.metadata_hash(inv)
        case['optimizer_groups_sha256'] = coord.bench.digest(gp)
    binding = {'schema': 'execution_p0_fullgrad_source_binding_v1', 'producer': {'root': str(coord.PRODUCER),
        'git_sha': coord.PRODUCER_GIT, 'files_sha256': {coord.REL: coord.PRODUCER_SHA}},
        'model': original['model'], 'forward_producer': {'root': original['producer']['root']},
        'forward_source_binding': coord.ref(oldtests.BINDING), 'original_acceptance_bindings': original['original_acceptance_bindings'],
        'contract': json.loads(json.dumps(coord.full.CONTRACT)), 'cases': cases,
        'frozen_UTC': (datetime.now(timezone.utc)-timedelta(seconds=30)).isoformat(),
        'verification_relationship': 'OWNED_SYNTHETIC_CPU new delegated context, never scientific evidence', 'fixture': 'OWNED_SYNTHETIC_CPU'}
    bp = root/'OWNED_SYNTHETIC_FULLGRAD_BINDING.json'; put(bp, binding)
    return binding, coord.ref(bp), inv, groups


def fixture(root):
    job, fwd = oldtests.fixture(root)
    binding, bref, inv, groups = new_binding(root)
    job.update(fullgradient_source_binding=bref, stage='fullgradient', trainer=str(coord.PRODUCER/coord.REL))
    # Origin path is an owned fixture; keep original optimizer-group bytes there.
    put(Path(job['origin']['run'])/'optimizer_groups.json', groups)
    binding['cases'][job['case']]['optimizer_groups_sha256'] = coord.bench.digest(Path(job['origin']['run'])/'optimizer_groups.json')
    put(bref['path'], binding); bref = coord.ref(bref['path']); job['fullgradient_source_binding'] = bref
    job['command'] = fixture_command(job, binding, bref)
    script = job['command'].index(str(coord.PRODUCER/coord.REL))
    job['CPU_preflight_command'] = [job['command'][0], '-B']+job['command'][script:]+['--preflight']
    # Real746 delegates6d16.run_observed with args.source_binding=new869, so its
    # original forward emission also binds the new context, not historical44.
    fwd['producer_source_binding'] = bref
    put(Path(job['observed'])/'P0_FORWARD_COMPARISON.json', fwd)
    prototype = Path(job['folder'])/'PROBE.json'
    p = coord.load(prototype); p.update(fullgradient_binding=bref, job=job); prototype.unlink(); coord.write(prototype, p)
    records = []
    for r in range(8):
        tensors = {}
        # Deliberately different three measured pairs, no A-derived AA_B stand-in.
        for name, shape, dtype in inv:
            n = shape[0]; x = float(r+1)
            pairs = {}
            for pair, a, b in [('A_A', x, x+.1), ('A_B', x, x-.1), ('AA_B', x+.1, x-.1)]:
                pairs[pair] = {'elements': n, 'reference_square': n*a*a, 'candidate_square': n*b*b,
                    'difference_square': n*(a-b)**2, 'dot': n*a*b, 'max_absolute': abs(a-b),
                    'nonfinite_elements': 0, 'bit_different_elements': n, 'fixed_numeric_violations': n}
            tensors[name] = {'component': name.split('.')[0], 'optimizer_group': '1' if name.startswith('encoder') else '0', 'pairs': pairs}
        inventory = [[n, shape, dtype, True, [shape, dtype]] for n, shape, dtype in inv]
        value = {'inventory': {r: copy.deepcopy(inventory) for r in ('A', 'AA', 'B')}, 'expected_trainable_inventory': inv,
            'gradient_inventory_sha256': coord.full.metadata_hash(inv), 'trainable_parameter_count': 2, 'trainable_elements': 3,
            'all_trainable_gradients_present': True, 'captured_roles': ['A', 'AA', 'B'], 'tensors': tensors, 'optimizer_groups': groups}
        records.append({'rank': r, 'tokens': fwd['per_rank'][r]['tokens'], 'fullgrad': value})
    report = coord.full.combine_gradients(records, coord.full.metadata_hash(inv))
    report.update(registration_sha256=coord.forward.REGISTRATION_SHA, candidate_mode='engineering_fast', contract=binding['contract'],
        producer_sha256=coord.PRODUCER_SHA, producer_source_binding=bref, model_source=binding['model'],
        original_forward_source_binding=job['source_binding'], original_acceptance_bindings=binding['original_acceptance_bindings'],
        checkpoint={'run': job['origin']['run'], 'tag': job['origin']['tag'], 'case': job['case']},
        original_strict=fwd['original_strict'], forward_observation=coord.ref(Path(job['observed'])/'P0_FORWARD_COMPARISON.json'))
    put(Path(job['observed'])/'P0_FULL_GRADIENT_COMPARISON.json', report)
    return job, report, binding, bref


def prefix_fixture(root):
    job, _, b, bref = fixture(root)
    oldroot = Path(job['folder']).parent
    m = {**oldtests.manifest(root), 'schema': 'p0_fullgradient_coordinator_manifest_v1',
        'python': job['command'][0], 'cache': '/OWNED_CPU_cache', 'dino_root': '/OWNED_CPU_dino',
        'fullgradient_source_binding': bref, 'native_budget': budget(), 'minimum_free_MiB': 100,
        'previous_forward_completion': str(root/'OWNED_previous_forward_COMPLETE.json')}
    qp = root/'OWNED_QUALIFICATION.json'
    put(qp, {'passed': True, 'host': m['host'], 'physical_gpu_UUIDs': m['gpu_uuids'],
        'environment': {'packages': {n: 'OWNED_CPU_fixture' for n in (*coord.bench.PACKAGES, 'flash_attn')}}})
    m['qualification'] = coord.ref(qp)
    mp = root/'OWNED_OLD_MANIFEST.json'; put(mp, m)
    with patch.object(coord, 'sources', return_value=(b, bref)):
        plan = coord.build_plan(m, oldroot)
    assert plan['jobs'][0] == job
    plan['manifest_sha256'] = coord.bench.digest(mp); put(oldroot/'PLAN.json', plan)
    put(oldroot/'MANIFEST_BINDING.json', coord.ref(mp))
    put(m['previous_forward_completion'], {'completed_forward_slots': 12, 'original_strict_relabelled': False,
        'observations': [{'case': j['case'], 'repeat': j['repeat']} for j in plan['jobs']]})
    driver = coord.ref('/mnt/project/VLA-Drive-structured-world-fgtr-perf-fullgrad-driver-v2-frozen-20261011/tools/structured_world/run_p0_fullgrad_probes.py')
    plan['driver_sha256'] = driver['sha256']; put(oldroot/'PLAN.json', plan)
    proto = coord.load(Path(job['folder'])/'PROBE.json'); proto['coordinator'] = driver
    put(Path(job['folder'])/'PROBE.json', proto)
    put(oldroot/'REGISTRATION.json', {'driver': driver, 'driver_git': 'ccbda534adc63e71425143c61a98af9381b31416',
        'qualification': m['qualification'], 'native_budget': m['native_budget'],
        'fullgradient_binding': bref, 'original_forward_binding': job['source_binding']})
    put(Path(job['run'])/'PROCESS.json', {'command': job['command'], 'OWNED_SYNTHETIC_CPU': True})
    put(Path(job['folder'])/'CPU_PREFLIGHT.json', {'preflight_passed': True, 'case': job['case'],
        'source_binding_sha256': coord.BINDING_SHA, 'model_source_sha': coord.forward.MODEL_GIT,
        'forward_source_sha': coord.full.FORWARD_COMMIT})
    put(Path(job['folder'])/'RESOURCE_GATE.json', {'decision': 'ready', 'compute_apps': [],
        'gpus': [{'uuid': u} for u in m['gpu_uuids']]})
    error = "ValueError('Aggregate full-gradient statistics differ')"
    put(Path(job['folder'])/'INCOMPLETE.json', {'error': error, 'PROCESS_EXIT': coord.ref(Path(job['run'])/'PROCESS_EXIT.json')})
    put(oldroot/'FAILURE.json', {'error': error, 'costs': coord.native.actual_process_charges([job])})
    rel = {'plan': 'PLAN.json', 'manifest_binding': 'MANIFEST_BINDING.json', 'registration': 'REGISTRATION.json',
        'failure': 'FAILURE.json', 'probe': job['key']+'/PROBE.json', 'incomplete': job['key']+'/INCOMPLETE.json',
        'process': job['key']+'/process/PROCESS.json', 'process_exit': job['key']+'/process/PROCESS_EXIT.json',
        'cpu_preflight': job['key']+'/CPU_PREFLIGHT.json', 'resource_gate': job['key']+'/RESOURCE_GATE.json',
        'fullgradient': job['key']+'/observed/P0_FULL_GRADIENT_COMPARISON.json',
        'forward': job['key']+'/observed/P0_FORWARD_COMPARISON.json', 'strict': job['key']+'/observed/strict/COMPLETE.json'}
    files = {k: coord.ref(oldroot/v) for k, v in rel.items()}
    files.update(manifest=coord.ref(mp), old_coordinator=driver)
    new = {**m, 'observed_prefix': {'output': str(oldroot), 'files': files}}
    return new, plan, b, bref


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='owned-fullgrad-coordinator-'); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def classify_fixture(self, mutate=None):
        job, report, b, bref = fixture(self.root)
        if mutate:
            mutate(report); put(Path(job['observed'])/'P0_FULL_GRADIENT_COMPARISON.json', report)
        with patch.object(coord, 'checked_binding', return_value=(b, bref)):
            return coord.classify(job)

    def test_twelve_NEW_jobs_original44_context_real_new_fixture_hash_and_budget(self):
        b, bref, _, _ = new_binding(self.root)
        m = {**oldtests.manifest(self.root), 'schema': 'p0_fullgradient_coordinator_manifest_v1',
            'fullgradient_source_binding': bref, 'native_budget': budget()}
        with patch.object(coord, 'sources', return_value=(b, bref)):
            plan = coord.build_plan(m, self.root/'NEW_fullgrad')
        self.assertEqual(len(plan['jobs']), 12); self.assertEqual(len({j['run'] for j in plan['jobs']}), 12)
        for j in plan['jobs']:
            self.assertEqual(j['source_binding']['sha256'], coord.forward.BINDING_SHA)
            self.assertEqual(j['fullgradient_source_binding'], bref)
            self.assertIn('--nproc_per_node=8', j['command']); self.assertIn('--max_restarts=0', j['command'])
            self.assertEqual(j['command'][j['command'].index('--')+1], str(coord.PRODUCER/coord.REL))
        gate = coord.native.gpu_admission(plan['native_budget'], plan['jobs'], {'actual_phase_GPU_hours': 0, 'entries': []}, whole_plan=True)
        self.assertTrue(gate['admitted'])

    def test_pending_binding_never_gets_fake_SHA(self):
        with self.assertRaises(FileNotFoundError):
            coord.checked_binding({'path': str(self.root/'NOT_REGISTERED.json'), 'sha256': None}, {})
        p = self.root/'owned.json'; put(p, {})
        with self.assertRaisesRegex(ValueError, 'binding changed'):
            coord.checked_binding({'path': str(p), 'sha256': 'a'*64}, {})

    def test_actual_registered869_sources_twelve_CPU_commands_no_worker(self):
        p = Path('/mnt/project/structured-world-fgtr-execution-artifacts/20261011/p0_fullgrad_v1_prospective/SOURCE_BINDING.json')
        b, reference = coord.checked_binding({'path': str(p), 'sha256': coord.BINDING_SHA}, coord.ref(oldtests.BINDING))
        m = {**oldtests.manifest(self.root), 'schema': 'p0_fullgradient_coordinator_manifest_v1',
            'fullgradient_source_binding': reference, 'native_budget': budget()}
        with patch.object(coord.bench, 'run_job') as run:
            plan = coord.build_plan(m, self.root/'actual_source_CPU_plan')
        run.assert_not_called(); self.assertEqual(len(plan['jobs']), 12)
        self.assertEqual(b['producer']['git_sha'], coord.PRODUCER_GIT)
        self.assertEqual(plan['fullgradient_source_binding']['sha256'], coord.BINDING_SHA)

    def test_strict_exit1_full_three_pairs_is_observation_not_acceptance(self):
        result = self.classify_fixture()
        self.assertEqual(result['status'], 'completed_scientific_observation_no_retry')
        self.assertEqual(result['actual_returncode'], 1); self.assertFalse(result['original_strict_passed'])
        self.assertTrue(result['full_parameter_AA_B_gradients_captured']); self.assertFalse(result['passed'])

    def test_sorted_JSON_producer_ties_restore_exact_aggregate_without_mutation(self):
        job, report, b, bref = fixture(self.root)
        p = Path(job['observed'])/'P0_FULL_GRADIENT_COMPARISON.json'
        p.write_text(json.dumps(report, sort_keys=True)); before = p.read_bytes(); loaded = coord.load(p)
        self.assertNotEqual(coord.full.combine_gradients(loaded['per_rank'], loaded['gradient_inventory_sha256'])['comparisons'], report['comparisons'])
        self.assertEqual(coord.combine_in_inventory_order(loaded['per_rank'], loaded['gradient_inventory_sha256'])['comparisons'], report['comparisons'])
        with patch.object(coord, 'checked_binding', return_value=(b, bref)):
            self.assertFalse(coord.classify(job)['passed'])
        self.assertEqual(p.read_bytes(), before)

    def test_inventory_order_hash_and_group_order_cannot_be_changed(self):
        _, report, _, _ = fixture(self.root)
        for mutation in ('inventory', 'groups', 'missing'):
            bad = copy.deepcopy(report['per_rank'])
            if mutation == 'inventory': bad[0]['fullgrad']['expected_trainable_inventory'].reverse()
            if mutation == 'groups': bad[0]['fullgrad']['optimizer_groups'].reverse()
            if mutation == 'missing': bad[0]['fullgrad']['tensors'].pop('head.bias')
            with self.assertRaisesRegex(ValueError, 'order/inventory'):
                coord.combine_in_inventory_order(bad, report['gradient_inventory_sha256'])

    def test_first_observed_prefix_plans_eleven_and_charges_original_failed_exit(self):
        m, old, b, bref = prefix_fixture(self.root)
        before = {r['path']: Path(r['path']).read_bytes() for r in m['observed_prefix']['files'].values()}
        with patch.object(coord, 'sources', return_value=(b, bref)), \
                patch.object(coord, 'checked_binding', return_value=(b, bref)), \
                patch.object(coord, 'QUALIFICATION_SHA', m['qualification']['sha256']), patch.object(coord.bench, 'run_job') as worker:
            plan = coord.build_plan(m, self.root/'FRESH_v3')
        worker.assert_not_called(); self.assertEqual(plan['new_GPU_jobs'], 11)
        self.assertEqual(plan['jobs'][0], old['jobs'][0]); self.assertFalse(plan['observed_prefix']['observation']['passed'])
        charges = coord.native.actual_process_charges(plan['jobs'])
        self.assertEqual(charges['actual_phase_GPU_hours'], 12*8/3600)
        admission = coord.native.gpu_admission(plan['native_budget'], plan['jobs'], charges, whole_plan=True)
        self.assertEqual(len(admission['remaining_jobs']), 11)
        self.assertAlmostEqual(admission['projected_phase_GPU_hours'], 27.5+12*8/3600)
        self.assertEqual(before, {p: Path(p).read_bytes() for p in before})

    def test_prefix_changed_environment_source_data_budget_rejected(self):
        for field, value in [('python', '/foreign_python'), ('cache', '/foreign_cache'), ('gpu_uuids', ['foreign']*8),
                ('qualification', {}), ('fullgradient_source_binding', {}), ('native_budget', {})]:
            m, old, b, bref = prefix_fixture(self.root/field)
            m[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'source/environment/data/budget'):
                coord.observed_prefix(m, old['jobs'], self.root/'fresh')

    def test_prefix_second_partial_or_old_output_refused(self):
        m, old, b, bref = prefix_fixture(self.root)
        with self.assertRaisesRegex(ValueError, 'separate NEW output'):
            coord.observed_prefix(m, old['jobs'], m['observed_prefix']['output'])
        Path(old['jobs'][1]['folder']).mkdir()
        with self.assertRaisesRegex(ValueError, 'another completed/unfinished'):
            coord.observed_prefix(m, old['jobs'], self.root/'fresh')

    def test_prefix_missing_receipt_or_tampered_hash_refused(self):
        m, old, _, _ = prefix_fixture(self.root)
        p = Path(m['observed_prefix']['files']['fullgradient']['path']); p.unlink()
        with self.assertRaises(FileNotFoundError): coord.observed_prefix(m, old['jobs'], self.root/'fresh')
        put(p, {})
        with self.assertRaisesRegex(ValueError, 'receipt changed'):
            coord.observed_prefix(m, old['jobs'], self.root/'fresh')

    def test_prefix_complete_looking_missing_pair_or_rank_refused(self):
        for missing in ('rank', 'pair'):
            m, old, b, bref = prefix_fixture(self.root/missing)
            p = Path(m['observed_prefix']['files']['fullgradient']['path']); r = coord.load(p)
            if missing == 'rank': r['per_rank'].pop()
            else: r['per_rank'][0]['fullgrad']['tensors']['head.bias']['pairs'].pop('AA_B')
            put(p, r); m['observed_prefix']['files']['fullgradient'] = coord.ref(p)
            with patch.object(coord, 'checked_binding', return_value=(b, bref)), \
                    patch.object(coord, 'QUALIFICATION_SHA', m['qualification']['sha256']), self.assertRaises(ValueError):
                coord.observed_prefix(m, old['jobs'], self.root/'fresh')

    def test_prefix_scientific_worker_argv_change_is_refused(self):
        m, old, _, _ = prefix_fixture(self.root)
        jobs = copy.deepcopy(old['jobs']); jobs[1]['command'].append('--foreign-scientific-flag')
        with self.assertRaisesRegex(ValueError, 'scientific job argv/order'):
            coord.observed_prefix(m, jobs, self.root/'fresh')

    def test_execution_routes_only_eleven_NEW_slots_and_keeps_first_charge(self):
        m, _, b, bref = prefix_fixture(self.root)
        output = self.root/'FRESH_v3'
        with patch.object(coord, 'sources', return_value=(b, bref)), \
                patch.object(coord, 'checked_binding', return_value=(b, bref)), \
                patch.object(coord, 'QUALIFICATION_SHA', m['qualification']['sha256']):
            plan = coord.build_plan(m, output); output.mkdir(); coord.write(output/'PLAN.json', plan)
            versions = coord.load(m['qualification']['path'])['environment']['packages']
            def operator_read(argv, *a, **k):
                if argv[:3] == ['git', 'rev-parse', 'HEAD']: return 'OWNED_CPU_operator_commit'
                if argv[:2] == ['git', 'ls-files']: return 'tools/structured_world/run_p0_fullgrad_probes.py'
                if '-c' in argv: return json.dumps(versions)
                job = next(j for j in plan['jobs'] if j['CPU_preflight_command'] == argv)
                return json.dumps({'preflight_passed': True, 'case': job['case'], 'source_binding_sha256': bref['sha256']})
            def owned_worker(job, locks):
                # Synthetic runner only: prove registration precedes each NEW
                # worker and never route the old first job back to run_attempt.
                self.assertTrue((output/'REGISTRATION.json').is_file())
                self.assertEqual(coord.load(Path(job['folder'])/'PROBE.json')['job'], job)
                put(Path(job['run'])/'PROCESS_EXIT.json', {'returncode': 1, 'monitor_error': None, 'wall_seconds': 1.})
                return {'case': job['case'], 'repeat': job['repeat'], 'passed': False, 'OWNED_CPU_fixture': True}
            with patch.object(coord.bench, 'checked', side_effect=operator_read), \
                    patch.object(coord.bench, 'immutable_source'), patch.object(coord.bench, 'acquire', return_value=None), \
                    patch.object(coord.bench, 'readiness', return_value={'OWNED_CPU_fixture': True}), \
                    patch.object(coord, 'run_attempt', side_effect=owned_worker) as worker, \
                    patch.object(coord.signal, 'signal'):
                coord.execute(m, plan, output)
            self.assertEqual(worker.call_count, 11)
            self.assertEqual([call.args[0]['key'] for call in worker.call_args_list], [j['key'] for j in plan['jobs'][1:]])
            self.assertFalse((output/'G1_initial_repeat0').exists())
            done = coord.load(output/'COMPLETE.json')
            self.assertEqual(done['completed_fullgradient_slots'], 12); self.assertTrue(done['all12_observed'])
            self.assertFalse(done['passed'])
            self.assertEqual(done['actual_process_costs']['entries'][0]['path'], str(Path(plan['jobs'][0]['run'])/'PROCESS_EXIT.json'))
            self.assertAlmostEqual(done['actual_process_costs']['actual_phase_GPU_hours'], (12+11)*8/3600)

    def test_actual_delegated_context_rejected_by_old_classifier_preserved_by_new(self):
        job, _, b, bref = fixture(self.root)
        sidecar = Path(job['observed'])/'P0_FORWARD_COMPARISON.json'
        original = sidecar.read_bytes()
        with self.assertRaisesRegex(ValueError, 'Wrong source binding/model'): coord.forward.classify(job)
        with patch.object(coord, 'checked_binding', return_value=(b, bref)):
            result = coord.classify(job)
        self.assertEqual(sidecar.read_bytes(), original)
        self.assertEqual(result['actual_delegated_forward_source_binding'], bref)
        self.assertEqual(result['preserved_original44_context'], job['source_binding'])
        self.assertFalse(result['forward_receipt_bytes_rewritten'])

    def test_old44_sidecar_cannot_be_relabelled_as_new_fullgradient_observation(self):
        job, _, b, bref = fixture(self.root)
        p = Path(job['observed'])/'P0_FORWARD_COMPARISON.json'
        f = coord.load(p); f['producer_source_binding'] = job['source_binding']; put(p, f)
        with patch.object(coord, 'checked_binding', return_value=(b, bref)), self.assertRaisesRegex(ValueError, 'must be869'):
            coord.classify(job)

    def test_delegated_forward_thresholds_and_original_strict_hash_still_exact(self):
        for mutate in (lambda f: f['modules']['Bt']['AA_B'].update(max_absolute=1e-12),
                lambda f: f['original_strict'].update(sha256='0'*64),
                lambda f: f['producer_source_binding'].update(sha256='0'*64)):
            job, _, b, bref = fixture(self.root/str(len(list(self.root.iterdir()))))
            p = Path(job['observed'])/'P0_FORWARD_COMPARISON.json'
            f = coord.load(p); mutate(f); put(p, f)
            with patch.object(coord, 'checked_binding', return_value=(b, bref)), self.assertRaises(ValueError): coord.classify(job)

    def test_missing_duplicate_rank_rejected(self):
        for mutate in (lambda r: r['per_rank'].pop(), lambda r: r['per_rank'][1].update(rank=0)):
            with self.subTest(mutate=mutate):
                job, report, b, bref = fixture(self.root/str(len(list(self.root.iterdir()))))
                mutate(report); put(Path(job['observed'])/'P0_FULL_GRADIENT_COMPARISON.json', report)
                with patch.object(coord, 'checked_binding', return_value=(b, bref)), self.assertRaisesRegex(ValueError, 'ranks'):
                    coord.classify(job)

    def test_missing_parameter_group_or_None_gradient_rejected(self):
        for mutate in (lambda r: r['per_rank'][0]['fullgrad']['tensors'].pop('head.bias'),
                lambda r: r['per_rank'][0]['fullgrad']['optimizer_groups'][0].update(elements=99),
                lambda r: r['per_rank'][0]['fullgrad']['inventory']['B'][0].__setitem__(-1, None)):
            job, report, b, bref = fixture(self.root/str(len(list(self.root.iterdir()))))
            mutate(report); put(Path(job['observed'])/'P0_FULL_GRADIENT_COMPARISON.json', report)
            with patch.object(coord, 'checked_binding', return_value=(b, bref)), self.assertRaises(ValueError): coord.classify(job)

    def test_missing_AA_B_and_aggregate_A_based_substitution_rejected(self):
        with self.assertRaises(ValueError): self.classify_fixture(lambda r: r['per_rank'][0]['fullgrad']['tensors']['head.bias']['pairs'].pop('AA_B'))
        job, report, b, bref = fixture(self.root/'other')
        report['comparisons']['AA_B'] = copy.deepcopy(report['comparisons']['A_B'])
        put(Path(job['observed'])/'P0_FULL_GRADIENT_COMPARISON.json', report)
        with patch.object(coord, 'checked_binding', return_value=(b, bref)), self.assertRaisesRegex(ValueError, 'Aggregate'): coord.classify(job)

    def test_nonfinite_raw_gradients_and_wrong_scene_order_rejected(self):
        with self.assertRaises(ValueError): self.classify_fixture(lambda r: r['per_rank'][0]['fullgrad']['tensors']['head.bias']['pairs']['AA_B'].update(nonfinite_elements=1))
        job, report, b, bref = fixture(self.root/'other'); report['per_rank'][0]['tokens'].reverse()
        put(Path(job['observed'])/'P0_FULL_GRADIENT_COMPARISON.json', report)
        with patch.object(coord, 'checked_binding', return_value=(b, bref)), self.assertRaisesRegex(ValueError, 'scene/rank'): coord.classify(job)

    def test_wrong_producer_binding_and_forged_acceptance_rejected(self):
        with self.assertRaises(ValueError): self.classify_fixture(lambda r: r.update(producer_sha256='f'*64))
        job, report, b, bref = fixture(self.root/'other'); report['passed'] = True
        put(Path(job['observed'])/'P0_FULL_GRADIENT_COMPARISON.json', report)
        with patch.object(coord, 'checked_binding', return_value=(b, bref)), self.assertRaisesRegex(ValueError, 'forged acceptance'): coord.classify(job)

    def test_new_binding_must_precede_first_A(self):
        job, report, b, bref = fixture(self.root); b['frozen_UTC'] = (datetime.now(timezone.utc)+timedelta(days=1)).isoformat()
        with patch.object(coord, 'checked_binding', return_value=(b, bref)), self.assertRaisesRegex(ValueError, 'first A'): coord.classify(job)

    def test_actual_old_forward_classify_not_bypassed(self):
        job, _, b, bref = fixture(self.root)
        p = Path(job['observed'])/'P0_FORWARD_COMPARISON.json'; f = coord.load(p); f['forward_exact']['Bt'] = False; put(p, f)
        with patch.object(coord, 'checked_binding', return_value=(b, bref)), self.assertRaisesRegex(ValueError, 'exact flags'): coord.classify(job)

    def test_phase_cost_includes_strict_exit1_refuses_next_when_reserved_exceeded(self):
        job, _, _, _ = fixture(self.root)
        job['stage'] = 'fullgradient'; jobs = [job, {**job, 'key': 'next', 'run': str(self.root/'newprocess')}]
        costs = coord.native.actual_process_charges(jobs)
        self.assertAlmostEqual(costs['actual_phase_GPU_hours'], 12*8/3600)
        self.assertTrue(costs['entries'][0]['charged_despite_failure'])
        b = budget(); b['phase_limit'] = 2.5
        with self.assertRaises(coord.native.BudgetRefused): coord.native.gpu_admission(b, jobs, costs, next_job=jobs[1])
        (Path(job['run'])/'PROCESS_EXIT.json').unlink(); put(Path(job['run'])/'PROCESS.json', {'owned_fixture': True})
        self.assertTrue(coord.cost_record(jobs)['cost_unknown']); self.assertIsNone(coord.cost_record(jobs)['actual_phase_GPU_hours'])

    def test_existing_partial_attempt_never_dispatches(self):
        job, _, _, _ = fixture(self.root)
        with patch.object(coord.bench, 'run_job') as run, self.assertRaisesRegex(ValueError, 'NEW process'): coord.run_attempt(job, [])
        run.assert_not_called()

    def test_foreign_compute_app_readiness_refuses_without_dispatch(self):
        m = {'host': 'owned-host', 'gpu_uuids': [f'GPU-{r}' for r in range(8)], 'minimum_free_MiB': 100}
        rows = '\n'.join(f'GPU-{r}, 0, 100000, 0' for r in range(8))
        with patch.object(coord.bench, 'authorize'), patch.object(coord.bench.socket, 'gethostname', return_value='owned-host'), \
                patch.object(coord.bench, 'checked', side_effect=[rows, 'GPU-0, 321, genuine-worker']), \
                patch.object(coord.bench, 'run_job') as run, self.assertRaisesRegex(ValueError, 'compute app'):
            coord.bench.readiness(m)
        run.assert_not_called()


def different_leaves(a, b, path=''):
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(a.keys() | b.keys()):
            if key not in a or key not in b: yield [path+'/'+key, a.get(key), b.get(key)]
            else: yield from different_leaves(a[key], b[key], path+'/'+key)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b): yield [path+'/length', len(a), len(b)]
        for i, (x, y) in enumerate(zip(a, b)): yield from different_leaves(x, y, path+'/'+str(i))
    elif a != b: yield [path, a, b]


class ActualReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = coord.load(cls.path)
        cls.before = coord.bench.digest(cls.path)

    def test_real_sorted_report_original_failure_is_only_tie_names_exact_numeric(self):
        r = self.report
        original = coord.full.combine_gradients(r['per_rank'], r['gradient_inventory_sha256'])['comparisons']
        diff = list(different_leaves(r['comparisons'], original))
        self.assertTrue(diff)
        self.assertTrue(all('/worst_tensors/' in p and p.endswith('/name') for p, a, b in diff))
        numeric = [x for x in diff if type(x[1]) in (int, float) and type(x[2]) in (int, float)]
        self.assertEqual(numeric, [])
        corrected = coord.combine_in_inventory_order(r['per_rank'], r['gradient_inventory_sha256'])
        self.assertEqual(r['comparisons'], corrected['comparisons'])
        self.assertEqual(coord.bench.digest(self.path), self.before)
        self.__class__.evidence = {'raw_report': coord.ref(self.path), 'different_leaf_count': len(diff),
            'numeric_difference_count': 0, 'max_numeric_difference': 0, 'all_differences': diff,
            'inventory_order_strict_comparisons_equal': True, 'raw_report_bytes_rewritten': False,
            'thresholds_changed': False, 'GPU_jobs_started': 0}

    def test_real_report_inventory_reordering_is_not_hash_forged(self):
        records = copy.deepcopy(self.report['per_rank'])
        records[0]['fullgrad']['expected_trainable_inventory'].reverse()
        with self.assertRaisesRegex(ValueError, 'order/inventory'):
            coord.combine_in_inventory_order(records, self.report['gradient_inventory_sha256'])

    def test_real_numeric_statistic_change_remains_strictly_different(self):
        records = copy.deepcopy(self.report['per_rank'])
        tensor = next(iter(records[0]['fullgrad']['tensors'].values()))
        tensor['pairs']['AA_B']['difference_square'] += 1.
        changed = coord.combine_in_inventory_order(records, self.report['gradient_inventory_sha256'])
        self.assertNotEqual(changed['comparisons'], self.report['comparisons'])


class ProducerIntegrationTests(unittest.TestCase):
    def test_actual_frozen746_to6d16_emit_then_context_classify(self):
        # CPU tensors are only a tiny owned stand-in for the legacy worker.
        # Both producer entry functions, collectors, pair reducers and writers
        # below are the actual unchanged frozen source, not expected-field mocks.
        import torch
        torch.set_num_threads(1)
        p0 = coord.full.upstream(coord.PRODUCER.parent/'VLA-Drive-structured-world-fgtr-perf-p0-forward-v1-frozen-20261011')
        model_root = Path(coord.load(oldtests.BINDING)['model']['root'])
        shared = p0.shared_collector(model_root)
        comparator = coord.full.comparator(model_root)
        with tempfile.TemporaryDirectory(prefix='OWNED_REAL_PRODUCER_CPU_') as temporary:
            job, _, binding, bref = fixture(Path(temporary))
            binding.update(chunk_elements=2, memory_reserve_bytes=0)
            put(bref['path'], binding); bref = coord.ref(bref['path']); job['fullgradient_source_binding'] = bref
            job['command'] = fixture_command(job, binding, bref)
            proto_path = Path(job['folder'])/'PROBE.json'
            proto = coord.load(proto_path); proto.update(fullgradient_binding=bref, job=job); put(proto_path, proto)
            output = Path(job['observed'])
            for name in ('P0_FORWARD_COMPARISON.json', 'P0_FULL_GRADIENT_COMPARISON.json'): (output/name).unlink()
            strict_path = output/'strict/COMPLETE.json'
            strict = coord.load(strict_path)
            groups = coord.load(Path(job['origin']['run'])/'optimizer_groups.json')
            framework = SimpleNamespace(sample_queries=lambda q0: {'poses': q0.detach().clone(),
                'time_index': torch.zeros((4, 2), dtype=torch.long)})

            class Tiny(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.encoder = torch.nn.Module(); self.encoder.register_parameter('weight', torch.nn.Parameter(torch.zeros(2)))
                    self.head = torch.nn.Module(); self.head.register_parameter('bias', torch.nn.Parameter(torch.zeros(1)))
                    self.frozen = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
                    self.geometry = torch.nn.Module()
                    self.geometry.register_buffer('running_mean', torch.zeros(1))
                    self.geometry.register_buffer('running_var', torch.ones(1))
                    self.geometry.register_buffer('num_batches_tracked', torch.zeros((), dtype=torch.long))
                    self.factor = 1.

                def forward(self):
                    q0 = torch.zeros((4, 2, 3)); framework.sample_queries(q0)
                    value = self.encoder.weight.sum()+self.head.bias.sum()
                    loss = (value-value.detach())*self.factor+1.
                    return {'diagnostics': {'q0': q0, 'q_final': q0.clone(),
                        'H_A': torch.zeros((4, 2)), 'B0': torch.zeros((4, 1, 2, 2)), 'Bt': torch.zeros((4, 1, 2, 2))},
                        'loss': loss, 'losses': {'tiny': loss},
                        'metrics': {'raw': {'tiny': loss}, 'effective_counts': {'scene': 4}, 'query_strata': {'tiny': 4},
                            'proposal_out_of_range': 0, 'refine_effective_weight': 4,
                            'activation_gradient_samples': {n: torch.zeros(1) for n in ('H_A', 'B0', 'Bt')}}}

            class FakeDist:
                def all_gather_object(self, out, row):
                    for rank in range(8):
                        out[rank] = {**copy.deepcopy(row), 'rank': rank,
                            'tokens': [f'OWNED_rank{rank}_scene{i}' for i in range(4)]}

            legacy = SimpleNamespace(VLAStructuredFGTR=Tiny, capture_rng=lambda _: {'OWNED_CPU_rng': [1, 2]},
                dist=FakeDist(), optimizer_batch_counts=lambda: {'scene': 32})
            rechecks = []
            def recheck():
                self.assertEqual(coord.ref(bref['path']), bref)
                rechecks.append(True)

            def main():
                model = legacy.VLAStructuredFGTR()
                for factor in (1., 2., 3.):
                    model.factor = factor; model.zero_grad(set_to_none=True)
                    legacy.optimizer_batch_counts()
                    result = model(); result['loss'].backward(); legacy.capture_rng({})
                rows = [None]*8
                legacy.dist.all_gather_object(rows, {'rank': 0, 'tokens': []})
                strict['per_rank'] = rows; put(strict_path, strict)
                raise ValueError('OWNED original strictFalse preserved')
            legacy.main = main
            args = SimpleNamespace(output=output, run=Path(job['origin']['run']), tag=job['origin']['tag'],
                cache=Path('/OWNED_CPU_cache'), dino_root=Path('/OWNED_CPU_dino'), microbatch=4,
                model_source_root=model_root, source_binding=Path(bref['path']), source_binding_sha256=bref['sha256'], model_state_file=None)
            with self.assertRaisesRegex(ValueError, 'original strictFalse'):
                coord.full.run_observed(args, p0, shared, comparator, legacy, framework, binding, recheck, groups, job['case'])
            sidecar = output/'P0_FORWARD_COMPARISON.json'; gradient = output/'P0_FULL_GRADIENT_COMPARISON.json'
            raw = sidecar.read_bytes(); emitted = coord.load(sidecar)
            self.assertEqual(emitted['producer_source_binding'], bref)
            self.assertEqual(emitted['producer_sha256'], coord.forward.PRODUCER_SHA)
            self.assertEqual(coord.load(gradient)['producer_sha256'], coord.PRODUCER_SHA)
            self.assertEqual(coord.load(gradient)['producer_source_binding'], bref)
            with self.assertRaisesRegex(ValueError, 'Wrong source binding/model'): coord.forward.classify(job)
            # Only owned fixture metadata admission is substituted; no producer
            # source hash, context check, writer or numeric validation is mocked.
            with patch.object(coord, 'checked_binding', return_value=(binding, bref)):
                observation = coord.classify(job)
            self.assertEqual(sidecar.read_bytes(), raw)
            self.assertTrue(rechecks); self.assertFalse(observation['passed'])
            self.assertEqual(observation['actual_returncode'], 1)
            self.assertTrue(observation['full_parameter_AA_B_gradients_captured'])
            self.assertEqual(coord.load(gradient)['comparisons']['AA_B']['gradient']['all']['difference_norm'], 24**.5)
            self.assertFalse(torch.cuda.is_initialized())
            self.__class__.evidence = {'schema': 'owned_CPU_real_producer_entrypaths_context_v1',
                'producer': coord.ref(coord.PRODUCER/coord.REL), 'delegated_forward_producer': coord.ref(Path(p0.__file__)),
                'functions_exercised': ['746.run_observed', '6d16.run_observed', '6d16.combine', 'be5.Collector', '746.combine_gradients', 'coordinator.classify'],
                'actual_emitted_forward_producer_SHA': emitted['producer_sha256'],
                'actual_emitted_fullgradient_producer_SHA': coord.load(gradient)['producer_sha256'],
                'actual_emitted_new_owned_fixture_context': emitted['producer_source_binding'],
                'old44_classifier_rejected_real_emission': True, 'new_classifier_preserved_sidecar_bytes': True,
                'forward_report_sha256': coord.bench.digest(sidecar), 'gradient_report_sha256': coord.bench.digest(gradient),
                'original_strict_false_and_exit1_preserved': True, 'numeric_thresholds_changed': False,
                'producer_source_hash_or_writer_mocked': False, 'only_fixture_metadata_admission_substituted': True,
                'tiny_owned_fixture_is_actual_scientific_evidence': False, 'GPU_jobs_started': 0, 'CUDA_initialized': False}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('--producer-integration', action='store_true', help='Also exercise frozen746->6d16 writers with tiny CPU tensors; requires private torch runtime')
    p.add_argument('--actual-report', type=Path, help='Read saved real fullgradient report; test exact order restoration and reject numeric/inventory changes; no dispatch')
    args = p.parse_args()
    require_cpu = os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    if not require_cpu: raise ValueError('CPU-only tests require empty CUDA_VISIBLE_DEVICES')
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    if args.producer_integration: suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(ProducerIntegrationTests))
    if args.actual_report:
        ActualReportTests.path = args.actual_report
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(ActualReportTests))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    no_torch = 'torch' not in sys.modules
    cpu_safe = no_torch or (args.producer_integration and not sys.modules['torch'].cuda.is_initialized())
    integration_evidence = getattr(ProducerIntegrationTests, 'evidence', None)
    if integration_evidence is not None:
        trace = args.receipt.with_name(args.receipt.stem+'_REAL_PRODUCER_TRACE.json')
        coord.write(trace, integration_evidence)
        integration_evidence = coord.ref(trace)
    actual_evidence = getattr(ActualReportTests, 'evidence', None)
    if actual_evidence is not None:
        trace = args.receipt.with_name(args.receipt.stem+'_ACTUAL_REPORT_ORDER_DIAGNOSIS.json')
        coord.write(trace, actual_evidence); actual_evidence = coord.ref(trace)
    coord.write(args.receipt, {'schema': 'p0_fullgradient_coordinator_CPU_tests_v1', 'tests_run': result.testsRun,
        'passed': result.wasSuccessful() and cpu_safe, 'torch_imported': not no_torch, 'GPU_jobs_started': 0,
        'actual_frozen_producer_entrypaths_exercised': args.producer_integration, 'CUDA_initialized': False if cpu_safe else None,
        'real_producer_integration_evidence': integration_evidence,
        'actual_report_order_regression_evidence': actual_evidence,
        'synthetic_fixture_receipts_are_scientific_evidence': False, 'new_BINDING_is_owned_mock_only': True,
        'source_files': [coord.ref(coord.__file__), coord.ref(__file__)]})
    raise SystemExit(0 if result.wasSuccessful() and cpu_safe else 1)


if __name__ == '__main__': main()
