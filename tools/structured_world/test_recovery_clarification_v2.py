"""Reproducible synthetic CPU tests, not model/GPU qualification.

Uses tiny native payloads and the original V1 numerical test fixture. Includes
the unchanged V1 evaluator rejecting an actual +16 load for its +32 trial.
"""
import argparse
import copy
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch

import evaluate_recovery_clarification_v2 as v2

SOURCE = Path('/mnt/project/VLA-Drive-structured-world-fgtr-perf-native-v4-frozen-20261011')
ART = Path('/mnt/project/structured-world-fgtr-round1-artifacts/20261008')
BINDINGS = ART/'training_efficiency_20261011T0546/P0_V1_PROSPECTIVE_BINDINGS/BINDINGS.json'
PLANS = ART/'perf_native_continuation_20261011/zt4_after_benchmark_p0_20261011T101431Z/same_mode_20261011T113145Z/records_v1'
sys.path.insert(0, str(SOURCE/'tools/structured_world'))
import test_execution_acceptance as old_tests

gate, native = v2.helpers(SOURCE)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    return v2.ref(path)


class RecoveryTests(unittest.TestCase):
    controls = []

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # Reuse V1 synthetic numerical statistics and inventories verbatim.
        self.fixture = old_tests.AcceptanceTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tmp.cleanup)
        self.policy = self.fixture.policy
        b = copy.deepcopy(self.fixture.bindings)
        actual = v2.load(BINDINGS)['sources']
        b['sources'] = copy.deepcopy(actual)
        self.bindings = self.root/'bindings.json'
        self.plan_paths = []
        self.identities, self.origins = {}, {}
        for group, start in (('G1', 25000), ('G3', 50000)):
            case = b['cases'][group+'/mid']
            case['origin']['completed'] = start
            case['origin']['identity'] = 'synthetic-origin-'+group
            original = self.bank(group, 'origin', start, case, b['sources']['B'], origin=True)
            case['origin']['COMPLETE_sha256'] = original.provenance()['COMPLETE_sha256']
            case['origin']['payload_manifest_sha256'] = v2.canonical(original.complete['file_sha256'])
            self.origins[group] = original
            p = v2.load(PLANS/(group+'_SAME_MODE_MANIFEST_BUDGET_PENDING.json'))
            p['qualification'] = write(self.root/(group+'_qualification.json'), {
                'passed': True, 'host': p['host'], 'physical_gpu_UUIDs': p['gpu_uuids'],
                'scope': 'SYNTHETIC CPU FIXTURE ONLY'})
            for job in p['jobs']:
                role = job['role']
                run = self.root/group/role
                job['run'] = str(run)
                job['origin']['complete_sha256'] = case['origin']['COMPLETE_sha256']
                command = list(job['command'])
                command[command.index('--output')+1] = str(run)
                job['command'] = command
            path = self.root/(group+'_plan.json'); write(path, p); self.plan_paths.append(path)
        write(self.bindings, b)
        self.binding_data = b
        self.c = v2.freeze_contract(SOURCE, self.bindings, self.plan_paths)
        self.clarification = self.root/'clarification.json'; write(self.clarification, self.c)
        self.receipts = {}
        for group in v2.GROUPS:
            self.receipts[group] = self.make_receipt(group)

    def bank(self, group, role, completed, case, source, origin=False, paused=False):
        run = self.root/group/role
        run.mkdir(parents=True, exist_ok=True)
        cfg = {'synthetic': 'no real model or dataset'}
        identity = {k: 'synthetic-'+k for k in native.SCIENCE_FIELDS}
        identity.update(identity=case['origin']['identity'] if origin else 'synthetic-'+group+'-'+role,
            training_source_sha=self.policy['scientific_training_source'] if origin else source['git_sha'],
            training_source_tree_sha256=v2.canonical(source['files_sha256']), config=cfg,
            **{k: case['science'][k] for k in ('current_and_GT_identity', 'DINO_identity', 'training_seed', 'FM_repeat', 'schedule_horizon')},
            effective_global_batch=32, schedule_warmup=case['science']['warmup'], world_size=8,
            execution_mode='io_preserving_v1' if origin else 'engineering_fast', benchmark_no_end_checkpoint=False,
            engineering_continue_origin={'identity': case['origin']['identity'], 'completed': case['origin']['completed'],
                                         'complete_sha256': case['origin']['COMPLETE_sha256']})
        if not origin:
            p = v2.load(self.root/(group+'_plan.json'))
            identity['engineering_native_capture'] = dict(p['jobs'][0]['capture_plan'], storage_budget_sha256=p['jobs'][0]['storage_budget']['sha256'])
        write(run/'identity.json', identity)
        write(run/'optimizer_groups.json', [{'synthetic': True}])
        tag = ('latest_' if paused else 'milestone_')+f'{completed:07d}'
        folder = run/'checkpoints'/tag; folder.mkdir(parents=True, exist_ok=True)
        model = {'identity': identity['identity'], 'completed': completed, 'dp_world_size': 8,
                 'module': {'weight': torch.ones(4), 'bn.running_mean': torch.zeros(4)},
                 'epoch': 7, 'offset': 33, 'exposure': 800000, 'scheduler_horizon': 100000}
        torch.save(model, folder/'mp_rank_00_model_states.pt')
        for rank in range(8):
            torch.save({'python': (1, 2), 'numpy': [1, 2], 'torch': torch.ones(2), 'cuda': [], 'generators': {}}, folder/f'rng_rank{rank}.pt')
            torch.save({'optimizer_state_dict': {'synthetic_rank': rank}}, folder/f'zero_pp_rank_{rank}_mp_rank_00_optim_states.pt')
        write(folder/'COMPLETE.json', {'identity': identity['identity'], 'completed': completed, 'ranks': 8, 'tag': tag,
              'status': 'PAUSED' if paused else 'COMPLETE', 'file_sha256': {p.name: v2.digest(p) for p in folder.glob('*.pt')}})
        return native.Bank(folder, verify_hashes=True)

    def make_receipt(self, group):
        case = self.binding_data['cases'][group+'/mid']; start = case['origin']['completed']
        banks = {role: self.bank(group, role, start+32, case, self.c['sources'][role]) for role in gate.ROLES}
        split_bank = self.bank(group, 'B', start+16, case, self.c['sources']['B'], paused=True)
        comparison = old_tests.comparison(self.policy)
        comparison['provenance'] = {r: bank.provenance() for r, bank in dict(banks, origin=self.origins[group]).items()}
        comparison['rank_checks'] = [{'rank': r, 'groups': 1} for r in range(8)]
        result = {'schema': v2.RECEIPT_SCHEMA, 'kind': 'recovery', 'stage': 'mid', 'group': group,
            'clarification_sha256': v2.digest(self.clarification), 'plan_sha256': self.c['plans'][group]['sha256'],
            'comparison': write(self.root/(group+'_comparison.json'), comparison), 'jobs': {},
            'roles': gate.expected_roles(dict(self.binding_data, sources=self.c['sources']), case)}
        plan = v2.load(self.c['plans'][group]['path'])
        for job in plan['jobs']:
            key = job['key']; bank = split_bank if key == 'B16' else banks[job['role']]
            boundary = dict(v2.BOUNDARY, identity=bank.identity['identity'], completed=job['expected_resume_completed'], epoch=7, offset=33, exposure=800000)
            stage = {'key': key, 'run_identity': bank.identity['identity'], 'identity_sha256': v2.digest(bank.run/'identity.json'),
                     'origin_boundary': boundary, 'checkpoint': {'path': str(bank.folder), 'complete': bank.complete,
                     'COMPLETE_sha256': v2.digest(bank.folder/'COMPLETE.json')},
                     'logging_cost_by_rank': [{'rank': r, 'completed': job['expected_exit_completed']} for r in range(8)]}
            result['jobs'][key] = {'host': plan['host'],
                'process': write(self.root/(group+key+'_process.json'), {'pid': 1234, 'started_unix': gate.time_value(self.c['frozen_utc']).timestamp()+1, 'command': job['command']}),
                'exit': write(self.root/(group+key+'_exit.json'), {'pid': 1234, 'returncode': 0, 'monitor_error': None}),
                'stage': write(self.root/(group+key+'_stage.json'), stage),
                'resource_gate': write(self.root/(group+key+'_resource.json'), {'decision': 'ready', 'compute_apps': [], 'gpus': [{'uuid': u} for u in plan['gpu_uuids']]})}
        path = self.root/(group+'_receipt.json'); write(path, result)
        return path

    def mutate(self, path, fn):
        value = v2.load(path); fn(value); write(path, value)

    def mutate_proof(self, group, key, name, fn):
        path = self.receipts[group]; receipt = v2.load(path)
        item = receipt['jobs'][key][name]
        self.mutate(item['path'], fn); receipt['jobs'][key][name] = v2.ref(item['path']); write(path, receipt)

    def mutate_comparison(self, fn):
        path = self.receipts['G1']; receipt = v2.load(path)
        self.mutate(receipt['comparison']['path'], fn)
        receipt['comparison'] = v2.ref(receipt['comparison']['path']); write(path, receipt)

    def check(self, decision='not_supported', contains=None, paths=None):
        result = v2.review(self.clarification, list(self.receipts.values()) if paths is None else paths)
        self.assertEqual(result['advisory_decision'], decision, result['failures'])
        if contains: self.assertTrue(any(contains in f for f in result['failures']), result['failures'])
        self.assertFalse(result['passed']); self.assertFalse(result['formal_authorized']); self.assertFalse(result['trainer_consumable'])
        self.assertIsNone(result['accepted_mode'])
        self.controls.append({'test': self.id(), 'decision': decision, 'failures': result['failures'], 'passed': False,
                              'scope': 'SYNTHETIC CPU VALIDATOR TEST, NOT REAL MODEL EVIDENCE'})
        return result

    def test_actual_load16_and_endpoint32_consistent_without_acceptance(self):
        result = self.check('recovery_evidence_complete_for_review')
        self.assertEqual(result['groups']['G1']['actual_load_completed'], 25016)
        self.assertEqual(result['groups']['G3']['endpoint_completed'], 50032)
        self.assertFalse(result['original_strict_results_preserved'][0]['original_strict']['A_B']['passed'])

    def test_missing_receipts_remain_pending(self): self.check('pending', paths=[])
    def test_one_group_does_not_complete_two_groups(self): self.check('pending', paths=[self.receipts['G1']])

    def test_bad_RNG_at_load(self):
        self.mutate_proof('G1', 'B_resume32', 'stage', lambda x: x['origin_boundary'].update(all_task_RNG='different rank7'))
        self.check(contains='all_task_RNG')

    def test_bad_RNG_at_endpoint(self):
        self.mutate_comparison(lambda x: x['E1']['A_B'].update(passed=False, failure_count=1, failure_paths=['RNG/rank7']))
        self.check(contains='RNG/groups/data/scheduler')

    def test_master_and_Adam_load_must_be_exact(self):
        for field in ('FP32_masters', 'Adam_moments_steps_and_LR'):
            with self.subTest(field=field):
                self.mutate_proof('G1', 'B_resume32', 'stage', lambda x: x['origin_boundary'].update({field: 'not exact'}))
                self.check(contains=field)
                self.mutate_proof('G1', 'B_resume32', 'stage', lambda x: x['origin_boundary'].update({field: v2.BOUNDARY[field]}))

    def test_model_and_BN_load_must_be_exact(self):
        self.mutate_proof('G1', 'B_resume32', 'stage', lambda x: x['origin_boundary'].update(FP32_model_and_buffers='not exact'))
        self.check(contains='FP32_model_and_buffers')

    def test_actual_load_client_offset_mismatch(self):
        self.mutate_proof('G1', 'B_resume32', 'stage', lambda x: x['origin_boundary'].update(offset=34))
        self.check(contains='Restored split client/offset')

    def test_modified_split_stop_CLI_rejected_during_freeze(self):
        self.mutate(self.plan_paths[0], lambda x: x['jobs'][2]['command'].__setitem__(x['jobs'][2]['command'].index('--stop-after')+1, '25032'))
        with self.assertRaisesRegex(ValueError, 'stop at16'):
            v2.freeze_contract(SOURCE, self.bindings, self.plan_paths)

    def test_cross_run_controls_cannot_replace_bound_own_runs(self):
        receipt = v2.load(self.receipts['G1']); cp = Path(receipt['comparison']['path']); comp = v2.load(cp)
        old = Path(comp['provenance']['AA']['path']); new_run = self.root/'unbound_control'
        import shutil
        shutil.copytree(old.parent.parent, new_run)
        comp['provenance']['AA'] = native.Bank(new_run/'checkpoints'/old.name, verify_hashes=True).provenance()
        write(cp, comp); receipt['comparison'] = v2.ref(cp); write(self.receipts['G1'], receipt)
        self.check(contains='bound own run')

    def test_missing_comparator_rank(self):
        self.mutate_comparison(lambda x: x['rank_checks'].pop())
        self.check(contains='ranks')

    def test_missing_stage_rank(self):
        self.mutate_proof('G1', 'B16', 'stage', lambda x: x['logging_cost_by_rank'].pop())
        self.check(contains='all-rank')

    def test_missing_native_rank_payload(self):
        p = v2.load(v2.load(self.receipts['G1'])['comparison']['path'])['provenance']['B']['path']
        (Path(p)/'rng_rank7.pt').unlink()
        self.check(contains='Missing native payload')

    def test_payload_corruption_even_when_report_claims_verified(self):
        p = v2.load(v2.load(self.receipts['G1'])['comparison']['path'])['provenance']['AA']['path']
        with (Path(p)/'rng_rank7.pt').open('ab') as stream: stream.write(b'corrupt')
        self.check(contains='Native payload hash changed')

    def test_load32_cannot_be_relabelled_as16(self):
        self.mutate_proof('G1', 'B_resume32', 'stage', lambda x: x['origin_boundary'].update(completed=25032))
        self.check(contains='load16')

    def test_endpoint16_cannot_be_relabelled_as32(self):
        self.mutate_comparison(lambda x: x['provenance']['B'].update(completed=25016))
        self.check(contains='provenance')

    def test_stage32_cannot_claim_split16_complete(self):
        self.mutate_proof('G1', 'B16', 'stage', lambda x: x['checkpoint']['complete'].update(completed=25032))
        self.check(contains='Preserved stage COMPLETE')

    def test_no_skipped_payload_hashing(self):
        self.mutate_comparison(lambda x: x.update(integrity_verified=False))
        self.check(contains='Unverified')

    def test_source_pin_drift(self):
        self.mutate(self.clarification, lambda x: x.update(source_sha='0'*40))
        self.check(contains='source changed')

    def test_environment_header_drift(self):
        self.mutate(self.receipts['G1'], lambda x: x['roles']['B']['environment'].update(torch='different'))
        self.check(contains='role facts')

    def test_same_GPU_pairing_required(self):
        self.mutate_proof('G1', 'AA32', 'resource_gate', lambda x: x['gpus'][0].update(uuid='GPU-other-host'))
        self.check(contains='same-device')

    def test_foreign_host_control_rejected(self):
        self.mutate(self.receipts['G1'], lambda x: x['jobs']['AA32'].update(host='training-vlawm-zt3'))
        self.check(contains='host pairing')

    def test_actual_B16_before_freeze_rejected(self):
        self.mutate_proof('G1', 'B16', 'process', lambda x: x.update(started_unix=gate.time_value(self.c['frozen_utc']).timestamp()-1))
        self.check(contains='preceded V2 freeze')

    def test_actual_failed_child_rejected(self):
        self.mutate_proof('G1', 'B_resume32', 'exit', lambda x: x.update(returncode=1))
        self.check(contains='failed actual stage')

    def test_changed_actual_commands_rejected(self):
        self.mutate_proof('G1', 'B_resume32', 'process', lambda x: x['command'].append('--scientific-change'))
        self.check(contains='Actual recovery command')

    def test_old_schema_not_reinterpreted(self):
        self.mutate(self.receipts['G1'], lambda x: x.update(schema='execution_acceptance_receipt_v1'))
        self.check(contains='cannot be reinterpreted')

    def test_native500_not_reinterpreted(self):
        self.mutate(self.receipts['G1'], lambda x: x.update(kind='native'))
        self.check(contains='cannot be reinterpreted')

    def test_duplicate_group_rejected(self): self.check(paths=[self.receipts['G1'], self.receipts['G1']], contains='Duplicate')

    def test_V1_numeric_and_dev_digest_not_relaxed(self):
        self.mutate(self.clarification, lambda x: x['unchanged_V1_sections_sha256'].update(numeric='0'*64))
        self.check(contains='criteria unchanged')

    def test_fresh_AA_envelope_remains_V1(self):
        self.mutate_comparison(lambda x: x.update(comparisons=old_tests.comparison(self.policy, b_error=1e-4)['comparisons']))
        self.check(contains='AA envelope')

    def test_absolute_ceilings_remain_V1(self):
        self.mutate_comparison(lambda x: x.update(comparisons=old_tests.comparison(self.policy, b_error=.02)['comparisons']))
        self.check(contains='absolute ceiling')

    def test_missing_BN_category_rejected(self):
        def remove(x):
            for pair in x['comparisons'].values(): del pair['BN_float']
        self.mutate_comparison(remove)
        self.check(contains='Incomplete native tensor categories')

    def test_original_V1_load16_remains_not_supported(self):
        f = self.fixture
        path = f.receipt('recovery', updates=32)
        receipt = gate.load(path)
        boundary = gate.load(receipt['payload']['load_boundary_receipt']['path'])
        boundary['completed'] = 25016
        receipt['payload']['load_boundary_receipt'] = old_tests.write(receipt['payload']['load_boundary_receipt']['path'], boundary)
        old_tests.write(path, receipt)
        result = gate.evaluate(f.registration, f.binding_path, [path], source_root=v2.V1_SOURCE_ROOT)
        self.assertEqual(result['advisory_decision'], 'not_supported', result)
        self.assertTrue(any('actual restore boundary' in x for x in result['failures']), result['failures'])
        self.controls.append({'test': self.id(), 'original_V1_decision': result['advisory_decision'], 'passed': result['passed'],
                              'failures': result['failures'], 'scope': 'Synthetic V1 schema-mismatch reproduction; unchanged original evaluator'})

    def test_original_V1_source_guard_not_rebound_to_be5(self):
        f = self.fixture
        result = gate.evaluate(f.registration, f.binding_path, [], source_root=SOURCE)
        self.assertEqual(result['advisory_decision'], 'not_supported')
        self.assertTrue(any('train_vla.py' in x for x in result['failures']), result['failures'])
        self.controls.append({'test': self.id(), 'original_V1_decision': result['advisory_decision'], 'passed': result['passed'],
                              'failures': result['failures'], 'scope': 'Original protected trainer hash remains unchanged; be5 is separately pinned only in V2'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    v2.require(not args.output.exists(), 'Fresh CPU test output required')
    args.output.mkdir(parents=True, exist_ok=False)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RecoveryTests))
    v2.write_new(args.output/'CPU_TEST_RECEIPT.json', {'schema': 'recovery_clarification_v2_synthetic_CPU_tests',
        'tests': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
        'CPU_tests_passed': result.wasSuccessful(), 'scope': 'SYNTHETIC ONLY; no actual model, recovery or host qualification claim',
        'GPU_jobs_started': 0, 'validator': v2.ref(v2.__file__), 'tests_source': v2.ref(__file__),
        'original_V1_registration': v2.ref(gate.REGISTRATION), 'controls': RecoveryTests.controls,
        'unittest_failures': [{'test': str(t), 'traceback': e} for t, e in result.failures+result.errors]})
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__': sys.exit(main())
