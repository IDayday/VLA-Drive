"""CPU controls with real pinned source/criteria and tiny observation fixtures.

No source/model construction, tensor loads, GPU jobs or real enrollment.
Synthetic environments do not qualify the recovered container.
"""
import copy
from datetime import datetime, timedelta
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.structured_world import prepare_execution_e2_envelope as helper


class EnvelopeControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old = helper.load(helper.HISTORY)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.spec = {'schema': 'prospective_execution_E2_envelope_spec_v1', 'phases': {
            k: {'host': 'OWNED_CPU_TEST_HOST', 'environment_receipt': None}
            for k in ('forward', 'native', 'recovery', 'dev')}}

    def path(self, name):
        return self.root/name

    def save(self, name, value):
        p = self.path(name); helper.write_new(p, value); return helper.ref(p)

    def prepare(self, freeze=False):
        spec = self.save('spec.json', self.spec)
        return helper.prepare(spec['path'], self.path('envelope'), freeze)

    def complete_synthetic_environments(self):
        value = {'host': 'OWNED_CPU_TEST_HOST', 'observed_utc': '2026-10-11T15:00:00+00:00',
                 'gpu_uuids': ['GPU-OWNED-SYNTHETIC-'+str(i) for i in range(8)],
                 'environment': self.old['environment']}
        ref = self.save('synthetic_environment.json', value)
        for phase in self.spec['phases'].values():
            phase['environment_receipt'] = ref
        self.spec['physical_binding_paused'] = False
        self.spec['storage_plan'] = self.save('synthetic_storage_plan.json', {
            'status': 'OWNED_SYNTHETIC_ONLY_NOT_ACTUAL_STORAGE', 'GPU_jobs_started': 0})

    def frozen(self):
        self.complete_synthetic_environments(); return self.prepare(True)

    def trial(self, mode='native'):
        script = {'native': helper.MODEL/'tools/structured_world/train_vla.py',
                  'forward': helper.PRODUCERS['forward'][0]}[mode]
        return self.save('trial.json', {'schema': 'prospective_execution_E2_trial_v1',
            'host': 'OWNED_CPU_TEST_HOST', 'key': [mode, 'G1', 'mid', 0, 1 if mode == 'native' else 0],
            'command': ['python', str(script)], 'future_artifact_paths': [str(self.path('future_observation.json'))]})

    def enroll(self, ref):
        with patch.object(helper.socket, 'gethostname', return_value='OWNED_CPU_TEST_HOST'):
            return helper.enroll(self.path('envelope'), ref['path'], self.path('enrollment.json'))

    def test_draft_is_pending_all28_no_old_environment_substitution(self):
        before = helper.digest(helper.HISTORY); x = self.prepare()
        self.assertEqual(len(x['required_receipt_keys']), 28)
        self.assertFalse(x['passed']); self.assertFalse(x['thresholds_changed'])
        b = helper.load(x['bindings']['path'])
        self.assertFalse(b['adopted_for_pilot']); self.assertTrue(all(v is None for v in b['environment'].values()))
        self.assertEqual(x['actual_training_source'], helper.PINS['model'][1])
        self.assertNotEqual(x['actual_training_source'], x['locked_scientific_origin_source'])
        self.assertEqual(before, helper.digest(helper.HISTORY))
        self.assertEqual(helper.load(self.path('envelope/ADVISORY_PENDING.json'))['advisory_decision'], 'pending')

    def test_actual_complete_synthetic_environments_bind_new_producers_not_adapter(self):
        x = self.frozen(); b = helper.load(x['bindings']['path'])
        self.assertTrue(b['adopted_for_pilot']); self.assertFalse(x['passed'])
        self.assertEqual(b['producer_sha256']['forward'], helper.FULL_SHA)
        self.assertEqual(b['producer_sha256']['recovery'], helper.COMPARATOR_SHA)
        self.assertEqual(b['producer_sha256']['dev'], helper.PRODUCERS['dev'][1])
        self.assertNotEqual(b['producer_sha256']['forward'], helper.ASSEMBLER_SHA)
        self.assertEqual(b['cases'], self.old['cases'])

    def test_unknown_environment_cannot_freeze(self):
        with self.assertRaisesRegex(ValueError, 'actual complete compatible'):
            self.prepare(True)
        self.assertFalse(self.path('envelope').exists())

    def test_latest_user_storage_pause_cannot_be_overridden_by_complete_environment(self):
        self.complete_synthetic_environments(); self.spec['physical_binding_paused'] = True
        with self.assertRaisesRegex(ValueError, 'Physical binding paused'):
            self.prepare(True)
        self.assertFalse(self.path('envelope').exists())

    def test_missing_requested_storage_plan_cannot_freeze(self):
        self.complete_synthetic_environments(); self.spec.pop('storage_plan')
        with self.assertRaisesRegex(ValueError, 'shared storage plan'):
            self.prepare(True)

    def test_foreign_environment_host_rejected(self):
        self.complete_synthetic_environments()
        self.spec['phases']['native']['host'] = 'FOREIGN_HOST'
        with self.assertRaisesRegex(ValueError, 'environment/route'):
            self.prepare(True)

    def test_actual_phase_environment_mismatch_is_not_silently_joined(self):
        self.complete_synthetic_environments()
        value = helper.document(self.spec['phases']['native']['environment_receipt'])
        value['environment']['cuda_driver'] = 'FOREIGN_DRIVER'
        self.spec['phases']['native']['environment_receipt'] = self.save('different_env.json', value)
        with self.assertRaisesRegex(ValueError, 'differing phase environments'):
            self.prepare(True)

    def test_environment_hash_corruption_rejected(self):
        self.complete_synthetic_environments()
        self.spec['phases']['native']['environment_receipt']['sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'Pinned bytes changed'):
            self.prepare(True)

    def test_no_output_overwrite(self):
        self.prepare()
        with self.assertRaisesRegex(ValueError, 'NEW envelope'):
            helper.prepare(self.path('spec.json'), self.path('envelope'))

    def test_draft_cannot_enroll(self):
        self.prepare(); trial = self.trial()
        with self.assertRaisesRegex(ValueError, 'Draft'):
            self.enroll(trial)

    def test_enrollment_on_worker_pins_binding_and_never_launches(self):
        x = self.frozen(); trial = self.trial(); r = self.enroll(trial)
        self.assertEqual(r['bindings'], x['bindings']); self.assertFalse(r['passed'])
        self.assertEqual(r['GPU_jobs_started'], 0)
        self.assertFalse(self.path('future_observation.json').exists())

    def test_existing_legacy_output_cannot_be_enrolled(self):
        self.frozen(); trial = self.trial(); self.path('future_observation.json').write_text('OLD\n')
        with self.assertRaisesRegex(ValueError, 'retrospectively'):
            self.enroll(trial)

    def test_wrong_namespace_cannot_enroll_remote_future_paths(self):
        self.frozen(); trial = self.trial()
        with patch.object(helper.socket, 'gethostname', return_value='WRONG_HOST'):
            with self.assertRaisesRegex(ValueError, 'actual worker'):
                helper.enroll(self.path('envelope'), trial['path'], self.path('enrollment.json'))

    def test_unpinned_entrypoint_cannot_enroll(self):
        self.frozen(); trial = self.trial(); j = helper.document(trial)
        j['command'] = ['python', '/tmp/unreviewed_training.py']
        p = self.save('bad_trial.json', j)
        with self.assertRaisesRegex(ValueError, 'Pinned actual entrypoint'):
            self.enroll(p)

    def test_historical_candidate_cannot_be_relabelled_with_new_binding(self):
        self.frozen(); trial = self.trial(); self.enroll(trial)
        header = self.save('old_header.json', {'kind': 'native', 'group': 'G1', 'stage': 'mid',
            'repeat': 0, 'updates': 1, 'started_utc': self.old['frozen_utc']})
        entry = self.save('entry.json', {'observation': header, 'external_enrollment': helper.ref(self.path('enrollment.json'))})
        r = helper.check_entry(self.path('envelope'), entry['path'], self.path('checked.json'))
        self.assertEqual(r['status'], 'not_supported'); self.assertIn('predates', r['error'])
        self.assertFalse(r['passed'])

    def test_fullgrad_old_legacy_summary_cannot_supply_actual_gradients(self):
        self.frozen(); trial = self.trial('forward'); self.enroll(trial)
        start = helper.load(self.path('enrollment.json'))['created_utc']
        header = self.save('header.json', {'kind': 'forward', 'group': 'G1', 'stage': 'mid',
            'repeat': 0, 'updates': 0, 'started_utc': start})
        comparison = self.save('legacy.json', {'producer_source_binding': {'path': 'old', 'sha256': '0'*64}})
        entry = self.save('entry.json', {'observation': header, 'comparison': comparison,
            'external_enrollment': helper.ref(self.path('enrollment.json'))})
        r = helper.check_entry(self.path('envelope'), entry['path'], self.path('checked.json'))
        self.assertEqual(r['status'], 'not_supported'); self.assertFalse(r['passed'])

    def test_original_criteria_full_scopes_and_strict_failure_remain(self):
        x = self.prepare(); c = helper.load(self.path('envelope/CRITERIA_UNCHANGED.json')); old = helper.load(helper.REG)
        for key in ('numeric', 'trial_plan', 'dev', 'E1'):
            self.assertEqual(c[key], old[key])
        self.assertEqual(c['numeric']['E2_AA_envelope']['factor'], 1.25)
        self.assertEqual(c['trial_plan']['mid_native_updates'], [1, 32, 100, 250, 500])
        self.assertEqual(x['fullgrad_source_binding']['sha256'], helper.GRAD_BINDING_SHA)
        self.assertFalse(x['legacy_reuse']['payload_recovery_verified'])

    def runtime(self):
        old = self.old['environment']
        return {'host': 'training-vlawm-zt4', 'observed_utc': '2026-10-11T15:00:00+00:00',
            'CUDA_initialized': False, 'python': old['python'], 'cuda_runtime': old['cuda_runtime'],
            'cudnn': old['cudnn'], 'nccl': old['nccl'],
            'packages': {k: old[k] for k in ('torch', 'deepspeed', 'transformers', 'flash_attn', 'numpy')},
            'gpus': [f'GPU-OWNED-{i}, {old["gpu_model"]}, {old["cuda_driver"]}' for i in range(8)]}

    def forward_binding(self):
        rt = self.save('runtime.json', self.runtime())
        return helper.freeze_forward(rt['path'], self.path('future_fullgrad'), self.path('forward_frozen'))

    def test_forward_only_freeze_does_not_require_future_native_routes_or_claim_whole(self):
        x = self.forward_binding()
        self.assertEqual(len(x['expected_slots']), 12)
        self.assertFalse(x['whole_E2_adopted']); self.assertFalse(x['passed'])
        self.assertEqual(x['producer_source_binding']['sha256'], helper.GRAD_BINDING_SHA)
        self.assertEqual(x['coordinator']['sha256'], helper.FULL_DRIVER_SHA)
        self.assertEqual(x['model']['git_sha'], helper.PINS['model'][1])
        self.assertEqual(x['environment']['torch'], '2.5.1+cu124')

    def test_forward_only_rejects_incomplete_hardware_inventory(self):
        rt = self.runtime(); rt['gpus'] = rt['gpus'][:7]
        p = self.save('runtime.json', rt)
        with self.assertRaisesRegex(ValueError, 'Full8'):
            helper.freeze_forward(p['path'], self.path('future'), self.path('frozen'))

    def test_forward_only_cuda_initialized_receipt_not_accepted_as_cpu_snapshot(self):
        rt = self.runtime(); rt['CUDA_initialized'] = True
        p = self.save('runtime.json', rt)
        with self.assertRaisesRegex(ValueError, 'CPU runtime'):
            helper.freeze_forward(p['path'], self.path('future'), self.path('frozen'))

    def test_forward_coordinator_pin_hash_mismatch_cannot_freeze(self):
        rt = self.save('runtime.json', self.runtime())
        pin = self.save('coordinator.json', {'root': str(helper.FULL_DRIVER),
            'git_sha': helper.FULL_DRIVER_GIT, 'script_sha256': '0'*64})
        with self.assertRaisesRegex(ValueError, 'Pinned bytes changed'):
            helper.freeze_forward(rt['path'], self.path('future'), self.path('frozen'), pin['path'])
        self.assertFalse(self.path('frozen').exists())

    def test_forward_coordinator_pin_is_explicit_and_hash_bound(self):
        rt = self.save('runtime.json', self.runtime())
        pin = self.save('coordinator.json', {'root': str(helper.FULL_DRIVER),
            'git_sha': helper.FULL_DRIVER_GIT, 'script_sha256': helper.FULL_DRIVER_SHA})
        x = helper.freeze_forward(rt['path'], self.path('future'), self.path('frozen'), pin['path'])
        self.assertEqual(x['coordinator_pin_receipt'], pin)
        self.assertFalse(x['passed'])

    def test_actual_native_CPU_reader_accepts17_hash_identical_payload_symlinks(self):
        # This checks the unchanged real reader and mmap transport only; tiny
        # native-shaped tensors are NOT real optimizer/numerical qualification.
        import torch
        torch.set_num_threads(2)
        source = helper.MODEL/'tools/structured_world/compare_execution_checkpoints.py'
        native = helper.module(source, helper.COMPARATOR_SHA, '_own_CPU_symlink_reader')
        run = self.path('shared/run'); bank = run/'checkpoints/milestone_0025001'; bank.mkdir(parents=True)
        scratch = self.path('local_scratch'); scratch.mkdir()
        identity = {'identity': 'OWNED_SYNTHETIC8'}
        (run/'identity.json').write_text(json.dumps(identity))
        (run/'optimizer_groups.json').write_text('[]')
        payload = {'mp_rank_00_model_states.pt': {'identity': identity['identity'], 'completed': 25001,
            'dp_world_size': 8, 'module': {'owned_probe': torch.tensor([1., 2.])}}}
        for rank in range(8):
            payload[f'rng_rank{rank}.pt'] = {'rank': rank, 'owned_probe': torch.tensor([rank])}
            payload[f'zero_pp_rank_{rank}_mp_rank_00_optim_states.pt'] = {'optimizer_state_dict': {'zero_stage': 2, 'base_optimizer_state': {}}}
        for name, value in payload.items(): torch.save(value, bank/name)
        hashes = {name: helper.digest(bank/name) for name in payload}
        complete = {'identity': identity['identity'], 'completed': 25001, 'ranks': 8, 'file_sha256': hashes}
        (bank/'COMPLETE.json').write_text(json.dumps(complete))
        before = helper.digest(bank/'COMPLETE.json')
        original = native.Bank(bank, verify_hashes=True).model()['module']['owned_probe'].clone()
        for name in payload:
            target = scratch/name; target.write_bytes((bank/name).read_bytes())
            self.assertEqual(helper.digest(target), hashes[name])
        for name in payload:
            (bank/name).unlink(); (bank/name).symlink_to(scratch/name)
        linked = native.Bank(bank, verify_hashes=True)
        self.assertEqual(linked.ranks, 8); self.assertEqual(len(complete['file_sha256']), 17)
        self.assertEqual(before, helper.digest(bank/'COMPLETE.json'))
        self.assertTrue(torch.equal(original, linked.model()['module']['owned_probe']))
        for rank in range(8): self.assertEqual(linked.optimizer(rank)['zero_stage'], 2)
        self.assertFalse(torch.cuda.is_initialized())

    def bridge_fixture(self):
        binding = self.forward_binding()
        folder = self.path('future_fullgrad/G1_mid_repeat0'); folder.mkdir(parents=True)
        job = {'case': 'G1/mid', 'repeat': 0, 'folder': str(folder)}
        probe = self.save('probe.json', {'job': job})
        start = (datetime.fromisoformat(binding['frozen_utc'])+timedelta(seconds=1)).isoformat()
        forward = self.save('forward.json', {'per_rank': [{'roles': {'B': {'started_UTC': start}}} for _ in range(8)],
            'forward_exact': {k: True for k in ('q0', 'Bt', 'q_final', 'loss', 'RNG', 'BN_float', 'BN_counter', 'buffer_inventory', 'gradient_inventory')}})
        row = {'elements': 4, 'nonfinite_elements': 0, 'reference_norm': 2., 'candidate_norm': 2.,
               'difference_norm': 2e-5, 'relative_L2': 1e-5, 'cosine': 1.-5e-11}
        report = {'model_source': binding['model'], 'producer_source_binding': binding['producer_source_binding'],
            'forward_observation': forward, 'original_strict': {'passed': False, 'sha256': '0'*64},
            'comparisons': {p: {'gradient': {'all': row, 'by_component': {'geometry': row}, 'by_optimizer_group': {'0': row}}} for p in ('A_A', 'A_B', 'AA_B')}}
        comparison = self.save('comparison.json', report)
        from types import SimpleNamespace
        classified = {'P0_FULL_GRADIENT_COMPARISON': comparison, 'expected_legacy_strict_failure': True, 'actual_returncode': 1}
        coordinator = SimpleNamespace(classify=lambda job: classified)
        original_module = helper.module
        def modules(path, expected, name):
            if name == '_prospective_actual_fullgrad_coordinator': return coordinator
            return original_module(path, expected, name)
        return binding, probe, forward, modules

    def test_forward_bridge_preserves_failed_strict_and_pending_whole(self):
        binding, probe, forward, modules = self.bridge_fixture()
        self.assertEqual(binding['model'], helper.load(helper.GRAD_BINDING)['model'])
        self.assertEqual(len(binding['model']['files_sha256']), 431)
        with patch.object(helper, 'module', side_effect=modules):
            r = helper.bridge_forward(binding['binding']['path'], probe['path'], self.path('bridge.json'))
        self.assertEqual(r['status'], 'forward_only_reviewed_whole_pending')
        self.assertFalse(r['original_strict']['passed']); self.assertFalse(r['passed'])
        self.assertFalse(r['whole_receipt_generated']); self.assertEqual(r['GPU_jobs_started'], 0)

    def test_forward_bridge_rejects_missing_or_corrupt_full_model_inventory(self):
        binding, probe, forward, modules = self.bridge_fixture()
        original = helper.load(self.path('comparison.json'))
        projected = {k: original['model_source'][k] for k in ('root', 'git_sha')}
        corrupt = copy.deepcopy(original['model_source'])
        corrupt['files_sha256'][next(iter(corrupt['files_sha256']))] = '0'*64
        from types import SimpleNamespace
        for invalid in (projected, corrupt):
            with self.subTest(inventory='missing' if invalid is projected else 'corrupt'):
                report = copy.deepcopy(original); report['model_source'] = invalid
                self.path('comparison.json').write_text(json.dumps(report))
                fresh = helper.ref(self.path('comparison.json'))
                with patch.object(helper, 'module', side_effect=lambda path, expected, name: SimpleNamespace(classify=lambda job: {'P0_FULL_GRADIENT_COMPARISON': fresh}) if name == '_prospective_actual_fullgrad_coordinator' else modules(path, expected, name)):
                    with self.assertRaisesRegex(ValueError, 'full869 model inventory'):
                        helper.bridge_forward(binding['binding']['path'], probe['path'], self.path('bridge.json'))
                self.assertFalse(self.path('bridge.json').exists())

    def test_forward_bridge_rejects_candidate_started_before_new_partial_freeze(self):
        binding, probe, forward, modules = self.bridge_fixture()
        j = helper.document(forward)
        for row in j['per_rank']: row['roles']['B']['started_UTC'] = self.old['frozen_utc']
        self.path('forward.json').write_text(json.dumps(j))
        report = helper.load(self.path('comparison.json')); report['forward_observation'] = helper.ref(self.path('forward.json'))
        self.path('comparison.json').write_text(json.dumps(report))
        fresh = helper.ref(self.path('comparison.json'))
        original_modules = modules
        from types import SimpleNamespace
        with patch.object(helper, 'module', side_effect=lambda path, expected, name: SimpleNamespace(classify=lambda job: {'P0_FULL_GRADIENT_COMPARISON': fresh}) if name == '_prospective_actual_fullgrad_coordinator' else original_modules(path, expected, name)):
            with self.assertRaisesRegex(ValueError, 'predates'):
                helper.bridge_forward(binding['binding']['path'], probe['path'], self.path('bridge.json'))
        self.assertFalse(self.path('bridge.json').exists())


if __name__ == '__main__':
    unittest.main()
