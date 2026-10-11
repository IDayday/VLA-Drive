"""Owned synthetic CPU evidence controls; no real qualification or GPU dispatch."""
from __future__ import annotations

import argparse
import copy
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import assemble_execution_evidence as adapter
import test_execution_acceptance as fixtures

ORIGINAL = Path('/mnt/project/VLA-Drive-structured-world-fgtr-perf-v3-frozen-20261011')


def write(path, value):
    adapter.new_json(path, value)
    return {'path': str(path), 'sha256': adapter.digest(path)}


class AssemblyTests(unittest.TestCase):
    def setUp(self):
        self.helper = fixtures.AcceptanceTests(methodName='runTest')
        self.helper.setUp()
        self.addCleanup(self.helper.doCleanups)
        self.root = self.helper.root
        self.helper.registration.write_bytes(adapter.REGISTRATION.read_bytes())
        self.helper.registration_hash = adapter.REGISTRATION_SHA
        self.helper.bindings['producer_sha256'] = {k: adapter.digest(adapter.__file__) for k in ('forward', 'native', 'recovery', 'dev')}
        self.helper.freeze_bindings()
        self.entries = []; self.serial = 0

    def path(self, name):
        self.serial += 1
        return self.root/(str(self.serial)+'_'+name)

    def header(self, receipt):
        value = {k: receipt[k] for k in adapter.HEADER_FIELDS}
        value['schema'] = 'execution_evidence_observation_v1'
        return write(self.path('observation.json'), value)

    def counts(self, comp, group, stage, updates):
        captures = {}; origin = comp['provenance']['origin']['completed']
        for role in adapter.ROLES:
            records = []
            for rank in range(8):
                rows = [{'updates': origin+i, 'epoch': 0, 'offset_after_update': i,
                         'global_counts': {'ego_scenes': 32.0, 'current_dino': 4096.0},
                         'schedule_horizon': 100000, 'schedule_warmup': 1000,
                         'continuation_endpoint': origin+updates, 'source_population_scenes': 101592,
                         'rank': rank, 'global_scene_indices': list(range(32)),
                         'local_scene_indices': list(range(rank, 32, 8)), 'loss_horizons': {}}
                        for i in range(1, updates+1)]
                path = self.path(role+f'_{rank}.jsonl')
                path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
                records.append({'rank': rank, 'path': str(path), 'sha256': adapter.digest(path), 'updates': updates})
            p = comp['provenance'][role]
            capture = {'schema': 'native_execution_capture_receipt_v1', 'passed': False, 'captures_complete': True,
                       'role': role, 'mode': p['execution_mode'], 'run_identity': p['identity'],
                       'training_source_sha': p['source'], 'count_evidence': records}
            captures[role] = write(self.path('capture.json'), capture)
        name = group+'_'+stage
        triplets = write(self.path('triplets.json'), {'discrete_evidence': [{'case': name, 'all8_count_scene_horizon_exact': True}], 'passed': False})
        return {'captures': captures, 'triplets': triplets, 'case': name}

    def entry(self, kind='native', group='G1', stage='mid', repeat=0, updates=1):
        path = self.helper.receipt(kind, group, stage, repeat, updates)
        receipt = adapter.load(path); payload = receipt['payload']
        entry = {'observation': self.header(receipt)}
        if kind == 'dev':
            native = next(adapter.load(e['comparison']['path'])
                          for e in self.entries if adapter.load(e['observation']['path'])['kind'] == 'native'
                          and adapter.load(e['observation']['path'])['group'] == group
                          and adapter.load(e['observation']['path'])['updates'] == 500)
            dev = self.helper.make_dev_payload(native['provenance'])
            entry['dev_payload'] = write(self.path('dev_payload.json'), dev)
            # make_dev_payload freezes an identical index/binding after initial setup.
            receipt['bindings_sha256'] = self.helper.binding_hash
            entry['observation'] = self.header(receipt)
        else:
            entry['comparison'] = payload['comparison']
            comp = adapter.load(entry['comparison']['path'])
            if kind == 'forward':
                entry['original_strict'] = payload['original_strict']
                entry['forward_observation'] = write(self.path('forward.json'), {'forward_exact': payload['forward_exact']})
            else:
                entry['counts'] = self.counts(comp, group, stage, updates)
                if kind == 'recovery':
                    entry['load_boundary_receipt'] = payload['load_boundary_receipt']
                    entry['load_boundary_source'] = write(self.path('native_boundary.json'), {
                        'identity': comp['provenance']['B']['identity'], 'completed': comp['provenance']['B']['completed'],
                        'passed': True, 'FP32_model_and_buffers': 'bitwise', 'FP32_masters': 'bitwise all ranks, actual parameters',
                        'Adam_moments_steps_and_LR': 'bitwise all ranks', 'all_task_RNG': 'exact all ranks',
                        'timing': 'before next forward/backward/update'})
        self.entries.append(entry)
        return entry

    def modify_ref(self, ref, change):
        path = Path(ref['path']); data = adapter.load(path); change(data)
        path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
        ref['sha256'] = adapter.digest(path)

    def run_assembly(self):
        manifest = self.path('manifest.json')
        adapter.new_json(manifest, {'schema': 'execution_evidence_assembly_manifest_v1', 'entries': self.entries})
        return adapter.assemble(manifest, self.helper.binding_path, self.helper.binding_hash,
                                self.path('output'), protected_source_root=ORIGINAL)

    def test_contract_exact_fields_and_required28_no_executed_facts(self):
        x = adapter.contract()
        self.assertEqual(len(x['required_receipt_keys']), 28)
        self.assertFalse(x['executed_trial_facts'])
        self.assertFalse(x['required_bindings']['adopted_for_pilot'])
        self.assertIsNone(x['required_bindings']['sources']['B']['git_sha'])

    def test_actual_producer_binding_separate_from_adapter_and_wrong_producer_rejected(self):
        producer = adapter.COMPARATOR_SHA
        self.helper.bindings['producer_sha256']['native'] = producer
        self.helper.freeze_bindings()
        entry = self.entry()
        result = self.run_assembly()
        self.assertEqual(len(result['receipt_paths']), 1, result['failures'])
        self.modify_ref(entry['observation'], lambda x: x.update(producer_sha256='0'*64))
        result = self.run_assembly()
        self.assertEqual(result['status'], 'not_supported')
        self.assertTrue(any('producer differs' in x for x in result['failures']))

    def binding_spec(self):
        # Entire source tree and qualifications below are OWNED synthetic data.
        root = self.root/'owned_source'; root.mkdir(exist_ok=True)
        files = ['starVLA/model/modules/structured_world/'+n for n in ('execution.py', 'losses.py', 'queries.py', 'grid.py')]
        files += ['starVLA/model/framework/vla_structured_fgtr.py', 'tools/ddpolicy_vehicle/training_state.py']
        files += ['tools/structured_world/'+n for n in ('train_vla.py', 'recovery_boundary.py', 'training_assets.py',
                  'stage_timing.py', 'prefetch.py', 'resume_origin.py', 'execution_log.py', 'native_capture.py')]
        for name in files:
            p = root/name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('# OWNED synthetic CPU binding fixture\n')
        packages = {k: 'owned-synthetic' for k in ('torch', 'deepspeed', 'transformers', 'flash_attn', 'numpy')}
        host = {'passed': True, 'FP32_master_update_verified': True, 'host': 'OWNED_SYNTHETIC_HOST',
                'physical_gpu_UUIDs': ['owned-synthetic-'+str(n) for n in range(8)],
                'environment': {'packages': packages, 'torch_cuda_runtime': 'owned', 'cudnn_version': 1, 'NCCL_version': [1, 0, 0]},
                'source': {'git_SHA': 'a'*40, 'training_source_tree_sha256': adapter.metadata_hash(adapter.source_inventory(root))}}
        cpu = {'passed_CPU': True, 'python_version': 'owned', 'physical_gpu_inventory': [
            {'uuid': u, 'name': 'OWNED_SYNTHETIC_GPU', 'driver': 'owned'} for u in host['physical_gpu_UUIDs']]}
        refs = {}
        for group in ('G1', 'G3'):
            for stage in ('initial', 'mid'):
                completed = 2 if stage == 'initial' else 25000 if group == 'G1' else 50000
                i = {'identity': 'OWNED_SYNTHETIC_ORIGIN_'+group+stage, 'world_size': 8, 'microbatch': 4,
                     'training_seed': 42, 'effective_global_batch': 32, 'FM_repeat': 8, 'DINO_identity': 'owned-dino',
                     'current_and_GT_identity': {'identity': 'owned-current'}, 'shared_geometry_identity': 'owned-geometry',
                     'schedule_horizon': 100000, 'schedule_warmup': 5000, 'determinism': {'warn_only': True},
                     'config': {'structured_world': {'group': group+'_OWNED', 'grid': {'dx': .5}, 'geometry_image_size': [256, 448]},
                                'trainer': {'lr_scheduler_type': 'cosine', 'scheduler_specific_kwargs': {'min_lr': 1e-7}, 'learning_rate': {'base': 1e-5}}}}
                layout = {'owned_parameter': [4]}; grad = [['owned_parameter', [4], 'torch.float32']]
                m = {'schema': 'execution_origin_metadata_v1', 'tensor_values_read': False, 'identity': i,
                     'COMPLETE': {'identity': i['identity'], 'completed': completed, 'file_sha256': {'owned': '1'*64}},
                     'provenance': {'identity': i['identity'], 'COMPLETE_sha256': '2'*64, 'optimizer_groups_sha256': '3'*64},
                     'model_layout': layout, 'expected_gradient_inventory': grad,
                     'parameter_inventory_sha256': adapter.metadata_hash(layout), 'gradient_inventory_sha256': adapter.metadata_hash(grad),
                     'gradient_inventory_semantics': 'OWNED prospective fixture only',
                     'category_inventory': self.helper.bindings['cases']['G1/mid']['category_inventory']}
                refs[group+'/'+stage] = write(self.path('owned_origin.json'), m)
        protocol = {k: 'owned-synthetic' for k in self.helper.policy['dev']['protocol_bindings']}
        return write(self.path('owned_build_spec.json'), {
            'schema': 'execution_binding_build_spec_v1', 'source_root': str(root),
            'host_qualification': write(self.path('owned_host.json'), host), 'CPU_qualification': write(self.path('owned_CPU.json'), cpu),
            'origins': refs, 'train_index': write(self.path('owned_train_index.json'), ['owned-scene']),
            'dev_protocol': write(self.path('owned_protocol.json'), protocol),
            'producers': {k: {'path': str(adapter.COMPARATOR), 'sha256': adapter.COMPARATOR_SHA} for k in ('forward', 'native', 'recovery', 'dev')}})

    def test_binding_builder_fills_four_origins_pins_sources_no_executed_facts_no_overwrite(self):
        spec = self.binding_spec(); output = self.path('owned_binding')
        with patch.object(adapter.subprocess, 'check_output', side_effect=lambda cmd, **kw: 'a'*40 if 'rev-parse' in cmd else ''):
            result = adapter.build_bindings(spec['path'], output)
            with self.assertRaisesRegex(ValueError, 'NEW binding output'):
                adapter.build_bindings(spec['path'], output)
        b = adapter.load(result['bindings']['path'])
        self.assertEqual(set(b['cases']), {'G1/initial', 'G1/mid', 'G3/initial', 'G3/mid'})
        self.assertEqual(b['cases']['G1/initial']['origin']['completed'], 2)
        self.assertEqual(b['cases']['G3/mid']['origin']['completed'], 50000)
        self.assertEqual(b['producer_sha256']['native'], adapter.COMPARATOR_SHA)
        self.assertFalse(result['executed_candidate_facts']); self.assertFalse(result['payload_hashes_verified'])
        self.assertFalse(result['formal_authorized']); self.assertEqual(result['GPU_jobs_started'], 0)

    def test_binding_builder_missing_case_and_corrupt_metadata_reject(self):
        for change in (lambda x: x['origins'].pop('G3/initial'),
                       lambda x: x['origins']['G1/mid'].update(sha256='0'*64)):
            spec = self.binding_spec(); self.modify_ref(spec, change)
            with patch.object(adapter.subprocess, 'check_output', side_effect=lambda cmd, **kw: 'a'*40 if 'rev-parse' in cmd else ''):
                with self.assertRaises(ValueError):
                    adapter.build_bindings(spec['path'], self.path('owned_rejected_binding'))

    def test_origin_inspection_owned_native_metadata_only_and_prescribed_coverage(self):
        import torch
        import test_compare_execution_checkpoints as native
        origin = native.native_fixture(self.root, 'owned_metadata_origin', completed=1, offset=1)
        m = adapter.inspect_origin(origin)
        self.assertFalse(m['tensor_values_read']); self.assertFalse(m['payload_hashes_verified'])
        self.assertEqual(m['provenance']['bytes_hashed'], 0)
        self.assertEqual(m['gradient_inventory_sha256'], adapter.metadata_hash(m['expected_gradient_inventory']))
        trainable = sum(__import__('math').prod(row[1]) for row in m['expected_gradient_inventory'])
        for category in ('master_FP32', 'Adam/exp_avg', 'Adam/exp_avg_sq'):
            self.assertEqual(m['category_inventory'][category]['all'], trainable)
        self.assertFalse(torch.cuda.is_initialized())

    def test_absent_and_missing_observations_stay_pending(self):
        self.entries = [{'observation': {'path': str(self.root/'missing.json'), 'sha256': '0'*64}}]
        r = self.run_assembly()
        self.assertFalse(r['independent_evidence_satisfied']); self.assertEqual(r['status'], 'pending')
        self.assertEqual(len(r['missing_receipt_keys']), 28)

    def test_native_oldstrict_false_and_actual_count_refs_preserved(self):
        entry = self.entry(); before = adapter.digest(entry['comparison']['path'])
        result = self.run_assembly()
        self.assertEqual(result['status'], 'pending', result['failures'])
        self.assertEqual(len(result['receipt_paths']), 1)
        receipt = adapter.load(result['receipt_paths'][0])
        self.assertTrue(receipt['payload']['count_scene_horizon']['all8_count_scene_horizon_exact'])
        self.assertFalse(result['old_strict_results_preserved'][0]['original_strict']['A_B']['passed'])
        self.assertEqual(before, adapter.digest(entry['comparison']['path']))
        self.assertFalse(result['passed']); self.assertFalse(result['trainer_consumable'])

    def test_count_triplet_true_cannot_hide_single_rank_denominator_drift(self):
        entry = self.entry()
        cap_ref = entry['counts']['captures']['B']; cap = adapter.load(cap_ref['path'])
        record = cap['count_evidence'][3]; path = Path(record['path']); rows = [json.loads(x) for x in path.read_text().splitlines()]
        rows[0]['global_counts']['current_dino'] += 1
        path.write_text(''.join(json.dumps(x)+'\n' for x in rows)); record['sha256'] = adapter.digest(path)
        self.modify_ref(cap_ref, lambda x: x.update(count_evidence=cap['count_evidence']))
        r = self.run_assembly(); self.assertEqual(r['status'], 'not_supported')
        self.assertTrue(any('triplet drift' in x for x in r['failures']))

    def test_comparison_incomplete_and_skipped_integrity_never_qualify(self):
        for key, status in (('comparison_performed', 'pending'), ('integrity_verified', 'not_supported')):
            self.entries = []; entry = self.entry()
            self.modify_ref(entry['comparison'], lambda x: x.update({key: False}))
            r = self.run_assembly(); self.assertEqual(r['status'], status)
            self.assertFalse(r['independent_evidence_satisfied'])

    def test_retrospective_and_observed_source_mismatch_reject(self):
        for mutate in (lambda x: x.update(started_utc='2020-01-01T00:00:00+00:00'),
                       lambda x: x['roles']['B']['source'].update(git_sha='c'*40)):
            self.entries = []; entry = self.entry()
            self.modify_ref(entry['observation'], mutate)
            r = self.run_assembly(); self.assertEqual(r['status'], 'not_supported')

    def test_recovery_split16_not_relabelled_as32(self):
        entry = self.entry('recovery', updates=32)
        self.modify_ref(entry['load_boundary_source'], lambda x: x.update(completed=x['completed']-16))
        r = self.run_assembly(); self.assertEqual(r['status'], 'pending', r['failures'])
        self.assertTrue(any('cannot relabel split16' in x for x in r['pending']))

    def test_legacy_gradient_summary_not_fabricated_and_P0_violation_rejects(self):
        entry = self.entry('forward', updates=0)
        self.modify_ref(entry['comparison'], lambda x: x.update(schema='legacy_summary_only'))
        r = self.run_assembly(); self.assertEqual(r['status'], 'pending')
        self.entries = []; entry = self.entry('forward', updates=0)
        self.modify_ref(entry['forward_observation'], lambda x: x['forward_exact'].update(Bt=False))
        r = self.run_assembly(); self.assertEqual(r['status'], 'not_supported')

    def prepare_index_binding(self):
        self.helper.make_dev_payload({r: {'synthetic_index_setup': True} for r in adapter.ROLES})

    def test_actual_dev_population_partial_csv_rejects(self):
        self.prepare_index_binding(); self.entry(updates=500); entry = self.entry('dev', updates=500)
        payload = adapter.load(entry['dev_payload']['path']); bank = payload['banks']['B']['q_final']
        path = Path(bank['csv']['path']); rows = path.read_text().splitlines(); path.write_text('\n'.join(rows[:-1])+'\n')
        bank['csv']['sha256'] = adapter.digest(path)
        self.modify_ref(entry['dev_payload'], lambda x: x.update(banks=payload['banks']))
        r = self.run_assembly(); self.assertEqual(r['status'], 'not_supported')

    def test_complete_owned_synthetic28_can_be_review_ready_with_oldstrict_false(self):
        self.prepare_index_binding()
        for kind, group, stage, repeat, updates in sorted(adapter.reviewer().required_keys(self.helper.policy), key=lambda k: k[0] == 'dev'):
            self.entry(kind, group, stage, repeat, updates)
        r = self.run_assembly()
        self.assertTrue(r['independent_evidence_satisfied'], adapter.load(r['independent_review']['path']))
        self.assertEqual(r['status'], 'evidence_complete_for_review')
        self.assertFalse(r['passed']); self.assertFalse(r['formal_authorized']); self.assertIsNone(r['accepted_mode'])
        self.assertTrue(any(not x['original_strict']['A_B']['passed'] for x in r['old_strict_results_preserved'] if 'original_strict' in x))

    def test_hash_corruption_and_duplicate_receipts_reject(self):
        entry = self.entry(); Path(entry['observation']['path']).write_text('{}')
        r = self.run_assembly(); self.assertEqual(r['status'], 'not_supported')
        self.entries = []; entry = self.entry(); self.entries.append(copy.deepcopy(entry))
        r = self.run_assembly(); self.assertEqual(r['status'], 'not_supported')

    def test_actual_protected_comparator_tiny_CPU_native_banks_exit_preserved(self):
        import torch
        import test_compare_execution_checkpoints as native
        origin = native.native_fixture(self.root, 'owned_origin', completed=1, offset=1)
        identity = adapter.load(origin.parent.parent/'identity.json')
        ancestry = {'run': str(origin.parent.parent), 'tag': origin.name, 'identity': identity['identity'],
                    'completed': 1, 'complete_sha256': adapter.digest(origin/'COMPLETE.json')}
        banks = [native.native_fixture(self.root, role, origin_binding=ancestry) for role in adapter.ROLES]
        file = banks[2]/'zero_pp_rank_0_mp_rank_00_optim_states.pt'
        value = torch.load(file, map_location='cpu', weights_only=False)
        value['optimizer_state_dict']['base_optimizer_state']['state'][0]['exp_avg'][2] += .002
        torch.save(value, file); native.rehash(banks[2])
        output = self.path('actual_comparison.json')
        result = adapter.run_comparison(*banks, output, origin=origin, threads=1, chunk_elements=2)
        self.assertEqual(result['actual_exit_code'], 2)
        self.assertTrue(result['comparison_performed']); self.assertTrue(result['integrity_verified'])
        self.assertFalse(result['original_passed']); self.assertFalse(torch.cuda.is_initialized())
        with self.assertRaisesRegex(ValueError, 'NEW comparator output'):
            adapter.run_comparison(*banks, output, origin=origin)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    paths = [Path(adapter.__file__), Path(__file__), adapter.REVIEWER, adapter.COMPARATOR, adapter.REGISTRATION]
    before = {str(p): adapter.digest(p) for p in paths}
    start = time.monotonic(); suite = unittest.defaultTestLoader.loadTestsFromTestCase(AssemblyTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    after = {str(p): adapter.digest(p) for p in paths}
    torch = sys.modules.get('torch')
    adapter.new_json(args.output, {'schema': 'execution_evidence_assembly_CPU_controls_v1', 'passed': result.wasSuccessful(),
                     'tests_run': result.testsRun, 'failures': [(str(t), e) for t, e in result.failures],
                     'errors': [(str(t), e) for t, e in result.errors], 'source_sha256': after, 'sources_unchanged': before == after,
                     'CUDA_initialized': False if torch is None else torch.cuda.is_initialized(), 'wall_seconds': time.monotonic()-start,
                     'qualification': 'Owned synthetic CPU controls only; includes actual protected comparator on tiny CPU native banks. No trained GPU/native/dev acceptance.'})
    raise SystemExit(0 if result.wasSuccessful() and before == after else 1)


if __name__ == '__main__':
    main()
