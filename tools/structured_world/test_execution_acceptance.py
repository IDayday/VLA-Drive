"""Synthetic CPU acceptance-validator tests; no real model qualification.

The fixtures deliberately fail OLD strict while meeting the NEW prospective
proposal, then corrupt provenance, coverage, recovery and numeric evidence.
Complete synthetic dev fixtures contain 1696 distinct rows across sixteen logs;
a one-line CSV claiming that population must fail independently of its summary.
"""
import argparse
import copy
import csv
from datetime import timedelta
import json
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np

import evaluate_execution_acceptance as gate


def write(path, value):
    path = Path(path)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    return {'path': str(path), 'sha256': gate.digest(path)}


def stats(a, b, origin, updates):
    d = float(np.linalg.norm(b-a))
    return {'elements': len(a), 'nonfinite_elements': 0,
            'reference_norm': float(np.linalg.norm(a)), 'candidate_norm': float(np.linalg.norm(b)),
            'difference_norm': d, 'relative_L2': d/float(np.linalg.norm(a)),
            'cosine': float(np.dot(a,b)/np.linalg.norm(a)/np.linalg.norm(b)),
            'max_absolute': float(np.abs(a-b).max()),
            'logical_tensor_max_error_quantiles': {p: float(np.abs(a-b).max()) for p in ('p50','p90','p99')},
            'worst_tensors': [], 'origin_relative_updates': {'update_norms': updates}}


def comparison(policy, native=True, aa_error=1e-5, b_error=1.1e-5):
    origin = np.full(4, .9)
    values = {'A': np.ones(4), 'AA': np.ones(4)+aa_error, 'B': np.ones(4)+b_error}
    updates = {k: float(np.linalg.norm(v-origin)) for k,v in values.items()}
    categories = policy['numeric']['required_native_categories'] if native else ['gradient']
    pairs = {}
    for pair, (a,b) in gate.PAIRS.items():
        pairs[pair] = {}
        for category in categories:
            row = stats(values[a], values[b], origin, updates)
            pairs[pair][category] = {'all': row, 'by_component': {'geometry': copy.deepcopy(row)},
                                    'by_optimizer_group': {} if category == 'BN_float' else {'0': copy.deepcopy(row)}}
    strict = {p: {'E1_exact': True, 'FP32_bitwise': False, 'passed': False} for p in gate.PAIRS}
    return {'schema': 'native_execution_checkpoint_comparison_v1' if native else 'execution_acceptance_gradient_stats_v1',
            'comparisons': pairs, 'original_strict': strict,
            'fixed_numeric_results': {p: {'E1_exact': True, 'passed': False, 'violating_elements': 1} for p in gate.PAIRS},
            'valid_native_inputs': True, 'comparison_performed': True, 'integrity_verified': True,
            'coverage': {'all_rank_slice_membership_complete': True},
            'E1': {p: {'passed': True, 'failure_count': 0, 'failure_paths': []} for p in gate.PAIRS},
            'tool_sha256': next(r['sha256'] for r in policy['protected_files'] if r['path'].endswith('/compare_execution_checkpoints.py'))}


class AcceptanceTests(unittest.TestCase):
    controls = []

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.policy = gate.load(gate.REGISTRATION)
        self.registration = self.root/'registration.json'
        write(self.registration, self.policy)
        self.registration_hash = gate.digest(self.registration)
        self.bindings = self.make_bindings()
        self.binding_path = self.root/'bindings.json'
        self.freeze_bindings()
        self.receipts = {}

    def make_bindings(self, mode='engineering_fast'):
        b = gate.binding_template(self.policy, self.registration_hash)
        b.update(adopted_for_pilot=True, candidate_mode=mode,
                 tier='E3' if mode == 'precision_fast' else 'E2',
                 frozen_utc=(gate.time_value(self.policy['last_criteria_update_utc'])+timedelta(seconds=1)).isoformat())
        sha = 'a'*64
        for role, source in b['sources'].items():
            source.update(git_sha='b'*40, files_sha256={'synthetic_source.py': sha},
                          execution_mode=mode if role == 'B' else 'io_preserving_v1',
                          execution_mode_flags_sha256=sha)
        b['environment'] = {k: 'synthetic-'+k for k in b['environment']}
        b['environment'].update(world_size=8, TF32=False, determinism=False, model_dtype='FP32', master_dtype='FP32', optimizer_dtype='FP32')
        inv = {c: {k: v['elements'] for k,v in gate.flatten_category(cat)}
               for c,cat in comparison(self.policy)['comparisons']['A_A'].items()}
        for case in b['cases'].values():
            case['science'] = {k: sha for k in case['science']}
            case['science'].update(training_source_sha=self.policy['scientific_training_source'], training_seed=42,
                                    global_batch=32, FM_repeat=8, schedule_horizon=100000, warmup=1000)
            case['origin'] = {'identity': 'synthetic-origin', 'completed': 25000, 'COMPLETE_sha256': sha, 'payload_manifest_sha256': sha}
            for key in ('optimizer_groups_sha256','parameter_inventory_sha256','gradient_inventory_sha256'):
                case[key] = sha
            case['category_inventory'] = copy.deepcopy(inv)
            case['forward_module_inventory'] = ['Bt','q_final','loss']
            cfg = {'synthetic': 'no real model or dataset'}
            case['science']['config_sha256'] = gate.hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
            case['optimizer_groups_sha256'] = write(self.root/'synthetic_optimizer_groups.json',[{'synthetic':True}])['sha256']
        for key in self.policy['dev']['protocol_bindings']:
            b['dev_protocol'][key] = sha
        b['producer_sha256'] = {k: sha for k in b['producer_sha256']}
        b['E3_declaration'] = self.policy['tiers']['E3']['must_declare'] if mode == 'precision_fast' else None
        return b

    def freeze_bindings(self):
        write(self.binding_path, self.bindings)
        self.binding_hash = gate.digest(self.binding_path)

    def receipt(self, kind='native', group='G1', stage='mid', repeat=0, updates=1, b_error=1.1e-5):
        case = self.bindings['cases'][group+'/'+stage]
        suffix = f'{kind}_{group}_{stage}_{repeat}_{updates}'
        payload = {}
        if kind != 'dev':
            comp = comparison(self.policy, native=kind != 'forward', b_error=b_error)
            comp['provenance'] = {role: {'identity': 'synthetic-'+group+'-'+stage+'-'+role,
                'source': self.bindings['sources'][role]['git_sha'],
                'execution_mode': self.bindings['sources'][role]['execution_mode'],
                'optimizer_groups_sha256': case['optimizer_groups_sha256'], 'ranks': 8,
                'completed': case['origin']['completed']+updates, 'payload_hashes_verified': True,
                'COMPLETE_sha256': 'c'*64} for role in gate.ROLES}
            comp['provenance']['origin'] = dict(identity=case['origin']['identity'], COMPLETE_sha256=case['origin']['COMPLETE_sha256'],
                completed=case['origin']['completed'], payload_hashes_verified=True)
            if kind == 'forward':
                comp.update(gradient_inventory_sha256=case['gradient_inventory_sha256'], full_parameter_rank_coverage=True)
                payload['forward_exact'] = {k: True for k in ['q0','Bt','q_final','loss','RNG','BN_float','BN_counter','buffer_inventory','gradient_inventory']}
                payload['original_strict'] = write(self.root/(suffix+'_old_strict.json'), {'passed': False, 'gradient_strict_passed': False,
                    'q0_qfinal_bitwise_passed': self.bindings['candidate_mode'] != 'engineering_timevec'})
                if self.bindings['candidate_mode'] == 'engineering_timevec':
                    fs = {'schema': 'execution_acceptance_forward_stats_v1', 'full_tensor_rank_coverage': True,
                          'modules': {}, 'physical_q_final': {'unit': 'm','full_tensor_rank_coverage': True,
                              'conversion_source_sha256': case['science']['physical_conversion_sha256']}}
                    for name in case['forward_module_inventory']:
                        fs['modules'][name] = {p: dict(elements=4,nonfinite_elements=0,atol=1e-6,rtol=1e-4,
                            relative_L2=1e-7,max_absolute=1e-7,violating_elements=0,zero_reference_norm=False) for p in ('A_B','AA_B')}
                    for p in ('A_B','AA_B'):
                        fs['physical_q_final'][p] = {'max_position_error_m': 1e-5}
                    payload['forward_comparison'] = write(self.root/(suffix+'_forward.json'), fs)
                    for k in ('Bt','q_final','loss'):
                        payload['forward_exact'][k] = False  # New numeric rule, OLD strict still fails.
            else:
                for role in gate.ROLES:
                    run = self.root/(group+'-'+stage+'-'+role)
                    folder = run/'checkpoints'/f'milestone_{case["origin"]["completed"]+updates:07d}'
                    folder.mkdir(parents=True, exist_ok=True)
                    p = comp['provenance'][role]
                    source = self.bindings['sources'][role]
                    identity = {'identity':p['identity'],'training_source_sha':source['git_sha'],
                        'training_source_tree_sha256':gate.hashlib.sha256(json.dumps(source['files_sha256'],sort_keys=True).encode()).hexdigest(),
                        'config':{'synthetic':'no real model or dataset'},
                        **{k:case['science'][k] for k in ('current_and_GT_identity','DINO_identity','training_seed','FM_repeat','schedule_horizon')},
                        'effective_global_batch':32, 'schedule_warmup':case['science']['warmup'],
                        'engineering_continue_origin':{'identity':case['origin']['identity'],'completed':case['origin']['completed'],
                            'complete_sha256':case['origin']['COMPLETE_sha256']}}
                    p['identity_sha256'] = write(run/'identity.json',identity)['sha256']
                    p['COMPLETE_sha256'] = write(folder/'COMPLETE.json',{'identity':p['identity'],'completed':p['completed']})['sha256']
                    write(run/'optimizer_groups.json',[{'synthetic':True}])
                    p['path']=str(folder)
            payload['comparison'] = write(self.root/(suffix+'_comparison.json'), comp)
            if kind == 'recovery':
                boundary = {'registration_sha256': self.registration_hash, 'bindings_sha256': self.binding_hash,
                            'native_payload_hashes_verified': True, 'completed': case['origin']['completed']+updates,
                            **{k: True for k in ('model_FP32','master_FP32','Adam','BN_float','RNG','discrete_state')}}
                payload.update(load_boundary_exact=True, load_boundary_receipt=write(self.root/(suffix+'_boundary.json'), boundary))
        receipt = {'schema': 'execution_acceptance_receipt_v1', 'registration_sha256': self.registration_hash,
                   'bindings_sha256': self.binding_hash, 'started_utc': (gate.time_value(self.bindings['frozen_utc'])+timedelta(seconds=1)).isoformat(),
                   'candidate_mode': self.bindings['candidate_mode'], 'tier': self.bindings['tier'], 'kind': kind,
                   'group': group, 'stage': stage, 'repeat': repeat, 'updates': updates,
                   'producer_sha256': self.bindings['producer_sha256'][kind], 'roles': gate.expected_roles(self.bindings,case),
                   'exact_checks': {k: True for k in self.policy['E1']['exact_checks']}, 'payload': payload}
        if self.bindings['tier'] == 'E3':
            receipt.update(E3_declaration=self.policy['tiers']['E3']['must_declare'], native_precision_exercised=True)
        path = self.root/(suffix+'_receipt.json')
        write(path, receipt)
        self.receipts[(kind,group,stage,repeat,updates)] = path
        return path

    def mutate(self, path, fn):
        data = gate.load(path)
        fn(data)
        write(path,data)

    def check(self, expected='pending', paths=None):
        report = gate.evaluate(self.registration,self.binding_path,list(self.receipts.values()) if paths is None else paths)
        self.assertEqual(report['advisory_decision'],expected,report['failures'])
        self.assertFalse(report['passed'])
        self.assertFalse(report['formal_authorized'])
        self.assertFalse(report['trainer_consumable'])
        self.assertIsNone(report['accepted_mode'])
        self.controls.append({'test':self.id().split('.')[-1], 'decision':report['advisory_decision'],
                              'failures':report['failures'], 'missing_or_inconclusive':len(report['pending'])})
        return report

    def test_absent_evidence_and_template_stay_pending(self):
        r=gate.evaluate(self.registration)
        self.assertEqual(r['advisory_decision'],'pending')
        write(self.binding_path,gate.binding_template(self.policy,self.registration_hash))
        self.check()

    def test_old_strict_failures_stay_false_with_new_numeric_candidate(self):
        self.receipt()
        r=self.check()
        self.assertFalse(r['old_strict_results_preserved'][0]['original_strict']['A_B']['passed'])

    def test_no_recompute_always_rejects(self):
        self.bindings['candidate_mode']='loss_preserving_v1'
        self.freeze_bindings()
        self.check('not_supported')

    def test_wrong_environment_source_data_or_ancestry_rejects(self):
        for part,key in [('environment','torch'),('source','git_sha'),('science','batch_order_sha256'),('origin','COMPLETE_sha256')]:
            with self.subTest(part=part):
                p=self.receipt()
                self.mutate(p,lambda r:r['roles']['B'][part].update({key:'wrong'}))
                self.check('not_supported')

    def test_candidate_started_before_freeze_rejects(self):
        p=self.receipt()
        self.mutate(p,lambda r:r.update(started_utc=self.policy['created_utc']))
        self.check('not_supported')

    def test_native_metadata_recheck_catches_false_header_source_claim(self):
        p=self.receipt();d=gate.load(p);cp=Path(d['payload']['comparison']['path']);comp=gate.load(cp)
        run=Path(comp['provenance']['B']['path']).parent.parent
        identity_path=run/'identity.json'
        self.mutate(identity_path,lambda i:i.update(training_source_tree_sha256='0'*64))
        # All enclosing hashes are recomputed: a valid checksum cannot make a
        # different native source match the fixed prospective source binding.
        comp['provenance']['B']['identity_sha256']=gate.digest(identity_path)
        write(cp,comp);d['payload']['comparison']['sha256']=gate.digest(cp);write(p,d)
        r=self.check('not_supported')
        self.assertTrue(any('native complete source-file inventory hash' in f for f in r['failures']))

    def test_modified_payload_hash_and_unverified_native_reject(self):
        p=self.receipt()
        comparison_path=Path(gate.load(p)['payload']['comparison']['path'])
        self.mutate(comparison_path,lambda r:r.update(integrity_verified=False))
        self.check('not_supported')  # Broken file hash first.
        r=gate.load(p);r['payload']['comparison']['sha256']=gate.digest(comparison_path);write(p,r)
        self.check('not_supported')  # Correct hash still cannot hide skipped integrity.

    def test_fresh_AA_envelope_and_absolute_ceilings(self):
        self.receipt(b_error=1e-4)
        r=self.check('not_supported');self.assertTrue(any('AA envelope' in f for f in r['failures']))
        self.receipt(b_error=.02)
        r=self.check('not_supported');self.assertTrue(any('absolute ceiling' in f for f in r['failures']))

    def test_zero_update_requires_exact_values_not_epsilon(self):
        comp=comparison(self.policy)
        for c in comp['comparisons']['A_B'].values():
            for _,row in gate.flatten_category(c):row['origin_relative_updates']['update_norms']['A']=0.
        with self.assertRaisesRegex(gate.InvalidEvidence,'Zero update'):
            gate.numerical_checks(self.policy,'E2',comp,native=True,inventory=self.bindings['cases']['G1/mid']['category_inventory'])

    def test_bn_has_no_optimizer_group_and_inventory_omission_rejects(self):
        p=self.receipt();self.check()
        comp_path=Path(gate.load(p)['payload']['comparison']['path'])
        d=gate.load(comp_path)
        for pair in d['comparisons']:del d['comparisons'][pair]['BN_float']
        write(comp_path,d)
        r=gate.load(p);r['payload']['comparison']['sha256']=gate.digest(comp_path);write(p,r)
        self.check('not_supported')

    def test_recovery_boundary_must_be_bitwise(self):
        p=self.receipt('recovery',updates=32);self.check()
        self.mutate(p,lambda r:r['payload'].update(load_boundary_exact=False))
        self.check('not_supported')

    def test_phase_receipts_cannot_switch_native_chain_identity(self):
        self.receipt(updates=1)
        p=self.receipt(updates=32)
        d=gate.load(p);cp=Path(d['payload']['comparison']['path'])
        self.mutate(cp,lambda r:r['provenance']['B'].update(identity='different-chain'))
        d['payload']['comparison']['sha256']=gate.digest(cp);write(p,d)
        self.check('not_supported')

    def test_timevec_uses_fixed_tolerance_keeps_original_bitwise_failure(self):
        self.bindings=self.make_bindings('engineering_timevec');self.freeze_bindings()
        p=self.receipt('forward',updates=0)
        r=self.check()
        self.assertFalse(r['old_strict_results_preserved'][0]['original']['q0_qfinal_bitwise_passed'])
        d=gate.load(p);fs=Path(d['payload']['forward_comparison']['path'])
        self.mutate(fs,lambda x:x['physical_q_final']['A_B'].update(max_position_error_m=1.01e-4))
        d['payload']['forward_comparison']['sha256']=gate.digest(fs);write(p,d)
        self.check('not_supported')

    def test_timevec_cannot_raise_old_module_tolerance(self):
        self.bindings=self.make_bindings('engineering_timevec');self.freeze_bindings()
        p=self.receipt('forward',updates=0);d=gate.load(p);fs=Path(d['payload']['forward_comparison']['path'])
        self.mutate(fs,lambda x:x['modules']['Bt']['A_B'].update(atol=1e-5))
        d['payload']['forward_comparison']['sha256']=gate.digest(fs);write(p,d)
        self.check('not_supported')

    def test_timevec_normalized_L2_is_independent_of_elementwise_gate(self):
        self.bindings=self.make_bindings('engineering_timevec');self.freeze_bindings()
        p=self.receipt('forward',updates=0);d=gate.load(p);fs=Path(d['payload']['forward_comparison']['path'])
        self.mutate(fs,lambda x:x['modules']['Bt']['A_B'].update(relative_L2=1.01e-4))
        d['payload']['forward_comparison']['sha256']=gate.digest(fs);write(p,d)
        self.check('not_supported')

    def test_P0_does_not_inherit_timevec_forward_allowance(self):
        p=self.receipt('forward',updates=0)
        self.mutate(p,lambda r:r['payload']['forward_exact'].update(Bt=False))
        self.check('not_supported')

    def test_P0_exact_claim_cannot_hide_original_trajectory_strict_failure(self):
        p=self.receipt('forward',updates=0);d=gate.load(p);sp=Path(d['payload']['original_strict']['path'])
        self.mutate(sp,lambda s:s.update(q0_qfinal_bitwise_passed=False))
        d['payload']['original_strict']['sha256']=gate.digest(sp);write(p,d)
        self.check('not_supported')

    def test_E3_declares_float_BN_difference_and_cannot_be_E2(self):
        self.bindings=self.make_bindings('precision_fast');self.freeze_bindings()
        p=self.receipt();self.check()
        self.mutate(p,lambda r:r.update(tier='E2'))
        self.check('not_supported')
        p=self.receipt();self.mutate(p,lambda r:r.update(native_precision_exercised=False))
        self.check('not_supported')

    def make_dev_payload(self, provenance, delta=0., partial=False):
        index=[{'token':f'synthetic-{i:04}', 'log':f'log-{i%16:02}'} for i in range(1696)]
        index_ref=write(self.root/'synthetic_index.json',index)
        self.bindings['dev_protocol']['metric_index_sha256']=index_ref['sha256']
        self.freeze_bindings()
        payload={'index':index_ref,'protocol':self.bindings['dev_protocol'],'banks':{}}
        fields=['token','log','status',*self.policy['dev']['metrics'].values()]
        for role in gate.ROLES:
            payload['banks'][role]={}
            for variant in ('q0','q_final'):
                path=self.root/f'{role}_{variant}.csv'
                with path.open('w') as stream:
                    writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
                    for item in index[:1] if partial and role=='B' else index:
                        vals={k:.9+(delta if role=='B' else 0) for k in self.policy['dev']['metrics'].values()}
                        writer.writerow({**item,'status':'ok',**vals})
                summary=write(self.root/f'{role}_{variant}_summary.json',{'scenes':1696,'failed':0,'valid':True,
                         'metrics':{k:.9+(delta if role=='B' else 0) for k in self.policy['dev']['metrics'].values()}})
                payload['banks'][role][variant]={'csv':{'path':str(path),'sha256':gate.digest(path)},'summary':summary,'checkpoint':provenance[role]}
        return payload

    def test_full_independent_population_and_mean_check(self):
        provenance={r:{'identity':'synthetic-'+r} for r in gate.ROLES}
        payload=self.make_dev_payload(provenance)
        failures,pending,_=gate.dev_checks(self.policy,payload,self.root,self.bindings,provenance)
        self.assertEqual((failures,pending),([],[]))
        payload=self.make_dev_payload(provenance,partial=True)
        with self.assertRaisesRegex(gate.InvalidEvidence,'Partial/duplicate'):
            gate.dev_checks(self.policy,payload,self.root,self.bindings,provenance)
        payload=self.make_dev_payload(provenance)
        summary=Path(payload['banks']['B']['q0']['summary']['path'])
        d=gate.load(summary);d['metrics']['score']=.8
        payload['banks']['B']['q0']['summary']=write(summary,d)
        with self.assertRaisesRegex(gate.InvalidEvidence,'mean mismatch'):
            gate.dev_checks(self.policy,payload,self.root,self.bindings,provenance)

    def test_dev_regression_and_wrong_index_membership_reject(self):
        provenance={r:{'identity':'synthetic-'+r} for r in gate.ROLES}
        payload=self.make_dev_payload(provenance,delta=-.002)
        failures,_,_=gate.dev_checks(self.policy,payload,self.root,self.bindings,provenance)
        self.assertTrue(any('PDMS' in f for f in failures))
        payload=self.make_dev_payload(provenance)
        cp=Path(payload['banks']['B']['q0']['csv']['path']);raw=cp.read_text()
        cp.write_text(raw.replace('synthetic-0000','different-token',1))
        payload['banks']['B']['q0']['csv']['sha256']=gate.digest(cp)
        with self.assertRaisesRegex(gate.InvalidEvidence,'Foreign scene/log'):
            gate.dev_checks(self.policy,payload,self.root,self.bindings,provenance)

    def test_every_actual_receipt_required_even_if_one_step_passes(self):
        self.receipt();report=self.check()
        self.assertTrue(any("'dev'" in p for p in report['pending']))
        self.assertTrue(any('500' in p for p in report['pending']))

    def test_complete_synthetic_chain_only_qualifies_for_review_never_trainer(self):
        # Bind the independent population before producing eligible receipts.
        self.make_dev_payload({r:{'identity':'synthetic-'+r} for r in gate.ROLES})
        for kind,g,s,r,n in sorted(gate.required_keys(self.policy)):
            if kind!='dev':
                self.receipt(kind,g,s,r,n)
        for group in ('G1','G3'):
            native=self.receipts['native',group,'mid',0,500]
            comparison=gate.load(gate.load(native)['payload']['comparison']['path'])
            payload=self.make_dev_payload(comparison['provenance'])
            p=self.receipt('dev',group,'mid',0,500)
            self.mutate(p,lambda d:d.update(payload=payload))
        report=self.check('evidence_complete_for_review')
        self.assertEqual(len(report['receipts']),len(gate.required_keys(self.policy)))
        self.assertTrue(all(not p['passed'] for record in report['old_strict_results_preserved'] if 'original_strict' in record for p in record['original_strict'].values()))
        # A missing registered +500 receipt removes eligibility, not the queue.
        paths=list(self.receipts.values());paths.remove(self.receipts['native','G3','mid',0,500])
        self.check('not_supported',paths)  # Dev cannot claim the absent native ancestry.


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    started=time.monotonic()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(AcceptanceTests))
    report={'schema':'execution_acceptance_CPU_validation_v1','passed':result.wasSuccessful(),
            'tests_run':result.testsRun,'controls':AcceptanceTests.controls,'wall_seconds':time.monotonic()-started,
            'source_sha256':{'script':gate.digest(gate.__file__),'tests':gate.digest(__file__),'registration':gate.digest(gate.REGISTRATION)},
            'scope':'CPU synthetic receipt validation only. No actual model/native/dev acceptance; no GPU jobs or trainer/guard edits.',
            'GPU_jobs_started':0,'actual_candidate_qualified':False,
            'errors':[text for _,text in result.errors], 'failures':[text for _,text in result.failures]}
    gate.write_new(args.output,report)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__=='__main__':
    main()
