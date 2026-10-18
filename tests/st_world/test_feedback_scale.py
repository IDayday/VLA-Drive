"""Synthetic compact JSON only; no real full-feedback receipts are read."""
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('feedback_scale', ROOT / 'tools/st_world_policy/analyze_feedback_scale.py')
m = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(m)


def write(path, obj, sealed=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = m.base.sealed(obj) if sealed else m.base.encoded(obj)
    path.write_bytes(raw)
    return raw


def analysis(version=0, groups=None, event=False, discrim=False):
    groups = groups if groups is not None else [[-.25, .25, 0., 0.], [0., 0., 0., 0.]]
    active = [i for i, g in enumerate(groups) if any(g)]
    slots = {'C1': active, 'D': [0] if discrim else [], 'C': [], 'T': [], 'EP_only': [] if discrim else active}
    return {'policy_version': version, 'scene_presentations': len(groups), 'candidate_presentations': 4*len(groups),
            'centered_A_float32': groups, 'scene_log_pairs': [['scene-%d' % i, 'log-%d' % (i//2)] for i in range(len(groups))],
            'support_scene_slots': slots, 'nonzero_advantage_scene_presentations': {k:len(slots[k]) for k in ('C1','D','C','T')},
            'EP_only_scene_presentations': len(slots['EP_only']), 'event_candidate_counts_D_C_T': [int(event),0,0],
            'reward_value_decomposition': {'max_absolute_rounding_residual': 1e-8}, 'canonical_score_reconstruction_max_error': 1e-16,
            'input_pins': {'COMPLETE': 'c'*64, 'RESULTS': 'd'*64}}


class PrimaryFixture:
    def __init__(self, root):
        self.root = root / 'primary'; self.root.mkdir()
        self.plan_path = root / 'plan.json'; self.analyses = [analysis(i, event=i==2, discrim=i==3) for i in range(8)]
        self.analyses[1] = analysis(1, [[0.]*4]*2)
        self.items = [{'ledger': m.identity({'v':i}), 'attempt': 'attempt_00', 'payload_bytes': 1,
                       'complete_sha256': 'c'*64, 'payload_sha256_declared': 'd'*64} for i in range(8)]
        self.expected = {'ledgers': 8, 'scene_presentations': 16, 'candidate_presentations': 64}
        plan = {'schema': m.base.SCHEMA, 'run': str(root/'original-run'), 'output_root': str(self.root),
                'source_sha256': m.PRIMARY_SOURCE_SHA, 'expected': self.expected, 'plan_metadata_read_bytes': 100,
                'shards': [{'shard':i, 'items':[self.items[i]]} for i in range(8)]}
        self.plan_id = m.identity(plan); self.plan = dict(plan, plan_id=self.plan_id); write(self.plan_path, self.plan)
        (self.root/'MERGE.lock').touch(); write(self.root/'EXECUTION.json', {'plan_id':self.plan_id,'deadline_unix':1.})
        hashes = []
        for i, a in enumerate(self.analyses):
            folder = self.root / ('shard_%02d' % i); folder.mkdir(); (folder/'WORKER.lock').touch()
            raw = write(folder/(self.items[i]['ledger']+'.json'), {'plan_id':self.plan_id,'item':self.items[i],'analysis':a}, True)
            summary = {'status':'SHARD_COMPLETE','plan_id':self.plan_id,'shard':i,
                       'receipts':[{'ledger':self.items[i]['ledger'],'sha256':m.digest(raw)}],
                       'budget':{'logical_read_bytes':100,'output_reserved_bytes':len(raw)}}
            complete_raw = write(folder/'SHARD_COMPLETE.json', summary, True); hashes.append(m.digest(complete_raw))
            write(folder/'BUDGET.json', {'plan_id':self.plan_id,'label':'shard_%02d'%i,'deadline_unix':1.,
                                        'logical_read_bytes':100,'output_reserved_bytes':len(raw)+len(complete_raw)})
        write(self.root/'merge/BUDGET.json', {'plan_id':self.plan_id,'label':'merge','deadline_unix':1.,'logical_read_bytes':100})
        self.full = {'status':'COMPLETE','plan_id':self.plan_id,'expected_counts_from_run_RESULT':self.expected,
                     'counts':{**self.expected,'nonzero_advantage_scene_presentations':{'C1':7}},
                     'policy_versions':list(range(8)), 'shard_receipt_sha256':hashes}
        write(self.root/'RESULT.json', self.full, True)


class FeedbackScaleTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.affinity = os.sched_getaffinity(0); self.cpu = min(self.affinity)
    def tearDown(self):
        os.sched_setaffinity(0, self.affinity); self.temp.cleanup()

    def test_two_pass_correction_one_and_no_second_center(self):
        values = [-.25,.25,0.,0.,0.,0.,0.,0.]
        d = m.describe_values(values)
        self.assertEqual(d['N'],8); self.assertEqual(d['energy_A'],.125)
        self.assertEqual(d['std_correction1'], math.sqrt(.125/7))
        self.assertEqual(d['gain'],1/(math.sqrt(.125/7)+1e-10))
        original = [0.,0.,0.,-2.**-24,0.,0.,0.,0.]
        before = copy.deepcopy(original); d = m.describe_values(original)
        self.assertEqual(d['mean_A'],-2.**-27); self.assertEqual(original,before)
        self.assertEqual(original[4:], [0.]*4)
        # A different mean genuinely requires centered M2, not sum(A^2)/(N-1).
        self.assertNotEqual(d['M2'],d['energy_A'])

    def test_zero_small_std_and_invalid_numeric_boundaries(self):
        d = m.describe_update(analysis(groups=[[0.]*4]*24))
        self.assertEqual(d['N'],96); self.assertEqual(d['zero_groups'],24)
        self.assertEqual(d['std_correction1'],0); self.assertIsNone(d['gain']); self.assertIsNone(d['rounding_bound_fraction'])
        self.assertNotIn('nonzero_A_but_std_zero',d['flags'])
        self.assertIn('N_below_two',m.describe_values([])['flags'])
        self.assertIn('N_below_two',m.describe_values([0.])['flags'])
        self.assertIn('nonfinite_input_or_statistics',m.describe_values([float('nan'),0.])['flags'])
        self.assertIn('nonzero_A_but_std_zero',m.describe_values([.25]*4)['flags'])
        self.assertIn('constant_nonzero_group',m.describe_update(analysis(groups=[[.25]*4]))['flags'])
        self.assertIn('positive_std_le_epsilon',m.describe_values([-2.**-40,2.**-40])['flags'])
        with self.assertRaisesRegex(ValueError,'exact stored float32'): m.describe_values([.1,-.1])

    def test_strata_overlap_rounding_bound_and_unique_exposures(self):
        safe = m.describe_update(analysis(0)); event = m.describe_update(analysis(1,event=True)); metric = m.describe_update(analysis(2,event=True,discrim=True))
        self.assertTrue(safe['_strata']['safe_EP_only_update']); self.assertFalse(event['_strata']['safe_EP_only_update'])
        self.assertTrue(metric['_strata']['DCT_event_present_update']); self.assertTrue(metric['_strata']['DCT_metric_discriminative_update'])
        a = analysis(3); a['reward_value_decomposition']['max_absolute_rounding_residual']=1.
        unresolved = m.describe_update(a)
        self.assertEqual(unresolved['rounding_bound_fraction'],8/.125)
        self.assertIn('rounding_bound_can_cover_all_energy',unresolved['flags'])
        result=m.summarize([safe,event,metric,unresolved]); allpop=result['strata']['all']
        self.assertEqual(allpop['sum_energy_A'], .5)
        self.assertEqual(allpop['ratio_of_rounding_bound_sum_to_energy_sum'], allpop['sum_rounding_energy_bound']/.5)
        self.assertEqual(allpop['scene_presentations'],8); self.assertEqual(allpop['unique_scenes'],2)
        self.assertEqual(allpop['nonzero_scene_presentations'],4); self.assertEqual(allpop['nonzero_unique_scenes'],1)
        self.assertEqual(result['strata']['DCT_metric_discriminative_update']['updates'],1)

    def test_exact_update_quantiles_and_fixed_windows(self):
        rows=[m.describe_update(analysis(v)) for v in (0,1023,1024,2047,2048,3071,3072,4095)]
        for row,x in zip(rows,[0.,0.,0.,0.,0.,0.,0.,8.]): row['std_correction1']=x
        result=m.summarize(rows)
        self.assertEqual(result['strata']['all']['distributions']['std_correction1']['quantiles']['0.5'],0.)
        self.assertEqual([p['all']['updates'] for p in result['fixed_policy_version_windows'].values()],[2,2,2,2])
        self.assertIsNone(result['strata']['DCT_event_present_update']['distributions']['gain']['mean'])

    def test_complete_compact_only_join_and_independent_identity(self):
        f=PrimaryFixture(self.root); oldexec=(f.root/'EXECUTION.json').read_bytes(); before={p:p.read_bytes() for p in f.root.rglob('BUDGET.json')}
        real_open=Path.open
        def guarded(path,*args,**kw):
            self.assertNotEqual(path.suffix,'.pt'); self.assertNotIn('original-run',path.parts)
            return real_open(path,*args,**kw)
        with mock.patch.object(Path,'open',guarded): result=m.execute(f.plan_path,self.root/'derived',self.cpu,production=False)
        self.assertEqual(result['strata']['all']['updates'],8)
        self.assertEqual(result['evidence']['primary_actual_logical_read_bytes'],1000)
        self.assertEqual((f.root/'EXECUTION.json').read_bytes(),oldexec)
        self.assertTrue(all(p.read_bytes()==raw for p,raw in before.items()))
        self.assertLess(sum(p.stat().st_size for p in (self.root/'derived').iterdir()),m.OUTPUT_LIMIT)
        with self.assertRaisesRegex(ValueError,'already exists'): m.execute(f.plan_path,self.root/'derived',self.cpu,production=False)
        with self.assertRaisesRegex(ValueError,'overlaps primary'): m.execute(f.plan_path,f.root/'never_create',self.cpu,production=False)
        self.assertFalse((f.root/'never_create').exists())

    def test_active_worker_or_incomplete_primary_blocks_compact_reads(self):
        f=PrimaryFixture(self.root)
        with m.readonly_lock(f.root/'shard_00/WORKER.lock'):
            with self.assertRaises(BlockingIOError): m.execute(f.plan_path,self.root/'active',self.cpu,production=False)
        self.assertTrue((self.root/'active/ERROR.json').exists())
        f.full['status']='INCOMPLETE'; write(f.root/'RESULT.json',f.full,True)
        with self.assertRaisesRegex(ValueError,'fully COMPLETE'): m.execute(f.plan_path,self.root/'incomplete',self.cpu,production=False)
        self.assertFalse((self.root/'incomplete/RESULT.json').exists())

    def test_hash_mismatch_and_original_total_read_cap_fail_closed(self):
        f=PrimaryFixture(self.root); path=f.root/'shard_00'/(f.items[0]['ledger']+'.json')
        path.write_bytes(path.read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError,'hash mismatch'): m.execute(f.plan_path,self.root/'bad_hash',self.cpu,production=False)
        state_path=f.root/'merge/BUDGET.json'; state=json.loads(state_path.read_bytes()); state['logical_read_bytes']=32<<30; write(state_path,state)
        with self.assertRaisesRegex(ValueError,'original 32 GiB'): m.execute(f.plan_path,self.root/'over_total',self.cpu,production=False)

    def test_read_precharge_and_runtime_bounds(self):
        folder=self.root/'budget'; folder.mkdir(); b=m.ReadBudget(folder,'test',time.monotonic()); data=self.root/'data.json'; data.write_bytes(b'12345')
        with mock.patch.object(b,'save',wraps=b.save) as saved:
            for _ in range(66): b.read(data,10)
            self.assertEqual(saved.call_count,1)
        self.assertEqual(b.state['read_operations'],66)
        b.state['precharged_read_bytes']=m.READ_LIMIT-5
        with mock.patch.object(Path,'open',side_effect=AssertionError('must not open')):
            with self.assertRaisesRegex(ValueError,'before read'): b.read(data,10)
        b.start-=61
        with self.assertRaisesRegex(ValueError,'60-second'): b.check()

    def test_strict_help_and_invalid_cli(self):
        script=Path(m.__file__)
        help_result=subprocess.run([sys.executable,str(script),'--help'],capture_output=True,text=True,timeout=5)
        self.assertEqual(help_result.returncode,0); self.assertIn('--primary-plan',help_result.stdout)
        bad=subprocess.run([sys.executable,str(script),'--primary',str(self.root/'never-open')],capture_output=True,text=True,timeout=5)
        self.assertEqual(bad.returncode,2); self.assertFalse((self.root/'never-open').exists())

    def test_4096_synthetic_update_summary_fits_output_budget(self):
        rows=[]
        for v in range(4096):
            x=m.struct.unpack('!f',m.struct.pack('!f',(v+1)*1e-8))[0]
            rows.append(m.describe_update(analysis(v, [[-x,x,0.,0.]]*32, event=v%3==0, discrim=v%5==0)))
        result=m.summarize(rows)
        self.assertEqual(result['strata']['all']['candidate_presentations'],524288)
        self.assertLess(len(m.base.sealed(result))+(16<<10),m.OUTPUT_LIMIT)


if __name__=='__main__':
    unittest.main()
