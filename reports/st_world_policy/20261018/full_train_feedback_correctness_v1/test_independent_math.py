"""Independent CPU-only math/transaction fixtures; never open real data assets."""
import copy
import importlib.util
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

SOURCE_ROOT = Path(os.environ['FULL_FEEDBACK_REVIEW_SOURCE_ROOT'])
spec = importlib.util.spec_from_file_location('full_feedback_author_fixture', SOURCE_ROOT/'tests/st_world/test_full_feedback_support.py')
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)
m = fixture.m


class IndependentChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
            os.environ[name] = '1'
        m.load_numeric()

    def test_pooled_quantile_is_not_mean_of_shard_quantiles(self):
        left, right = [0.]*100, [100.]
        self.assertEqual(m.scalar_summary(left+right)['quantiles']['0.5'], 0.)
        self.assertEqual((m.scalar_summary(left)['quantiles']['0.5']+m.scalar_summary(right)['quantiles']['0.5'])/2, 50.)
        self.assertEqual(m.scalar_summary([])['quantiles']['0.99'], None)
        self.assertEqual(m.scalar_summary([-4.,0.,8.,12.])['quantiles']['0.25'], -1.)

    def test_nonzero_group_keeps_zero_candidate_entries(self):
        item = {'scene_presentations':2,'centered_A_float32':[[0.,0.,0.,0.],[-1.,0.,0.,1.]],
                'within_group_A_range_float32':[0.,2.]}
        r = m.amplitude_statistics([item])
        self.assertEqual(r['all_scene_presentations']['abs_A_per_candidate']['count'],8)
        self.assertEqual(r['nonzero_advantage_scene_presentations']['abs_A_per_candidate']['count'],4)
        self.assertEqual(r['nonzero_advantage_scene_presentations']['abs_A_per_candidate']['mean'],.5)
        self.assertEqual(r['nonzero_advantage_scene_presentations']['abs_A_per_candidate']['quantiles']['0.5'],.5)

    def test_variable_safety_gate_changes_all_gated_terms(self):
        ep=m.np.full((1,4),.8)
        nc=m.np.array([[0.,.5,1.,1.]])
        ones=m.np.ones((1,4))
        metric={'no_at_fault_collisions':nc,'drivable_area_compliance':ones,'ego_progress':ep,
                'time_to_collision_within_bound':ones,'comfort':ones,
                'score':nc*(5*ep+7)/12}
        A, terms, d=m.decompose(metric)
        self.assertTrue(m.np.any(A))
        self.assertEqual(m.np.ptp(ep),0.)
        self.assertTrue(all(m.np.ptp(terms[:,:,j])>0 for j in range(3)))
        self.assertAlmostEqual(sum(d['signed_covariance_fractions'].values()),1.,places=12)
        self.assertGreater(d['signed_covariance_numerators']['gated_TTC'],0.)
        # Thus gated-TTC algebra is not, generally, an independently varying TTC metric.

    def test_all_zero_algebra_and_empty_events_have_undefined_denominators(self):
        records=[fixture.event_record(7,0) for _ in range(4)]
        profiles=[m.process_profile(r) for r in records]
        metrics={k:m.np.array([r['metrics'][k] for r in records]).reshape(1,4) for k in m.METRICS}
        _,_,d=m.decompose(metrics)
        self.assertEqual(d['advantage_squared_sum'],0.)
        self.assertIsNone(d['signed_covariance_fractions'])
        for row in m.tie_counts(records,profiles).values():
            self.assertEqual(row['eligible_pairs'],6)
            self.assertEqual(row['nonempty_event_pairs'],0)
            self.assertIsNone(row['conditional_time_difference_fraction_given_nonempty'])

    def test_repeated_scene_exposure_and_weighted_covariance(self):
        with tempfile.TemporaryDirectory() as temp:
            f=fixture.Fixture(Path(temp),sizes=(32,24))
            budget=m.Budget(Path(temp)/'b','test','unit',time.time()+60,8<<20,2<<20,False)
            items=[i for shard in f.plan['shards'] for i in shard['items']]
            analyses=[m.verify_and_analyze(f.plan,i,budget) for i in items]
            total=m.aggregate(analyses)
            self.assertEqual(total['unique_scenes'],32)
            self.assertEqual(total['unique_logs'],16)
            self.assertEqual(total['counts']['scene_presentations'],56)
            self.assertEqual(sum(total['scene_presentations_by_log'].values()),56)
            self.assertEqual(total['counts']['candidate_pair_presentations'],336)
            self.assertEqual(total['scene_exposure_distribution']['histogram_scene_count_by_presentations'],{1:8,2:24})
            for label in analyses[0]['support_scene_slots']:
                selected=[a['scene_log_pairs'][i] for a in analyses for i in a['support_scene_slots'][label]]
                self.assertEqual(total['support_unique_population'][label],
                    {'scene_presentations':len(selected),'unique_scenes':len({s for s,_ in selected}),
                     'unique_logs':len({log for _,log in selected})})
            V=sum(a['reward_value_decomposition']['advantage_squared_sum'] for a in analyses)
            for key,value in total['reward_value_decomposition']['signed_covariance_fractions'].items():
                expected=sum(a['reward_value_decomposition']['signed_covariance_numerators'][key] for a in analyses)/V
                self.assertAlmostEqual(value,expected,places=14)
            changed=copy.deepcopy(analyses)
            changed[1]['scene_log_pairs'][0][1]='different_log'
            with self.assertRaisesRegex(ValueError,'multiple source logs'): m.aggregate(changed)

    def test_interruption_after_committed_ledger_reuses_exact_summary(self):
        with tempfile.TemporaryDirectory() as temp:
            f=fixture.Fixture(Path(temp),sizes=(32,))
            shard=next(s for s in f.plan['shards'] if s['items'])
            deadline=time.time()+60
            cpu=min(os.sched_getaffinity(0)); old_affinity=os.sched_getaffinity(0)
            publish=m.Budget.publish
            fired=[]
            def inject(budget,path,raw,**kw):
                publish(budget,path,raw,**kw)
                if Path(path).name==shard['items'][0]['ledger']+'.json' and not fired:
                    fired.append(True)
                    raise RuntimeError('fixture_after_ledger_publish')
            try:
                with patch.object(m.Budget,'publish',inject):
                    with self.assertRaisesRegex(RuntimeError,'fixture_after_ledger_publish'):
                        m.worker(f.path,shard['shard'],cpu,deadline)
                folder=f.output/('shard_%02d'%shard['shard'])
                summary=folder/(shard['items'][0]['ledger']+'.json')
                before=summary.read_bytes()
                self.assertTrue((folder/'ERROR_0001.json').exists())
                self.assertFalse((folder/'SHARD_COMPLETE.json').exists())
                with patch.object(m,'verify_and_analyze',side_effect=AssertionError('committed payload must not rerun')):
                    done=m.worker(f.path,shard['shard'],cpu,deadline,resume=True)
                self.assertEqual(done['status'],'SHARD_COMPLETE')
                self.assertEqual(summary.read_bytes(),before)
                self.assertTrue((folder/'ERROR_0001.json').exists())
            finally:
                os.sched_setaffinity(0,old_affinity)


if __name__=='__main__': unittest.main()
