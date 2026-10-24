"""Finite CPU arithmetic and actual tiny native ActionDiT rehearsal checks."""
import argparse
import copy
import json
from pathlib import Path
import unittest
import numpy as np
import torch
from starVLA.rl.qwen_flow_exploration.preservation.archive import ArchiveConfig, FrozenArchive, trajectory_distance
from starVLA.rl.qwen_flow_exploration.preservation.objective import (ProjectedCostDual, official_costs,
    cost_scalarized_advantage, source_fm_replay_loss, gradient_comparison, calibrate_replay_coefficient)
from starVLA.rl.qwen_flow_exploration.preservation.diagnostics import physical_drift, select_conservative_trial, retention_summary
from starVLA.rl.qwen_flow_exploration.preservation.update import PreservationUpdate


def candidate(i, *, token='s0', reward=.95, y=0., nc=1.):
    trajectory = [[(j+1)*1.5, y, 0.] for j in range(8)]
    return dict(candidate_id=str(i), scene_token=token, log_id='log0', split='TRAIN',
                checkpoint_id='base', scorer_config_hash='canonical-v1.1', raw_reward=reward,
                planned_trajectory=trajectory, proposal_source='base_ODE',
                native_subscores=dict(no_at_fault_collisions=nc, drivable_area_compliance=1., time_to_collision_within_bound=1.))


class ArchiveTests(unittest.TestCase):
    def build(self, records, config=ArchiveConfig()):
        return FrozenArchive.build(records, train_tokens={'s0','s1'}, token_logs={'s0':'log0','s1':'log0'},
                                   config=config, reference_candidate_ids={'s0':'0'})

    def test_fixed_quality_and_diverse_capacity(self):
        records = [candidate(i, y=i*.5, reward=.95-i*.001) for i in range(8)]
        records += [candidate(8, y=4., reward=.5),candidate(9,y=5.,nc=.5)]
        archive = self.build(records)
        self.assertEqual(len(archive.entries['s0']), 4)
        self.assertEqual(archive.entries['s0'][0]['candidate_id'], '0')
        self.assertEqual(archive.provenance['input_candidates'], 10)
        self.assertTrue(all(e['on_policy_probability'] is None for e in archive.entries['s0']))
        clone=FrozenArchive.from_state_dict(json.loads(json.dumps(archive.state_dict())))
        self.assertEqual(archive.digest(),clone.digest())

    def test_dedup_and_angle_wrap(self):
        records=[candidate(0),candidate(1,y=.01),candidate(2,y=.5)]
        archive=self.build(records)
        self.assertEqual(len(archive.entries['s0']),2)
        a=np.asarray(records[0]['planned_trajectory']);b=a.copy();b[:,2]+=2*np.pi
        self.assertLess(trajectory_distance(a,b)[1],1e-12)

    def test_no_good_trajectory_no_success_label(self):
        archive=self.build([candidate(0,reward=0.),candidate(1,reward=.5,nc=0.)])
        self.assertEqual(archive.scenes,[])
        with self.assertRaises(ValueError):archive.sample_scenes(1,generator=torch.Generator().manual_seed(1))

    def test_shared_single_multi_population_and_positive_sampling(self):
        archive=self.build([candidate(0),candidate(1,y=.5),candidate(2,token='s1')])
        g=torch.Generator().manual_seed(12)
        scenes=archive.sample_scenes(64,generator=g)
        single,ids=archive.sample(scenes,generator=g,single_best=True)
        multi,mids=archive.sample(scenes,generator=g)
        self.assertEqual(single.shape,multi.shape)
        self.assertEqual(set(scenes),set(archive.scenes))
        self.assertTrue(set(mids)<=set(['0','1','2']))

    def test_no_leakage_or_false_encoding(self):
        for key,value in [('split','DEV'),('scene_token','test_scene'),('log_id','wrong')]:
            row=candidate(0);row[key]=value
            with self.assertRaises(ValueError):self.build([row])
        row=candidate(0);row['normalized_action']=np.zeros((8,3)).tolist()
        with self.assertRaises(ValueError):self.build([row])

    def test_immutable(self):
        archive=self.build([candidate(0)]);archive.entries['s0'][0]['reward']=1.
        with self.assertRaises(ValueError):archive.assert_fixed()


class CostTests(unittest.TestCase):
    def test_official_fractional_cost_and_dual_sign(self):
        costs=official_costs([candidate(0,nc=.5)['native_subscores']],['no_at_fault_collisions'])
        self.assertEqual(float(costs[0,0]),.5)
        dual=ProjectedCostDual([.1,.2,.3],learning_rate=.1)
        before=dual.snapshot();dual.update([.2,.1,.3],0)
        self.assertGreater(dual.values[0],before[0]);self.assertLess(dual.values[1],before[1]);self.assertEqual(dual.values[2],before[2])
        with self.assertRaises(ValueError):dual.update([.2,.1,.3],0)
        dual.values.zero_();dual.update([0.,0.,0.],1);self.assertTrue((dual.values==0).all())

    def test_dual_resume_and_budget_freeze(self):
        a=ProjectedCostDual([.1,.2]);a.update([.2,.3],0)
        b=ProjectedCostDual([.1,.2]);b.load_state_dict(json.loads(json.dumps(a.state_dict())))
        a.update([.3,.1],1);b.update([.3,.1],1);self.assertEqual(a.state_dict(),b.state_dict())
        a.budgets[0]=.3
        with self.assertRaises(ValueError):a.update([.3,.1],2)

    def test_lambda_not_cancelled_and_equal_groups_exact_zero(self):
        reward=torch.zeros(2,8);cost=torch.zeros(2,8,3,dtype=torch.float64);cost[0,0,0]=1.
        a=cost_scalarized_advantage(reward,cost,torch.ones(3),common_scale=.1)
        b=cost_scalarized_advantage(reward,cost,torch.tensor([2.,1.,1.]),common_scale=.1)
        self.assertLess(b[0,0],a[0,0]);self.assertTrue(torch.equal(a[1],torch.zeros(8)))
        self.assertAlmostEqual(float(a.sum()),0.,places=6)
        self.assertFalse(a.requires_grad)


class DiagnosticsTests(unittest.TestCase):
    def test_drift_is_not_KL_and_bounded_TRAIN_matching(self):
        a=np.zeros((2,8,3));b=a.copy();b[:,:,0]=.5
        drift=physical_drift(a,b);self.assertEqual(drift['mean_ADE_m'],.5)
        trials=[dict(split='TRAIN',learning_rate=x,experiment_id=str(i),drift=dict(mean_ADE_m=d,RMS_heading_rad=0.))
                for i,(x,d) in enumerate([(1e-5,.2),(2.5e-5,.48)])]
        result=select_conservative_trial(trials,dict(mean_ADE_m=.5,RMS_heading_rad=0.))
        self.assertEqual(result['status'],'MATCHED')
        result=select_conservative_trial(trials,dict(mean_ADE_m=1.,RMS_heading_rad=0.))
        self.assertEqual(result['status'],'NOT_MATCHED')
        trials[0]['split']='DEV'
        with self.assertRaises(ValueError):select_conservative_trial(trials,dict(mean_ADE_m=.5,RMS_heading_rad=0.))

    def test_zero_gradient_is_undefined_and_nonnegative_coefficient(self):
        p=torch.tensor(1.,requires_grad=True)
        stats=gradient_comparison(p*0,p*p,[p]);self.assertIsNone(stats['cosine'])
        self.assertIsNone(calibrate_replay_coefficient(0.,1.)['coefficient'])
        with self.assertRaises(ValueError):calibrate_replay_coefficient(1.,1.,target_ratio=-1.)

    def test_net_retention_uses_fixed_population(self):
        stat=retention_summary([1.,.95,0.,.3],[0.,.94,.7,.4],[True,True,False,False],fixed_high_mask=[True,True,False,False])
        self.assertEqual(stat['high_to_zero'],1)
        self.assertAlmostEqual(stat['net_PDMS_points'],100*((-1-.01+.7+.1)/4))
        self.assertEqual(stat['high_mask_source'],'fixed_multi_draw_stratum')


class ActualNativeFMTests(unittest.TestCase):
    def test_actual_tiny_source_FM_and_LoRA_gradient(self):
        from starVLA.model.modules.qwen_sourceflow.vendor.action_dit import ActionDiT
        from starVLA.model.modules.qwen_sourceflow.source_flow_policy import SourceFlowPolicy
        from starVLA.rl.qwen_sourceflow.actor import NativeRecipeActor
        from starVLA.rl.qwen_sourceflow.recipe import expected_recipe
        torch.manual_seed(42);torch.set_num_threads(2)
        head=ActionDiT(action_dim=3,hidden_dim=32,ffn_dim=64,text_dim=40,freq_dim=16,
                       num_heads=2,attn_head_dim=12,num_layers=3,eps=1e-6,use_gradient_checkpointing=False).eval()
        actor=NativeRecipeActor(head,expected_recipe('R_CANONICAL'))
        condition=dict(context=torch.randn(2,4,40),context_mask=torch.ones(2,4,dtype=torch.bool),
                       visual_mask=torch.ones(2,5,dtype=torch.bool),visual_kv=[dict(k=torch.randn(2,5,24),v=torch.randn(2,5,24)) for _ in range(3)])
        target=torch.randn(2,8,3);noise=torch.randn_like(target);time=torch.tensor([250.,700.])
        actual,_=source_fm_replay_loss(actor,condition,target,noise=noise,timestep=time)
        direct=SourceFlowPolicy(actor.action_expert)
        expected,_=direct(condition,target,noise=noise,timestep=time)
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)
        before={n:p.detach().clone() for n,p in actor.named_parameters() if not p.requires_grad}
        opt=torch.optim.AdamW([p for p in actor.parameters() if p.requires_grad],lr=5e-5)
        actual.backward();self.assertTrue(all(p.grad is not None and p.grad.isfinite().all() for p in actor.parameters() if p.requires_grad))
        self.assertTrue(any(p.grad.abs().sum()>0 for n,p in actor.named_parameters() if n.endswith('lora_B')))
        opt.step();opt.zero_grad()
        loss,_=source_fm_replay_loss(actor,condition,target,generator=torch.Generator().manual_seed(987))
        loss.backward();self.assertTrue(any(p.grad.abs().sum()>0 for n,p in actor.named_parameters() if n.endswith('lora_A')))
        self.assertTrue(all(torch.equal(p,before[n]) for n,p in actor.named_parameters() if not p.requires_grad))
        rng=torch.get_rng_state().clone()
        a,_=source_fm_replay_loss(actor,condition,target,generator=torch.Generator().manual_seed(1234))
        torch.testing.assert_close(torch.get_rng_state(),rng,atol=0,rtol=0)
        b,_=source_fm_replay_loss(actor,condition,target,generator=torch.Generator().manual_seed(1234))
        torch.testing.assert_close(a,b,atol=0,rtol=0)


class TinyNativeEnvironment:
    """Qualification wrapper only; production uses the common NativeEnvironment."""
    def __init__(self, head, conditions):
        from starVLA.rl.qwen_sourceflow.actor import NativeRecipeActor
        from starVLA.rl.qwen_sourceflow.recipe import expected_recipe
        cfg=expected_recipe('R_CANONICAL');cfg['grpo']['train']['adv_clip_max']=None
        self.actor=NativeRecipeActor(copy.deepcopy(head),cfg)
        self.device=torch.device('cpu');self.conditions=conditions

    def condition(self, token):return self.conditions[token]

    def policy_chunks(self, rollout, advantages, mask, condition):
        group=len(advantages)
        for start in range(0,group,4):
            end=min(start+4,group);chunk=end-start
            cond={k:([{x:y.expand(chunk,-1,-1).contiguous() for x,y in row.items()} for row in v] if k=='visual_kv' else
                     v.expand(chunk,*v.shape[1:]).contiguous()) for k,v in condition.items()}
            cropped={k:v for k,v in rollout.items() if k not in ('chain','old_logp','rewards')}
            cropped.update(cond_bg=cond,chain=rollout['chain'][start:end],old_logp=rollout['old_logp'][start:end],
                           advantages=advantages[start:end],adv_mask=mask[start:end])
            loss,metrics=self.actor(cropped)
            yield loss,metrics,chunk/group

    def source_fm_loss(self, condition, target, *, generator):
        return source_fm_replay_loss(self.actor,condition,target,generator=generator)


class ActualNativeUpdateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from starVLA.model.modules.qwen_sourceflow.vendor.action_dit import ActionDiT
        torch.set_num_threads(2);torch.manual_seed(17)
        cls.head=ActionDiT(action_dim=3,hidden_dim=16,ffn_dim=32,text_dim=20,freq_dim=8,
                          num_heads=2,attn_head_dim=8,num_layers=2,eps=1e-6,use_gradient_checkpointing=False).eval()
        cls.conditions={key:dict(context=torch.randn(1,3,20),context_mask=torch.ones(1,3,dtype=torch.bool),
            visual_mask=torch.ones(1,4,dtype=torch.bool),visual_kv=[dict(k=torch.randn(1,4,16),v=torch.randn(1,4,16)) for _ in range(2)]) for key in ['s0','s1']}
        cls.archive=FrozenArchive.build([candidate(i,token=t,y=i*.5) for t in ['s0','s1'] for i in range(3)],
                                      train_tokens={'s0','s1'},token_logs={'s0':'log0','s1':'log0'})

    def collect(self, env):
        rollouts,records=[],[]
        with torch.no_grad():
            for n,token in enumerate(['s0','s1']):
                condition=env.condition(token)
                cond={k:([{x:y.expand(8,-1,-1).contiguous() for x,y in row.items()} for row in v] if k=='visual_kv' else
                         v.expand(8,*v.shape[1:]).contiguous()) for k,v in condition.items()}
                chain,times,deltas,steps=env.actor.sample_action_chain(cond,8,generator=torch.Generator().manual_seed(700+n))
                old=env.actor.action_chain_logprobs(cond,chain,times,deltas,stoch_idx=steps)
                values=[.4+.05*j for j in range(8)]
                rollouts.append(dict(chain=chain,old_logp=old.detach(),rewards=torch.tensor([values]),timesteps=times,
                                     deltas=deltas,stoch_idx=steps,token=token,log='log0'))
                rows=[candidate(j,token=token,reward=value,nc=.5 if j==0 else 1.) for j,value in enumerate(values)]
                records.append(rows)
        return rollouts,records

    def test_all_six_arms_execute_four_real_source_updates(self):
        for arm in ['P0','P1','P2','P3','P4','P5']:
            env=TinyNativeEnvironment(self.head,self.conditions)
            replay=arm in ['P2','P3','P5'];constrained=arm in ['P4','P5']
            update=PreservationUpdate(arm,seed=42,archive=self.archive if replay else None,replay_coefficient=.001 if replay else 0.,
                                      budgets=[.1,.1,.1] if constrained else None)
            rollouts,records=self.collect(env)
            prepared=update.prepare(rollouts,records,collection=0)
            self.assertFalse(prepared['raw_reward_tied'].any())
            optimizer=torch.optim.AdamW([p for p in env.actor.parameters() if p.requires_grad],lr=1e-5)
            result=update.update(env,rollouts,prepared,optimizer)
            self.assertEqual(len(result),4);self.assertTrue(all(float(state['step'])==4 for state in optimizer.state.values()))
            self.assertEqual(sum(row['supervised_scene_exposures'] for row in result),8 if replay else 0)
            self.assertTrue(result[-1]['gradient_norm']>0)
            if constrained:self.assertEqual(update.dual.collections,1)

    def test_final_cost_mask_survives_equal_PDMS(self):
        env=TinyNativeEnvironment(self.head,self.conditions);rollouts,records=self.collect(env)
        for r,rows in zip(rollouts,records):
            r['rewards'].zero_()
            for row in rows:row['raw_reward']=0.
        p0=PreservationUpdate('P0',seed=42).prepare(rollouts,records,collection=0)
        p4=PreservationUpdate('P4',seed=42,budgets=[.1,.1,.1]).prepare(rollouts,records,collection=0)
        self.assertFalse(p0['mask'].any());self.assertTrue(p4['mask'].all())

    def test_P2_P3_replay_scene_exposure_is_identical(self):
        output=[]
        for arm in ['P2','P3']:
            env=TinyNativeEnvironment(self.head,self.conditions);rollouts,records=self.collect(env)
            update=PreservationUpdate(arm,seed=42,archive=self.archive,replay_coefficient=.001)
            prepared=update.prepare(rollouts,records,collection=0)
            optimizer=torch.optim.AdamW([p for p in env.actor.parameters() if p.requires_grad],lr=1e-5)
            rows=update.update(env,rollouts,prepared,optimizer)
            output.append([row['replay_scenes'] for row in rows])
        self.assertEqual(output[0],output[1])

    def test_mid_inner_resume_preserves_archive_rng_dual_optimizer(self):
        from starVLA.rl.simwam_recipe.recovery_state import tensor_digest
        env=TinyNativeEnvironment(self.head,self.conditions);rollouts,records=self.collect(env)
        update=PreservationUpdate('P5',seed=42,archive=self.archive,replay_coefficient=.001,budgets=[.1,.1,.1])
        prepared=update.prepare(rollouts,records,collection=0)
        optimizer=torch.optim.AdamW([p for p in env.actor.parameters() if p.requires_grad],lr=1e-5)
        saved={}
        def capture(inner,row,state):
            if inner==1:saved.update(actor=copy.deepcopy(env.actor.state_dict()),optimizer=copy.deepcopy(optimizer.state_dict()),state=copy.deepcopy(state))
        continuous=update.update(env,rollouts,prepared,optimizer,after_inner=capture)
        restored=TinyNativeEnvironment(self.head,self.conditions);restored.actor.load_state_dict(saved['actor'])
        ropt=torch.optim.AdamW([p for p in restored.actor.parameters() if p.requires_grad],lr=1e-5);ropt.load_state_dict(saved['optimizer'])
        rupd=PreservationUpdate('P5',seed=42,archive=self.archive,replay_coefficient=.001,budgets=[.1,.1,.1]);rupd.load_state_dict(saved['state'])
        resumed=rupd.update(restored,rollouts,prepared,ropt,start_inner=1)
        self.assertEqual(tensor_digest(env.actor.state_dict()),tensor_digest(restored.actor.state_dict()))
        self.assertEqual(tensor_digest(optimizer.state_dict()),tensor_digest(ropt.state_dict()))
        self.assertEqual(tensor_digest(update.state_dict()),tensor_digest(rupd.state_dict()))
        self.assertEqual([r['replay_candidate_ids'] for r in continuous[1:]],[r['replay_candidate_ids'] for r in resumed])


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',required=True);args=p.parse_args()
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(t) for t in [ArchiveTests,CostTests,DiagnosticsTests,ActualNativeFMTests,ActualNativeUpdateTests])
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,
                                    scope='CPU arithmetic, fixed archive, tiny actual native source ActionDiT/LoRA',
                                    real_driving_results=False, failures=len(result.failures), errors=len(result.errors)),indent=2)+'\n')
    if not result.wasSuccessful():raise SystemExit(1)


if __name__=='__main__':main()
