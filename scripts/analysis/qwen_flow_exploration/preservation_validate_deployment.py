"""CPU direct-merge/source-wrapper/ten-step-ODE and pure-package qualification."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import torch
from starVLA.model.modules.qwen_sourceflow.vendor.action_dit import ActionDiT
from starVLA.model.modules.qwen_sourceflow.cached_action_runner import CachedActionRunner
from starVLA.model.modules.qwen_sourceflow.source_flow_policy import SourceFlowPolicy
from starVLA.rl.qwen_sourceflow.actor import NativeRecipeActor
from starVLA.rl.qwen_sourceflow.recipe import expected_recipe
from starVLA.rl.simwam_recipe.lora_adapter import adapter_state
from starVLA.rl.simwam_recipe.vendor.lora import merged_state_dict
from starVLA.rl.qwen_flow_exploration.preservation.deployment import merge_native_adapters_, export_adapters, read_adapter_package
from starVLA.rl.qwen_flow_exploration.checkpoint import save_state
from starVLA.rl.qwen_sourceflow.contracts import fingerprint, file_sha
from starVLA.rl.simwam_recipe.recovery_state import tensor_digest
from starVLA.model.modules.qwen_sourceflow.action_codec import ActionCodec
from scripts.analysis.qwen_flow_exploration.preservation_evaluate import verified_row, summarize


class DirectMergeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(83)
        self.original=ActionDiT(action_dim=3,hidden_dim=32,ffn_dim=64,text_dim=40,freq_dim=16,num_heads=2,
            attn_head_dim=12,num_layers=3,eps=1e-6,use_gradient_checkpointing=False).eval()
        self.actor=NativeRecipeActor(copy.deepcopy(self.original),expected_recipe('R_CANONICAL'))
        with torch.no_grad():
            for name,value in self.actor.named_parameters():
                if name.endswith('lora_B'):value.normal_(std=.01)
        self.adapters=adapter_state(self.actor)
        self.condition=dict(context=torch.randn(2,4,40),context_mask=torch.ones(2,4,dtype=torch.bool),
            visual_mask=torch.ones(2,5,dtype=torch.bool),visual_kv=[dict(k=torch.randn(2,5,24),v=torch.randn(2,5,24)) for _ in range(3)])
        self.condition['visual_mask'][1,-1]=False

    def test_exact_upstream_merged_state_and_no_reference(self):
        head=copy.deepcopy(self.original)
        record=merge_native_adapters_(head,self.adapters,expected_layers=3)
        expected=merged_state_dict(self.actor.action_expert)
        self.assertEqual(set(head.state_dict()),set(expected))
        for name,value in head.state_dict().items():
            torch.testing.assert_close(value,expected[name],atol=0,rtol=0)
        self.assertFalse(record['reference_actor_constructed'])
        self.assertTrue(all(not p.requires_grad for p in head.parameters()))
        self.assertFalse(any('lora_' in name or 'ref_action_expert' in name for name in head.state_dict()))

    def test_wrapper_velocity_and_full_ten_step_ODE(self):
        head=copy.deepcopy(self.original);merge_native_adapters_(head,self.adapters)
        x=torch.randn(2,8,3);times=torch.tensor([250.,700.])
        expected=self.actor.action_velocity(x,times,self.condition)
        actual=CachedActionRunner(head)(x,times,self.condition)
        torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-6)
        root=torch.randn(2,8,3)
        old=SourceFlowPolicy(self.actor.action_expert).sample_ode(self.condition,root)
        new=SourceFlowPolicy(head).sample_ode(self.condition,root)
        torch.testing.assert_close(old,new,atol=1e-5,rtol=1e-5)

    def test_bad_or_extra_adapters_rejected_before_mutation(self):
        for field in ['shape','nonfinite','extra']:
            values=copy.deepcopy(self.adapters);key=sorted(values)[-1]
            if field=='shape':values[key]=torch.zeros(1)
            if field=='nonfinite':values[key].view(-1)[-1]=float('nan')
            if field=='extra':values['unexpected.weight']=torch.zeros(1)
            head=copy.deepcopy(self.original);before=tensor_digest(head.state_dict())
            with self.assertRaises(ValueError):merge_native_adapters_(head,values)
            self.assertEqual(before,tensor_digest(head.state_dict()))

    def test_double_merge_rejected(self):
        head=copy.deepcopy(self.original);merge_native_adapters_(head,self.adapters)
        with self.assertRaises(ValueError):merge_native_adapters_(head,self.adapters)

    def test_zero_B_is_original_policy(self):
        values={key:(torch.zeros_like(value) if key.endswith('lora_B') else value) for key,value in self.adapters.items()}
        head=copy.deepcopy(self.original);merge_native_adapters_(head,values)
        self.assertEqual(tensor_digest(head.state_dict()),tensor_digest(self.original.state_dict()))

    def test_pure_package_survives_removal_of_training_state(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);run=root/'training';run.mkdir()
            protocol=dict(sha256='qualification-protocol',base_model_sha256='qualification-base')
            identity=dict(arm='P0',seed=42,qualification=True,common_protocol_sha256=protocol['sha256'],base_model_sha256=protocol['base_model_sha256'])
            identity['sha256']=fingerprint(identity);(run/'identity.json').write_text(json.dumps(identity))
            state=dict(phase='COMPLETE',round=1,updates=4,next_inner=0,arm='P0',seed=42)
            payload=dict(batch=None,identity_sha256=identity['sha256'],adapters=self.adapters,
                archive='/nonexistent/training/archive',dual={'values':[99.,99.]},optimizer={'unused':torch.zeros(3)})
            checkpoint=save_state(run,state,payload,protocol['sha256'])
            package=export_adapters(checkpoint,root/'deployment',protocol,arm='P0',seed=42,qualification=True)
            # All files here are synthetic CPU qualification assets. The final
            # package must remain loadable after removing its training parent.
            shutil.rmtree(run)
            values,metadata=read_adapter_package(package,protocol,arm='P0',seed=42,qualification=True)
            self.assertEqual(tensor_digest(values),tensor_digest(self.adapters))
            self.assertTrue(metadata['no_archive_dual_optimizer_reference_or_teacher_in_payload'])
            with self.assertRaises(ValueError):read_adapter_package(package,protocol,arm='P0',seed=42,qualification=False)
            with self.assertRaises(ValueError):read_adapter_package(package,protocol,arm='P1',seed=42,qualification=True)
            with self.assertRaises(ValueError):read_adapter_package(package,protocol,arm='P0',seed=43,qualification=True)
            head=copy.deepcopy(self.original);merge_native_adapters_(head,values)
            self.assertEqual(tensor_digest(head.state_dict()),tensor_digest(merged_state_dict(self.actor.action_expert)))

    def test_feedback_S1_S2_S3_checkpoint_arm_mapping(self):
        for arm in ('S1','S2','S3'):
            with self.subTest(arm=arm),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);run=root/'training';run.mkdir()
                protocol=dict(sha256='qualification-protocol',base_model_sha256='qualification-base')
                config=dict(schema='native_feedback_student_v1',arm=arm,seed=42,updates=256,
                            common_protocol_sha256=protocol['sha256'])
                config['sha256']=fingerprint(config);(run/'config.json').write_text(json.dumps(config))
                state=dict(phase='COMPLETE',round=256,updates=256,next_inner=0,arm='feedback_'+arm,seed=42)
                payload=dict(batch=None,config_sha256=config['sha256'],adapters=self.adapters,
                             teacher_bank_hash='never_open_this_teacher',optimizer={'unused':torch.zeros(3)})
                checkpoint=save_state(run,state,payload,protocol['sha256'])
                package=export_adapters(checkpoint,root/'deployment',protocol,arm=arm,seed=42)
                values,metadata=read_adapter_package(package,protocol,arm=arm,seed=42)
                self.assertEqual(metadata['checkpoint_arm'],'feedback_'+arm)
                self.assertEqual(metadata['source_training_identity_field'],'config_sha256')
                self.assertEqual(tensor_digest(values),tensor_digest(self.adapters))
                with self.assertRaises(ValueError):export_adapters(checkpoint,root/'bad_arm',protocol,arm='P0',seed=42)
                with self.assertRaises(ValueError):export_adapters(checkpoint,root/'bad_seed',protocol,arm=arm,seed=43)
                with self.assertRaises(ValueError):export_adapters(checkpoint,root/'bad_qual',protocol,arm=arm,seed=42,qualification=True)
                # No identity.json, teacher file or optimizer may be required by inference.
                shutil.rmtree(run)
                read_adapter_package(package,protocol,arm=arm,seed=42)


class EvaluationRecordTests(unittest.TestCase):
    def record(self):
        metrics=dict(score=.95,no_at_fault_collisions=1.,drivable_area_compliance=1.,time_to_collision_within_bound=1.,
                     ego_progress=.88,comfort=1.,driving_direction_compliance=1.)
        planned=[[1.5*(i+1),0.,0.] for i in range(8)];simulated=[[0.,0.,0.] for _ in range(41)]
        row=dict(scene_token='synthetic',log_id='log',seed=2101,split='TRAIN',status='OK',scorer_config_hash='scorer',sampler_config_hash='sampler',
            evaluation_identity='evaluation',native_subscores=copy.deepcopy(metrics),raw_reward=.95,
            planned_trajectory=planned,normalized_action=ActionCodec.norm_odo(torch.tensor(planned)).tolist(),simulated_trajectory=simulated,
            feedback=dict(status='OK',token='synthetic',log='log',protocol_sha256='scorer',metrics=metrics,
                          requested_trajectory=copy.deepcopy(planned),simulated_states=copy.deepcopy(simulated)))
        return row

    def check(self,row):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'record.json';row['sha256']=fingerprint(row);path.write_text(json.dumps(row))
            return verified_row(path,dict(sha256='evaluation',split='TRAIN',scorer_config_hash='scorer',sampler_config_hash='sampler'),'synthetic',2101)

    def test_exact_request_and_feedback_associations(self):
        row=self.record();self.assertEqual(self.check(row)['raw_reward'],.95)
        for field in ['request','reward','metrics','seed','sampler']:
            bad=self.record()
            if field=='request':bad['feedback']['requested_trajectory'][0][0]+=.5
            if field=='reward':bad['raw_reward']=.96
            if field=='metrics':bad['native_subscores']['time_to_collision_within_bound']=0.
            if field=='seed':bad['seed']=2102
            if field=='sampler':bad['sampler_config_hash']='other_sampler'
            with self.assertRaises(ValueError):self.check(bad)

    def test_seed_population_cannot_be_reduced(self):
        row=self.record()
        self.assertEqual(summarize([row],['synthetic'],[2101])['score'],.95)
        with self.assertRaises(ValueError):summarize([row],['synthetic'],[2101,2102])


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',required=True)
    p.add_argument('--actual-evaluation',help='Read-only existing real scene/feedback association check; no inference/scoring')
    p.add_argument('--protocol',help='Required for legacy evaluation identities whose scorer is bound through common protocol')
    args=p.parse_args()
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(cls) for cls in [DirectMergeTests,EvaluationRecordTests])
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    actual=None
    if result.wasSuccessful() and args.actual_evaluation:
        path=Path(args.actual_evaluation);identity=json.loads((path/'identity.json').read_text())
        from starVLA.rl.qwen_flow_exploration.protocol import read_protocol
        protocol=read_protocol(args.protocol)
        body=dict(identity)
        if (body.pop('sha256')!=fingerprint(body) or identity['protocol_sha256']!=protocol['sha256'] or
                identity['base_model_sha256']!=protocol['base_model_sha256']):
            raise ValueError('Actual evaluation does not bind common protocol/base')
        # Legacy L0 identity binds scorer/sampler through its sealed protocol.
        # The original identity and its original hash remain untouched on disk.
        identity=dict(identity,scorer_config_hash=protocol['scorer']['sha256'],sampler_config_hash=fingerprint(protocol['samplers']))
        record_path=next(iter(sorted((path/'scenes').glob('*.json'))))
        row=json.loads(record_path.read_text());verified_row(record_path,identity,row['scene_token'],row['seed'])
        rejected=[]
        for field in ('seed','raw_reward','requested_trajectory'):
            bad=copy.deepcopy(row)
            if field=='seed':bad['seed']+=100
            elif field=='raw_reward':bad['raw_reward']+=.001
            else:bad['feedback']['requested_trajectory'][0][0]+=.1
            bad.pop('sha256');bad['sha256']=fingerprint(bad)
            with tempfile.TemporaryDirectory() as temp:
                changed=Path(temp)/'changed.json';changed.write_text(json.dumps(bad))
                try:verified_row(changed,identity,row['scene_token'],row['seed'])
                except ValueError:rejected.append(field)
                else:raise AssertionError('Real canonical record association accepted corrupted '+field)
        actual=dict(status='PASS',source_evaluation=str(path),source_identity_sha256=identity['sha256'],
                    source_record_file_sha256=file_sha(record_path),wrong_field_rejections=rejected,
                    no_new_inference_or_score=True,not_a_new_deployment_performance_result=True)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,
        scope='CPU actual tiny source ActionDiT, nonzero native LoRA, original upstream merge/velocity/ODE',
        real_deployment_efficacy=False,actual_existing_record_check=actual,
        failures=len(result.failures),errors=len(result.errors)),indent=2)+'\n')
    if not result.wasSuccessful():raise SystemExit(1)


if __name__=='__main__':main()
