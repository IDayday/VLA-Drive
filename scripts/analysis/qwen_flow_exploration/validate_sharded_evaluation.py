"""CPU partition/publication/admission tests; synthetic metrics are not PDMS."""
import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest
import torch
from starVLA.model.modules.qwen_sourceflow.action_codec import ActionCodec
from starVLA.rl.qwen_flow_exploration import evaluation as shared
from starVLA.rl.qwen_flow_exploration.native import NativeEnvironment
from starVLA.rl.qwen_flow_exploration.common_analysis import load_evaluation
from starVLA.rl.qwen_flow_exploration.evaluation_admission import validate_capability,authorization_fingerprint,GRANT,TASK
from scripts.analysis.qwen_flow_exploration.aggregate_sharded_evaluation import aggregate
from starVLA.rl.qwen_sourceflow.contracts import fingerprint


def sealed(value):return dict(value,sha256=fingerprint(value))


class Tests(unittest.TestCase):
    def test_partition_exact_union(self):
        rows=list('abcdefgh')
        halves=[shared.partition(rows,i,2) for i in range(2)]
        self.assertEqual(set(halves[0])|set(halves[1]),set(rows));self.assertFalse(set(halves[0])&set(halves[1]))
        for rank,world in [(0,0),(2,2),(0,9)]:
            with self.assertRaises(ValueError):shared.partition(rows,rank,world)

    def make_shards(self,root):
        tokens=list('abcd');seeds=[2101,2102]
        p=dict(sha256='synthetic-protocol',base_model_sha256='synthetic-base',evaluation_seeds=seeds,
            splits={'DEV_SELECT':tokens},diagnostic={'DEV_SELECT':tokens},scorer={'sha256':'synthetic-scorer'},
            samplers={'deployment':'single source shift5 ODE10, original standard Gaussian root'})
        identity=shared.identity(p,'base','BASE',42,None,tokens,True,None);identity['split']='DEV_SELECT';identity=sealed(identity)
        paths=[]
        for rank in range(2):
            d=root/f'rank{rank}';(d/'scenes').mkdir(parents=True);local=shared.partition(tokens,rank,2)
            ex=dict(rank=rank,world=2,full_tokens_sha256=fingerprint(tokens),local_tokens_sha256=fingerprint(local),
                    entry_sha256='synthetic',identity_sha256=identity['sha256'])
            (d/'identity.json').write_text(json.dumps(identity));(d/'execution.json').write_text(json.dumps(ex))
            for token in local:
                condition=dict(context=torch.zeros(1,1,4096),context_mask=torch.ones(1,1,dtype=torch.bool),
                    visual_kv=[dict(v=torch.zeros(1,1,3072))],visual_mask=torch.ones(1,1,dtype=torch.bool))
                for seed,_,metadata in shared.roots_for(None,condition,token,seeds,identity):
                    metrics={k:1. for k in ['score','no_at_fault_collisions','drivable_area_compliance',
                        'time_to_collision_within_bound','ego_progress','comfort','driving_direction_compliance']}
                    plan=ActionCodec.denorm_odo(torch.zeros(8,3)).tolist();states=torch.zeros(41,11).tolist()
                    row=dict(status='OK',scene_token=token,log_id='log'+str(tokens.index(token)//2),split='DEV_SELECT',seed=seed,
                        scorer_config_hash=p['scorer']['sha256'],sampler_config_hash=fingerprint(p['samplers']),
                        raw_reward=1.,native_subscores=metrics,normalized_action=torch.zeros(8,3).tolist(),
                        planned_trajectory=plan,simulated_trajectory=states,latent_or_proposal_parameters=metadata,
                        evaluation_identity=identity['sha256'],qualification_data='SYNTHETIC_NOT_DRIVING')
                    row['feedback']=dict(status='OK',token=token,log=row['log_id'],protocol_sha256=p['scorer']['sha256'],
                        metrics=metrics,requested_trajectory=plan,simulated_states=states)
                    (d/'scenes'/f'{token}_seed{seed}.json').write_text(json.dumps(sealed(row)))
            (d/'SHARD_COMPLETE.json').write_text(json.dumps(sealed(dict(status='COMPLETE',execution=ex))))
            invocation=d/'invocations/one';invocation.mkdir(parents=True)
            (invocation/'invocation.json').write_text(json.dumps(dict(status='COMPLETE',start_unix=1.,end_unix=2.,wall_seconds=1.,
                physical_inference_attempts=4,completed_inferences=4,new=4,reused=0,model_loading_seconds=.1)))
            paths.append(d)
        return p,paths

    def test_atomic_complete_assembly_and_scene_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);p,paths=self.make_shards(root);out=root/'published'
            result=aggregate(p,paths,out);self.assertEqual(result['logical_single_output_evaluations'],8)
            self.assertEqual(result['physical_inference_attempts_including_retries'],8)
            value=load_evaluation(out,p,allow_qualification=True);self.assertEqual(value.values.shape[:2],(4,2))
            self.assertTrue(all(x.is_symlink() for x in (out/'scenes').glob('*.json')))

    def test_missing_or_duplicate_shard_never_publishes(self):
        for mode in ['missing','duplicate']:
            with tempfile.TemporaryDirectory() as folder:
                root=Path(folder);p,paths=self.make_shards(root);out=root/'rejected'
                with self.assertRaises(ValueError):aggregate(p,paths[:1] if mode=='missing' else [paths[0],paths[0]],out)
                self.assertFalse(out.exists())

    def test_resealed_wrong_root_or_reward_rejected(self):
        for field in ['root','reward']:
            with tempfile.TemporaryDirectory() as folder:
                root=Path(folder);p,paths=self.make_shards(root);target=next((paths[0]/'scenes').glob('*.json'))
                row=json.loads(target.read_text());row.pop('sha256')
                if field=='root':row['latent_or_proposal_parameters']['root'][0][0]+=.1
                else:row['raw_reward']=.5
                target.write_text(json.dumps(sealed(row)))
                with self.assertRaises(ValueError):aggregate(p,paths,root/'rejected')
                self.assertFalse((root/'rejected').exists())

    def test_frozen_scope_cannot_construct_training_or_access_train(self):
        with self.assertRaises(PermissionError):NativeEnvironment({},trainable='lora',evaluation_scope='NAVTEST_ONLY')
        env=NativeEnvironment.__new__(NativeEnvironment);env.evaluation_scope='NAVTEST_ONLY'
        with self.assertRaises(PermissionError):env.dataset('TRAIN')
        with self.assertRaises(PermissionError):env.dataset('DEV_SELECT')
        with self.assertRaises(PermissionError):env.score_actions('token',torch.zeros(8,3),'TRAIN')
        with self.assertRaises(PermissionError):env.source_fm_loss({},torch.zeros(8,3))

    def test_exact_spare_gpu_capability_and_revocation(self):
        host='training-vlawm-tsinghua-zt6'
        policy=dict(allowed_hosts={},task_authorizations={TASK:{'allowed_hosts':[]},GRANT:{'allowed_hosts':[host],
            'authorized_gpu_indices':{host:list(range(1,8))}}},global_dispatch_prohibitions={host:{'new_task_dispatch_allowed':False}})
        lock=sealed(dict(protocol_sha256='synthetic',approved_evaluation_configurations=[]))
        lease=sealed(dict(status='ACTIVE',expires_unix=10.,hosts={host:dict(authorization_sha256=authorization_fingerprint(policy,host),gpu_indices=list(range(1,8)))},
            configuration_lock_sha256=lock['sha256'],evaluation_entry_sha256='qualified-entry'))
        self.assertEqual(validate_capability(policy,lease,lock,host,[1],'qualified-entry',1.),host)
        for indices,entry,now in [([0],'qualified-entry',1.),([1],'train.py',1.),([1],'qualified-entry',11.)]:
            with self.assertRaises(PermissionError):validate_capability(policy,lease,lock,host,indices,entry,now)
        revoked=copy.deepcopy(policy);revoked['global_dispatch_prohibitions'][host]['withdrawal']='new user withdrawal'
        with self.assertRaises(PermissionError):validate_capability(revoked,lease,lock,host,[1],'qualified-entry',1.)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',required=True);a=p.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    document=dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,
        data='synthetic publication arithmetic only, not canonical simulation or driving efficacy',actual_GPU_or_scorer_calls=0)
    Path(a.output).write_text(json.dumps(document,indent=2)+'\n')
    if not result.wasSuccessful():raise SystemExit(1)


if __name__=='__main__':main()
