"""CPU actual tiny-source audit qualification; synthetic feedback is not PDMS evidence."""
import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import torch
from starVLA.model.modules.qwen_sourceflow.vendor.action_dit import ActionDiT
from starVLA.model.modules.qwen_sourceflow.cached_action_runner import expand_condition
from starVLA.model.modules.qwen_sourceflow.action_codec import ActionCodec
from starVLA.rl.qwen_sourceflow.actor import NativeRecipeActor
from starVLA.rl.qwen_sourceflow.recipe import expected_recipe
from starVLA.rl.simwam_recipe.lora_adapter import adapter_state,restore_adapters
from starVLA.rl.simwam_recipe.recovery_state import tensor_digest
from starVLA.rl.simwam_recipe.contracts import seed_for
from starVLA.rl.qwen_sourceflow.contracts import fingerprint,file_sha
from starVLA.rl.qwen_flow_exploration.preservation.sampler_audit import (validate_budget,validate_group,
    reference_rng_book,freeze_and_hash,model_hash,publish_group,load_group,mean_summary,NAMESPACE)


def mock_records(chain,token,protocol):
    rows=[]
    for i,action in enumerate(chain[:,-1]):
        planned=ActionCodec.denorm_odo(action).tolist();simulated=[[0.,0.,0.] for _ in range(41)]
        metrics=dict(score=i/20,no_at_fault_collisions=1.,drivable_area_compliance=1.,time_to_collision_within_bound=1.,
                     ego_progress=.5,comfort=1.,driving_direction_compliance=1.)
        rows.append(dict(scene_token=token,log_id='synthetic_log',status='OK',split='TRAIN',scorer_config_hash='synthetic_scorer',
            sampler_config_hash=fingerprint(protocol['samplers']),raw_reward=metrics['score'],native_subscores=metrics,
            normalized_action=action.tolist(),planned_trajectory=planned,simulated_trajectory=simulated,
            latent_or_proposal_parameters=dict(root=chain[i,0].tolist(),candidate_index=i,oracle_selection=False),
            feedback=dict(status='OK',token=token,log='synthetic_log',protocol_sha256='synthetic_scorer',metrics=copy.deepcopy(metrics),
                          requested_trajectory=copy.deepcopy(planned),simulated_states=copy.deepcopy(simulated))))
    return rows


class AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);torch.manual_seed(219)
        cls.original=ActionDiT(action_dim=3,hidden_dim=16,ffn_dim=32,text_dim=20,freq_dim=8,
            num_heads=2,attn_head_dim=8,num_layers=2,eps=1e-6,use_gradient_checkpointing=False).eval()
        cls.condition=dict(context=torch.randn(1,3,20),context_mask=torch.ones(1,3,dtype=torch.bool),
            visual_mask=torch.ones(1,4,dtype=torch.bool),visual_kv=[dict(k=torch.randn(1,4,16),v=torch.randn(1,4,16)) for _ in range(2)])
        cls.protocol=dict(sha256='synthetic_protocol',base_model_sha256='synthetic_base',splits={'TRAIN':['synthetic0','synthetic1']},
                          execution={'candidate_chunk':4},samplers={'synthetic_fixture_only':True},scorer={'sha256':'synthetic_scorer'})
        cls.actor=NativeRecipeActor(copy.deepcopy(cls.original),expected_recipe('R_CANONICAL')).eval()
        rng=torch.Generator().manual_seed(seed_for(NAMESPACE,42,'unfiltered128'))
        cls.rollouts=[];cls.groups=[]
        with torch.no_grad():
            for token in cls.protocol['splits']['TRAIN']:
                rollout,rows=cls.collect(cls.actor,rng,token)
                cls.rollouts.append(rollout);cls.groups.append(rows)
        cls.pool=dict(tokens=cls.protocol['splits']['TRAIN'],rollouts=cls.rollouts,record_groups=cls.groups)

    @classmethod
    def collect(cls,actor,rng,token):
        chains=[];probabilities=[]
        with torch.no_grad():
            for _ in range(4):
                condition=expand_condition(cls.condition,4)
                chain,times,deltas,indices=actor.sample_action_chain(condition,8,generator=rng)
                old=actor.action_chain_logprobs(condition,chain,times,deltas,stoch_idx=indices)
                chains.append(chain);probabilities.append(old)
        chain=torch.cat(chains);rows=mock_records(chain,token,cls.protocol)
        return dict(chain=chain,old_logp=torch.cat(probabilities),rewards=torch.tensor([[row['raw_reward'] for row in rows]]),
                    timesteps=times,deltas=deltas,stoch_idx=indices,sampling_chunk=4,token=token,split='TRAIN'),rows

    def test_reference_roots_and_full_original_RNG_boundaries(self):
        book=reference_rng_book(self.pool)
        rng=torch.Generator().manual_seed(seed_for(NAMESPACE,42,'unfiltered128'))
        for i,token in enumerate(self.pool['tokens']):
            self.assertTrue(torch.equal(rng.get_state(),book[i]['before']))
            actual,_=self.collect(self.actor,rng,token)
            self.assertTrue(torch.equal(actual['chain'],self.rollouts[i]['chain']))
            self.assertTrue(torch.equal(rng.get_state(),book[i]['after']))
        wrong=copy.deepcopy(self.pool);wrong['rollouts'][1]['chain'][0,0,0,0]+=.1
        with self.assertRaises(ValueError):reference_rng_book(wrong)
        wrong=copy.deepcopy(self.pool);wrong['tokens'].reverse()
        with self.assertRaises(ValueError):reference_rng_book(wrong)

    def test_actual_adapter_restore_then_frozen_no_update(self):
        actor=NativeRecipeActor(copy.deepcopy(self.original),expected_recipe('R_CANONICAL'))
        values=adapter_state(actor)
        for name,value in values.items():
            if name.endswith('lora_B'):value.add_(.01)
        restore_adapters(actor,values)
        self.assertEqual(tensor_digest(adapter_state(actor)),tensor_digest(values))
        environment=SimpleNamespace(model=actor.action_expert,actor=actor)
        before=freeze_and_hash(environment)
        rng=torch.Generator().manual_seed(seed_for(NAMESPACE,42,'unfiltered128'))
        current,_=self.collect(actor,rng,'synthetic0')
        self.assertTrue(torch.equal(current['chain'][:,0],self.rollouts[0]['chain'][:,0]))
        self.assertFalse(torch.equal(current['chain'][:,-1],self.rollouts[0]['chain'][:,-1]))
        self.assertEqual(before,model_hash(environment))
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in actor.parameters()))

    def test_atomic_partial_reuse_preserves_stream_and_records(self):
        book=reference_rng_book(self.pool);identity={'sha256':'synthetic_audit'}
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            publish_group(root,identity,0,'synthetic0',self.rollouts[0],self.groups[0],self.groups[0],book[0])
            payload=load_group(root,identity,0,'synthetic0',book[0],self.protocol)
            self.assertEqual(tensor_digest(payload['rollout']),tensor_digest(self.rollouts[0]))
            rng=torch.Generator();rng.set_state(payload['RNG_after'])
            resumed,rows=self.collect(self.actor,rng,'synthetic1')
            self.assertEqual(tensor_digest(resumed),tensor_digest(self.rollouts[1]))
            self.assertEqual(rows,self.groups[1])
            with self.assertRaises(FileExistsError):publish_group(root,identity,0,'synthetic0',self.rollouts[0],self.groups[0],self.groups[0],book[0])
            with self.assertRaises(ValueError):load_group(root,{'sha256':'wrong'},0,'synthetic0',book[0],self.protocol)

    def test_missing_candidates_or_wrong_feedback_never_drop(self):
        for rollout,rows,token in zip(self.rollouts,self.groups,self.pool['tokens']):validate_group(rollout,rows,token,self.protocol)
        with self.assertRaises(ValueError):validate_group(self.rollouts[0],self.groups[0][:-1],'synthetic0',self.protocol)
        rows=copy.deepcopy(self.groups[0]);rows[0]['feedback']['requested_trajectory'][0][0]+=.1
        with self.assertRaises(ValueError):validate_group(self.rollouts[0],rows,'synthetic0',self.protocol)

    def test_candidate_means_never_oracle(self):
        summary=mean_summary(self.groups)
        self.assertAlmostEqual(summary['mean_PDMS'],37.5)
        self.assertNotEqual(summary['mean_PDMS'],75.)
        self.assertEqual(summary['candidates'],32)

    def test_registered_exact_population_and_costs(self):
        protocol=copy.deepcopy(self.protocol);protocol['splits']['TRAIN']=[f'synthetic{i}' for i in range(128)]
        calibration=dict(sha256='calibration',reference_SDE_pool_sha256='pool')
        lock=dict(schema='preservation_terminal_sampler_audit_budget_v1',phase='TERMINAL_AUDIT_LOCKED',
            protocol_sha256=protocol['sha256'],calibration_sha256='calibration',reference_pool_sha256='pool',
            TRAIN_tokens_sha256=fingerprint(protocol['splits']['TRAIN']),scenes=128,G=16,K=10,candidate_chunk=4,
            stochastic_steps=[7,8,9],new_candidates_per_terminal_arm=4096,new_candidates_BASE=2048,total_new_candidates=26624,
            reference_rng_seed=42,reference_rng_namespace=NAMESPACE,arms=['BASE','P0','P1','P2','P3','P4','P5'],training_seeds=[42])
        lock['sha256']=fingerprint(lock);validate_budget(lock,protocol,calibration,arm='BASE',training_seed=42)
        validate_budget(lock,protocol,calibration,arm='P5',training_seed=42)
        with self.assertRaises(ValueError):validate_budget(lock,protocol,calibration,arm='P5',training_seed=43)
        wrong=copy.deepcopy(lock);wrong['G']=8;wrong.pop('sha256');wrong['sha256']=fingerprint(wrong)
        with self.assertRaises(ValueError):validate_budget(wrong,protocol,calibration,arm='P5',training_seed=42)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',required=True);p.add_argument('--actual-reference-pool')
    a=p.parse_args();result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(AuditTests))
    actual=None
    if a.actual_reference_pool:
        pool=torch.load(a.actual_reference_pool,map_location='cpu',weights_only=False);book=reference_rng_book(pool)
        actual=dict(reference_file_sha256=file_sha(a.actual_reference_pool),scenes=len(book),G=len(pool['rollouts'][0]['chain']),
                    exact_saved_roots_and_original_CPU_RNG_stream=True,book_sha256=tensor_digest(book),no_new_scores=True)
    record=dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,
        real_model=False,qualification_only=True,scope='Actual tiny original source actor/adapter, original_randn, synthetic scoring feedback',
        actual_existing_reference_RNG_check=actual,failures=len(result.failures),errors=len(result.errors))
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise FileExistsError('Preserve earlier qualification attempts')
    out.write_text(json.dumps(record,indent=2)+'\n')
    if not result.wasSuccessful():raise SystemExit(1)


if __name__=='__main__':main()
