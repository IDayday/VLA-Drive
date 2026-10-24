"""Exercise the ACTUAL latent CLI transaction/resume on a CPU mock scorer.

This validates orchestration and persisted RNG/old-batch identity, not driving
rewards. The tiny generator still runs the unchanged native source ActionDiT.
"""
import argparse
import copy
import json
import os
import signal
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import torch
from torch import nn
from starVLA.model.modules.qwen_sourceflow.source_flow_policy import SourceFlowPolicy
from starVLA.model.modules.qwen_sourceflow.vendor.action_dit import ActionDiT
from starVLA.model.modules.qwen_sourceflow.cached_action_runner import expand_condition
from starVLA.rl.qwen_flow_exploration.latent.frozen import freeze_generator
from starVLA.rl.qwen_flow_exploration.checkpoint import load_state, save_state
from starVLA.rl.qwen_flow_exploration.protocol import Ledger
from starVLA.rl.qwen_sourceflow.contracts import fingerprint
from starVLA.rl.simwam_recipe.recovery_state import tensor_digest
from scripts.analysis.qwen_flow_exploration import latent_train


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(52)
    generator = nn.Module()
    generator.policy = SourceFlowPolicy(ActionDiT(action_dim=3, hidden_dim=32, ffn_dim=64, text_dim=40,
        freq_dim=16, num_heads=2, attn_head_dim=12, num_layers=3, eps=1e-6, use_gradient_checkpointing=False))
    generator.register_buffer('nonpersistent', torch.tensor([5.]), persistent=False)
    freeze_generator(generator)
    condition = dict(context=torch.randn(1,4,40),context_mask=torch.ones(1,4,dtype=torch.bool),
        visual_mask=torch.ones(1,7,dtype=torch.bool),visual_kv=[dict(k=torch.randn(1,7,24),v=torch.randn(1,7,24)) for _ in range(3)])

    class MockEnvironment:
        def __init__(self, protocol, **kwargs):
            self.protocol = protocol
            self.device = torch.device('cpu')
            self.model = copy.deepcopy(generator)
            self.experiment_id = kwargs['experiment_id']
            self.ledger = Ledger(kwargs['ledger_path'])

        def condition(self, token, split):
            assert split == 'TRAIN' and token in self.protocol['splits']['TRAIN']
            return copy.deepcopy(condition)

        def set_behavior_identity(self, identity):
            self.behavior = copy.deepcopy(identity)

        def sample_ode(self, scene_condition, roots):
            assert scene_condition['context'].shape[0] == 1
            output = []
            for start in range(0,len(roots),4):
                noise = roots[start:start+4]
                output.append(self.model.policy.sample_ode(expand_condition(scene_condition,len(noise)),noise))
            return torch.cat(output)

        def score_actions(self, token, actions, split, **kwargs):
            # Bounded deterministic MOCK scalar is solely a transaction oracle.
            rewards = torch.sigmoid(-(actions-.4).square().mean((-1,-2)))
            return [dict(status='OK',raw_reward=float(value),token=token,
                         qualification_reward='CPU_mock_not_PDMS') for value in rewards]

        def close(self):
            self.ledger.save()

    protocol = dict(stage='SMOKE_AND_PROFILE_ONLY',splits=dict(TRAIN=['CPU_MOCK_A','CPU_MOCK_B']),
                    screen=dict(B=2,G=4,inner_epochs=4,rollout_rounds=2),
                    base_model_sha256='CPU_MOCK_NO_DRIVING',advantages='common FP64 globalstd floor.01')
    protocol['sha256'] = fingerprint(protocol)
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        protocol_file = root/'protocol.json'
        protocol_file.write_text(json.dumps(protocol))
        def run(target, resume=False):
            argv = ['latent_train','--protocol',str(protocol_file),'--output',str(target),
                    '--arm','L1','--seed','42','--qualification','--limit','2']
            if resume:
                argv += ['--resume',str(target)]
            with patch.object(latent_train,'NativeEnvironment',MockEnvironment), patch.object(sys,'argv',argv):
                latent_train.main()
        continuous, interrupted = root/'continuous',root/'interrupted'
        run(continuous)
        class ExpectedInterrupt(Exception):pass
        actual_save = latent_train.save_state
        def interrupted_save(run_path,state,payload,sha):
            checkpoint = actual_save(run_path,state,payload,sha)
            if state['updates'] == 1:
                raise ExpectedInterrupt('CPU injection AFTER valid inner1 publication')
            return checkpoint
        with patch.object(latent_train,'save_state',interrupted_save):
            try:
                run(interrupted)
            except ExpectedInterrupt:
                pass
            else:
                raise AssertionError('Expected injection did not happen')
        state,payload=load_state(interrupted,protocol['sha256'],restore_rng=False)
        assert state['phase']=='INNER_COMMITTED' and state['updates']==1 and state['next_inner']==1
        old_batch=payload['batch'].digest()
        run(interrupted,resume=True)
        left,lp=load_state(continuous,protocol['sha256'],restore_rng=False)
        right,rp=load_state(interrupted,protocol['sha256'],restore_rng=False)
        assert left==right and left['updates']==8 and left['round']==2
        for key in ['policy','optimizer','scheduler','root_generator_state']:
            assert tensor_digest(lp[key])==tensor_digest(rp[key]),key
        assert lp['history']==rp['history']
        assert lp['history'][0]['behavior_batch_digest']==old_batch
        for mode in ['file','SIGUSR1']:
            stopped=root/('stopped_'+mode)
            def native_stop_save(run_path,state,payload,sha):
                checkpoint=actual_save(run_path,state,payload,sha)
                if state['updates']==1:
                    if mode=='file':
                        (Path(run_path)/'STOP_AFTER_BOUNDARY').touch()
                    else:
                        os.kill(os.getpid(),signal.SIGUSR1)
                return checkpoint
            with patch.object(latent_train,'save_state',native_stop_save):
                run(stopped)
            status=json.loads((stopped/'progress.json').read_text())
            assert status['status']=='STOPPED' and status['phase']=='INNER_COMMITTED' and status['updates']==1
            if mode=='file':
                (stopped/'STOP_AFTER_BOUNDARY').unlink()
            run(stopped,resume=True)
            stopped_state,sp=load_state(stopped,protocol['sha256'],restore_rng=False)
            assert stopped_state==left
            for key in ['policy','optimizer','scheduler','root_generator_state']:
                assert tensor_digest(sp[key])==tensor_digest(lp[key]),(mode,key)
            assert sp['history']==lp['history']
        # A collecting failure is not a restorable boundary and cannot replace latest.
        original_latest=(interrupted/'latest.json').read_bytes()
        bad=dict(right,phase='COLLECTING')
        try:
            save_state(interrupted,bad,rp,protocol['sha256'])
        except ValueError:
            pass
        else:
            raise AssertionError('COLLECTING was published')
        assert (interrupted/'latest.json').read_bytes()==original_latest
        result=dict(status='PASS',qualification_only=True,actual_CLI_executed=True,
            episodes=dict(rounds=2,actual_optimizer_steps=8),
            interruption='after inner1 checkpoint then actual CLI --resume',
            exact_matches=['policy','Adam','scheduler','custom root RNG','effective metrics ancestry','subsequent sampled behavior batch'],
            failure_does_not_advance_latest=True, common_atomic_checkpoint_exercised=True,
            native_SIGUSR1_and_STOP_file_resume_bitwise=True,
            reward='CPU mock; not canonical PDMS or driving evidence',
            generator='actual native tiny ActionDiT source shift5 ODE')
    out=Path(args.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',output=str(out))))


if __name__=='__main__':
    main()
