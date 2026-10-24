"""CPU fault injection into the real private CLI; no model/GPU/job dispatch."""
import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
import types
from unittest.mock import patch
import torch
from starVLA.rl.qwen_flow_exploration.checkpoint import load_state
from starVLA.rl.qwen_sourceflow.contracts import fingerprint,file_sha


class MockLoss:
    adv_clip_max=None
    def _build_scheduler(self,*_):return torch.optim.lr_scheduler.LambdaLR(self.optimizer,lambda step:1.)


class MockActor(torch.nn.Module):
    def __init__(self):
        super().__init__();self.lora_A=torch.nn.Parameter(torch.tensor([.1]));self.lora_B=torch.nn.Parameter(torch.tensor([0.]))
        self._loss=MockLoss();self._velocity_calls=self._reference_calls=0


class MockLedger:
    def add(self,**_):pass
    def save(self):pass


class MockEnvironment:
    def __init__(self,*_,**__):
        self.actor=MockActor();self.device=torch.device('cpu');self.ledger=MockLedger();self.calls=0
    def condition(self,token):return None
    def set_behavior_identity(self,value):pass
    def sample_source_sde(self,condition,G,generator,**kwargs):
        self.calls+=1
        if self.fail_collection and self.calls==2:raise RuntimeError('injected_second_scene_scoring_failure')
        token=kwargs['token'];rewards=torch.linspace(.4,.7,G)
        rollout=dict(chain=torch.randn(G,11,8,3,generator=generator),old_logp=torch.zeros(G,3),rewards=rewards[None],token=token)
        records=[dict(raw_reward=float(value),native_subscores=dict(no_at_fault_collisions=1.,drivable_area_compliance=1.,
                      time_to_collision_within_bound=1.)) for value in rewards]
        return rollout,records
    def policy_chunks(self,*_):raise torch.OutOfMemoryError('injected_first_inner_OOM')
    def close(self):pass


def run_fault(source,root,*,fail_collection=False,expected_exception):
    module=types.ModuleType('private_cli_failure_test');module.__file__=str(source)
    exec(compile(source.read_text(),str(source),'exec'),module.__dict__)
    protocol=dict(sha256='CPU_qualification_protocol',base_model_sha256='CPU_qualification_base',
                  splits={'TRAIN':['synthetic0','synthetic1']},safety_metrics=['no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound'])
    environment=MockEnvironment();environment.fail_collection=fail_collection
    module.NativeEnvironment=lambda *args,**kwargs:environment
    module.read_protocol=lambda _:protocol
    module.require_dispatch=lambda:None
    module.require_scene=lambda *args,**kwargs:None
    argv=['private_cli','--protocol','CPU','--output',str(root),'--arm','P0','--seed','42',
          '--qualification','--rounds','1','--B','2','--G','4']
    caught=None
    with patch.object(sys,'argv',argv):
        try:module.main()
        except BaseException as error:caught=error
    if not isinstance(caught,expected_exception):raise AssertionError(f'Original exception lost: {type(caught).__name__}: {caught}')
    state,payload=load_state(root,protocol['sha256'],restore_rng=False)
    if state['updates']!=0 or payload['optimizer']['state']:raise AssertionError('Failed inner published optimizer state')
    expected_phase='READY' if fail_collection else 'ROLLOUT_COMMITTED'
    if state['phase']!=expected_phase:raise AssertionError('Uncommitted failure advanced latest')
    evidence=list((root/'failure_evidence').glob('failure_*.json'))
    if len(evidence)!=1:raise AssertionError('Original failure JSON was not preserved')
    value=json.loads(evidence[0].read_text())
    if not value['failure_handler_does_not_publish_latest']:raise AssertionError('Failure masquerades as a resume boundary')
    if fail_collection:
        if payload['batch'] is not None or len(list((root/'failure_evidence').glob('collection_*.pt')))!=1:
            raise AssertionError('Partial collected scene escaped as committed batch')
    elif payload['batch'] is None or state['next_inner']!=0:
        raise AssertionError('Valid pending fixed behavior batch lost on first-inner failure')
    return dict(original_exception=type(caught).__name__,latest_phase=state['phase'],updates=0,
                old_chain_kept=payload['batch'] is not None,failure_evidence_preserved=True)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',required=True);a=p.parse_args()
    source=Path(__file__).with_name('preservation_train.py')
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp)
        inner=run_fault(source,root/'first_inner',expected_exception=torch.OutOfMemoryError)
        collecting=run_fault(source,root/'collecting',fail_collection=True,expected_exception=RuntimeError)
        # Minimum independent reproduction of the pre-fix secondary exception.
        from starVLA.rl.flow_grpo.transactions import atomic_json
        try:atomic_json(root/'missing_parent'/'failure.json',{'original':'OOM'})
        except FileNotFoundError:previous=True
        else:raise AssertionError('Historical missing-parent reproduction failed')
    result=dict(status='PASS',qualification_only=True,no_GPU_or_real_model=True,
                pre_fix_missing_parent_reproduced=previous,first_inner_failure=inner,collection_failure=collecting,
                source_sha256=file_sha(source),patch='Create failure_evidence before model construction; no actor/sampler/loss/optimizer changes')
    result['sha256']=fingerprint(result);out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise FileExistsError('Preserve earlier qualification output')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
