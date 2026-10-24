"""CPU saved-initial-batch acceptance for L1_CENTER_V1; never dispatches GPU."""
import argparse
import copy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import tempfile
import sys
from unittest.mock import patch
import torch
from starVLA.rl.qwen_flow_exploration.protocol import ROOT,read_protocol
from starVLA.rl.qwen_flow_exploration.checkpoint import load_state,save_state
from starVLA.rl.qwen_flow_exploration.latent.policy import LatentNoisePolicy
from starVLA.rl.qwen_flow_exploration.latent.normalization import CENTER_ARM,load_center_normalizer,policy_from_identity,assert_center_unchanged,require_center_lock
from starVLA.rl.qwen_flow_exploration.latent.gaussian import joint_log_prob,sample_roots
from starVLA.rl.qwen_flow_exploration.latent.training import train_collected_batch
from starVLA.rl.qwen_sourceflow.contracts import fingerprint,file_sha
from starVLA.rl.simwam_recipe.recovery_state import tensor_digest
from starVLA.rl.flow_grpo.transactions import atomic_json


def actual_cli_resume(protocol,normalizer,features):
    """Actual CLI/factory/transactions; explicitly tiny model and mock scalar."""
    from scripts.analysis.qwen_flow_exploration import latent_train
    from starVLA.rl.qwen_flow_exploration.protocol import Ledger
    from starVLA.rl.qwen_flow_exploration.latent.frozen import freeze_generator
    from starVLA.model.modules.qwen_sourceflow.source_flow_policy import SourceFlowPolicy
    from starVLA.model.modules.qwen_sourceflow.vendor.action_dit import ActionDiT
    from starVLA.model.modules.qwen_sourceflow.cached_action_runner import expand_condition
    generator=torch.nn.Module()
    torch.manual_seed(52)
    generator.policy=SourceFlowPolicy(ActionDiT(action_dim=3,hidden_dim=32,ffn_dim=64,text_dim=40,
        freq_dim=16,num_heads=2,attn_head_dim=12,num_layers=3,eps=1e-6,use_gradient_checkpointing=False))
    freeze_generator(generator)
    class Environment:
        def __init__(self,p,**kwargs):
            self.protocol=p;self.device=torch.device('cpu');self.model=copy.deepcopy(generator)
            self.experiment_id=kwargs['experiment_id'];self.ledger=Ledger(kwargs['ledger_path'])
        def condition(self,token,split):
            assert split=='TRAIN' and token in self.protocol['splits']['TRAIN']
            row=features[self.protocol['splits']['TRAIN'].index(token)%len(features)]
            return dict(context=row[:4096].reshape(1,1,4096),context_mask=torch.ones(1,1,dtype=torch.bool),
                        visual_mask=torch.ones(1,1,dtype=torch.bool),
                        visual_kv=[dict(k=row[4096:].reshape(1,1,3072),v=row[4096:].reshape(1,1,3072)) for _ in range(3)])
        def set_behavior_identity(self,value):self.behavior=value
        def sample_ode(self,c,roots):
            tiny=dict(context=c['context'][...,:40],context_mask=c['context_mask'],visual_mask=c['visual_mask'],
                      visual_kv=[dict(k=v['k'][...,:24],v=v['v'][...,:24]) for v in c['visual_kv']])
            return torch.cat([self.model.policy.sample_ode(expand_condition(tiny,len(z)),z) for z in roots.split(4)])
        def score_actions(self,token,actions,split,**kwargs):
            return [dict(status='OK',raw_reward=float(x),qualification_reward='CPU_MOCK_NOT_PDMS')
                    for x in torch.sigmoid(-(actions-.4).square().mean((-1,-2)))]
        def close(self):self.ledger.save()
    with tempfile.TemporaryDirectory() as folder:
        folder=Path(folder);p=folder/'protocol.json';p.write_text(json.dumps(protocol))
        def run(path,resume=False):
            argv=['latent_train','--protocol',str(p),'--output',str(path),'--arm',CENTER_ARM,
                  '--center-normalizer',str(normalizer),'--seed','42','--qualification','--limit','1']
            if resume:argv+=['--resume',str(path)]
            with patch.object(latent_train,'NativeEnvironment',Environment),patch.object(sys,'argv',argv):latent_train.main()
        continuous,interrupted=folder/'continuous',folder/'interrupted';run(continuous)
        original_save=latent_train.save_state
        class Stop(Exception):pass
        def interrupted_save(path,state,payload,sha):
            checkpoint=original_save(path,state,payload,sha)
            if state['updates']==1:raise Stop()
            return checkpoint
        with patch.object(latent_train,'save_state',interrupted_save):
            try:run(interrupted)
            except Stop:pass
            else:raise AssertionError('Actual CLI interruption was not exercised')
        run(interrupted,True)
        left,lp=load_state(continuous,protocol['sha256'],restore_rng=False)
        right,rp=load_state(interrupted,protocol['sha256'],restore_rng=False)
        assert left==right and left['updates']==4 and left['arm']==CENTER_ARM
        assert lp['history']==rp['history']
        for key in ['policy','optimizer','scheduler','root_generator_state']:
            assert tensor_digest(lp[key])==tensor_digest(rp[key]),key


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--normalizer',default=str(ROOT/'qualification/latent_center_saved_data_v2/TRAIN512_feature_mean.pt'))
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if torch.cuda.is_initialized():raise RuntimeError('CPU acceptance cannot initialize CUDA')
    torch.set_num_threads(2)
    protocol=read_protocol();mean,receipt=load_center_normalizer(args.normalizer,protocol)
    run=ROOT/'runs/L1_seed42'
    _,initial=load_state(next((run/'checkpoints').glob('round0000_update000000_inner0_*')),protocol['sha256'],restore_rng=False)
    _,sampled=load_state(next((run/'checkpoints').glob('round0001_update000000_inner0_*')),protocol['sha256'],restore_rng=False)
    _,terminal=load_state(run,protocol['sha256'],restore_rng=False)
    original=sampled['batch'];features=original.features;original_digest=original.digest();checks=[]
    # Load actual immutable old class, not another copy of the changed forward.
    old_path=ROOT/'snapshots/public_screen_v1/starVLA/rl/qwen_flow_exploration/latent/policy.py'
    spec=importlib.util.spec_from_file_location('starVLA.rl.qwen_flow_exploration.latent._immutable97_policy',old_path)
    oldmodule=importlib.util.module_from_spec(spec);spec.loader.exec_module(oldmodule)
    for state in [initial['policy'],terminal['policy']]:
        old=oldmodule.LatentNoisePolicy(7168);new=LatentNoisePolicy(7168)
        old.load_state_dict(state,strict=True);new.load_state_dict(state,strict=True)
        assert list(old.state_dict())==list(new.state_dict()) and old.identity()==new.identity()
        a,b=old(features),new(features)
        assert torch.equal(a.mean,b.mean) and torch.equal(a.log_std,b.log_std)
        loss_a=joint_log_prob(a,original.roots).mean();loss_b=joint_log_prob(b,original.roots).mean()
        loss_a.backward();loss_b.backward()
        assert all(torch.equal(pa.grad,pb.grad) for pa,pb in zip(old.parameters(),new.parameters()))
    checks.append('Default uncentered class matches immutable97 actual initial/nonzero checkpoints: keys, identity, outputs and every parameter gradient')
    torch.manual_seed(42);old_init=oldmodule.LatentNoisePolicy(7168)
    torch.manual_seed(42);new_init=LatentNoisePolicy(7168,feature_center=mean,center_identity=receipt['identity'])
    assert all(torch.equal(old_init.state_dict()[key],new_init.state_dict()[key]) for key in old_init.state_dict())
    assert all(torch.equal(initial['policy'][key],new_init.state_dict()[key]) for key in initial['policy'])
    assert set(new_init.state_dict())-set(old_init.state_dict())=={'feature_center'}
    checks.append('Adding the fixed buffer consumes no RNG; seed42 initializes identical trainable tensors and only adds the mean buffer')
    identity=dict(arm=CENTER_ARM,policy_identity=new_init.identity(),center_normalizer=receipt,
                  correction_budget_lock_sha256=None,qualification=True)
    def make_policy(state=None):
        model=policy_from_identity(identity,protocol)
        model.load_state_dict(state or new_init.state_dict(),strict=True)
        assert_center_unchanged(model,identity)
        return model
    model=make_policy();q=model(features)
    assert torch.count_nonzero(q.mean)==0 and torch.count_nonzero(q.log_std)==0
    byone=[model(row[None]) for row in features]
    assert torch.equal(q.mean,torch.cat([v.mean for v in byone])) and torch.equal(q.log_std,torch.cat([v.log_std for v in byone]))
    plain=LatentNoisePolicy(7168);plain.load_state_dict(initial['policy'],strict=True)
    assert torch.equal(q.mean,plain(features-mean).mean)
    # Independent root replay uses the actual initial generator state.
    from starVLA.rl.safe_flow_improvement.contracts import seed_for
    generator=torch.Generator().manual_seed(seed_for('three_routes_latent_train',42,'root_stream'))
    root=sample_roots(model(features[:1]),16,generator)
    assert torch.equal(root,original.roots[:1])
    cpu_old=joint_log_prob(original.old_parameters,original.roots).detach()
    gap=float((cpu_old-original.old_joint_logp).abs().max())
    assert torch.equal(joint_log_prob(q,original.roots),cpu_old)
    batch=replace(copy.deepcopy(original),old_joint_logp=cpu_old,behavior_digest=tensor_digest(model.state_dict()))
    checks.append('Real mean and saved initial TRAIN batch: q0/root replay/joint density and B1 versus B16 fixed-layout outputs are identical')
    def optimizer_pair(policy):
        optimizer=torch.optim.Adam(policy.parameters(),lr=1e-4)
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda i:min(1.,(i+1)/8))
        return optimizer,scheduler
    baseline=make_policy();optimizer,scheduler=optimizer_pair(baseline)
    history=train_collected_batch(baseline,optimizer,batch,inner_epochs=4,
                                  checkpoint_callback=lambda *_:scheduler.step())
    assert history[0]['ratio_min']==history[0]['ratio_max']==1 and len(history)==4
    assert all(v is not None and torch.isfinite(v).all() for v in [p.grad for p in baseline.parameters()])
    assert_center_unchanged(baseline,identity)
    assert not baseline.feature_center.requires_grad and baseline.feature_center.grad is None
    q=baseline(features);byone=[baseline(row[None]) for row in features]
    assert torch.equal(q.mean,torch.cat([v.mean for v in byone])) and torch.equal(q.log_std,torch.cat([v.log_std for v in byone]))
    checks.append('Four real Adam steps with original roots/rewards/advantages: all actor gradients finite, mean buffer unchanged, nonzero-policy B1/B16 outputs identical')
    expected_policy=copy.deepcopy(baseline.state_dict());expected_opt=copy.deepcopy(optimizer.state_dict());expected_sched=copy.deepcopy(scheduler.state_dict())
    with tempfile.TemporaryDirectory() as temp:
        for boundary in [1,2,3]:
            current=make_policy();opt,sched=optimizer_pair(current);target=Path(temp)/str(boundary);target.mkdir()
            class Stop(Exception):pass
            def commit(inner,_,metrics):
                sched.step()
                if inner==boundary:
                    state=dict(phase='INNER_COMMITTED',round=1,updates=inner,next_inner=inner,arm=CENTER_ARM,seed=42)
                    payload=dict(identity=identity,policy=copy.deepcopy(current.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),
                                 scheduler=copy.deepcopy(sched.state_dict()),batch=batch,
                                 root_generator_state=sampled['root_generator_state'])
                    save_state(target,state,payload,protocol['sha256']);raise Stop()
            try:train_collected_batch(current,opt,batch,inner_epochs=4,checkpoint_callback=commit)
            except Stop:pass
            else:raise AssertionError('Requested inner boundary not exercised')
            state,payload=load_state(target,protocol['sha256'],restore_rng=True)
            restored=policy_from_identity(payload['identity'],protocol);restored.load_state_dict(payload['policy'],strict=True)
            ropt,rsched=optimizer_pair(restored);ropt.load_state_dict(payload['optimizer']);rsched.load_state_dict(payload['scheduler'])
            train_collected_batch(restored,ropt,payload['batch'],inner_epochs=4,start_inner=state['next_inner'],checkpoint_callback=lambda *_:rsched.step())
            assert tensor_digest(restored.state_dict())==tensor_digest(expected_policy)
            assert tensor_digest(ropt.state_dict())==tensor_digest(expected_opt) and tensor_digest(rsched.state_dict())==tensor_digest(expected_sched)
            assert payload['batch'].digest()==batch.digest();assert_center_unchanged(restored,payload['identity'])
            a=torch.Generator();a.set_state(sampled['root_generator_state']);b=torch.Generator();b.set_state(payload['root_generator_state'])
            assert torch.equal(sample_roots(restored(features[:1]),4,a),sample_roots(baseline(features[:1]),4,b))
        checks.append('Actual atomic mid-inner1/2/3 restore: centered policy factory, mean buffer, Adam, scheduler, fixed behavior batch and subsequent root RNG bitwise match')
        try:require_center_lock(None,protocol,receipt)
        except ValueError:pass
        else:raise AssertionError('Formal correction without budget lock accepted')
        checks.append('Formal CENTER without separate correction budget lock is rejected')
    actual_cli_resume(protocol,args.normalizer,features)
    checks.append('Actual centered latent_train CLI with real mean/features and tiny source ActionDiT/mock scalar: inner1 interrupt/resume is bitwise equal to continuous4steps')
    assert original.digest()==original_digest
    output=Path(args.output)
    if output.exists():raise FileExistsError(output)
    result=dict(schema='L1_CENTER_V1_CPU_acceptance_v1',status='PASS',qualification_only=True,new_GPU_tasks=0,new_rollouts=0,new_scoring_calls=0,
                real_normalizer=receipt,checks=checks,CPU_vs_saved_GPU_initial_joint_logprob_max_abs=gap,
                source_files={str(Path(__file__)):file_sha(__file__),**{str(path):file_sha(path) for path in Path('starVLA/rl/qwen_flow_exploration/latent').glob('*.py')}},
                formal_budget_dispatch_authorized=False,driving_improvement_not_tested=True)
    atomic_json(output,result);print(json.dumps(dict(status='PASS',checks=len(checks),output=str(output))))


if __name__=='__main__':main()
