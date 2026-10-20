"""Independent affected-scope review; small CPU fixtures are not real training."""
from copy import deepcopy
import subprocess
import sys
import types

import pytest
import torch

from tools.action_effect import train_effect_model as current
from tools.action_effect.common import digest
from tools.action_effect.training import run_training
from starVLA.model.modules.action_effect.effect_model import ActionEffectModel, MODES
from starVLA.model.modules.action_effect.effect_losses import effect_loss_contributions,reduce_contributions
from tests.action_effect import test_model_effect as fixture


@pytest.fixture(scope='module')
def parent_module():
    name='tools.action_effect._review_temporal_parent_b7cc'
    text=subprocess.check_output(['git','show','d80f658e^:tools/action_effect/train_effect_model.py'],text=True)
    module=types.ModuleType(name);module.__package__='tools.action_effect';sys.modules[name]=module
    exec(compile(text,'train_effect_model.py@d80-parent','exec'),module.__dict__)
    return module


@pytest.mark.parametrize('mode',MODES)
def test_default_all_six_original_loss_gradients_and_train_identity(parent_module,tmp_path,monkeypatch,mode):
    torch.set_num_threads(2)
    # Compare with the actual pinned parent's constants, not a second reference
    # to the current mapping. Same source identity is intentional to isolate the
    # objective change; real source revisions remain disclosed in run identity.
    assert parent_module.WEIGHTS==current.loss_weights(.2,arm=mode)
    model=ActionEffectModel(fixture.cfg(mode),fixture.spec())
    context,before,after=fixture.batch();targets=fixture.supervision(model,len(before))
    kwargs={'context_permutation':torch.tensor([2,0,1])} if mode=='shuffled_context' else {}
    contributions=effect_loss_contributions(model(context,before,after,**kwargs),**targets)
    prior=reduce_contributions(contributions,parent_module.WEIGHTS)
    now=reduce_contributions(contributions,current.loss_weights(.2,arm=mode))
    assert torch.equal(prior.total,now.total) and prior.statistics==now.statistics
    params=tuple(model.parameters())
    left=torch.autograd.grad(prior.total,params,retain_graph=True,allow_unused=True)
    right=torch.autograd.grad(now.total,params,allow_unused=True)
    assert all((a is None and b is None) or (a is not None and b is not None and torch.equal(a,b)) for a,b in zip(left,right))

    context_file=tmp_path/'context.json';context_file.write_text('{}')
    pair_file=tmp_path/'pairs.json';pair_file.write_text('{}')
    cfg={'artifact_root':str(tmp_path),'training':{'hidden':8,'layers':1,'heads':2,'microbatch_scenes':1}}
    class Data:
        spec=fixture.spec();shuffle_map={'a':'b'};scenes=['one']
        def __init__(self,*args):pass
    def capture(model,**kwargs):
        return {'identity':kwargs['identity'],'output':str(kwargs['output']),
                'private_state':fixture.state_hash(model),'sampler_rng':kwargs['stateful']['pair_rng'].get_state()}
    outputs=[]
    for module in (parent_module,current):
        monkeypatch.setattr(module,'ContextReader',lambda *args:types.SimpleNamespace(binding={'base_checkpoint_binding_hash':'bound-own-base'}))
        monkeypatch.setattr(module,'PairData',Data)
        monkeypatch.setattr(module,'run_training',capture)
        monkeypatch.setattr(module.subprocess,'check_output',lambda *args,**kwargs:'pinned-same-source\n')
        outputs.append(module.train(cfg,context_manifest=context_file,pair_dataset=pair_file,arm=mode,
                                    device='cpu',seed=42,max_updates=1000))
    a,b=outputs
    assert a['identity']==b['identity'] and a['output']==b['output'] and a['private_state']==b['private_state']
    assert torch.equal(a['sampler_rng'],b['sampler_rng'])


def test_resume_zero_to_default_is_rejected_before_batch_consumption(tmp_path):
    cfg={'training':{'learning_rate':1e-3,'weight_decay':0.,'warmup_updates':1,'batch_scenes':2,
        'microbatch_scenes':1,'gradient_clip':1.,'validation_every':4,'max_updates':4}}
    seen=[]
    def step(model,indices,n):
        seen.append(indices);loss=model(torch.tensor(indices,dtype=torch.float32)[:,None]).square().mean()
        loss.backward();return {'loss':float(loss)}
    identity={'source_commit':'independent_fixture','arm':'delta_feedback','loss_weights':current.loss_weights(0.)}
    first=run_training(torch.nn.Linear(1,1),output=tmp_path,identity=identity,dataset_size=7,step_fn=step,
        cfg=cfg,device='cpu',seed=42,max_updates=4,stop_after=2)
    assert first['optimizer_step']==2
    with pytest.raises(ValueError,match='Immutable'):
        run_training(torch.nn.Linear(1,1),output=tmp_path,identity={**identity,'loss_weights':current.loss_weights(.2)},
            dataset_size=7,step_fn=step,cfg=cfg,device='cpu',seed=42,max_updates=4,resume=True)
    assert len(seen)==2


@pytest.mark.parametrize('mutation',('data','event_weight','batch','exposure','backend'))
def test_temporal_comparator_refuses_other_unmatched_axes(tmp_path,monkeypatch,mutation):
    from tests.action_effect.test_temporal_ablation import matched_fixtures
    from tools.action_effect import compare_temporal_ablation as compare
    a,b,states=matched_fixtures(tmp_path)
    if mutation=='data':states[1][0]['data_artifact']='foreign'
    elif mutation=='event_weight':states[1][0]['loss_weights']['events']=0.
    elif mutation=='batch':states[1][0]['recipe']['batch_scenes']=64
    elif mutation=='exposure':states[1][1]['consumed_scenes']+=1
    else:states[1][0]['runtime']['gpu_name']='different hardware'
    it=iter(states);monkeypatch.setattr(compare,'checkpoint_state',lambda *_:next(it))
    with pytest.raises(ValueError):compare._matched(a,b,step=1000)
