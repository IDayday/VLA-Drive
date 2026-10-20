"""Single-factor CPU contracts; no real G1/policy improvement is inferred."""
from copy import deepcopy
from pathlib import Path
import torch
import pytest
from tools.action_effect import train_effect_model as train
from tools.action_effect.training import run_training
from starVLA.model.modules.action_effect.effect_model import ActionEffectModel
from starVLA.model.modules.action_effect.effect_losses import effect_loss_contributions,reduce_contributions
from tests.action_effect import test_model_effect as fixture


def test_only_registered_two_weights_and_delta_ablation_namespace(tmp_path):
    default=deepcopy(train.WEIGHTS)
    assert train.loss_weights()==default
    zero=train.loss_weights(0.)
    assert zero=={**default,'temporal':0.} and train.WEIGHTS==default
    zero['ranking']=999.
    assert train.loss_weights(0.)['ranking']==.2 and train.WEIGHTS==default
    for bad in (True,False,.1,-.2,float('nan'),float('inf'),'0',torch.tensor(.2)):
        with pytest.raises(ValueError,match='exactly'):train.loss_weights(bad)
    with pytest.raises(ValueError,match='only for delta_feedback'):train.loss_weights(0.,arm='terminal_pdms')
    cfg={'artifact_root':str(tmp_path)}
    assert train.training_output(cfg,'delta_feedback',42,None,train.loss_weights())==tmp_path/'g1/delta_feedback/seed_42'
    assert train.training_output(cfg,'delta_feedback',42,'',train.loss_weights())==tmp_path/'g1/delta_feedback/seed_42'
    assert train.training_output(cfg,'delta_feedback',42,None,train.loss_weights(0.))==tmp_path/'g1_ablation/no_temporal/seed_42'


def test_default_value_terms_and_gradients_are_exact_original_recipe():
    model=ActionEffectModel(fixture.cfg(),fixture.spec())
    context,before,after=fixture.batch();targets=fixture.supervision(model,len(before))
    prediction=model(context,before,after)
    contributions=effect_loss_contributions(prediction,**targets)
    original=reduce_contributions(contributions,train.WEIGHTS)
    explicit=reduce_contributions(contributions,train.loss_weights(.2))
    assert torch.equal(original.total,explicit.total)
    parameters=tuple(model.parameters())
    old=torch.autograd.grad(original.total,parameters,retain_graph=True,allow_unused=True)
    new=torch.autograd.grad(explicit.total,parameters,retain_graph=True,allow_unused=True)
    for a,b in zip(old,new):
        assert a is b is None or a is not None and b is not None and torch.equal(a,b)
    assert all(torch.equal(original.terms[k],explicit.terms[k]) for k in original.terms)


def test_only_dense_temporal_gradient_removed_all_eleven_value_channels_retained():
    model=ActionEffectModel(fixture.cfg(),fixture.spec())
    context,before,after=fixture.batch();targets=fixture.supervision(model,len(before))
    prediction=model(context,before,after)
    contributions=effect_loss_contributions(prediction,**targets)
    full=reduce_contributions(contributions,train.loss_weights(.2))
    ablated=reduce_contributions(contributions,train.loss_weights(0.))
    assert all(torch.equal(full.terms[k],ablated.terms[k]) for k in full.terms)
    assert full.statistics==ablated.statistics
    params=tuple(model.parameters())
    non_temporal=sum(v*train.WEIGHTS[k] for k,v in full.terms.items() if k!='temporal')
    observed=torch.autograd.grad(ablated.total,params,retain_graph=True,allow_unused=True)
    expected=torch.autograd.grad(non_temporal,params,retain_graph=True,allow_unused=True)
    for a,b in zip(observed,expected):
        if a is None or b is None:
            assert (a is None or torch.count_nonzero(a)==0) and (b is None or torch.count_nonzero(b)==0)
        else:assert torch.equal(a,b)
    # First-event times remain part of the eleven-channel value objective.
    value_grad,=torch.autograd.grad(ablated.terms['value'],model.mean_head.weight,retain_graph=True)
    assert len(model.label_spec.names)==11 and torch.count_nonzero(value_grad[-3:])>0
    temporal_grad,=torch.autograd.grad(ablated.total,model.temporal_head.weight)
    assert torch.equal(temporal_grad,torch.zeros_like(temporal_grad))
    # Same model inventory, target arrays and masks; this does not drop time labels.
    assert contributions['temporal'].denominator>0 and targets['temporal_valid'].all()


def test_same_run_rejects_weight_change_before_next_batch(tmp_path):
    cfg={'training':{'learning_rate':1e-3,'weight_decay':.01,'warmup_updates':1,'batch_scenes':2,
        'microbatch_scenes':1,'gradient_clip':1.,'validation_every':2,'max_updates':4}}
    source=deepcopy(torch.nn.Linear(1,1));seen=[]
    def step(model,indices,n):
        seen.append(list(indices));loss=model(torch.tensor(indices).float()[:,None]).square().mean();loss.backward();return {'value':float(loss)}
    identity={'source_commit':'same_cpu_fixture','arm':'delta_feedback','loss_weights':train.loss_weights(.2)}
    first=run_training(source,output=tmp_path,identity=identity,dataset_size=7,step_fn=step,cfg=cfg,
        device='cpu',seed=42,max_updates=4,stop_after=2)
    assert first['optimizer_step']==2 and len(seen)==2
    changed={**identity,'loss_weights':train.loss_weights(0.)}
    with pytest.raises(ValueError,match='Immutable'):
        run_training(deepcopy(source),output=tmp_path,identity=changed,dataset_size=7,step_fn=step,cfg=cfg,
            device='cpu',seed=42,max_updates=4,resume=True)
    assert len(seen)==2


def test_reporting_keeps_first_event_metrics_but_marks_untrained_time_grid():
    from tests.action_effect import test_g1_comparison as report_fixture
    from tools.action_effect.g1_metrics import summarize_records
    rows=[report_fixture.record(time_observed=True)]
    kwargs=dict(names=report_fixture.NAMES,units=report_fixture.UNITS,scales=report_fixture.SCALES,
        requested_scenes=report_fixture.requested('a'),mode='delta_feedback')
    full=summarize_records(rows,**kwargs)
    no=summarize_records(rows,**kwargs,temporal_supervised=False)
    assert full['temporal_supervised_in_this_arm'] and not no['temporal_supervised_in_this_arm']
    assert no['label_metrics']==full['label_metrics']
    assert all(no['label_metrics'][k]['supervised_in_this_arm'] for k in report_fixture.NAMES)
    assert {k:v for k,v in full.items() if k!='temporal_supervised_in_this_arm'}=={
        k:v for k,v in no.items() if k!='temporal_supervised_in_this_arm'}


def test_ablation_cannot_silently_replace_the_primary_delta_control():
    from tools.action_effect.compare_effect_models import require_primary_loss_recipe
    require_primary_loss_recipe({'loss_weights':train.loss_weights()})
    with pytest.raises(ValueError,match='compare the temporal ablation separately'):
        require_primary_loss_recipe({'loss_weights':train.loss_weights(0.)})


def test_two_explicit_missing_evaluations_remain_pending(tmp_path):
    from tools.action_effect.compare_temporal_ablation import compare
    result=compare(tmp_path/'full_not_ready',tmp_path/'no_temporal_not_ready',step=1000,
        output=tmp_path/'comparison',bootstrap=10)
    assert result['status']=='NOT_RUN' and result['completion']=='PENDING'
    assert result['scientific_conclusion']=='UNTESTED'
    assert result['policy_temporal_credit_claim']=='NOT_ESTABLISHED'
    assert result['training_or_simulator_started'] is False


def matched_fixtures(tmp_path):
    from tests.action_effect import test_g1_comparison as report_fixture
    from tools.action_effect.common import artifact
    path=tmp_path/'checkpoint';path.write_bytes(b'unit stub, not a real trained state')
    data={k:k for k in ('base_binding_hash','context_artifact','data_artifact','label_spec','heldout_pair_identities')}
    data.update(arm='delta_feedback',optimizer_step=1000,requested_scenes=report_fixture.requested('a'),checkpoint=artifact(path))
    metrics=report_fixture.summary([report_fixture.record()])
    sample={'identity':data,'metrics':metrics,'training':{'private_parameters':42}}
    state={'sampler':torch.Generator().manual_seed(143).get_state(),
        'stateful':{'pair_rng':torch.Generator().manual_seed(244).get_state()},'consumed_scenes':32000}
    identity={'source_commit':'full','arm':'delta_feedback','seed':42,'model_config':{'same':'capacity'},
        'data_artifact':'same','recipe':{'batch_scenes':32,'microbatch_scenes':8},'max_updates':1000,
        'loss_weights':train.loss_weights(.2),'runtime':{'device':'cuda:0','gpu_name':'same GPU','dtype':'float32'}}
    other=deepcopy(identity);other.update(source_commit='ablation',loss_weights=train.loss_weights(0.))
    other['runtime']['device']='cuda:1'
    return sample,deepcopy(sample),[(identity,state,{}),(other,deepcopy(state),{})]


def test_pair_comparison_accepts_one_change_and_rejects_other_loss_or_sampling_changes(tmp_path,monkeypatch):
    from tools.action_effect import compare_temporal_ablation as compare
    first,second,states=matched_fixtures(tmp_path)
    def check():
        it=iter(states);monkeypatch.setattr(compare,'checkpoint_state',lambda _:next(it))
        return compare._matched(first,second,step=1000)
    assert check()
    states[1][0]['loss_weights']['ranking']=0.
    with pytest.raises(ValueError,match='Only the dense'):check()
    states[1][0]['loss_weights']['ranking']=.2
    states[1][0]['recipe']['microbatch_scenes']=4
    with pytest.raises(ValueError,match='Unmatched model'):check()
    states[1][0]['recipe']['microbatch_scenes']=8
    states[1][1]['stateful']['pair_rng']=torch.Generator().manual_seed(245).get_state()
    with pytest.raises(ValueError,match='pair sampler'):check()
