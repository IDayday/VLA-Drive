import copy
import pytest

from tools.foresight.lock_navtest import validate_lock
from tools.foresight.score_pdms import identity_hash
from tools.full_foresight.lock_checkpoint_navtest import validate_probe


def fixture():
    identity={k:'same' for k in ('source_sha','data','ego','selected_index_hash','scene_count','global_batch','updates','schedule','precision','world_size','micro_batch')}
    identity.update(scope='formal',schema='ddp_full_foresight_student_v1',config={'framework':{'action_model':{'num_inference_timesteps':10}}})
    checkpoints=[dict(sha256=c,arm=c,training_seed=42,scope='formal',completed=25000) for c in ('C0','C1')]
    report=dict(schema='ddp_full_foresight_planning_v1',valid=True,diagnostic=False,split='dev',
                groups={f'{c["arm"]}_train42':dict(valid=True,checkpoint=c) for c in checkpoints})
    return identity,checkpoints,report


def test_fixed_milestone_does_not_require_completed_training():
    identity,checkpoints,report=fixture()
    records,steps,common=validate_probe([(identity,c) for c in checkpoints],report,25000)
    assert records==checkpoints and steps==10
    assert common['updates']=='same'


@pytest.mark.parametrize('fault',['checkpoint','data','development'])
def test_milestone_pair_rejects_changed_identities(fault):
    identity,checkpoints,report=fixture();other=copy.deepcopy(identity)
    if fault=='checkpoint':checkpoints[1]['completed']=10000
    if fault=='data':other['data']='other'
    if fault=='development':report['groups']['C1_train42']['checkpoint']={'sha256':'other'}
    with pytest.raises(ValueError):validate_probe([(identity,checkpoints[0]),(other,checkpoints[1])],report,25000)


def test_probe_lock_binds_source_checkpoint_population_seed():
    _,checkpoints,_=fixture();checkpoint=checkpoints[0]
    current=dict(identity='inputs',index_sha256='index')
    lock=dict(schema='foresight_navtest_checkpoint_probe_lock_v1',evaluation_purpose='user_requested_fixed_checkpoint',
              requested_update=25000,precision='FP32',candidates_per_scene=1,learned_scorer=None,
              log_count=136,scene_count=12146,checkpoints=['C0'],checkpoint_records={'C0':checkpoint},
              evaluation_source_sha='code',current_data_identity='inputs',current_index_identity='index',
              sampling_seeds=[42],inference_steps=10)
    lock['identity']=identity_hash(lock)
    validate_lock(lock,checkpoint,'code',current,12146,42,10)
    for source,n,seed in [('changed',12146,42),('code',10,42),('code',12146,43)]:
        with pytest.raises(ValueError):validate_lock(lock,checkpoint,source,current,n,seed,10)
    bad=copy.deepcopy(lock);bad['requested_update']=10000;bad['identity']=identity_hash({k:v for k,v in bad.items() if k!='identity'})
    with pytest.raises(ValueError):validate_lock(bad,checkpoint,'code',current,12146,42,10)


@pytest.mark.parametrize('fault',[None,'run_identity','training_source_sha','newer'])
def test_prior_development_evidence_is_explicit_and_same_run(fault):
    identity,checkpoints,report=fixture()
    for c in checkpoints:
        c.update(run_identity=c['arm']+'-run',training_source_sha='source',model_class='DDPFullForesight')
    later=copy.deepcopy(checkpoints)
    for c in later:c.update(completed=31600,sha256=c['sha256']+'-later',tag='periodic_031600')
    with pytest.raises(ValueError):validate_probe([(identity,c) for c in later],report,31600)
    if fault in ('run_identity','training_source_sha'):later[0][fault]='foreign'
    if fault=='newer':report['groups']['C0_train42']['checkpoint']['completed']=32000
    if fault:
        with pytest.raises(ValueError):validate_probe([(identity,c) for c in later],report,31600,True)
    else:
        records,_,_=validate_probe([(identity,c) for c in later],report,31600,True)
        assert records==later
