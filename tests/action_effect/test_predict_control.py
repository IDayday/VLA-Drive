"""Deterministic physical controls and thin registration, no real scoring."""
from copy import deepcopy
from pathlib import Path
import json
import inspect
import numpy as np
import pytest
import torch
from tools.action_effect import predict_control as c
from tools.action_effect.common import artifact,digest,read_json,write_json


def base():return np.column_stack((np.arange(1,9,dtype=np.float64)*2.,np.zeros((8,2))))


def noise(seed=2):return np.random.default_rng(seed).normal(size=(8,4)).astype(np.float32)


def test_zero_is_exact_and_fixed_brake_is_only_late_half_metre():
    x=base();out=c.apply_control_coefficients(x,np.zeros(8))
    assert out['trajectory'].tobytes()==x.tobytes() and out['geometry_accepted']
    fixed=c.generate_control(x,noise(),control='FixedLateBrake')
    assert np.flatnonzero(fixed['coefficients']).tolist()==[2] and fixed['coefficients'][2]==-.5
    assert np.array_equal(fixed['trajectory'][:4],x[:4])
    assert fixed['trajectory'][-1,0]==x[-1,0]-.5
    assert fixed['geometry_screen']['initial_velocity_observed'] is False
    assert fixed['geometry_screen']['is_safety_label'] is False


def test_random_uses_one_of_eight_primitives_without_global_rng_or_input_mutation():
    x=base();eps=noise();before=(x.copy(),eps.copy(),np.random.get_state(),torch.get_rng_state().clone())
    a=c.generate_control(x,eps,control='RandomLocal',seed=42)
    b=c.generate_control(x,eps.copy(),control='RandomLocal',seed=42)
    assert a['primitive_index']==b['primitive_index']
    assert a['trajectory'].tobytes()==b['trajectory'].tobytes()
    assert len(np.flatnonzero(a['coefficients']))==1 and a['deploy_candidate_count']==1 and not a['oracle_selection_used']
    assert any(np.array_equal(a['coefficients'],v) for v in c.primitive_coefficients())
    assert np.array_equal(x,before[0]) and np.array_equal(eps,before[1])
    assert all(np.array_equal(a,b) for a,b in zip(np.random.get_state(),before[2]))
    assert torch.equal(torch.get_rng_state(),before[3])
    indices={c.generate_control(x,noise(i),control='RandomLocal')['primitive_index'] for i in range(128)}
    assert indices==set(range(8))


def test_no_scene_label_or_score_inputs_and_fixed_control_ignores_noise_choice():
    assert set(inspect.signature(c.generate_control).parameters)=={'base_physical','initial_noise','control','seed'}
    for key in ('scene_id','GT','score','feedback','metric_cache'):
        with pytest.raises(TypeError):c.generate_control(base(),noise(),control='RandomLocal',**{key:object()})
    a=c.generate_control(base(),noise(1),control='FixedLateBrake')
    b=c.generate_control(base(),noise(9),control='FixedLateBrake')
    assert a['trajectory'].tobytes()==b['trajectory'].tobytes()


def test_invalid_geometry_falls_back_to_original_without_native_safety_claim():
    x=base();x[:,0]*=100.
    out=c.generate_control(x,noise(),control='FixedLateBrake')
    assert out['fallback_to_Base'] and not out['geometry_accepted']
    assert out['trajectory'].tobytes()==x.tobytes() and not out['native_safety_verified']
    assert np.isfinite(out['trajectory']).all()
    with pytest.raises(ValueError,match='finite physical'):c.generate_control(x*np.nan,noise(),control='RandomLocal')
    with pytest.raises(ValueError,match='FP32'):c.generate_control(base(),noise().astype(np.float64),control='RandomLocal')
    with pytest.raises(ValueError,match='FP32'):c.generate_control(base(),noise()*np.nan,control='RandomLocal')


def fake_context(tmp_path,monkeypatch):
    from tools.action_effect import evaluation_context_data as data
    from tools.st_world_policy import base_binding
    context=tmp_path/'CONTEXT.json';write_json(context,{'fixture':'small explicit unit context, not DEV1696'})
    binding=tmp_path/'BASE_BINDING.json';write_json(binding,{'fixture':'not learned weights'})
    source={'schema':'action_effect_evaluation_registration_v1','split':'dev','for_training':False,
        'base_checkpoint_binding_hash':'base','base_checkpoint':{'sha256':'originalS0'},'base_binding':artifact(binding),
        'numeric_runtime_hash':'fp32-original','score_protocol_hash':'official-v1','spec':{'score_protocol_hash':'official-v1'}}
    source['identity']=digest(source);write_json(tmp_path/'SOURCE_REG.json',source)
    rows={};records={}
    for i in range(2):
        sid='scene'+str(i);rows[sid]={'scene_id':sid,'source_log':'log'+str(i),'role':'dev_report','observation_hash':'obs'+str(i)}
        records[sid]={'physical':torch.tensor(base(),dtype=torch.float32),'noise':torch.tensor(noise(i)),
                      'GT':object(),'feedback':object()} # These are never forwarded to the generator.
    class Reader:
        def __init__(self,*args,**kwargs):
            self.binding=source;self.manifest={'registration':artifact(tmp_path/'SOURCE_REG.json')};self.rows=rows
        def __getitem__(self,sid):return records[sid]
    monkeypatch.setattr(data,'EvaluationContextReader',Reader);monkeypatch.setattr(data,'FULL_SCENES',2)
    monkeypatch.setattr(base_binding,'read_binding',lambda path:{'binding_sha256':'base'})
    return {'seed':42,'cohort_size':1024,'base_binding':str(binding)},context,Reader,records


def test_real_baseline_registration_is_reused_control_identity_is_immutable(tmp_path,monkeypatch):
    cfg,context,Reader,records=fake_context(tmp_path,monkeypatch)
    output=tmp_path/'control'
    result=c.predict(cfg,context_manifest=context,output=output,control='RandomLocal',role='dev_report',max_samples=2)
    assert result['new_optimizer_updates']==result['new_VLM_calls']==result['new_simulation_calls']==0
    m=read_json(output/'PREDICTIONS.json');reg=read_json(output/'PREDICTION_REGISTRATION.json')
    assert reg['checkpoint']==artifact(output/'GENERATOR_SPEC.json')
    assert reg['checkpoint_kind']=='engineering_control_generator_spec_not_trained_checkpoint'
    assert reg['source_s0_predictions']==artifact(output/'source_s0/PREDICTIONS.json')
    assert all(r['deploy_candidate_count']==1 and not r['oracle_selection_used'] for r in m['entries'])
    old=(output/'PREDICTIONS.json').read_bytes()
    for row in records.values():row['GT']='different';row['feedback']='different'
    c.predict(cfg,context_manifest=context,output=output,control='RandomLocal',role='dev_report',max_samples=2,resume=True)
    assert (output/'PREDICTIONS.json').read_bytes()==old
    with pytest.raises(ValueError,match='explicit --resume'):
        c.predict(cfg,context_manifest=context,output=output,control='RandomLocal',role='dev_report',max_samples=2)
    with pytest.raises(ValueError,match='Immutable'):
        c.predict(cfg,context_manifest=context,output=output,control='FixedLateBrake',role='dev_report',max_samples=2,resume=True)


def test_wrong_checkpoint_context_or_base_physical_rejected(tmp_path,monkeypatch):
    cfg,context,Reader,records=fake_context(tmp_path,monkeypatch)
    output=tmp_path/'control'
    c.predict(cfg,context_manifest=context,output=output,control='FixedLateBrake',role='dev_report',max_samples=2)
    source=output/'source_s0/PREDICTIONS.json'
    other=tmp_path/'OTHER_CONTEXT.json';write_json(other,{'different':'context'})
    with pytest.raises(ValueError,match='exact completed original S0'):
        c.read_source_predictions(source,reader=Reader(),context_manifest=other,role='dev_report',max_samples=2)
    manifest=read_json(source);registration=read_json(Path(manifest['registration']['path']))
    registration['checkpoint']={'path':'/foreign/Base.json','sha256':'wrong'}
    registration['identity']=digest({k:v for k,v in registration.items() if k!='identity'})
    write_json(tmp_path/'WRONG_REG.json',registration)
    manifest['registration']=artifact(tmp_path/'WRONG_REG.json');manifest['registration_identity']=registration['identity']
    manifest['identity']=digest({k:v for k,v in manifest.items() if k!='identity'})
    write_json(tmp_path/'WRONG_PREDICTIONS.json',manifest)
    with pytest.raises(ValueError,match='exact completed original S0'):
        c.read_source_predictions(tmp_path/'WRONG_PREDICTIONS.json',reader=Reader(),context_manifest=context,role='dev_report',max_samples=2)
    records['scene0']['physical'][0,0]+=.1
    with pytest.raises(ValueError,match='does not equal'):
        c.predict(cfg,context_manifest=context,output=tmp_path/'wrong',control='FixedLateBrake',role='dev_report',
                  max_samples=2,s0_predictions=source)


def test_cli_dry_run_never_calls_predict_or_scoring(monkeypatch,capsys):
    monkeypatch.setattr(c,'predict',lambda *a,**kw:pytest.fail('dry run called predict'))
    argv=['--config','configs/action_effect/shared_budget_v2.yaml','--context-manifest','/future/DEV.json',
          '--output','/future/control','--control','RandomLocal','--role','dev_report','--max-samples','1696','--dry-run']
    assert c.main(argv)==0 and json.loads(capsys.readouterr().out)['status']=='NOT_RUN'
    assert c.main(argv[:-2]+['1','--dry-run'])==1
    assert 'fixed full population' in capsys.readouterr().out


def test_no_hidden_multi_candidate_search(monkeypatch):
    calls=[];original=c.reconstruct
    def one(*a,**kw):calls.append(1);return original(*a,**kw)
    monkeypatch.setattr(c,'reconstruct',one)
    c.generate_control(base(),noise(),control='RandomLocal')
    assert len(calls)==1
