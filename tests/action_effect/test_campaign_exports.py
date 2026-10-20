"""CPU controller/export contracts; synthetic fixtures do not prove scientific gain."""
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import torch
import pytest

from tools.action_effect import run_campaign as campaign
from tools.action_effect.common import artifact,digest,write_json,sha256
from tools.action_effect.export_models import source_inventory,load_bundle,predict_bundle,DenyTrainingRuntime
from tools.action_effect.evaluate_effect_flow import canonical_inference_backend


def test_campaign_dry_run_does_not_launch_jobs_and_requires_bounded_config(capsys,monkeypatch):
    monkeypatch.setattr(campaign.subprocess,'run',lambda *a,**k:pytest.fail('Dry run launched child'))
    rc=campaign.main(['--config','configs/action_effect/main.yaml','--phase','all','--dry-run',
        '--max-updates','32','--max-simulation-budget','2048'])
    result=json.loads(capsys.readouterr().out)
    assert rc==0 and result['status']=='NOT_RUN' and result['no_jobs_launched']
    assert result['phases']==list(campaign.PHASES[:-1])
    assert campaign.main(['--config','configs/action_effect/main.yaml','--phase','all'])==1
    assert 'execute-ready-stages' in capsys.readouterr().out


def test_latest_checkpoint_only_named_run_rejects_newer_corrupt_and_foreign(tmp_path):
    identity={'source_commit':'fixture'};ih=digest(identity);write_json(tmp_path/'IDENTITY.json',identity)
    for step,foreign in ((1,False),(2,False),(9,True)):
        path=tmp_path/f'step_{step:06d}.pt'
        torch.save({'run_identity':'foreign' if foreign else ih,'optimizer_step':step},path)
        write_json(path.with_suffix('.COMPLETE.json'),{'run_identity':ih,'optimizer_step':step,'checkpoint':artifact(path)})
    broken=tmp_path/'step_000099.pt';broken.write_bytes(b'partial')
    write_json(broken.with_suffix('.COMPLETE.json'),{'run_identity':ih,'optimizer_step':99,'checkpoint':artifact(broken)})
    assert campaign.latest_complete(tmp_path).name=='step_000002.pt'
    with pytest.raises(campaign.Blocked):campaign.require(tmp_path/'missing-input')


def test_aggregate_budget_counts_reserved_slots_and_rejects_expansion(tmp_path):
    a=tmp_path/'g0';b=tmp_path/'eval'
    for p,count in ((a,10),(b,3)):
        write_json(p/'calls'/'COUNTER.json',{'reserved_calls':count})
    d=object.__new__(campaign.Dispatcher);d.g0=a;d.evaluations=[b]
    d.a=SimpleNamespace(max_simulation_budget=15)
    assert campaign.used_calls([a,b])==13 and d.budget(b)==2
    reg={'max_calls':7};reg['identity']=digest(reg);write_json(b/'EVALUATION_REGISTRATION.json',reg)
    with pytest.raises(campaign.Blocked,match='ceiling exceeds'):d.budget(b)
    d.a.max_simulation_budget=13
    with pytest.raises(campaign.Blocked,match='exhausted'):d.budget(b)


def test_population_run_namespace_and_canonical_inference_flags(tmp_path):
    d=object.__new__(campaign.Dispatcher);d.root=tmp_path;d.seed=42;d.a=SimpleNamespace(max_samples=64)
    first=d.model_run('route_b','B0');d.a.max_samples=1024
    assert first!=d.model_run('route_b','B0')
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    flags=canonical_inference_backend()
    assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
    assert torch.are_deterministic_algorithms_enabled() and not flags['autocast']


def test_source_archive_closure_relative_dependencies_only_python(tmp_path):
    (tmp_path/'tools/x').mkdir(parents=True)
    (tmp_path/'tools/x/main.py').write_text('from .helper import f\nimport torch\n')
    (tmp_path/'tools/x/helper.py').write_text('def f(): return 1\n')
    (tmp_path/'tools/x/weights.pt').write_bytes(b'not a source')
    actual=source_inventory(tmp_path,['tools/x/main.py'])
    assert set(actual)=={'tools/x/main.py','tools/x/helper.py'}
    assert all(name.endswith('.py') for name in actual)


def test_private_A_bundle_strict_reload_without_training_checkpoint_or_feedback(tmp_path,monkeypatch):
    # Small real refiner modules; explicit fake binding reader only inside unit test.
    from tools.st_world_policy import base_binding
    from starVLA.model.modules.action_effect.effect_model import EffectModelConfig
    from starVLA.model.modules.action_effect.refinement_policy import CoefficientRefiner,RefinerConfig
    from starVLA.model.modules.action_effect.trajectory_constraints import ConstraintSpec
    torch.set_num_threads(1)
    cfg=RefinerConfig(model=EffectModelConfig(context_dim=12,world_tokens=2,action_tokens=2,hidden_dim=16,layers=1,heads=2))
    original=CoefficientRefiner(cfg).eval()
    payload={'schema':'action_effect_private_deploy_v1','route':'A','arm':'A0',
        'config':asdict(cfg),'constraint_spec':asdict(ConstraintSpec()),'state_dict':original.state_dict()}
    private=tmp_path/'PRIVATE.pt';torch.save(payload,private)
    binding=tmp_path/'BINDING.json';binding.write_text('{}')
    monkeypatch.setattr(base_binding,'read_binding',lambda p:{'binding_sha256':'fixture-base'})
    code=Path('tools/action_effect/common.py')
    m={'schema':'action_effect_research_deployment_bundle_v1','route':'A','arm':'A0','private_head':artifact(private),
        'base_binding':artifact(binding),'base_binding_hash':'fixture-base','effect_teacher_required':False,
        'simulator_required':False,'training_cache_required':False,'GT_required':False,'LiDAR_required':False,
        'source_inventory':{str(code):sha256(code)},'training_provenance':{'checkpoint':'deleted/not-needed'}}
    m['identity']=digest(m);path=tmp_path/'BUNDLE.json';write_json(path,m)
    model=load_bundle(path)
    assert model.effect_teacher is None and model.base_policy is None
    assert all(torch.equal(v,model.refiner.state_dict()[k]) for k,v in original.state_dict().items())
    context={'memory':torch.zeros(1,4,12),'ego':torch.zeros(1,1,4),'mask':torch.ones(1,4,dtype=torch.bool)}
    trajectory=torch.zeros(1,8,3);trajectory[0,:,0]=torch.arange(1,9)
    result=predict_bundle(model,{'context':context,'base_physical':trajectory},'A','cpu')
    assert torch.equal(result,trajectory)
    with pytest.raises(ValueError,match='Only current context'):
        predict_bundle(model,{'context':context,'base_physical':trajectory,'GT':trajectory},'A','cpu')
    with pytest.raises(ImportError):DenyTrainingRuntime().find_spec('navsim.planning')
    with pytest.raises(ImportError):DenyTrainingRuntime().find_spec('tools.action_effect.train_effect_model')


def test_private_B_input_boundary_no_second_rollout_or_feedback():
    calls=[]
    class Model:
        def predict_action(self,c,*,initial_noise):calls.append((c,initial_noise));return initial_noise
    inputs={'condition':torch.ones(1,8,12),'initial_noise':torch.zeros(1,8,4)}
    assert torch.equal(predict_bundle(Model(),inputs,'B','cpu'),inputs['initial_noise']) and len(calls)==1
    with pytest.raises(ValueError,match='Only current condition'):
        predict_bundle(Model(),{**inputs,'real_feedback':torch.ones(1)},'B','cpu')


def test_DEV_B_prediction_uses_separate_reader_no_repair_targets(tmp_path,monkeypatch):
    from tools.action_effect import evaluate_effect_flow as module
    context=tmp_path/'DEV_CONTEXT.json';context.write_text('{}')
    reg=tmp_path/'DEV_REG.json';reg.write_text('{}')
    checkpoint=tmp_path/'step_000001.pt';checkpoint.write_bytes(b'explicit unit fixture')
    raw=torch.zeros(8,4);raw[:,3]=1.
    from starVLA.dataloader.foresight_dataset import decode_ego
    record={'scene_id':'devscene','source_log':'devlog','role':'dev_report','observation_hash':'observation',
        'condition':torch.ones(8,12),'noise':raw.clone(),'physical':decode_ego(raw)}
    class Reader:
        binding={'schema':'action_effect_evaluation_registration_v1','split':'dev','for_training':False,
            'identity':'evalbinding','base_checkpoint_binding_hash':'base','base_checkpoint':{'sha256':'ownBase'},
            'score_protocol_hash':'protocol','numeric_runtime_hash':'numeric','base_binding':artifact(reg)}
        manifest={'registration':artifact(reg)};rows={'devscene':{k:record[k] for k in ('scene_id','source_log','role','observation_hash')}}
        def __init__(self,*args,**kwargs):pass
        def __getitem__(self,scene_id):return record
    monkeypatch.setitem(sys.modules,'tools.action_effect.evaluation_context_data',SimpleNamespace(EvaluationContextReader=Reader))
    calls=[]
    def load(*args,**kwargs):
        assert kwargs['expected_context_artifact'] is None and kwargs['expected_repair_artifact'] is None
        def predict(condition,*,initial_noise):calls.append((condition,initial_noise));return initial_noise
        return SimpleNamespace(provenance={'arm':'B0','checkpoint':artifact(checkpoint)},predict_action=predict)
    monkeypatch.setattr(module,'load_flow_policy',load)
    result=module.predict({'cohort_size':1,'base_binding':'fixture'},dataset_path=None,context_manifest=context,
        checkpoint=checkpoint,output=tmp_path/'out',role='dev_report',max_samples=1,device='cpu')
    assert result['status']=='COMPLETE' and len(calls)==1
    prediction=json.loads((tmp_path/'out/PREDICTIONS.json').read_text());row=prediction['entries'][0]
    registration=json.loads((tmp_path/'out/PREDICTION_REGISTRATION.json').read_text())
    assert row['base_metrics'] is None and row['base_feedback'] is None and row['teacher_forward_calls']==0
    assert 'repair_dataset' not in registration and registration['population_artifact_kind']=='DEV_context_only_no_repairs'
    assert registration['evaluation_split']=='dev' and row['complete_generations']==1
    baseline=module.predict_baseline({'cohort_size':1},context_manifest=context,output=tmp_path/'S0',
        role='dev_report',max_samples=1)
    assert baseline['model_forward_calls']==0 and len(calls)==1
    baseline_rows=json.loads((tmp_path/'S0/PREDICTIONS.json').read_text())['entries']
    assert baseline_rows[0]['physical']==record['physical'].double().tolist()
