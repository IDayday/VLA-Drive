"""Synthetic admission tests, not new learned/simulator evidence."""
from copy import deepcopy
from pathlib import Path
import json
import torch
import pytest
from tools.action_effect import compare_eligible_effect_models as derived
from tools.action_effect.common import artifact,digest,write_json
from tools.action_effect.g1_metrics import paired_group_bootstrap,summarize_records
from tests.action_effect import test_g1_comparison as f


def resign(x):
    x['identity']=digest({k:v for k,v in x.items() if k!='identity'});return x


def fixtures(monkeypatch,*,arm='delta_feedback',temporal=.2):
    records=[f.record('a',delta=.05,pred=.1),f.record('a','a2',delta=-.2,pred=-.1),
             f.record('b',delta=.02,pred=.03)]
    requested=f.requested('a','b','c');entries=[]
    for r in records:
        entries.append({**{k:r[k] for k in ('scene_id','source_log','role','candidate_id','kind')},'identity':r['pair_identity']})
    dataset=resign({'schema':'action_effect_pair_dataset_v1','status':'COMPLETE','split':'train',
        'deployment_inputs':False,'invalid_scene_evidence':[],'requested_scenes':3,'scene_index':requested,'entries':entries})
    spec={'names':list(f.NAMES),'units':list(f.UNITS),'scales':list(f.SCALES)}
    original={'identity':{'checkpoint':{'path':'unit-not-read','sha256':'fixture'},'requested_scenes':requested,
        'heldout_pair_identities':[r['pair_identity'] for r in records],'arm':arm,'label_spec':spec},
        'metrics':summarize_records(records,**spec,requested_scenes=requested,mode=arm),
        'complete':{'metrics':{'path':'original-METRICS.json','sha256':'fixture'}}}
    monkeypatch.setattr(derived,'verify_artifact',lambda pin:Path(pin['path']))
    ti={'source_commit':'fixture','loss_weights':{'temporal':temporal}}
    monkeypatch.setattr(derived,'checkpoint_state',lambda _:(ti,{},{}))
    return original,dataset,records


def test_structural_no_pair_admission_preserves_original_and_does_not_invent_absolute_score(monkeypatch):
    original,dataset,records=fixtures(monkeypatch);saved=deepcopy(original)
    x=derived.admit_population(original,dataset,records);pop=x['population_admission']
    assert original==saved and not original['metrics']['selection']['coverage_complete']
    assert pop['original_requested_scenes']==3 and pop['eligible_native_pair_scenes']==2
    assert pop['complete_native_validation_pairs']==3 and pop['structural_no_available_intervention_scenes']==1
    assert pop['structural_scene_inventory'][0]['scene_id']=='c'
    assert x['metrics']['requested_scenes']==2 and not x['metrics']['missing_scene_ids']
    assert x['metrics']['selection']['coverage_complete']
    extension=x['structural_identity_delta_extension']
    assert extension['mean_increment_over_original_requested_scenes']==pytest.approx((.05+.02)/3)
    assert extension['absolute_Base_PDMS'] is extension['absolute_selected_PDMS'] is None
    assert extension['forced_identity_structural_scenes']==1
    with pytest.raises(ValueError,match='common-subset'):paired_group_bootstrap(original['metrics'],original['metrics'],iterations=10)
    ci=paired_group_bootstrap(x['metrics'],x['metrics'],iterations=10)
    assert ci['scenes']==2 and ci['independent_groups']==2 and ci['mean_difference']==0.


@pytest.mark.parametrize('drop',[0,2])
def test_any_missing_pair_rejects_even_if_its_scene_still_has_other_predictions(monkeypatch,drop):
    original,dataset,records=fixtures(monkeypatch)
    rows=records[:drop]+records[drop+1:]
    # Even a rehashed/relabelled partial request cannot change source eligibility.
    original['identity']['heldout_pair_identities']=[r['pair_identity'] for r in rows]
    with pytest.raises(ValueError,match='Every native validation pair'):
        derived.admit_population(original,dataset,rows)


@pytest.mark.parametrize('case',['duplicate','wrong_log','swapped_native_pair_scene','requested_omission','bad_original_count','invalid_native_evidence'])
def test_other_negative_population_and_identity_cases(monkeypatch,case):
    original,dataset,records=fixtures(monkeypatch)
    if case=='duplicate':records.append(deepcopy(records[0]))
    if case=='wrong_log':records[0]['source_log']='other'
    if case=='swapped_native_pair_scene':
        records[0]['scene_id']='b';records[0]['source_log']='log_b'
    if case=='requested_omission':original['identity']['requested_scenes']=f.requested('a','b')
    if case=='bad_original_count':original['metrics']['covered_scenes']=1
    if case=='invalid_native_evidence':dataset['invalid_scene_evidence']=[{'scene_id':'c','error':'true native failure'}];resign(dataset)
    with pytest.raises(ValueError):derived.admit_population(original,dataset,records)


def test_temporal_disabled_report_keeps_all_physical_mean_targets(monkeypatch):
    original,dataset,records=fixtures(monkeypatch,temporal=0.)
    x=derived.admit_population(original,dataset,records)
    assert not x['metrics']['temporal_supervised_in_this_arm']
    assert all(row['supervised_in_this_arm'] for row in x['metrics']['label_metrics'].values())


def write_original_evaluation(root):
    """A tiny signed artifact chain exercises the real strict loader, CPU only."""
    path=f.write_evaluation(root,'delta_feedback')
    data=json.loads((root/'DATA.json').read_text())
    data.update(schema='action_effect_pair_dataset_v1',status='COMPLETE',split='train',deployment_inputs=False,
        requested_scenes=3,invalid_scene_evidence=[],scene_index=f.requested('a','b','c'))
    resign(data);(root/'DATA.json').write_text(json.dumps(data))
    ti=json.loads((root/'IDENTITY.json').read_text());ti['data_artifact']=artifact(root/'DATA.json')
    (root/'IDENTITY.json').write_text(json.dumps(ti))
    state=torch.load(root/'step_000002.pt',weights_only=False);state['run_identity']=digest(ti)
    torch.save(state,root/'step_000002.pt')
    (root/'step_000002.COMPLETE.json').write_text(json.dumps({'checkpoint':artifact(root/'step_000002.pt'),
        'run_identity':digest(ti),'optimizer_step':2}))
    ev=json.loads((root/'EVAL_IDENTITY.json').read_text());ev.update(data_artifact=ti['data_artifact'],
        checkpoint=artifact(root/'step_000002.pt'),checkpoint_identity=artifact(root/'IDENTITY.json'),requested_scenes=f.requested('a','b','c'))
    resign(ev);(root/'EVAL_IDENTITY.json').write_text(json.dumps(ev))
    pred=json.loads((root/'PAIR_PREDICTIONS.json').read_text());pred['evaluation_identity']=ev['identity']
    (root/'PAIR_PREDICTIONS.json').write_text(json.dumps(pred))
    metrics={'schema':'action_effect_g1_metrics_v1','evaluation_identity':ev['identity'],
        **summarize_records(pred['entries'],**ev['label_spec'],requested_scenes=ev['requested_scenes'],mode='delta_feedback')}
    (root/'METRICS.json').write_text(json.dumps(metrics))
    cp=json.loads(path.read_text());cp.update(evaluation_identity=ev['identity'],identity_artifact=artifact(root/'EVAL_IDENTITY.json'),
        predictions=artifact(root/'PAIR_PREDICTIONS.json'),metrics=artifact(root/'METRICS.json'))
    resign(cp);path.write_text(json.dumps(cp));return path


def test_real_loader_admission_is_read_only_and_rejects_forged_missing_prediction(tmp_path):
    original=write_original_evaluation(tmp_path/'original')
    tracked=list(original.parent.glob('*'));before={p:artifact(p) for p in tracked if p.is_file()}
    x=derived.derive(original)
    assert x['population_admission']['original_requested_scenes']==3
    assert x['population_admission']['eligible_native_pair_scenes']==2
    assert all(artifact(p)==v for p,v in before.items())
    cp=json.loads(original.read_text());p=Path(cp['predictions']['path']);rows=json.loads(p.read_text());rows['entries'].pop()
    p.write_text(json.dumps(rows));cp['predictions']=artifact(p);resign(cp);original.write_text(json.dumps(cp))
    with pytest.raises(ValueError):derived.derive(original)


@pytest.mark.parametrize('kind,arms',[('six',tuple(derived.MODES)),('temporal',('full','no_temporal'))])
def test_missing_experiments_keep_pending_and_versioned_output(tmp_path,kind,arms):
    x=derived.compare(dict.fromkeys(arms),kind=kind,step=1000,output=tmp_path,bootstrap=10)
    assert x['status']=='NOT_RUN' and x['completion']=='PENDING'
    assert x['new_model_forwards']==x['new_optimizer_updates']==x['new_simulator_calls']==0
    assert x['not_full_original_population_PDMS'] and x['scientific_conclusion']=='UNTESTED'
    assert Path(x['report_directory']).name==x['identity']
    y=derived.compare(dict.fromkeys(arms),kind=kind,step=1000,output=tmp_path,bootstrap=10)
    assert x==y


def model_report(monkeypatch,arm):
    original,data,records=fixtures(monkeypatch,arm=arm)
    x=derived.admit_population(original,data,records)
    x['identity'].update(arm=arm,optimizer_step=1000,base_binding_hash='base',context_artifact='context',data_artifact='dataset')
    x['training']={'private_parameters':42}
    from tools.action_effect.train_effect_model import loss_weights
    x['training_identity']={'source_commit':'unit','loss_weights':loss_weights(),
        'model_config':{'mode':arm,'init_seed':42,'hidden_dim':16},'recipe':{'batch_scenes':32},'seed':42,
        'runtime':{'device':'cpu','dtype':'float32'}}
    return x


def test_complete_six_uses_all_native_pair_scenes_and_rejects_different_admission(tmp_path,monkeypatch):
    examples={arm:model_report(monkeypatch,arm) for arm in derived.MODES}
    paths={arm:tmp_path/arm for arm in derived.MODES}
    for p in paths.values():p.write_text('explicit orchestration stub; strict loader tested separately')
    monkeypatch.setattr(derived,'derive',lambda path:examples[Path(path).name])
    x=derived.compare(paths,kind='six',step=1000,output=tmp_path/'reports',bootstrap=10)
    assert x['status']=='COMPLETE' and len(x['paired_log_intervals_on_complete_eligible_population'])==5
    assert x['population_admission']['original_requested_scenes']==3
    assert all(row['eligible_native_pair_scenes']==2 and row['original_full_population_PDMS'] is None for row in x['rows'])
    examples['terminal_pdms']['training_identity']['runtime']['tf32']=True
    with pytest.raises(ValueError,match='numerical runtime'):
        derived.compare(paths,kind='six',step=1000,output=tmp_path/'wrong_runtime',bootstrap=10)
    del examples['terminal_pdms']['training_identity']['runtime']['tf32']
    examples['terminal_pdms']['population_admission']['structural_scene_inventory'][0]['source_log']='foreign'
    with pytest.raises(ValueError,match='Different source/eligible/requested'):
        derived.compare(paths,kind='six',step=1000,output=tmp_path/'wrong',bootstrap=10)


def test_complete_temporal_uses_same_derived_admission_and_strict_original_weight_sampler_match(tmp_path,monkeypatch):
    from tools.action_effect import compare_temporal_ablation as temporal
    from tools.action_effect.train_effect_model import loss_weights
    examples={arm:model_report(monkeypatch,'delta_feedback') for arm in ('full','no_temporal')}
    states={};paths={}
    for arm,x in examples.items():
        p=tmp_path/arm;p.write_text('explicit orchestration stub');paths[arm]=p
        x['identity']['checkpoint']=artifact(p)
        x['training_identity']['loss_weights']=loss_weights(.2 if arm=='full' else 0.)
        states[p.resolve()]=(x['training_identity'],{'sampler':torch.Generator().manual_seed(143).get_state(),
            'stateful':{'pair_rng':torch.Generator().manual_seed(244).get_state()},'consumed_scenes':32000},{})
    monkeypatch.setattr(derived,'derive',lambda path:examples[Path(path).name])
    monkeypatch.setattr(temporal,'checkpoint_state',lambda path:states[Path(path).resolve()])
    x=derived.compare(paths,kind='temporal',step=1000,output=tmp_path/'report',bootstrap=10)
    assert x['status']=='COMPLETE' and x['comparison']=='temporal'
    assert x['paired_log_intervals_on_complete_eligible_population']['full_minus_no_temporal']['scenes']==2
    examples['no_temporal']['training_identity']['loss_weights']['ranking']=0.
    with pytest.raises(ValueError,match='exact named full/zero objective'):
        derived.compare(paths,kind='temporal',step=1000,output=tmp_path/'bad',bootstrap=10)
