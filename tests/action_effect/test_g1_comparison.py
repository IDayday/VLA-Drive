"""Synthetic reporting tests; not learned G1 or real simulator evidence."""
from copy import deepcopy
from pathlib import Path
import json
import math
import subprocess
import sys

import numpy as np
import pytest
import torch

from tools.action_effect.common import artifact,digest,write_json
from tools.action_effect.g1_metrics import calibration,paired_group_bootstrap,summarize_records
from tools.action_effect.compare_effect_models import compare,training_summary,_load_evaluation
from starVLA.model.modules.action_effect.effect_model import MODES

NAMES=('score','no_at_fault_collisions','drivable_area_compliance','ego_progress',
 'time_to_collision_within_bound','comfort','driving_direction_compliance','progress_raw_meters',
 'first_at_fault_time_seconds','first_ttc_state_time_seconds','first_road_time_seconds')
UNITS=('0..1',)*7+('m','s','s','s');SCALES=(1.,)*7+(20.,4.,4.,4.)


def record(scene='a',candidate='p0',*,delta=.05,pred=.1,kind='longitudinal',time_observed=False):
    before=[.9,1.,1.,.8,1.,1.,1.,20.,0.,0.,0.]
    after=before.copy();after[0]+=delta
    valid=[True]*8+[time_observed]*3
    if time_observed:before[8:]=[1.,2.,3.];after[8:]=[2.,2.5,3.5]
    mean=[0.]*11;mean[0]=pred
    temporal=[[0.,0.,0.],[0.,0.,0.]]
    return {'pair_identity':scene+'_'+candidate,'native_pair_identity':'native_'+scene+candidate,
        'scene_id':scene,'source_log':'log_'+scene,'role':'validation','candidate_id':candidate,'kind':kind,
        'before':before,'after':after,'before_valid':valid.copy(),'after_valid':valid.copy(),'valid':valid.copy(),
        'mean_delta':mean,'logvar_delta':[0.]*11,'event_probabilities':[[.05,.9,.05]]*3,
        'event_transition':{e:{'repaired':False,'introduced':False,'unchanged':True} for e in ('collision','road','ttc')},
        'repair_probability':.7,'eligible_repair':True,'temporal_delta':deepcopy(temporal),
        'temporal_target':deepcopy(temporal),'temporal_valid':[[True]*3]*2}


def requested(*scenes):return [{'scene_id':s,'source_log':'log_'+s,'role':'validation'} for s in scenes]


def summary(rows,scenes=('a',),mode='delta_feedback'):
    return summarize_records(rows,names=NAMES,units=UNITS,scales=SCALES,requested_scenes=requested(*scenes),mode=mode)


def test_complete_selection_all_physical_channels_and_risk_denominators():
    good=record('a',delta=.05,pred=.1);bad=record('a','bad',delta=-.9,pred=.2)
    bad['event_transition']['collision']={'repaired':False,'introduced':True,'unchanged':False}
    s=summary([good,bad]);v=s['selection']
    assert s['requested_scenes']==s['covered_scenes']==1 and s['heldout_pairs']==2
    assert set(s['label_metrics'])==set(NAMES)
    assert v['selected_delta']==pytest.approx(-.9) and v['oracle_delta']==pytest.approx(.05)
    assert v['regret']==pytest.approx(.95) and v['high_score_to_zero_count']==1
    assert v['introduced_events']['collision']==1 and v['switch_count']==1
    assert v['topk_oracle_true_delta']['3']==pytest.approx(.05)
    assert s['event_transitions']['collision']['class_counts']==[0,1,1]


def test_no_predicted_gain_retains_identity_and_stable_positive_ties_use_candidate_identity():
    rows=[record('a','z',delta=-.2,pred=0.),record('a','a',delta=.05,pred=-.1)]
    s=summary(rows);assert s['scenes'][0]['selection']['selected_candidate']=='identity'
    assert s['selection']['selected_delta']==0 and s['selection']['switch_count']==0
    for r in rows:r['mean_delta'][0]=.1
    x=summary(rows);y=summary(list(reversed(rows)))
    assert x['scenes'][0]['selection']['selected_candidate']==y['scenes'][0]['selection']['selected_candidate']=='a'


def test_scene_weighting_differs_from_pooling_variable_pair_counts():
    rows=[record('a',str(i),delta=0,pred=1.) for i in range(4)]+[record('b',delta=0,pred=0.)]
    s=summary(rows,('a','b'));label=s['label_metrics']['score']
    assert label['scene_weighted_rmse']==pytest.approx(math.sqrt(.5))
    assert label['pair_weighted_rmse']==pytest.approx(math.sqrt(.8))
    assert label['valid_pairs']==5 and label['valid_scenes']==2


def test_first_event_absence_is_masked_not_treated_as_zero_time():
    no_event=record();both=record('b','both',time_observed=True)
    after_only=record('c','after_only');after_only['after_valid'][8]=True;after_only['after'][8]=2.
    s=summary([no_event,both,after_only],('a','b','c'));metric=s['label_metrics']['first_at_fault_time_seconds']
    assert metric['valid_pairs']==1 and metric['scene_weighted_mae']==pytest.approx(1.)
    assert metric['event_observation_patterns']=={'neither_observed':1,'before_only':0,'after_only':1,'both_observed':1}
    assert metric['censored_time_regression']['status']=='UNAVAILABLE'


def test_untrained_terminal_control_heads_are_not_presented_as_trained_diagnostics():
    s=summary([record()],mode='terminal_pdms')
    assert s['label_metrics']['score']['supervised_in_this_arm']
    assert not s['label_metrics']['progress_raw_meters']['supervised_in_this_arm']
    assert not s['event_transitions']['collision']['supervised_in_this_arm']
    assert not s['repair_eligibility_calibration']['supervised_in_this_arm']


def test_joint_combinations_are_a_separate_heldout_intervention_subset():
    s=summary([record(kind='joint'),record('a','other',kind='lateral')])
    assert s['intervention_subsets']['joint']['generalization']=='withheld joint combination'
    assert s['intervention_subsets']['joint']['pairs']==1


def test_missing_scene_does_not_become_complete_population_PDMS():
    s=summary([record()],('a','b'))
    assert s['missing_scene_ids']==['b'] and not s['selection']['coverage_complete']
    assert s['selection']['PDMS_on_complete_requested_population'] is None
    assert s['selection']['selected_PDMS_valid_subset']==pytest.approx(.95)
    with pytest.raises(ValueError,match='common-subset'):paired_group_bootstrap(s,s,iterations=10)


def test_same_log_paired_bootstrap_is_deterministic_and_uses_scene_coverage():
    a=summary([record('a'),record('b')],('a','b'))
    b=summary([record('a',pred=-1),record('b',pred=-1)],('a','b'))
    x=paired_group_bootstrap(a,b,iterations=100,seed=9);y=paired_group_bootstrap(a,b,iterations=100,seed=9)
    assert x==y and x['independent_groups']==2 and x['scenes']==2
    assert x['mean_difference']==pytest.approx(.05) and x['ci95_percentile']==pytest.approx([.05,.05])


def test_binary_calibration_fixed_bins_and_never_calibrated_claim():
    x=calibration([.9,.2],[1.,0.]);assert x['brier']==pytest.approx(.025)
    assert x['ece']==pytest.approx(.15) and 'not declared calibrated' in x['interpretation']
    assert calibration([],[])['count']==0
    with pytest.raises(ValueError):calibration([1.1],[1.])


@pytest.mark.parametrize('mutation',[lambda r:r.update(role='fit'),lambda r:r.update(source_log='other'),
    lambda r:r['mean_delta'].__setitem__(0,float('nan')),lambda r:r['valid'].__setitem__(0,False)])
def test_bad_role_alignment_finiteness_or_mask_is_rejected(mutation):
    r=record();mutation(r)
    with pytest.raises(ValueError):summary([r])


def test_duplicate_pair_is_not_more_heldout_evidence():
    r=record()
    with pytest.raises(ValueError,match='Repeated pair'):summary([r,r])


def test_all_six_pending_arms_remain_pending_and_no_model_is_selected(tmp_path):
    x=compare(dict.fromkeys(MODES),output=tmp_path,step=1000,bootstrap=10)
    assert x['status']=='NOT_RUN' and x['completion']=='PENDING'
    assert len(x['rows'])==6 and all(r['completion']=='PENDING' for r in x['rows'])
    assert x['scientific_conclusion']=='UNTESTED' and x['checkpoint_selection']['status']=='NOT_SELECTED_BY_REPORT'
    assert not x['gpu_or_simulator_started']
    y=compare(dict.fromkeys(MODES),output=tmp_path,step=1000,bootstrap=10);assert x==y


def test_training_history_preserves_full_update_cost_and_validation_but_not_checkpoint_amortization():
    identity={'runtime':{'device':'cpu'},'source_commit':'synthetic'}
    state={'history':[{'optimizer_step':1,'seconds':.5},{'optimizer_step':2,'seconds':.7,'validation':{'all':'metrics'}}],
        'optimizer_step':2,'terminal':True,'termination_reason':'MAX_UPDATES','consumed_scenes':64,
        'model':{'p':torch.zeros(3)}}
    x=training_summary(identity,state)
    assert x['private_parameters']==3 and x['update_timing']['sum_measured_update_seconds']==pytest.approx(1.2)
    assert len(x['validation_history'])==1 and 'UNAVAILABLE' in x['update_timing']['checkpoint_and_validation_cost']
    state['history'][1]['optimizer_step']=3
    with pytest.raises(ValueError,match='each completed'):training_summary(identity,state)


def test_cli_help_and_pending_exit_code(tmp_path):
    cmd=[sys.executable,'-m','tools.action_effect.compare_effect_models']
    result=subprocess.run(cmd+['--help'],capture_output=True,text=True)
    assert result.returncode==0 and 'evaluate' in result.stdout and 'compare' in result.stdout
    mapping=tmp_path/'mapping.json';mapping.write_text(json.dumps(dict.fromkeys(MODES)))
    result=subprocess.run(cmd+['compare','--evaluations-json',str(mapping),'--step','1000',
        '--bootstrap','10','--output',str(tmp_path/'out')],capture_output=True,text=True)
    assert result.returncode==2 and json.loads(result.stdout)['status']=='NOT_RUN'


def write_evaluation(root,arm,*,seed=42):
    """Self-contained fake artifacts test joins only, never simulator/model fit."""
    root.mkdir(parents=True);rows=[record('a'),record('b',kind='joint')]
    source=[]
    for r in rows:
        source.append({**{k:r[k] for k in ('scene_id','source_log','role','candidate_id','kind','event_transition','eligible_repair')},
            'identity':r['pair_identity'],'base_labels':{'values':r['before'],'valid':r['before_valid']},
            'modified_labels':{'values':r['after'],'valid':r['after_valid']},
            'temporal_event_delta':r['temporal_target'],'temporal_event_valid':r['temporal_valid']})
    dataset={'schema':'synthetic_compact_fixture_only','entries':source,'scene_index':requested('a','b')};dataset['identity']=digest(dataset)
    write_json(root/'DATA.json',dataset);write_json(root/'CONTEXT.json',{'synthetic':True});write_json(root/'BINDING.json',{'synthetic':True})
    spec={'names':list(NAMES),'units':list(UNITS),'scales':list(SCALES)}
    ti={'arm':arm,'model_config':{'mode':arm,'hidden_dim':16,'init_seed':42},'recipe':{'batch_scenes':2},
        'private_names':['p'],'label_spec':spec,'base_binding_hash':'b'*64,'data_artifact':artifact(root/'DATA.json'),
        'context_artifact':artifact(root/'CONTEXT.json'),'source_commit':'synthetic_cpu_fixture','seed':seed}
    write_json(root/'IDENTITY.json',ti)
    state={'schema':'action_effect_private_checkpoint_v1','run_identity':digest(ti),'optimizer_step':2,
        'model':{'p':torch.zeros(3)},'history':[{'optimizer_step':1,'seconds':.3},
            {'optimizer_step':2,'seconds':.4,'validation':{'diagnostic':'only'}}],
        'terminal':True,'termination_reason':'MAX_UPDATES','consumed_scenes':4}
    torch.save(state,root/'step_000002.pt')
    write_json(root/'step_000002.COMPLETE.json',{'checkpoint':artifact(root/'step_000002.pt'),'run_identity':digest(ti),'optimizer_step':2})
    evaluation={'schema':'action_effect_g1_evaluation_identity_v1','arm':arm,'optimizer_step':2,
        'base_binding_hash':ti['base_binding_hash'],'context_artifact':ti['context_artifact'],'data_artifact':ti['data_artifact'],
        'label_spec':spec,'requested_scenes':requested('a','b'),'heldout_pair_identities':[r['pair_identity'] for r in rows],
        'checkpoint':artifact(root/'step_000002.pt'),'checkpoint_identity':artifact(root/'IDENTITY.json'),
        'campaign_binding':artifact(root/'BINDING.json')};evaluation['identity']=digest(evaluation)
    write_json(root/'EVAL_IDENTITY.json',evaluation)
    predictions={'schema':'action_effect_g1_predictions_v1','evaluation_identity':evaluation['identity'],'entries':rows}
    metric={'schema':'action_effect_g1_metrics_v1','evaluation_identity':evaluation['identity'],**summary(rows,('a','b'),mode=arm)}
    write_json(root/'PAIR_PREDICTIONS.json',predictions);write_json(root/'METRICS.json',metric)
    write_json(root/'TRAINING_SNAPSHOT.json',training_summary(ti,state))
    complete={'schema':'action_effect_g1_evaluation_complete_v1','status':'COMPLETE','evaluation_identity':evaluation['identity'],
        'identity_artifact':artifact(root/'EVAL_IDENTITY.json'),'predictions':artifact(root/'PAIR_PREDICTIONS.json'),
        'metrics':artifact(root/'METRICS.json'),'training_snapshot':artifact(root/'TRAINING_SNAPSHOT.json'),'cpu_cost':{'total_wall_seconds':.1}}
    complete['identity']=digest(complete);write_json(root/'COMPLETE.json',complete)
    return root/'COMPLETE.json'


def test_evaluation_source_join_rejects_rehashed_fake_labels(tmp_path):
    path=write_evaluation(tmp_path/'fake','delta_feedback');x=_load_evaluation(path)
    assert x['metrics']['heldout_pairs']==2
    cp=json.loads(path.read_text());predpath=Path(cp['predictions']['path']);pred=json.loads(predpath.read_text())
    pred['entries'][0]['after'][0]=1. # Not the immutable native compact label.
    predpath.write_text(json.dumps(pred));cp['predictions']=artifact(predpath)
    cp['identity']=digest({k:v for k,v in cp.items() if k!='identity'});path.write_text(json.dumps(cp))
    with pytest.raises(ValueError,match='native physical labels'):_load_evaluation(path)


def test_exact_six_control_compare_checks_capacity_fixed_step_and_seed(tmp_path):
    # The native data/context pins must be the same across arms; fixtures share
    # those immutable dependencies while keeping separate trained identities.
    paths={a:write_evaluation(tmp_path/a,a) for a in MODES}
    ref=json.loads((tmp_path/MODES[0]/'EVAL_IDENTITY.json').read_text())
    for arm in MODES[1:]:
        root=tmp_path/arm;ti=json.loads((root/'IDENTITY.json').read_text());ev=json.loads((root/'EVAL_IDENTITY.json').read_text())
        for k in ('data_artifact','context_artifact'):ti[k]=ref[k];ev[k]=ref[k]
        (root/'IDENTITY.json').write_text(json.dumps(ti));state=torch.load(root/'step_000002.pt',weights_only=False)
        state['run_identity']=digest(ti);torch.save(state,root/'step_000002.pt')
        (root/'step_000002.COMPLETE.json').write_text(json.dumps({'checkpoint':artifact(root/'step_000002.pt'),
            'run_identity':digest(ti),'optimizer_step':2}))
        ev['checkpoint']=artifact(root/'step_000002.pt');ev['checkpoint_identity']=artifact(root/'IDENTITY.json')
        ev['identity']=digest({k:v for k,v in ev.items() if k!='identity'});(root/'EVAL_IDENTITY.json').write_text(json.dumps(ev))
        for filename in ('PAIR_PREDICTIONS.json','METRICS.json'):
            obj=json.loads((root/filename).read_text());obj['evaluation_identity']=ev['identity'];(root/filename).write_text(json.dumps(obj))
        cp=json.loads((root/'COMPLETE.json').read_text());cp['evaluation_identity']=ev['identity'];cp['identity_artifact']=artifact(root/'EVAL_IDENTITY.json')
        cp['predictions']=artifact(root/'PAIR_PREDICTIONS.json');cp['metrics']=artifact(root/'METRICS.json')
        cp['identity']=digest({k:v for k,v in cp.items() if k!='identity'});(root/'COMPLETE.json').write_text(json.dumps(cp))
    x=compare(paths,output=tmp_path/'comparison',step=2,bootstrap=20)
    assert x['status']=='COMPLETE' and x['scientific_conclusion']=='UNTESTED' and len(x['paired_log_intervals'])==5
    assert x['matched_parameter_count']==3 and x['checkpoint_selection']['status']=='NOT_SELECTED_BY_REPORT'
    with pytest.raises(ValueError,match='fixed optimizer endpoint'):
        compare(paths,output=tmp_path/'wrongstep',step=1,bootstrap=20)
