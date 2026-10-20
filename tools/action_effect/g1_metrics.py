"""Descriptive held-out G1 metrics from immutable prediction records.

No model, simulator, checkpoint choice, threshold fitting or GPU is used here.
Scene averages are primary; pair averages retain explicit separate denominators.
"""
from __future__ import annotations
from collections import defaultdict
import math
import numpy as np

EVENTS=('collision','road','ttc')
FIRST_TIMES=('first_at_fault_time_seconds','first_road_time_seconds','first_ttc_state_time_seconds')


def average(values):
    return float(np.mean(values)) if len(values) else None


def calibration(probability, target, *, bins=10):
    """Fixed-bin diagnostic, never a guarantee or a fitted calibrator."""
    p=np.asarray(probability,dtype=float);y=np.asarray(target,dtype=float)
    if p.shape!=y.shape or p.ndim!=1 or not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError('Finite same-shape binary diagnostic arrays required')
    if ((p<0)|(p>1)|(y<0)|(y>1)).any():raise ValueError('Probability/target outside [0,1]')
    if not len(p):return {'count':0,'brier':None,'ece':None,'bins':[]}
    idx=np.minimum((p*bins).astype(int),bins-1);rows=[];ece=0.
    for i in range(bins):
        mask=idx==i;n=int(mask.sum())
        row={'lower':i/bins,'upper':(i+1)/bins,'count':n,'mean_probability':average(p[mask]),'frequency':average(y[mask])}
        if n:ece+=n/len(p)*abs(row['mean_probability']-row['frequency'])
        rows.append(row)
    return {'count':len(p),'brier':float(np.mean((p-y)**2)),'ece':float(ece),'bins':rows,
            'interpretation':'descriptive fixed bins; probabilities are not declared calibrated'}


def classification(confusion):
    cm=np.asarray(confusion,dtype=np.int64);recall=[];precision=[]
    for i in range(3):
        recall.append(float(cm[i,i]/cm[i].sum()) if cm[i].sum() else None)
        precision.append(float(cm[i,i]/cm[:,i].sum()) if cm[:,i].sum() else None)
    available=[x for x in recall if x is not None]
    return {'count':int(cm.sum()),'confusion_true_by_predicted':cm.tolist(),
            'class_order':['repaired','unchanged','introduced'],
            'class_counts':cm.sum(1).tolist(),'accuracy':float(np.trace(cm)/cm.sum()) if cm.sum() else None,
            'balanced_accuracy_present_classes':average(available),'recall':recall,'precision':precision,
            'macro_recall_support':len(available)}


def _selection(rows, names, *, identity=True):
    score=names.index('score');available=[r for r in rows if r['valid'][score]]
    if not available:return None
    # Stable candidate identity controls ties; identity wins an exact zero tie.
    ordered=sorted(available,key=lambda r:(-r['mean_delta'][score],r['candidate_id']))
    chosen=ordered[0] if not identity or ordered[0]['mean_delta'][score]>0 else None
    before=np.asarray(available[0]['before']);bvalid=np.asarray(available[0]['before_valid'],dtype=bool)
    if any(not np.array_equal(np.asarray(r['before']),before) or r['before_valid']!=bvalid.tolist() for r in available):
        raise ValueError('One scene has inconsistent native Base labels')
    if not bvalid[score]:raise ValueError('Valid delta without valid Base label')
    true=np.array([r['after'][score]-r['before'][score] for r in available])
    actual=chosen['after'][score]-chosen['before'][score] if chosen is not None else 0.
    oracle=max(0.,float(true.max()))
    base=float(before[score]);selected=base+actual
    events={e:bool(chosen['event_transition'][e]['introduced']) if chosen is not None else False for e in EVENTS}
    out={'selected_candidate':chosen['candidate_id'] if chosen else 'identity',
         'candidate_count':len(available),'selected_predicted_delta':float(chosen['mean_delta'][score]) if chosen else 0.,
         'base_score':base,'selected_score':selected,'actual_selected_delta':float(actual),
         'oracle_delta_including_identity':oracle,'selection_regret':oracle-actual,
         'random_intervention_delta':float(true.mean()),'random_with_identity_delta':float(true.sum()/(len(true)+1)),
         'switched':chosen is not None,'harmful_switch':bool(actual<-.01),'improving_switch':bool(actual>.01),
         'high_score_to_zero':bool(base>=.8 and selected<=1e-8),
         'base_nonzero_to_zero':bool(base>1e-8 and selected<=1e-8),'introduced_events':events,
         'selected_component_deltas':{},'topk_oracle_delta':{}}
    for j,name in enumerate(names):
        if chosen is None:out['selected_component_deltas'][name]=0. if bvalid[j] else None
        else:out['selected_component_deltas'][name]=(float(chosen['after'][j]-chosen['before'][j]) if chosen['valid'][j] else None)
    # Offline true-score upper bound within the k highest predicted proposals;
    # this is not a simulator-free deployment selector.
    for k in (1,3,5):
        prefix=ordered[:k]
        out['topk_oracle_delta'][str(k)]=max(0.,max((r['after'][score]-r['before'][score] for r in prefix),default=0.))
    return out


def summarize_records(records, *, names, units, scales, requested_scenes, mode):
    if len(names)!=len(units) or len(scales)!=len(names) or len(set(names))!=len(names) or 'score' not in names:
        raise ValueError('Actual named physical label schema required')
    if not records:raise ValueError('No held-out predictions; do not manufacture a zero score')
    if mode not in ('absolute_score','terminal_pdms','absolute_feedback','delta_feedback','trajectory_only','shuffled_context'):
        raise ValueError('Unknown G1 control mode')
    requested={r['scene_id']:r for r in requested_scenes}
    if len(requested)!=len(requested_scenes):raise ValueError('Duplicate requested held-out scene')
    by=defaultdict(list);seen=set();d=len(names)
    for r in records:
        if r['pair_identity'] in seen:raise ValueError('Repeated pair is not independent held-out evidence')
        seen.add(r['pair_identity']);sid=r['scene_id']
        if sid not in requested or r['source_log']!=requested[sid]['source_log'] or r.get('role')!='validation':
            raise ValueError('Prediction outside locked held-out TRAIN population')
        for key in ('before','after','mean_delta','logvar_delta','valid','before_valid','after_valid'):
            a=np.asarray(r[key])
            if a.shape!=(d,) or not np.isfinite(a).all():raise ValueError('Malformed physical prediction/label: '+key)
        for key in ('valid','before_valid','after_valid'):
            if np.asarray(r[key]).dtype!=np.bool_:raise ValueError('Boolean validity mask required')
        if r['valid']!=(np.asarray(r['before_valid'])&np.asarray(r['after_valid'])).tolist():
            raise ValueError('Delta validity must be the native before/after intersection')
        by[sid].append(r)
    supervised=([u=='0..1' for u in units] if mode=='absolute_score' else
        [n=='score' for n in names] if mode=='terminal_pdms' else [True]*d)
    full_feedback=mode not in ('absolute_score','terminal_pdms')
    labels={};scene_metrics=[];event_conf={e:np.zeros((3,3),dtype=np.int64) for e in EVENTS}
    event_prob={e:[] for e in EVENTS};event_target={e:[] for e in EVENTS}
    repair_p=[];repair_y=[];temporal=[]
    for j,name in enumerate(names):
        scene_mse=[];scene_mae=[];scene_bias=[];all_errors=[];cover1=[];cover95=[];standardized=[];valid_count=0
        censor={'neither_observed':0,'before_only':0,'after_only':0,'both_observed':0}
        for rows in by.values():
            errors=[]
            for r in rows:
                if units[j]=='s' and name.startswith('first_'):
                    before,after=r['before_valid'][j],r['after_valid'][j]
                    censor['both_observed' if before and after else 'before_only' if before else 'after_only' if after else 'neither_observed']+=1
                if not r['valid'][j]:continue
                err=r['mean_delta'][j]-(r['after'][j]-r['before'][j]);errors.append(err);all_errors.append(err);valid_count+=1
                std=math.exp(.5*r['logvar_delta'][j])*scales[j]
                cover1.append(abs(err)<=std);cover95.append(abs(err)<=1.96*std);standardized.append(abs(err)/std)
            if errors:
                a=np.asarray(errors);scene_mse.append(float(np.mean(a*a)));scene_mae.append(float(np.mean(abs(a))));scene_bias.append(float(a.mean()))
        labels[name]={'unit':units[j],'normalization_scale':scales[j],'supervised_in_this_arm':supervised[j],
            'valid_pairs':valid_count,'valid_scenes':len(scene_mse),
            'scene_weighted_rmse':math.sqrt(average(scene_mse)) if scene_mse else None,
            'scene_weighted_mae':average(scene_mae),'scene_weighted_bias':average(scene_bias),
            'pair_weighted_rmse':float(np.sqrt(np.mean(np.square(all_errors)))) if all_errors else None,
            'normalized_scene_rmse':math.sqrt(average(scene_mse))/scales[j] if scene_mse else None,
            'uncertainty_diagnostic':{'supervised_in_this_arm':supervised[j], 'valid_pairs':valid_count,
                'coverage_1sigma_pair_weighted':average(cover1),'coverage_1p96sigma_pair_weighted':average(cover95),
                'mean_standardized_absolute_error':average(standardized),'calibrated':False}}
        if units[j]=='s' and name.startswith('first_'):
            labels[name]['event_observation_patterns']=censor
            labels[name]['censored_time_regression']={'status':'UNAVAILABLE',
                'reason':'Absent event times are masked native observations; zero is a placeholder, not an event time. No censored likelihood was trained.'}
    score=names.index('score');all_directions=[]
    for sid,rows in sorted(by.items()):
        nt=[r for r in rows if r['valid'][score] and abs(r['after'][score]-r['before'][score])>.01]
        correct=[np.sign(r['mean_delta'][score])==np.sign(r['after'][score]-r['before'][score]) for r in nt]
        all_directions.extend(correct)
        selection=_selection(rows,names)
        scene={'scene_id':sid,'source_log':requested[sid]['source_log'],'pairs':len(rows),'non_tie_pairs':len(nt),
            'direction_accuracy':average(correct),'selection':selection,'intervention_subsets':{}}
        for kind in sorted({r['kind'] for r in rows}):
            subset=[r for r in rows if r['kind']==kind]
            sc=[r for r in subset if r['valid'][score] and abs(r['after'][score]-r['before'][score])>.01]
            scene['intervention_subsets'][kind]={'pairs':len(subset),'non_tie_pairs':len(sc),
                'direction_accuracy':average([np.sign(r['mean_delta'][score])==np.sign(r['after'][score]-r['before'][score]) for r in sc]),
                'selection':_selection(subset,names)}
        scene_metrics.append(scene)
        for r in rows:
            probabilities=np.asarray(r['event_probabilities'],dtype=float)
            if probabilities.shape!=(3,3) or not np.isfinite(probabilities).all() or (probabilities<0).any() or not np.allclose(probabilities.sum(-1),1,atol=1e-6):
                raise ValueError('Finite three-event transition probabilities required')
            for j,event in enumerate(EVENTS):
                y=0 if r['event_transition'][event]['repaired'] else 2 if r['event_transition'][event]['introduced'] else 1
                event_conf[event][y,int(probabilities[j].argmax())]+=1
                event_prob[event].append(probabilities[j].tolist());event_target[event].append(y)
            repair_p.append(r['repair_probability']);repair_y.append(float(r['eligible_repair']))
            p=np.asarray(r['temporal_delta']);y=np.asarray(r['temporal_target']);v=np.asarray(r['temporal_valid'])
            if p.shape!=y.shape or v.shape!=y.shape or v.dtype!=np.bool_ or not np.isfinite(p).all() or not np.isfinite(y).all():
                raise ValueError('Finite temporal targets and explicit masks required')
            if v.any():temporal.append(float(np.mean(abs(p[v]-y[v]))))
    choices=[s['selection'] for s in scene_metrics if s['selection'] is not None]
    gains=[s['actual_selected_delta'] for s in choices];oracle=[s['oracle_delta_including_identity'] for s in choices]
    requested_count=len(requested);covered=len(by);score_covered=len(choices)
    selection_summary={'requested_scenes':requested_count,'valid_scored_scenes':score_covered,
        'coverage_complete':score_covered==requested_count,
        'PDMS_on_complete_requested_population':average([s['selected_score'] for s in choices]) if score_covered==requested_count else None,
        'base_PDMS_valid_subset':average([s['base_score'] for s in choices]),
        'selected_PDMS_valid_subset':average([s['selected_score'] for s in choices]),
        'selected_delta':average(gains),'oracle_delta':average(oracle),
        'regret':average([s['selection_regret'] for s in choices]),
        'random_intervention_delta':average([s['random_intervention_delta'] for s in choices]),
        'random_with_identity_delta':average([s['random_with_identity_delta'] for s in choices]),
        'switch_count':sum(s['switched'] for s in choices),'harmful_switch_count':sum(s['harmful_switch'] for s in choices),
        'improving_switch_count':sum(s['improving_switch'] for s in choices),
        'high_score_to_zero_count':sum(s['high_score_to_zero'] for s in choices),
        'base_nonzero_to_zero_count':sum(s['base_nonzero_to_zero'] for s in choices),
        'introduced_events':{e:sum(s['introduced_events'][e] for s in choices) for e in EVENTS},
        'selected_component_deltas':{n:average([s['selected_component_deltas'][n] for s in choices if s['selected_component_deltas'][n] is not None]) for n in names},
        'topk_oracle_true_delta':{str(k):average([s['topk_oracle_delta'][str(k)] for s in choices]) for k in (1,3,5)},
        'topk_definition':'offline best true gain among k top-predicted interventions plus identity; not deployed oracle'}
    event_summary={}
    for e in EVENTS:
        event_summary[e]=classification(event_conf[e]);event_summary[e]['supervised_in_this_arm']=full_feedback
        p=np.array(event_prob[e]);y=np.array(event_target[e])
        event_summary[e]['introduced_event_calibration']=calibration(p[:,2],(y==2).astype(float))
        event_summary[e]['multiclass_brier']=float(np.mean(np.sum((p-np.eye(3)[y])**2,axis=-1)))
    kinds=sorted({r['kind'] for r in records});subsets={}
    for kind in kinds:
        ss=[s['intervention_subsets'][kind] for s in scene_metrics if kind in s['intervention_subsets']]
        selected=[x['selection'] for x in ss if x['selection'] is not None]
        subsets[kind]={'scenes':len(ss),'pairs':sum(x['pairs'] for x in ss),
            'scene_weighted_direction_accuracy':average([x['direction_accuracy'] for x in ss if x['direction_accuracy'] is not None]),
            'selected_delta':average([x['actual_selected_delta'] for x in selected]),
            'oracle_delta':average([x['oracle_delta_including_identity'] for x in selected]),
            'generalization':'withheld joint combination' if kind=='joint' else 'held-out log, represented intervention family'}
    return {'requested_scenes':requested_count,'covered_scenes':covered,'missing_scene_ids':sorted(set(requested)-set(by)),
        'heldout_pairs':len(records),'heldout_groups':len({r['source_log'] for r in requested_scenes}),
        'label_metrics':labels,'direction_accuracy_scene_weighted':average([s['direction_accuracy'] for s in scene_metrics if s['direction_accuracy'] is not None]),
        'direction_accuracy_pair_weighted':average(all_directions),'direction_non_tie_pairs':len(all_directions),
        'event_transitions':event_summary,'repair_eligibility_calibration':{**calibration(repair_p,repair_y),
            'supervised_in_this_arm':full_feedback,'target':'native eligible_repair, not necessarily strictly positive score delta'},
        'temporal_event_mae_pair_mean':average(temporal),'temporal_supervised_in_this_arm':full_feedback,
        'selection':selection_summary,'intervention_subsets':subsets,'scenes':scene_metrics,
        'population':'held-out TRAIN logs, not untouched DEV or NAVTEST','scientific_conclusion':'UNTESTED'}


def paired_group_bootstrap(first,second,*,metric='actual_selected_delta',iterations=2000,seed=42):
    """Paired log bootstrap of the same complete, score-covered scene list."""
    if not 1<=iterations<=10000:raise ValueError('Explicit bootstrap budget 1..10000 required')
    a={s['scene_id']:s for s in first['scenes']};b={s['scene_id']:s for s in second['scenes']}
    if set(a)!=set(b) or first['missing_scene_ids'] or second['missing_scene_ids']:
        raise ValueError('No silent common-subset bootstrap of incomplete evaluation populations')
    groups=defaultdict(list)
    for sid,x in a.items():
        y=b[sid]
        if x['source_log']!=y['source_log'] or x['selection'] is None or y['selection'] is None:
            raise ValueError('Matched scene/log and valid score required for paired bootstrap')
        groups[x['source_log']].append(float(x['selection'][metric]-y['selection'][metric]))
    logs=sorted(groups);sums=np.array([sum(groups[g]) for g in logs]);counts=np.array([len(groups[g]) for g in logs])
    if not logs:raise ValueError('No independent held-out group')
    rng=np.random.default_rng(seed);draw=rng.integers(len(logs),size=(iterations,len(logs)))
    means=sums[draw].sum(1)/counts[draw].sum(1)
    return {'metric':metric,'mean_difference':float(sums.sum()/counts.sum()),'ci95_percentile':np.quantile(means,[.025,.975]).tolist(),
        'independent_groups':len(logs),'scenes':int(counts.sum()),'iterations':iterations,'seed':seed,
        'single_seed_limit':'within-run held-out sampling interval, not training-seed stability'}
