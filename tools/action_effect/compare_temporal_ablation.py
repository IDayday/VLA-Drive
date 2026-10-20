"""Read two explicit G1 receipts for the dense temporal-grid loss ablation.

This is a fixed-endpoint prediction/ranking comparison on held-out TRAIN logs.
It is not a policy evaluation or proof of causal temporal credit assignment.
No training, VLM inference, new simulation, model selection or DEV access occurs.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from .common import artifact,atomic_bytes,digest,read_json,verify_artifact,write_json
from .compare_effect_models import _load_evaluation,checkpoint_state,_source
from .g1_metrics import paired_group_bootstrap
from .train_effect_model import loss_weights


def _matched(full,no_temporal,*,step):
    """Verify the one allowed objective difference against actual checkpoints."""
    values=[]
    for x,weight in ((full,.2),(no_temporal,0.)):
        ei=x['identity'];ti,state,_=checkpoint_state(verify_artifact(ei['checkpoint']))
        if ei['arm']!='delta_feedback' or ei['optimizer_step']!=step:
            raise ValueError('Both explicit snapshots must be delta_feedback at the predeclared same endpoint')
        if ti['loss_weights']!=loss_weights(weight):
            raise ValueError('Only the dense temporal weight may differ: complete=0.2, no_temporal=0')
        if x['metrics']['missing_scene_ids'] or not x['metrics']['selection']['coverage_complete']:
            raise ValueError('Full fixed held-out scene coverage required; no common-subset filtering')
        values.append((ti,state))
    a,b=values
    # Source revisions are disclosed. Device ordinal is an assignment, not a
    # numerical backend; the device type, hardware and all flags still match.
    def comparable(ti):
        runtime={k:v for k,v in ti['runtime'].items() if k!='device'}
        runtime['device_type']=ti['runtime']['device'].split(':')[0]
        return {**{k:v for k,v in ti.items() if k not in ('source_commit','loss_weights','runtime')},'runtime':runtime}
    if digest(comparable(a[0]))!=digest(comparable(b[0])):
        raise ValueError('Unmatched model, seed, data, numerical runtime, optimizer/sampler recipe or horizon')
    for key in ('sampler',):
        if not torch.equal(a[1][key],b[1][key]):raise ValueError('Consumed scene sampler states differ')
    if set(a[1]['stateful'])!=set(b[1]['stateful']) or 'pair_rng' not in a[1]['stateful']:
        raise ValueError('Explicit same pair-stream RNG inventory required')
    if not torch.equal(a[1]['stateful']['pair_rng'],b[1]['stateful']['pair_rng']):
        raise ValueError('Consumed pair sampler states differ')
    if a[1]['consumed_scenes']!=b[1]['consumed_scenes']:
        raise ValueError('Actual sample exposures differ')
    keys=('base_binding_hash','context_artifact','data_artifact','label_spec','requested_scenes','heldout_pair_identities')
    if any(full['identity'][k]!=no_temporal['identity'][k] for k in keys):
        raise ValueError('Unmatched physical labels, observation bank or held-out pair identities')
    if full['training']['private_parameters']!=no_temporal['training']['private_parameters']:
        raise ValueError('Model capacity differs')
    return values


def compare(full_evaluation,no_temporal_evaluation,*,step,output,bootstrap=2000,seed=42):
    if type(step) is not int or step<1 or not 1<=bootstrap<=10000:
        raise ValueError('Explicit positive fixed step and bounded bootstrap required')
    declared={'full':full_evaluation,'no_temporal':no_temporal_evaluation};loaded={};status={}
    for arm,path in declared.items():
        if path is None or not Path(path).is_file():
            status[arm]={'status':'NOT_RUN','path':str(path) if path is not None else None}
        else:
            loaded[arm]=_load_evaluation(path);status[arm]={'status':'COMPLETE','artifact':artifact(path)}
    result={'schema':'action_effect_g1_temporal_ablation_v1','status':'NOT_RUN','completion':'PENDING',
        'source_commit':_source(),'fixed_optimizer_step':step,'evaluations':status,
        'only_intended_difference':'dense temporal-grid loss weight 0.2 versus 0; first-event mean channels, event transitions, repair and ranking retained',
        'full_metrics':{k:x['complete']['metrics'] for k,x in loaded.items()},
        'scientific_conclusion':'UNTESTED','policy_temporal_credit_claim':'NOT_ESTABLISHED',
        'selection_rule':'explicit same fixed endpoint, no model/threshold selection','training_or_simulator_started':False}
    if len(loaded)==2:
        states=_matched(loaded['full'],loaded['no_temporal'],step=step)
        result.update(status='COMPLETE',completion='COMPLETE_DESCRIPTIVE',
            private_parameters=loaded['full']['training']['private_parameters'],
            consumed_scene_presentations=states[0][1]['consumed_scenes'],
            training_source_commits=[ti['source_commit'] for ti,_ in states],
            source_note='Different revisions are disclosed; CPU affected-scope tests establish the unchanged default objective, not universal source equivalence.',
            prediction_ranking_paired_log_interval=paired_group_bootstrap(loaded['full']['metrics'],loaded['no_temporal']['metrics'],iterations=bootstrap,seed=seed),
            rows=[])
        for arm,x in loaded.items():
            m=x['metrics'];sel=m['selection']
            result['rows'].append({'arm':arm,'temporal_loss_weight':.2 if arm=='full' else 0.,
                'heldout_scenes':m['covered_scenes'],'heldout_pairs':m['heldout_pairs'],
                'direction_accuracy_scene_weighted':m['direction_accuracy_scene_weighted'],
                'score_delta_scene_rmse':m['label_metrics']['score']['scene_weighted_rmse'],
                'temporal_grid_mae':m['temporal_event_mae_pair_mean'],
                'temporal_grid_is_supervised':m['temporal_supervised_in_this_arm'],
                'selected_native_delta':sel['selected_delta'],'regret':sel['regret'],
                'introduced_events':sel['introduced_events'],'high_score_to_zero_count':sel['high_score_to_zero_count'],
                'all_physical_channels':m['label_metrics'],'event_transitions':m['event_transitions'],
                'training_cost':x['training']['update_timing'],'cpu_evaluation_cost':x['complete']['cpu_cost']})
    result['identity']=digest(result);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    write_json(output/'TEMPORAL_ABLATION.json',result)
    text=['# Dense temporal-feedback G1 ablation','',f"Completion: {result['completion']}; fixed endpoint {step}.",'',
        'Only the dense time-grid loss coefficient changes from 0.2 to zero. Both models retain the eleven physical mean labels, including observed first-event times, all event-transition, repair and gap-sensitive ranking supervision.',
        '','The JSON pins full per-label, event, uncertainty, per-scene and risk reporting. The no-temporal head remains allocated but is not supervised; its error is diagnostic.',
        '','No policy improvement or temporal-credit benefit is inferred. This comparison addresses held-out TRAIN effect prediction and within-bank intervention ranking. Policy-level temporal credit requires separately matched A/B deployment experiments.',
        '','Missing inputs remain PENDING; no GPU update or simulation is dispatched by this tool.']
    atomic_bytes(output/'TEMPORAL_ABLATION.md',('\n'.join(text)+'\n').encode());return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--full-evaluation',required=True);p.add_argument('--no-temporal-evaluation',required=True)
    p.add_argument('--step',type=int,required=True);p.add_argument('--output',required=True)
    p.add_argument('--bootstrap',type=int,default=2000);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--dry-run',action='store_true');a=p.parse_args(argv)
    try:
        if a.dry_run:print(json.dumps({'status':'NOT_RUN','no_jobs_started':True}));return 0
        r=compare(a.full_evaluation,a.no_temporal_evaluation,step=a.step,output=a.output,bootstrap=a.bootstrap,seed=a.seed)
        print(json.dumps({'status':r['status'],'identity':r['identity']}));return 0 if r['status']=='COMPLETE' else 2
    except Exception as e:
        print(json.dumps({'status':'FAIL','error_type':type(e).__name__,'error':str(e)}));return 1


if __name__=='__main__':raise SystemExit(main())
