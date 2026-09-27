"""Preregistered paired schedule decisions; no planning test score is accepted."""
import argparse
import json
import math
from pathlib import Path


def require_finished(runs, epoch):
    states = [json.loads((Path(run)/'status.json').read_text()) for run in runs]
    if any(state.get('status') != 'complete' or state.get('epochs') != epoch for state in states):
        raise ValueError('Every paired run must have completed the registered decision epoch')
    if len({(state['step'], state['presentations']) for state in states}) != 1:
        raise ValueError('Paired runs have unequal updates/data exposure')


def graph_extension(runs):
    """Compare both arms on the same fixed graph and supervised population."""
    if len(runs) != 2: raise ValueError('Need exactly two graph runs')
    require_finished(runs, 16)
    manifests=[json.loads((Path(p)/'manifest.json').read_text()) for p in runs]
    if {m['arguments']['mode'] for m in manifests}!={'all','mask'}:raise ValueError('Need exactly ALL and MASK')
    for key in ('code_sha','cache_identity','labels','holdout_identity','holdout_labels','config'):
        if manifests[0][key]!=manifests[1][key]:raise ValueError('Unmatched paired experiment: '+key)
    for key in ('batch','seed','schedule_epochs'):
        if manifests[0]['arguments'][key]!=manifests[1]['arguments'][key]:raise ValueError('Unmatched training budget/config: '+key)
    reasons=[];missing=[];curves=[]
    for run in runs:
        old=json.loads((Path(run)/'holdout_12.json').read_text());new=json.loads((Path(run)/'holdout_16.json').read_text())
        components=('deploy_ego_ADE_m','deploy_agents_ADE_m')
        if any(old.get(k) is None or new.get(k) is None for k in components):
            missing.append(str(run));continue
        before=sum(old[k] for k in components)/2;after=sum(new[k] for k in components)/2
        relative=abs(after-before)/max(abs(before),1e-8)
        worse=[actor for actor in ('ego','agents') if new['deploy_'+actor+'_ADE_m']>new['deploy_'+actor+'_stationary_ADE_m']]
        curves.append({'run':str(run),'relative_change_12_to_16':relative,'worse_than_stationary':worse})
        if relative>.02 or worse:reasons.append(str(run))
    return {'decision':'REVIEW_MISSING_COVERAGE' if missing else ('EXTEND_BOTH_TO_32' if reasons else 'STOP_BOTH_AT_16'),
        'curves':curves,'missing_neighborhood_metrics':missing,'triggered_runs':reasons,'uses_PDMS':False,
        'note':'A stable curve alone does not establish scientific sufficiency or planning improvement.'}


def planner_extension(runs):
    """Apply the registered 6-to-8 holdout DiT ADE rule jointly to three equal controls."""
    if len(runs) != 3: raise ValueError('Need CURRENT, ALL and MASK planner runs')
    require_finished(runs, 8)
    manifests = [json.loads((Path(run)/'manifest.json').read_text()) for run in runs]
    if {item['arguments']['mode'] for item in manifests} != {'current','all','mask'}:
        raise ValueError('Planner controls are incomplete or duplicated')
    for key in ('code_sha','current_identity','labels','holdout_labels','graph_config','foundation_sha256',
                'trainable_parameters','frozen_DiT_parameters'):
        if any(item[key] != manifests[0][key] for item in manifests[1:]):
            raise ValueError('Unmatched planner comparison: '+key)
    for key in ('batch','seed','schedule_epochs'):
        if any(item['arguments'][key] != manifests[0]['arguments'][key] for item in manifests[1:]):
            raise ValueError('Unmatched planner schedule: '+key)
    curves=[]; reasons=[]; missing=[]
    for run in runs:
        old=json.loads((Path(run)/'holdout_6.json').read_text()); new=json.loads((Path(run)/'holdout_8.json').read_text())
        values=[item.get('ego_ADE_m') for item in (old,new)]
        if any(value is None or not math.isfinite(value) for value in values) or any(item.get('failed',0) for item in (old,new)):
            missing.append(str(run)); continue
        if old['scenes'] != new['scenes']: raise ValueError('Planner holdout population changed')
        change=abs(values[1]-values[0])/max(abs(values[0]),1e-8)
        curves.append({'run':str(run),'ego_ADE_epoch6_m':values[0],'ego_ADE_epoch8_m':values[1],
                       'relative_change_6_to_8':change})
        if change>.02: reasons.append(str(run))
    return {'decision':'REVIEW_MISSING_METRICS' if missing else ('EXTEND_ALL_THREE_TO_16' if reasons else 'STOP_ALL_THREE_AT_8'),
        'curves':curves,'triggered_runs':reasons,'missing_runs':missing,'uses_PDMS':False,
        'note':'This is the preregistered finite schedule rule, not proof of convergence. Preserve the16epoch scheduler and common terminal length.'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',nargs='+',required=True)
    p.add_argument('--phase',choices=['graph','planner'],default='graph');p.add_argument('--output',required=True);a=p.parse_args()
    result=(graph_extension if a.phase=='graph' else planner_extension)(a.runs);path=Path(a.output)
    if path.exists():raise FileExistsError('Keep prior preregistered decision')
    path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
