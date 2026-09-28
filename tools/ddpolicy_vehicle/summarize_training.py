"""Aggregate all ranks, retaining task exposure and incomplete-log evidence."""
import argparse
import json
from pathlib import Path
import numpy as np
from .prepare_data import atomic_json


def summarize(run):
    run = Path(run)
    identity=json.loads((run/'identity.json').read_text())
    status=json.loads((run/'status.json').read_text())
    ranks=[]
    for rank in range(identity['world_size']):
        path=run/f'train_rank{rank}.jsonl'
        values=[json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        indexed={r['update']:r for r in values}
        if len(indexed)!=len(values):raise ValueError('Duplicate optimizer update in rank log')
        ranks.append(indexed)
    common=set.intersection(*(set(r) for r in ranks))
    union=set.union(*(set(r) for r in ranks))
    rows=[]
    for step in sorted(common):
        parts=[rank[step] for rank in ranks]
        if len({p['global_scene_exposure'] for p in parts})!=1:raise ValueError('Rank exposure mismatch')
        row={'update':step,'scene_presentations':parts[0]['global_scene_exposure'],'lr':parts[0]['lr']}
        keys=set.union(*(set(p['losses']) for p in parts))
        if any(set(p['losses'])!=keys for p in parts):raise ValueError('Rank task cadence mismatch')
        for key in keys:
            values=[p['losses'][key] for p in parts]
            if not np.isfinite(values).all():raise ValueError('Nonfinite logged loss')
            row[key]=float(np.mean(values))
        for key in ('selected_vehicles','context_vehicles','ego_only'):
            row[key]=sum(p['graphs'][key] for p in parts)
        for key in set.union(*(set(p['coordinates']) for p in parts)):
            row['coordinates_'+key]=sum(p['coordinates'][key] for p in parts)
        row['applied_role_scenes']=sum(p.get('tasks',{}).get('applied_role_scenes',0) for p in parts)
        rows.append(row)
    complete=not (union-common) and len(common)==status['real_optimizer_updates']
    arm=identity['arm'];counters={}
    if common:
        for rank in ranks:
            for key,value in rank[max(common)]['roles'].items():counters[key]=counters.get(key,0)+value
    report={'run_id':run.name,'arm':arm,'source_sha':identity['source_sha'],'status':status['status'],
            'optimizer_updates':status['real_optimizer_updates'],'scene_presentations':status['sample_presentations'],
            'common_logged_updates':len(common),'incomplete_rank_steps':sorted(union-common),
            'complete_log_coverage':complete,'world_size':identity['world_size'],
            'seed':identity['config']['seed'],'diagnostic':identity['startup'] or identity.get('small_fit',False),
            'scheduler_role_counts':counters,
            'actual_role_objective_scenes':counters.get('actual_ego',0)+counters.get('actual_neighbor',0) if arm=='C' else 0,
            'role_count_note':'B eligibility scheduler counts are candidate tasks; B objective is all-hidden',
            'main_scene_presentations':status['sample_presentations'],
            'selected_vehicle_scene_presentations':sum(r['selected_vehicles'] for r in rows),
            'context_vehicle_scene_presentations':sum(r['context_vehicles'] for r in rows),
            'main_vehicle_coordinates':sum(r.get('coordinates_vehicle',0) for r in rows),
            'main_ego_coordinates':sum(r.get('coordinates_ego',0) for r in rows)}
    report['optimizer_calls']=report.pop('optimizer_updates')
    report['verified_effective_updates']=sum(
        ranks[0][step].get('optimizer_update',{}).get('fp32_master_update_verified',False) for step in common)
    report['training_validity']='PER_STEP_MASTER_UPDATE_PROBE' if report['verified_effective_updates']==len(common) and common else 'NOT_VERIFIED'
    for overlay in run.parents[1].glob('optimizer_stasis_correction*/validity_overlay.json'):
        for item in json.loads(overlay.read_text())['runs']:
            if item['run_id']==run.name:
                report['training_validity']='INVALID_OPTIMIZER_STASIS'
                report['verified_effective_updates']=0
    if arm=='A' and report['main_ego_coordinates']==0:
        repeat=identity['config']['framework']['action_model']['repeated_diffusion_steps']
        report['main_ego_coordinates']=status['sample_presentations']*repeat*8*4
        report['legacy_base_coordinate_logging_correction']='exposure x FM repeats x8steps x4coordinates; actual loss was present'
    report['auxiliary_scene_presentations']=sum(
        row['scene_presentations']-(rows[i-1]['scene_presentations'] if i else 0)
        for i,row in enumerate(rows) if 'role_auxiliary' in row)
    report['last16_loss_mean']={key:float(np.mean([r[key] for r in rows[-16:] if key in r]))
        for key in ('main_fm','vehicle_cls','vehicle_box','vehicle_motion','video','depth') if any(key in r for r in rows[-16:])}
    return report,rows


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--runs',nargs='+',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    reports=[]
    for run in a.runs:
        report,rows=summarize(run);reports.append(report)
        atomic_json(out/(report['run_id']+'.json'),{'summary':report,'curve':rows})
        for ax,key,title in zip(axes.flat,('main_fm','vehicle_box','vehicle_motion','selected_vehicles'),
            ('Main FM, trailing16-update mean','Current box loss, trailing16-update mean',
             'Auxiliary motion head loss, trailing16-update mean','Selected graph vehicles per global batch')):
            values=[(r['update'],r[key]) for r in rows if key in r]
            if not values:continue
            x,y=map(np.array,zip(*values));window=min(16,len(y)) if key!='selected_vehicles' else 1
            ax.plot(x[window-1:],np.convolve(y,np.ones(window)/window,mode='valid'),label=report['run_id'])
            ax.set_title(title);ax.set_xlabel('Optimizer calls (validity recorded in JSON)');ax.grid(alpha=.2)
    handles,labels=axes[0,0].get_legend_handles_labels()
    if handles:fig.legend(handles,labels,loc='lower center',fontsize=7)
    fig.suptitle('Training diagnostics; curves do not establish planning performance')
    fig.tight_layout(rect=(0,.08,1,.95));fig.savefig(out/'learning_curves.png',dpi=180);fig.savefig(out/'learning_curves.svg');plt.close(fig)
    atomic_json(out/'summary.json',{'runs':reports,'scope':'training curves, not development/Navtest results'})


if __name__=='__main__':main()
