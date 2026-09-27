"""Point-aware paired mechanism metrics; uncertainty resamples whole holdout logs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from tools.joint_local_scene_v3.budget import atomic_json


def value(x):
    if x=='':return None
    if x in ('True','False'):return x=='True'
    try:return float(x) if any(c in x for c in '.eE') else int(x)
    except ValueError:return x


def read_queries(path):
    with Path(path).open() as f:rows=[{k:(v if k in ('query_id','token','log','graph_origin','scope','valid_timesteps') else value(v)) for k,v in r.items()} for r in csv.DictReader(f)]
    scenes=Path(path).parent/'all_hidden_scenes.csv'
    if scenes.exists():
        with scenes.open() as f:population={r['token']:int(r['selected_neighbors']) for r in csv.DictReader(f)}
        for r in rows:r['scene_selected_neighbors']=population[r['token']]
    return rows


def select(rows,actor='all',motion='all',other='all',scene='all'):
    return [r for r in rows if (actor=='all' or (r['slot']==0)==(actor=='ego')) and (motion=='all' or r.get('dynamic')==(motion=='dynamic')) and (other=='all' or (r['known_other_xy_points']>0)==(other=='available')) and (scene=='all' or (r['scene_selected_neighbors']>0)==(scene=='neighbors'))]


def metrics(rows,prefix):
    result={'queries':len(rows),'failed':sum(r.get('status','ok')!='ok' for r in rows),'xy_points':sum(r['valid_xy_points'] for r in rows),'complete_targets':sum(r['complete_horizon'] for r in rows),'partial_targets':sum(not r['complete_horizon'] for r in rows),'final_time_targets':sum(r['final_time_valid'] for r in rows)}
    if not rows or result['failed']:return result
    points=result['xy_points']
    result['ADE_m']=sum(r[prefix+'_xy_error_sum_m'] for r in rows)/points
    final=[r[prefix+'_xy_FDE_m'] for r in rows if r[prefix+'_xy_FDE_m'] is not None];result['FDE_m']=float(np.mean(final)) if final else None
    yaw=sum(r[prefix+'_valid_yaw_points'] for r in rows)
    result['yaw_error_rad']=sum((r[prefix+'_yaw_error_rad'] or 0)*r[prefix+'_valid_yaw_points'] for r in rows)/yaw if yaw else None
    result['yaw_points']=yaw
    result['stationary_ADE_m']=sum(r[prefix+'_stationary_ADE_m']*r['valid_xy_points'] for r in rows)/points
    cv=[r for r in rows if r[prefix+'_CV_ADE_m'] is not None]
    result['CV_ADE_m']=sum(r[prefix+'_CV_ADE_m']*r['valid_xy_points'] for r in cv)/sum(r['valid_xy_points'] for r in cv) if cv else None
    return result


def summarize_queries(rows):
    result={}
    for actor in ('ego','neighbor'):
        for motion in ('all','static','dynamic'):
            for other in ('all','available','none'):
                group=select(rows,actor,motion,other)
                result[f'{actor}_{motion}_other_{other}']={prefix:metrics(group,prefix) for prefix in ('all_hidden','conditional')}
        for scene in ('ego_only','neighbors'):
            group=select(rows,actor,scene=scene)
            result[f'{actor}_scene_{scene}']={prefix:metrics(group,prefix) for prefix in ('all_hidden','conditional')}
    return result


def bootstrap(rows,replicates=5000):
    # Rows already paired at query identity; resample complete logs with all their scenes/targets.
    logs={}
    for r in rows:
        totals=logs.setdefault(r['log'],[0.,0.,0.]);totals[0]+=r['all_sum'];totals[1]+=r['mask_sum'];totals[2]+=r['points']
    if not logs:return {'queries':0,'logs':0}
    arr=np.array(list(logs.values()));denom=arr[:,2].sum();all_mean=arr[:,0].sum()/denom;mask_mean=arr[:,1].sum()/denom
    result={'queries':len(rows),'logs':len(logs),'valid_xy_points':int(denom),'J_ALL_ADE_m':float(all_mean),'J_MASK_ADE_m':float(mask_mean),'MASK_minus_ALL_m':float(mask_mean-all_mean),'relative_change_percent':float((mask_mean/all_mean-1)*100),'ci_kind':'paired whole-log bootstrap, not training-seed uncertainty','ci95_m':None}
    if len(logs)>1:
        rng=np.random.default_rng(20260927);idx=rng.integers(len(arr),size=(replicates,len(arr)));draw=arr[idx].sum(1);diff=(draw[:,1]-draw[:,0])/draw[:,2]
        result['ci95_m']=np.quantile(diff,[.025,.975]).tolist()
    return result


def paired(first,second,prefix,actor,other='all',motion='all'):
    one={r['query_id']:r for r in select(first,actor,motion,other)};two={r['query_id']:r for r in select(second,actor,motion,other)}
    if set(one)!=set(two):raise ValueError('Paired query population mismatch')
    rows=[]
    for q,a in one.items():
        b=two[q]
        if a['status']!='ok' or b['status']!='ok' or a['valid_xy_points']!=b['valid_xy_points'] or a['valid_timesteps']!=b['valid_timesteps']:raise ValueError('Incomplete or mismatched paired results')
        rows.append({'query_id':q,'token':a['token'],'log':a['log'],'all_sum':a[prefix+'_xy_error_sum_m'],'mask_sum':b[prefix+'_xy_error_sum_m'],'points':a['valid_xy_points']})
    scene_rows=[]
    for field in ('token','log'):
        groups={}
        for r in rows:
            group=groups.setdefault(r[field],[0.,0.,0.]);group[0]+=r['all_sum'];group[1]+=r['mask_sum'];group[2]+=r['points']
        for identity,(a,b,n) in groups.items():scene_rows.append({'level':field,'id':identity,'J_ALL_ADE_m':a/n,'J_MASK_ADE_m':b/n,'MASK_minus_ALL_m':(b-a)/n,'valid_xy_points':int(n)})
    return bootstrap(rows),scene_rows


def primary(rows):
    return {f'{prefix}_{actor}':metrics(select(rows,actor,other='available' if prefix=='conditional' else 'all'),prefix).get('ADE_m') for prefix in ('all_hidden','conditional') for actor in ('ego','neighbor')}


def extension_decision(root):
    evidence={};overfit=[];improving=[]
    for arm in ('all','mask'):
        run=root/f'formal42_{arm}';hold={};train={}
        for epoch in (16,24,32):
            hold[epoch]=primary(read_queries(run/f'holdout_{epoch*228}/conditional_queries.csv'))
            train[epoch]=primary(read_queries(run/f'train64_{epoch*228}/conditional_queries.csv'))
        for metric in hold[16]:
            vals=[hold[e][metric] for e in (16,24,32)]
            if any(v is None for v in vals):continue
            imp=[1-vals[i+1]/vals[i] for i in range(2)]
            evidence[arm+'_'+metric]={'holdout':vals,'relative_improvement':imp,'train64':[train[e][metric] for e in (16,24,32)]}
            if min(imp)>.01:improving.append(arm+'_'+metric)
            tv=[train[e][metric] for e in (16,24,32)]
            if metric.startswith('all_hidden') and all(v is not None for v in tv) and max(imp)<-.1 and tv[2]<tv[1]<tv[0]:overfit.append(arm+'_'+metric)
    return {'extend_both_to64_by_learning_rule':bool(improving) and not overfit,'improving_registered_metrics':improving,'overfit_proxy_metrics':overfit,'evidence':evidence,'decision_uses_MASK_winning':False,'budget_feasibility_must_be_checked_separately':True}


def csv_out(path,rows):
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=sorted(set().union(*(r.keys() for r in rows))),lineterminator='\n');writer.writeheader();writer.writerows(rows)


def harvest(root,out):
    out.mkdir(parents=True,exist_ok=True);curves=[];summaries={};run_stats={}
    for run in sorted(root.iterdir()):
        if not run.is_dir() or not run.name.startswith(('small_','formal42_','formal43_')):continue
        log=run/'train.jsonl'
        if not log.exists():continue
        steps=[json.loads(line) for line in log.read_text().splitlines()];manifest=json.loads((run/'manifest.json').read_text())
        run_stats[run.name]={'source':manifest['source'],'identity':manifest,'updates':steps[-1]['step'],'epoch':steps[-1]['epoch'],'offset':steps[-1]['offset'],'presentations':steps[-1]['presentations'],'exposures':steps[-1]['exposures'],'task_totals':steps[-1]['task_totals'],'supervised':steps[-1]['supervised'],'first_32_loss_mean':float(np.mean([s['weighted_loss'] for s in steps[:32]])),'last_32_loss_mean':float(np.mean([s['weighted_loss'] for s in steps[-32:]])),'mean_step_seconds':float(np.mean([s['seconds'] for s in steps])),'peak_gpu_bytes':max(s['peak_gpu_bytes'] for s in steps)}
        for folder in sorted(run.glob('*_*')):
            if not folder.is_dir() or not (folder/'summary.json').exists() or not (folder/'conditional_queries.csv').exists():continue
            if not folder.name.startswith(('holdout_','train64_')):continue
            summary=json.loads((folder/'summary.json').read_text())
            if not summary['aggregate_valid']:raise ValueError('Incomplete evaluation: '+str(folder))
            rows=read_queries(folder/'conditional_queries.csv');summaries[run.name+'/'+folder.name]=summarize_queries(rows)
            split,step=folder.name.rsplit('_',1)
            for prefix in ('all_hidden','conditional'):
                for actor in ('ego','neighbor'):
                    m=metrics(select(rows,actor,other='available' if prefix=='conditional' else 'all'),prefix)
                    curves.append({'run':run.name,'split':split,'step':int(step),'prefix':prefix,'actor':actor,**m})
    atomic_json(out/'run_statistics.json',run_stats);atomic_json(out/'milestone_summaries.json',summaries);csv_out(out/'milestone_curves.csv',curves)
    pairs={};deltas=[];pairing_checks={}
    for seed in (42,43):
        ra=root/f'formal{seed}_all';rb=root/f'formal{seed}_mask'
        common=sorted({int(p.name.split('_')[1]) for p in ra.glob('holdout_*') if (p/'summary.json').exists()} & {int(p.name.split('_')[1]) for p in rb.glob('holdout_*') if (p/'summary.json').exists()})
        if not common:continue
        step=max(common);one=read_queries(ra/f'holdout_{step}/conditional_queries.csv');two=read_queries(rb/f'holdout_{step}/conditional_queries.csv')
        for prefix in ('all_hidden','conditional'):
            for actor in ('ego','neighbor'):
                for other in ('all','available','none'):
                    key=f'seed{seed}_{prefix}_{actor}_other_{other}'
                    result,rows=paired(one,two,prefix,actor,other);pairs[key]=dict(result,step=step)
                    deltas.extend(dict(r,comparison=key,step=step) for r in rows)
        la=[json.loads(x) for x in (ra/'train.jsonl').read_text().splitlines()];lb=[json.loads(x) for x in (rb/'train.jsonl').read_text().splitlines()];n=min(len(la),len(lb))
        pairing_checks[str(seed)]={'steps_compared':n,'data_order_identical':all(a['data_order_sha256']==b['data_order_sha256'] for a,b in zip(la,lb)),'noise_time_hashes_identical':all(a['noise_time_sha256']==b['noise_time_sha256'] for a,b in zip(la,lb)),'equal_completed_length':len(la)==len(lb)}
    atomic_json(out/'paired_results.json',pairs);atomic_json(out/'pairing_checks.json',pairing_checks);csv_out(out/'scene_log_paired_deltas.csv',deltas)
    if curves:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,2,figsize=(13,9))
        for ax,(prefix,actor) in zip(axes.flat,[(p,a) for p in ('all_hidden','conditional') for a in ('ego','neighbor')]):
            for name in run_stats:
                for split,style in [('holdout','-'),('train64','--')]:
                    vals=sorted([r for r in curves if r['run']==name and r['split']==split and r['prefix']==prefix and r['actor']==actor],key=lambda r:r['step'])
                    if vals:ax.plot([r['step'] for r in vals],[r.get('ADE_m',np.nan) for r in vals],style,marker='.',label=name+' '+split)
            ax.set_title(prefix+' '+actor);ax.set_xlabel('optimizer updates');ax.set_ylabel('xy ADE [m]');ax.grid(alpha=.3);ax.legend(fontsize=6)
        fig.tight_layout();fig.savefig(out/'learning_curves.png');plt.close(fig)
    return pairs


def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign',required=True);p.add_argument('--output',required=True);p.add_argument('--extension-decision',action='store_true');a=p.parse_args();root=Path(a.campaign);out=Path(a.output)
    if a.extension_decision:atomic_json(out,extension_decision(root))
    else:harvest(root,out)


if __name__=='__main__':main()
