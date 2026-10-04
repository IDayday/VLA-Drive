"""Read-only C/S experiment history; keep physical steps, splits and protocol IDs.

No inference, metric recomputation, optimizer updates, or best-test selection.
Use the original complete canonical score CSVs, not rounded chat summaries.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from tools.full_foresight.navtest_schedule import atomic, read, sha, write_csv
from tools.foresight.summarize_experiments import collapse_sampling_runs, paired_difference
from tools.local_interaction_mask_v2.compare_pdms import read as read_scenes

FACTORS = {'NC':'no_at_fault_collisions','DAC':'drivable_area_compliance',
           'TTC':'time_to_collision_within_bound','EP':'ego_progress','Comfort':'comfort'}
ARMS = ['C0','C1','C4','S0','S1','S2','S3','S4']


def main():
    p = argparse.ArgumentParser(__doc__)
    for k in ('c-root','s-root','c-history','output'):
        p.add_argument('--'+k,required=True)
    args = p.parse_args(); croot=Path(args.c_root);sroot=Path(args.s_root)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    sources=[]
    for row in csv.DictReader(Path(args.c_history).open()):
        path=Path(row['source'])
        if path.name=='result.json':path=path.parent/'scores/summary.json'
        sources.append((row['arm'],'navtest',path))
    for result in sorted((croot/'navtest_milestones_20261001/jobs').glob('*/result.json')):
        sources.append((read(result)['arm'],'navtest',result.parent/'scores/summary.json'))
    for path in sorted((croot/'canonical_evaluation_v1').glob('dev*_registry.json')):
        for row in read(path):sources.append((row['arm'],'dev',Path(row['score_dir'])/'summary.json'))
    for path in sorted((croot/'scores').glob('formal_C4*_dev*_seed42/summary.json')):
        sources.append(('C4','dev',path))
    for result in sorted((sroot/'navtest_50k_100k_20261003/jobs').glob('*/result.json')):
        sources.append((read(result)['arm'],'navtest',result.parent/'scores/summary.json'))
    for folder in ('scores','scores_reconciled_development_v1'):
        for path in sorted((sroot/folder).glob('formal_S*_dev*_seed42/summary.json')):
            sources.append((path.parent.name.split('_')[1],'dev',path))
    rows={}; populations={}; contracts={}; manifests={}
    for arm,split,path in sources:
        summary=read(path);export=summary['export_identity'];cp=export['checkpoint'];protocol=export['protocol']
        step=cp['completed'];key=(split,arm,step)
        if key in rows:
            if cp['sha256']!=rows[key]['checkpoint_sha256'] or abs(100*summary['PDMS']-rows[key]['PDMS'])>1e-9:
                raise ValueError('Conflicting duplicate result '+str(key))
            continue
        expected=(12146,136) if split=='navtest' else (1696,16)
        if (not summary['valid'] or summary['failed'] or summary['diagnostic']
                or (summary['scenes'],summary['logs'])!=expected or export['limit']
                or export['current_identity']['split']!=split or cp['scope']!='formal'
                or cp['training_seed']!=42 or protocol['sampling_seed']!=42
                or protocol['precision']!='FP32' or protocol['tf32'] or protocol['steps']!=10
                or protocol['future_conditioning'] or protocol['scorer'] is not None
                or protocol['candidates_per_scene']!=1 or not protocol['auxiliary_heads_removed']):
            raise ValueError('Incomplete/changed official protocol '+str(path))
        if arm.startswith('S'):
            regpath=sroot/'registrations'/f'formal_{arm}_seed42_full100k_v2.json'
            train=read(sroot/'students'/f'formal_{arm}_seed42_full100k_v2'/'identity.json')
            if read(regpath)['arm']!=arm or train['identity']!=cp['run_identity'] or train['registration_sha256']!=sha(regpath):
                raise ValueError('S-arm is not the C1 visual-candidate field')
        elif cp['arm']!=arm:
            raise ValueError('Wrong C configuration')
        contract={'current':export['current_identity'],'evaluator':summary['evaluator_identity'],
                  'protocol':{k:v for k,v in protocol.items() if k not in ('W_retained','evaluation_purpose')}}
        if split in contracts and contracts[split]!=contract:
            raise ValueError('Cross-series scoring protocol mismatch for '+str(path))
        contracts[split]=contract
        scenes=read_scenes(path.parent/'scenes.csv')
        if len(scenes)!=expected[0] or len({r['log'] for r in scenes.values()})!=expected[1] or any(r['status']!='ok' for r in scenes.values()):
            raise ValueError('Invalid scene CSV; never filter failures')
        population={t:(r['log'],r['metric_cache_sha256']) for t,r in scenes.items()}
        if split in manifests and population!=manifests[split]:
            raise ValueError('Scene/log/original metric-cache population changed')
        manifests[split]=population
        collapsed,checked=collapse_sampling_runs({42:scenes},[42])
        if abs(checked['metrics']['score']-summary['PDMS'])>1e-10:
            raise ValueError('Score summary disagrees with all scene rows')
        for metric,value in summary['metrics'].items():
            if not math.isfinite(value) or abs(value-checked['metrics'][metric])>1e-10:
                raise ValueError('Factor summary mismatch')
        record=dict(series=arm[0],arm=arm,split=split,updates=step,PDMS=100*summary['PDMS'],
            **{short:100*summary['metrics'][name] for short,name in FACTORS.items()},
            zero_scenes=sum(float(r['score'])==0 for r in scenes.values()),failed=summary['failed'],
            scenes=summary['scenes'],logs=summary['logs'],training_seed=42,sampling_seed=42,
            precision='FP32',FM_steps=10,candidates=1,checkpoint_sha256=cp['sha256'],
            training_source=cp['training_source_sha'],evaluation_source=export['evaluation_source'],
            score_summary=str(path),summary_sha256=sha(path),scene_csv_sha256=sha(path.parent/'scenes.csv'))
        rows[key]=record;populations[key]=collapsed
    ordered=sorted(rows.values(),key=lambda r:(r['split'],r['updates'],ARMS.index(r['arm'])))
    write_csv(out/'ALL_RESULTS.csv',ordered)
    for split,name in [('navtest','NAVTEST_HISTORY.csv'),('dev','DEVELOPMENT_HISTORY.csv')]:
        write_csv(out/name,[r for r in ordered if r['split']==split])
    latest=[]
    for arm in ARMS:
        item={'arm':arm}
        for split in ('navtest','dev'):
            r=max((r for r in ordered if r['arm']==arm and r['split']==split),key=lambda x:x['updates'])
            item.update({split+'_updates':r['updates'],split+'_PDMS':r['PDMS'],
                         split+'_zero_scenes':r['zero_scenes'],split+'_failed':r['failed']})
        latest.append(item)
    write_csv(out/'LATEST.csv',latest)
    comparisons={}
    pairs=[('C1','C0',100000),('C4','C1',100000),('C4','C0',100000),
           ('S1','S0',50000),('S3','S2',50000),('S2','S0',50000),('S3','S1',50000),('S4','S3',50000)]
    for split in ('navtest','dev'):
        for first,baseline,step in pairs:
            report,_,_=paired_difference(populations[(split,first,step)],populations[(split,baseline,step)])
            comparisons[f'{split}_{first}-{baseline}_{step}']=report
    atomic(out/'PAIRED_COMPARISONS.json',comparisons)
    atomic(out/'IDENTITY.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        score_contracts=contracts,source_summaries={r['score_summary']:r['summary_sha256'] for r in ordered},
        complete_evaluations=len(rows),navtest_evaluations=sum(r['split']=='navtest' for r in ordered),
        failed_scenes=sum(r['failed'] for r in ordered),script_sha256=sha(__file__),
        new_optimizer_updates=0,new_model_inference=0,
        limitations='One training seed and one sampling seed. Different checkpoints/series are not equal-exposure causal ablations. Repeated Navtest observations are not a blind final test.'))
    def table(split,arms):
        lines=['| updates | '+' | '.join(arms)+' |','|---:|'+'---:|'*len(arms)]
        for step in sorted({r['updates'] for r in ordered if r['split']==split and r['arm'] in arms}):
            lines.append('| '+str(step)+' | '+' | '.join(f'{rows[(split,a,step)]["PDMS"]:.4f}' if (split,a,step) in rows else '—' for a in arms)+' |')
        return '\n'.join(lines)
    text='# C/S results: complete canonical evaluations\n\n'+read(out/'IDENTITY.json')['created_utc']+'\n\n'
    text+='Same per-split scene/cache/scoring/sampling contracts verified against every complete CSV. PDMS units0–100; all failures retained. No new inference or training.\n\n'
    for split in ('navtest','dev'):
        for names in (ARMS[:3],ARMS[3:]):
            text+='## '+split+' '+names[0][0]+' series\n\n'+table(split,names)+'\n\n'
    text+='## Matched-checkpoint Navtest differences\n\n| Pair | Delta PDMS points | Log-cluster95% interval |\n|---|---:|---|\n'
    for key,report in comparisons.items():
        if key.startswith('navtest'):
            m=report['metrics']['score'];lo,hi=m['log_cluster_95_interval']
            text+=f'| {key} | {100*m["delta"]:+.4f} | [{100*lo:+.4f}, {100*hi:+.4f}] |\n'
    text+='\nC0/C1/C4 use144W and legacy single-future supervision; target sizes128×96,256×192/pool2,512×384/pool4. S0–S4 fix C1 current targets: S0 frame sequence; S1 frame sequence+GT auxiliary action; S2 video; S3 video+GT auxiliary action; S4 S3+directW planner. All retain currentDINO and frozenGTMAE. C2/C3/C5 have no formal model results in these campaigns.\n\n'
    text+='C/S future target normalization, exposure and calibrated weights differ. Do not attribute a cross-series difference to a single mechanism. All figures here use one training and one inference seed42, original10FMsteps, no oracle/scorer. Historical89.41 is external unverified context and excluded.\n'
    (out/'SUMMARY.md').write_text(text)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,split in zip(axes,('dev','navtest')):
        for arm in ARMS:
            part=sorted((r for r in ordered if r['split']==split and r['arm']==arm),key=lambda r:r['updates'])
            ax.plot([r['updates']/1000 for r in part],[r['PDMS'] for r in part],
                    marker='o',linestyle='--' if arm.startswith('C') else '-',label=arm)
        ax.set(xlabel='Optimizer updates (thousands)',ylabel='PDMS (points)',title=split)
        ax.grid(alpha=.25);ax.legend(ncol=2)
    fig.suptitle('C / S experiments — fixed seed42, FP32, 10 FM steps');fig.tight_layout()
    fig.savefig(out/'PDMS_CURVES.svg');plt.close(fig)
    print(json.dumps({'evaluations':len(rows),'navtest':sum(r['split']=='navtest' for r in ordered),'latest':latest}))


if __name__=='__main__':main()
