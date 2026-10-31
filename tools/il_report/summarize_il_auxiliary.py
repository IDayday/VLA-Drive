"""Actual eight-rank IL auxiliary trends; no forwards or training mutations."""
import argparse
import json
from pathlib import Path
import statistics
from datetime import datetime, timezone


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--window',type=int,default=500)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    result={}
    for arch in'ABCD':
        values={};run=a.root/'runs'/('ARCH_'+arch)/'IL_seed42'
        for rank in range(8):
            last=0
            with(run/f'metrics_rank{rank}.jsonl').open()as stream:
                for line in stream:
                    try:r=json.loads(line)
                    except json.JSONDecodeError:continue  # A live final write may be incomplete.
                    u=r['updates']
                    if u!=last+1:raise ValueError('Missing/duplicate/disordered effective IL update')
                    last=u
                    if r['cursor']!=32*u or r['optimizer_global_steps']!=u or r['optimizer_skipped_steps']!=0:
                        raise ValueError('Optimizer/exposure contract changed')
                    v=values.setdefault(u,dict(ranks=0,losses={},grad_norm=0.));v['ranks']+=1
                    for k,x in r['losses'].items():v['losses'][k]=v['losses'].get(k,0)+x/8
                    v['grad_norm']+=r['grad_norm']/8
        end=max(u for u,v in values.items()if v['ranks']==8);rows=[]
        for e in sorted(set(x for x in[2000,5000,10000,15000,20000,25000,30000,end]if x<=end)):
            ids=[u for u in range(max(1001,e-a.window+1),e+1)if values.get(u,{}).get('ranks')==8]
            if not ids:continue
            loss={k:statistics.mean(values[u]['losses'][k]for u in ids)for k in values[ids[0]]['losses']}
            gradients=[values[u]['grad_norm']for u in ids]
            rows.append(dict(end=e,first=ids[0],n=len(ids),losses=loss,
                mean_global_grad_norm=statistics.mean(gradients),fraction_preclip_norm_above1=sum(x>1 for x in gradients)/len(gradients)))
        start=next(r for r in rows if r['end']==10000);last=rows[-1]
        result[arch]=dict(latest_common_update=end,windows=rows,
            relative_change_from10k={k:last['losses'][k]/start['losses'][k]-1 for k in last['losses']})
    output=dict(status='PASS',generated_utc=datetime.now(timezone.utc).isoformat(),arms=result,
        notes=['Mean of eight local world/global-denominator-scaled loss contributions recovers each logical-update loss.',
               '500-update windows with changing TRAIN scenes, noise and candidate labels; not a fixed-panel learning curve.',
               'Scalar loss magnitude and global clipping cannot establish component gradient dominance or conflict.'],
        optimizer_updates=0,new_scores=0,new_forward_calls=0)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:dict(last=v['latest_common_update'],relative_change_from10k=v['relative_change_from10k'])for k,v in result.items()}))


if __name__=='__main__':main()
