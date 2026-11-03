"""Compare fixed same-D0 journal windows; CPU only, no model/score work."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

ROOT=Path('/mnt/project/d46-priority-rl-20261103')


def window(path,start,end,identity):
    selected=[json.loads(line)for line in Path(path).read_text().splitlines()]
    selected=[r for r in selected if start<=r['updates']<=end]
    if [r['updates']for r in selected]!=list(range(start,end+1)):
        raise ValueError('Missing or repeated committed updates')
    if any(r['run']!=identity['sha256']or r['active_batch']['scenes']!=64
           or r['active_batch']['group']!=16 for r in selected):
        raise ValueError('Foreign run or changed B64/G16')
    return selected


def summarize(rows):
    complete=[r for r in rows if r['inner_completed']==2]
    spans=[dict(start=a['updates'],end=b['updates'],seconds=b['at']-a['at'],
                scenes=b['scene_exposures']-a['scene_exposures'],
                checkpoint_between=a['updates']%100==0)
           for a,b in zip(complete,complete[1:])if b['updates']==a['updates']+2]
    steady=[s for s in spans if not s['checkpoint_between']]
    if not steady or any(s['seconds']<=0 or s['scenes']!=64 for s in spans):
        raise ValueError('Need complete positive-time B64 intervals')
    seconds=sum(s['seconds']for s in steady)
    scenes=sum(s['scenes']for s in steady)
    encoded=json.dumps(rows,sort_keys=True,separators=(',',':')).encode()
    return dict(updates=[rows[0]['updates'],rows[-1]['updates']],update_count=len(rows),
        optimizer_seconds_median=statistics.median(r['update_seconds']for r in rows),
        generation_seconds_median=statistics.median(r['collection_seconds_generation']for r in complete),
        scoring_seconds_median=statistics.median(r['collection_seconds_scoring']for r in complete),
        full_spans=spans,steady_collections=len(steady),steady_seconds=seconds,steady_scenes=scenes,
        steady_collection_seconds=seconds/len(steady),steady_scenes_per_second=scenes/seconds,
        all_span_scenes_per_second=sum(s['scenes']for s in spans)/sum(s['seconds']for s in spans),
        selected_journal_rows_sha256=hashlib.sha256(encoded).hexdigest())


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--old-start',type=int,default=1677);p.add_argument('--old-end',type=int,default=1714)
    p.add_argument('--new-start',type=int,default=1719);p.add_argument('--new-end',type=int,default=1796)
    p.add_argument('--output',required=True);a=p.parse_args()
    old=ROOT/'runs/D46_D0';new=ROOT/'runs/D46_D0_WORLD16_v2'
    oi=json.loads((old/'identity.json').read_text());ni=json.loads((new/'identity.json').read_text())
    if (oi['world'],ni['world'])!=(8,16)or oi['recipe']!=ni['recipe']or oi['scope']!=ni['scope']:
        raise ValueError('Need the same D0 recipe and parameter scope')
    if oi['qualification']or ni['qualification']or oi['recipe']['arm']!='D0' or oi['recipe']['inner_updates']!=2:
        raise ValueError('Only actual formal D0 B64/G16/M2')
    left=summarize(window(old/'metrics_rank0.jsonl',a.old_start,a.old_end,oi))
    right=summarize(window(new/'metrics_rank0.jsonl',a.new_start,a.new_end,ni))
    value=dict(schema='d46_fixed_window_actual_journal_throughput_v1',at=time.time(),
        world8=left,world16=right,optimizer_speedup=left['optimizer_seconds_median']/right['optimizer_seconds_median'],
        full_cycle_speedup=right['steady_scenes_per_second']/left['steady_scenes_per_second'],
        protocol=dict(B=64,G=16,M=2,population=103288,rank=0,
            clock_host='training-vlawm-tsinghua-zt5',
            old_window='After old I EXIT0 plus15s; exclusive8 GPUs',
            new_window='Skip first4 actual optimizer updates after native1714 transition',
            timing='Even-update timestamps include fresh rollout, scoring, two Adam updates and synchronization',
            exclusions='Startup, full NAVTEST, checkpoint-crossing intervals for steady statistics',
            limits='Different sequential TRAIN scenes; short steady-state windows, not a matched hardware ablation or long-run ETA guarantee'))
    dest=Path(a.output)
    if dest.exists():raise FileExistsError('Keep prior throughput evidence')
    dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(dict(output=str(dest),full_cycle_speedup=value['full_cycle_speedup'],
                         optimizer_speedup=value['optimizer_speedup'])))


if __name__=='__main__':main()
