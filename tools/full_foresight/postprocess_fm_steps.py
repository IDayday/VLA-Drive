"""Wait for the frozen-step campaign, then add offline ego fit and complete CSV."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import time

from .navtest_schedule import atomic,read,sha,source_identity,lease
from .navtest_milestones import environment


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--registration',required=True);p.add_argument('--max-seconds',type=int,default=21600)
    a=p.parse_args();registration=read(a.registration);config=registration['config']
    root=Path(config['artifact_root']);out=root/'postprocess';out.mkdir(exist_ok=True)
    source=source_identity(Path.cwd());start=time.time()
    with lease(out/'postprocess.lock'):
        try:
            atomic(out/'status.json',{'status':'WAITING','source':source,'sampling_source':registration['source_sha'],
                                    'pid':os.getpid(),'gpu_count':0,'real_optimizer_updates':0})
            while True:
                status=read(root/'status.json')
                if status['status']=='COMPLETE':break
                if status['status']!='RUNNING':raise RuntimeError(f'Step campaign {status["status"]}; preserve partial results')
                if time.time()-start>a.max_seconds:raise TimeoutError('Step campaign not complete')
                time.sleep(10)
            results=read(root/'RESULTS.json')
            if results['registration']!=registration['identity'] or results['training_updates']!=0:
                raise ValueError('Sweep identity changed')
            atomic(out/'status.json',{'status':'RUNNING','source':source,'gpu_count':0,'real_optimizer_updates':0})
            def fit(n):
                job=root/f'steps_{n:02d}';target=job/'ego_fit'
                if not target.exists():
                    args=[config['inference_python'],'-m','tools.foresight.evaluate_ego',
                          '--predictions',str(job/'predictions'),'--current-root',config['current_root'],
                          '--processed-root',config['ego_labels'],'--output',str(target)]
                    with (job/'ego_fit.log').open('x') as log:
                        subprocess.run(args,env=environment(),stdout=log,stderr=subprocess.STDOUT,
                                       check=True,timeout=a.max_seconds)
                summary=read(target/'summary.json')
                if not summary['valid'] or summary['failed'] or summary['scenes']!=12146:
                    raise ValueError('Failed ego rows retained; not a complete result')
                return n,summary
            with ThreadPoolExecutor(max_workers=3) as pool:
                fits=dict(pool.map(fit,config['steps']))
            for row in results['rows']:
                n=row['steps'];fit_summary=fits[n]['groups']['all']
                row.update({k:fit_summary[k] for k in ('ADE','FDE','yaw_MAE_rad','yaw_endpoint_rad')})
                interval=results['paired_vs10'][str(n)]['metrics']['score']['log_cluster_95_interval']
                row.update(delta_log95_lower_points=interval[0]*100,delta_log95_upper_points=interval[1]*100)
            with (out/'RESULTS_WITH_EGO.csv').open('w') as stream:
                writer=csv.DictWriter(stream,list(results['rows'][0]));writer.writeheader();writer.writerows(results['rows'])
            destination=out/'ALL_SCENES_WITH_EGO.csv';count=0;keys=None
            with destination.with_suffix('.tmp').open('w') as stream:
                for n in config['steps']:
                    job=root/f'steps_{n:02d}'
                    with (job/'scores/scenes.csv').open() as f:score_rows=list(csv.DictReader(f))
                    with (job/'ego_fit/scenes.csv').open() as f:ego={r['token']:r for r in csv.DictReader(f)}
                    if len(score_rows)!=12146 or len(ego)!=12146:raise ValueError('Incomplete paired population')
                    for row in score_rows:
                        token=row['token'];item=ego[token]
                        if row['status']!='ok' or item['failure'] or item['log']!=row['log']:
                            raise ValueError('Failed/mismatched scene rows retained')
                        metadata=read(job/'predictions/predictions'/(token+'.json'))
                        row.update(steps=n,dt=1/n,checkpoint_update=registration['checkpoint']['completed'],
                                   sampling_seed=config['sampling_seed'],precision='FP32',
                                   **{k:item[k] for k in ('ADE','FDE','yaw_MAE_rad','yaw_endpoint_rad','motion_group')},
                                   solver_seconds=metadata['solver_seconds'],encode_current_seconds=metadata['encode_current_seconds'])
                        if any(not math.isfinite(float(row[k])) for k in ('score','ADE','FDE')):
                            raise ValueError('Nonfinite scene output')
                        if keys is None:
                            keys=list(row);writer=csv.DictWriter(stream,keys);writer.writeheader()
                        writer.writerow(row);count+=1
            destination.with_suffix('.tmp').replace(destination)
            if count!=12146*len(config['steps']):raise ValueError('Incorrect complete CSV denominator')
            results.update(postprocess_source=source,ego_fit=fits,complete_scene_rows=count,
                           complete_csv_sha256=sha(destination),postprocess_completed_unix=time.time())
            atomic(out/'RESULTS_WITH_EGO.json',results)
            atomic(out/'status.json',{'status':'COMPLETE','source':source,'gpu_count':0,'real_optimizer_updates':0,
                                     'scene_rows':count,'wall_seconds':time.time()-start})
        except BaseException as error:
            atomic(out/'status.json',{'status':'FAILED','source':source,'error':repr(error),'gpu_count':0,
                                     'real_optimizer_updates':0,'wall_seconds':time.time()-start})
            raise


if __name__=='__main__':main()
