"""Immutable eight-real-frame index; shared source population for both teachers."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import subprocess
from tools.ddpolicy_vehicle.prepare_data import load_trusted, camera_path, CAMERAS, atomic_json
from tools.foresight.prepare_vehicle_trajectories import timed_frames
from starVLA.model.modules.vehicle_joint.initialization import identity_hash, file_sha256

TIMES=[.5*i for i in range(1,9)]


def one_log(job):
    log,tokens,split,cfg=job
    path=Path(cfg['raw_log_root'])/(log+'.pkl');frames=load_trusted(path)
    positions={f['token']:i for i,f in enumerate(frames)}
    if len(positions)!=len(frames):raise ValueError('Duplicate raw frame token')
    scenes=[]
    for token in tokens:
        current=json.loads((Path(cfg[split+'_data'])/'current'/(token+'.json')).read_text())
        at=positions[token];frame=frames[at]
        if int(frame['timestamp'])!=current['timestamp']:raise ValueError('Wrong decision timestamp')
        future=timed_frames(frames,at,TIMES,cfg['tolerance']);views=[];current_views=[]
        actual=[None if f is None else (int(f['timestamp'])-int(frame['timestamp']))/1e6 for f in future]
        for view,cam in enumerate(CAMERAS):
            cp=camera_path(frame['cams'][cam]['data_path'],cfg)
            if cp.resolve()!=Path(current['image_paths'][view]).resolve():raise ValueError('Current image mismatch')
            current_views.append(str(cp));refs=[]
            for f in future:
                if f is None:refs.append(None);continue
                try:ip=camera_path(f['cams'][cam]['data_path'],cfg)
                except FileNotFoundError:refs.append(None);continue
                st=ip.stat();refs.append({'path':str(ip),'timestamp':int(f['timestamp']),'bytes':st.st_size,'mtime_ns':st.st_mtime_ns})
            views.append(refs)
        scenes.append({'token':token,'log':log,'split':split,'timestamp':int(frame['timestamp']),
            'requested_times_s':TIMES,'actual_times_s':actual,'current_paths':current_views,
            'frames_by_view':views,'clip_view_valid':[all(x is not None for x in v) for v in views]})
    return scenes,{'log':log,'split':split,'source_sha256':file_sha256(path)}


def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ('train-data','dev-data','split-manifest','raw-log-root','sensor-root','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--fallback-sensor-root',action='append',default=[]);p.add_argument('--workers',type=int,default=8)
    p.add_argument('--tolerance',type=float,default=.05);a=p.parse_args()
    if not 1<=a.workers<=16 or not 0<a.tolerance<=.05:raise ValueError('Bounded timestamp/index settings required')
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if subprocess.check_output(['git','status','--porcelain']).strip():raise ValueError('Freeze source before indexing')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);(out/'logs').mkdir(exist_ok=True)
    indices={s:json.loads((Path(getattr(a,s+'_data'))/'index.json').read_text()) for s in ('train','dev')}
    partition=json.loads(Path(a.split_manifest).read_text())
    if {r['log'] for r in indices['train']}&{r['log'] for r in indices['dev']}:raise ValueError('Log overlap')
    for s in indices:
        if [r['token'] for r in indices[s]]!=partition[s+'_tokens']:raise ValueError('Population changed')
    registration={'schema':'action_video_clip_index_registration_v1','source':source,'args':vars(a),
        'partition_sha256':file_sha256(a.split_manifest),'index_hashes':{s:identity_hash(v) for s,v in indices.items()}}
    if (out/'registration.json').exists() and json.loads((out/'registration.json').read_text())!=registration:raise ValueError('Resume index changed')
    atomic_json(out/'registration.json',registration)
    jobs=[];rows={};logs=[]
    for s,index in indices.items():
        grouped=defaultdict(list)
        for r in index:grouped[r['log']].append(r['token'])
        for log,tokens in grouped.items():
            saved=out/'logs'/f'{s}_{log}.json'
            if saved.exists():
                value=json.loads(saved.read_text());rows.update({r['token']:r for r in value['scenes']});logs.append(value['audit'])
            else:jobs.append((log,tokens,s,vars(a)))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for scenes,audit in pool.map(one_log,jobs):
            atomic_json(out/'logs'/f"{audit['split']}_{audit['log']}.json",{'scenes':scenes,'audit':audit})
            rows.update({r['token']:r for r in scenes});logs.append(audit)
            atomic_json(out/'progress.json',{'scenes':len(rows),'logs':len(logs),'remaining_logs':len(jobs)-len(logs)})
    hashes={};coverage={}
    for s,index in indices.items():
        selected=[rows[r['token']] for r in index];atomic_json(out/(s+'_scenes.json'),selected)
        hashes[s]=file_sha256(out/(s+'_scenes.json'))
        coverage[s]={'scenes':len(selected),'valid_views':sum(sum(r['clip_view_valid']) for r in selected),
            'all_views_valid':sum(all(r['clip_view_valid']) for r in selected),'no_valid_clip':sum(not any(r['clip_view_valid']) for r in selected)}
    identity={'schema':'action_video_clip_index_v1','partition_sha256':registration['partition_sha256'],
        'index_hashes':registration['index_hashes'],'files':hashes,'times_s':TIMES,'views':list(CAMERAS),
        'timestamp_tolerance_s':a.tolerance,'source':source,'raw_logs':sorted(logs,key=lambda r:(r['split'],r['log'])),
        'missing_policy':'whole view invalid if any real frame missing; retain ego scene'}
    identity['identity']=identity_hash(identity);atomic_json(out/'identity.json',identity)
    atomic_json(out/'COMPLETE.json',{'identity':identity['identity'],'coverage':coverage,'failures':0})

if __name__=='__main__':main()
