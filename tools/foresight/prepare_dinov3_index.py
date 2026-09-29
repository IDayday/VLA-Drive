"""Read-only current/future timestamp index, with shared image entities per split."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import subprocess
from tools.ddpolicy_vehicle.prepare_data import load_trusted, camera_path, CAMERAS, atomic_json
from starVLA.model.modules.vehicle_joint.initialization import identity_hash, file_sha256
from .prepare_vehicle_trajectories import timed_frames


def one_log(job):
    log, tokens, split, cfg = job
    frames = load_trusted(Path(cfg['raw_log_root'])/(log+'.pkl'))
    positions = {f['token']:i for i,f in enumerate(frames)}
    if len(positions) != len(frames): raise ValueError('Duplicate frame token')
    images, scenes = {}, []
    for token in tokens:
        current = json.loads((Path(cfg[split+'_data'])/'current'/(token+'.json')).read_text())
        at = positions[token]; frame = frames[at]
        if current['timestamp'] != int(frame['timestamp']): raise ValueError('Current decision time mismatch')
        selected = [frame, *timed_frames(frames, at, [1.,2.,4.], cfg['tolerance'])]
        refs, actual = [], []
        for h, target in enumerate(selected):
            row = []
            actual.append(None if target is None else (int(target['timestamp'])-int(frame['timestamp']))/1e6)
            for v, cam in enumerate(CAMERAS):
                if target is None: row.append(None); continue
                try: path = camera_path(target['cams'][cam]['data_path'], cfg)
                except FileNotFoundError:
                    if h == 0: raise
                    row.append(None); continue
                if h == 0 and path.resolve() != Path(current['image_paths'][v]).resolve():
                    raise ValueError('h=0 differs from student input image')
                entry = {'split':split,'log':log,'path':str(path),'timestamp':int(target['timestamp']),'view':v}
                key = identity_hash(entry); st = path.stat()
                entry.update(key=key,bytes=st.st_size,mtime_ns=st.st_mtime_ns)
                if key in images and images[key] != entry: raise ValueError('Image identity conflict')
                images[key] = entry; row.append(key)
            refs.append(row)
        scenes.append({'token':token,'log':log,'split':split,'timestamp':int(frame['timestamp']),
                       'requested_horizons_s':[0.,1.,2.,4.],'actual_horizons_s':actual,'images':refs})
    return images, scenes, {'log':log,'raw_log_sha256':file_sha256(Path(cfg['raw_log_root'])/(log+'.pkl'))}


def main():
    p=argparse.ArgumentParser(__doc__)
    for k in ('train-data','dev-data','split-manifest','raw-log-root','sensor-root','output'):p.add_argument('--'+k,required=True)
    p.add_argument('--fallback-sensor-root',action='append',default=[])
    p.add_argument('--workers',type=int,default=8);p.add_argument('--tolerance',type=float,default=.05)
    a=p.parse_args();out=Path(a.output)
    if out.exists():raise FileExistsError('New immutable index directory required')
    if not 1<=a.workers<=16 or not 0<a.tolerance<=.05:raise ValueError('Invalid bounded index settings')
    out.mkdir(parents=True);partition=json.loads(Path(a.split_manifest).read_text());jobs=[];indices={}
    for split in ('train','dev'):
        root=Path(getattr(a,split+'_data'));index=json.loads((root/'index.json').read_text());indices[split]=index
        if [r['token'] for r in index]!=partition[split+'_tokens']:raise ValueError('Scene population/order changed')
        groups=defaultdict(list)
        for row in index:groups[row['log']].append(row['token'])
        jobs.extend((log,tokens,split,vars(a)) for log,tokens in groups.items())
    if {r['log'] for r in indices['train']}&{r['log'] for r in indices['dev']}:raise ValueError('Log leakage')
    images={};scenes={};logs=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for imgs,rows,audit in pool.map(one_log,jobs):
            images.update(imgs);scenes.update({r['token']:r for r in rows});logs.append(audit)
            atomic_json(out/'progress.json',{'scenes':len(scenes),'images':len(images),'logs':len(logs)})
    entities=sorted(images.values(),key=lambda r:(r['split'],r['log'],r['timestamp'],r['view']))
    lookup={r['key']:i for i,r in enumerate(entities)}
    atomic_json(out/'images.json',entities)
    hashes={'images':file_sha256(out/'images.json')}
    for split,index in indices.items():
        rows=[scenes[r['token']] for r in index]
        for row in rows:row['images']=[[lookup[k] if k is not None else -1 for k in h] for h in row['images']]
        atomic_json(out/(split+'_scenes.json'),rows);hashes[split]=file_sha256(out/(split+'_scenes.json'))
    identity={'schema':'dinov3_image_index_v1','partition_sha256':file_sha256(a.split_manifest),
        'index_hashes':{k:identity_hash(v) for k,v in indices.items()},'files':hashes,'timestamp_tolerance_s':a.tolerance,
        'horizons_s':[0.,1.,2.,4.],'views':list(CAMERAS),'raw_logs':logs,
        'writer_sha256':file_sha256(__file__),'source_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
    identity['identity']=identity_hash(identity);atomic_json(out/'identity.json',identity)
    atomic_json(out/'COMPLETE.json',{'scenes':{k:len(v) for k,v in indices.items()},'unique_images':len(entities),
        'failures':0,'estimate_feature_bytes_fp16_grid16x29x1024':len(entities)*16*29*1024*2,
        'current_refs':sum(sum(i>=0 for i in r['images'][0]) for r in scenes.values()),
        'future_refs':sum(sum(i>=0 for h in r['images'][1:] for i in h) for r in scenes.values())})

if __name__=='__main__':main()
