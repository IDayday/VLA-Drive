"""Verify old current-only C1 targets correspond to this exact native TRAIN."""
import argparse,json,time
from pathlib import Path
from starVLA.rl.qwen_sourceflow.temporal_data import TemporalSceneData
from starVLA.rl.qwen_sourceflow.temporal_contracts import ROOT,TEACHER_ID
from starVLA.rl.qwen_sourceflow.contracts import require_dispatch,file_sha,fingerprint
from starVLA.rl.flow_grpo.transactions import atomic_json


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',required=True)
    p.add_argument('--teacher',default='/var/tmp/structured-world-fgtr-round1-20261008/navsim_current_C1_only_v1')
    p.add_argument('--index',default='/mnt/project/ddp-full-foresight-study-artifacts/20260929/dino_index_v1')
    a=p.parse_args();host=require_dispatch();start=time.monotonic();data=TemporalSceneData(ROOT/'data/TRAIN_v3',cohort='TRAIN')
    t=Path(a.teacher);idx=Path(a.index);identity=json.loads((t/'identity.json').read_text());done=json.loads((t/'COMPLETE.json').read_text())
    ii=json.loads((idx/'identity.json').read_text())
    if identity['identity']!=TEACHER_ID or done['identity']!=TEACHER_ID or not done['current_only']:raise ValueError('Wrong C1')
    if file_sha(idx/'train_scenes.json')!=identity['source_teacher_index_sha256'] or file_sha(idx/'train_scenes.json')!=ii['files']['train']:
        raise ValueError('Teacher source scene index changed')
    if file_sha(idx/'images.json')!=ii['files']['images']:raise ValueError('Teacher image index changed')
    images=json.loads((idx/'images.json').read_text());scenes={r['token']:r for r in json.loads((idx/'train_scenes.json').read_text())}
    tokens={p.stem for p in (t/'records').glob('*.json')}
    if tokens!=set(data.tokens) or done['scenes']!=len(data.tokens):raise ValueError('Teacher population differs from exact native TRAIN')
    targets={}
    for token in data.tokens:
        raw=data.records[token];scene=scenes[token];meta=json.loads((t/'records'/(token+'.json')).read_text())
        if scene['log']!=raw['log'] or scene['timestamp']!=raw['timestamp'] or scene['actual_horizons_s'][0]!=0.:
            raise ValueError('Teacher is not this scene current frame')
        if meta['identity']!=TEACHER_ID or meta['legacy_current_image_indices']!=scene['images'][0]:raise ValueError('Teacher temporal selector changed')
        for camera,index in zip(['cam_f0','cam_l0','cam_r0'],scene['images'][0]):
            if raw['images'][camera] is not None and Path(raw['images'][camera]).parts[-3:]!=Path(images[index]['path']).parts[-3:]:
                raise ValueError('Teacher image field/camera mismatches native current observation')
        if not (t/'targets'/(token+'.npz')).is_file():raise FileNotFoundError(token)
        targets[token]=meta['sha256']
    record=dict(schema='native_temporal_current_teacher_manifest_v1',status='COMPLETE',host=host,
        teacher_identity=TEACHER_ID,teacher_identity_file_sha256=file_sha(t/'identity.json'),
        data_sha256=data.sha,scenes=len(tokens),tokens_sha256=fingerprint(data.tokens),
        camera_source=['cam_f0','cam_l0','cam_r0'],camera_actor=['cam_l0','cam_f0','cam_r0'],reorder=[1,0,2],
        current_only=True,targets=targets,source_index_sha256=file_sha(idx/'identity.json'),wall_seconds=time.monotonic()-start)
    record['sha256']=fingerprint(record);atomic_json(a.output,record)
    print(json.dumps({k:v for k,v in record.items() if k!='targets'}))


if __name__=='__main__':main()
