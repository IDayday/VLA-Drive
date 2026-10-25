"""Seal current observations, current annotations and train-only teacher clips.

The fixed DEV/NAVTEST populations are unchanged. Whole physical DEV logs are
embargoed from TRAIN before any score/target inspection. Future images live in
teacher_frames only; observation() never returns that field.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from starVLA.rl.qwen_world_postcot.contracts import file_sha, fingerprint

POPULATION=Path('/mnt/project/flow-safe-improvement-20261013/population/manifest.json')
RAW=Path('/mnt/project/DriveDreamer-Policy/navsim_raw')
CAMERAS=('cam_l0','cam_f0','cam_r0')
CLASSES=('vehicle','pedestrian','bicycle','traffic_cone','barrier','czone_sign','generic_object')

def physical(log):
    return re.sub(r'_\d+_\d+$','',log)

def atomic(path, value):
    path=Path(path);tmp=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    with tmp.open('w') as f:
        json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def current_objects(frame, image_sizes):
    """Current annotated box centers in ego; conservative observed positives.

    Visibility uses calibrated pinhole projection (same as NAVSIM visualization)
    and requires a center inside a real input image. Occlusion is unknown: no
    unmatched query is declared a negative. Lidar is not a model input.
    """
    q=frame['lidar2ego_rotation'];r=Rotation.from_quat([q[1],q[2],q[3],q[0]]).as_matrix()
    t=np.asarray(frame['lidar2ego_translation'],dtype=np.float64)
    boxes=[];classes=[];depths=[];visible_views=[]
    for box,name in zip(frame['anns']['gt_boxes'],frame['anns']['gt_names']):
        if name not in CLASSES or not np.isfinite(box).all() or np.any(box[3:6]<=0):continue
        views=[];ds=[]
        for camera in CAMERAS:
            cam=frame['cams'][camera.upper()];width,height=image_sizes[camera]
            camxyz=(np.asarray(box[:3])-np.asarray(cam['sensor2lidar_translation']))@np.asarray(cam['sensor2lidar_rotation'])
            if camxyz[2]<=.1:ds.append(None);continue
            pixel=np.asarray(cam['cam_intrinsic'])@camxyz;pixel=pixel[:2]/pixel[2]
            ds.append(float(camxyz[2]))
            if 4<=pixel[0]<width-4 and 4<=pixel[1]<height-4:views.append(camera)
        if not views:continue
        center=r@box[:3]+t;direction=r@np.array([np.cos(box[6]),np.sin(box[6]),0.])
        yaw=np.arctan2(direction[1],direction[0])
        boxes.append([*center,*box[3:6],np.sin(yaw),np.cos(yaw)])
        classes.append(CLASSES.index(name));depths.append(ds);visible_views.append(views)
    order=sorted(range(len(boxes)),key=lambda i:(float(np.linalg.norm(boxes[i][:2])),i))[:32]
    return dict(boxes=[boxes[i] for i in order],classes=[classes[i] for i in order],
        valid=[True]*len(order),coverage_valid=False,classes_schema=list(CLASSES),
        camera_depths_m=[depths[i] for i in order],visible_views=[visible_views[i] for i in order],
        coordinate='ego_xyz_lwh_sinyaw_cosyaw',units='meters_radians',
        visibility='current_calibrated_center_frustum; occlusion_unknown; unmatched_not_negative',
        annotation_source='current_frame_anns_only',lidar_is_training_annotation_transform_only=True)

SOURCE=None
def build_group(args):
    log, tokens = args
    source=SOURCE
    frames=pickle.load(open(RAW/'navsim_logs/trainval'/f'{log}.pkl','rb'))
    by_token={f['token']:i for i,f in enumerate(frames)};rows={};sizes={}
    for token in tokens:
        row=dict(source[token]);index=by_token[token];f=frames[index]
        if int(row['timestamp'])!=int(f['timestamp']):raise ValueError('Current timestamp mismatch')
        if not sizes:
            for camera in CAMERAS:
                with Image.open(row['images'][camera]) as image:sizes[camera]=image.size
        teacher=[]
        for offset in range(-3,9):
            j=index+offset;g=frames[j] if 0<=j<len(frames) else None
            timestamp=None if g is None else int(g['timestamp'])
            timing_ok=g is not None and abs((timestamp-int(f['timestamp']))/1e6-offset*.5)<.06
            images={c:None if not timing_ok else str(RAW/'sensor_blobs/trainval'/g['cams'][c.upper()]['data_path']) for c in CAMERAS}
            images={c:p if p is not None and Path(p).is_file() else None for c,p in images.items()}
            teacher.append(dict(offset=offset,timestamp=timestamp,images=images,timing_valid=timing_ok))
        row['teacher_frames']=teacher;row['objects']=current_objects(f,sizes)
        row['source_record_hash']=fingerprint({k:row[k] for k in ('token','log','timestamp','images','proprio','navigation','teacher_frames')})
        rows[token]=row
    return rows

def main():
    global SOURCE
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',required=True);p.add_argument('--workers',type=int,default=12)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    pop=json.loads(POPULATION.read_text());embargo={physical(pop['token_logs'][t]) for t in pop['dev_tokens']+pop['navtest_tokens']}
    train=[t for t in pop['train_tokens'] if physical(pop['token_logs'][t]) not in embargo]
    train_set=set(train)
    identity=dict(schema='world_postcot_roles_v1',population_sha256=file_sha(POPULATION),
        train_tokens=train,dev_tokens=pop['dev_tokens'],navtest_tokens=pop['navtest_tokens'],
        physical_log_rule='strip final two numeric segment endpoints',
        train_physical_logs=sorted({physical(pop['token_logs'][t]) for t in train}),
        dev_physical_logs=sorted({physical(pop['token_logs'][t]) for t in pop['dev_tokens']}),
        navtest_physical_logs=sorted({physical(pop['token_logs'][t]) for t in pop['navtest_tokens']}),
        embargoed_train_tokens=[t for t in pop['train_tokens'] if t not in train_set],
        selected_before_score_inspection=True,old_base_may_have_seen_DEV_physical_logs=True)
    for role,tokens,src in [('TRAIN',train,'TRAIN_v3'),('DEV',pop['dev_tokens'],'DEV_v2')]:
        dest=out/role;dest.mkdir(exist_ok=True)
        if (dest/'COMPLETE').exists():
            previous=json.loads((dest/'manifest.json').read_text())
            if previous['tokens']!=tokens or previous['role_partition_sha256']!=fingerprint(identity):raise ValueError('Existing sealed data identity mismatch')
            print(json.dumps(dict(role=role,status='REUSED',count=len(tokens))),flush=True);continue
        with open('/mnt/project/qwen-sourceflow-native-20261018/data/'+src+'/records.pkl','rb') as f:source=pickle.load(f)
        grouped={}
        for t in tokens:grouped.setdefault(pop['token_logs'][t],[]).append(t)
        rows={};SOURCE=source
        with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('fork')) as pool:
            for part in pool.map(build_group,grouped.items()):rows.update(part)
        tmp=dest/'records.pkl.building'
        with tmp.open('wb') as f:pickle.dump(rows,f,protocol=5);f.flush();os.fsync(f.fileno())
        os.replace(tmp,dest/'records.pkl')
        manifest=dict(schema='world_postcot_data_v1',cohort=role,count=len(tokens),tokens=tokens,
            records_file='records.pkl',records_sha256=file_sha(dest/'records.pkl'),population_sha256=file_sha(POPULATION),
            role_partition_sha256=fingerprint(identity),teacher_frames_schema='offsets_-3_through_8_current_at_3_2Hz',
            current_camera_order=CAMERAS,current_object_classes=CLASSES,
            source_raw_manifest_sha256=file_sha(Path('/mnt/project/qwen-sourceflow-native-20261018/data')/src/'manifest.json'))
        atomic(dest/'manifest.json',manifest);atomic(dest/'COMPLETE',dict(manifest_sha256=file_sha(dest/'manifest.json'),status='COMPLETE'))
        print(json.dumps(dict(role=role,status='COMPLETE',count=len(tokens),bytes=(dest/'records.pkl').stat().st_size)),flush=True)
    atomic(out/'DATA_PARTITION.json',identity)
    hard=json.load(open('/mnt/project/simwam-source-process-20261018/source_manifest.json'))
    candidates=hard.get('tokens',hard.get('train_tokens',hard.get('hard_tokens')))
    if candidates is None:raise ValueError('Unknown source hard manifest schema')
    selected=[t for t in candidates if t in train_set]
    atomic(out/'RL_HARD.json',dict(schema='world_postcot_rl_tokens_v1',tokens=selected,count=len(selected),
        parent_manifest_sha256=file_sha('/mnt/project/simwam-source-process-20261018/source_manifest.json'),
        role_partition_sha256=fingerprint(identity),train_manifest_sha256=file_sha(out/'TRAIN/manifest.json'),
        description='original hard population intersect whole-physical-log-disjoint TRAIN'))

if __name__=='__main__':main()
