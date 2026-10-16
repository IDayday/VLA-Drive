"""Bounded, registered CPU probes; no teacher/student training and no width choice.

Only current raw LiDAR/vehicle boxes are projected. Labels are NEVER model
inputs. This diagnostic is not a planning/PDMS result or a semantic-segmentation
benchmark: positive small-box patches require nearest-LiDAR support.
"""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from torch import nn

from starVLA.dataloader.st_world_history import file_sha256,resolve_camera_path
from starVLA.dataloader.st_world_schema import identity_hash
from starVLA.dataloader.st_world_targets import TeacherCacheReader,atomic_json
from tools.st_world_policy.prepare_history import TrustedNumpyUnpickler
from tools.st_world_policy.teacher_pca import TeacherPCA


def project(points, camera, original_wh, image_hw=(288,384)):
    """Raw source-camera distortion, then exact full-field pixel-center resize."""
    points=np.asarray(points,dtype=np.float64)
    R=np.asarray(camera['sensor2lidar_rotation'],dtype=np.float64)
    t=np.asarray(camera['sensor2lidar_translation'],dtype=np.float64)
    local=(points-t)@R
    finite=np.isfinite(local).all(1)&(local[:,2]>0)
    uv=np.full((len(points),2),np.nan,dtype=np.float64)
    if finite.any():
        uv[finite]=cv2.projectPoints(local[finite],np.zeros(3),np.zeros(3),
            np.asarray(camera['cam_intrinsic'],dtype=np.float64),
            np.asarray(camera['distortion'],dtype=np.float64))[0].reshape(-1,2)
        scale=np.array([image_hw[1]/original_wh[0],image_hw[0]/original_wh[1]])
        uv[finite]=(uv[finite]+.5)*scale-.5
    return uv,local[:,2],finite


def box_corners(box):
    x,y,z,length,width,height,yaw=np.asarray(box,dtype=np.float64)
    if not np.isfinite(box).all() or min(length,width,height)<=0:raise ValueError('Invalid current source box')
    local=np.array([[a,b,c] for a in [-.5,.5] for b in [-.5,.5] for c in [-.5,.5]])
    local*=np.array([length,width,height]);co,si=np.cos(yaw),np.sin(yaw)
    local[:,:2]=local[:,:2]@np.array([[co,si],[-si,co]])
    return local+np.array([x,y,z])


def current_labels(points, boxes, camera, original_wh):
    height,width=288,384;patch=16
    uv,z,front=project(points,camera,original_wh)
    good=front & np.isfinite(uv).all(1) & (z>=1)&(z<=80)
    pixel=np.rint(uv[good]).astype(np.int64);z=z[good]
    inside=(pixel[:,0]>=0)&(pixel[:,0]<width)&(pixel[:,1]>=0)&(pixel[:,1]<height)
    pixel,z=pixel[inside],z[inside]
    depth=np.full(height*width,np.inf,dtype=np.float32)
    np.minimum.at(depth,pixel[:,1]*width+pixel[:,0],z)
    depth=depth.reshape(height,width);known=np.isfinite(depth)
    target=np.zeros((18,24),dtype=np.float32);valid=np.zeros((18,24),dtype=bool)
    for y in range(18):
        for x in range(24):
            q=depth[y*16:(y+1)*16,x*16:(x+1)*16];ok=np.isfinite(q)
            if ok.sum()>=5:target[y,x]=np.median(q[ok]);valid[y,x]=True
    positive=np.zeros((18,24),dtype=bool);excluded=np.zeros((18,24),dtype=bool)
    stats={'current_boxes':len(boxes),'front_projected_boxes':0,'small_boxes':0,'supported_small_boxes':0,
           'straddling_camera_plane_boxes':0,'known_lidar_pixels':int(known.sum())}
    for box in boxes:
        corners=box_corners(box);box_uv,box_z,box_front=project(corners,camera,original_wh)
        if not box_front.any():continue
        if not box_front.all():
            # Avoid calling unbounded/invalid projected box regions background.
            excluded[:]=True;stats['straddling_camera_plane_boxes']+=1;continue
        lo=box_uv.min(0);hi=box_uv.max(0)
        if hi[0]<0 or hi[1]<0 or lo[0]>=width or lo[1]>=height:continue
        stats['front_projected_boxes']+=1
        x0,y0=np.maximum(np.floor(lo).astype(int),0);x1,y1=np.minimum(np.ceil(hi).astype(int)+1,[width,height])
        excluded[y0//patch:(y1-1)//patch+1,x0//patch:(x1-1)//patch+1]=True
        area=float(np.prod(hi-lo))
        if not 0<area<=1024:continue
        stats['small_boxes']+=1
        roi=depth[y0:y1,x0:x1]
        support=np.isfinite(roi)&(roi>=box_z.min()-.5)&(roi<=box_z.max()+.5)
        if support.sum()<3:continue
        stats['supported_small_boxes']+=1
        yy,xx=np.where(support);positive[(yy+y0)//patch,(xx+x0)//patch]=True
    negative=valid & ~excluded
    return {'depth':target,'depth_valid':valid,'small_vehicle':positive.astype(np.float32),
            'small_valid':positive|negative,'statistics':stats}


def validate_registration(registration):
    if registration.get('identity')!=identity_hash({k:v for k,v in registration.items() if k!='identity'}):
        raise ValueError('Probe registration changed')
    fixed = {
        'schema':'st_two_spatial_probe_registration_v1', 'dimensions':[256,384],
        'image_hw':[288,384], 'native_patch_hw':[16,16],
        'cameras':['CAM_F0','CAM_L0','CAM_R0'], 'max_scenes':{'train':64,'dev':16},
        'gpu_count':0,'navtest_access':False,'old_results_modified':False,
    }
    if any(registration.get(k)!=v for k,v in fixed.items()):
        raise ValueError('Unregistered geometry/population/probe family')
    depth=registration['depth'];small=registration['small_vehicle'];cfg=registration['readout']
    if (depth['range_m']!=[1,80] or depth['minimum_valid_pixels_per_patch']!=5 or
            small['maximum_projected_box_area_pixels']!=1024 or small['minimum_visible_lidar_pixels_per_box']!=3 or
            small['depth_tolerance_m']!=.5 or cfg['optimizer']!='AdamW' or
            cfg['architecture']!='independent single Linear(dimension,1) per probe'):
        raise ValueError('Unsupported registered label or readout definition')
    if not (type(cfg['updates']) is int and 1<=cfg['updates']<=300 and
            type(cfg['batch_patches']) is int and 1<=cfg['batch_patches']<=512 and
            type(registration['cpu_threads']) is int and 1<=registration['cpu_threads']<=4 and
            0<registration['max_wall_seconds']<=1200 and cfg['lr']>0 and
            cfg['weight_decay']>=0 and cfg['clip_grad_norm']>0):
        raise ValueError('Unregistered probe budget')


def prepare_labels(a,registration,readers,check_budget):
    from nuplan.database.utils.pointclouds.lidar import LidarPointCloud
    artifacts=Path(a.artifact_root);out=Path(a.output);result={}
    for split,count in [('train',64),('dev',16)]:
        manifest_path=artifacts/f'smoke_{split}{count}_pinned_v2/SELECTED_SCENES.json'
        if file_sha256(manifest_path)!=registration[split+'_manifest_sha256']:raise ValueError('Probe scene selection changed')
        selection=json.loads(manifest_path.read_text())
        if selection['split']!=split or len(selection['scenes'])!=registration['max_scenes'][split]:
            raise ValueError('Probe role/population mismatch')
        history=json.loads((artifacts/f'smoke_{split}{count}_v1/HISTORY_MANIFEST.json').read_text())
        log_hashes={x['log']:x['sha256'] for x in history['source_logs']}
        records=[]
        for row in selection['scenes']:
            check_budget()
            raw=Path(a.raw_log_root)/(row['log']+'.pkl')
            if file_sha256(raw)!=log_hashes[row['log']]:raise ValueError('Raw source log changed')
            with raw.open('rb') as f:frames=TrustedNumpyUnpickler(f).load()
            selected=[f for f in frames if f['token']==row['token']]
            if len(selected)!=1:raise ValueError('Ambiguous t0 frame')
            frame=selected[0];scene=readers[split].scenes[row['token']]
            if frame['timestamp']!=scene['timestamp_us']:raise ValueError('Probe teacher/current time mismatch')
            path=resolve_camera_path(frame['lidar_path'],[a.sensor_root,*a.fallback_sensor_root])
            if path is None:raise FileNotFoundError('Current LiDAR missing: '+frame['lidar_path'])
            points=LidarPointCloud.from_buffer(Path(path).read_bytes(),'pcd').points[:3].T.astype(np.float32)
            # Both boxes and source sensor2lidar calibration use the raw LiDAR
            # coordinate frame. Do not apply an unrelated ego/world transform.
            boxes=np.asarray(frame['anns']['gt_boxes'])[np.asarray(frame['anns']['gt_names'])=='vehicle']
            per_view=[]
            for camera,ref in zip(registration['cameras'],scene['frames'][0]['views']):
                cam=frame['cams'][camera]
                actual=resolve_camera_path(cam['data_path'],[a.sensor_root,*a.fallback_sensor_root])
                if actual is None or Path(actual).resolve()!=Path(ref['path']).resolve():raise ValueError('Probe view/image mismatch')
                if file_sha256(actual)!=ref['sha256']:raise ValueError('Probe original image changed')
                per_view.append(current_labels(points,boxes,cam,ref['original_wh']))
            tensors={key:np.stack([v[key] for v in per_view]) for key in ('depth','depth_valid','small_vehicle','small_valid')}
            dst=out/'labels'/split/(row['token']+'.npz');dst.parent.mkdir(parents=True,exist_ok=True)
            if dst.exists():
                with np.load(dst,allow_pickle=False) as old:
                    if any(not np.array_equal(old[k],v) for k,v in tensors.items()):raise ValueError('Conflicting complete probe label')
            else:
                temporary=dst.with_suffix('.tmp')
                with temporary.open('wb') as f:np.savez(f,**tensors)
                temporary.replace(dst)
            records.append({'token':row['token'],'log':row['log'],'raw_log_sha256':log_hashes[row['log']],
                'lidar_path':str(path),'lidar_sha256':file_sha256(path),'label_sha256':file_sha256(dst),
                'current_timestamp_us':frame['timestamp'],'camera_calibration_hash':identity_hash({k:{name:np.asarray(v).tolist() for name,v in frame['cams'][k].items() if name!='data_path'} for k in registration['cameras']}),
                'views':[v['statistics'] for v in per_view], 'depth_valid_patches':int(tensors['depth_valid'].sum()),
                'small_positive_patches':int(tensors['small_vehicle'].sum()),
                'small_negative_patches':int((tensors['small_valid'] & (tensors['small_vehicle']==0)).sum())})
        result[split]=records
    atomic_json(out/'LABEL_MANIFEST.json',{'registration':registration['identity'],'roles':result,
                'label_schema':'current_raw_lidar_distorted_full_rgb_native384_patch_v1'},immutable=True)
    return result


def binary_metrics(logits,truth):
    from sklearn.metrics import roc_auc_score,average_precision_score
    p=logits.numpy();y=truth.numpy().astype(bool);pred=p>0
    return {'ROC_AUC':float(roc_auc_score(y,p)),'PR_AUC':float(average_precision_score(y,p)),
        'balanced_accuracy_at_logit0':float(((pred[y]).mean()+(~pred[~y]).mean())*.5),
        'positive_patches':int(y.sum()),'negative_patches':int((~y).sum())}


def run(a):
    started=time.monotonic();out=Path(a.output);artifacts=Path(a.artifact_root)
    registration=json.loads(Path(a.registration).read_text())
    validate_registration(registration)
    def check_budget():
        if time.monotonic()-started>registration['max_wall_seconds']:raise RuntimeError('Probe wall budget exhausted')
    cfg=registration['readout']
    torch.set_num_threads(registration['cpu_threads']);torch.manual_seed(registration['readout']['seed'])
    atomic_json(out/'REGISTRATION.json',registration,immutable=True)
    atomic_json(out/'SOURCE.json',{'source_sha256':file_sha256(__file__),'command':vars(a),
                'readout_initialization':'zero weight and bias, all dimensions and both probes',
                'readout_selection':'fixed final update; no best/dev selection'},immutable=True)
    if (out/'COMPLETE.json').exists():
        receipt=json.loads((out/'COMPLETE.json').read_text())
        if receipt['registration']!=registration['identity'] or receipt['probe_results_sha256']!=file_sha256(out/'PROBE_RESULTS.json'):
            raise ValueError('Completed probe artifact changed')
        return {'reused':True,'receipt':receipt}
    readers={}
    for split in ['train','dev']:
        root=artifacts/registration['teacher_cache_'+split];ident=json.loads((root/'identity.json').read_text())['identity']
        readers[split]=TeacherCacheReader(root,expected_identity=ident)
    labels=prepare_labels(a,registration,readers,check_budget)
    basis=TeacherPCA.load(a.pca_root,dimension=384,expected_identity=registration['pca_identity'],expected_teacher_contract=readers['train'].contract.identity)
    sets={}
    for split in ['train','dev']:
        feature=[];target=[];valid=[];small=[];small_valid=[]
        for row in labels[split]:
            check_budget()
            token=row['token'];sample=readers[split].read_scene(token,expected_split=split,need_future=False)
            z=basis.transform(sample['current_dense_features']).reshape(-1,384)
            with np.load(out/'labels'/split/(token+'.npz'),allow_pickle=False) as y:
                feature.append(z);target.append(torch.from_numpy(y['depth'].copy()).flatten());valid.append(torch.from_numpy(y['depth_valid'].copy()).flatten())
                small.append(torch.from_numpy(y['small_vehicle'].copy()).flatten());small_valid.append(torch.from_numpy(y['small_valid'].copy()).flatten())
        sets[split]={'features':torch.cat(feature),'depth':torch.cat(target),'depth_valid':torch.cat(valid),
                     'small':torch.cat(small),'small_valid':torch.cat(small_valid)}
    result={'registration':registration['identity'],'pca_identity':basis.identity,'dimension_selected':None,'results':{},
            'label_source':'current raw LiDAR and current raw vehicle boxes; no teacher pseudo-labels',
            'interpretation':'bounded readout diagnostic, not PDMS or semantic segmentation', 'fit_role':'train'}
    for task in ['depth','small']:
        train=sets['train'];dev=sets['dev'];mask=task+'_valid'
        tx=train['features'][train[mask]];ty=train[task][train[mask]]
        vx=dev['features'][dev[mask]];vy=dev[task][dev[mask]]
        if len(tx)<cfg['batch_patches'] or len(vx)<1 or task=='small' and (ty.sum()<5 or vy.sum()<1 or ty.sum()==len(ty) or vy.sum()==len(vy)):
            result['results'][task]={'status':'INSUFFICIENT_VALID_LABELS','train_valid':len(tx),'dev_valid':len(vx)};continue
        if task=='depth':
            mean=ty.log().mean();std=ty.log().std(unbiased=False).clamp_min(1e-6);train_y=(ty.log()-mean)/std
        else:train_y=ty
        order_rng=torch.Generator().manual_seed(registration['readout']['seed'])
        batch_ids=torch.randint(len(tx),(cfg['updates'],cfg['batch_patches']),generator=order_rng)
        result['results'][task]={}
        for dimension in registration['dimensions']:
            # Independent readout, same model family and fixed exposure stream.
            model=nn.Linear(dimension,1);nn.init.zeros_(model.weight);nn.init.zeros_(model.bias)
            optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
            losses=[]
            for ids in batch_ids:
                pred=model(tx[ids,:dimension]).squeeze(-1)
                if not torch.isfinite(pred).all():raise ValueError('Nonfinite probe prediction')
                if task=='depth':loss=torch.nn.functional.smooth_l1_loss(pred,train_y[ids],beta=.1)
                else:loss=torch.nn.functional.binary_cross_entropy_with_logits(pred,train_y[ids],pos_weight=(len(ty)-ty.sum())/ty.sum())
                optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['clip_grad_norm'],error_if_nonfinite=True);optimizer.step();losses.append(float(loss))
                check_budget()
            metrics={}
            with torch.no_grad():
                for role,x,y in [('train',tx,ty),('dev',vx,vy)]:
                    pred=model(x[:,:dimension]).squeeze(-1)
                    if not torch.isfinite(pred).all():raise ValueError('Nonfinite probe validation prediction')
                    if task=='depth':
                        meters=(pred*std+mean).exp();error=meters-y
                        if not torch.isfinite(meters).all():raise ValueError('Nonfinite reconstructed probe depth')
                        metrics[role]={'MAE_m':float(error.abs().mean()),'RMSE_m':float(error.square().mean().sqrt()),
                            'AbsRel':float((error.abs()/y).mean()),'valid_patches':len(y)}
                    else:metrics[role]=binary_metrics(pred,y)
            torch.save({'state_dict':model.state_dict(),'task':task,'dimension':dimension,'registration':registration['identity'],
                        'pca_identity':basis.identity,'updates':cfg['updates'],'optimizer':optimizer.state_dict()},out/f'{task}_{dimension}.pt')
            result['results'][task][str(dimension)]={'metrics':metrics,'train_losses':losses,'parameters':sum(p.numel() for p in model.parameters()),
                'updates':cfg['updates'],'patch_presentations':int(batch_ids.numel()),'sample_order_sha256':identity_hash(batch_ids.tolist()),
                'normalization':{'log_depth_mean':float(mean),'log_depth_std':float(std)} if task=='depth' else {'pos_weight':float((len(ty)-ty.sum())/ty.sum())}}
    result['elapsed_s']=time.monotonic()-started
    atomic_json(out/'PROBE_RESULTS.json',result,immutable=True)
    atomic_json(out/'COMPLETE.json',{'status':'COMPLETE','registration':registration['identity'],
        'probe_results_sha256':file_sha256(out/'PROBE_RESULTS.json'),'dimension_selected':None,'gpu_used':False},immutable=True)
    return {'elapsed_s':result['elapsed_s'],'dimension_selected':None,'results':{
        task:({dimension:cell['metrics'] for dimension,cell in cells.items()} if 'status' not in cells else cells)
        for task,cells in result['results'].items()}}


def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ['registration','artifact-root','pca-root','raw-log-root','sensor-root','output']:p.add_argument('--'+key,required=True)
    p.add_argument('--fallback-sensor-root',action='append',default=[])
    a=p.parse_args();print(json.dumps(run(a),sort_keys=True))


if __name__=='__main__':main()
