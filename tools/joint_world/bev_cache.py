"""Current-only BEV index and bounded CPU store; observations, never targets, validate it."""
import argparse
from collections import OrderedDict
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch


WEIGHT_SHA='a7ea19fa0ed99244e67b624c72b8580b7e9553043245905be58796a608eb9345'
SENSOR={'cameras':['CAM_F0','CAM_L0','CAM_R0'],'time':'current_only'}
GRID={'bounds':[1.,-20.,50.,20.],'resolution':1.,'shape':[49,40],'heights':[0.,1.,2.]}
META_FIELDS={'schema_version','provider','sensor_contract','coordinates','extrinsics','grid',
 'feature_dimension','support_semantics','pretrained_bev','backbone_weights_sha256',
 'backbone_source_sha256','backbone_internal_resize','backbone_normalization','learned_bev_fusion',
 'scene_token','decision_time','image_sha256','image_transforms','calibration_sha256','dtype','provider_code_sha'}


def validate_payload(payload,token):
    if set(payload)!={'metadata','features','coordinates','observation_support'}:raise ValueError('Unexpected BEV cache fields')
    meta=payload['metadata']
    if set(meta)!=META_FIELDS:raise ValueError('BEV metadata whitelist mismatch')
    for key,value in {'schema_version':1,'scene_token':token,'sensor_contract':SENSOR,'grid':GRID,'backbone_weights_sha256':WEIGHT_SHA,
                      'pretrained_bev':False,'coordinates':'ego_t0_x_forward_y_left_z_up_metres',
                      'extrinsics':'camera_to_ego','feature_dimension':1024}.items():
        # Torch metadata preserves tuples while JSON indices use lists; geometry
        # values/keys remain exact after canonical serialization of both forms.
        if json.dumps(meta[key],sort_keys=True)!=json.dumps(value,sort_keys=True):raise ValueError('BEV identity mismatch: '+key)
    for key,shape in [('features',(1,1960,1024)),('coordinates',(1,1960,3)),('observation_support',(1,1960))]:
        if tuple(payload[key].shape)!=shape or not torch.isfinite(payload[key]).all():raise ValueError('Invalid BEV '+key)
    if payload['observation_support'].dtype!=torch.bool:raise ValueError('BEV support must be boolean')
    if str(payload['features'].dtype)!=meta['dtype']:raise ValueError('BEV dtype mismatch')


def validate_observation(meta,path,sensor_root):
    with np.load(path,allow_pickle=False) as z:obs={k:z[k] for k in z.files}
    fields={'intrinsics','extrinsics','image_transforms','distortion','timestamp','camera_names','image_paths','schema_version'}
    if set(obs)!=fields or int(obs['schema_version'])!=1:raise ValueError('Unexpected current observation fields')
    if list(obs['camera_names'])!=SENSOR['cameras'] or int(obs['timestamp'])!=meta['decision_time']:
        raise ValueError('Current observation camera/time mismatch')
    digest=hashlib.sha256()
    for dst,src in [('camera_intrinsics','intrinsics'),('camera_extrinsics','extrinsics'),('image_transforms','image_transforms'),('distortion','distortion')]:
        digest.update(dst.encode());digest.update(np.asarray(obs[src],dtype=np.float32)[None].tobytes())
    if digest.hexdigest()!=meta['calibration_sha256']:raise ValueError('Changed current calibration')
    images=[hashlib.sha256((Path(sensor_root)/str(p)).read_bytes()).hexdigest() for p in obs['image_paths']]
    if images!=meta['image_sha256']:raise ValueError('BEV features do not match current images')
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class BEVFeatureStore:
    def __init__(self,index,resident=32):
        self.index=json.loads(Path(index).read_text());self.records={r['token']:r for r in self.index['records']}
        identity=self.index['identity']
        if identity['sensor_contract']!=SENSOR or identity['weights_sha256']!=WEIGHT_SHA or identity['grid']!=GRID:
            raise ValueError('Unsupported BEV index identity')
        if not identity['current_observations_verified'] or identity['targets_opened']:raise ValueError('Unverified BEV index')
        if len(self.records)!=len(self.index['records']):raise ValueError('Duplicate BEV token')
        if not 1<=resident<=64:raise ValueError('BEV CPU residency cap64')
        self.resident=resident;self.memo=OrderedDict()
        self.identity_sha256=hashlib.sha256(Path(index).read_bytes()).hexdigest()

    def __getitem__(self,token):
        if token not in self.memo:
            row=self.records[token];blob=Path(row['path']).read_bytes()
            if hashlib.sha256(blob).hexdigest()!=row['sha256']:raise ValueError('BEV file changed after verification')
            payload=torch.load(io.BytesIO(blob),map_location='cpu',weights_only=True);validate_payload(payload,token)
            self.memo[token]=payload
            if len(self.memo)>self.resident:self.memo.popitem(last=False)
        self.memo.move_to_end(token);return self.memo[token]


def batch_bev(samples,device='cuda'):
    return {k:torch.cat([s['bev'][k] for s in samples]).to(device) for k in ['features','coordinates','observation_support']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['manifest','observations','sensor-root','output']:p.add_argument('--'+key,required=True)
    p.add_argument('--shards',nargs='+',required=True);p.add_argument('--allow-subset',action='store_true')
    a=p.parse_args();out=Path(a.output)
    if out.exists():raise FileExistsError('Use an immutable new index')
    tokens=json.loads(Path(a.manifest).read_text());paths={};sources=[]
    for shard in a.shards:
        root=Path(shard);audit=json.loads((root/'extraction.json').read_text())
        if audit['failed']:raise ValueError('Feature extraction is incomplete')
        sources.append(json.loads((root/'manifest.json').read_text()))
        for row in audit['records']:
            if row['token'] in paths:raise ValueError('Duplicated BEV token')
            paths[row['token']]=root/(row['token']+'.pt')
    if not set(tokens)<=set(paths) or len(tokens)!=len(set(tokens)):raise ValueError('Missing/duplicate requested BEV scenes')
    if not a.allow_subset and set(tokens)!=set(paths):raise ValueError('Extra scenes; explicit --allow-subset required')
    rows=[];provider_sources=set()
    for token in tokens:
        blob=paths[token].read_bytes();payload=torch.load(io.BytesIO(blob),map_location='cpu',weights_only=True)
        validate_payload(payload,token)
        observation_sha=validate_observation(payload['metadata'],Path(a.observations)/(token+'.npz'),a.sensor_root)
        provider_sources.add(payload['metadata']['backbone_source_sha256'])
        rows.append({'token':token,'path':str(paths[token].resolve()),'sha256':hashlib.sha256(blob).hexdigest(),'observation_sha256':observation_sha})
        if len(rows)%512==0:print(json.dumps({'verified':len(rows),'requested':len(tokens)}),flush=True)
    if len(provider_sources)!=1:raise ValueError('Mixed provider implementation')
    identity={'weights_sha256':WEIGHT_SHA,'sensor_contract':SENSOR,'grid':GRID,'pretrained_bev':False,
              'current_observations_verified':True,'targets_opened':False,'provider_source_sha256':next(iter(provider_sources)),
              'requested_scenes':len(tokens),'source_scenes':len(paths),'explicit_subset':a.allow_subset,
              'manifest_sha256':hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest(),'source_manifests':sources}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps({'identity':identity,'records':rows},indent=2))
    print(json.dumps({'verified':len(rows),'output':str(out),'targets_opened':False}))


if __name__=='__main__':main()
