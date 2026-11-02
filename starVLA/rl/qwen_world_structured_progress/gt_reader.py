"""Sealed TRAIN GT/current-ego source, with bounded immutable CPU caches.

Raw source coordinates follow official Scene exactly; never import another
NAVSIM module into the GPU interpreter. No raw frames past the eight labels.
"""
from collections import OrderedDict
from pathlib import Path
import pickle
import numpy as np
from pyquaternion import Quaternion
import torch
from starVLA.rl.qwen_world_postcot.contracts import file_sha,fingerprint
from starVLA.rl.qwen_world_safety_exploration.retention_material import sealed,verify_seal,digest

SCHEMA='world_structured_progress_verified_TRAIN_GT_source_v1'
RAW=Path('/mnt/project/DriveDreamer-Policy/navsim_raw/navsim_logs/trainval')


class GTReader:
    def __init__(self,data,*,raw_root=RAW,token_capacity=256,log_capacity=2):
        self.data=data;self.raw_root=Path(raw_root);self.token_capacity=token_capacity;self.log_capacity=log_capacity
        self.tokens=OrderedDict();self.logs=OrderedDict();self.raw_hashes={};sources={}
        for source in data.manifest['source_records']:
            path=Path(source['root'])/'manifest.json'
            if file_sha(path)!=source['manifest_sha256']:raise ValueError('Original IL GT manifest changed')
            import json
            m=json.loads(path.read_text());self.raw_hashes.update(m['raw_logs_sha256']);sources[str(path)]=source['manifest_sha256']
        self.identity=dict(schema=SCHEMA,data_manifest_sha256=data.sha,records_sha256=data.manifest['records_sha256'],
            population_sha256=data.manifest['population_sha256'],original_manifest_sha256=sources,
            coordinate='official rear-axle local x-forward/y-left/relative-radian,8 absolute future poses',
            timing='all8 FP32 cumulative raw_dt within.06s of0.5..4s; never interpolate missing time',
            caches='immutable source-only tokenLRU256/rawlogLRU2; no policy graph, optimizer or RNG; recomputable on resume')
        self.sha=fingerprint(self.identity)

    def current_ego(self,token,log):
        if token not in self.data.records or self.data.logs[token]!=log:raise ValueError('Non-TRAIN token/log')
        ego=torch.tensor(self.data.records[token]['proprio'][:4],dtype=torch.float32)
        if ego.shape!=(4,) or not torch.isfinite(ego).all():raise ValueError('Invalid current ego')
        return ego

    def record_fingerprint(self,token,log):
        r=self.data.records[token]
        return digest(dict(token=token,log=log,GT=torch.tensor(r['requested_GT'],dtype=torch.float32),
            dt=r['raw_future_dt_seconds'],timestamp=r['timestamp'],ego=self.current_ego(token,log)))

    def validate_saved(self,source,token,log):
        """Validate immutable input proof, never refresh a pending family gate.

        An unavailable raw source at collection time must not turn into a new
        GT family on restore, and an already-verified source needs no new I/O.
        """
        verify_seal(source,'source_sha256')
        if source.get('schema')!=SCHEMA or source.get('token')!=token or source.get('log')!=log or source.get('reader_identity_sha256')!=self.sha:
            raise ValueError('Saved GT source identity changed')
        if source['record_fingerprint'] is not None and source['record_fingerprint']!=self.record_fingerprint(token,log):raise ValueError('Saved immutable GT/current-ego record changed')
        if source['valid']:
            want=torch.tensor(self.data.records[token]['requested_GT'],dtype=torch.float32)
            if source['requested'].requires_grad or source['requested'].dtype!=torch.float32 or not torch.equal(want,source['requested']) or source['raw_log_sha256']!=self.raw_hashes[log] or not bool(source['time_valid'].all()):
                raise ValueError('Saved verified GT target/time/raw provenance changed')
        elif source['requested'] is not None:raise ValueError('Invalid GT source retained a teacher target')
        return source

    def read(self,token,log):
        key=(token,log)
        if key in self.tokens:
            value=self.tokens.pop(key);self.tokens[key]=value;return value
        base=dict(schema=SCHEMA,token=token,log=log,reader_identity_sha256=self.sha,valid=False,reason=None,
            requested=None,time_valid=None,raw_log_sha256=None,record_fingerprint=None)
        try:
            self.current_ego(token,log);r=self.data.records[token]
            gt=torch.tensor(r['requested_GT'],dtype=torch.float32)
            dt=torch.tensor(r['raw_future_dt_seconds'],dtype=torch.float32)
            if gt.shape!=(8,3) or not torch.isfinite(gt).all() or dt.shape!=(8,) or not torch.isfinite(dt).all() or not bool((dt>0).all()):
                raise ValueError('invalid_GT_shape_or_time')
            mask=(dt.cumsum(0)-torch.arange(1,9)*.5).abs()<.06
            base['time_valid']=mask;base['record_fingerprint']=self.record_fingerprint(token,log)
            if not bool(mask.all()):raise ValueError('invalid_fixed_time_GT_only')
            if log in self.logs:
                frames,raw_sha=self.logs.pop(log);self.logs[log]=(frames,raw_sha)
            else:
                path=self.raw_root/(log+'.pkl');raw_sha=file_sha(path)
                if raw_sha!=self.raw_hashes[log]:raise ValueError('raw_TRAIN_log_hash_changed')
                with path.open('rb') as f:frames=pickle.load(f)
                self.logs[log]=(frames,raw_sha)
                while len(self.logs)>self.log_capacity:self.logs.popitem(last=False)
            index=next(i for i,f in enumerate(frames) if f['token']==token);future=frames[index:index+9]
            if len(future)!=9 or int(future[0]['timestamp'])!=r['timestamp']:raise ValueError('raw_token_or_horizon_changed')
            actual_dt=np.diff([float(f['timestamp']) for f in future])/1e6
            if not np.array_equal(actual_dt,np.asarray(r['raw_future_dt_seconds'])):raise ValueError('raw_timestamp_changed')
            poses=np.array([[f['ego2global_translation'][0],f['ego2global_translation'][1],
                Quaternion(*f['ego2global_rotation']).yaw_pitch_roll[0]] for f in future],dtype=np.float64)
            theta=-poses[0,2];rotation=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
            local=poses[1:]-poses[0:1];local[:,:2]=local[:,:2]@rotation.T
            local[:,2]=np.arctan2(np.sin(local[:,2]),np.cos(local[:,2]))
            if not np.array_equal(local.astype(np.float32),gt.numpy()):raise ValueError('official_raw_coordinate_reconstruction_changed')
            base.update(valid=True,requested=gt,raw_log_sha256=raw_sha,
                record_fingerprint=self.record_fingerprint(token,log))
        except (OSError,ValueError,KeyError,IndexError,StopIteration,pickle.PickleError,ZeroDivisionError) as exc:
            base['reason']=type(exc).__name__+':'+str(exc)
        value=sealed(base,'source_sha256');self.tokens[key]=value
        while len(self.tokens)>self.token_capacity:self.tokens.popitem(last=False)
        return value
