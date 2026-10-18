"""CPU road-direction diagnostic: original fixed32 or explicitly bound failures.

Prepare emits a reviewable immutable registration. Execute requires an exact
Lead admission plus independently authored correctness/resource approvals bound
to that registration and these source hashes. Existing calls are never repeated.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import fcntl
import hashlib
import io
import json
import lzma
import math
import os
from pathlib import Path
import pickle
import resource
import signal
import socket
import stat
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from tools.st_world_policy.diagnose_road_direction import EVIDENCE, prepare as prepare_evidence
from tools.st_world_policy.consequence_smoke import atomic_json, json_ready
from starVLA.model.modules.st_world.consequence import (
    ConsequenceEngine, ConsequenceRequest, ConsequenceSpec, array_sha256,
    fingerprint, validate_cache, validated_reference_input, METRICS)

SCHEMA = 'road_direction_execution_v1'
FILES = ('tools/st_world_policy/run_road_direction.py',
         'tools/st_world_policy/road_failure_selection.py',
         'tools/st_world_policy/diagnose_road_direction.py',
         'tools/st_world_policy/consequence_smoke.py',
         'starVLA/model/modules/flow_posttraining/road_margin_proxy.py',
         'starVLA/model/modules/flow_posttraining/rollout_contract.py',
         'starVLA/model/modules/st_world/consequence.py',
         'starVLA/dataloader/foresight_dataset.py')
LIMITS = {'wall_seconds':1800,'rss_and_address_space_bytes':8*2**30,
          'output_bytes':2**30,'asset_read_bytes':256*2**20,
          'read_rate_bytes_per_second':32*2**20,'cache_expanded_bytes':64*2**20,
          'max_official_calls':128,'workers':1,'threads':1,'scenes':32,'variants':4}
FAILURE_LIMITS = {**LIMITS, 'max_official_calls':96}
BINDING_LIMITS = {**LIMITS, 'wall_seconds':600, 'rss_and_address_space_bytes':2*2**30,
                  'output_bytes':16*2**20, 'asset_read_bytes':512*2**20,
                  'cpu_seconds':120, 'max_official_calls':0, 'workers':1, 'threads':1}
DECODE_GATE = {'raw_dtype':'float32_exact_widening','xy':'constant_rounding_plus_half_ULP_product_and_sum_plus_float64_roundoff',
              'yaw_periodic_tolerance_rad':8*2**-23,'yaw_tolerance_is_not_gpu_error_theorem':True}
VARIANTS = ('original','negative','positive','random')


def sha(data): return hashlib.sha256(data).hexdigest()
def source_commit(): return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
def source_hashes(): return {p:sha((ROOT/p).read_bytes()) for p in FILES}
def small_json(path):
    path=Path(path)
    with path.open('rb') as stream: raw=stream.read(2*2**20+1)
    if len(raw)>2*2**20: raise ValueError('Oversized control JSON')
    return json.loads(raw)


def owned_output_bytes(root):
    """One stat per entry; committed temp unlink races are not monitor failures."""
    total=0
    for path in Path(root).rglob('*'):
        try: value=path.lstat()
        except FileNotFoundError: continue
        if stat.S_ISREG(value.st_mode): total+=value.st_size
    return total


class LimitError(RuntimeError): pass


class GeometryParityError(ValueError):
    def __init__(self, audit):
        super().__init__('REAL_OFFICIAL_GEOMETRY_PARITY_FAILURE')
        self.audit = audit


def require_geometry_parity(audit):
    if max(audit[k] for k in ('command_xy_error_m','command_yaw_error_rad','corner_max_error_m')) > 1e-9:
        raise GeometryParityError(audit)


class Budget:
    def __init__(self, root, limits=LIMITS):
        self.root=Path(root);self.limits=limits;self.started=time.monotonic()
        cpu=resource.getrusage(resource.RUSAGE_SELF);self.cpu_started=cpu.ru_utime+cpu.ru_stime
        self.read_bytes=0;self.reads=[];self.verified_stats={}
    def check(self):
        if time.monotonic()-self.started>self.limits['wall_seconds']:raise LimitError('WALL_CAP')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>self.limits['rss_and_address_space_bytes']:
            raise LimitError('RSS_CAP')
        cpu=resource.getrusage(resource.RUSAGE_SELF)
        if cpu.ru_utime+cpu.ru_stime-self.cpu_started>self.limits.get('cpu_seconds',math.inf):raise LimitError('CPU_CAP')
    def read(self,path,digest,maximum=None):
        self.check();p=Path(path);before=p.stat();size=before.st_size
        if size>(maximum or self.limits['asset_read_bytes']) or self.read_bytes+size>self.limits['asset_read_bytes']:
            raise LimitError('ASSET_READ_CAP: '+str(p))
        # Reserve the bounded probe before IO. A failed read retains its upper
        # bound charge; a returned read is charged by actual returned bytes.
        read_limit=min(size+1,self.limits['asset_read_bytes']-self.read_bytes)
        self.read_bytes+=read_limit
        with p.open('rb') as stream: raw=stream.read(read_limit)
        self.read_bytes-=read_limit-len(raw)
        after=p.stat()
        stamp=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
        actual=sha(raw)
        self.reads.append({'path':str(p),'bytes':len(raw),'sha256':actual})
        if len(raw)!=size or stamp(before)!=stamp(after) or actual!=digest:raise ValueError('ASSET_CHANGED: '+str(p))
        target=self.read_bytes/self.limits['read_rate_bytes_per_second']
        delay=target-(time.monotonic()-self.started)
        if delay>0:time.sleep(delay)
        self.verified_stats[str(p.resolve())]=stamp(after)
        self.check();return raw
    def write(self,path,value):
        self.check();p=Path(path)
        if not p.resolve().is_relative_to(self.root.resolve()):raise ValueError('OUTPUT_ESCAPE')
        payload=(json.dumps(json_ready(value),indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
        existing=owned_output_bytes(self.root)
        if existing+2*len(payload)>self.limits['output_bytes']-2*2**20:raise LimitError('OUTPUT_CAP')
        atomic_json(p,value)
    def report(self):
        return {'elapsed_seconds':time.monotonic()-self.started,'logical_asset_bytes':self.read_bytes,
                'asset_reads':self.reads,'max_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                'cpu_seconds':resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime-self.cpu_started}


def bounded_unpack(compressed,maximum):
    decoder=lzma.LZMADecompressor()
    expanded=decoder.decompress(compressed,max_length=maximum+1)
    if len(expanded)>maximum or not decoder.eof or decoder.unused_data:
        raise LimitError('CACHE_EXPANSION_CAP_OR_TRAILING_DATA')
    return pickle.loads(expanded)


def decode_discrepancy_gate(raw,physical):
    """Predeclared FP32 eager XY roundoff envelope; yaw tolerance is empirical."""
    import numpy as np
    import torch
    from starVLA.dataloader.foresight_dataset import decode_ego,X_MEAN,X_STD,Y_MEAN,Y_STD
    if raw.dtype!=torch.float32 or raw.shape!=(8,4) or not torch.isfinite(raw).all():
        raise ValueError('Original saved raw must be finite float32 [8,4]')
    p=np.asarray(physical,dtype=np.float64)
    if p.shape!=(8,3) or not np.isfinite(p).all():raise ValueError('Invalid saved physical command')
    x=raw.detach().cpu().numpy()[:,:2].astype(np.float64)
    scale=np.array([X_STD,Y_STD]);mean=np.array([X_MEAN,Y_MEAN])
    scale32=scale.astype(np.float32);mean32=mean.astype(np.float32)
    product=(x.astype(np.float32)*scale32).astype(np.float32)
    summed=(product+mean32).astype(np.float32)
    def spacing(v):
        hi=np.nextafter(v,np.float32(np.inf)).astype(np.float64)-v.astype(np.float64)
        lo=v.astype(np.float64)-np.nextafter(v,np.float32(-np.inf)).astype(np.float64)
        return np.maximum(hi,lo)
    bound=(abs(x)*abs(scale32.astype(np.float64)-scale)+abs(mean32.astype(np.float64)-mean)
           +.5*spacing(product)+.5*spacing(summed)
           +8*np.finfo(np.float64).eps*(abs(x*scale)+abs(mean)+1))
    unanchored=decode_ego(raw.double()).detach().numpy()
    offset=p-unanchored;offset[:,2]=np.arctan2(np.sin(offset[:,2]),np.cos(offset[:,2]))
    if not np.isfinite(bound).all() or np.any(abs(offset[:,:2])>bound):raise ValueError('XY_DECODE_ROUNDOFF_ENVELOPE_EXCEEDED')
    if np.any(abs(offset[:,2])>DECODE_GATE['yaw_periodic_tolerance_rad']):raise ValueError('YAW_DECODE_TOLERANCE_EXCEEDED')
    return {'gate':DECODE_GATE,'xy_bounds_m':bound.tolist(),'observed_offsets':offset.tolist(),
            'cpu_float32_decode':decode_ego(raw).detach().numpy().astype(np.float64).tolist(),
            'cpu_float64_decode':unanchored.tolist(),'original_raw_sha256':sha(raw.numpy().tobytes())}


class VerifiedContextEngine(ConsequenceEngine):
    """Use the native score function, allowing only a bounded preloaded context."""
    def bind(self,request,cache,verified_stamp):
        p=Path(request.metric_cache_path).resolve()
        if not any(p.is_relative_to(Path(r).resolve()) for r in self.spec.cache_roots):raise ValueError('Cache root mismatch')
        if list(p.parts[-4:])!=[request.source_log,'unknown',request.scene_id,'metric_cache.pkl']:raise ValueError('Cache path mismatch')
        validate_cache(cache)
        st=p.stat();self._cache_stat=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
        if self._cache_stat!=tuple(verified_stamp):raise ValueError('Verified cache changed before binding')
        self._cache_key=(str(p),request.metric_cache_sha256);self._cache=cache
    def load_context(self,request):
        if self._cache_key!=(str(Path(request.metric_cache_path).resolve()),request.metric_cache_sha256):
            raise ValueError('Only explicitly verified bounded context allowed')
        return super().load_context(request)


def registration(evidence_path,output_root,selection_mode='fixed32_v1'):
    if selection_mode == 'failure_cohort_v1':
        from tools.st_world_policy.road_failure_selection import prepare, EXECUTION_SCHEMA
        preparation=prepare(evidence_path); schema=EXECUTION_SCHEMA;limits=FAILURE_LIMITS
    elif selection_mode == 'fixed32_v1':
        preparation=prepare_evidence(evidence_path);schema=SCHEMA;limits=LIMITS
    else:raise ValueError('Unsupported explicit selection mode')
    result={'schema':schema,'status':'REGISTERED_NOT_EXECUTED','evidence':preparation['evidence'],
            'selection_hash':preparation['selection_hash'],'source_hashes':source_hashes(),'source_commit':source_commit(),
            'limits':limits,'decode_gate':DECODE_GATE,'output_root':str(Path(output_root).resolve()),
            'fixed_original_DAC':preparation['existing_DAC'],'dispatch_authorized':False}
    if selection_mode == 'failure_cohort_v1':
        result.update(selection_mode=selection_mode,requested_scenes=preparation['scene_count'])
    return result


def execution_limits(reg):
    from tools.st_world_policy.road_failure_selection import EXECUTION_SCHEMA
    if reg['schema'] == SCHEMA and reg.get('selection_mode','fixed32_v1')=='fixed32_v1':return LIMITS
    if reg['schema'] == EXECUTION_SCHEMA and reg.get('selection_mode') == 'failure_cohort_v1':return FAILURE_LIMITS
    raise ValueError('Unknown execution schema or implicit failure selection')


def check_admission(reg_path,admission_path,admission_hash):
    reg=small_json(reg_path);admission_data=Path(admission_path).read_bytes()
    if sha(admission_data)!=admission_hash:raise ValueError('Lead admission checksum mismatch')
    admission=json.loads(admission_data)
    if reg.get('source_commit')!=source_commit():raise ValueError('Execution worktree commit differs from registration')
    limits=execution_limits(reg)
    if reg['limits']!=limits or reg['decode_gate']!=DECODE_GATE or reg['source_hashes']!=source_hashes():
        raise ValueError('Registration code or constants changed')
    if reg.get('selection_mode') == 'failure_cohort_v1':
        from tools.st_world_policy.road_failure_selection import prepare
        preparation=prepare(reg['evidence']['path'])
        if (preparation['evidence']!=reg['evidence'] or preparation['selection_hash']!=reg['selection_hash']
                or preparation['scene_count']!=reg['requested_scenes']):raise ValueError('Failure selection changed')
    reg_hash=sha(Path(reg_path).read_bytes())
    if admission.get('schema')!='road_direction_lead_admission_v1' or admission.get('status')!='APPROVED':raise ValueError('No Lead execution admission')
    if admission.get('registration_sha256')!=reg_hash or admission.get('source_hashes')!=reg['source_hashes']:raise ValueError('Admission binding mismatch')
    if admission.get('host')!=socket.gethostname() or not admission['not_before_unix']<=time.time()<admission['expires_unix']:
        raise ValueError('Admission host/window mismatch')
    if admission['expires_unix']-time.time()<limits['wall_seconds']:raise ValueError('Insufficient admitted wall window')
    policy_path=Path(admission['dispatch_policy']['path']);policy_raw=policy_path.read_bytes()
    if sha(policy_raw)!=admission['dispatch_policy']['sha256']:raise ValueError('Dispatch policy changed')
    policy=json.loads(policy_raw)
    if socket.gethostname() not in policy['allowed_aliases']:raise ValueError('Host no longer allowed')
    reviewers=[]
    for kind in ('correctness','resources'):
        ref=admission['reviews'][kind];data=Path(ref['path']).read_bytes()
        if sha(data)!=ref['sha256']:raise ValueError('Review checksum mismatch')
        review=json.loads(data)
        if (review.get('status')!='APPROVE_ROAD_DIRECTION_EXECUTION' or review.get('review_kind')!=kind
            or review.get('registration_sha256')!=reg_hash or review.get('source_hashes')!=reg['source_hashes']):
            raise ValueError('Independent review does not approve this exact execution')
        reviewers.append(review['reviewer'])
    if len(set(reviewers))!=2 or admission['issuer'] in reviewers:raise ValueError('Independent reviewer identities required')
    return reg,admission


def load_inputs(evidence,budget):
    if evidence.get('schema') == 'road_failure_direction_selection_v1':
        from tools.st_world_policy.road_failure_selection import load_inputs as load_failure_inputs
        return load_failure_inputs(evidence,budget)
    import torch
    entries={r['path']:r for r in evidence['budget']['entries']}
    def read(path):
        return budget.read(path,entries[str(path)]['sha256'])
    rollouts={}
    for ref in evidence['rollout_receipts']:
        receipt=json.loads(read(ref['path']));payload=receipt['payload']
        if payload['sha256']!=ref['payload_sha256']:raise ValueError('Rollout receipt mismatch')
        value=torch.load(io.BytesIO(budget.read(payload['path'],payload['sha256'])),map_location='cpu',weights_only=False)
        if value['identity']!=receipt['identity']:raise ValueError('Rollout identity mismatch')
        rollout=value['rollout']
        if rollout['policy']!=evidence['policy'] or rollout['flow_steps']!=10 or rollout['raw_representation']!='normalized_xy_sin_cos':
            raise ValueError('Wrong policy or actual rollout representation')
        if tuple(rollout['raw_actions'].shape)!=(8,4,8,4):raise ValueError('Wrong rollout population')
        rank=int(Path(ref['path']).name.split('_')[1]);rollouts[rank]=rollout
    result=evidence['stored_official_result'];result_ref=result['result'];folder=Path(result_ref['path']).parent
    complete=json.loads(read(folder/'COMPLETE.json'))
    if complete!=result or sha(read(folder/'ADMITTED.json'))!=complete['admission_sha256']:raise ValueError('Original feedback receipt mismatch')
    records=torch.load(io.BytesIO(budget.read(result_ref['path'],result_ref['sha256'])),map_location='cpu',weights_only=False)
    requests=json.loads(read(folder.parent/'REQUESTS.json'))
    if requests['policy_version']!='4095':raise ValueError('Wrong saved request policy')
    request_map={r['candidate_id']:r for r in requests['requests']}
    record_map={r['identity']['candidate_id']:r for r in records}
    # Bounded independent train membership; never read the176MiB enriched manifest.
    index_entry=next(e for e in entries.values() if e['path'].endswith('/student_train_v1/index.json'))
    index=json.loads(read(index_entry['path']))
    if fingerprint(index)!=evidence['train_index_identity']['index_sha256']:raise ValueError('Train index mismatch')
    train={r['token']:r['log'] for r in index};loaded=[]
    for item in evidence['selected']:
        rollout=rollouts[item['rank']];slot=item['slot'];raw=rollout['raw_actions'][slot,0].detach().cpu()
        if (rollout['scene_ids'][slot]!=item['scene_id'] or rollout['observation_hashes'][slot]!=item['observation_hash']
            or rollout['condition_hashes'][slot]!=item['condition_hash'] or sha(raw.numpy().tobytes())!=item['raw_sha256']
            or train.get(item['scene_id'])!=item['source_log']):raise ValueError('Selected scene/raw/provenance mismatch')
        identity=request_map[item['candidate_id']];record=record_map[item['candidate_id']]
        for key in ('scene_id','source_log','trajectory_sha256','metric_cache_sha256','score_protocol_hash','metric_cache_path'):
            if identity[key]!=item[key]:raise ValueError('Original request/selection mismatch: '+key)
        request=ConsequenceRequest(**identity,trajectory=record['requested_trajectory'],observe=True)
        if identity!=record['identity'] or request.identity()!=identity:raise ValueError('Original feedback join mismatch')
        validated_reference_input(record,request)
        if (array_sha256(record['requested_trajectory'])!=request.trajectory_sha256
            or record.get('score_units')!='0..1' or set(record['metrics'])!=set(METRICS)
            or any(record['metrics_valid'].get(k) is not True for k in METRICS)
            or any(not math.isfinite(record['metrics'][k]) or not 0<=record['metrics'][k]<=1 for k in METRICS)):
            raise ValueError('Invalid original canonical labels')
        loaded.append((item,raw,request,record))
    return loaded


def charge_official_call(root,item,identity,budget,limits,*,failure_mode=False):
    calls=sorted((root/'calls').glob('*.json')) if (root/'calls').exists() else []
    if len(calls)>=limits['max_official_calls']:raise LimitError('OFFICIAL_CALL_CAP')
    if failure_mode:
        permitted={item['candidate_id']+':road_direction_v1:'+name for name in VARIANTS[1:]}
        saved=[small_json(path) for path in calls]
        own=[row for row in saved if row['scene_id']==item['scene_id']]
        if (identity['candidate_id'] not in permitted or len(own)>=3
                or any(row['identity']['candidate_id']==identity['candidate_id'] for row in own)):
            raise LimitError('PER_SCENE_THREE_NEW_VARIANTS_NO_RETRY_CAP')
    budget.check()
    budget.write(root/'calls'/f'{len(calls):03d}.json',{'identity':identity,
        'scene_id':item['scene_id'],'charged_before_call':True,'attempted_unix':time.time()})


def require_repairable_native_t0(record):
    from tools.st_world_policy.road_failure_selection import native_t0
    from starVLA.model.modules.flow_posttraining.road_margin_proxy import DiagnosticUndefined
    if native_t0(record):raise DiagnosticUndefined('IRREPARABLE_NATIVE_ROAD_FAILURE_AT_T0')


def command_geometry_audit(physical,initial_state,action_sampling,sampling,vehicle):
    """No map, LQR or score call: compare the commanded geometry only."""
    import numpy as np
    import torch
    from starVLA.model.modules.flow_posttraining.road_margin_proxy import (
        Footprint,command_poses,command_time_grid,footprint_corners,periodic_difference)
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import transform_trajectory,get_trajectory_as_array
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import state_array_to_coords_array
    physical=np.asarray(physical,dtype=np.float64)
    initial=torch.tensor(initial_state.rear_axle.serialize(),dtype=torch.float64)
    poses,_=command_poses(torch.from_numpy(physical),initial,initial_state.time_us)
    trajectory=transform_trajectory(Trajectory(physical,action_sampling),initial_state)
    official=get_trajectory_as_array(trajectory,sampling,initial_state.time_point)
    corners=footprint_corners(poses,Footprint.from_vehicle(vehicle)).detach().numpy()
    official_corners=state_array_to_coords_array(official[None],vehicle)[0,:,:4]
    values=poses.detach().numpy()
    xy_error=values[:,:2]-official[:,:2]
    yaw_error=periodic_difference(poses[:,2],torch.tensor(official[:,2])).numpy()
    corner_error=corners-official_corners
    if not all(np.isfinite(x).all() for x in (values,official,corners,official_corners)):
        raise ValueError('NONFINITE_OFFICIAL_OR_PROXY_GEOMETRY')
    source_us,target_us=command_time_grid(initial_state.time_us)
    # Read source knots from the real official trajectory, not our grid helper.
    official_source=[p.time_us for p in trajectory.get_sampled_trajectory()]
    official_times=np.arange(0.,sampling.time_horizon+sampling.interval_length,sampling.interval_length)
    official_times+=initial_state.time_point.time_s
    official_target=np.clip([int(t*1e6) for t in official_times],trajectory.start_time.time_us,trajectory.end_time.time_us)
    def worst(error):
        index=np.unravel_index(np.argmax(abs(error)),error.shape)
        return {'index':list(map(int,index)),'signed_error':float(error[index])}
    return {'command_xy_error_m':float(abs(xy_error).max()),
        'command_yaw_error_rad':float(abs(yaw_error).max()),'corner_max_error_m':float(abs(corner_error).max()),
        'geometry_parity_tolerance':1e-9,'initial_time_us':initial_state.time_us,
        'initial_rear_axle':initial.tolist(),'timepoint_seconds':initial_state.time_point.time_s,
        'legacy_division_seconds':initial_state.time_us/1e6,
        'proxy_source_time_us':source_us.tolist(),'proxy_target_time_us':target_us.tolist(),
        'official_source_time_us':official_source,'official_target_time_us':official_target.tolist(),
        'worst_xy':worst(xy_error),'worst_yaw':worst(yaw_error),'worst_corner':worst(corner_error),
        'proxy_poses':values.tolist(),'official_poses':official[:,:3].tolist(),
        'proxy_corners':corners.tolist(),'official_corners':official_corners.tolist(),
        'official_score_calls':0,'includes_LQR':False}


def parity_and_objective(raw,request,cache,engine):
    import torch
    from starVLA.model.modules.flow_posttraining.road_margin_proxy import (
        AnchoredDecode,CommandRoadCost,Footprint,RoadUnion)
    gate=decode_discrepancy_gate(raw,request.trajectory)
    actual=Footprint.from_vehicle(cache.ego_state.car_footprint.vehicle_parameters)
    scored=Footprint.from_vehicle(engine.scorer._vehicle_parameters)
    if actual!=scored:raise ValueError('Simulator/scorer footprint mismatch')
    if type(cache).__name__!='MetricCache' or set(vars(cache))!={'file_path','trajectory','ego_state','observation','centerline','route_lane_ids','drivable_area_map'}:
        raise ValueError('Foreign metric cache schema')
    area=cache.drivable_area_map
    if len(area._geometries)!=len(area.tokens) or len(area.tokens)!=len(area.map_types):raise ValueError('Invalid map layout')
    # Limit pre-union workload in addition to process address-space/RSS limits.
    if len(area._geometries)>100000:raise LimitError('MAP_POLYGON_COUNT_CAP')
    decoder=AnchoredDecode(raw.double(),torch.as_tensor(request.trajectory,dtype=torch.float64))
    initial=torch.tensor(cache.ego_state.rear_axle.serialize(),dtype=torch.float64)
    audit=command_geometry_audit(decoder(raw.double()).numpy(),cache.ego_state,
        engine.action_sampling,engine.sampling,engine.scorer._vehicle_parameters)
    audit.update(decode_gate=gate,vehicle=asdict(actual))
    require_geometry_parity(audit)
    road=RoadUnion.from_map(area)
    objective=CommandRoadCost(decoder,road,scored,initial,cache.ego_state.time_us)
    return objective,{**audit,'segments':len(road.segments)}


def tracking_diagnostics(command,executed):
    import numpy as np
    command=np.asarray(command,dtype=np.float64);executed=np.asarray(executed,dtype=np.float64)
    velocity=np.diff(command[:,:2],axis=0)/.1
    acceleration=np.diff(velocity,axis=0)/.1
    heading_delta=np.arctan2(np.sin(np.diff(command[:,2])),np.cos(np.diff(command[:,2])))
    ds=np.linalg.norm(np.diff(command[:,:2],axis=0),axis=-1);moving=ds>0
    yaw_error=np.arctan2(np.sin(executed[:,2]-command[:,2]),np.cos(executed[:,2]-command[:,2]))
    return {'command_finite_difference_dt_seconds':.1,
        'command_max_speed_mps':float(np.linalg.norm(velocity,axis=-1).max()),
        'command_max_acceleration_mps2':float(np.linalg.norm(acceleration,axis=-1).max()),
        'command_curvature_max_abs_per_m_moving_only':float(np.max(abs(heading_delta[moving]/ds[moving]))) if moving.any() else None,
        'command_curvature_undefined_zero_distance_intervals':int((~moving).sum()),
        'executed_max_speed_mps':float(np.linalg.norm(executed[:,3:5],axis=-1).max()),
        'executed_max_acceleration_mps2':float(np.linalg.norm(executed[:,5:7],axis=-1).max()),
        'tracking_max_xy_error_m':float(np.linalg.norm(executed[:,:2]-command[:,:2],axis=-1).max()),
        'tracking_max_yaw_error_rad':float(abs(yaw_error).max())}


def run_worker(reg,admission):
    import torch
    from starVLA.model.modules.flow_posttraining.road_margin_proxy import (
        DiagnosticUndefined,direction_variants,score_diagnostic_variants)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    root=Path(reg['output_root']);limits=execution_limits(reg);budget=Budget(root,limits)
    failure_mode=reg.get('selection_mode')=='failure_cohort_v1'
    def stop(signum,frame):raise LimitError('SIGNAL_OR_DEADLINE: '+str(signum))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop)
    signal.alarm(limits['wall_seconds']-20)
    resource.setrlimit(resource.RLIMIT_AS,(limits['rss_and_address_space_bytes'],)*2)
    outcome={'status':'PARTIAL','error':None}
    try:
        evidence=json.loads(budget.read(reg['evidence']['path'],reg['evidence']['sha256'],2*2**20 if failure_mode else 256*1024))
        if (evidence.get('schema')=='road_failure_direction_selection_v1')!=failure_mode:
            raise ValueError('Failure selection requires its explicit versioned execution schema')
        for source in evidence['sources']:budget.read(source['path'],source['sha256'],2*2**20)
        loaded=load_inputs(evidence,budget)
        first=loaded[0][2]
        if not Path(admission['maps_root']).is_dir():raise ValueError('Actual maps root unavailable')
        cache_roots=tuple(sorted({str(Path(item['metric_cache_path']).parents[3]) for item,_,_,_ in loaded}))
        spec=ConsequenceSpec(str(Path(evidence['source_runtime'])/'navsim_v1.1/navsim'),
            admission['maps_root'],cache_roots,first.campaign_id,first.score_protocol_hash,True)
        engine=VerifiedContextEngine(spec)
        for item,raw,request,record in loaded:
            budget.check();scene=root/'scenes'/item['scene_id']
            if (scene/'SCENE_RESULT.json').exists():continue
            if (scene/'STARTED.json').exists():
                # An interrupted attempt is not replayed: an admitted call may
                # have reached native execution without a committed return.
                budget.write(scene/'SCENE_RESULT.json',{'scene_id':item['scene_id'],
                    'status':'UNCERTAIN_INTERRUPTED_NOT_RETRIED','original_metrics':record['metrics']})
                continue
            budget.write(scene/'STARTED.json',{'scene_id':item['scene_id'],'request':request.identity(),
                'original_metrics':record['metrics'],'original_physical':record['requested_trajectory'],
                'original_raw':raw.tolist(),'variants':{n:'PENDING' for n in VARIANTS}})
            try:
                if failure_mode:
                    require_repairable_native_t0(record)
                compressed=budget.read(request.metric_cache_path,request.metric_cache_sha256,64*2**20)
                cache=bounded_unpack(compressed,limits['cache_expanded_bytes']);del compressed
                engine.bind(request,cache,budget.verified_stats[str(Path(request.metric_cache_path).resolve())]);budget.check()
                obj,audit=parity_and_objective(raw,request,cache,engine);budget.check()
                budget.write(scene/'NUMERICAL_PARITY.json',audit)
                directions=direction_variants(obj,raw.double(),item['scene_id'],reg['selection_hash']);budget.check()
                # Original/perturbed payloads are durable BEFORE the first score.
                values=json_ready({**directions,'gradient':directions['gradient'].tolist(),
                    'variants':{n:{k:(v.detach().cpu().tolist() if torch.is_tensor(v) else v)
                        for k,v in row.items()} for n,row in directions['variants'].items()}})
                budget.write(scene/'DIRECTIONS.json',values)
                def charge(identity):
                    charge_official_call(root,item,identity,budget,limits,failure_mode=failure_mode)
                def save(value):
                    if value['record'].get('valid'):
                        value['tracking_and_kinematics']=tracking_diagnostics(
                            directions['variants'][value['variant']]['command_poses'].numpy(),
                            value['record']['executed_states'])
                    budget.write(scene/(value['variant']+'_RESULT.json'),value)
                scored=score_diagnostic_variants(engine,request,record,directions,obj.road,obj.vehicle,
                    fingerprint({'registration':reg,'source':reg['source_hashes']}),charge,save)
                result={'scene_id':item['scene_id'],'status':'COMPLETE' if all(v['record'].get('valid') for v in scored.values()) else 'PARTIAL_SCORING_FAILURE',
                    'original_metrics':record['metrics'],'variants':{n:{'valid':v['record'].get('valid',False),
                        'status':v['record'].get('status'),'metrics':v['record'].get('metrics'),
                        'geometry':v['geometry'],'command_cost_m':directions['variants'][n]['command_cost_m']}
                        for n,v in scored.items()}}
                budget.write(scene/'SCENE_RESULT.json',result)
            except (DiagnosticUndefined,ValueError,LimitError,MemoryError,RuntimeError,KeyError,OSError) as error:
                # No substitute scene, epsilon or trajectory; maintain original denominator.
                result={'scene_id':item['scene_id'],'status':'UNDEFINED_DIRECTION' if isinstance(error,DiagnosticUndefined) else 'FAILED',
                        'error_type':type(error).__name__,'error':str(error),'original_metrics':record['metrics']}
                if isinstance(error,GeometryParityError):
                    budget.write(scene/'NUMERICAL_PARITY_FAILURE.json',error.audit)
                    result['numerical_parity']=error.audit
                # Signal/time/RSS exceptions must end this worker, not continue scoring.
                if isinstance(error,(LimitError,MemoryError)):raise
                budget.write(scene/'SCENE_RESULT.json',result)
        outcome['status']='WORKER_FINISHED'
    except BaseException as error:
        outcome.update(error_type=type(error).__name__,error=str(error))
    finally:
        signal.alarm(0)
        outcome['resources']=budget.report();outcome['cuda_initialized']=torch.cuda.is_initialized()
        # Small terminal receipt uses the reserved2MiB, even when normal budget
        # writes are blocked. Parent separately handles native crash/hard timeout.
        atomic_json(root/('WORKER_RESULT_'+str(admission['attempt_id'])+'.json'),outcome)
    return 0 if outcome['status']=='WORKER_FINISHED' else 2


def summarize(root,evidence,supervisor):
    rows=[]
    for item in evidence['selected']:
        p=root/'scenes'/item['scene_id']/'SCENE_RESULT.json'
        if p.exists():row=small_json(p)
        else:row={'scene_id':item['scene_id'],'status':'PENDING_OR_INTERRUPTED','variants':{v:'UNRESOLVED' for v in VARIANTS}}
        rows.append(row)
    complete=[r for r in rows if r['status']=='COMPLETE']
    calls=len(list((root/'calls').glob('*.json'))) if (root/'calls').exists() else 0
    failure_mode=evidence.get('schema')=='road_failure_direction_selection_v1'
    from tools.st_world_policy.road_failure_selection import EXECUTION_SCHEMA
    requested=len(evidence['selected'])
    originals={item['scene_id']:item['failed']['metrics'] for item in evidence['selected']} if failure_mode else {
        row['scene_id']:row['original_metrics'] for row in rows if 'original_metrics' in row}
    # Legacy fixture metadata can omit labels. Unknown originals are explicitly
    # distinct from zero failures; real legacy preparation still binds DAC=1.
    original_dac=[value['drivable_area_compliance'] for value in originals.values()]
    failed=sum(value<1.0 for value in original_dac)
    result={'schema':EXECUTION_SCHEMA if failure_mode else SCHEMA,
            'requested_scenes':requested,'complete_scenes':len(complete),
            'unresolved_or_undefined_scenes':requested-len(complete),'charged_official_calls':calls,
            'status':'COMPLETE_DIAGNOSTIC_NOT_SCIENTIFIC_PASS' if len(complete)==requested else 'PARTIAL_OR_UNDEFINED',
            'all_original_DAC_equal_1':len(original_dac)==requested and all(v==1. for v in original_dac),
            'original_DAC_labels_known':len(original_dac),'road_repair_denominator':failed,
            'selection_hash':evidence['selection_sha256_before_proxy'] if failure_mode else evidence['selection_sha256_before_scores'],
            'supervisor':supervisor,'scenes':rows}
    if failure_mode:
        for row in rows:
            if 'original_metrics' in row and row['original_metrics']!=originals[row['scene_id']]:
                raise ValueError('Summary original labels differ from frozen selection')
        result.update(original_DAC_failure_scenes=failed,
            eligible_original_DAC_failures=sum(not item['failed']['native_t0'] for item in evidence['selected']),
            original_t0_failure_scenes=sum(item['failed']['native_t0'] for item in evidence['selected']),
            population_summary_complete=len(complete)==requested,
            requested_original_compliant_denominators={metric:sum(row[metric]==1. for row in originals.values())
                for metric in ('no_at_fault_collisions','drivable_area_compliance',
                               'time_to_collision_within_bound','comfort')})
        repair={}
        for name in VARIANTS:
            valid=[row for row in complete if originals[row['scene_id']]['drivable_area_compliance']<1]
            repaired=sum(row['variants'][name]['metrics']['drivable_area_compliance']==1 for row in valid)
            repair[name]={'requested_original_failure_denominator':failed,
                'common_valid_original_failure_denominator':len(valid), 'repaired_DAC_scenes':repaired,
                'unresolved_original_failures':failed-len(valid),
                'repair_rate_valid_subset':repaired/len(valid) if valid else None,
                'known_repairs_over_requested_lower_bound':repaired/failed if failed else None}
        result['DAC_repair']=repair
    if complete:
        contrast={}
        for comparator in ('positive','random'):
            diffs=[r['variants']['negative']['geometry']['custom_executed_cost_m']-
                   r['variants'][comparator]['geometry']['custom_executed_cost_m'] for r in complete]
            contrast['negative_minus_'+comparator]={'mean_custom_executed_cost_difference_m':sum(diffs)/len(diffs),
                'common_valid_scenes':len(diffs),'negative_lower_cost_scenes':sum(v<0 for v in diffs),
                'equal_cost_scenes':sum(v==0 for v in diffs),'negative_higher_cost_scenes':sum(v>0 for v in diffs)}
        result['valid_subset_only_direction_contrasts']=contrast
        risk={}
        for name in VARIANTS:
            risk[name]={'valid_scenes':len(complete),'mean_official_score_valid_subset':sum(r['variants'][name]['metrics']['score'] for r in complete)/len(complete),
                'original_compliant_denominators_valid_subset':{metric:sum(r['original_metrics'][metric]==1. for r in complete)
                    for metric in ('no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound','comfort')},
                'new_DAC_failures':sum(r['variants'][name]['metrics']['drivable_area_compliance']<1 and r['original_metrics']['drivable_area_compliance']==1 for r in complete),
                'new_NC_failures':sum(r['variants'][name]['metrics']['no_at_fault_collisions']<1 and r['original_metrics']['no_at_fault_collisions']==1 for r in complete),
                'high_to_zero':sum(r['original_metrics']['score']>=.8 and r['variants'][name]['metrics']['score']<=1e-8 for r in complete),
                'zero_scores':sum(r['variants'][name]['metrics']['score']<=1e-8 for r in complete),
                'mean_component_differences_vs_original':{metric:sum(r['variants'][name]['metrics'][metric]-r['original_metrics'][metric] for r in complete)/len(complete) for metric in METRICS},
                'new_TTC_failures':sum(r['variants'][name]['metrics']['time_to_collision_within_bound']<1 and r['original_metrics']['time_to_collision_within_bound']==1 for r in complete)}
        result['valid_subset_only_risk']=risk
    return result


def _child_entry(reg_path,admission_path,admission_hash,log_path):
    with Path(log_path).open('xb') as stream:
        os.dup2(stream.fileno(),1);os.dup2(stream.fileno(),2)
        reg,admission=check_admission(reg_path,admission_path,admission_hash)
        raise SystemExit(run_worker(reg,admission))


def supervise(reg_path,admission_path,admission_hash,resume):
    import multiprocessing
    reg,admission=check_admission(reg_path,admission_path,admission_hash)
    limits=execution_limits(reg) if 'schema' in reg else LIMITS
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or any(os.environ.get(k)!='1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')):
        raise ValueError('CPU-only CUDA hidden and all numerical threads=1 required')
    root=Path(reg['output_root']);root.mkdir(parents=True,exist_ok=True)
    with (root/'RUN.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        started=root/'EXECUTION_BINDING.json'
        identity={'registration_sha256':sha(Path(reg_path).read_bytes()),'source_hashes':reg['source_hashes']}
        if started.exists() and not resume:raise ValueError('Output already started; explicit --resume required; no retries')
        if started.exists() and small_json(started)!=identity:raise ValueError('Cannot resume foreign execution')
        atomic_json(started,identity)
        attempt=root/('ADMISSION_'+str(admission['attempt_id'])+'.json')
        if attempt.exists():raise ValueError('Admission attempt already consumed; no automatic replay')
        atomic_json(attempt,admission)
        raw=Path(reg['evidence']['path']).read_bytes()
        if sha(raw)!=reg['evidence']['sha256']:raise ValueError('Evidence changed')
        evidence=json.loads(raw)
        if (evidence.get('schema')=='road_failure_direction_selection_v1')!=(reg.get('selection_mode')=='failure_cohort_v1'):
            raise ValueError('Failure selection requires its explicit versioned execution schema')
        start=time.monotonic();why=None;peak=0
        log_path=root/('WORKER_'+str(admission['attempt_id'])+'.log')
        child=multiprocessing.get_context('spawn').Process(target=_child_entry,
            args=(str(reg_path),str(admission_path),admission_hash,str(log_path)))
        previous_handlers={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT)}
        tracker_pid=None;monitor_error=None;cleanup_error=None;exit_confirmed=False
        def interrupted(signum,frame):
            raise LimitError('SUPERVISOR_SIGNAL_'+str(signum))
        try:
            for sig in previous_handlers:signal.signal(sig,interrupted)
            child.start()
            # CPython spawn's known owned helper, not broad process discovery.
            from multiprocessing import resource_tracker
            tracker_pid=resource_tracker._resource_tracker._pid
            while child.is_alive():
                elapsed=time.monotonic()-start
                rss=0
                for pid in {os.getpid(),child.pid,tracker_pid}-{None}:
                    try:
                        fields=Path(f'/proc/{pid}/status').read_text().splitlines()
                        rss+=next(int(line.split()[1])*1024 for line in fields if line.startswith('VmRSS:'))
                    except (FileNotFoundError,ProcessLookupError,StopIteration):pass
                peak=max(peak,rss)
                output=owned_output_bytes(root)
                if elapsed>=limits['wall_seconds']-20:why='WALL_CAP'
                elif rss>limits['rss_and_address_space_bytes']:why='AGGREGATE_RSS_CAP'
                elif output>limits['output_bytes']-2*2**20:why='OUTPUT_CAP'
                if why:break
                time.sleep(.2)
        except BaseException as error:
            why='SUPERVISOR_INTERRUPTED' if isinstance(error,(KeyboardInterrupt,LimitError)) else 'MONITOR_ERROR'
            monitor_error={'type':type(error).__name__,'message':str(error)}
        finally:
            # Keep the run lock until the only owned worker has exited. Repeated
            # operator signals cannot interrupt the bounded TERM/KILL cleanup.
            for sig in previous_handlers:signal.signal(sig,signal.SIG_IGN)
            try:
                if child.pid is not None:
                    if child.is_alive():child.terminate()
                    child.join(timeout=15)
                    if child.is_alive():child.kill();child.join(timeout=3)
                    exit_confirmed=not child.is_alive()
                else:exit_confirmed=True
            except BaseException as error:
                cleanup_error={'type':type(error).__name__,'message':str(error)}
                # Still make one explicit owned-child kill attempt if TERM/join
                # itself failed; never target other jobs or process groups.
                if child.pid is not None:
                    try:
                        if child.is_alive():child.kill()
                        child.join(timeout=3);exit_confirmed=not child.is_alive()
                    except BaseException as second:
                        cleanup_error['final_error']=repr(second)
        try:
            info={'worker_exit_code':child.exitcode,'stop_reason':why,'elapsed_seconds':time.monotonic()-start,
                  'parent_child_tracker_peak_rss_bytes':peak,'resource_tracker_pid':tracker_pid,
                  'owned_worker_pid':child.pid,'owned_worker_exit_confirmed':exit_confirmed,
                  'monitor_error':monitor_error,'cleanup_error':cleanup_error,'attempt_id':admission['attempt_id']}
            try:result=summarize(root,evidence,info)
            except BaseException as error:
                requested=len(evidence['selected'])
                result={'schema':reg.get('schema',SCHEMA),'requested_scenes':requested,'complete_scenes':0,
                    'unresolved_or_undefined_scenes':requested,'status':'PARTIAL_SUMMARY_ERROR',
                    'charged_official_calls':None,'supervisor':info,'summary_error':repr(error),
                    'scenes':[{'scene_id':item['scene_id'],'status':'UNRESOLVED_SUMMARY_FAILURE'} for item in evidence['selected']]}
            atomic_json(root/('RESULT_'+str(admission['attempt_id'])+'.json'),result)
            print(json.dumps({'status':result['status'],'complete_scenes':result['complete_scenes'],
                              'charged_official_calls':result['charged_official_calls'],'output_root':str(root)}))
            return 0 if result['complete_scenes']==len(evidence['selected']) and child.exitcode==0 and exit_confirmed and why is None else 2
        finally:
            for sig,handler in previous_handlers.items():signal.signal(sig,handler)


def bind_failure_selection(frame_path,frame_hash,evidence_path,output):
    """Explicit, bounded CPU preparation; never create an execution admission."""
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or any(os.environ.get(k)!='1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')):
        raise ValueError('Binding requires CUDA hidden and numerical threads=1')
    output=Path(output)
    if output.exists():raise ValueError('Selection output already exists; preserve original binding')
    output.parent.mkdir(parents=True,exist_ok=True)
    report_path=output.with_name(output.stem+'_BINDING_REPORT.json')
    if report_path.exists():raise ValueError('Binding receipt already exists; use an explicit new attempt path')
    budget=Budget(output.parent,BINDING_LIMITS)
    allowed=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,set(allowed[:2]))
    original_alarm=signal.getsignal(signal.SIGALRM)
    original_cpu_handler=signal.getsignal(signal.SIGXCPU);original_cpu_limit=resource.getrlimit(resource.RLIMIT_CPU)
    def expired(signum,frame):raise LimitError('BINDING_WALL_CAP')
    signal.signal(signal.SIGALRM,expired);signal.alarm(BINDING_LIMITS['wall_seconds'])
    def cpu_expired(signum,frame):raise LimitError('BINDING_CPU_CAP')
    signal.signal(signal.SIGXCPU,cpu_expired)
    current_cpu=resource.getrusage(resource.RUSAGE_SELF)
    soft_cpu=math.floor(current_cpu.ru_utime+current_cpu.ru_stime+BINDING_LIMITS['cpu_seconds'])
    if original_cpu_limit[0]!=resource.RLIM_INFINITY:soft_cpu=min(soft_cpu,original_cpu_limit[0])
    resource.setrlimit(resource.RLIMIT_CPU,(soft_cpu,original_cpu_limit[1]))
    result={'schema':'road_failure_selection_binding_receipt_v1','status':'FAILED_CLOSED',
        'frame':{'path':str(Path(frame_path).resolve()),'sha256':frame_hash},
        'limits':BINDING_LIMITS,'training_updates':0,'new_official_calls':0,'proxy_calls':0}
    try:
        import torch
        from tools.st_world_policy.road_failure_selection import bind_frame,execution_read_plan
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        value,audit=bind_frame(frame_path,frame_hash,evidence_path,budget,decode_discrepancy_gate)
        audit['execution_read_plan']=execution_read_plan(value)
        if audit['execution_read_plan']['total_worker_asset_bytes']>FAILURE_LIMITS['asset_read_bytes']:
            raise LimitError('EXECUTION_READ_PLAN_EXCEEDS_UNCHANGED_256MiB_CAP')
        budget.write(output,value);budget.check()
        result.update(status='BOUND_BEFORE_PROXY_NOT_EXECUTION_APPROVAL',audit=audit,
            selection={'path':str(output.resolve()),'sha256':sha(output.read_bytes())})
    except (ValueError,TypeError,KeyError,OSError,RuntimeError,MemoryError) as error:
        result.update(error_type=type(error).__name__,error=str(error))
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,original_alarm)
        resource.setrlimit(resource.RLIMIT_CPU,original_cpu_limit);signal.signal(signal.SIGXCPU,original_cpu_handler)
        result['resources']=budget.report();result['cpu_affinity']=sorted(os.sched_getaffinity(0))
        result['population']={'requested_scene_ids':getattr(budget,'binding_requested_scene_ids',[]),
            'bound_scene_ids':getattr(budget,'binding_bound_scene_ids',[]),
            'last_active_scene_id':getattr(budget,'binding_active_scene_id',None),'replacement_scenes':0}
        os.sched_setaffinity(0,set(allowed))
        atomic_json(report_path,result)
    print(json.dumps({'status':result['status'],'report':str(report_path),'new_official_calls':0,'proxy_calls':0}))
    return 0 if result['status']=='BOUND_BEFORE_PROXY_NOT_EXECUTION_APPROVAL' else 2


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true');mode.add_argument('--execute',action='store_true')
    mode.add_argument('--bind-failure-selection',action='store_true',help='Read only the frozen original C1 TRAIN frame and bind its exact candidates; no proxy/scoring')
    parser.add_argument('--evidence',type=Path,default=EVIDENCE)
    parser.add_argument('--registration',type=Path)
    parser.add_argument('--selection-mode',choices=('fixed32_v1','failure_cohort_v1'),default='fixed32_v1')
    parser.add_argument('--frame',type=Path);parser.add_argument('--frame-sha256')
    parser.add_argument('--selection-output',type=Path,help='New immutable bound-selection JSON; binding mode only')
    parser.add_argument('--output-root',type=Path,help='Required by prepare; cannot change at execute')
    parser.add_argument('--admission',type=Path);parser.add_argument('--admission-sha256')
    parser.add_argument('--resume',action='store_true',help='Continue untouched scenes only, never retry an admitted/partial scene')
    args=parser.parse_args(argv)
    try:
        if args.bind_failure_selection:
            if (not args.frame or not args.frame_sha256 or not args.selection_output
                    or args.registration or args.output_root or args.admission or args.resume):
                raise ValueError('Binding needs frame/hash/new selection-output and no execution arguments')
            return bind_failure_selection(args.frame,args.frame_sha256,args.evidence,args.selection_output)
        if args.frame or args.frame_sha256 or args.selection_output:raise ValueError('Binding arguments require explicit binding mode')
        if args.registration is None:raise ValueError('Prepare/execute requires registration')
        if args.prepare:
            if args.output_root is None or args.admission or args.resume:raise ValueError('Prepare requires output root and no admission/resume')
            value=registration(args.evidence,args.output_root,args.selection_mode);atomic_json(args.registration,value)
            print(json.dumps({'status':value['status'],'registration':str(args.registration),
                              'sha256':sha(args.registration.read_bytes()),'dispatch_authorized':False}));return 0
        if args.output_root or not args.admission or not args.admission_sha256:raise ValueError('Execution needs exact admission hash and registered output root')
        return supervise(args.registration,args.admission,args.admission_sha256,args.resume)
    except (ValueError,TypeError,KeyError,OSError,RuntimeError) as error:
        print(json.dumps({'status':'FAILED_CLOSED','error_type':type(error).__name__,'error':str(error)}),file=sys.stderr)
        return 2


if __name__=='__main__':raise SystemExit(main())
