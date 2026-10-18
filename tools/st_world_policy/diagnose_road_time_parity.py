"""One geometry-only replay of the nine fixed v2 parity failures; never scores.

Prepare reads small existing receipts only. Execute is for the Lead after an
exact admission; it reads those nine original caches, not rollouts or weights.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import socket
import sys
import time
from unittest.mock import patch

# This executable has no GPU mode. Preserve original model/precision settings.
os.environ['CUDA_VISIBLE_DEVICES']=''
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools.st_world_policy import run_road_direction as r

ORIGINAL_ROOT=Path('/mnt/project/st-world-policy-20261016/audits/consequence/road_direction_fixed32_v2')
NAVSIM_ROOT=Path('/mnt/project/VLA-Drive-st-flow-context-runtime-v3-20261017/navsim_v1.1/navsim')
IDS=tuple(sorted(('18c080c7deeb5788','261964b9701e5ee5','429854b6e1a257a5',
    '594c142e00fc53eb','66957e8032bf54ca','7837d9aa8c285e14','97c6b7c64014592c',
    'a31f5736e4c658c4','a3c04b44e71f59d7')))
LIMITS={**r.LIMITS,'wall_seconds':120,'asset_read_bytes':32*2**20,
    'output_bytes':4*2**20,'rss_and_address_space_bytes':8*2**30,
    'max_official_calls':0,'scenes':9,'variants':0}


def legacy_time_grid(time_us):
    """Only the old division differs; used exclusively for diagnostic replay."""
    import numpy as np
    start=time_us/1e6
    source=np.array([time_us]+[int(t*1e6) for t in np.arange(0.,4.,.5)+.5+start],dtype=np.int64)
    target=np.clip(np.array([int(t*1e6) for t in np.arange(0.,4.+.1,.1)+start],dtype=np.int64),source[0],source[-1])
    return source,target


def code_hashes():
    values={str(ROOT/p):h for p,h in r.source_hashes().items()}
    path=Path(__file__).resolve();values[str(path)]=r.sha(path.read_bytes())
    nuplan=Path(importlib.util.find_spec('nuplan').origin).parent
    files=[NAVSIM_ROOT/'navsim'/p for p in ('evaluate/pdm_score.py','common/dataclasses.py',
        'planning/simulation/planner/pdm_planner/utils/pdm_array_representation.py')]
    files += [nuplan/p for p in ('common/actor_state/state_representation.py',
        'common/actor_state/ego_state.py','common/actor_state/car_footprint.py',
        'common/actor_state/oriented_box.py','common/actor_state/vehicle_parameters.py',
        'common/geometry/convert.py','common/geometry/compute.py',
        'planning/simulation/trajectory/interpolated_trajectory.py',
        'planning/simulation/planner/ml_planner/transform_utils.py')]
    for p in files:values[str(p)]=r.sha(p.read_bytes())
    return values


def prepare(output_root):
    """No cache stat/open. Capture exact small receipt identities before replay."""
    output_root=Path(output_root).resolve()
    if output_root.is_relative_to(ORIGINAL_ROOT):raise ValueError('Do not write under original v2 evidence')
    rows=[];refs=[]
    paths=sorted((ORIGINAL_ROOT/'scenes').glob('*/SCENE_RESULT.json'))
    if len(paths)!=32:raise ValueError('Expected original fixed32 denominator')
    for p in paths:
        result=r.small_json(p)
        refs.append({'path':str(p),'sha256':r.sha(p.read_bytes())})
        if result.get('error')!='REAL_OFFICIAL_GEOMETRY_PARITY_FAILURE':continue
        scene=p.parent.name;started_path=p.parent/'STARTED.json'
        started=r.small_json(started_path);request=started['request']
        if scene!=started['scene_id'] or scene!=request['scene_id']:raise ValueError('Original receipt scene mismatch')
        if r.array_sha256(started['original_physical'])!=request['trajectory_sha256']:
            raise ValueError('Original saved trajectory hash differs')
        cache=Path(request['metric_cache_path'])
        if list(cache.parts[-4:])!=[request['source_log'],'unknown',scene,'metric_cache.pkl']:
            raise ValueError('Original cache path identity mismatch')
        refs.append({'path':str(started_path),'sha256':r.sha(started_path.read_bytes())})
        rows.append({'scene_id':scene,'started_path':str(started_path),'request':request})
    if tuple(row['scene_id'] for row in rows)!=IDS:raise ValueError('Not the fixed nine original parity failures')
    binding=ORIGINAL_ROOT/'EXECUTION_BINDING.json'
    refs.append({'path':str(binding),'sha256':r.sha(binding.read_bytes())})
    return {'schema':'road_time_parity_replay_v1','status':'PREPARED_NOT_EXECUTED',
        'original_root':str(ORIGINAL_ROOT),'original_denominator':32,'selected_failed_denominator':9,
        'scene_ids':list(IDS),'rows':rows,'receipt_refs':refs,'source_hashes':code_hashes(),
        'source_commit':r.source_commit(),'navsim_root':str(NAVSIM_ROOT),'limits':LIMITS,
        'output_root':str(output_root),'official_score_calls':0,'training':False,
        'geometry_tolerance_unchanged':1e-9,'dispatch_authorized':False}


def check_admission(plan_path,admission_path,admission_hash):
    plan=r.small_json(plan_path);admission=r.small_json(admission_path)
    if r.sha(Path(admission_path).read_bytes())!=admission_hash:raise ValueError('Admission checksum mismatch')
    if (plan['schema']!='road_time_parity_replay_v1' or plan['limits']!=LIMITS
        or plan['source_hashes']!=code_hashes() or plan['source_commit']!=r.source_commit()
        or tuple(plan['scene_ids'])!=IDS or plan['navsim_root']!=str(NAVSIM_ROOT)):
        raise ValueError('Replay code/constants/scene identity changed')
    if (admission.get('status')!='APPROVED_GEOMETRY_REPLAY_ONLY' or admission.get('issuer')!='lead'
        or admission.get('plan_sha256')!=r.sha(Path(plan_path).read_bytes())
        or admission.get('host')!=socket.gethostname()
        or not admission.get('not_before_unix',float('inf'))<=time.time()<admission.get('expires_unix',0)
        or admission['expires_unix']-time.time()<LIMITS['wall_seconds']):
        raise ValueError('Exact Lead geometry-only admission required')
    return plan


def execute(plan):
    import numpy as np
    import torch
    from starVLA.model.modules.flow_posttraining import road_margin_proxy as proxy
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    root=Path(plan['output_root']);root.mkdir(parents=True,exist_ok=False)
    budget=r.Budget(root,LIMITS)
    rows={scene:{'scene_id':scene,'status':'UNATTEMPTED'} for scene in IDS}
    r.atomic_json(root/'ATTEMPT_STARTED.json',{'plan':plan,'rows':list(rows.values())})
    old_handlers={}
    def stop(signum,frame):raise r.LimitError('GEOMETRY_REPLAY_SIGNAL_OR_DEADLINE:'+str(signum))
    for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT):
        old_handlers[sig]=signal.signal(sig,stop)
    signal.alarm(115)
    resource.setrlimit(resource.RLIMIT_AS,(LIMITS['rss_and_address_space_bytes'],)*2)
    overall_error=None
    try:
        for ref in plan['receipt_refs']:budget.read(ref['path'],ref['sha256'],2**20)
        # Geometry-source revalidation is accounted; no large checkpoint or manifest.
        for path,digest in plan['source_hashes'].items():budget.read(path,digest,2**20)
        sys.path.insert(0,plan['navsim_root'])
        import navsim
        if not Path(navsim.__file__).resolve().is_relative_to(NAVSIM_ROOT):raise ValueError('Foreign navsim module')
        from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
        from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
        action=TrajectorySampling(num_poses=8,interval_length=.5)
        sampling=TrajectorySampling(num_poses=40,interval_length=.1)
        vehicle=get_pacifica_parameters()
        for row in plan['rows']:
            scene=row['scene_id'];rows[scene]['status']='STARTED'
            try:
                ref=next(ref for ref in plan['receipt_refs'] if ref['path']==row['started_path'])
                started=json.loads(budget.read(ref['path'],ref['sha256'],2**20))
                request=started['request']
                if request!=row['request']:raise ValueError('Changed original request')
                raw=torch.tensor(started['original_raw'],dtype=torch.float32)
                if not np.array_equal(raw.double().numpy(),np.asarray(started['original_raw'])):
                    raise ValueError('Saved raw is not exact FP32 widening')
                gate=r.decode_discrepancy_gate(raw,started['original_physical'])
                compressed=budget.read(request['metric_cache_path'],request['metric_cache_sha256'],32*2**20)
                cache=r.bounded_unpack(compressed,LIMITS['cache_expanded_bytes']);del compressed
                r.validate_cache(cache)
                if proxy.Footprint.from_vehicle(cache.ego_state.car_footprint.vehicle_parameters)!=proxy.Footprint.from_vehicle(vehicle):
                    raise ValueError('Canonical vehicle mismatch')
                args=(started['original_physical'],cache.ego_state,action,sampling,vehicle)
                with patch.object(proxy,'command_time_grid',legacy_time_grid):
                    legacy=r.command_geometry_audit(*args)
                corrected=r.command_geometry_audit(*args)
                value={'scene_id':scene,'status':'GEOMETRY_REPLAY_COMPLETE','original_status':'FAILED',
                    'original_error':'REAL_OFFICIAL_GEOMETRY_PARITY_FAILURE','decode_gate':gate,
                    'legacy_division':legacy,'official_timepoint_multiplication':corrected,
                    'corrected_parity_pass':max(corrected[k] for k in
                        ('command_xy_error_m','command_yaw_error_rad','corner_max_error_m'))<=1e-9,
                    'official_score_calls':0,'source_request':request}
                budget.write(root/(scene+'.json'),value)
                rows[scene]={k:value[k] for k in ('scene_id','status','corrected_parity_pass')}
                del cache
            except (ValueError,RuntimeError,MemoryError,OSError) as error:
                rows[scene].update(status='FAILED',error_type=type(error).__name__,error=str(error))
                if isinstance(error,(r.LimitError,MemoryError)):raise
    except BaseException as error:
        overall_error={'type':type(error).__name__,'message':str(error)}
    finally:
        signal.alarm(0)
        result={'schema':'road_time_parity_result_v1','original_denominator':32,
            'requested_failed_scenes':9,'rows':list(rows.values()),'error':overall_error,
            'resources':budget.report(),'official_score_calls':0,'cuda_initialized':torch.cuda.is_initialized(),
            'old_evidence_modified':False,'science_pass_claimed':False}
        r.atomic_json(root/'RESULT.json',result)
        for sig,handler in old_handlers.items():signal.signal(sig,handler)
    return 0 if all(row['status']=='GEOMETRY_REPLAY_COMPLETE' for row in rows.values()) else 2


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true');mode.add_argument('--execute',action='store_true')
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output-root',type=Path)
    parser.add_argument('--admission',type=Path);parser.add_argument('--admission-sha256')
    args=parser.parse_args(argv)
    try:
        if args.prepare:
            if args.output_root is None:raise ValueError('--output-root required for prepare')
            r.atomic_json(args.plan,prepare(args.output_root));return 0
        if not args.admission or not args.admission_sha256:raise ValueError('Lead admission required; prepare does not authorize cache reads')
        return execute(check_admission(args.plan,args.admission,args.admission_sha256))
    except (ValueError,OSError,RuntimeError,KeyError) as error:
        print(json.dumps({'status':'REFUSED','error':str(error)}),file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
