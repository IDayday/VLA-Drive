"""A-internal learning curve using the unchanged S0 checkpoint/predict path.

This monitors learning, not checkpoint selection on B/dev. The formal shared
anchor remains the exposure-matched terminal checkpoint. Preflight only reads
real A-calibration current inputs and GT; it is not a model evaluation.
"""
import argparse,importlib.util,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('legacy-source','population-contract','current-root','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--training-run',type=Path);p.add_argument('--tag');p.add_argument('--preflight',action='store_true')
    p.add_argument('--engineering-scenes',type=int,default=0,
        help='Explicit small native-checkpoint inference check; never a full learning-curve point')
    p.add_argument('--rank',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    p.add_argument('--max-seconds',type=float,required=True);a=p.parse_args()
    if not 0<=a.rank<a.shards or a.max_seconds<=0:raise ValueError('Bounded shard/time required')
    if a.engineering_scenes<0 or (a.engineering_scenes and (a.preflight or a.shards!=1)):
        raise ValueError('Engineering checkpoint inference is separate from CPU preflight and full shards')
    def module(name,path):
        spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
        sys.modules[name]=m;spec.loader.exec_module(m);return m
    h=module('_learning_helpers',ROOT/'tools/decision_local/common.py');h.check_host()
    contract=h.read(a.population_contract);audit=h.read(contract['source_audit']);codec=h.read(contract['action_codec'])
    if (contract['schema']!='a_only_s0_training_population_v1' or contract['source_audit_identity']!=audit['identity'] or
        codec['identity']!=contract['action_codec_identity']):raise ValueError('Unverified A-only identities')
    rows=h.read(Path(contract['source_audit']).parent/'A_calibration.json')
    if not rows or any(audit['roles'][r['log'].rsplit('_',2)[0]]!='A_calibration' for r in rows):raise ValueError('Cross-role evaluation')
    if a.engineering_scenes>=len(rows):raise ValueError('Use full evaluation for the full population')
    selected=rows[:4] if a.preflight else rows[:a.engineering_scenes] if a.engineering_scenes else rows
    if h.source_sha(a.legacy_source)!='1493deda247107efe076b9294863a6753391298a':raise ValueError('Wrong S0 runtime')
    sys.path.insert(0,str(a.legacy_source))
    adapter=module('_learning_codec',ROOT/'starVLA/dataloader/a_only_codec.py');adapter.install_legacy_codec(codec)
    import numpy as np,torch
    from starVLA.dataloader.foresight_dataset import ForesightCurrentDataset,decode_ego
    from tools.ddpolicy_vehicle.evaluate_ego import trajectory_errors
    data=ForesightCurrentDataset(a.current_root);lookup={r['token']:r for r in data.index}
    if any(lookup.get(r['token'])!=r for r in rows):raise ValueError('Foreign A-calibration current data')
    data.index=rows;ego_identity=h.read(a.current_root/'ego_identity.json')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;training=checkpoint=model=None;start=time.monotonic();cpu=time.process_time()
    if not a.preflight:
        if not a.training_run or not a.tag:raise ValueError('Actual native checkpoint required')
        from tools.foresight.checkpoints import checkpoint_identity,load_student,scene_noise
        training,checkpoint=checkpoint_identity(a.training_run,a.tag)
        if (training['config']['natural_error_provenance']['population_contract_sha256']!=h.digest(a.population_contract) or
            training['data']['population_contract']!=contract or training['config']['foresight']['planner_condition_mode']!='action_only'):
            raise ValueError('Only the registered A-only S0 may use this learning curve')
        model=load_student(a.training_run,a.tag,training,precision='fp32',strip=True)
    elif a.training_run or a.tag or a.shards!=1:raise ValueError('Preflight has no checkpoint or GPU shards')
    declaration=dict(schema='a_internal_base_learning_v2',population_contract_sha256=h.digest(a.population_contract),
        role='A_calibration',index_identity=h.identity(rows),scenes=len(rows),checkpoint=checkpoint,
        runtime_source_sha=h.source_sha(a.legacy_source),wrapper_source_sha=h.source_sha(ROOT),tool_sha256=h.digest(__file__),
        action_codec_identity=codec['identity'],preflight=a.preflight,shards=a.shards,
        engineering_only=bool(a.engineering_scenes),evaluated_scenes=len(selected),
        selected_index_identity=h.identity(selected),
        checkpoint_selection='Fixed shared terminal; this curve cannot select on B or independent development',
        protocol='FP32 native master, TF32 off, original Euler10, one scene-bound q0 seed42; no optimizer')
    ident=h.identity(declaration);a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'identity.json').exists() and h.read(a.output/'identity.json')['identity']!=ident:raise ValueError('Preserve previous curve version')
    h.atomic_json(a.output/'identity.json',dict(identity=ident,**declaration));records=[]
    for i in list(range(len(selected)))[a.rank::a.shards]:
        if time.monotonic()-start>a.max_seconds:raise TimeoutError('Finite A-calibration budget')
        row=rows[i];observation=data[i]
        if model is not None:
            torch.cuda.synchronize();tick=time.monotonic()
            with torch.inference_mode():prediction=model.predict_action([observation],initial_noise=scene_noise(row['token'],42,'cuda'))[0].cpu().numpy()
            torch.cuda.synchronize();latency=time.monotonic()-tick
        # GT is loaded only after the current-only prediction.
        label=torch.load(a.current_root/'ego'/(row['token']+'.pt'),map_location='cpu',weights_only=True)
        if label['identity']!=ego_identity['identity'] or label['token']!=row['token']:raise ValueError('Wrong A-calibration ego label')
        encoded=adapter.recode_historical_targets(label['ego'],codec);truth=decode_ego(encoded).numpy()
        original_xy=label['ego'][...,:2]*label['ego'].new_tensor(adapter.OLD_STD)+label['ego'].new_tensor(adapter.OLD_MEAN)
        np.testing.assert_allclose(truth[:,:2],original_xy.numpy(),rtol=1e-6,atol=2e-6)
        record=dict(**row,current_velocity_mps=(decode_ego(torch.from_numpy(observation['state']))[0,:2]/.5).tolist(),
            label_source_sha256=h.digest(a.current_root/'ego'/(row['token']+'.pt')))
        if model is not None:
            if prediction.shape!=(8,3) or not np.isfinite(prediction).all():raise ValueError('Invalid q0; retain full population')
            record.update(trajectory_errors(prediction,truth));record['inference_seconds']=latency
            record['q0']=prediction.tolist()
        records.append(record)
    h.atomic_json(a.output/f'rank_{a.rank:03d}.json',dict(identity=ident,rows=records,wall_seconds=time.monotonic()-start,
        parent_CPU_seconds=time.process_time()-cpu,GPU_process_hours=0. if a.preflight else (time.monotonic()-start)/3600,
        peak_allocated_bytes=0 if a.preflight else torch.cuda.max_memory_allocated(),resources=h.check_host()))
    files=[a.output/f'rank_{i:03d}.json' for i in range(a.shards)]
    if all(path.exists() for path in files):
        receipts=[h.read(path) for path in files];allrows=[row for receipt in receipts for row in receipt['rows']]
        expected=selected
        if (any(rec['identity']!=ident for rec in receipts) or len(allrows)!=len(expected) or
            {r['token'] for r in allrows}!={r['token'] for r in expected}):raise ValueError('Incomplete A-calibration population')
        summary=dict(identity=ident,scenes=len(allrows),logs=len({r['log'] for r in allrows}),preflight=a.preflight,
            engineering_only=bool(a.engineering_scenes),full_population=not(a.preflight or a.engineering_scenes),
            scope='Actual current/label contract preflight only' if a.preflight else
                'Native FP32-master checkpoint engineering check only; not a learning-curve point' if a.engineering_scenes else
                'A-internal ego fit; not PDMS or an unseen-B result',
            GPU_process_hours=sum(rec['GPU_process_hours'] for rec in receipts))
        if not a.preflight:
            summary['metrics']={key:float(np.mean([row[key] for row in allrows])) for key in ('ADE','FDE','yaw_MAE_rad','yaw_endpoint_rad')}
        h.atomic_json(a.output/'COMPLETE.json',summary);print(json.dumps(summary))


if __name__=='__main__':main()
