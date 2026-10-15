"""Real current images to one trajectory; no label or feature-cache argument."""
import argparse
import importlib.util
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('dl_online_helpers',Path(__file__).with_name('common.py'))
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('anchor','refiner','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--predictor',type=Path);p.add_argument('--index',type=Path)
    p.add_argument('--audit-token-quantization', action='store_true',
        help='Extra current-image diagnostic forward with FP32 readout tokens; never replaces the registered executed trajectory')
    p.add_argument('--split',choices=('train','dev','navtest'),required=True)
    p.add_argument('--max-seconds',type=float,required=True);a=p.parse_args();started=time.monotonic();resource=helpers.check_host()
    if a.split=='navtest' and not (a.output.parent/'FINAL_METHOD_LOCK.json').exists():raise ValueError('Lock endpoints before final Navtest')
    anchor=helpers.read(a.anchor)
    # Fail closed on actual Python-level training-data access, including the
    # existing raw log/box source. Current image files remain permitted. This
    # is an execution check, not a claim of OS-level filesystem sandboxing.
    forbidden=[]
    for record in anchor['data'].values():
        if 'label_root' in record:forbidden.append(str(Path(record['label_root']).resolve())+'/')
        forbidden.append(str(Path(record['current_root']).resolve()/'ego')+'/')
    forbidden.append('/mnt/navsim/trainval_navsim_logs/')
    forbidden.append('/mnt/navsim/test_navsim_logs/')
    forbidden.append('/mnt/navsim/maps/')
    def reject_training_open(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes)):return
        path=str(Path(args[0].decode() if isinstance(args[0],bytes) else args[0]).absolute())
        if any(path.startswith(root) for root in forbidden) or '/LIDAR_PC/' in path:
            raise PermissionError('Camera-only deployment attempted to open training annotations: '+path)
        if str(a.anchor.parent)+'/' in path and any(part.startswith(('labels_','anchor_train_','anchor_dev_','anchor_diagnostic','anchor_profile')) for part in Path(path).parts):
            raise PermissionError('Online deployment cannot open a compact training label/feature cache')
        if '/minimal-repair-artifacts/' in path and any(part.startswith(('targets_','target_review','labels_')) for part in Path(path).parts):
            raise PermissionError('Online repair cannot read offline solver/target/geometry artifacts')
        if '/natural-error-local-artifacts/' in path and any(part.startswith(('engineering_local_','local_labels_','natural_targets_','anchor_cache_')) for part in Path(path).parts):
            raise PermissionError('Online natural-error policy cannot read frozen features or label evidence')
    sys.addaudithook(reject_training_open)
    spec=importlib.util.spec_from_file_location('dl_bootstrap',Path(__file__).with_name('verify_current_only.py'))
    bootstrap=importlib.util.module_from_spec(spec);spec.loader.exec_module(bootstrap);runtime=bootstrap.load_runtime(anchor)
    import numpy as np
    import torch
    from tools.foresight.checkpoints import checkpoint_identity,load_student
    from starVLA.dataloader.foresight_dataset import ForesightCurrentDataset
    from starVLA.model.modules.decision_local.consequence_head import ConsequenceQueryHead
    from starVLA.model.modules.decision_local.residual_refiner import ResidualRefiner
    from starVLA.model.modules.decision_local.trajectory_queries import QueryConfig
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rr=torch.load(a.refiner,map_location='cpu',weights_only=False);kind=rr['identity']['kind']
    natural=rr['identity'].get('protocol')=='natural_local_v1'
    repair=rr['identity'].get('protocol') in ('minimal_repair_v1','natural_local_v1')
    allowed=(('N_NONE','R_CURRENT','R_ABS_FUTURE','R_DIFF_FUTURE') if natural else
        (('LOG_NONE','MIN_NONE','R_MIN_CURRENT','R_MIN_ABS','R_MIN_DIFF') if repair else
         ('E_NONE','R_CURRENT_PRED','R_FUTURE_ABS','R_FUTURE_DIFF')))
    if kind not in allowed:raise ValueError('Oracle probes are not deployable')
    if rr['identity']['anchor_identity']!=anchor['identity']:raise ValueError('Wrong base policy')
    predictor=None
    if kind.startswith('R_'):
        if a.predictor is None or helpers.digest(a.predictor)!=rr['identity']['predictor_sha256']:raise ValueError('Use the predictor seen during refiner training')
        pp=torch.load(a.predictor,map_location='cpu',weights_only=False)
        expected='P_DIFF' if kind in ('R_FUTURE_DIFF','R_MIN_DIFF','R_DIFF_FUTURE') else 'P_ABS'
        if pp['identity']['kind']!=expected or pp['identity']['anchor_identity']!=anchor['identity'] or pp['identity']['query_config']!=rr['identity']['query_config']:raise ValueError('Predictor contract mismatch')
        if natural and (pp['identity'].get('protocol')!='natural_local_v1' or pp['identity']['repair_label_semantics']!=rr['identity']['repair_label_semantics']):
            raise ValueError('Local predictor and refiner contracts differ')
        predictor=ConsequenceQueryHead(dim=pp['identity']['dim']).cuda();predictor.load_state_dict(pp['model'],strict=True)
    elif a.predictor is not None:raise ValueError('E_NONE has no predictor')
    if repair:
        from starVLA.model.modules.decision_local.minimal_repair_contract import MinimalRepairRefiner
        from starVLA.model.framework.frozen_vla_minimal_repair import FrozenVLAMinimalRepairRefiner
        refiner=MinimalRepairRefiner(dim=rr['identity']['dim']).cuda()
    else:refiner=ResidualRefiner(dim=rr['identity']['dim']).cuda()
    refiner.load_state_dict(rr['model'],strict=True)
    training,checked=checkpoint_identity(anchor['run'],anchor['tag'])
    if checked['sha256']!=anchor['checkpoint']['sha256']:raise ValueError('S0 checkpoint changed')
    base=load_student(anchor['run'],anchor['tag'],training,precision='fp32',strip=True).requires_grad_(False).eval()
    policy_class=FrozenVLAMinimalRepairRefiner if repair else runtime.FrozenVLAConsequenceRefiner
    policy=policy_class(base,refiner,predictor,mode=kind,query_config=QueryConfig(**rr['identity']['query_config'])).eval()
    # The pinned S0 invokes Qwen's visual and language submodules directly,
    # bypassing the outer conditional-generation forward. Hook the executed
    # modules instead of a wrapper that is never called on this native path.
    calls={'qwen_visual':0,'qwen_language':0,'proposal':0}
    def count_visual(module, args):calls['qwen_visual']+=1
    def count_language(module, args):calls['qwen_language']+=1
    native_qwen=base.qwen_vl_interface.model.model
    qwen_hooks=[native_qwen.visual.register_forward_pre_hook(count_visual),
                native_qwen.language_model.register_forward_pre_hook(count_language)]
    expected_calls={'qwen_visual':1,'qwen_language':1,'proposal':1}
    original_proposal=base.action_model.predict_action
    def counted_proposal(*args,**kwargs):
        calls['proposal']+=1
        return original_proposal(*args,**kwargs)
    base.action_model.predict_action=counted_proposal
    data=ForesightCurrentDataset(anchor['data'][a.split]['current_root'])
    if data.identity!=anchor['data'][a.split]['identity']:raise ValueError('Current population changed')
    lookup={row['token']:i for i,row in enumerate(data.index)}
    selected=helpers.read(a.index) if a.index else data.index
    if len({r['token'] for r in selected})!=len(selected) or any(r['token'] not in lookup for r in selected):raise ValueError('Invalid registered current subset')
    contract={'kind':kind,'mode':'camera_only','execution':'online current images; one frozen S0 and one residual',
        'source_sha':helpers.source_sha(ROOT),'reader_sha256':helpers.digest(__file__),
        'anchor_identity':anchor['identity'],'current_identity':data.identity['identity'],
        'head_checkpoint_sha256':helpers.digest(a.refiner),'predictor_sha256':helpers.digest(a.predictor) if a.predictor else None,
        'index':selected,'sampling_seed':42,'precision':'FP32 base/q0, FP16 token roundtrip, FP32 heads; TF32off',
        'queries':'seven functions, no candidate scoring/selection','main_trajectory':'q_final'}
    ident=helpers.identity(contract)
    if a.output.exists():raise FileExistsError('Preserve existing online outputs')
    a.output.mkdir(parents=True)
    for k in ('predictions','records'):(a.output/k).mkdir()
    helpers.atomic_json(a.output/'identity.json',{'identity':ident,**contract})
    latencies=[];execution_checks=[];quantization=[]
    for row in selected:
        if time.monotonic()-started>=a.max_seconds:raise TimeoutError('Bounded online deployment verification')
        obs=data[lookup[row['token']]];previous=dict(calls)
        torch.cuda.synchronize();before=time.monotonic();out=policy.predict(obs,sampling_seed=42);torch.cuda.synchronize()
        latencies.append(time.monotonic()-before)
        actual={key:calls[key]-previous[key] for key in calls}
        if actual!=expected_calls:raise AssertionError('Expected one native visual/language pass and one proposal sampler, got '+repr(actual))
        execution_checks.append({'token':row['token'],**actual})
        if a.audit_token_quantization:
            # This is a separately counted diagnostic, not a second model or
            # candidate in the executed policy, and it cannot choose q_final.
            old_dtype=policy.feature_dtype;previous=dict(calls)
            try:
                policy.feature_dtype=torch.float32
                unquantized=policy.predict(obs,sampling_seed=42)
            finally:policy.feature_dtype=old_dtype
            if {key:calls[key]-previous[key] for key in calls}!=expected_calls:
                raise AssertionError('Unexpected extra calls in the precision diagnostic')
            torch.testing.assert_close(out['q0'],unquantized['q0'],rtol=0,atol=0)
            item={'token':row['token'],'extra_diagnostic_Qwen_forwards':1,'q0_exact':True,
                'final_xy_max_abs_m':float((out['q_final'][...,:2]-unquantized['q_final'][...,:2]).abs().max())}
            difference=out['q_final'][...,2]-unquantized['q_final'][...,2]
            item['final_heading_max_abs_rad']=float(torch.atan2(difference.sin(),difference.cos()).abs().max())
            for key in ('W','H_A'):
                full=getattr(unquantized['inputs'],key);stored=getattr(out['inputs'],key)
                item[key+'_relative_L2']=float((stored-full).norm()/full.norm().clamp_min(1e-12))
            quantization.append(item)
        dest=a.output/'predictions'/(row['token']+'.npz');temp=dest.with_suffix('.tmp')
        with temp.open('wb') as stream:np.savez_compressed(stream,q0=out['q0'][0].cpu().numpy(),q_final=out['q_final'][0].cpu().numpy())
        temp.replace(dest);helpers.atomic_json(a.output/'records'/(row['token']+'.json'),{'identity':ident,'token':row['token'],'sha256':helpers.digest(dest)})
    for hook in qwen_hooks:hook.remove()
    base.action_model.predict_action=original_proposal
    helpers.atomic_json(a.output/'COMPLETE.json',{'identity':ident,'scenes':len(selected),'GPU_hours':(time.monotonic()-started)/3600.,
        'seconds':time.monotonic()-started,'current_image_read_excluded_from_latency':True,'latency_seconds':latencies,
        'median_latency_excluding_first':float(np.median(latencies[1:] or latencies)),
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'resources':resource,
        'opened_GT_relation_or_feature_cache':False,'python_training_file_access_guard':True,
        'guard_is_not_an_OS_sandbox':True,'q0_and_final_exported':True,
        'actual_module_forward_checks':execution_checks,'token_quantization_diagnostic':quantization,
        'precision_diagnostic_never_used_to_select_executed_trajectory':True})


if __name__=='__main__':main()
