"""One real current-image S0 forward, EGR/IEG deployment and bounded latency.

The launcher starts a fresh worker. Only original Base masters, inference-only
private bundles and current observations enter that worker. Optional cached
prediction comparison happens afterwards in the launcher, never in the policy.
"""
from pathlib import Path
import argparse
import json
import os
import resource
import re
import subprocess
import sys
import time
import traceback
import hashlib
import numpy as np
import torch

from .common import artifact,read_json,sha256,verify_artifact,verify_identity,write_json

_ACTIVE_RUNTIME_GUARDS=None


def safe_resource_path(path):
    """Only source/project paths, never file contents, locals or URL credentials."""
    value=os.fsdecode(path)
    if any(value.startswith(str(root)+os.sep) for root in (Path('/mnt/project'),Path(sys.prefix),Path(sys.base_prefix))):
        return re.sub(r'(?<=//)[^/@]+@','<redacted>@',value)
    return '<external-path-sha256:'+hashlib.sha256(value.encode()).hexdigest()+'>'


def diagnostic_stack():
    return [{'file':safe_resource_path(frame.filename),'line':frame.lineno,'function':frame.name}
            for frame in traceback.extract_stack(limit=9)[:-1]]


class DeploymentReadGuard:
    """Deny training-label/physical-feedback data opens in the worker only."""
    parts={'ego','metric_cache','metric_cache_v1','contexts','future','future_targets',
           'interaction_targets','feedback','native_feedback','lidar','teacher_cache'}
    names={'PAIR_DATASET.json','REPAIR_DATASET.json','TRAINING_MANIFEST.json','PREDICTIONS.json'}

    def __init__(self,code_roots=None):
        root=Path(__file__).resolve().parents[2]
        self.code_roots=tuple(Path(p).resolve() for p in (code_roots or
            (root/'starVLA',root/'tools',Path(sys.prefix)/'lib',Path(sys.base_prefix)/'lib')))
        self.blocked=[];self.permitted_code_cache_writes=[];self.current_images=set();self.current_image_opens=0

    def atomic_python_cache_write(self,path,args):
        # importlib._write_atomic creates NAME.cpython-310.pyc.<id(path)>.
        # This is an optional code write, not future-label data access. Restrict
        # the exception to a real source module under declared trusted roots.
        if path.parent.name!='__pycache__' or len(args)<3 or not isinstance(args[2],int):return False
        flags=args[2]
        if (flags&os.O_ACCMODE)!=os.O_WRONLY or not flags&os.O_CREAT:return False
        match=re.fullmatch(r'([A-Za-z_][A-Za-z0-9_]*)\.cpython-[0-9]+(?:\.opt-[12])?\.pyc\.[0-9]+',path.name)
        if match is None:return False
        source=(path.parent.parent/(match.group(1)+'.py')).resolve()
        return source.is_file() and any(source.is_relative_to(root) for root in self.code_roots)

    def __call__(self,event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
        path=Path(os.fsdecode(args[0]));parts={x.lower() for x in path.parts}
        if str(path) in self.current_images:self.current_image_opens+=1
        if self.atomic_python_cache_write(path,args):
            self.permitted_code_cache_writes.append({'path':safe_resource_path(str(path)),
                'reason':'trusted_importlib_atomic_python_bytecode_write','stack':diagnostic_stack()})
            return
        if path.suffix in ('.py','.pyc','.so'):return
        if (parts&self.parts or path.name in self.names or path.name.lower().startswith(('metric_cache','feedback_'))
            or any(x.startswith('future_') or x.endswith('_targets') for x in parts)):
            detail={'path':safe_resource_path(str(path)),'event':'open','reason':'supervision_or_feature_cache_data',
                    'stack':diagnostic_stack()}
            self.blocked.append(detail)
            raise PermissionError('Deployment process denied supervision/cache data: '+detail['path'])


class DeploymentImportGuard:
    denied=('navsim','tools.action_effect.simulate_interventions','tools.action_effect.train_effect_model',
        'tools.action_effect.train_refiner','tools.action_effect.evaluate_refiner',
        'tools.action_effect.context_data','tools.action_effect.evaluation_context_data')

    def __init__(self):self.blocked=[]

    def find_spec(self,fullname,path=None,target=None):
        if any(fullname==x or fullname.startswith(x+'.') for x in self.denied):
            self.blocked.append({'module':fullname,'stack':diagnostic_stack()})
            raise ImportError('Deployment process denied teacher/simulator/cache module: '+fullname)


def dependency_audit(output,guard,imports):
    loaded=[{'module':name,'source':safe_resource_path(getattr(module,'__file__','<namespace>') or '<namespace>')}
            for name,module in sys.modules.items()
            if any(name==x or name.startswith(x+'.') for x in imports.denied)]
    evidence={'schema':'action_effect_deployment_dependency_audit_v1',
        'status':'FAIL' if loaded or guard.blocked or imports.blocked else 'PASS',
        'blocked_data_opens':guard.blocked,'blocked_import_attempts':imports.blocked,
        'forbidden_loaded_modules':loaded,'permitted_atomic_code_cache_writes':guard.permitted_code_cache_writes,
        'logged_content':'safe paths, module names and filename/line/function stacks only; no locals/source lines/file contents',
        'scope':'dependency boundary only, not model quality or latency'}
    write_json(Path(output)/'DEPENDENCY_AUDIT.json',evidence,immutable=False)
    return evidence


def current_only(observation):
    """Explicit same-observation whitelist; token never changes fixed noise."""
    current={key:observation[key] for key in ('image','lang','state')}
    if len(current['image'])!=3 or not isinstance(current['lang'],str):
        raise ValueError('Original three-view/current instruction contract required')
    state=np.asarray(current['state'])
    if state.shape!=(1,4) or not np.isfinite(state).all():
        raise ValueError('Current ego must be original normalized history delta-pose [1,4], not velocity')
    return current


class OnlinePipeline:
    """Exactly one shared encoder owner and one original action-head object."""
    def __init__(self,base,*,refiner=None,flow=None):
        self.base=base;self.refiner=refiner;self.flow=flow
        if base.training or any(p.requires_grad for p in base.parameters()):raise ValueError('Frozen eval original S0 required')
        if refiner is not None and (refiner.training or getattr(refiner,'effect_teacher',None) is not None
                                   or getattr(refiner,'base_policy',None) is not None):
            raise ValueError('EGR deployment must contain only its private refiner')
        if flow is not None:
            if flow.base.action_head is not base.action_model:raise ValueError('IEG must reuse the exact original Base action head')
            if flow.training or flow.adapter.config.effect_dim!=0:raise ValueError('IEG deploy adapter must strip training-only auxiliary heads')
        self.counts={};self.handles=[]
        model=base.qwen_vl_interface.model.model
        modules={'visual':model.visual,'language':model.language_model,
                 'planner_projection':base.action_model.qwen_proj,'DiT':base.action_model.model}
        if refiner is not None:modules['refiner']=refiner.refiner
        if flow is not None:modules['velocity_adapter']=flow.adapter
        for name,module in modules.items():
            self.counts[name]=0
            def hook(*_,name=name):self.counts[name]+=1
            self.handles.append(module.register_forward_hook(hook))

    def close(self):
        for h in self.handles:h.remove()
        self.handles=[]

    @torch.no_grad()
    def run(self,mode,observation,initial_noise):
        if mode not in ('S0','A','B'):raise ValueError('Explicit S0/A/B mode required')
        if mode=='A' and self.refiner is None:raise ValueError('No private A bundle')
        if mode=='B' and self.flow is None:raise ValueError('No private B bundle')
        obs=current_only(observation);before=dict(self.counts)
        with torch.autocast(device_type=initial_noise.device.type,enabled=False):
            encoded=self.base.encode_current([obs])
            condition=self.base.build_planner_condition(encoded)
            context={'memory':torch.cat((encoded['W'],encoded['action_queries']),1).detach(),
                     'ego':torch.as_tensor(np.asarray([obs['state']]),device=condition.device,dtype=torch.float32),
                     'mask':torch.ones((1,encoded['W'].shape[1]+encoded['action_queries'].shape[1]),device=condition.device,dtype=torch.bool)}
            if mode=='B':
                # No original full rollout before this single Euler10 generation.
                raw=self.flow.predict_action(condition,initial_noise=initial_noise)
            else:raw=self.base.action_model.predict_action(condition,initial_noise=initial_noise)
            from starVLA.dataloader.foresight_dataset import decode_ego
            physical=decode_ego(raw.float())
            proposal=self.refiner.deploy(context,physical) if mode=='A' else None
            final=proposal['trajectory'] if proposal is not None else physical
        calls={key:self.counts[key]-before[key] for key in self.counts}
        expected={key:0 for key in calls};expected.update(visual=1,language=1,planner_projection=1,DiT=10)
        if mode=='A':expected['refiner']=1
        if mode=='B':expected['velocity_adapter']=10
        if calls!=expected:raise ValueError('Online deployment call contract differs: '+str(calls))
        return {'W':encoded['W'],'A':encoded['action_queries'],'condition':condition,
                'ego':context['ego'],'raw':raw,'physical_before_refiner':physical,'physical':final,'calls':calls,
                'complete_generations':1,'teacher_calls':0,'simulator_calls':0}


def frozen_output(result):
    keys=('W','A','condition','ego','raw','physical_before_refiner','physical')
    out={}
    for key in keys:
        value=result[key].detach().cpu().contiguous()
        if value.dtype!=torch.float32 or not torch.isfinite(value).all():raise ValueError('Noncanonical/nonfinite online result: '+key)
        out[key]=value
    return out


def compare_outputs(left,right):
    if left.keys()!=right.keys():return False
    return all(a.dtype==right[k].dtype and a.shape==right[k].shape and
        torch.equal(a.reshape(-1).view(torch.uint8),right[k].reshape(-1).view(torch.uint8)) for k,a in left.items())


def poisoned(observation):
    return {**observation,'token':'poisoned_scene_identity_not_noise_seed',
        'ego':torch.full((8,4),float('nan')),'future_gt':'must never be inspected',
        'GT':object(),'lidar':object(),'feedback':object(),'metric_cache':'/forbidden/metric_cache.pkl',
        'teacher_cache':'/forbidden/teacher_cache','future_actors':object()}


def latency_summary(values):
    if len(values)<8 or not all(np.isfinite(x) and x>0 for x in values):raise ValueError('At least eight finite measured samples required')
    return {'n':len(values),'mean_seconds':float(np.mean(values)),'p50_seconds':float(np.percentile(values,50)),
            'p95_seconds':float(np.percentile(values,95)),'measured_seconds':values}


def worker(args):
    global _ACTIVE_RUNTIME_GUARDS
    from .evaluate_effect_flow import canonical_inference_backend
    backend=canonical_inference_backend()
    if not args.device.startswith('cuda') or not torch.cuda.is_available():raise ValueError('Actual allocated CUDA device required')
    if torch.cuda.device_count()!=1:raise ValueError('Expose exactly one admitted CUDA device')
    torch.cuda.set_device(args.device);torch.set_num_threads(2)
    guard=DeploymentReadGuard();imports=DeploymentImportGuard();_ACTIVE_RUNTIME_GUARDS=(guard,imports)
    sys.addaudithook(guard);sys.meta_path.insert(0,imports)
    from tools.st_world_policy.base_binding import read_binding
    from tools.foresight.checkpoints import load_student,scene_noise
    from starVLA.dataloader.foresight_dataset import ForesightCurrentDataset
    from .export_models import load_bundle
    base_binding=read_binding(args.base_binding,verify_files=True)
    repo=Path(__file__).resolve().parents[2]
    for path,want in base_binding['source_contracts'].items():
        if sha256(repo/path)!=want:raise ValueError('Original S0 input/model contract changed: '+path)
    cohort=verify_identity(read_json(args.cohort))
    if cohort.get('schema')!='action_effect_observation_manifest_v1' or cohort.get('split')!='train' or cohort['base_binding_hash']!=base_binding['binding_sha256']:
        raise ValueError('Use the fixed real original-S0 TRAIN observation cohort')
    selected=[row for row in cohort['entries'] if row['scene_id']==args.scene_id]
    if len(selected)!=1:raise ValueError('Exactly one registered real scene must match --scene-id')
    row=selected[0]
    for pin in [row['current'],*row['images']]:verify_artifact(pin)
    dataset=ForesightCurrentDataset(cohort['data_root'])
    if dataset.identity!=base_binding['parent_identity']['data'] or dataset.index[row['dataset_index']]['token']!=args.scene_id:
        raise ValueError('Current-only observation lineage/index differs from bound original S0')
    current_path=Path(cohort['data_root'])/'current'/(args.scene_id+'.json')
    if verify_artifact(row['current']).resolve()!=current_path.resolve():raise ValueError('Pinned current record is not the actual current-only dataset input')
    current_record=read_json(current_path)
    if current_record['image_paths']!=[pin['path'] for pin in row['images']]:raise ValueError('Current image view identity/order differs from the pinned observation')
    guard.current_images=set(current_record['image_paths'])
    base=load_student(base_binding['run'],base_binding['tag'],base_binding['parent_identity'],device=args.device,precision='fp32',strip=True)
    base.requires_grad_(False).eval()
    if any(hasattr(base,name) for name in ('future_head','dino_head','interaction_head','spatiotemporal_head')):
        raise ValueError('Original deployment helper did not strip auxiliary prediction networks')
    a=load_bundle(args.a_bundle,device=args.device) if 'A' in args.modes else None
    b=load_bundle(args.b_bundle,device=args.device,original_action_head=base.action_model,
                  original_base_binding_hash=base_binding['binding_sha256']) if 'B' in args.modes else None
    for mode,model in [('A',a),('B',b)]:
        if model is not None and (model.provenance['route']!=mode or model.provenance['base_binding_hash']!=base_binding['binding_sha256']):
            raise ValueError('Private deployment bundle has another route/Base')
    pipeline=OnlinePipeline(base,refiner=a,flow=b);noise=scene_noise(args.scene_id,42,args.device)
    noise_before=noise.clone();results={};outputs={};started=time.perf_counter()
    try:
        for mode in args.modes:
            torch.cuda.reset_peak_memory_stats(args.device);samples=[];reference=None;calls=None
            for n in range(args.warmup+args.repeats):
                torch.cuda.synchronize(args.device);begin=time.perf_counter()
                image_opens_before=guard.current_image_opens
                obs=dataset[row['dataset_index']]
                result=pipeline.run(mode,obs,noise)
                # Include delivery of the final physical trajectory to CPU.
                final=result['physical'].detach().cpu()
                torch.cuda.synchronize(args.device);seconds=time.perf_counter()-begin
                if guard.current_image_opens-image_opens_before!=3:raise ValueError('Online path must really open the original three current images')
                snap=frozen_output(result) # Diagnostic copies excluded from timing.
                if reference is None:reference=snap;calls=result['calls']
                elif not compare_outputs(reference,snap):raise ValueError('Same current images/noise gave different online outputs')
                if n>=args.warmup:samples.append(seconds)
            peak_allocated=torch.cuda.max_memory_allocated(args.device)
            peak_reserved=torch.cuda.max_memory_reserved(args.device)
            poison=frozen_output(pipeline.run(mode,poisoned(obs),noise))
            if not compare_outputs(reference,poison):raise ValueError('Forbidden future/label data changed deployment outputs')
            if not torch.equal(noise_before,noise):raise ValueError('Original fixed scene noise was modified')
            outputs[mode]=reference
            results[mode]={**latency_summary(samples),'warmup':args.warmup,'hook_calls_per_prediction':calls,
                'real_current_image_opens_per_timed_prediction':3,
                'label_independence_exact':True,'same_input_same_noise_exact':True,
                'peak_cuda_allocated_bytes':peak_allocated,
                'peak_cuda_reserved_bytes':peak_reserved,
                'resident_policy_scope':'one shared S0 plus all requested small private heads',
                'raw':reference['raw'][0].tolist(),'physical':reference['physical'][0].tolist(),
                'base_physical':reference['physical_before_refiner'][0].tolist() if mode!='B' else None,
                'complete_generations':1,'teacher_calls':0,'simulator_calls':0,'VLM_calls':1}
            del result,snap,poison,reference,final
    finally:pipeline.close()
    dependencies=dependency_audit(args.output,guard,imports)
    if dependencies['status']!='PASS':
        raise ValueError('Forbidden training/simulator/cache dependency used; inspect DEPENDENCY_AUDIT.json safe paths/stacks')
    if 'S0' in outputs:
        for mode in ('A','B'):
            if mode in outputs:
                for key in ('W','A','condition','ego'):
                    if not compare_outputs({key:outputs['S0'][key]},{key:outputs[mode][key]}):
                        raise ValueError('Same-call original shared context changed between routes: '+key)
    result={'schema':'action_effect_online_deployment_v1','status':'PASS','scope':'single real TRAIN scene repeated; engineering/latency only',
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        'base_binding':artifact(args.base_binding),'base_binding_hash':base_binding['binding_sha256'],
        'cohort':artifact(args.cohort),'scene_id':args.scene_id,'source_log':row['source_log'],'observation_hash':row['observation_hash'],
        'current_input':{'current':row['current'],'images':row['images'],'state':'normalized history delta-pose, not velocity'},
        'bundles':{key:artifact(path) for key,path in [('A',args.a_bundle),('B',args.b_bundle)] if key in args.modes},
        'runtime':{**backend,'torch':torch.__version__,'cuda':torch.version.cuda,'device':args.device,
            'gpu':torch.cuda.get_device_name(args.device),'batch':1,'seed':42},'modes':results,
        'parameters':{'shared_original_S0':sum(p.numel() for p in base.parameters()),
            'original_action_head':sum(p.numel() for p in base.action_model.parameters()),
            'A_private':sum(p.numel() for p in a.refiner.parameters()) if a is not None else 0,
            'B_private_velocity':sum(p.numel() for p in b.adapter.parameters()) if b is not None else 0,
            'second_VLM':0,'second_action_head':0,'training_auxiliary_heads_deployed':0},
        'max_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'timing_definition':'current JSON + original three-image decode/crop/resize + original processor/VLM + Euler10 + optional A refiner + CPU physical output; loading weights, diagnostics and cached-output comparison excluded',
        'file_cache_state':'repeated same scene; OS/file caches warm, not population throughput or cold-storage benchmark',
        'elapsed_seconds':time.perf_counter()-started,'scientific_gain':'UNTESTED','blocked_data_opens':guard.blocked,
        'dependency_audit':artifact(Path(args.output)/'DEPENDENCY_AUDIT.json'),
        'blocked_imports_verified':True,'source_policy':'same frozen repository/source contract, independent fresh Python worker'}
    write_json(Path(args.output)/'WORKER_RESULT.json',result)
    return result


def compare_prediction(pin,mode,result,worker_result):
    expected=artifact(pin);m=verify_identity(read_json(pin));reg=verify_identity(read_json(verify_artifact(m['registration'])))
    if (m.get('schema')!='action_effect_refiner_predictions_v1' or m.get('status')!='COMPLETE'
        or reg.get('schema')!='action_effect_refiner_prediction_registration_v1'
        or m.get('registration_identity')!=reg['identity'] or reg.get('base_checkpoint') is None
        or m.get('requested_scenes')!=len(m['entries'])
        or [r['scene_id'] for r in m['entries']]!=reg['scene_ids']):
        raise ValueError('Complete identity-bound explicit prediction artifact required')
    for row in m['entries']:verify_identity(row)
    rows=[r for r in m['entries'] if r['scene_id']==worker_result['scene_id']]
    if len(rows)!=1 or rows[0].get('observation_hash')!=worker_result['observation_hash']:
        raise ValueError('Explicit cached prediction scene/observation mismatch')
    from tools.st_world_policy.base_binding import read_binding
    binding=read_binding(verify_artifact(worker_result['base_binding']))
    if reg['base_checkpoint']!=binding['checkpoint']:raise ValueError('Cached prediction comes from a different Base')
    if mode!='S0':
        bundle=verify_identity(read_json(verify_artifact(worker_result['bundles'][mode])))
        if reg.get('checkpoint')!=bundle['training_provenance']['checkpoint'] or reg.get('arm')!=bundle['arm']:
            raise ValueError('Cached prediction uses a different private head')
    elif reg.get('arm')!='S0':raise ValueError('S0 comparison must explicitly be original Base predictions')
    cached=torch.tensor(rows[0]['physical'],dtype=torch.float32)
    online=torch.tensor(result['physical'],dtype=torch.float32)
    exact=compare_outputs({'physical':online},{'physical':cached})
    return {'prediction_artifact':expected,'exact_physical':exact,'max_abs_error':float((online-cached).abs().max())}


def launch(args):
    out=Path(args.output)
    if out.exists():raise FileExistsError('Keep prior deployment evidence; use a new output directory')
    if args.dry_run:return {'status':'NOT_RUN','modes':args.modes,'warmup':args.warmup,'measured_repeats':args.repeats,'jobs_launched':0}
    out.mkdir(parents=True)
    repo=Path(__file__).resolve().parents[2]
    cmd=[sys.executable,'-m','tools.action_effect.qualify_online_deployment','--worker','--base-binding',str(Path(args.base_binding).resolve()),
        '--cohort',str(Path(args.cohort).resolve()),'--scene-id',args.scene_id,'--output',str(out.resolve()),'--device',args.device,
        '--warmup',str(args.warmup),'--repeats',str(args.repeats),'--modes',*args.modes]
    for mode,path in [('a',args.a_bundle),('b',args.b_bundle)]:
        if path:cmd.extend(['--'+mode+'-bundle',str(Path(path).resolve())])
    write_json(out/'COMMAND.json',{'command':cmd,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        'max_seconds':args.max_seconds,'scientific_gain':'UNTESTED'})
    with (out/'stdout.log').open('w') as stdout,(out/'stderr.log').open('w') as stderr:
        child=subprocess.run(cmd,cwd=repo,stdout=stdout,stderr=stderr,timeout=args.max_seconds,
            env={**os.environ,'CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
    if child.returncode:raise RuntimeError('Independent online deployment child failed; stdout/stderr preserved')
    result=read_json(out/'WORKER_RESULT.json');comparisons={}
    for mode,pin in [('S0',args.s0_predictions),('A',args.a_predictions),('B',args.b_predictions)]:
        if pin:
            if mode not in result['modes']:raise ValueError('Cannot compare an unexecuted mode')
            comparisons[mode]=compare_prediction(pin,mode,result['modes'][mode],result)
    exact=all(x['exact_physical'] for x in comparisons.values())
    complete={'schema':'action_effect_online_deployment_receipt_v1','status':'PASS' if exact else 'FAIL',
        'worker_result':artifact(out/'WORKER_RESULT.json'),'command':artifact(out/'COMMAND.json'),
        'stdout':artifact(out/'stdout.log'),'stderr':artifact(out/'stderr.log'),'cached_prediction_comparisons':comparisons,
        'cached_comparison_status':'NOT_REQUESTED' if not comparisons else 'PASS' if exact else 'FAIL',
        'scientific_gain':'UNTESTED: engineering integration and one-scene latency only'}
    write_json(out/'COMPLETE.json' if exact else out/'FAILED.json',complete)
    return complete


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-binding',required=True);p.add_argument('--cohort',required=True);p.add_argument('--scene-id',required=True)
    p.add_argument('--a-bundle');p.add_argument('--b-bundle');p.add_argument('--output',required=True);p.add_argument('--device',required=True)
    p.add_argument('--modes',choices=('S0','A','B'),nargs='+',default=['S0','A','B'])
    p.add_argument('--warmup',type=int,default=2);p.add_argument('--repeats',type=int,default=8);p.add_argument('--max-seconds',type=float,default=1800.)
    p.add_argument('--s0-predictions');p.add_argument('--a-predictions');p.add_argument('--b-predictions')
    p.add_argument('--dry-run',action='store_true');p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args(argv)
    try:
        if len(set(a.modes))!=len(a.modes) or not 2<=a.warmup<=32 or not 8<=a.repeats<=128 or not 1<=a.max_seconds<=7200:
            raise ValueError('Unique modes, 2..32 warmup, 8..128 measured repetitions and explicit bounded wall time required')
        if ('A' in a.modes and not a.a_bundle) or ('B' in a.modes and not a.b_bundle):raise ValueError('Requested route requires its actual small deployment bundle')
        result=worker(a) if a.worker and not a.dry_run else launch(a)
        print(json.dumps(result));return 0 if result['status'] in ('PASS','NOT_RUN') else 1
    except Exception as e:
        if a.worker and _ACTIVE_RUNTIME_GUARDS is not None:
            try:dependency_audit(a.output,*_ACTIVE_RUNTIME_GUARDS)
            except Exception:pass # Preserve the first error; normal worker outputs are outside prohibited data roots.
        print(json.dumps({'status':'FAIL','error_type':type(e).__name__,'error':str(e),'scientific_gain':'UNTESTED'}));return 1


if __name__=='__main__':raise SystemExit(main())
