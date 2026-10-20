"""Export inference-only private heads with canonical Base references and source.

No training samples, simulator, effect teacher or optimizer are deployment
dependencies. The shared original VLA is referenced once, not duplicated.
"""
from pathlib import Path
from dataclasses import asdict
import argparse
import ast
import io
import json
import os
import subprocess
import sys
import tempfile
import zipfile
import torch

from .common import artifact,atomic_bytes,digest,read_json,verify_artifact,verify_identity,write_json,sha256


def source_inventory(root,seeds):
    """Transitive local Python sources only; no images/data/weights/secrets."""
    root=Path(root);pending=list(seeds);seen={}
    while pending:
        relative=pending.pop();path=root/relative
        if relative in seen or not path.is_file():continue
        seen[relative]=sha256(path)
        parts=list(Path(relative).with_suffix('').parts)
        for n in range(1,len(parts)):
            parent='/'.join(parts[:n])+'/__init__.py'
            if (root/parent).is_file():pending.append(parent)
        tree=ast.parse(path.read_text())
        for node in ast.walk(tree):
            candidates=[]
            if isinstance(node,ast.Import):candidates=[a.name for a in node.names]
            elif isinstance(node,ast.ImportFrom):
                base=parts[:-node.level] if node.level else []
                module=base+(node.module.split('.') if node.module else [])
                candidates=['.'.join(module)]+['.'.join(module+[a.name]) for a in node.names]
            for module in candidates:
                if module.split('.')[0] not in ('tools','starVLA'):continue
                for candidate in (module.replace('.','/')+'.py',module.replace('.','/')+'/__init__.py'):
                    if (root/candidate).is_file():pending.append(candidate)
    return dict(sorted(seen.items()))


def _refiner_config(value):
    from starVLA.model.modules.action_effect.effect_model import EffectModelConfig
    from starVLA.model.modules.action_effect.intervention_basis import BasisSpec
    from starVLA.model.modules.action_effect.refinement_policy import RefinerConfig
    value=dict(value);value['model']=EffectModelConfig(**value['model'])
    b=dict(value['basis']);b['windows']=tuple(tuple(x) for x in b['windows']);value['basis']=BasisSpec(**b)
    for k in ('coefficient_scales','active_coefficients'):value[k]=tuple(value[k])
    return RefinerConfig(**value)


def export(checkpoint,*,base_binding,output,route,device='cpu'):
    from tools.st_world_policy.base_binding import read_binding
    binding=read_binding(base_binding);output=Path(output)
    if output.exists():raise FileExistsError('Immutable export: choose a new output directory')
    if route=='B':
        from .evaluate_effect_flow import load_flow_policy
        model=load_flow_policy(checkpoint,base_binding=base_binding,
            expected_base_binding_hash=binding['binding_sha256'],device=device)
        identity=read_json(Path(checkpoint).parent/'IDENTITY.json')
        payload={'schema':'action_effect_private_deploy_v1','route':'B','arm':model.arm,
            **model.adapter.velocity_export()}
        head_receipt=identity['head']['head_receipt']
        required=['current_original_planner_condition','original_initial_noise']
    elif route=='A':
        from .evaluate_refiner import load_refiner
        model=load_refiner(checkpoint,expected_base_binding_hash=binding['binding_sha256']).to(device)
        payload={'schema':'action_effect_private_deploy_v1','route':'A','arm':model.provenance['arm'],
            'config':asdict(model.refiner.config),'constraint_spec':asdict(model.constraint_spec),
            'state_dict':model.refiner.state_dict()}
        head_receipt=None;required=['same_call_current_W_A_ego_context','original_Base_physical_trajectory']
    else:raise ValueError('Explicit independent route A or B required')
    payload['state_dict']={k:v.detach().cpu().clone() for k,v in payload['state_dict'].items()}
    output.mkdir(parents=True)
    buffer=io.BytesIO();torch.save(payload,buffer);atomic_bytes(output/'PRIVATE_HEAD.pt',buffer.getvalue())
    root=Path(__file__).resolve().parents[2]
    inventory=source_inventory(root,['tools/action_effect/export_models.py',
        'tools/action_effect/train_effect_flow.py','starVLA/model/framework/effect_guided_refiner.py'])
    z=io.BytesIO()
    with zipfile.ZipFile(z,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in inventory:archive.writestr(relative,(root/relative).read_bytes())
    atomic_bytes(output/'SOURCE.zip',z.getvalue())
    manifest={'schema':'action_effect_research_deployment_bundle_v1','route':route,'arm':payload['arm'],
        'private_head':artifact(output/'PRIVATE_HEAD.pt'),'source_package':artifact(output/'SOURCE.zip'),
        'source_inventory':inventory,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'base_binding':artifact(base_binding),'base_binding_hash':binding['binding_sha256'],
        'original_action_head_receipt':head_receipt,'training_provenance':model.provenance,
        'inference_inputs':required,'external_scorer_weights_loaded':False,'effect_teacher_required':False,
        'simulator_required':False,'training_cache_required':False,'GT_required':False,'LiDAR_required':False,
        'ACTIVE_updated':False,'online_VLA_integration':'requires original same-call S0 observation encoder; not measured by head-only export'}
    manifest['identity']=digest(manifest);write_json(output/'BUNDLE.json',manifest)
    return {'status':'COMPLETE','bundle':artifact(output/'BUNDLE.json'),
        'source_files':len(inventory),'private_parameters':sum(v.numel() for v in payload['state_dict'].values()),
        'scientific_gain':'UNTESTED by export','independent_process_roundtrip':'NOT_RUN'}


def load_bundle(path,*,device='cpu'):
    """No training checkpoint, dataset, feedback, GT or metric-cache reads."""
    from .evaluate_effect_flow import canonical_inference_backend
    canonical_inference_backend()
    from tools.st_world_policy.base_binding import read_binding
    manifest=verify_identity(read_json(path))
    if manifest.get('schema')!='action_effect_research_deployment_bundle_v1':raise ValueError('Wrong deployment schema')
    if any(manifest[k] is not False for k in ('effect_teacher_required','simulator_required','training_cache_required','GT_required','LiDAR_required')):
        raise ValueError('Deployment bundle contains forbidden runtime requirements')
    source_root=Path(__file__).resolve().parents[2]
    if not manifest.get('source_inventory'):raise ValueError('Deployment source contract is empty')
    for relative,want in manifest['source_inventory'].items():
        if Path(relative).is_absolute() or '..' in Path(relative).parts or sha256(source_root/relative)!=want:
            raise ValueError('Deployment source contract changed; use the exported source package')
    binding_path=verify_artifact(manifest['base_binding']);binding=read_binding(binding_path)
    if binding['binding_sha256']!=manifest['base_binding_hash']:raise ValueError('Deployment Base changed')
    payload=torch.load(verify_artifact(manifest['private_head']),map_location='cpu',weights_only=True)
    if (payload['schema']!='action_effect_private_deploy_v1' or payload['route']!=manifest['route']
            or payload['arm']!=manifest['arm']):raise ValueError('Private head route/identity mismatch')
    for tensor in payload['state_dict'].values():
        if not isinstance(tensor,torch.Tensor) or (tensor.is_floating_point() and
                (tensor.dtype!=torch.float32 or not torch.isfinite(tensor).all())):
            raise ValueError('Deployment private state must contain finite canonical FP32 tensors')
    if payload['route']=='B':
        from .train_effect_flow import load_original_action_head
        from starVLA.model.modules.action_effect.effect_flow_adapter import EffectFlowAdapter,EffectFlowAdapterConfig
        from starVLA.model.framework.effect_internalized_flow import EffectInternalizedFlow
        head,_=load_original_action_head(verify_artifact(manifest['original_action_head_receipt']),binding_path,device=device)
        adapter=EffectFlowAdapter(EffectFlowAdapterConfig(**payload['config']))
        adapter.load_state_dict(payload['state_dict'],strict=True)
        model=EffectInternalizedFlow(head,adapter,arm=payload['arm'])
    else:
        from starVLA.model.framework.effect_guided_refiner import EffectGuidedRefiner
        from starVLA.model.modules.action_effect.refinement_policy import CoefficientRefiner
        from starVLA.model.modules.action_effect.trajectory_constraints import ConstraintSpec
        refiner=CoefficientRefiner(_refiner_config(payload['config']));refiner.load_state_dict(payload['state_dict'],strict=True)
        model=EffectGuidedRefiner(refiner,constraint_spec=ConstraintSpec(**payload['constraint_spec']))
    model.to(device).requires_grad_(False).eval();model.provenance=manifest
    return model


def predict_bundle(model,data,route,device):
    with torch.no_grad(),torch.autocast(device_type=torch.device(device).type,enabled=False):
        if route=='B':
            if set(data)!={'condition','initial_noise'}:raise ValueError('Only current condition and noise enter IEG deployment')
            return model.predict_action(data['condition'].to(device),initial_noise=data['initial_noise'].to(device)).cpu()
        if set(data)!={'context','base_physical'} or set(data['context'])!={'memory','ego','mask'}:
            raise ValueError('Only current context and original physical Base enter EGR deployment')
        context={k:v.to(device) for k,v in data['context'].items()}
        return model.deploy(context,data['base_physical'].to(device))['trajectory'].cpu()


class DenyTrainingRuntime:
    def find_spec(self,fullname,path=None,target=None):
        denied=('navsim','tools.action_effect.simulate_interventions','tools.action_effect.train_effect_model',
                'tools.action_effect.train_refiner','tools.action_effect.evaluate_refiner','tools.action_effect.feedback_schema')
        if any(fullname==p or fullname.startswith(p+'.') for p in denied):
            raise ImportError('Deployment process denied training/simulator module '+fullname)


def roundtrip(bundle,*,inputs,output,device='cpu'):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    output=Path(output)
    if output.exists():raise FileExistsError('Keep prior deployment verification')
    manifest=verify_identity(read_json(bundle));data=torch.load(inputs,map_location='cpu',weights_only=True)
    reference=predict_bundle(load_bundle(bundle,device=device),data,manifest['route'],device)
    output.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix='effect-deploy-source-') as temporary:
        with zipfile.ZipFile(verify_artifact(manifest['source_package'])) as archive:
            for name in archive.namelist():
                if name.startswith('/') or '..' in Path(name).parts:raise ValueError('Unsafe source archive member')
            archive.extractall(temporary)
        for name,want in manifest['source_inventory'].items():
            if sha256(Path(temporary)/name)!=want:raise ValueError('Exported source integrity mismatch')
        cmd=[sys.executable,'-m','tools.action_effect.export_models','--worker','--bundle',str(Path(bundle).resolve()),
             '--inputs',str(Path(inputs).resolve()),'--output',str(output/'OUTPUT.pt'),'--device',device]
        with (output/'stdout.log').open('w') as stdout,(output/'stderr.log').open('w') as stderr:
            result=subprocess.run(cmd,cwd=temporary,stdout=stdout,stderr=stderr,timeout=600,
                env={**os.environ,'PYTHONPATH':temporary,'CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'2',
                     'OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
        if result.returncode:raise RuntimeError('Independent deployment process failed; preserve logs')
    actual=torch.load(output/'OUTPUT.pt',map_location='cpu',weights_only=True)
    if not torch.equal(reference,actual):raise ValueError('Exported same-input/same-noise output differs; no relaxed tolerance')
    receipt={'schema':'action_effect_deploy_roundtrip_v1','status':'PASS','bundle':artifact(bundle),
        'input':artifact(inputs),'output':artifact(output/'OUTPUT.pt'),'exact':True,'device':device,
        'forbidden_modules_denied':True,'image_to_action_integration':'NOT_RUN','scientific_gain':'UNTESTED'}
    write_json(output/'COMPLETE.json',receipt);return receipt


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('export','roundtrip'),default='export');p.add_argument('--checkpoint')
    p.add_argument('--route',choices=('A','B'));p.add_argument('--base-binding');p.add_argument('--output',required=True)
    p.add_argument('--bundle');p.add_argument('--inputs');p.add_argument('--device',default='cpu');p.add_argument('--dry-run',action='store_true')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args(argv)
    try:
        if a.worker:
            sys.meta_path.insert(0,DenyTrainingRuntime());torch.set_num_threads(2)
            torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            manifest=verify_identity(read_json(a.bundle));model=load_bundle(a.bundle,device=a.device)
            torch.save(predict_bundle(model,torch.load(a.inputs,map_location='cpu',weights_only=True),manifest['route'],a.device),a.output)
            return 0
        if a.phase=='export' and not all((a.checkpoint,a.route,a.base_binding)):raise ValueError('Explicit own checkpoint/route/Base binding required')
        if a.phase=='roundtrip' and not all((a.bundle,a.inputs)):raise ValueError('Roundtrip needs bundle and actual legal input tensors')
        if a.dry_run:print(json.dumps({'status':'NOT_RUN','phase':a.phase}));return 0
        result=(export(a.checkpoint,base_binding=a.base_binding,output=a.output,route=a.route,device=a.device)
                if a.phase=='export' else roundtrip(a.bundle,inputs=a.inputs,output=a.output,device=a.device))
        print(json.dumps(result));return 0
    except Exception as exc:
        print(json.dumps({'status':'FAIL','error_type':type(exc).__name__,'error':str(exc)}));return 1


if __name__=='__main__':raise SystemExit(main())
