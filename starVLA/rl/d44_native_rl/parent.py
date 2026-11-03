"""One immutable native D44 parent, including IL-trained R16/readouts."""
import copy
import json
import os
from pathlib import Path
import torch
from starVLA.rl.qwen_world_compare.loading import inspect_IL_checkpoint, model_from_identity, strict_tensor_load
from starVLA.rl.qwen_world_compare.contracts import file_sha, fingerprint

PARENT=Path('/mnt/project/qwen-world-four-frameworks-20261029/runs/ARCH_D/IL_seed42/checkpoints/update044000')
ROOT=Path('/mnt/project/d44-native-rl-20261103')
D46_ROOT=Path('/mnt/project/d46-priority-rl-20261103')
D46_SCHEMA='d46_verified_deployment_parent_v1'
D46_MODEL_SHA='386a0a27bd197cb002501bcb90c343659001d2e6d4f7bad0cbd5e31a91f8477d'
D46_EXPORT_SHA='6ffe4b20e47f52b2fae1266a5c525c7abf714d43a26b8d38372b946263d7decf'
D46_NATIVE_SHA='815ff2ca72293ca6b81fe873141c3a175a747dc067eddb1971944a6d80ed752b'
_D46_VERIFIED={}


def _d46_root():
    root=Path(os.environ.get('WORLD_COMPARE_ROOT',str(D46_ROOT))).resolve()
    if root!=D46_ROOT.resolve():raise ValueError('D46 policy requires its independently authorized campaign root')
    return root


def registration(binding=None):
    """A new D46 parent is opt-in; no native44 default or seal is rewritten."""
    root=_d46_root();name='PARENT_REGISTRATION.json'if binding is None else Path(binding.get('path','')).name
    versions={'PARENT_REGISTRATION.json':'PARENT_CONTRACT.json',
        'PARENT_REGISTRATION_v1.json':'PARENT_CONTRACT_v1.json',
        'PARENT_REGISTRATION_v2.json':'PARENT_CONTRACT_v2.json'}
    if name not in versions:raise ValueError('Unregistered D46 parent-contract version')
    path=root/name;contract=root/versions[name]
    raw=path.read_bytes()
    if raw!=contract.read_bytes():raise ValueError('D46 contract/registration differ')
    if binding is not None and binding!={'path':str(path),'file_sha256':file_sha(path)}:
        raise ValueError('Recipe is not bound to the exact sealed D46 registration')
    value=json.loads(raw);body=dict(value);seal=body.pop('sha256',None)
    if (seal!=fingerprint(body)or value.get('schema')!=D46_SCHEMA
            or value.get('parent_id')!='D_IL46K'or value.get('parent_kind')!='verified_deployment_export'
            or value.get('architecture_id')!='D'or value.get('IL_updates')!=46000
            or value.get('uses_rl')is not False
            or value.get('readouts_true_allowed')is not False):raise ValueError('Wrong D46 policy-parent contract')
    revision=value.get('contract_revision',1)
    allowed={'D0','D2','D5'}if revision==1 else {'D0','D1','D2','D3_FPO','D3_AWM','D5'}if revision==2 else None
    if (allowed is None or len(value.get('allowed_arms',[]))!=len(allowed)or set(value['allowed_arms'])!=allowed
            or(name.endswith('_v1.json')and revision!=1)or(name.endswith('_v2.json')and revision!=2)):
        raise ValueError('D46 parent contract has an unregistered policy-only route set')
    export=value['export'];ancestry=value['ancestry']
    if (Path(export['path']).resolve()!=root/'IL46_PARENT'or export['model_file']!='model.pt'
            or export['model_sha256']!=D46_MODEL_SHA or export['export_json_sha256']!=D46_EXPORT_SHA
            or ancestry['native_model_sha256']!=D46_NATIVE_SHA or ancestry['full_native_available']is not False):
        raise ValueError('D46 deployment/native ancestry changed')
    proof=value['parity'];nav=value['complete_navtest']
    if file_sha(proof['path'])!=proof['file_sha256']or file_sha(nav['summary'])!=nav['summary_file_sha256']:
        raise ValueError('D46 original parity/complete evaluation proof changed')
    parity=json.loads(Path(proof['path']).read_text());summary=json.loads(Path(nav['summary']).read_text());summary_body=dict(summary);summary_seal=summary_body.pop('sha256',None)
    if (parity.get('status')!='PASS'or parity.get('architecture_id')!='D'
            or parity.get('model_sha256')!=D46_MODEL_SHA or parity.get('native_model_sha256')!=D46_NATIVE_SHA
            or parity.get('normalized_max_abs')!=0 or parity.get('requested_max_abs')!=0
            or summary_seal!=fingerprint(summary_body)or summary.get('status')!='COMPLETE'
            or summary.get('scenes')!=12146 or summary.get('physical_logs')!=44 or summary.get('invalid_scenes')!=0
            or summary.get('qualification')is not False or summary.get('seed')!=2101
            or summary.get('model_sha256')!=D46_MODEL_SHA or summary.get('PDMS')!=nav['PDMS']
            or summary.get('protocol_sha256')!=nav['protocol_sha256']):
        raise ValueError('D46 parent lacks matching actual full-score/parity evidence')
    return value


def _d46_meta(path,value):
    from starVLA.rl.qwen_world_compare import deployment as deployment
    from starVLA.rl.qwen_world_compare.loading import contained_file
    path=Path(path).resolve();e=value['export']
    if path!=Path(e['path']).resolve():raise ValueError('Not the registered D46 export')
    meta=json.loads((path/'export.json').read_text());complete=json.loads((path/'COMPLETE').read_text())
    if (file_sha(path/'export.json')!=e['export_json_sha256']or file_sha(path/'COMPLETE')!=e['COMPLETE_file_sha256']
            or complete.get('export_sha256')!=e['export_json_sha256']or meta.get('schema')!=deployment.EXPORT_SCHEMA
            or meta.get('architecture_id')!='D'or meta.get('IL_updates')!=46000 or meta.get('qualification')is not False
            or meta.get('RL_parent')is not None or meta.get('rl_lora')is not None or meta.get('LoRA_merge')is not False
            or meta.get('prediction_type')!='x0'or meta.get('action_shape')!=[8,3]
            or meta.get('deployment_inference')!='single_candidate_cleanx_K10_shift5'
            or meta.get('no_teacher_future_candidate_metric_or_train_heads')is not True
            or meta.get('model_sha256')!=D46_MODEL_SHA or contained_file(path,'model.pt').stat().st_size!=e['model_bytes']):
        raise ValueError('D46 immutable policy-export contract changed')
    old=meta['IL_parent'];ancestry=value['ancestry']
    if (old['model_sha256']!=D46_NATIVE_SHA or old['seal']['updates']!=46000
            or old['seal']['files_sha256']!=ancestry['native_files_sha256']
            or old['seal']['identity_sha256']!=ancestry['IL_training_identity_sha256']
            or old['identity']['architecture_id']!='D'or old['identity'].get('uses_rl')is not False):
        raise ValueError('Export does not descend from the verified D46 IL checkpoint')
    if meta['processor']!=e['processor']or meta['runtime_source_files']!=value['runtime']['source_files']or meta['runtime_libraries']!=value['runtime']['libraries']:
        raise ValueError('D46 processor/runtime inventory changed')
    for name,want in e['processor'].items():
        if file_sha(contained_file(path/'processor',name))!=want:raise ValueError('D46 processor file changed')
    # Reuse the already reviewed exact execution migration: old default-True
    # checkpoint semantics, never accept arbitrary source or library drift.
    from starVLA.rl.qwen_world_abcd_research.parents import ADAPTER,ADAPTER_MIGRATION
    expected=meta['runtime_source_files'];actual=deployment._source_runtime();added=set(actual)-set(expected)
    helper='starVLA/model/modules/qwen_world_compare/execution.py';helper_sha='4f1cef430829f29b3c2938486c0bbc39782812be17255dbb784519d789270f22'
    changes={name:(expected[name],actual.get(name))for name in expected if expected[name]!=actual.get(name)}
    override=value['source_compatibility']['exact_override']
    if (override!={'path':ADAPTER,'scored_sha256':ADAPTER_MIGRATION[0],'allowed_execution_sha256':ADAPTER_MIGRATION[1]}
            or set(expected)-set(actual)or added not in(set(),{helper})or(added and actual[helper]!=helper_sha)
            or changes not in({}, {ADAPTER:ADAPTER_MIGRATION})or deployment._library_runtime()!=meta['runtime_libraries']):
        raise ValueError('Unregistered D46 execution-source/library change')
    return meta,dict(differences={k:list(v)for k,v in changes.items()},added_execution_helper={k:actual[k]for k in added},checkpoint_short_suffix=True,original_export_seals_unchanged=True)


def audit_export(path,*,verify=True,binding=None):
    value=registration(binding);meta,migration=_d46_meta(path,value);path=Path(path).resolve()
    if verify:
        stat=(path/'model.pt').stat();key=(str(path),stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)
        if key not in _D46_VERIFIED:
            # Full canonical export verification includes the actual model
            # bytes and every processor; source comparison is narrowed above.
            from starVLA.rl.qwen_world_abcd_research.parents import inspect_parent_export
            checked=inspect_parent_export(path)
            if checked['model_sha256']!=D46_MODEL_SHA:raise ValueError('Changed D46 model bytes')
            _D46_VERIFIED[key]=True
    original=meta['IL_parent']
    info=dict(schema='d46_verified_export_policy_parent_info_v1',parent_kind='verified_deployment_export',
        path=str(path),model_sha256=D46_MODEL_SHA,identity=copy.deepcopy(original['identity']),
        seal=dict(kind='deployment_export_COMPLETE',updates=46000,export_sha256=D46_EXPORT_SHA,model_sha256=D46_MODEL_SHA),
        files={'model.pt':D46_MODEL_SHA,'export.json':D46_EXPORT_SHA,'COMPLETE':value['export']['COMPLETE_file_sha256']},
        processor=copy.deepcopy(meta['processor']),
        export_parent=dict(path=str(path),model_sha256=D46_MODEL_SHA,export_sha256=D46_EXPORT_SHA,source_migration=migration),
        native_ancestry=copy.deepcopy(value['ancestry']),native_recovery_available=False,
        training_readouts_available=False,R16_query_available=False,
        parent_registration=dict(path=str(_d46_root()/'PARENT_REGISTRATION.json')if binding is None else binding['path'],
            file_sha256=file_sha(_d46_root()/'PARENT_REGISTRATION.json')if binding is None else binding['file_sha256'],seal=value['sha256']))
    return json.loads(json.dumps(info))


def audit(path=PARENT, *, verify=True):
    if (Path(path)/'export.json').is_file():return audit_export(path,verify=verify)
    info=inspect_IL_checkpoint(path,verify_files='model'if verify else'inventory')
    if info['identity']['architecture_id']!='D' or info['seal']['updates']!=44000 or info['identity']['qualification']:
        raise ValueError('Only the actual formal D IL44000 parent is allowed')
    cfg=info['identity']['config']['deployment']
    if cfg['action_shape']!=[8,3] or cfg['steps']!=10 or cfg['shift']!=5 or cfg['seed']!=2101:
        raise ValueError('D44 deployment identity changed')
    return info


def pin():
    info=audit();dest=ROOT/'IL44_PARENT';dest.mkdir(parents=True,exist_ok=True)
    for name in list(info['files'])+['COMPLETE','identity.json','files.json']:
        src=PARENT/name;out=dest/name;out.parent.mkdir(parents=True,exist_ok=True)
        if not out.exists():os.link(src,out)
        if not os.path.samefile(src,out):raise ValueError('Parent pin differs: '+name)
    scored=Path('/mnt/project/qwen-world-four-frameworks-20261029/periodic_NAVTEST_ARCH_D_044000_20261102T142147Z/D_IL_044000/export/export.json')
    pinned=dest/'scored_export_identity.json'
    if not pinned.exists():os.link(scored,pinned)
    if file_sha(pinned)!=file_sha(scored):raise ValueError('Scored export metadata pin differs')
    return audit(dest,verify=False)


def load(path,device,*,readouts=False,binding=None):
    if (Path(path)/'export.json').is_file():
        if readouts:raise ValueError('D46 deployment parent has no IL R16/readout weights; D4/D6 are not admitted')
        info=audit_export(path,verify=True,binding=binding)
        # Same canonical load_export topology and strict tensor load, with the
        # exact reviewed source migration above; no randomly recreated heads.
        from starVLA.rl.qwen_world_compare.deployment import prune_deployment
        model=model_from_identity(info['identity'],train_heads=True)
        model=prune_deployment(model,copy_model=False)
        state=torch.load(Path(path)/'model.pt',map_location='cpu',weights_only=True,mmap=True)
        strict_tensor_load(model,state)
        if model.train_heads or hasattr(model,'readouts')or model.decision.r_embeddings is not None:
            raise ValueError('D46 deployed policy must not reconstruct unavailable training-only modules')
        return model.eval().requires_grad_(False).to(device),info
    if binding is not None:raise ValueError('A versioned D46 binding cannot initialize from another native parent')
    info=audit(path,verify=False)
    from starVLA.rl.qwen_world_compare.deployment import _library_runtime
    original=ROOT/'IL44_PARENT/scored_export_identity.json'
    expected=json.loads(original.read_text())['runtime_libraries']
    if _library_runtime()!=expected:raise ValueError('Native D44 Qwen/Torch execution library identity differs from its scored export')
    raw=torch.load(Path(path)/info['seal']['model_file'],map_location='cpu',weights_only=False,mmap=True)
    model=model_from_identity(info['identity'],train_heads=True)
    strict_tensor_load(model,raw['module'])
    if not readouts:
        from starVLA.rl.qwen_world_compare.deployment import prune_deployment
        model=prune_deployment(model,copy_model=False)
    return model.eval().requires_grad_(False).to(device),info
