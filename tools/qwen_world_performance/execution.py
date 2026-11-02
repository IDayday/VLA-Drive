"""Typed frozen-IL-reference execution migration; never changes scientific cfg.

The only accepted CLI opt-in is the reviewed boolean frozen-reference cache.
Full old/new identities and existing native-lineage verifier remain authoritative.
No receipt is created, registry written, model loaded, or process started here.
"""
import copy,json,subprocess
from pathlib import Path
from .launch import ROOT,FAST,read,sha
from starVLA.rl.simwam_recipe.lineage import scientific_identity,verify_migration
from starVLA.rl.qwen_world_research_data.manifest import fingerprint
from starVLA.rl.qwen_world_safety_exploration.recovery import tensor_digest

SCHEMA='world_performance_frozen_reference_cache_execution_v1'
FIELDS=('execution_source','execution_source_commit','execution_optimizations',
        'execution_migration_receipt','execution_migration_receipt_sha256','execution_parent_checkpoint')
WORLD8='world_full_IL70_reference_cache_world8_native_equivalence_v1'
SINGLE='world_real_full_IL70_reference_cache_singleGPU_four_inner_v1'
CACHE_HELPER='starVLA/rl/qwen_world_research_algorithms/frozen_reference.py'
_PROOFS={}


def verify_identity(identity):
    body=dict(identity);digest=body.pop('sha256',None)
    if not digest or fingerprint(body)!=digest:raise ValueError('Execution identity seal changed')


def validate_execution_fields(run):
    present=[k for k in FIELDS if k in run]
    if 'execution_flags'in run:raise ValueError('Arbitrary execution flags are forbidden')
    if not present:return None
    if set(present)!=set(FIELDS):raise ValueError('Typed execution source requires exact receipt/parent/source/optimization fields')
    if run['id']==FAST or run['trainer_module']!='tools.qwen_world_performance.train':
        raise ValueError('FAST_PACK4 is an independent fresh lineage, not single-scene cache migration')
    if run['execution_optimizations']!={'schema':SCHEMA,'frozen_reference_cache':True}:
        raise ValueError('Only explicit frozen IL reference cache execution opt-in')
    receipt_path=Path(run['execution_migration_receipt']).resolve();receipt_sha=sha(receipt_path)
    if receipt_sha!=run['execution_migration_receipt_sha256']:raise ValueError('Execution migration receipt file changed')
    source=Path(run['execution_source']).resolve();parent=Path(run['execution_parent_checkpoint']).resolve();root=Path(run['root']).resolve()
    if not parent.is_relative_to(root/'checkpoints')or not(parent/'COMPLETE').is_file():raise ValueError('Only own sealed native execution parent')
    old=read(root/'identity.json');verify_identity(old)
    cache_key=(str(receipt_path),receipt_sha,str(source),run['execution_source_commit'],old['sha256'])
    proof=_PROOFS.get(cache_key)
    if proof is None:
        receipt=read(receipt_path);new=receipt['new_identity'];verify_identity(new)
        if receipt.get('schema')!='world_frozen_reference_cache_execution_migration_draft_v1':raise ValueError('Unknown typed migration receipt schema')
        if receipt['old_identity']!=old or read(parent/'identity.json')!=old:raise ValueError('Execution migration must retain exact existing logical-run identity')
        if old.get('purpose')!='formal'or new.get('purpose')!='formal'or scientific_identity(old)!=scientific_identity(new):raise ValueError('Execution migration changed scientific identity')
        if old.get('recipe')!=new.get('recipe')or old['recipe']['research']['arm']!=run['id']:raise ValueError('Execution recipe/arm changed')
        allowed={'source_files','execution_identity','execution_optimizations','sha256'}
        if any(old.get(k)!=new.get(k)for k in (set(old)|set(new))-allowed):raise ValueError('Unregistered non-execution identity delta')
        options=copy.deepcopy(old.get('execution_optimizations',{}));cache=new.get('execution_optimizations',{}).get('frozen_IL_reference_cache')
        if not isinstance(cache,dict)or cache.get('schema')!='world_frozen_IL_reference_W3_terms_cache_v1':raise ValueError('Missing exact frozen-reference execution protocol')
        options['frozen_IL_reference_cache']=cache
        if new.get('execution_optimizations')!=options:raise ValueError('Unregistered execution optimization')
        actual_commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
        if actual_commit!=run['execution_source_commit']:raise ValueError('Actual execution source commit differs')
        if Path(new['execution_identity']['fields']['module_root']).resolve()!=source:raise ValueError('Execution module root differs')
        if new['execution_identity']['fields']['source_files']!=new['source_files']:raise ValueError('Execution source inventory differs')
        for rel,digest in new['source_files'].items():
            path=(source/rel).resolve()
            if Path(rel).is_absolute()or not path.is_relative_to(source)or sha(path)!=digest:raise ValueError('Execution source payload differs: '+rel)
        if CACHE_HELPER not in new['source_files']:raise ValueError('Cache helper not bound to execution inventory')
        evidence=[read(x['path'])for x in receipt['evidence']]
        if not{SINGLE,WORLD8}<={x.get('schema')for x in evidence}:raise ValueError('Actual singleGPU and world8 native PASS evidence required')
        for x in evidence:
            if x.get('status')!='PASS':raise ValueError('Execution equivalence evidence is not PASS')
            if x.get('schema')==WORLD8 and(x.get('world')!=8 or x.get('cache_helper_sha256')!=new['source_files'][CACHE_HELPER]
                    or x.get('qualified_source_files_sha256')!=tensor_digest(new['source_files'])):
                raise ValueError('World8 qualification does not bind this exact execution source')
        verify_migration(receipt_path,old,new,parent)
        proof=dict(source=str(source),receipt=str(receipt_path),new_identity=new,old_identity=old)
        _PROOFS[cache_key]=proof
    else:
        # Immutable source proof cached; parent/evidence lineage is checked again
        # for a concrete resume in command_execution, not inferred from progress.
        verify_migration(receipt_path,proof['old_identity'],proof['new_identity'],parent)
    return proof


def command_execution(run,resume):
    proof=validate_execution_fields(run)
    if proof is None:return run['source'],[]
    if resume is None:raise ValueError('Reference cache migration requires own strict native resume, never fresh IL reinitialization')
    cp=Path(resume).resolve()
    if not cp.is_relative_to(Path(run['root']).resolve()/'checkpoints')or not(cp/'COMPLETE').exists():raise ValueError('Missing own sealed resume checkpoint')
    verify_migration(proof['receipt'],proof['old_identity'],proof['new_identity'],cp)
    return proof['source'],['--frozen-reference-cache','--migration-receipt',proof['receipt']]


def validate_checkpoint_in_execution(run,checkpoint):
    """CPU-only isolated validator from the actual migrated scientific source.

    The new recovery checks present cache payloads. It is not silently replaced
    by an older ops interpreter's recovery; no actor, GPU or model is loaded.
    """
    source,_=command_execution(run,checkpoint)
    from starVLA.rl.qwen_world_safety_exploration.resources import MODEL_PYTHON
    import os
    code='''import json,sys,torch
from pathlib import Path
from starVLA.rl.qwen_world_performance import recovery
from starVLA.rl.qwen_world_performance.sampling_priority import stream_state_validation
p=Path(sys.argv[1]);identity=json.loads((p/'identity.json').read_text())
common=torch.load(p/'optimizer.pt',map_location='cpu',weights_only=False);summaries=[];digests=[]
for rank in range(8):
 own=torch.load(p/f'rank{rank}.pt',map_location='cpu',weights_only=False)
 if own['identity_sha256']!=identity['sha256']:raise ValueError('Export rank identity changed')
 summaries.append(recovery.validate_boundary(own['state'],own['rollout'],identity,common['optimizer'],common['scheduler'],expected_rank=rank))
 digests.append(stream_state_validation(own['state'],run_identity=identity['sha256'],stream_identity=identity['data_stream_identity'])['stream_state_sha256'])
recovery.validate_ranks(summaries)
if len(set(digests))!=1:raise ValueError('Export ranks disagree on scene/feedback state')
print(json.dumps(dict(status='PASS',scene_stream_sha256=digests[0],validated_ranks=8)))
'''
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=source)
    output=subprocess.check_output([MODEL_PYTHON,'-c',code,str(Path(checkpoint).resolve())],cwd=source,env=env,text=True)
    proof=json.loads(output.strip().splitlines()[-1])
    if proof.get('status')!='PASS'or proof.get('validated_ranks')!=8:raise ValueError('Migrated CPU native validator did not pass')
    return proof
