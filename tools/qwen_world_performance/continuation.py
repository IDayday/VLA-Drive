"""Opt-in measured sharing and exact own-run cooperative continuation intents.

No symlink, registry amendment, stop file, signal, or dispatch occurs on import.
The alias is a real output directory view, never an unused argv ownership hint.
"""
import time,uuid
from pathlib import Path
from .launch import ROOT,read,sha,inspect_native
from starVLA.rl.qwen_world_research_data.manifest import fingerprint,sealed,read_sealed

ALIAS=Path('/mnt/project/qwen-world-safety-exploration-20261031/performance_training_20261103')
SHARING_HOSTS={'training-vlawm-tsinghua-zt3','training-vlawm-tsinghua-zt5'}
PROTOCOL=dict(schema='world_performance_measured_native_sharing_v1',maximum_campaign_native_per_host=1,
    maximum_known_native_per_host=2,minimum_free_MiB=32768,output_alias=str(ALIAS))
PROTOCOL_V2=dict(PROTOCOL,schema='world_performance_measured_native_sharing_v2',
    maximum_campaign_native_per_host=2)
MIGRATION='world_performance_owned_same_run_continuation_v1'


def same_path(a,b):
    try:return Path(a).resolve()==Path(b).resolve()
    except(TypeError,ValueError,OSError):return False


def validate_resource_fields(run):
    protocol=run.get('resource_protocol')
    if protocol is None:
        if any(r['lease_slot']!='primary'for r in run['deployment_candidates']):raise ValueError('Shared native slots require explicit opt-in')
        return
    if protocol not in (PROTOCOL,PROTOCOL_V2):raise ValueError('Changed measured-sharing protocol')
    expected=ALIAS/'runs'/run['id']
    if(not ALIAS.is_symlink()or not same_path(ALIAS,ROOT)
       or run.get('execution_output_root')!=str(expected)or not same_path(expected,run['root'])):
        raise ValueError('Transparent real same-directory output alias required')
    for route in run['deployment_candidates']:
        if route['lease_slot']=='primary':continue
        # v2 permits a second own learner on an already registered host. Live
        # authorization, owner visibility, headroom and UUID locks still decide
        # dispatch; this field confers no permanent or temporary host grant.
        hosts=SHARING_HOSTS if protocol==PROTOCOL else {
            r['host'] for r in run['deployment_candidates'] if r['lease_slot']=='primary'
        }-{'training-vla-zt','training-vlawm-tsinghua-zt'}
        if(route['lease_slot']not in('shared','research')or route['host']not in hosts
           or route['minimum_free_MiB']<32768):raise ValueError('Only explicit registered qualified sharing routes')


def shared_sample_ready(sample,run=None):
    """Additional cap only. Caller/worker still enforce every existing GPU check."""
    outputs=set()
    for owner in sample.get('owners',[]):
        entry=owner.get('entry','')
        if 'train'not in entry or'evaluat'in entry:continue
        if not owner.get('visible')or not isinstance(owner.get('output'),str):return False,'unresolved_native_owner'
        outputs.add(Path(owner['output']).resolve())
    protocol=(run or {}).get('resource_protocol',PROTOCOL)
    if protocol not in (PROTOCOL,PROTOCOL_V2):return False,'unknown_sharing_protocol'
    own={p for p in outputs if p.is_relative_to(ROOT)}
    if run and Path(run['root']).resolve() in outputs:return False,'same_run_native_already_present'
    if len(own)>=protocol['maximum_campaign_native_per_host']:return False,'campaign_native_capacity_reached'
    if len(outputs)>=protocol['maximum_known_native_per_host']:return False,'two_known_native_jobs_already_present'
    return True,'measured_native_capacity_available'


def amendment_preview(registry,arms=('S1_NOMINAL_RESCUE',)):
    """Pure proposed operational fields. Root must approve/create alias/apply."""
    changes=[]
    for run in registry['runs']:
        if run['id']not in arms:continue
        routes=[dict(host='training-vlawm-tsinghua-zt5',lease_slot=slot,minimum_free_MiB=32768)for slot in('shared','research')]
        routes+=run['deployment_candidates']
        changes.append(dict(arm=run['id'],resource_protocol=PROTOCOL,execution_output_root=str(ALIAS/'runs'/run['id']),deployment_candidates=routes))
    return dict(status='PREVIEW_NOT_APPLIED',schema=PROTOCOL['schema'],alias=str(ALIAS),alias_target=str(ROOT),
        changes=changes,scientific_source_config_parent_unchanged=True,
        required_root_actions=['Review and create the exact transparent symlink if absent; reject any existing different target',
            'Apply only the proposed operational fields under RUNS publication lock; preserve all other concurrent fields',
            'CPU-only adopt reviewed ops_v2, preserving all native/eval children',
            'For a move, seal make_migration_intent, then exact job ownSTOP; wait EXIT0+8rankCOMPLETE+oldworker exit before continuation'])


def make_migration_intent(run,node,target_host,lease_slot='shared'):
    """Pure receipt body; Root writes it then ownSTOP after exact fresh review."""
    if target_host not in SHARING_HOSTS or lease_slot not in('shared','research'):raise ValueError('Only explicit permanent continuation routes')
    job=Path(node['job']).resolve()
    if not job.is_relative_to(ROOT/'jobs'/run['id']):raise ValueError('Foreign source job')
    running=read(job/'RUNNING.json',{});dispatch=read(job/'DISPATCH.json',{})
    if(job/'RESULT.json').exists()or running.get('status')!='RUNNING':raise ValueError('Exact current native attempt must still be running')
    if(running.get('host')!=node['host']or dispatch.get('worker_pid')!=running.get('worker_pid')
       or dispatch.get('worker_start_ticks')!=running.get('worker_start_ticks')
       or dispatch.get('command')!=running.get('command')or running.get('cwd')!=run['source']):raise ValueError('Worker/source attempt binding changed')
    latest=read(Path(run['root'])/'latest.json');cp=inspect_native(run,latest['checkpoint'])
    body=dict(schema=MIGRATION,id=uuid.uuid4().hex,requested_unix=time.time(),requester='root_operator',
        reason='PERMANENT_HOST_MEASURED_SHARING_CONTINUATION',run=run['id'],run_root=str(Path(run['root']).resolve()),
        run_identity=read(Path(run['root'])/'identity.json')['sha256'],source_job=str(job),source_host=node['host'],
        source=run['source'],source_commit=run['source_commit'],config_sha256=run['config_sha256'],
        source_worker_pid=running['worker_pid'],source_worker_start_ticks=running['worker_start_ticks'],
        command_sha256=fingerprint(running['command']),requested_latest_checkpoint=cp,
        target_host=target_host,target_lease_slot=lease_slot,execution_output_root=str(ALIAS/'runs'/run['id']),
        optimizer_rng_scene_cursor_pending_inherited=True,weights_only=False,phase_reset=False)
    return sealed(body)


def verified_migration(run,node,result,cp):
    job=Path(node['job']).resolve();path=job/'OWN_NATIVE_CONTINUATION.json'
    if not path.exists()or not(job/'STOP_REQUESTED').exists():return None
    try:
        intent=read_sealed(path);command=result.get('command',[])
        expected=dict(schema=MIGRATION,requester='root_operator',reason='PERMANENT_HOST_MEASURED_SHARING_CONTINUATION',
            run=run['id'],run_root=str(Path(run['root']).resolve()),run_identity=read(Path(run['root'])/'identity.json')['sha256'],
            source_job=str(job),source_host=node['host'],source=run['source'],source_commit=run['source_commit'],config_sha256=run['config_sha256'],
            source_worker_pid=result.get('worker_pid'),source_worker_start_ticks=result.get('worker_start_ticks'),command_sha256=fingerprint(command),
            optimizer_rng_scene_cursor_pending_inherited=True,weights_only=False,phase_reset=False)
        if any(intent.get(k)!=v for k,v in expected.items()):return None
        if(result.get('returncode')!=0 or result.get('forced_expiry')or intent['target_host']not in SHARING_HOSTS
           or intent['target_lease_slot']not in('shared','research')or intent['requested_unix']>result.get('end_unix',0)
           or(job/'STOP_REQUESTED').stat().st_mtime<intent['requested_unix']-2
           or cp['updates']<intent['requested_latest_checkpoint']['updates']):return None
        return intent
    except(OSError,KeyError,TypeError,ValueError):return None
