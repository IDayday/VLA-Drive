"""Opt-in future native attempts to the approved cap; published-CP eval discovery.

No model, GPU, registry, owner, or checkpoint mutation is performed here.
Legacy active attempt commands retain their original targets and receipts.
"""
import hashlib,json
from pathlib import Path
import yaml

SCHEMA='world_performance_continuous_to_registered_cap_v1'
PROTOCOL=dict(schema=SCHEMA,scope='future_attempts_only')


def enabled(run):
    value=run.get('native_execution')
    if value is None:return False
    if value!=PROTOCOL:raise ValueError('Unknown continuous native execution protocol')
    return True


def attempt_targets(run,native,root):
    """Adopt one in-flight legacy attempt, then use only the approved cap.

    A returned but unharvested job remains an existing attempt. The original
    run_stage handles its actual RESULT (including scientific failures), rather
    than hiding it behind a new cap job or a stale top-level status.
    """
    if not enabled(run):return list(run['native_gates'])
    active=[];cap=run['target_updates']
    for key,node in native.get('stages',{}).items():
        if node.get('complete')or not node.get('job'):continue
        target=int(key);job=Path(node['job']).resolve()
        if(target not in run['native_gates']or target>cap or
           not job.is_relative_to(Path(root).resolve()/'jobs'/run['id'])):
            raise ValueError('Unknown/foreign existing native attempt')
        command=node.get('command')
        if command is not None:
            if(command.count('--stop-after-update')!=1 or
               command[command.index('--stop-after-update')+1]!=str(target)):
                raise ValueError('Existing attempt target/command changed')
        active.append(target)
    if len(active)>1:raise ValueError('Multiple unharvested native attempts require explicit review')
    return active+[cap]if active and active[0]!=cap else [cap]


def _read(path):return json.loads(Path(path).read_text())
def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _fingerprint(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def published_checkpoint_at_gate(run,updates):
    """Metadata-only discovery; the unchanged exporter validates full payloads.

    Return exactly the legacy checkpoint dict, whose identity_sha256 is the
    identity *file* SHA, preserving all existing SPEC bytes/world/job bindings.
    Never manufactures native stage completion or selects a newer live model.
    """
    if not enabled(run):raise ValueError('Published native discovery requires explicit continuous protocol')
    if updates not in run['gates']or updates>run['target_updates']:raise ValueError('Unregistered evaluation update')
    state=_read(run['owner_state'])if Path(run['owner_state']).exists()else{}
    native=state.get('runs',{}).get(run['id'],{})
    root=Path(run['root']).resolve();path=root/'checkpoints'/f'update{updates:06}_inner0'
    if not(path/'COMPLETE').is_file():return None,native
    if path.resolve()!=path or not path.is_relative_to(root/'checkpoints'):raise ValueError('Checkpoint path escaped native root')
    seal=_read(path/'COMPLETE');files=_read(path/'files.json');identity=_read(path/'identity.json');tx=_read(path/'transaction.json')
    required={'optimizer.pt','identity.json','transaction.json'}|{f'rank{r}.pt'for r in range(8)}
    if not required<=set(files)or any(Path(n).is_absolute()or not(path/n).resolve().is_relative_to(path)or not(path/n).is_file()for n in files):
        raise ValueError('Incomplete/escaping native inventory')
    if(seal.get('schema')!=2 or seal.get('world')!=8 or seal.get('updates')!=updates or seal.get('next_inner')!=0
       or seal.get('phase')not in('READY','COMPLETE')or seal.get('files_sha256')!=_sha(path/'files.json')):
        raise ValueError('Not the exact atomically published eight-rank READY checkpoint')
    body=dict(identity);saved=body.pop('sha256',None)
    if(saved!=_fingerprint(body)or identity!=_read(root/'identity.json')or identity.get('purpose')!='formal'
       or identity.get('world')!=8 or _sha(run['config'])!=run['config_sha256']
       or identity.get('recipe')!=yaml.safe_load(Path(run['config']).read_text())
       or identity['recipe'].get('adaptation')!='ATTN_LORA_MEMORY'):
        raise ValueError('Published native scientific identity/recipe changed')
    for name in('identity.json','transaction.json'):
        if _sha(path/name)!=files[name]:raise ValueError('Changed native metadata inventory')
    point=next((x for x in run['exposure_mapping'].values()if x['updates']==updates),None)
    if point is None:raise ValueError('No exact registered exposure gate')
    rows=tx.get('ranks',[])
    if(tx.get('schema')!='simwam_checkpoint_transaction_v2'or len(rows)!=8
       or [r.get('rank')for r in rows]!=list(range(8))
       or any(r.get('updates')!=updates or r.get('next_inner')!=0 or r.get('phase')!=seal['phase']
              or r.get('run')!=saved or r.get('valid_scene_exposures')!=point['valid_scene_exposures']
              or r.get('collections')!=point['collections']for r in rows)):
        raise ValueError('Eight-rank actual exposure/transaction mismatch')
    return dict(path=str(path),updates=updates,files_sha256=seal['files_sha256'],
                identity_sha256=_sha(path/'identity.json')),native
