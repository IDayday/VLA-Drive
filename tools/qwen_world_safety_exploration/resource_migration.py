"""Review only a clean native pause on a newly withdrawn resource.

No actor, scheduler, reward, old-chain or checkpoint bytes are modified here.
The original strict full-state loader verifies all tensor hashes on resume.
"""
import hashlib
import json
from pathlib import Path
import time


def requeue_clean_resource_pause(node,run,latest,receipt,allowed):
    if node.get('host')in allowed or receipt.get('returncode')!=0:return False
    path=Path(latest['checkpoint']);seal=json.loads((path/'COMPLETE').read_text())
    if not path.resolve().is_relative_to((Path(run['root'])/'checkpoints').resolve()):raise ValueError('Foreign native recovery point')
    if seal.get('world')!=8 or seal.get('updates')!=latest['updates']or seal.get('phase')not in ('READY','ROLLOUT_COMMITTED','INNER_COMMITTED'):
        raise ValueError('Resource migration requires a legal complete native stage')
    if not 0<=seal.get('next_inner',-1)<4:raise ValueError('Invalid pending inner cursor')
    if hashlib.sha256((path/'files.json').read_bytes()).hexdigest()!=seal['files_sha256']:raise ValueError('Changed native inventory seal')
    inventory=json.loads((path/'files.json').read_text())
    if any(not(path/name).resolve().is_relative_to(path.resolve())or not(path/name).is_file()for name in inventory):raise ValueError('Missing or unsafe native file')
    for rank in range(8):
        result=json.loads((Path(run['root'])/f'result_rank{rank}.json').read_text())
        if (result.get('status')!='STOPPED_AT_SAVED_BOUNDARY'or result.get('checkpoint')!=str(path)
                or any(result.get(k)!=seal.get(k)for k in ('updates','phase','next_inner'))):
            raise ValueError('Resource pause lacks eight matching native rank results')
    event=dict(previous_job=node['job'],previous_host=node['host'],resume_checkpoint=str(path),
        updates=seal['updates'],phase=seal['phase'],next_inner=seal['next_inner'],at_unix=time.time(),
        reason='User withdrew this host; clean native pause verified; same-budget continuation only')
    node.setdefault('resource_migrations',[]).append(event)
    for key in ('job','host','resume','start_unix','dispatch_error'):node.pop(key,None)
    return True


def requeue_clean_scene_pause(node,receipt,allowed):
    """A verified scene-boundary stop keeps exact source/command/cache identity."""
    command=node.get('command',[])
    if (node.get('host')in allowed or receipt.get('returncode')!=0 or receipt.get('status')!='EXITED'
            or receipt.get('host')!=node.get('host')or receipt.get('command')!=command
            or 'tools.qwen_world_safety_exploration.evaluate'not in command
            or '--rank'not in command or '--world'not in command or '--output'not in command or '--summarize'in command):return False
    rank=int(command[command.index('--rank')+1]);world=int(command[command.index('--world')+1]);output=Path(command[command.index('--output')+1])
    path=output/f'rank{rank}.json'
    if not path.exists():return False
    result=json.loads(path.read_text())
    if (result.get('status')!='STOPPED_AT_SCENE_BOUNDARY'or result.get('rank')!=rank or result.get('world')!=world
            or result.get('host')!=node['host']or not result.get('identity')):return False
    node.setdefault('resource_pauses',[]).append(dict(job=node['job'],host=node['host'],
        rank_receipt=str(path),rank_receipt_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),at_unix=time.time()))
    for key in ('job','result','error'):node.pop(key,None)
    node['status']='RESOURCE_REQUEUE'
    return True
