"""Strict READY world8->16 native transition, preserving common Adam payload.

Writes ONLY a new caller-selected execution root. Existing source/native files
and the original world's journal are never edited or copied as new history.
"""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import uuid
import torch
from starVLA.rl.qwen_world_compare.contracts import fingerprint,file_sha
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.d44_native_rl.recovery import Boundary
from starVLA.rl.qwen_world_abcd_research.checkpoint import atomic_torch,CHECKPOINT_CONTRACT
from .planning import validate_recipe

SCHEMA='d46_exact_native_READY_world8_to16_v1'
STATE_SCHEMA='d46_world16_virtual8_sampler_state_v1'
BASE='ff3013c4045e9505ae9cd39b9418247cc03a596d'
CONTROL_PROTOCOL=dict(schema='d46_world16_CPU_control_plane_v1',backend='gloo',timeout_seconds=600,
    purpose='allow full model load and true B1 startup skew; no policy/optimizer change')


def source_inventory():
    from tools.d44_native_rl.train import code_identity
    files=code_identity()
    for root in ('starVLA/rl/d46_world16','tools/d46_world16'):
        files.update({str(p):file_sha(p)for p in sorted(Path(root).glob('*.py'))})
    return files


def tensor_sha(x):
    if x.dtype!=torch.uint8 or x.ndim!=1:raise ValueError('Invalid saved generator state')
    import hashlib
    return hashlib.sha256(x.detach().cpu().numpy().tobytes()).hexdigest()


def validate_virtual(state):
    value=state.get('world16_virtual_sampler',{})
    lanes=value.get('lanes',[])
    if value.get('schema')!=STATE_SCHEMA or len(lanes)!=8 or value.get('committed_collections')!=state['collections']:
        raise ValueError('Missing/changed virtual8 sampler ancestry')
    for lane in lanes:
        tensor_sha(lane);g=torch.Generator();g.set_state(lane)
    return fingerprint([tensor_sha(x)for x in lanes])


class World16Boundary(Boundary):
    @staticmethod
    def validate_boundary(state,batch,identity,optimizer=None,scheduler=None,**kw):
        if identity.get('world')!=16 or identity.get('native_world_transition',{}).get('schema')!=SCHEMA:
            raise ValueError('World16 requires typed native transition identity')
        if identity['recipe']['arm']not in('D0','D5'):raise ValueError('Variable-draw/other arms not admitted')
        if identity['native_world_transition'].get('control_plane')!=CONTROL_PROTOCOL:raise ValueError('World16 control protocol changed')
        out=Boundary.validate_boundary(state,batch,identity,optimizer,scheduler,**kw)
        out['virtual8_rng_sha256']=validate_virtual(state)
        if state.get('journal_origin_updates')!=identity['native_world_transition']['origin_updates']:
            raise ValueError('Native journal origin changed')
        return out
    @staticmethod
    def validate_ranks(rows):
        Boundary.validate_ranks(rows)
        if len({r['virtual8_rng_sha256']for r in rows})!=1:raise ValueError('Ranks disagree on sampler lanes')
        return rows


def transition_identity(old,checkpoint,files,new_sources,qualification):
    body=copy.deepcopy(old);body.pop('sha256',None);body.update(world=16,fresh_Adam=False,source_files=new_sources,qualification=qualification)
    body['native_world_transition']=dict(schema=SCHEMA,source_world=8,target_world=16,same_algorithm=True,
        origin_updates=json.loads((Path(checkpoint)/'COMPLETE').read_text())['updates'],source_qualification=old['qualification'],target_qualification=qualification,
        source_run_identity_sha256=old['sha256'],source_checkpoint=str(Path(checkpoint).resolve()),
        source_files_sha256=file_sha(Path(checkpoint)/'files.json'),common_optimizer_sha256=files['optimizer.pt'],
        virtual_sampler_lanes=8,global_scenes=64,group=16,inner_updates=2,control_plane=copy.deepcopy(CONTROL_PROTOCOL),
        compatibility='same real-valued update target; FP32 DDP summation topology may differ, not cross-world bitexact',
        parent_policy_unchanged=True,optimizer='inherited_AdamW_no_reset',old_owner_may_adopt=False)
    body['sha256']=fingerprint(body);return body


def prepare(old_checkpoint,new_root,*,qualification=False):
    """Actual byte verification and seal; no model/GPU load, no dispatch."""
    old_checkpoint=Path(old_checkpoint).resolve();root=Path(new_root).resolve()
    if root.exists():raise FileExistsError('New execution root must not exist')
    if root.is_relative_to(old_checkpoint.parent.parent):raise ValueError('Never write a new execution under the old run')
    if '.incomplete'in old_checkpoint.name:raise ValueError('Unsealed parent')
    seal=json.loads((old_checkpoint/'COMPLETE').read_text());files=json.loads((old_checkpoint/'files.json').read_text())
    old=json.loads((old_checkpoint/'identity.json').read_text());body=dict(old);sha=body.pop('sha256',None)
    if (seal.get('world')!=8 or old.get('world')!=8 or seal.get('contract')!=CHECKPOINT_CONTRACT or seal.get('schema')!=2
            or fingerprint(body)!=sha or seal.get('files_sha256')!=file_sha(old_checkpoint/'files.json')
            or seal.get('phase')!='READY' or seal.get('next_inner')!=0):raise ValueError('Only complete original world8 READY checkpoint')
    cfg=validate_recipe(old['recipe'])
    from starVLA.rl.d44_native_rl.config import validate
    validate(cfg)
    if cfg['arm']not in('D0','D5')or(old.get('qualification')and not qualification):raise ValueError('Only D0/D5; qualification weights may never enter formal execution')
    from starVLA.rl.d44_native_rl.parent import D46_MODEL_SHA
    tiny=(qualification and old.get('qualification_fixture')=='d46_tiny_native_world_transition_CPU_v1'
          and old.get('parent_model_sha256','').startswith('tiny_native_qualification:')
          and old['scope']['trainable_numel']<1000000)
    if not tiny and old.get('parent_model_sha256')!=D46_MODEL_SHA:raise ValueError('Must continue own IL46 algorithm, not another RL policy')
    required={'identity.json','transaction.json','optimizer.pt'}|{f'rank{r}.pt'for r in range(8)}
    if not required<=files.keys():raise ValueError('Incomplete old native inventory')
    for name in files:
        if Path(name).name!=name or file_sha(old_checkpoint/name)!=files[name]:raise ValueError('Old native bytes changed: '+name)
    current=source_inventory();train='tools/d44_native_rl/train.py'
    oldtrain=subprocess.run(['git','show',BASE+':'+train],check=True,stdout=subprocess.PIPE).stdout
    import hashlib
    for name,value in old['source_files'].items():
        expected=hashlib.sha256(oldtrain).hexdigest()if name==train else current.get(name)
        if expected!=value:raise ValueError('Unreviewed old source difference: '+name)
    common=torch.load(old_checkpoint/'optimizer.pt',map_location='cpu',weights_only=False)
    if set(common['adapters'])!=set(old['scope']['trainable_names']):raise ValueError('Incomplete common trainable closure')
    if any(x.requires_grad or x.dtype!=torch.float32 or not torch.isfinite(x).all()for x in common['adapters'].values()):raise ValueError('Invalid common parameter bytes')
    owns=[torch.load(old_checkpoint/f'rank{r}.pt',map_location='cpu',weights_only=False)for r in range(8)]
    oldtx=json.loads((old_checkpoint/'transaction.json').read_text());rows=[]
    for r,own in enumerate(owns):
        if own['identity_sha256']!=old['sha256']or own['rollout']is not None:raise ValueError('Not exact READY ancestor')
        summary=Boundary.validate_boundary(own['state'],None,old,common['optimizer'],common['scheduler']);rows.append(summary)
        if summary!=oldtx['ranks'][r]or own['state']['updates']!=seal['updates']:raise ValueError('Old transaction changed')
        if any(k in own['state']for k in ('world16_virtual_sampler','journal_origin_updates')):raise ValueError('Already migrated parent')
    Boundary.validate_ranks(rows)
    identity=transition_identity(old,old_checkpoint,files,current,qualification)
    target=root/'checkpoints'/f"update{seal['updates']:06}_inner0";attempt=root.with_name(root.name+'.incomplete.'+uuid.uuid4().hex)
    cp=attempt/'checkpoints'/target.name;cp.mkdir(parents=True)
    try:
        summaries=[];lanes=[own['rng']['rollout_generator'].clone()for own in owns]
        for r in range(16):
            own=copy.deepcopy(owns[r%8]);own['identity_sha256']=identity['sha256']
            own['state']['world16_virtual_sampler']=dict(schema=STATE_SCHEMA,lanes=copy.deepcopy(lanes),committed_collections=own['state']['collections'])
            own['state']['journal_origin_updates']=seal['updates']
            summaries.append(World16Boundary.validate_boundary(own['state'],None,identity,common['optimizer'],common['scheduler']))
            atomic_torch(cp/f'rank{r}.pt',own)
        World16Boundary.validate_ranks(summaries)
        # Exact common bytes: never deserialize/reserialize Adam as a new initialization.
        shutil.copyfile(old_checkpoint/'optimizer.pt',cp/'optimizer.pt')
        if file_sha(cp/'optimizer.pt')!=files['optimizer.pt']:raise ValueError('Common optimizer copy changed')
        atomic_json(cp/'identity.json',identity)
        receipt=dict(schema=SCHEMA,source_identity_sha256=old['sha256'],target_identity_sha256=identity['sha256'],
            source_checkpoint=str(old_checkpoint),source_files_sha256=seal['files_sha256'],source_rank_files_sha256={str(r):files[f'rank{r}.pt']for r in range(8)},
            common_optimizer_sha256=files['optimizer.pt'],origin_updates=seal['updates'],phase='READY',
            native_weights_and_Adam_inherited=True,source_inventory=current,qualification=qualification,source_qualification=old['qualification'],control_plane=copy.deepcopy(CONTROL_PROTOCOL),qualification_weights_never_return_to_formal=True,source_world=8,target_world=16)
        receipt['sha256']=fingerprint(receipt)
        atomic_json(cp/'world_transition.json',receipt);atomic_json(attempt/'WORLD_TRANSITION.json',receipt)
        atomic_json(cp/'transaction.json',dict(schema=CHECKPOINT_CONTRACT,attempt=uuid.uuid4().hex,ranks=summaries,
            execution_identity=None,migration_receipt_sha256=None,world_transition_sha256=receipt['sha256']))
        inventory={p.name:file_sha(p)for p in sorted(cp.iterdir())if p.is_file()};atomic_json(cp/'files.json',inventory)
        atomic_json(cp/'COMPLETE',dict(schema=2,contract=CHECKPOINT_CONTRACT,phase='READY',world=16,updates=seal['updates'],next_inner=0,files_sha256=file_sha(cp/'files.json')))
        atomic_json(attempt/'identity.json',identity);atomic_json(attempt/'latest.json',dict(checkpoint=str(target),updates=seal['updates'],files_sha256=file_sha(cp/'files.json')))
        os.rename(attempt,root)
        if not tiny:
            from starVLA.rl.d44_native_rl.deployment import inspect
            inspect(target)  # Existing official strict metadata/parent/closure loader guard.
    except BaseException:
        # Never remove failed conversion evidence or alter the old checkpoint.
        raise
    return dict(root=str(root),checkpoint=str(target),identity_sha256=identity['sha256'],receipt_sha256=receipt['sha256'],official_deployment_inspect_passed=not tiny)
