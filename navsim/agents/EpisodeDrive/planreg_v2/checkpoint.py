"""Strict V2 initialization, complete resume payloads and actual student topology export."""
import copy
import json
from pathlib import Path
import torch
from .agent import PlanRegV2Agent,file_sha256
from . import CHECKPOINT_SCHEMA,STUDENT_SCHEMA

TRAINING_PREFIXES = ('ema_teacher.','wm_predictor.','motion_normalizer.','kinematics_codec.')
TRAINING_KEYS = {'optimizer_updates'}


def export_student(source,output):
    if Path(output).exists(): raise FileExistsError('Never overwrite an existing checkpoint')
    ckpt = torch.load(source,map_location='cpu',weights_only=False)
    if ckpt.get('schema') != CHECKPOINT_SCHEMA: raise ValueError('Not an exact V2.2 training checkpoint; old recipe needs explicit warm start')
    state = ckpt['model']
    retained = {n:v for n,v in state.items() if not n.startswith(TRAINING_PREFIXES) and n not in TRAINING_KEYS}
    config = copy.deepcopy(ckpt['config'])
    config.update(shared_init_path=None,world_model_enabled=False,gradient_checkpointing=False)
    config['normalizer_statistics'] = dict(mean=state['action_head.normalizer.mean'].tolist(),
        std=state['action_head.normalizer.raw_std'].tolist(),metadata=state['action_head.normalizer._extra_state'])
    config['normalizer_path'] = None
    # The deployment constructor has no teacher, predictor, or motion-target modules.
    retained['optimizer_updates'] = torch.zeros((),dtype=torch.long)
    torch.save(dict(schema=STUDENT_SCHEMA,model=retained,config=config),output)
    manifest = dict(source_sha256=file_sha256(source),export_sha256=file_sha256(output),
        removed_keys=sorted(set(state)-set(retained)),retained_key_count=len(retained),config=config,
        source_git_commit=ckpt.get('git_commit'),precision='BF16 VLM compute; FP32 trainable storage and action/scorer')
    Path(str(output)+'.manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest


def load_student(path,device='cpu'):
    ckpt = torch.load(path,map_location='cpu',weights_only=False)
    if ckpt.get('schema') != STUDENT_SCHEMA: raise ValueError('Student-only V2.2 artifact required; old V2 uses its original worktree')
    agent = PlanRegV2Agent(ckpt['config'],device=device,deployment=True)
    agent.load_state_dict(ckpt['model'],strict=True)
    return agent.eval()


def convert_physical_output_head(weight,bias,normalizer):
    scale,mean = normalizer.std.flatten(),normalizer.mean.flatten()
    if weight.shape[0] != 24 or bias.shape != (24,): raise ValueError('Expected physical 8x3 output head')
    return weight/scale[:,None],(bias-mean)/scale


def warm_start_v1(v2,source_state):
    """Explicit allowlist transfer. Rebuild EMA; this is NOT whole-model parity or full resume."""
    source = {n.removeprefix('agent.'):v for n,v in source_state.items()}
    required={'action_head.traj_head.4.mlp.6.weight','action_head.traj_head.4.mlp.6.bias'}
    if not required.issubset(source):
        raise ValueError('Not the declared V1 architecture: missing final physical trajectory head')
    target = v2.state_dict()
    copied,skipped,missing = [],[],[]
    coverage={}
    for name,value in target.items():
        allowed = name.startswith(('backbone.model.vision_model.','backbone.planning_register_adapter.register_',
            'backbone.planning_register_adapter.planning_registers','action_head.scorer.',
            'action_head.scorer_attention.','action_head.pos_embed.','action_head.attention.',
            'action_head.hist_encoding.','action_head.init_feature.'))
        old = name
        if name.startswith('action_head.attention.'):
            old = name.replace('action_head.attention.','action_head.trajectory_decoder.')
        if name.startswith('action_head.trajectory_head.'):
            old = name.replace('action_head.trajectory_head.','action_head.traj_head.4.')
            allowed = True
        module='.'.join(name.split('.')[:2])
        row=coverage.setdefault(module,dict(expected=0,copied=0,explicitly_new=0,excluded=0,unexpected_missing=[]))
        if allowed: row['expected']+=1
        if allowed and old in source:
            if not torch.is_tensor(value) or source[old].shape != value.shape:
                raise ValueError('Warm-start shape mismatch: '+name)
            target[name] = source[old].clone()
            copied.append((name,old))
            row['copied']+=1
        elif allowed:
            missing.append((name,old));row['unexpected_missing'].append(name)
        else:
            skipped.append(name);row['explicitly_new']+=1
    if missing:
        raise ValueError('Warm-start required compatible modules were not copied: '+str(missing))
    # V1 reads raw navigation/velocity/acceleration; V2 divides these input
    # columns by declared scales. Compensate W, keeping all nonzero ego outputs.
    hist='action_head.hist_encoding.weight'
    if hist in target:
        scale=torch.cat((target[hist].new_ones(3),v2.action_head.ego_normalizer.scales.to(target[hist])))
        target[hist]=target[hist]*scale[None]
    w,b = 'action_head.trajectory_head.mlp.6.weight','action_head.trajectory_head.mlp.6.bias'
    if any(n == w for n,_ in copied):
        target[w],target[b] = convert_physical_output_head(target[w],target[b],v2.action_head.normalizer)
    # Preserve new topology, never swallow missing/unexpected keys with strict=False.
    v2.load_state_dict(target,strict=True)
    if v2.world_model_enabled:
        from .ema import FP32MasterEMA
        v2.ema_teacher = FP32MasterEMA(v2.backbone,v2.config['total_steps'],v2.config['global_batch'])
    unused_source=sorted(set(source)-{old for _,old in copied})
    coverage['excluded_source']=dict(expected=0,copied=0,explicitly_new=0,excluded=len(unused_source),unexpected_missing=[])
    return dict(copied=copied,coverage=coverage,newly_initialized=skipped,not_migrated_source_keys=unused_source,
                hist_encoding_input_columns_rescaled=True,
                legacy_bf16_ema_history_unrecoverable=True,
                whole_model_parity_claimed=False)


def load_declared_warm_start(agent,path):
    """Consume the migration artifact explicitly; its old EMA schedule is NOT resumed."""
    artifact=torch.load(path,map_location='cpu',weights_only=False)
    if artifact.get('schema')!='planreg_v2_declared_warm_start_v2':
        raise ValueError('Run the audited V1→V2 migration first')
    permitted={'total_steps','global_batch','shared_init_path'}
    if {k:v for k,v in artifact['config'].items() if k not in permitted} != {
            k:v for k,v in agent.config.items() if k not in permitted}:
        raise ValueError('Warm-start architecture/normalizer/config mismatch')
    state=artifact['model']
    expected=agent.state_dict()
    if set(state)!=set(expected):raise ValueError('Warm-start state topology mismatch')
    # Only these declared schedule buffers differ. All actual module tensors load strictly.
    for name in ('optimizer_updates','ema_teacher.updates','ema_teacher.total_steps','ema_teacher.global_batch'):
        if name in state:state[name]=expected[name]
    agent.load_state_dict(state,strict=True)
    if agent.world_model_enabled:
        from .ema import FP32MasterEMA
        agent.ema_teacher=FP32MasterEMA(agent.backbone,agent.config['total_steps'],agent.config['global_batch'])
    return dict(artifact['audit'],artifact_sha256=file_sha256(path),
                initialization_kind='declared_v1_warm_start_not_resume',teacher_rebuilt=True)
