"""Explicit exact-native Safety10k LR forks; source PPO/credit stays unchanged."""
from __future__ import annotations
import ast
import copy
import json
import math
from pathlib import Path

from starVLA.rl.qwen_world_postcot.contracts import file_sha, fingerprint
from starVLA.rl.simwam_recipe import checkpoint
from starVLA.rl.simwam_recipe.vendor.trainer_math import SimWAMGRPOTrainer
from .policy import ResearchLoss
from .safety_efficiency import (
    validate_safety_efficiency_recipe, validate_safety_efficiency_boundary,
    validate_safety_efficiency_ranks,
)

FORK_SCHEMA='world_Safety10k_exact_native_LR_fork_v1'
SCHEDULE_SCHEMA='world_native_fork_relative_LR_v1'
STATE_SCHEMA='world_native_fork_relative_LR_state_v1'
ARMS=('PPO_SAFETY_LRLOW','PPO_SAFETY_LRCOS','PPO_SAFETY_DATA','PPO_SAFETY_LRDATA')
MIX_ARMS=ARMS[2:]
MIX_MANIFEST=Path('/mnt/project/qwen-world-rl-optimization-20261030/data/manifests/RESCUE_80_10_10_STATIC_103288.json')
MIX_MANIFEST_SHA='5767f6fc52d8c9cef24750da69f2fa63cf6883279cb68299f015c3d2c34b0175'
GROUP_NAMES=('action_lora','scene_predictive_memory')
BASE_LRS=(5e-5,5e-6)
FORK_KEYS={'schema','checkpoint','expected_updates','checkpoint_files_sha256','checkpoint_identity_sha256',
           'optimizer_file_sha256','expected_parent_model_sha256','optimizer','rng','reference'}
INFERENCE_FILES=(
    'starVLA/rl/qwen_world_research_algorithms/policy.py',
    'starVLA/rl/qwen_world_research_algorithms/safety_efficiency.py',
    'starVLA/rl/qwen_world_postcot/scope_actor.py',
    'starVLA/rl/qwen_world_postcot/scope_conditions.py',
    'starVLA/rl/qwen_world_postcot/scope_runtime.py',
    'starVLA/rl/qwen_world_postcot/rl_actor.py',
    'starVLA/rl/simwam_recipe/vendor/action_math.py',
    'starVLA/rl/simwam_recipe/vendor/trainer_math.py',
    'starVLA/model/modules/qwen_world_postcot/flow.py',
)


def expected_schedule(arm):
    if arm not in ARMS:raise ValueError('Unregistered native LR fork')
    cosine=arm==ARMS[1];factor=.2 if arm in (ARMS[0],ARMS[3]) else 1.
    return dict(schema=SCHEDULE_SCHEMA,kind='cosine' if cosine else 'constant',start_updates=10000,end_updates=20000,
                start_factor=factor,end_factor=.1 if cosine else factor)


def validate_schedule(spec):
    if not any(spec==expected_schedule(arm) for arm in ARMS):raise ValueError('Unregistered relative LR schedule')
    return spec


def factor_at(spec,completed_updates):
    validate_schedule(spec)
    if type(completed_updates) is not int or completed_updates<0:raise ValueError('LR clock is a nonnegative global optimizer-update counter')
    if spec['kind']=='constant':return spec['start_factor']
    phase=min(1.,max(0.,(completed_updates-spec['start_updates'])/(spec['end_updates']-spec['start_updates'])))
    return spec['end_factor']+(spec['start_factor']-spec['end_factor'])*.5*(1+math.cos(math.pi*phase))


class RelativeSchedule:
    """NEXT-update LR from completed global updates; never resets Adam or RNG.

    No vendor shared absolute eta_min: each group keeps its own base LR.
    load_state_dict requires optimizer.load_state_dict first, as native restore
    already does, and refuses a saved optimizer/schedule LR disagreement.
    """
    def __init__(self,optimizer,spec):
        self.optimizer=optimizer;self.spec=copy.deepcopy(validate_schedule(spec));self.base_lrs=list(BASE_LRS)
        if [g.get('name') for g in optimizer.param_groups]!=list(GROUP_NAMES):raise ValueError('Exact action/memory optimizer groups required')
        for g,base in zip(optimizer.param_groups,self.base_lrs):
            if g.get('initial_lr',base)!=base:raise ValueError('Relative schedule base LR changed')
            g['initial_lr']=base
        self.last_epoch=0;self._last_lr=[];self.seek(0)
    def get_last_lr(self):return list(self._last_lr)
    def seek(self,completed_updates):
        factor=factor_at(self.spec,completed_updates);self.last_epoch=completed_updates
        self._last_lr=[base*factor for base in self.base_lrs]
        for group,lr in zip(self.optimizer.param_groups,self._last_lr):group['lr']=lr
    def step(self):self.seek(self.last_epoch+1)
    def state_dict(self):
        return dict(schema=STATE_SCHEMA,spec=copy.deepcopy(self.spec),base_lrs=list(self.base_lrs),group_names=list(GROUP_NAMES),last_epoch=self.last_epoch,_last_lr=self.get_last_lr())
    def load_state_dict(self,state):
        validate_schedule_state(state,self.spec,state.get('last_epoch'),self.optimizer)
        self.last_epoch=state['last_epoch'];self._last_lr=list(state['_last_lr'])


def validate_schedule_state(scheduler,spec,updates,optimizer=None):
    state=scheduler.state_dict() if hasattr(scheduler,'state_dict') else scheduler
    factor=factor_at(spec,updates);expected=[base*factor for base in BASE_LRS]
    want=dict(schema=STATE_SCHEMA,spec=spec,base_lrs=list(BASE_LRS),group_names=list(GROUP_NAMES),last_epoch=updates,_last_lr=expected)
    if state!=want:raise ValueError('Relative schedule identity/global counter/LR changed')
    if optimizer is not None:
        opt=optimizer.state_dict() if hasattr(optimizer,'state_dict') else optimizer
        groups=opt['param_groups']
        if len(groups)!=2 or any(g.get('name')!=name or g.get('lr')!=lr or g.get('initial_lr')!=base for g,name,lr,base in zip(groups,GROUP_NAMES,expected,BASE_LRS)):
            raise ValueError('Native optimizer group LR differs from its saved relative schedule')
    return expected


class ScheduledSafetyLoss(ResearchLoss):
    def __init__(self,cfg):super().__init__(cfg);self.relative_schedule=copy.deepcopy(cfg['research']['lr_schedule'])
    def _build_scheduler(self,scheduler_type,total_train_steps,warmup_steps=0):
        if scheduler_type!=self.relative_schedule['kind'] or total_train_steps!=50000 or warmup_steps!=100:
            raise ValueError('Native fork keeps parent config horizon/warmup metadata; relative global schedule takes over without rewarmup')
        return RelativeSchedule(self.optimizer,self.relative_schedule)


def validate_fork_recipe(cfg,parent_identity):
    arm=cfg.get('research',{}).get('arm');r=cfg['research'];spec=r.get('lr_schedule',{});f=r.get('native_fork',{})
    if arm not in ARMS or spec!=expected_schedule(arm):raise ValueError('Exact registered LR/data native fork recipe required')
    if (set(f)!=FORK_KEYS or f.get('schema')!=FORK_SCHEMA or f.get('expected_updates')!=10000
            or f.get('optimizer')!='inherited_AdamW' or f.get('rng')!='inherited_all_native_per_rank' or f.get('reference')!='original_frozen_IL70'):
        raise ValueError('Full inherited native Adam/RNG/IL70 provenance required')
    parent=parent_identity['recipe'];validate_safety_efficiency_recipe(parent)
    if (parent['research']['arm']!='PPO_SAFETY_EP' or parent['lr_scheduler_type']!='constant'
            or parent['max_steps']!=50000 or parent['lr_warmup_steps']!=100):raise ValueError('Only original SafetyEP constant10k ancestry is permitted')
    expected=copy.deepcopy(parent);expected['research'].update(arm=arm,lr_schedule=spec,native_fork=f);expected['lr_scheduler_type']=spec['kind']
    if arm in MIX_ARMS:
        expected['research'].update(data_policy='static_rescue_retention',manifest_sha256=MIX_MANIFEST_SHA,sampling_distribution='full80_low10_high10_static_v1')
    if cfg!=expected:raise ValueError('Native fork may change only registered LR fields and,for DATA roles,registered static sampling fields')
    validate_safety_efficiency_recipe(cfg);return cfg


def source_compatible(rel,parent,checkpoint_path):
    if file_sha(rel)==parent['source_files'].get(rel):return True
    if rel!='starVLA/rl/qwen_world_research_algorithms/safety_efficiency.py':return False
    archived=Path(checkpoint_path).parent.parent/'source'/rel
    if not archived.is_file() or file_sha(archived)!=parent['source_files'].get(rel):return False
    def math_ast(text):
        tree=ast.parse(text)
        # Root's reviewed recipe-only DATA admission gate is the only allowed
        # difference; all advantage/loss/boundary math must stay identical.
        tree.body=[n for n in tree.body if not isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) or n.name!='validate_safety_efficiency_recipe']
        return ast.dump(tree,include_attributes=False)
    return math_ast(Path(rel).read_text())==math_ast(archived.read_text())


def validate_static_population(cfg,args,parent_identity):
    if cfg['research']['arm'] not in MIX_ARMS:return None
    if Path(args.manifest).resolve()!=MIX_MANIFEST or file_sha(args.manifest)!=MIX_MANIFEST_SHA:raise ValueError('DATA role requires exact presealed80/10/10 stream')
    stream=json.loads(MIX_MANIFEST.read_text());body=dict(stream)
    if body.pop('sha256',None)!=fingerprint(body):raise ValueError('Changed static scene stream seal')
    bank=Path(stream['teacher_manifest'])
    if file_sha(bank)!=stream['teacher_manifest_file_sha256']:raise ValueError('Static census population binding changed')
    bank_meta=json.loads(bank.read_text());full=Path(bank_meta['full_population_manifest'])
    if file_sha(full)!=bank_meta['full_population_manifest_sha256'] or file_sha(full)!=parent_identity['recipe']['research']['manifest_sha256']:
        raise ValueError('Static DATA eligible population differs from original full103k')
    population=json.loads(full.read_text());tokens=population['tokens']
    navtest=Path(args.manifest).parent.parent/'panels/NAVTEST_012146.json';test=json.loads(navtest.read_text())
    if (stream['schema']!='world_rl_research_scene_stream_v1' or stream['mode']!='adaptive' or stream['seed']!=42
            or stream['stage_exposures']!=10000 or len(tokens)!=103288 or len(set(tokens))!=103288 or stream['tokens']!=tokens
            or stream['data_manifest_sha256']!=population['data_manifest_sha256'] or bank_meta['full_population_count']!=103288
            or set(tokens)&set(test['tokens']) or len(test['tokens'])!=12146):raise ValueError('DATA stream must retain all103288 TRAIN-only eligible scenes without NAVTEST tokens')
    return dict(stream_manifest=str(MIX_MANIFEST),stream_sha256=MIX_MANIFEST_SHA,full_population_manifest=str(full),
        full_population_manifest_sha256=bank_meta['full_population_manifest_sha256'],full_population_count=103288,
        NAVTEST_tokens_in_population=0,source_stage_at_parent_cursor=2,parent_cursor=20000,
        adaptive_reader_mechanical_only=True,static80_10_10=True,teacher_CFM_used=False,PG_source='fresh8 candidate canonical SafetyEP only')


class SafetyLRRuntimeMixin:
    """Stack before SafetyEfficiencyRuntimeMixin and ResearchRuntime."""
    def load_recipe(self,args):
        cfg=super().load_recipe(args);f=cfg['research'].get('native_fork',{});cp=Path(f.get('checkpoint','')).resolve()
        seal=json.loads((cp/'COMPLETE').read_text());files=json.loads((cp/'files.json').read_text());parent=json.loads((cp/'identity.json').read_text())
        validate_fork_recipe(cfg,parent)
        if (seal.get('world')!=8 or seal['updates']!=10000 or seal['next_inner']!=0 or seal['phase']!='READY'
                or seal['files_sha256']!=f['checkpoint_files_sha256'] or file_sha(cp/'files.json')!=seal['files_sha256']
                or files['identity.json']!=f['checkpoint_identity_sha256'] or file_sha(cp/'identity.json')!=f['checkpoint_identity_sha256']
                or files['optimizer.pt']!=f['optimizer_file_sha256'] or parent['purpose']!='formal'
                or parent['IL_parent']['model_sha256']!=self.parent['model_sha256'] or f['expected_parent_model_sha256']!=self.parent['model_sha256']):
            raise ValueError('Native fork parent seal/identity/complete8rank10k/IL70 binding changed')
        if Path(args.output).resolve()==cp.parent.parent:raise ValueError('Native fork requires a new runroot, preserving the original parent')
        if any(not source_compatible(rel,parent,cp) for rel in INFERENCE_FILES):raise ValueError('Native fork must retain parent model,credit,probability and optimizer grouping source')
        self.static_population_provenance=validate_static_population(cfg,args,parent)
        self.native_fork_checkpoint=cp;self.native_parent_identity=parent
        self.native_fork_provenance=dict(copy.deepcopy(f),checkpoint=str(cp),initial_global_updates=10000,phase_reset=False,weights_only=False,
            optimizer_state_inherited=True,RNG_state_inherited=True,stream_cursor_inherited=True,reference_reset=False,
            scheduler_replacement=copy.deepcopy(cfg['research']['lr_schedule']),execution_migration=False,
            sampling_distribution_changed=cfg['research']['arm'] in MIX_ARMS,static_population=copy.deepcopy(self.static_population_provenance),
            interpretation=('scientific exact-native LR/data combination fork,new identity; inherit full Adam/moments/RNG/globalcursor from sealed Safety10k,then apply registered LR and static80/10/10 sampling' if cfg['research']['arm'] in MIX_ARMS else 'scientific LR-only exact-native fork,new identity; inherit full Adam/moments/RNG/cursor from sealed Safety10k,then change only two group LRs'))
        return cfg
    def make_actor(self,cfg,args,device):
        actor,oldcfg,sft=super().make_actor(cfg,args,device);actor._loss=ScheduledSafetyLoss(cfg);return actor,oldcfg,sft
    def initialize_training_state(self,actor,optimizer,scheduler,generator,identity):
        """Called only when not --resume; native restore validates old ancestry."""
        if identity['parameter_contract']!=self.native_parent_identity['parameter_contract']:
            raise ValueError('Native fork changed trainable parameter closure')
        if identity['IL_parent']['model_sha256']!=self.native_parent_identity['IL_parent']['model_sha256']:
            raise ValueError('Native fork reference ancestry changed')
        original=type('OriginalSchedulerOwner',(),dict(optimizer=optimizer,learning_rate=5e-5))()
        parent_scheduler=SimWAMGRPOTrainer._build_scheduler(original,'constant',50000,100)
        state,rollout=checkpoint.restore(self.native_fork_checkpoint,actor,optimizer,parent_scheduler,generator,self.native_parent_identity,
            boundary_validator=validate_safety_efficiency_boundary,rank_validator=validate_safety_efficiency_ranks)
        if (state['updates']!=10000 or state['phase']!='READY' or state['next_inner']!=0 or rollout is not None
                or state['collections']!=2500 or state['scene_cursor']!=20000):raise ValueError('Only parent10k READY without pending rollout may seed this fork')
        scheduler.seek(state['updates'])
        # Pure metadata validation; no forward/no RNG draw/no optimizer step.
        self.validate_boundary(state,rollout,identity,optimizer,scheduler)
        return state,rollout
    def validate_boundary(self,state,rollout,identity,optimizer=None,scheduler=None,*,legacy=False):
        if legacy:raise ValueError('New native LR fork never inherits an unversioned recovery boundary')
        cfg=identity['recipe'];spec=cfg['research']['lr_schedule']
        validate_fork_recipe(cfg,self.native_parent_identity)
        if state['updates']<10000:raise ValueError('Native fork global counter must never reset below10k')
        summary=validate_safety_efficiency_boundary(state,rollout,identity,optimizer,scheduler,legacy=False)
        if scheduler is not None:validate_schedule_state(scheduler,spec,state['updates'],optimizer)
        summary.update(relative_LR_schema=STATE_SCHEMA,relative_LR_spec=copy.deepcopy(spec),relative_LR_base=list(BASE_LRS),
            native_fork_parent_files_sha256=cfg['research']['native_fork']['checkpoint_files_sha256'])
        return summary
    @staticmethod
    def validate_ranks(summaries):
        validate_safety_efficiency_ranks(summaries)
        keys=('relative_LR_schema','relative_LR_spec','relative_LR_base','native_fork_parent_files_sha256')
        if any({k:s[k] for k in keys}!={k:summaries[0][k] for k in keys} for s in summaries):raise ValueError('Ranks disagree on native LR schedule/fork provenance')
        if any(s['relative_LR_schema']!=STATE_SCHEMA or s['relative_LR_base']!=list(BASE_LRS) for s in summaries):raise ValueError('Relative schedule rank protocol changed')
        return summaries
    def recipe_identity(self,cfg,args,condition_sha):
        value=super().recipe_identity(cfg,args,condition_sha);value.update(native_fork_provenance=copy.deepcopy(self.native_fork_provenance),
            relative_LR_protocol=dict(schema=SCHEDULE_SCHEMA,schedule=copy.deepcopy(cfg['research']['lr_schedule']),
                base_lrs=list(BASE_LRS),groups=list(GROUP_NAMES),clock='completed global optimizer updates; next-update LR',
                no_rewarmup=True,Adam_moments_inherited=True,ratio_action_to_memory=10.))
        return value
