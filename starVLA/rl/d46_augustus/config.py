"""Explicit AUGUSTUS-inspired three-step driving recipes; deployment stays K10."""
import copy
from pathlib import Path
import yaml
from starVLA.rl.d44_native_rl.config import defaults46,validate as legacy_validate
from starVLA.rl.qwen_world_compare.contracts import fingerprint

SCHEMA='d46_augustus_native_v1'
EXPERIMENTS={'AUG1_FULL':'full_text_action','AUG2_LORA':'lora_text_action'}
PROTOCOL=dict(schema='d46_augustus_three_step_policy_v1',training_steps=3,
    training_sigma_grid=[1.,2/3,1/3,0.],noise_scale=[.05,.05,.02],
    noise='official_g_a_sqrt_t',coordinate_reduction='mean24',
    reward='canonical_PDMS_raw_0_1',advantage='canonical_group_center_sample_std',
    advantage_epsilon=1e-6,group_std_unbiased=True,second_global_normalization=False,
    clip=.2,fixed_reference_beta=.001,initialization='fresh_verified_IL46_or_own_native_resume',
    deployment=dict(steps=10,shift=5.,seed=2101,training_deployment_grid_mismatch=True))
GRPO_SOURCE='https://raw.githubusercontent.com/volcengine/verl/main/verl/trainer/ppo/core_algos.py'


def defaults(experiment_id,*,global_scenes=None,inner_updates=1,registration_path=None):
    if experiment_id not in EXPERIMENTS:raise ValueError('Only two explicit AUGUSTUS driving experiments')
    cfg=defaults46('D0',registration_path=registration_path)
    cfg.update(experiment_id=experiment_id,global_scenes=global_scenes or(128 if experiment_id=='AUG1_FULL'else 64),
        group=8,inner_updates=inner_updates,learning_rate=1e-6,clip=.2,beta=.001,target_epochs=6.,maximum_epochs=20.,checkpoint_policy='all_off',
        film_last_layers=0 if experiment_id=='AUG1_FULL'else 8,train_action_io=True)
    cfg['augustus']=copy.deepcopy(PROTOCOL)|dict(scope=EXPERIMENTS[experiment_id],execution=dict(fast_prefix=True,rotary_cache=True))
    return validate(cfg)


def validate(cfg,*,parent_checks=True):
    if not isinstance(cfg,dict)or cfg.get('experiment_id')not in EXPERIMENTS:raise ValueError('Unknown AUGUSTUS experiment')
    exp=cfg['experiment_id'];execution=cfg.get('augustus',{}).get('execution',{})
    expected=copy.deepcopy(PROTOCOL)|dict(scope=EXPERIMENTS[exp],execution=execution)
    if cfg.get('augustus')!=expected or set(execution)!={'fast_prefix','rotary_cache'}or any(type(v)is not bool for v in execution.values()):
        raise ValueError('Changed AUGUSTUS three-step/normalization/execution protocol')
    if (cfg.get('schema')!='d44_native_rl_v1'or cfg.get('arm')!='D0'or cfg.get('parent')!='D_IL46K'
        or type(cfg.get('global_scenes'))is not int or cfg.get('global_scenes')not in(64,128)or type(cfg.get('group'))is not int or cfg.get('group')!=8
        or type(cfg.get('inner_updates'))is not int or cfg.get('inner_updates')not in(1,2)
        or cfg.get('learning_rate')!=1e-6 or cfg.get('beta')!=.001 or cfg.get('clip')!=.2
        or cfg.get('text_rank')!=16 or cfg.get('film_last_layers')!=(0 if exp=='AUG1_FULL'else 8)
        or cfg.get('train_action_io')is not True or cfg.get('post_feedback')is not False
        or cfg.get('seed')!=42 or cfg.get('evaluation_seed')!=2101 or cfg.get('training_uses_navtest')is not False
        or(cfg.get('steps'),cfg.get('shift'),cfg.get('stochastic_steps'),cfg.get('action_shape'))!=(10,5.,[7,8,9],[8,3])
        or(cfg.get('warmup_epochs'),cfg.get('hold_until_epoch'),cfg.get('cosine_end_epoch'),cfg.get('min_lr_ratio'))!=(.08,.60,4.,.05)
        or not 0<cfg.get('target_epochs',0)<=cfg.get('maximum_epochs',0)==20):
        raise ValueError('Changed AUGUSTUS IL46/data/scope/Adam/scene schedule or deployment contract')
    if parent_checks:
        base=copy.deepcopy(cfg);base['inner_updates']=2  # Legacy admitsM2; never rewrite saved M1 identity.
        legacy_validate(base)
    return cfg


def read(path):return validate(yaml.safe_load(Path(path).read_text()))


def kernel_identity(cfg):
    return dict(schema=SCHEMA,config_sha256=fingerprint(cfg),protocol_sha256=fingerprint(cfg['augustus']))
