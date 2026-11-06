"""Explicit AUGUSTUS-inspired three-step driving recipes; deployment stays K10."""
import copy
from pathlib import Path
import yaml
from starVLA.rl.d44_native_rl.config import defaults46,validate as legacy_validate
from starVLA.rl.qwen_world_compare.contracts import fingerprint

SCHEMA='d46_augustus_native_v1'
EXPERIMENTS={'AUG1_FULL':'full_text_action','AUG2_LORA':'lora_text_action'}
EXPERIMENTS.update(AUG3_FULL_ALIGNED='full_text_action',AUG4_LORA_ALIGNED='lora_text_action')
# A separate identity: never change the root distribution of an archived run.
EXPERIMENTS['AUG5_LORA_TOKEN_ROOT']='lora_text_action'
EXPERIMENTS['AUG6_NATIVE_CALIBRATED']='lora_text_action'
EXPERIMENTS['AUG7_NATIVE_FP32']='lora_text_action'
EXPERIMENTS['AUG8_NATIVE_KL001']='lora_text_action'
EXPERIMENTS['AUG9_SPLIT_ACTION']='lora_text_action'
PROTOCOL=dict(schema='d46_augustus_three_step_policy_v1',training_steps=3,
    training_sigma_grid=[1.,2/3,1/3,0.],noise_scale=[.05,.05,.02],
    noise='official_g_a_sqrt_t',coordinate_reduction='mean24',
    reward='canonical_PDMS_raw_0_1',advantage='canonical_group_center_sample_std',
    advantage_epsilon=1e-6,group_std_unbiased=True,second_global_normalization=False,
    clip=.2,fixed_reference_beta=.001,initialization='fresh_verified_IL46_or_own_native_resume',
    deployment=dict(steps=10,shift=5.,seed=2101,training_deployment_grid_mismatch=True))
GRPO_SOURCE='https://raw.githubusercontent.com/volcengine/verl/main/verl/trainer/ppo/core_algos.py'


def calibrated(experiment):return experiment in ('AUG6_NATIVE_CALIBRATED','AUG7_NATIVE_FP32','AUG8_NATIVE_KL001','AUG9_SPLIT_ACTION')


def official_split(cfg):return cfg['experiment_id']=='AUG9_SPLIT_ACTION'


def aligned(experiment):return experiment in ('AUG3_FULL_ALIGNED','AUG4_LORA_ALIGNED','AUG5_LORA_TOKEN_ROOT') or calibrated(experiment)


def profile(experiment):
    value=copy.deepcopy(PROTOCOL)
    if aligned(experiment):
        # Exact FP32 native deployment grid, NOT a three-step approximation.
        import torch
        from starVLA.model.modules.qwen_world_postcot.flow import shift_sigma
        grid=shift_sigma(torch.linspace(1,0,11,dtype=torch.float32),5.).tolist()
        value.update(schema='d46_augustus_deployment_suffix_v2',training_sigma_grid=grid[7:],
            deterministic_prefix_grid=grid[:8],initial_root_seed=2101,
            reward='canonical_PDMS_scoring_with_hard_safety_EP_credit',
            advantage='SafetyEP_group_center_sample_std',fixed_reference_beta=.1,
            deployment=dict(steps=10,shift=5.,seed=2101,training_deployment_grid_mismatch=False),
            correction='local D46 adaptation; official JaT kernel on native final3 transitions')
    if experiment=='AUG5_LORA_TOKEN_ROOT' or calibrated(experiment):
        value.update(schema='d46_augustus_deployment_token_root_v3',
                     initial_root_scheme='deployment_token_seed_v1')
    if calibrated(experiment):
        import json
        from .physical_noise import calibrated_scale
        from starVLA.rl.qwen_world_compare.contracts import file_sha
        path=Path(__file__).resolve().parents[3]/'reports/d46_augustus_regression_20261103/results/TRAIN103K_NOISE_CALIBRATION.json'
        calibration=json.loads(path.read_text())
        if calibration['count']!=103288 or calibration['NAVTEST_used'] or calibration['data']['NAVTEST_overlap']:
            raise ValueError('Noise calibration must use the full legal TRAIN only')
        scale,_=calibrated_scale(calibration['q01'],calibration['q99'])
        value.update(schema='d46_augustus_native_calibrated_v4',noise_scale=scale.tolist(),
            noise_calibration_sha256=file_sha(path),action_output_precision='float32_linear',
            advantage='SafetyEP_group_center_global_population_std_cap5',
            group_std_unbiased=False,second_global_normalization=False,
            correction='local JaT adaptation: deployment root, physical TRAIN covariance, FP32 output, global SafetyEP')
    if experiment in ('AUG7_NATIVE_FP32','AUG8_NATIVE_KL001','AUG9_SPLIT_ACTION'):
        value.update(schema='d46_augustus_native_text_FP32_v5',text_action_precision='float32',
            correction='AUG6 numerical correction: native text C/P and action denoiser FP32; frozen visual encoding unchanged')
    if experiment=='AUG8_NATIVE_KL001':
        value.update(schema='d46_augustus_native_text_FP32_kl001_v6',fixed_reference_beta=.001,
            correction='AUG7 independent same-IL46 run; only reference KL coefficient changed .1 to .001 after actual full-batch gradient replay')
    if experiment=='AUG9_SPLIT_ACTION':
        value.update(schema='d46_augustus_released_split_action_v7',
            reward='released_stage1_5PDMS_reverse_gate_1cm',
            advantage='released_split_group_population_std_no_epsilon',
            advantage_epsilon=0.,group_std_unbiased=False,group_std_zero_threshold=1e-6,
            correction='Released Stage-I action reward and split normalization; D sampling/precision/LoRA unchanged; retain local beta0.1 after low-KL safety regression; no CoT reproduction')
    return value


def defaults(experiment_id,*,global_scenes=None,inner_updates=None,registration_path=None):
    if experiment_id not in EXPERIMENTS:raise ValueError('Unknown explicit AUGUSTUS driving experiment')
    cfg=defaults46('D0',registration_path=registration_path)
    full=EXPERIMENTS[experiment_id]=='full_text_action';v2=aligned(experiment_id)
    cfg.update(experiment_id=experiment_id,global_scenes=global_scenes or(128 if full else 64),
        group=16 if calibrated(experiment_id)else 8,inner_updates=inner_updates if inner_updates is not None else (2 if v2 else 1),
        learning_rate=2e-7 if v2 and full else 1e-6,clip=.2,beta=profile(experiment_id)['fixed_reference_beta'],
        target_epochs=6.,maximum_epochs=20.,checkpoint_policy='all_off',
        film_last_layers=0 if full else 8,train_action_io=True)
    cfg['augustus']=profile(experiment_id)|dict(scope=EXPERIMENTS[experiment_id],execution=dict(fast_prefix=True,rotary_cache=True))
    if cfg['augustus'].get('text_action_precision')=='float32':cfg['dtype']='FP32_parameters_text_action_states_probability_BF16_frozen_visual'
    return validate(cfg)


def validate(cfg,*,parent_checks=True):
    if not isinstance(cfg,dict)or cfg.get('experiment_id')not in EXPERIMENTS:raise ValueError('Unknown AUGUSTUS experiment')
    exp=cfg['experiment_id'];execution=cfg.get('augustus',{}).get('execution',{})
    if profile(exp).get('text_action_precision')=='float32' and cfg.get('dtype')!='FP32_parameters_text_action_states_probability_BF16_frozen_visual':
        raise ValueError('Text/action precision must match scientific dtype metadata')
    v2=aligned(exp);full=EXPERIMENTS[exp]=='full_text_action'
    expected=profile(exp)|dict(scope=EXPERIMENTS[exp],execution=execution)
    if cfg.get('augustus')!=expected or set(execution)!={'fast_prefix','rotary_cache'}or any(type(v)is not bool for v in execution.values()):
        raise ValueError('Changed AUGUSTUS three-step/normalization/execution protocol')
    if (cfg.get('schema')!='d44_native_rl_v1'or cfg.get('arm')!='D0'or cfg.get('parent')!='D_IL46K'
        or type(cfg.get('global_scenes'))is not int or cfg.get('global_scenes')not in(64,128)or type(cfg.get('group'))is not int or cfg.get('group')!=(16 if calibrated(exp)else 8)
        or type(cfg.get('inner_updates'))is not int or cfg.get('inner_updates')not in(1,2)
        or cfg.get('learning_rate')!=(2e-7 if v2 and full else 1e-6) or cfg.get('beta')!=expected['fixed_reference_beta'] or cfg.get('clip')!=.2
        or (v2 and cfg.get('inner_updates')!=2)
        or cfg.get('text_rank')!=16 or cfg.get('film_last_layers')!=(0 if full else 8)
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
