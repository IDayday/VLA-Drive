"""Explicit new campaign identities; existing live recipes remain immutable."""
import copy
import hashlib
import json
from pathlib import Path
from starVLA.rl.qwen_world_postcot.scope_contracts import expected_scope_recipe, MEMORY

ROOT = Path('/mnt/project/qwen-world-performance-rl-20261103')
SCHEMA = 'qwen_world_performance_rl_v1'
IL70_SHA256 = '7dda5b14b1ea67fc631b06c7fa336950fa07a0b84c969f4e714bdeb5b5436d40'
EVALUATION_EPOCHS = [.25, .5, 1, 2, 3, 4, 6, 8, 12, 20]


def validate_recipe(cfg):
    p = cfg['research'].get('performance', {})
    b, g, m = (p[k] for k in ('scenes_per_collection', 'group_size', 'inner_updates'))
    if p.get('schema') != SCHEMA or p.get('parent_model_sha256') != IL70_SHA256:
        raise ValueError('All independent algorithms must use registered complete IL70')
    if b not in (64, 128) or g not in (16, 32) or m not in (2, 4):
        raise ValueError('Unregistered B/G/M')
    if cfg['world_size'] != 8 or cfg['seed'] != 42 or cfg['adaptation'] != MEMORY:
        raise ValueError('World8, training seed42, trainable action+memory required')
    if cfg['research']['policy_objective'] != 'ppo':
        raise ValueError('The inherited constructor interface remains PPO; algorithm selected explicitly below')
    inherited = expected_scope_recipe(MEMORY)['grpo']
    wanted = copy.deepcopy(inherited)
    wanted['sample']['group_size'] = g
    wanted['train']['num_inner_epochs'] = m
    if cfg['grpo'] != wanted:
        raise ValueError('Changed source W3 chain, original anchor or native PPO conventions')
    if p['algorithm'] not in ('safetyep', 'nominal_rescue', 'awfm', 'fpo', 'temporal_rescue', 'performance_max'):
        raise ValueError('Unknown explicitly identified independent algorithm')
    if cfg['learning_rate'] not in (2e-5, 3e-5, 5e-5) or cfg['memory_learning_rate'] != cfg['learning_rate']/10:
        raise ValueError('Register bounded independent LR recipe')
    if p['population'] <= 0 or p['data_audit_sha256'] == 'PENDING':
        raise ValueError('Formal and qualification runs require real sealed data audit')
    audit_path=Path(p['data_audit'])
    if hashlib.sha256(audit_path.read_bytes()).hexdigest()!=p['data_audit_sha256']:
        raise ValueError('Registered data audit changed')
    audit=json.loads(audit_path.read_text());checks=audit['checks']
    if (audit['status']!='COMPLETE' or checks['unique_tokens']!=p['population']
            or checks['token_duplicates'] or checks['token_overlap'] or checks['physical_log_overlap']
            or checks['cache_missing_scenes'] or checks['error_counts']):
        raise ValueError('Training population has overlap, duplicates, missing cache or failed audit')
    if cfg['research'].get('NAVTEST_feedback_to_training') is not False:
        raise ValueError('No NAVTEST scene feedback into training')
    if p['initialization'] != 'same_IL70_fresh_Adam0' or p['cross_algorithm_state_transfer']:
        raise ValueError('No cross-algorithm weights or optimizer transfer')
    nominal=p['nominal_rescue']
    expected=dict(start_epoch=.5,anchor_budget_share=.05,min_improvement=.02,strict_high_min_pdms=.9,
        sigma_min=.1,sigma_max=.9,mc_samples=2,cfm_share=.75,ode_last_share=.25,
        calibration_slots=100,gradient_diagnostic_slots=500,adaptive_extra_G=True)
    if nominal!=expected:
        raise ValueError('Nominal helper uses this exact registered protocol; new values require a new implemented recipe')
    if p['microbatch_scenes']!=1:
        raise ValueError('Only real one-scene microbatch is implemented; do not relabel accumulation as a larger microbatch')
    return cfg
