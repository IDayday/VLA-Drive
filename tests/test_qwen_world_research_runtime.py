import copy
import pytest
from starVLA.rl.qwen_world_postcot.scope_contracts import MEMORY, expected_scope_recipe
from starVLA.rl.qwen_world_research.runtime import validate_recipe, SCHEMA, CAMPAIGN


def config():
    cfg=expected_scope_recipe(MEMORY)
    cfg['research']=dict(schema=SCHEMA,campaign_root=str(CAMPAIGN),policy_objective='ppo',data_policy='uniform_full')
    return cfg


def test_distinguishes_algorithm_family_without_relaxing_original_math():
    cfg=config();assert validate_recipe(cfg)==cfg
    cfg['research']['policy_objective']='cispo';assert validate_recipe(cfg)==cfg
    cfg['research']['policy_objective']='nft'
    with pytest.raises(ValueError):validate_recipe(cfg)
    cfg['grpo']['train']['kl_beta']=0;assert validate_recipe(cfg)==cfg


@pytest.mark.parametrize('key,value',[('adaptation','TIME_MOD_LORA'),('seed',99),('world_size',4),('action_shape',[8,4])])
def test_rejects_unregistered_scope_or_action(key,value):
    cfg=config();cfg[key]=value
    with pytest.raises(ValueError):validate_recipe(cfg)


def test_cispo_does_not_silently_change_sampler_reward_or_anchor():
    cfg=config();cfg['research']['policy_objective']='cispo'
    for key,value in [('kl_beta',0),('ppo_clip_range',.2),('num_inner_epochs',2)]:
        altered=copy.deepcopy(cfg);altered['grpo']['train'][key]=value
        with pytest.raises(ValueError):validate_recipe(altered)


def test_lower_rigorous_noise_requires_its_own_single_factor_registration():
    cfg=config();cfg['grpo']['sample']['noise_level']=.03
    with pytest.raises(ValueError,match='inherited scientific change'):validate_recipe(cfg)
    cfg['research'].update(arm='PPO_NOISE03',noise_level_ablation='rigorous_eta003_v1')
    assert validate_recipe(cfg)==cfg
    assert cfg['grpo']['sample']['sample_noise_std']==.1
    for field in ('reward_transform','advantage_normalization','anchor_ablation','memory_lr_ablation','preference_auxiliary'):
        changed=copy.deepcopy(cfg);changed['research'][field]='unregistered combination'
        with pytest.raises(ValueError):validate_recipe(changed)


def test_final_deterministic_recipe_keeps_eta_and_the_first_two_exploration_steps():
    cfg=config()
    cfg['research'].update(arm='PPO_LASTDET',sampler_ablation='final_deterministic_steps78_v1')
    cfg['grpo']['sample'].update(train_sde_steps=2,sde_exclude_last_step=True,sde_step_indices=[7,8])
    assert validate_recipe(cfg)==cfg
    assert cfg['grpo']['sample']['noise_level']==.1
    changed=copy.deepcopy(cfg);changed['grpo']['sample']['sde_step_indices']=[6,7,8]
    with pytest.raises(ValueError):validate_recipe(changed)
    changed=copy.deepcopy(cfg);changed['research']['noise_level_ablation']='rigorous_eta003_v1'
    with pytest.raises(ValueError):validate_recipe(changed)
