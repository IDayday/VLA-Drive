"""Independent recipe identities; never weaken the legacy full-head contracts."""
import json
from pathlib import Path
import socket
import copy

from starVLA.rl.safe_flow_improvement.contracts import file_sha, fingerprint, seed_for, SFT_SHA256

UPSTREAM_COMMIT = '1433ee7e06a62bd8b89da74455df0d82bda7b726'
SFT_EXPORT = '/mnt/project/DriveDreamer-Policy-flow-grpo/artifacts/action-only-checkpoints-v1/frozen_visual'
POPULATION = '/mnt/project/flow-safe-improvement-20261013/population/manifest.json'
RAW_CONDITION = '/mnt/project/constrained-flow-noise-20261014/condition_cache'
CONDITION_SHA = '80a2dd4b417a3154cbbf575044daea586d945b54e014eee22269d7bd48d39e0b'
ROOT = '/mnt/project/simwam-recipe-20261015'
CACHE_ROOTS = ['/mnt/project/flow-safe-improvement-20261013/metric_cache_v1',
               '/mnt/project/constrained-flow-noise-20261014/metric_cache_v1']
SAFETY25_VARIANT = 'R_CANONICAL_X1_SAFETY25'
SAFETY25_DATA_ORDER = 'fixed SFT single-seed NC/DAC/TTC strata; 6 safe + 2 unsafe per global8; rotating ranks; independent source epoch permutations'
SAFETY25_SAMPLING = dict(schema='fixed_SFT_safety_6plus2_v1', safe_groups=6,
    unsafe_groups=2, rotate_ranks=True, stratum_permutation_seed=42,
    reference_metrics=['no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound'],
    reference_pass_value=1., reference_population='same fixed hard TRAIN manifest',
    importance_correction=False)


def registered_stochastic_steps(variant):
    if variant == 'R_CANONICAL_LAST_ODE':
        return [6, 7, 8]
    if variant in ('R_SOURCE', 'R_CANONICAL', 'R_CANONICAL_X1', SAFETY25_VARIANT):
        return [7, 8, 9]
    raise ValueError('Unknown registered sampler variant: ' + str(variant))


LASTDET_SCHEMA = 'final_deterministic_steps78_v1'


def registered_stochastic_steps_for_recipe(cfg):
    """Narrow World W2 registration; all historical variant rules stay intact."""
    research=cfg.get('research',{})
    if research.get('arm')=='PPO_LASTDET' or research.get('sampler_ablation') is not None:
        sample=cfg.get('grpo',{}).get('sample',{})
        expected=dict(group_size=8,num_inference_steps=10,sde_mode='rigorous',noise_level=.1,
            train_sde_steps=2,sde_exclude_last_step=True,sde_step_indices=[7,8],
            merge_sde_window=False,final_action_clip=None,min_std=.001,randn_clip=5.,
            sample_noise_std=.1,logprob_noise_std=None,anneal_noise=True,
            logprob_clamp_mode='relative',logprob_clamp_width=7.)
        common_bad=(research.get('sampler_ablation')!=LASTDET_SCHEMA
                or research.get('policy_objective')!='ppo'
                or cfg.get('variant')!='R_CANONICAL' or cfg.get('model_family')!='QwenWorldPostCoT'
                or cfg.get('prediction_type')!='x0' or cfg.get('adaptation')!='ATTN_LORA_MEMORY'
                or cfg.get('action_shape')!=[8,3]
                or any(research.get(k) is not None for k in ('advantage_normalization',
                    'reward_transform','anchor_ablation','memory_lr_ablation','preference_auxiliary','noise_level_ablation'))
                or any(sample.get(k)!=v for k,v in expected.items()))
        lastdet=(research.get('arm')=='PPO_LASTDET' and research.get('data_policy')=='uniform_full'
                 and research.get('structured_advantage') is None and research.get('rescue_auxiliary') is None)
        rescue_aux=dict(schema='world_verified_positive_CFM_teacher_aux_v1',lambda_low=1.,lambda_high=.2,
                        sigma_min=.2,sigma_max=.8,stateless_seed=42)
        rescue=(research.get('arm') in ('PPO_RESCUE_REPLAY','PPO_RESCUE_CFM')
                and research.get('data_policy')=='rescue_mixture'
                and research.get('rescue_distribution')=='full80_low10_high10_static_v1'
                and research.get('structured_advantage')=='candidate_metric_safety_feasible_EP_global_std_v1'
                and isinstance(research.get('rescue_manifest'),str)
                and isinstance(research.get('rescue_manifest_sha256'),str)
                and len(research['rescue_manifest_sha256'])==64
                and research.get('rescue_auxiliary')==(rescue_aux if research['arm']=='PPO_RESCUE_CFM' else None))
        if common_bad or not (lastdet or rescue):
            raise ValueError('W2[7,8] requires the exact independently registered World PPO_LASTDET or Rescue recipe')
        return [7,8]
    return registered_stochastic_steps(cfg['variant'])


def require_dispatch():
    p = json.loads(Path('/mnt/project/server_dispatch_policy.json').read_text())
    host = socket.gethostname().removesuffix('-worker-0')
    allowed = p['task_authorizations']['flow_safe_improvement_20261013']['allowed_hosts']
    if host not in allowed or host not in p['allowed_hosts']:
        raise PermissionError('Host not authorized for SimWAM transfer: '+host)
    if p.get('global_dispatch_prohibitions',{}).get(host,{}).get('new_task_dispatch_allowed') is False:
        raise PermissionError('Host explicitly withdrawn: '+host)
    return host


def load_recipe(path):
    from omegaconf import OmegaConf
    cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if cfg['variant'] == SAFETY25_VARIANT:
        base_path = Path(__file__).resolve().parents[3] / 'configs/flow_rl/simwam_canonical_x1_ddp.yaml'
        expected = copy.deepcopy(load_recipe(base_path))
        expected.update(variant=SAFETY25_VARIANT, data_order=SAFETY25_DATA_ORDER,
                        scene_sampling=copy.deepcopy(SAFETY25_SAMPLING))
        if cfg != expected:
            raise ValueError('SAFETY25 permits only the registered static TRAIN exposure change relative to X1')
        return cfg
    if cfg['variant'] == 'R_CANONICAL_LAST_ODE':
        # A separate single-window follow-up; the source and X1 contracts stay
        # immutable. The existing upstream kernel, not a new transition, is used.
        base_path = Path(__file__).resolve().parents[3] / 'configs/flow_rl/simwam_canonical_ddp.yaml'
        expected = copy.deepcopy(load_recipe(base_path))
        expected['variant'] = 'R_CANONICAL_LAST_ODE'
        expected['grpo']['sample']['sde_exclude_last_step'] = True
        if cfg != expected:
            raise ValueError('LAST_ODE permits only the registered stochastic-window change')
        return cfg
    if cfg['variant'] == 'R_CANONICAL_X1':
        # Separately authorized follow-up. Validate against the entire locked
        # canonical recipe; never relax either original experiment's contract.
        base_path = Path(__file__).resolve().parents[3] / 'configs/flow_rl/simwam_canonical_ddp.yaml'
        expected = copy.deepcopy(load_recipe(base_path))
        expected['variant'] = 'R_CANONICAL_X1'
        expected['grpo']['train']['kl_dim_weights'] = [1., 1., .5, .5]
        if cfg != expected:
            raise ValueError('X1 permits only the registered longitudinal anchor change')
        return cfg
    if cfg['variant'] not in ('R_SOURCE','R_CANONICAL'):
        raise ValueError('Only the two registered variants are formal')
    # Full resolved configuration, including formerly unchecked optimizer/LoRA/
    # numerical/budget fields. Only the registered reward and name may differ.
    normalized=copy.deepcopy(cfg)
    expected_progress=10. if cfg['variant']=='R_SOURCE' else 5.
    if cfg['grpo']['reward']['progress_weight']!=expected_progress:
        raise ValueError('Reward does not match the registered variant')
    normalized['variant']='R_SOURCE';normalized['grpo']['reward']['progress_weight']=10.
    if fingerprint(normalized)!='28c9ef5a16a1a70bb4970085f3ca5ead37a6633dab6a464f65a273d7b62b8fbb':
        raise ValueError('Resolved recipe differs from the complete immutable 12b8e0d contract')
    g = cfg['grpo']; s = g['sample']; t = g['train']
    expected = dict(group_size=8,num_inference_steps=10,train_sde_steps=3,sde_mode='rigorous',
                    noise_level=.1,min_std=.001,randn_clip=5.,logprob_clamp_mode='relative',
                    logprob_clamp_width=7.,final_action_clip=None,sde_exclude_last_step=False,
                    sde_step_indices=None,merge_sde_window=False)
    if any(s.get(k)!=v for k,v in expected.items()): raise ValueError('Changed source sampler contract')
    if (t['num_inner_epochs']!=4 or t['use_bc_loss'] or t['kl_type']!='x' or t['kl_beta']!=50
        or t['ppo_clip_range']!=.005 or t['kl_dim_weights']!=[.2,1,.5,.5]
        or not t['adv_global_std'] or t.get('kl_gate',{}).get('enabled')):
        raise ValueError('Changed source update contract')
    if cfg['seed']!=42 or cfg['evaluation_seed']!=2101 or cfg['gradient_accumulation_steps']!=1:
        raise ValueError('Changed seed/optimizer boundary contract')
    return cfg


def recipe_identity(cfg, hard_manifest, projected_identity):
    return dict(schema='ddp_simwam_recipe_v1',recipe=cfg,upstream_commit=UPSTREAM_COMMIT,
                SFT_sha256=SFT_SHA256,hard_manifest_sha256=file_sha(hard_manifest),
                projected_condition_identity=projected_identity,
                policy_ratio='source relative-clamped dimension-mean stochastic-transition surrogate',
                deployment='original merged DDP; uniform native and shift5 ODE reported separately')
