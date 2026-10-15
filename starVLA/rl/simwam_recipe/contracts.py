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
