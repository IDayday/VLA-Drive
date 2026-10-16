"""Independent new-base contracts; historical SimWAM/C/N contracts stay strict."""
import json
from pathlib import Path
import socket
from starVLA.rl.simwam_recipe.contracts import fingerprint, file_sha, SFT_SHA256, CONDITION_SHA, POPULATION

ROOT = Path('/mnt/project/fm-transfer-20261016')
TASK = 'ddp_simwam_fm_transfer_20261016'
CONFIGS = {
    'A_DDP_TIME': dict(architecture='ddp', initialization='original_F_SFT_step100000',
        max_updates=20000, learning_rate=1e-5, milestones=[0,5000,10000,20000]),
    'B_SOURCE_ACTION_DIT': dict(architecture='source_action_dit', initialization='source_random_initialization_seed42',
        max_updates=100000, learning_rate=1e-4, milestones=[0,10000,25000,50000,75000,100000])}


def config(variant):
    if variant not in CONFIGS:raise ValueError(variant)
    return dict(schema='fm_transfer_SFT_v1',variant=variant,**CONFIGS[variant],seed=42,
        evaluation_seed=2101,global_batch=32,per_rank_batch=4,world=8,FM_repeats=8,
        time_sampling='source continuous uniform quantile shifted5',
        objective='source time-weighted velocity MSE',train_shift=5.,
        action_encoding='unchanged DDP eight-by-four XY standardized plus sin/cos',
        betas=[.9,.95],eps=1e-8,weight_decay=.01,gradient_clip=1.,warmup_fraction=.03,
        scheduler='warmup_cosine_min0.1',parameter_storage='float32',
        operator_autocast='bfloat16',loss_statistics='float32',optimizer_states='float32',
        upstream_frozen=True,condition_boundary='before qwen_proj',save_every=2000,
        inference_grids=['native_uniform10','shift5_10'],
        ddp=dict(find_unused_parameters=True,static_graph=False,bucket_cap_mb=25,broadcast_buffers=False))


def require_dispatch():
    policy=json.loads(Path('/mnt/project/server_dispatch_policy.json').read_text())
    host=socket.gethostname().removesuffix('-worker-0')
    if (host not in policy['allowed_hosts'] or host in policy.get('global_dispatch_prohibitions',{}) or
        host not in policy['task_authorizations'][TASK]['allowed_hosts']):
        raise PermissionError('Host not authorized for new FM transfer: '+host)
    return host


def source_files():
    root=Path(__file__).resolve().parents[3]
    files=list((root/'starVLA/rl/fm_transfer').rglob('*.py'))+list((root/'scripts/analysis/fm_transfer').glob('*.py'))
    relatives=['starVLA/rl/simwam_recipe/vendor/scheduler_continuous.py',
        'starVLA/rl/simwam_recipe/checkpoint.py','starVLA/rl/constrained_flow_noise/loading.py',
        'starVLA/rl/constrained_flow_noise/condition_cache.py',
        'starVLA/model/modules/action_model/GR00T_ActionHeader.py',
        'starVLA/model/modules/action_model/flow_matching_head/action_encoder.py',
        'starVLA/model/modules/action_model/flow_matching_head/cross_attention_dit.py',
        'starVLA/rl/flow_grpo/config.py','starVLA/rl/flow_grpo/data.py',
        'starVLA/rl/flow_grpo/reproducibility.py','starVLA/rl/flow_grpo/transactions.py']
    files.extend(root/p for p in relatives if (root/p).exists())
    return {str(p.relative_to(root)):file_sha(p) for p in sorted(set(files))}
