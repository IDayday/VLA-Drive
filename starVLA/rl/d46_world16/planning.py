"""Keep 64 independent scene groups while repartitioning execution.

These helpers DO NOT publish a resumable checkpoint or relax the native world
checks. New runtime/worker admission and actual distributed qualification are
still required before use on a running experiment.
"""
import copy
import torch
from starVLA.rl.d44_native_rl.recovery import Boundary,material_digest
from starVLA.rl.qwen_world_compare.contracts import fingerprint

SCHEMA='d46_B64G16M2_world8_to16_qualification_plan_v1'
FIXED_DRAW_ARMS=('D0','D1','D3_FPO','D3_AWM','D5')


def rank_slots(rank,world):
    """Preserve eight original RNG lanes: rank r+8 owns old local slots4..7."""
    if world not in(8,16)or not 0<=rank<world:raise ValueError('Only world8/16')
    lane=rank%8;start=0 if world==8 else(rank//8)*4
    size=8 if world==8 else 4
    return tuple((start+j)*8+lane for j in range(size))


def validate_recipe(cfg):
    if cfg.get('arm')not in FIXED_DRAW_ARMS+('D2',)or cfg.get('parent')!='D_IL46K':
        raise ValueError('Only six registered IL46 policy routes, not fresh aux or D6')
    if (cfg.get('global_scenes'),cfg.get('group'),cfg.get('inner_updates'))!=(64,16,2):
        raise ValueError('Must preserve B64/G16/M2')
    if (cfg.get('steps'),cfg.get('shift'),cfg.get('stochastic_steps'),cfg.get('innovation_clip'),cfg.get('probability_layout'))!=(10,5.,[7,8,9],None,'sequential'):
        raise ValueError('Only the exact existing source W3 kernel')
    if cfg.get('seed')!=42 or cfg.get('evaluation_seed')!=2101:raise ValueError('Seed changed')
    return cfg


def _burn_scene(generator,arm,mc):
    if arm not in FIXED_DRAW_ARMS:raise ValueError('D2 teacher branches have data-dependent draws; cannot skip them')
    # Actual ActionGRPOMixin: one [16,8,3] root plus three independent innovations.
    for _ in range(4):torch.randn((16,8,3),generator=generator,dtype=torch.float32)
    if arm in('D3_FPO','D3_AWM'):
        torch.rand((mc,16),generator=generator)
        torch.randn((mc,16,8,3),generator=generator,dtype=torch.float32)


def fixed_scene_rng_plan(before,arm,*,mc=4):
    """Preview states ONLY; never consumes global RNG or changes old draws."""
    if mc!=4:raise ValueError('Only registered MC4 native variants')
    g=torch.Generator(device='cpu');g.set_state(before)
    starts=[];ends=[]
    for _ in range(8):
        starts.append(g.get_state().clone());_burn_scene(g,arm,mc);ends.append(g.get_state().clone())
    return dict(starts=starts,ends=ends,next_virtual_lane_state=g.get_state().clone(),
        schema='d46_fixed_draw_virtual8_sampler_lane_v1',arm=arm,mc_samples=mc)


def assert_scene_rng_end(generator,expected):
    if not torch.equal(generator.get_state(),expected):raise ValueError('Actual scene RNG consumption differs; do not advance or rewrite old noise')


def reshard_pending_for_qualification(payloads,identity):
    """In-memory, detached material redistribution; explicitly NOT native output.

    Optimizer/adapters/scheduler are replicated, unsharded common payload and
    remain byte-for-byte untouched. The old eight final sampler streams are
    retained separately as virtual lanes; copying them to16 ranks alone is NOT
    a valid future sampler implementation.
    """
    validate_recipe(identity['recipe'])
    if identity.get('world')!=8 or len(payloads)!=8:raise ValueError('Requires complete old eight-rank material')
    keys=('updates','collections','scene_cursor','next_inner','phase','valid_scene_exposures','raw_scene_exposures','new_candidates','valid_PG_candidates','extra_B1_queries','adaptive_G32_queries','padding_scene_exposures','adaptive_kl','evaluation_gates')
    shared={k:payloads[0]['state'][k]for k in keys}
    old_summaries=[]
    for own in payloads:
        if own.get('identity_sha256')!=identity['sha256']or {k:own['state'][k]for k in keys}!=shared:
            raise ValueError('Rank ancestry/counter mismatch')
        if own['rollout']is None:raise ValueError('This helper specifically exercises pending material; READY needs no leaf reshard')
        old_summaries.append(Boundary.validate_boundary(own['state'],own['rollout'],identity))
    Boundary.validate_ranks(old_summaries)
    new_identity=copy.deepcopy(identity);new_identity.pop('sha256',None);new_identity['world']=16
    new_identity['world_reshard_qualification']=dict(schema=SCHEMA,parent_identity_sha256=identity['sha256'],resumable_native_checkpoint=False)
    new_identity['sha256']=fingerprint(new_identity)
    rank_plans=[];summaries=[]
    for rank in range(16):
        lane=rank%8;start=(rank//8)*4;source=payloads[lane]
        batch=copy.deepcopy(source['rollout']);batch['scenes']=batch['scenes'][start:start+4];batch['digest']=material_digest(batch)
        plan=dict(schema='d46_non_native_rank_reshard_qualification_v1',rank=rank,source_rank=lane,
            source_local_indices=list(range(start,start+4)),global_slots=rank_slots(rank,16),state=copy.deepcopy(source['state']),rollout=batch,
            original_global_RNG_bundle=copy.deepcopy(source['rng']),source_identity_sha256=identity['sha256'])
        # Uses existing native pending-shape/count checks without weakening it.
        summaries.append(Boundary.validate_boundary(plan['state'],batch,new_identity));rank_plans.append(plan)
    Boundary.validate_ranks(summaries)
    return dict(schema=SCHEMA,resumable_native_checkpoint=False,qualification_identity=new_identity,ranks=rank_plans,
        virtual_sampler_RNG_states=[p['rng']['rollout_generator'].clone()for p in payloads],
        common_optimizer_adapters_scheduler='retain original shared optimizer.pt bytes; no optimizer reset or reshard',
        global_counts_unchanged=shared)


def specification(cfg):
    validate_recipe(cfg)
    return dict(schema=SCHEMA,status='CPU_PLAN_ONLY_NOT_RUNNABLE_WORLD16_TRAINER',B=64,G=16,M=2,world_old=8,world_new=16,
        local_scenes_old=8,local_scenes_new=4,candidate_forward_batch_unchanged=16,global_fresh_candidates=1024,
        rank_global_slots={str(r):rank_slots(r,16)for r in range(16)},
        objective='rank local sum(scene_loss*world/valid_scenes); DDP mean over world => sum(valid scene_loss)/valid_scenes',
        normalization='same 64x16 raw SafetyEP values, same padding exclusion, one FP64 population std; deterministic global-slot order recommended',
        sampler='fixed arm: retain original virtual8 RNG lanes and exact per-scene initial/end state; D2 requires old8-lane serial collection or explicit new RNG experiment',
        exactness='preserve theta/Adam/scheduler/dual/cursor/old material/RNG state; changed FP32 DDP accumulation/reduction order is NOT promised bitwise-equivalent to world8',
        checkpoint='new typed world-reshard lineage required; current native restore rejects changed world/identity/rank files correctly',
        deployment='same actual single B1 K10shift5 seed2101; deployment has no training world dimension')
