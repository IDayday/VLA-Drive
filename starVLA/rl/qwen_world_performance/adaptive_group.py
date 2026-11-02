"""One optional teacher-only G16; never enlarges on-policy PG/std material."""
import copy,time
import torch
from starVLA.model.modules.qwen_sourceflow.action_codec import ActionCodec
from starVLA.rl.simwam_recipe import checkpoint
from starVLA.rl.simwam_recipe.contracts import seed_for
from starVLA.rl.qwen_world_safety_exploration.retention_material import digest,sealed,verify_seal,validate_feedback
from .ode_target import strict_high

SCHEMA='world_performance_teacher_only_conditional_extra_G16_v1'
PROTOCOL=dict(schema=SCHEMA,main_G=16,extra_G=16,stochastic_steps=[7,8,9],
    trigger='nominalB1 notstrict-high AND no strict-high candidate in firstG16; gain is not part of trigger',
    role='teacher-only; never PG/rewardstd/advantages/exposure or best-of-G deployment',
    random='private statelessCPU generator; never consumes mainPG generator',max_extra_scored_candidates=16)


def needs_extra(rollout,nominal):
    if rollout.get('group_size')!=16 or len(rollout['records'])!=16:
        raise ValueError('Nominal study starts from actual primaryG16')
    return not strict_high(nominal['feedback']) and not any(strict_high(r)for r in rollout['records'])


def private_seed(rollout,rank):
    return seed_for(SCHEMA,42,rollout['behavior_version']['run'],rollout['behavior_version']['update'],
        rollout['behavior_version'].get('adapter_sha256'),rollout['token'],rank)


@torch.no_grad()
def collect_extra_if_needed(actor,rollout,nominal,service,version=None,*,rank=0):
    if version is not None and version!=rollout['behavior_version']:
        raise ValueError('Extra candidates must come from same current run/theta')
    if nominal['source_version']!=rollout['behavior_version'] or nominal['token']!=rollout['token']:
        raise ValueError('Foreign-run/version/scene B1 cannot control additional sampling')
    if not needs_extra(rollout,nominal):return None
    seed=private_seed(rollout,rank);generator=torch.Generator(device='cpu').manual_seed(seed)
    initial=generator.get_state().clone();audit=actor._probability_audit;noise_audit=actor._noise_audit;sampling=actor.sampling_terms
    actor._noise_audit=[]
    started=time.monotonic()
    try:
        chain,times,deltas,indices=actor.sample_action_chain(rollout['cond_bg'],8,generator=generator)
        noise=copy.deepcopy(actor._noise_audit)
    finally:
        actor._probability_audit=audit;actor._noise_audit=noise_audit;actor.sampling_terms=sampling
    if chain.shape!=(16,11,8,3)or indices.tolist()!=[7,8,9]or not torch.isfinite(chain).all():
        raise ValueError('Teacher extra must keep realG16/W3 source sampling')
    requested=ActionCodec.denorm_odo(chain[:,-1]).float().cpu();forward=time.monotonic()-started;started=time.monotonic()
    records=service.score((rollout['token'],rollout['log'],requested.numpy()))
    if len(records)!=16:raise ValueError('Score all original16 extra candidates, no replacements')
    for i,r in enumerate(records):validate_feedback(r,rollout['token'],rollout['log'],requested[i],service.protocol['sha256'])
    return sealed(dict(schema=SCHEMA,protocol=copy.deepcopy(PROTOCOL),token=rollout['token'],log=rollout['log'],
        source_version=copy.deepcopy(rollout['behavior_version']),rank=rank,seed=seed,
        condition=copy.deepcopy(rollout['cond_bg']),protocol_sha256=service.protocol['sha256'],
        chain=chain.detach().cpu(),timesteps=times.detach().cpu(),deltas=deltas.detach().cpu(),stoch_idx=indices.detach().cpu(),
        requested=requested,records=checkpoint.cpu_tree(records),noise_audit=checkpoint.cpu_tree(noise),
        generator_before=initial,generator_after=generator.get_state(),scored_candidates=16,PG_eligible=False,
        seconds=dict(forward=forward,scoring=time.monotonic()-started)), 'material_sha256')


def validate_extra(extra,rollout,nominal,*,rank):
    if (extra is not None)!=needs_extra(rollout,nominal):
        raise ValueError('ExtraG16 trigger changed; strict-high existence is independent of gain')
    if extra is None:return None
    verify_seal(extra,'material_sha256')
    if(extra.get('schema')!=SCHEMA or extra.get('protocol')!=PROTOCOL or extra['token']!=rollout['token']
       or extra['log']!=rollout['log']or extra['source_version']!=rollout['behavior_version']or extra['rank']!=rank
       or extra['condition']!=rollout['cond_bg']or extra['protocol_sha256']!=rollout['protocol_sha256']
       or extra['seed']!=private_seed(rollout,rank)or extra['scored_candidates']!=16 or extra['PG_eligible']is not False):
        raise ValueError('Teacher-only extraG16 ancestry/protocol changed')
    g=torch.Generator().manual_seed(extra['seed'])
    if not torch.equal(g.get_state(),extra['generator_before'].cpu()):raise ValueError('Extra privateRNG origin changed')
    root=torch.randn((16,8,3),generator=g,dtype=torch.float32)
    for _ in range(3):torch.randn((16,8,3),generator=g,dtype=torch.float32)
    if not torch.equal(g.get_state(),extra['generator_after'].cpu())or not torch.equal(extra['chain'][:,0].cpu(),root):
        raise ValueError('Extra privateRNG source draw order changed')
    if extra['chain'].shape!=(16,11,8,3)or extra['chain'].dtype!=torch.float32 or extra['chain'].requires_grad or not torch.isfinite(extra['chain']).all()or extra['stoch_idx'].tolist()!=[7,8,9]:
        raise ValueError('Detached originalG16/W3 extra chain required')
    for key in ('timesteps','deltas','stoch_idx'):
        if extra[key].requires_grad or not torch.equal(extra[key].cpu(),rollout[key].cpu()):
            raise ValueError('Extra kernel time/window differs from primary source sampling')
    if extra['requested'].shape!=(16,8,3)or len(extra['records'])!=16:raise ValueError('Teacher-only actual scoring inventory changed')
    decoded=ActionCodec.denorm_odo(extra['chain'][:,-1].cpu());actual=extra['requested'].cpu()
    bound=8*torch.finfo(torch.float32).eps*(decoded.abs()+decoded.new_tensor([66.74,42.,3.53]))
    if bool(((decoded-actual).abs()>bound).any()):raise ValueError('Extra endpoint differs from actually scored physical action')
    for i,r in enumerate(extra['records']):validate_feedback(r,rollout['token'],rollout['log'],actual[i],rollout['protocol_sha256'])
    return extra
