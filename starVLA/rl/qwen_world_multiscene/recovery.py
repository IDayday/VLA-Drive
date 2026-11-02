"""Original complete native semantics plus genuine packed sampling provenance."""
import copy
import torch
from starVLA.rl.qwen_world_performance.recovery import validate_boundary as validate_base,validate_ranks as validate_base_ranks
from starVLA.rl.qwen_world_safety_exploration.retention_material import digest
from .recipe import validate_recipe,base_validation_recipe


def validate_packs(window,execution):
    samples=window['samples'];size=execution['micro_scenes'];g=execution['candidate_group'];packs=window.get('packed_groups',[])
    if (window.get('packed_execution')!=execution or len(samples)%size or len(packs)*size!=len(samples)):
        raise ValueError('Pending collection packed layout/identity changed')
    previous=None
    for number,material in enumerate(packs):
        leaves=samples[number*size:(number+1)*size];tokens=[s['token']for s in leaves]
        if (material['pack_index']!=number or material['tokens']!=tokens or material['scenes']!=size or material['G']!=g
                or material['steps']!=[7,8,9]):raise ValueError('Packed ordinal, group or source steps changed')
        for j,leaf in enumerate(leaves):
            if leaf['packed_position']!=dict(pack_index=number,scene_ordinal=j,pack_tokens=tokens,scenes=size):
                raise ValueError('A leaf was reordered or assigned another packed kernel')
        chain=torch.cat([s['chain'].cpu()for s in leaves]);old=torch.cat([s['old_logp'].cpu()for s in leaves])
        audit={key:(leaves[0]['old_probability'][key].cpu()if key=='std'else torch.cat([s['old_probability'][key].cpu()for s in leaves]))
            for key,value in leaves[0]['old_probability'].items()if torch.is_tensor(value)}
        if (digest(chain)!=material['chain_sha256']or digest(audit)!=material['old_probability_sha256']
                or not torch.equal(old,material['old_logp'].cpu())or not torch.equal(audit['mean'],material['sampled_mean'].cpu())
                or not torch.equal(audit['std'],material['sampled_std'].cpu())):
            raise ValueError('Stored packed sampler means/probabilities/chain changed')
        means=material['sampled_mean'];std=material['sampled_std']
        if means.shape!=(size*g,3,8,3)or std.shape!=(3,)or means.requires_grad or std.requires_grad:
            raise ValueError('Packed detached probability material shape changed')
        before,after=material['generator_before'],material['generator_after']
        if previous is not None and not torch.equal(previous,before):raise ValueError('Packed RNG sequence was reordered or skipped')
        rng=torch.Generator().set_state(before.cpu());shape=(size*g,8,3)
        root=torch.randn(shape,generator=rng,dtype=torch.float32)
        if not torch.equal(root,chain[:,0]):raise ValueError('Packed actual independent root draw changed')
        for j,step in enumerate((7,8,9)):
            eps=torch.randn(shape,generator=rng,dtype=torch.float32).clamp(-5,5)
            want=means[:,j].cpu()+std[j].cpu()*eps
            if not torch.allclose(want,chain[:,step+1],atol=1e-6,rtol=8*torch.finfo(torch.float32).eps):
                raise ValueError('Packed actual source innovation/mean endpoint changed')
        if not torch.equal(rng.get_state(),after.cpu()):raise ValueError('Packed native generator state changed')
        noise=material['noise_audit']
        if len(noise)!=4 or any(x['shape']!=list(shape)or x['dtype']!='torch.float32'or x['count']!=size*g*24 for x in noise):
            raise ValueError('Not four genuine packed source RNG calls')
        previous=after
    return dict(packed_groups=len(packs),packed_scene_count=size,packed_execution=copy.deepcopy(execution))


def validate_boundary(state,window,identity,optimizer=None,scheduler=None,**kwargs):
    cfg=validate_recipe(identity['recipe']);execution=cfg['research']['packed_execution']
    if identity.get('packed_execution')!=execution:raise ValueError('Native identity lacks exact packed numeric protocol')
    #The original native validator checks unchanged math/data/Adam/scheduler.
    #Only its single-scene-execution recipe gate is projected; the actual cfg
    #is validated above and remains unmodified in identity/checkpoint.
    projected=dict(identity,recipe=base_validation_recipe(cfg))
    summary=validate_base(state,window,projected,optimizer,scheduler,**kwargs)
    summary['packed_execution']=copy.deepcopy(execution)
    if window is not None:summary.update(validate_packs(window,execution))
    return summary


def validate_ranks(summaries):
    validate_base_ranks(summaries)
    if any(s.get('packed_execution')!=summaries[0].get('packed_execution')for s in summaries):
        raise ValueError('Ranks disagree on coherent packed execution')
    return summaries
