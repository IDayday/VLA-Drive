"""Balanced cross-call role scheduler with an independent, resumable RNG."""
import torch


def xy_point_valid(feature_valid):
    if feature_valid.dtype!=torch.bool or feature_valid.ndim!=4 or feature_valid.shape[-1]!=4:raise ValueError('Expected bool B,A,T,4 label mask')
    return feature_valid[...,:2].all(-1)


class RoleScheduler:
    def __init__(self,seed=42):
        self.generator=torch.Generator().manual_seed(seed)
        self.position=0
    def roles(self,batch,device):
        if batch<1:raise ValueError('Empty role batch')
        # Alternating over examples, not reset at each batch. Odd/tail batches carry phase.
        roles=(torch.arange(self.position,self.position+batch)%2==1).to(device)
        self.position+=batch
        return roles
    def state_dict(self):return {'version':1,'position':self.position,'rng':self.generator.get_state()}
    def load_state_dict(self,state):
        if set(state)!={'version','position','rng'} or state['version']!=1 or not isinstance(state['position'],int) or state['position']<0:raise ValueError('Invalid role scheduler state')
        self.position=state['position'];self.generator.set_state(state['rng'])


def role_completion_mask(feature_valid,active,scheduler,neighbor_role=None):
    points=xy_point_valid(feature_valid)
    if active.shape!=feature_valid.shape[:2] or active.dtype!=torch.bool:raise ValueError('Role mask shape mismatch')
    if not isinstance(scheduler,RoleScheduler):raise ValueError('Use the resumable cross-call RoleScheduler')
    eligible=points.any(-1)&active;b,a=active.shape
    if not eligible[:,0].all():raise ValueError('A role task requires at least one valid ego xy point')
    if neighbor_role is None:neighbor_role=scheduler.roles(b,active.device)
    elif neighbor_role.shape!=(b,) or neighbor_role.dtype!=torch.bool:raise ValueError('Invalid requested roles')
    hidden=torch.zeros_like(active);chosen=torch.zeros(b,device=active.device,dtype=torch.long);fallback=torch.zeros(b,device=active.device,dtype=torch.bool)
    for i in range(b):
        candidates=torch.where(eligible[i,1:])[0]+1
        if bool(neighbor_role[i]) and len(candidates):chosen[i]=candidates[int(torch.randint(len(candidates),(1,),generator=scheduler.generator))]
        else:fallback[i]=bool(neighbor_role[i])
        hidden[i,chosen[i]]=True
    coordinates=feature_valid&hidden[:,:,None,None]
    if not (points&hidden[:,:,None]).flatten(1).any(-1).all():raise RuntimeError('Selected role has no complete xy position')
    actual=chosen>0
    return hidden,{'neighbor_requested':neighbor_role,'neighbor_actual':actual,'chosen_actor':chosen,'ego_only_fallback':fallback,
        'valid_hidden_coordinates':coordinates.sum((-1,-2,-3)),
        'ego_valid_coordinates':coordinates[:,0].sum((-1,-2)),
        'neighbor_valid_coordinates':coordinates[:,1:].sum((-1,-2,-3))}
