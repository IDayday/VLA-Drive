"""Balanced independent resumable role sampling; complete actor masking."""
from collections import Counter
import torch


class TeacherMaskScheduler:
    def __init__(self, seed=42):
        self.rng=torch.Generator().manual_seed(seed); self.cursor=0; self.counts=Counter()

    def sample(self, active, point_valid):
        if active.dtype!=torch.bool or point_valid.dtype!=torch.bool or point_valid.shape[:2]!=active.shape:
            raise ValueError('Invalid teacher task masks')
        if (point_valid & ~active[...,None]).any(): raise ValueError('Inactive labels')
        targets=[]
        for a,v in zip(active.cpu(),point_valid.cpu()):
            want_neighbor=self.cursor%2==1;self.cursor+=1
            self.counts['nominal_neighbor' if want_neighbor else 'nominal_ego']+=1
            eligible=torch.where(a[1:] & v[1:].any(-1))[0]+1
            t=int(eligible[torch.randint(len(eligible),(1,),generator=self.rng)]) if want_neighbor and len(eligible) else 0
            self.counts['actual_neighbor' if t else 'actual_ego']+=1
            self.counts['ego_fallback']+=int(want_neighbor and not len(eligible))
            targets.append(t)
        target=torch.tensor(targets,device=active.device)
        visible=point_valid.clone()
        visible[torch.arange(len(active),device=active.device),target]=False
        self.counts['with_peer_future']+=int(visible.flatten(1).any(-1).sum())
        return target,visible

    def state_dict(self): return {'rng':self.rng.get_state(),'cursor':self.cursor,'counts':dict(self.counts)}
    def load_state_dict(self,state):
        self.rng.set_state(state['rng']);self.cursor=state['cursor'];self.counts=Counter(state['counts'])
