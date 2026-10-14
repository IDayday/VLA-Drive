from types import SimpleNamespace
import numpy as np
import torch
from iqe.specialist_study.scorer_training import batch_fetch,CONTEXT_KEYS


class FrozenTestCache:
    def batch(self, scenes, device):
        n=len(scenes)
        return {'world':torch.zeros(n,2,3),'action':torch.zeros(n,1,3),'current_ego':torch.zeros(n,4),
                'world_valid':torch.ones(n,2,dtype=torch.bool),'action_valid':torch.ones(n,1,dtype=torch.bool),
                'target_raw':torch.ones(n,8,4)*99}


def bank(k):
    return {'trajectories':np.arange(2*k*8*3,dtype=np.float32).reshape(2,k,8,3),
            'scores':np.tile(np.arange(k,dtype=np.float32)/k,(2,1)),
            'valid':np.ones((2,k),bool),'components':np.ones((2,k,6),np.float32),
            'component_valid':np.ones((2,k,6),bool)}


def test_natural_vs_comparison_candidates_and_label_whitelist():
    scenes=[SimpleNamespace(scene_id='a'),SimpleNamespace(scene_id='b')]
    deploy=bank(3);rich=bank(5)
    fetch=batch_fetch(None,scenes,FrozenTestCache(),None,deploy,rich,True)
    entries=[{'scene_id':'a','phase':2,'comparison':False,'natural':True,'pair_i':0,'pair_j':1},
             {'scene_id':'b','phase':2,'comparison':True,'natural':False,'pair_i':1,'pair_j':2}]
    batch=fetch(entries,'cpu')
    assert set(batch['context'])==set(CONTEXT_KEYS)
    assert torch.equal(batch['scores'][0],torch.from_numpy(deploy['scores'][0]))
    assert torch.equal(batch['scores'][1,:2],torch.from_numpy(deploy['scores'][1,[1,2]]))
    assert batch['valid'][1].tolist()==[True,True,False]
    assert batch['natural'].tolist()==[True,False] and batch['comparison'].tolist()==[False,True]
    phase1=[dict(entries[0],phase=1)]
    assert fetch(phase1,'cpu')['scores'].shape==(1,5)
