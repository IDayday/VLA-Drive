from copy import deepcopy
import json
import numpy as np
import torch
from torch import nn
from iqe.losses import LossTerm
from iqe.training.trainer import train
from iqe.specialist_study.models import EMAExpert
from iqe.specialist_study.objectives import SceneSampler


def test_optimizer_boundary_exact_resume_includes_private_ema_and_rng(tmp_path):
    torch.set_num_threads(1);torch.manual_seed(17)
    initial=nn.Sequential(nn.Linear(2,4),nn.Dropout(.2),nn.Linear(4,2))
    cfg={'lr':1e-3,'weight_decay':0.,'global_batch_size':4,'micro_batch_size':2,'max_optimizer_steps':6,
         'warmup_steps':2,'lr_min_ratio':.1,'grad_clip_norm':1.,'validation_every_steps':3,'checkpoint_every_steps':3}
    deps={'data':'same','architecture':'tiny_fixture'}
    def fetch(entries,dev):
        ids=torch.tensor([int(e['scene_id']) for e in entries],dtype=torch.float32,device=dev)
        return {'x':torch.stack((ids/10,ids*0+1),-1),'y':torch.stack((ids/2,ids/4),-1)}
    def loss(model,b):return {'il':LossTerm((model(b['x'])-b['y']).square().mean(-1).sum(),torch.tensor(len(b['x']),device=b['x'].device).float())}
    def run(folder,stop=None,resume=None):
        model=EMAExpert(deepcopy(initial));sampler=SceneSampler([str(i) for i in range(10)],21,'manifest')
        train(model,loss,fetch,sampler,cfg,deps,folder,seed=31,stop_after=stop,resume=resume,
              denominator_function=lambda b:{'il':torch.tensor(len(b['x'])).float()},after_optimizer_step=lambda m,s:m.update())
        return model,sampler
    continuous,cs=run(tmp_path/'continuous')
    run(tmp_path/'interrupted',stop=3)
    restored,rs=run(tmp_path/'interrupted',resume='auto')
    assert all(torch.equal(x,y) for x,y in zip(continuous.state_dict().values(),restored.state_dict().values()))
    assert cs.state_dict()==rs.state_dict()
    assert cs.peek(4)==rs.peek(4)
    a=torch.load(tmp_path/'continuous/step_000006.pt',weights_only=False)
    b=torch.load(tmp_path/'interrupted/step_000006.pt',weights_only=False)
    assert a['optimizer']['param_groups']==b['optimizer']['param_groups']
    for name,values in a['optimizer']['state'].items():
        for key,value in values.items():
            assert torch.equal(value,b['optimizer']['state'][name][key])
    # Completed result/receipt can be repaired without an extra EMA update or sample.
    result=json.loads((tmp_path/'interrupted/result.json').read_text());assert result['optimizer_steps']==6
    _,again=run(tmp_path/'interrupted',resume='auto')
    assert again.state_dict()==rs.state_dict()
