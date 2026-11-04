import copy
import torch
from starVLA.rl.d44_native_rl.update import optimizer_update


class Actor(torch.nn.Module):
    def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.tensor(1.5))
    def forward(self,row):
        loss=(self.weight-row['target']).square()
        return loss,{'loss':loss.detach(),'reference_KL_joint':loss.detach().double()}


class Scheduler:
    def __init__(self):self.last_epoch=0
    def step(self):self.last_epoch+=1


def test_native_microbatch_adam_clock_and_bulk_metrics_are_exact():
    cfg=dict(arm='D3_FPO',warmup_epochs=.08,hold_until_epoch=.6,cosine_end_epoch=4.,min_lr_ratio=.05)
    batch={'epoch':.5,'valid_scenes':2,'scenes':[dict(target=t,padding=False)for t in(0.,1.)]}
    results=[]
    for bulk in(False,True):
        actor=Actor();scheduler=Scheduler()
        optimizer=torch.optim.AdamW([dict(params=[actor.weight],name='text_lora',initial_lr=.001)],lr=.001,weight_decay=0.)
        row=optimizer_update(actor,actor,optimizer,scheduler,batch,cfg,.5,1,[actor.weight],bulk_metrics=bulk)
        assert scheduler.last_epoch==1 and int(optimizer.state[actor.weight]['step'])==1
        assert row['norm'].item()==2 and abs(actor.weight.item()-1.499)<1e-7
        results.append((actor.weight.detach().clone(),row['totals'],row['valid_kl_sum']))
    assert torch.equal(results[0][0],results[1][0])and results[0][1:]==results[1][1:]
