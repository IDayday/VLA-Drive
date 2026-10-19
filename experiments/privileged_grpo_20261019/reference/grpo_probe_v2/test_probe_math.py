"""CPU checks for accumulation scaling, parameter groups and geometric metrics."""
import json
from pathlib import Path
import numpy as np
import torch
from diversity import summarize

torch.manual_seed(23)
vlm=torch.nn.Linear(3,5);action=torch.nn.Linear(5,2)
inputs=torch.randn(4,8,3);targets=torch.randn(4,8,2)
parameters=list(vlm.parameters())+list(action.parameters())
loss=(action(vlm(inputs))-targets).square().mean();loss.backward()
joint=[p.grad.clone() for p in parameters]
for p in parameters:p.grad=None
for scene in range(4):
    # The same 10-step loss repeated: averaging steps must not change scale.
    for step in range(10):
        loss=(action(vlm(inputs[scene]))-targets[scene]).square().mean()
        (loss/40).backward()
for p,expected in zip(parameters,joint):torch.testing.assert_close(p.grad,expected,rtol=1e-5,atol=1e-7)
optimizer=torch.optim.AdamW([{'params':vlm.parameters(),'lr':2e-7},{'params':action.parameters(),'lr':1e-6}],weight_decay=0)
assert [g['lr'] for g in optimizer.param_groups]==[2e-7,1e-6]
assert len({id(p) for g in optimizer.param_groups for p in g['params']})==len(parameters)
metrics=[dict(PDMS=1.,NC=1.,DAC=1.,TTC=1.,EP=1.,CF=1.,DDC=1.) for _ in range(8)]
a=np.zeros((8,8,3));s=summarize(a,metrics)
assert s['pairwise_ADE_m']==s['pairwise_FDE_m']==0 and s['pair_count']==28
a[:,:,0]=np.arange(8)[:,None];s=summarize(a,metrics)
assert s['pairwise_ADE_m']==s['pairwise_FDE_m']==3
a[:,:,2]=np.pi-.01;a[0,:,2]=-np.pi+.01
s=summarize(a,metrics);assert s['pairwise_final_yaw_deg']<1
result={'status':'PASS','checks':['four-scene ten-step accumulated gradient equals joint mean','disjoint full optimizer groups with specified LR','28 pairwise distances analytic check','wrapped yaw does not report spurious 360-degree diversity']}
Path(__file__).with_name('UNIT_CHECKS.json').write_text(json.dumps(result,indent=2)+'\n')
print(result)
