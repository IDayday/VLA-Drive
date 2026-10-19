import json
from pathlib import Path
import torch
from grpo_math import transition, log_prob, advantage, objective

torch.manual_seed(7)
x=torch.randn(4,8,4,dtype=torch.float64); v=torch.randn_like(x)
mean,std=transition(x,v,3,eta=0)
torch.testing.assert_close(mean,x.float()+.1*v.float())
mean,std=transition(x,v,3)
y=mean+std*torch.randn_like(mean)
lp=log_prob(y,mean,std)
oracle=torch.distributions.Independent(torch.distributions.Normal(mean,std),2).log_prob(y)
torch.testing.assert_close(lp,oracle)
adv,s=advantage(torch.tensor([.1,.2,.6,.9]))
assert abs(float(adv.mean()))<1e-6 and float(s)>0
loss,ratio,kl=objective(lp,lp,adv,mean,mean,std)
assert torch.equal(ratio,torch.ones_like(ratio)) and torch.equal(kl,torch.zeros_like(kl))
zero,s=advantage(torch.ones(4)); assert torch.equal(zero,torch.zeros_like(zero)) and s==0
# The likelihood-ratio gradient increases likelihood for positive advantages.
variable=lp.detach().clone().requires_grad_()
loss,_,_=objective(variable,lp,adv,mean,mean,std,beta=0); loss.backward()
torch.testing.assert_close(variable.grad,-adv/4)
# Detached-context gradient accumulation must equal joint end-to-end differentiation.
w=torch.randn(3,5,requires_grad=True); z=torch.randn(2,3)
context=z@w; (context.square().mean()+context.sin().mean()).backward(); direct=w.grad.clone(); w.grad=None
context=z@w; leaf=context.detach().requires_grad_()
leaf.square().mean().backward(); leaf.sin().mean().backward(); context.backward(leaf.grad)
torch.testing.assert_close(w.grad,direct)
for step in range(10):
    m,s=transition(x,v,step); assert torch.isfinite(m).all() and s>0
out={'status':'PASS','checks':['eta0 Euler','joint Gaussian log probability','identity ratio and zero reference KL','zero-variance advantages','policy gradient sign','detached context exact chain rule','finite nonzero variance at all10steps']}
Path(__file__).with_name('unit_checks.json').write_text(json.dumps(out,indent=2)+'\n')
print(out)
