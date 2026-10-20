"""CPU engineering checks; not evidence about driving model quality."""
import copy,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
import engine
CFG=dict(vlm_lr=2e-7,action_lr=1e-6,beta=.01,eta=.1,clip=.2,gradient_clip=1.)
class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__();self.qwen_vl_interface=torch.nn.Linear(3,4);self.action_input_model=torch.nn.Linear(2,4);self.action_model=torch.nn.ModuleDict({'qwen_proj':torch.nn.Linear(4,4),'head':torch.nn.Linear(4,4)});self.register_buffer('sentinel',torch.tensor([3.]))
    def condition(self,x):return self.action_model['qwen_proj'](self.qwen_vl_interface(x[:3])+self.action_input_model(x[3:]))
    def velocity(self,c,x,step):return .01*self.action_model['head'](x+c[None,None,:])
def update(model,ref,opt,number):
    opt.zero_grad(set_to_none=True)
    for i in range(4):
        x=torch.linspace(-.2,.2,5)+i*.01;condition=lambda m:m.condition(x);velocity=lambda m,c,z,s:m.velocity(c,z,s);noise=engine.make_noise('cpu',engine.seed('train',1,number,i))
        def reward(raw):return torch.sigmoid(raw.mean((1,2))),{}
        engine.backward_scene(model,ref,condition,velocity,noise,reward,CFG,4)
    return engine.step(model,opt,1.)
class Tests(unittest.TestCase):
    def test_large_embedding_indices_are_integer_exact(self):
        for size in [1,3,151863*2048,152130*2048]:
            idx=engine.sample_indices(size,'cpu');self.assertEqual(int(idx[-1]),size-1);self.assertTrue(bool(((idx>=0)&(idx<size)).all()))

    def test_all_scopes_and_frozen_reference(self):
        for scope in engine.SCOPES:
            torch.manual_seed(19);m=Toy();ref=copy.deepcopy(m).requires_grad_(False);h=engine.state_hash(ref,'all');engine.scope(m,scope);frozen=engine.state_hash(m);before=engine.witnesses(m);opt=engine.optimizer(m,CFG);stats=update(m,ref,opt,0);change=engine.changes(before,engine.witnesses(m))
            required={'joint':['vlm','action'],'vlm_only':['vlm'],'dit_only':['action']}[scope]
            self.assertEqual(stats['gradient_norms']['history'],0.)
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in m.action_input_model.parameters()))
            optimizer_ids={id(p) for group in opt.param_groups for p in group['params']}
            self.assertTrue(all(id(p) not in optimizer_ids for p in m.action_input_model.parameters()))
            for key in required:self.assertGreater(stats['gradient_norms'][key],0);self.assertGreater(change[key],0)
            self.assertEqual(engine.state_hash(m),frozen);self.assertEqual(engine.state_hash(ref,'all'),h)
    def test_scope_union_and_forward_identity(self):
        torch.manual_seed(23);m=Toy();x=torch.ones(5);original=m.condition(x).detach();names={}
        for scope in engine.SCOPES:
            inventory=engine.scope(m,scope);names[scope]={p['name'] for p in inventory if p['trainable']}
            self.assertTrue(torch.equal(original,m.condition(x).detach()))
        self.assertEqual(names['joint'],names['vlm_only']|names['dit_only'])
        self.assertFalse(names['vlm_only']&names['dit_only'])
    def test_checkpoint_resume_next_update(self):
        torch.manual_seed(20);base=Toy();ref=copy.deepcopy(base).requires_grad_(False);a=copy.deepcopy(base);engine.scope(a,'joint');oa=engine.optimizer(a,CFG);update(a,ref,oa,0)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'checkpoint.pt';torch.save({'model':a.state_dict(),'optimizer':oa.state_dict()},p);update(a,ref,oa,1)
            b=copy.deepcopy(base);engine.scope(b,'joint');ob=engine.optimizer(b,CFG);saved=torch.load(p,weights_only=True);b.load_state_dict(saved['model']);ob.load_state_dict(saved['optimizer']);update(b,ref,ob,1)
            self.assertEqual(engine.state_hash(a,'all'),engine.state_hash(b,'all'))
    def test_zero_variance_keeps_kl_and_finite(self):
        torch.manual_seed(3);m=Toy();ref=copy.deepcopy(m).requires_grad_(False);engine.scope(m,'vlm_only')
        with torch.no_grad():m.qwen_vl_interface.weight.add_(.1)
        opt=engine.optimizer(m,CFG);opt.zero_grad();x=torch.ones(5)
        r=engine.backward_scene(m,ref,lambda p:p.condition(x),lambda p,c,z,s:p.velocity(c,z,s),engine.make_noise('cpu',9),lambda raw:([0.]*8,{}),CFG,1)
        stats=engine.step(m,opt,1.);self.assertEqual(r['reward_std'],0.);self.assertGreater(r['kl_mean'],0.);self.assertGreater(stats['gradient_norms']['vlm'],0.)
    def test_pairs_order_and_exploration(self):
        tokens=['a','b','c','d'];self.assertEqual(engine.order(tokens,4,0),engine.order(tokens[::-1],4,0));self.assertEqual(engine.noise_hash(engine.make_noise('cpu',7)),engine.noise_hash(engine.make_noise('cpu',7)))
        metrics=[{'PDMS':float(i==0),'NC':1,'DAC':1,'TTC':1} for i in range(8)];x=engine.exploration(np.zeros((8,8,3)),metrics,0.)
        for k in [1,2,4,8]:self.assertAlmostEqual(x['curves'][str(k)]['pass_at_k'],k/8)
        self.assertEqual(x['curves']['8']['mean_best'],1.)
if __name__=='__main__':torch.set_num_threads(1);unittest.main()
