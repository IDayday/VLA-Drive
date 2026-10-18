import copy,importlib.util,json,sys,unittest,weakref
from pathlib import Path
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('diagnose',HERE/'diagnose.py');d=importlib.util.module_from_spec(s);s.loader.exec_module(d)
B=json.loads((HERE/'BINDING.json').read_text());sys.path.insert(0,B['runtime']['path'])
import numpy as np
import torch
from torch import nn
from starVLA.model.modules.st_world.requests import sample_requests,gather_current
from starVLA.model.modules.st_world.sparse_readout import SparseFeatureReadout
ms=importlib.util.spec_from_file_location('metric',B['metric_function']['path']);metric=importlib.util.module_from_spec(ms);ms.loader.exec_module(metric)
class Tiny(nn.Module):
    def __init__(self,c):
        super().__init__();self.sparse_readout=SparseFeatureReadout(8,c,dim=8,heads=2,layers=1)
        self.register_buffer('world',torch.randn(1,3,8));self.base=SimpleNamespace(inference_fp32=True,amp=nullcontext)
        self.st_config=SimpleNamespace(teacher=SimpleNamespace(channels=c));self.encode_count=0
    def encode_inputs(self,observations):self.encode_count+=1;return {'W':self.world.expand(len(observations),-1,-1)}
    def forward(self,*_a,**_k):raise AssertionError('FM/full forward forbidden')
class Tests(unittest.TestCase):
    def test_actual_small_metadata(self):
        configs,regs,selection=d.preflight(B);self.assertEqual(len(selection['scenes']),8);self.assertEqual(selection['split'],'train')
        self.assertFalse(Path(B['output_root']).exists())
    def test_wrong_population_and_recipe(self):
        b=copy.deepcopy(B);b['max_scenes']=9
        with self.assertRaises(ValueError):d.preflight(b)
        cfg={k:d.read_pin(a['config']) for k,a in B['arms'].items()};cfg['256']['space_time']['world_queries_per_view']=96
        with self.assertRaises(ValueError):d.paired_configs(cfg)
        pin=dict(B['selection']);pin['sha256']='0'*64
        with self.assertRaises(ValueError):d.read_pin(pin)
    def test_actual_request_gather_and_metric(self):
        g=torch.Generator().manual_seed(4242);r=sample_requests(2,3,(18,24),96,patch_generator=g)
        z=torch.arange(2*3*18*24*384,dtype=torch.float32).reshape(2,3,18,24,384)/100
        mask=torch.ones(z.shape[:-1],dtype=torch.bool);mask[0,0]=False
        target,valid=gather_current(z,mask,r);ids=np.array([[[str(n)+'/'+str(v)+'/'+str(x) for x in r.patch_indices[n,v]]for v in range(3)]for n in range(2)])
        t=target.numpy();a=t[...,:256]+1;b=t.copy();b[...,:256]+=2;b[...,256:]+=3
        result=metric.paired_channel_errors(a,b,t,valid.numpy(),request_ids256=ids,request_ids384=ids,target_request_ids=ids)
        self.assertAlmostEqual(result['common_first256']['arm256']['mse'],1,places=3)
        self.assertAlmostEqual(result['extra_128']['arm384']['mse'],9,places=3)
        self.assertEqual(result['extra_128']['arm256'],'NOT_PREDICTED_NOT_ZERO_ERROR')
        bad=ids.copy();bad[0,0,0]='foreign'
        with self.assertRaises(ValueError):metric.paired_channel_errors(a,b,t,valid.numpy(),request_ids256=bad,request_ids384=ids,target_request_ids=ids)
    def test_original_PCA_common_prefix(self):
        from tools.st_world_policy.teacher_pca import TeacherPCA
        g=torch.Generator().manual_seed(9)
        t={'channel_mean':torch.randn(1024,generator=g),'channel_std':torch.rand(1024,generator=g)+.5,
            'pca_mean':torch.randn(1024,generator=g),'components':torch.randn(384,1024,generator=g)}
        x=torch.randn(3,18,24,1024,generator=g)
        a=TeacherPCA(t,dimension=256,identity='fixture',teacher_contract_identity='fixture').transform(x)
        b=TeacherPCA(t,dimension=384,identity='fixture',teacher_contract_identity='fixture').transform(x)
        self.assertTrue(torch.equal(a,b[...,:256]))
    def test_real_readout_only_eval_and_mask(self):
        m=Tiny(256).eval().requires_grad_(False);r=sample_requests(1,3,(18,24),4,patch_generator=torch.Generator().manual_seed(4))
        original={k:v.clone() for k,v in m.state_dict().items()}
        p=d.predict_current(m,[object()],r);self.assertEqual(p.shape,(1,3,4,256));self.assertEqual(m.encode_count,1)
        self.assertTrue(all(torch.equal(original[k],v) for k,v in m.state_dict().items()))
        m.train()
        with self.assertRaises(ValueError):d.predict_current(m,[object()],r)
    def test_sequential_strict_load_and_release(self):
        refs=[];calls=[]
        def load(reg,tag,*,device,strip):
            self.assertFalse(any(r() is not None for r in refs));self.assertFalse(strip);self.assertEqual(device,'cuda:0')
            c=256 if 'PCA256' in reg else 384
            m=Tiny(c);refs.append(weakref.ref(m));calls.append(tag);return m,{'completed':5000}
        api=SimpleNamespace(load_world_checkpoint=load);r=sample_requests(2,3,(18,24),4,patch_generator=torch.Generator().manual_seed(3))
        with mock.patch.multiple(torch.cuda,manual_seed_all=lambda *_:None,get_rng_state_all=lambda:[],reset_peak_memory_stats=lambda:None,
            synchronize=lambda:None,max_memory_allocated=lambda:0,max_memory_reserved=lambda:0,empty_cache=lambda:None):
            for arm in ('256','384'):
                pred,record=d.run_arm(B,arm,[object(),object()],r,api);self.assertEqual(pred.shape[-1],int(arm));self.assertTrue(record['rng_unchanged'])
        self.assertEqual(len(calls),2);self.assertFalse(any(r() is not None for r in refs))
if __name__=='__main__':unittest.main(verbosity=2)
