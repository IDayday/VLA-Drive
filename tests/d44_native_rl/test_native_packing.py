"""Actual native tiny-Qwen, packed G8/16/32 vs independent candidate execution."""
import importlib.util
from pathlib import Path
import torch
import pytest

spec=importlib.util.spec_from_file_location('native_d_test_oracle',Path(__file__).parents[1]/'world_compare/test_d_qwen_flow_adapter.py')
oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)


@pytest.mark.parametrize('group',[8,16,32])
def test_output_prefix_gradient_no_candidate_leak(group):
    torch.set_num_threads(2);torch.manual_seed(42);m=oracle.model()
    b=oracle.batch();current=m.encode_current({k:v[:1]if torch.is_tensor(v)else v[:1]for k,v in b.items()})
    p=m.prepare_prefix(current);p=p.index_select(torch.zeros(group,dtype=torch.long))
    state=torch.randn(group,8,3);sigma=torch.full((group,),.5)
    m.pack_shared_candidates=False;plain=m.denoise_x0(state,sigma,p)
    g1=torch.autograd.grad(plain.square().sum(),m.p_embeddings,retain_graph=True)[0]
    m.pack_shared_candidates=True;packed=m.denoise_x0(state,sigma,p)
    g2=torch.autograd.grad(packed.square().sum(),m.p_embeddings,retain_graph=True)[0]
    torch.testing.assert_close(plain,packed,atol=2e-6,rtol=2e-5)
    torch.testing.assert_close(g1,g2,atol=2e-5,rtol=2e-4)
    changed=state.clone();changed[-1]+=2;out=m.denoise_x0(changed,sigma,p)
    torch.testing.assert_close(out[:-1],packed[:-1],atol=2e-6,rtol=2e-5)
