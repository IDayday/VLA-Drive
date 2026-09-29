import pytest,torch
from PIL import Image
from starVLA.model.modules.foresight.tradeoff import CANDIDATES,pool_patches,preprocess_current,TokenProjectionHead
from starVLA.model.modules.foresight.config import ForesightConfig
from starVLA.model.modules.foresight.losses import masked_regression
from starVLA.model.modules.vehicle_joint.initialization import initialization_seed

@pytest.mark.parametrize('name,hw,tokens',[('C0',(6,8),144),('C1',(6,8),144),('C2',(12,16),576),('C3',(9,12),324),('C4',(6,8),144),('C5',(12,16),576)])
def test_geometry_projection_and_pool(name,hw,tokens):
 c=CANDIDATES[name];assert c.grid_hw==hw and c.num_queries==tokens
 native=torch.arange(c.native_hw[0]*c.native_hw[1]).float().reshape(1,1,*c.native_hw)
 native=torch.cat([native+i*10000 for i in range(3)])
 z=pool_patches(native,c)
 assert z.shape==(3,1,*hw)
 assert z[0,0,0,0]==native[0,0,:c.pool,:c.pool].mean()
 assert torch.allclose(z[1]-z[0],torch.full_like(z[0],10000.))
 q=torch.randn(2,tokens,4,requires_grad=True);head=TokenProjectionHead(4,1)
 y=head(q,torch.zeros(2),hw)
 assert y.shape==(2,3,1,*hw)
 torch.testing.assert_close(y.permute(0,1,3,4,2).reshape(2,tokens,1),head.projection(q))
 y.sum().backward();assert q.grad.abs().sum()>0
 ForesightConfig(arm=name,num_queries=tokens,dino_height=hw[0],dino_width=hw[1],enable_current_dino=True,lambda_cur=1.,auxiliary_warmup=1).validate()

def test_original_field_and_actual_sizes(tmp_path):
 path=tmp_path/'original.png';Image.new('RGB',(1600,900),(100,50,20)).save(path)
 for c in CANDIDATES.values():
  pixels,meta=preprocess_current(path,c.width,c.height)
  assert pixels.shape==(3,c.height,c.width)
  assert meta['crop_xyxy']==[0,0,1600,900]
  expected=(torch.tensor([100,50,20])/255-torch.tensor([.485,.456,.406]))/torch.tensor([.229,.224,.225])
  torch.testing.assert_close(pixels[:,0,0],expected)

def test_mean_and_invalid_padding():
 for size in (48,108,192):
  p=torch.full((2,size,4),2.,requires_grad=True);t=torch.ones_like(p);valid=torch.ones_like(p,dtype=torch.bool)
  with torch.no_grad():p[1]=float('nan');t[1]=float('nan');valid[1]=False
  loss,_=masked_regression(p,t,valid);assert loss==1
  loss.backward();assert torch.isfinite(p.grad).all()
 with pytest.raises(ValueError):masked_regression(torch.tensor([float('nan')]),torch.zeros(1),torch.ones(1,dtype=torch.bool))

def test_head_initialization_independent_of_query_budget():
 before=torch.get_rng_state();weights=[]
 for c in CANDIDATES.values():
  with initialization_seed(2242):q=torch.randn(c.num_queries,8)
  with initialization_seed(2342):weights.append(TokenProjectionHead(8,4).state_dict())
 assert torch.equal(before,torch.get_rng_state())
 for state in weights[1:]:
  for key in state:torch.testing.assert_close(state[key],weights[0][key],rtol=0,atol=0)

def test_strict_current_only():
 with pytest.raises(ValueError):ForesightConfig(arm='C0',num_queries=64,enable_current_dino=True,lambda_cur=1.,auxiliary_warmup=1).validate()
 with pytest.raises(ValueError):TokenProjectionHead(4,8)(torch.zeros(1,144,4),torch.ones(1),(6,8))
