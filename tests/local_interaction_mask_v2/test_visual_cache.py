from types import SimpleNamespace
import torch
import pytest
from starVLA.model.modules.joint_world.visual_cache import FrozenVisualCache


def test_identity_corruption_and_trainable_vision_rejected(tmp_path):
    vision=torch.nn.Linear(2,2).requires_grad_(False)
    model=SimpleNamespace(model=SimpleNamespace(visual=vision,get_image_features=lambda pixels,grid:([pixels.bfloat16()],[pixels.bfloat16()+1])))
    identity={'cameras':['CAM_F0','CAM_L0','CAM_R0'],'version':2}
    cache=FrozenVisualCache(tmp_path,identity,write=True)
    q={'pixel_values':torch.ones(2,2),'image_grid_thw':torch.ones(3,dtype=torch.int64)}
    example={'token':'scene1','image':None,'state':None,'lang':'current'}
    a,b=cache.get(example,q,model);c,d=cache.get(example,q,model)
    assert torch.equal(a[0],c[0]) and cache.hits==1
    with pytest.raises(ValueError,match='version mismatch'):FrozenVisualCache(tmp_path,{'version':1})
    with pytest.raises(ValueError,match='observation mismatch'):cache.get(example,{**q,'pixel_values':q['pixel_values']*2},model)
    with pytest.raises(ValueError,match='whitelist'):cache.get({**example,'future':'forbidden'},q,model)
    vision.requires_grad_(True)
    with pytest.raises(ValueError,match='trainable'):cache.get(example,q,model)
    vision.requires_grad_(False)
    path=tmp_path/'scene1.pt';data=torch.load(path,weights_only=True);data['parts'][0][0,0]=8;torch.save(data,path)
    with pytest.raises(ValueError,match='corruption'):cache.get(example,q,model)
