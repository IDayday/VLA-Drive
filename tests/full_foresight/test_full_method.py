from dataclasses import replace
from pathlib import Path
import torch
import pytest
from omegaconf import OmegaConf
from starVLA.model.modules.foresight.config import ForesightConfig
from starVLA.model.modules.foresight.queries import SpatialViewEncoding
from starVLA.model.modules.foresight.dino_feature_head import DINOFeatureHead
from starVLA.model.modules.foresight.interaction_latent_head import InteractionLatentHead
from starVLA.model.modules.vehicle_joint.initialization import initialization_seed
from starVLA.model.modules.foresight.tradeoff import CANDIDATES


def configuration(name, monkeypatch):
    monkeypatch.setenv('FULL_LAMBDA_FUT','0.5');monkeypatch.setenv('FULL_LAMBDA_INT','0.25')
    cfg=OmegaConf.load(Path(__file__).parents[2]/'configs/foresight_resolution'/f'{name.lower()}.yaml')
    options=OmegaConf.to_container(cfg.foresight,resolve=True)
    for k in ('lambda_fut','lambda_int'):options[k]=float(options[k])
    return ForesightConfig(**options).validate()


@pytest.mark.parametrize('name',list(CANDIDATES))
def test_every_candidate_is_four_tasks_and_rejects_silent_disable(name,monkeypatch):
    cfg=configuration(name,monkeypatch);c=CANDIDATES[name]
    assert cfg.full_algorithm and cfg.is_dino and not cfg.is_tradeoff
    assert cfg.enable_current_dino and cfg.enable_future_dino and cfg.uses_interaction
    assert cfg.num_queries==c.num_queries and (cfg.dino_height,cfg.dino_width)==c.grid_hw
    for field in ('enable_current_dino','enable_future_dino','enable_interaction'):
        with pytest.raises(ValueError):replace(cfg,**{field:False}).validate()
    for field in ('lambda_cur','lambda_fut','lambda_int'):
        with pytest.raises(ValueError):replace(cfg,**{field:0.}).validate()
    with pytest.raises(ValueError):replace(cfg,readout_layers=3).validate()


def test_spatial_view_queries_order_gradients_and_initialization_isolation():
    torch.manual_seed(9);state=torch.random.get_rng_state().clone()
    with initialization_seed(2250):geometry=SpatialViewEncoding(24,(6,8))
    assert torch.equal(state,torch.random.get_rng_state())
    queries=torch.zeros(144,24,requires_grad=True);encoded=geometry(queries)
    view=encoded.reshape(3,6,8,24)
    assert not torch.equal(view[0],view[1]) and not torch.equal(view[0,0],view[0,1])
    assert torch.equal(geometry.positions[:8,1],geometry.positions[:1,1].expand(8))
    assert (geometry.positions[1:8,0]>geometry.positions[:7,0]).all()
    encoded.square().mean().backward()
    for value in (queries.grad,geometry.view.weight.grad,geometry.position.weight.grad):
        assert torch.isfinite(value).all() and value.abs().sum()>0
    with pytest.raises(ValueError):geometry(torch.zeros(48,24))


def test_three_auxiliary_readouts_share_identical_W_without_target_inputs():
    torch.manual_seed(42)
    w=torch.randn(1,144,24,requires_grad=True)
    visual=DINOFeatureHead(24,1024,dim=512,layers=2)
    interaction=InteractionLatentHead(24,512,layers=2)
    current=visual(w,torch.zeros(1),(6,8));future=visual(w,torch.ones(1),(6,8))
    latent=interaction(w)
    assert current.shape==future.shape==(1,3,1024,6,8) and latent.shape==(1,8,512)
    for output in (current,future,latent):
        gradient,=torch.autograd.grad(output.square().mean(),w,retain_graph=True)
        assert torch.isfinite(gradient).all() and gradient.abs().sum()>0
    assert not torch.equal(current,future)
    assert torch.equal(current,visual(w,torch.zeros(1),(6,8)))


def test_legacy_current_only_definition_is_not_relabelled_full(monkeypatch):
    full=configuration('C0',monkeypatch)
    with pytest.raises(ValueError):replace(full,full_algorithm=False).validate()
    old=ForesightConfig(arm='C0',num_queries=144,dino_height=6,dino_width=8,
                       enable_current_dino=True,lambda_cur=1.,auxiliary_warmup=1).validate()
    assert old.is_tradeoff and not old.uses_interaction


def test_removed_objective_is_an_error_not_silent_zero(monkeypatch):
    from starVLA.model.framework.ddp_full_foresight import DDPFullForesight
    # No fake forward result: exercise the guard before any expensive backbone.
    model=DDPFullForesight.__new__(DDPFullForesight);torch.nn.Module.__init__(model)
    model.foresight_config=configuration('C0',monkeypatch)
    with pytest.raises(RuntimeError,match='inventory'):
        model.forward_train([], {})
    model.dino_head=torch.nn.Identity();model.interaction_head=torch.nn.Identity()
    model.strip_auxiliary_heads()
    with pytest.raises(RuntimeError,match='Deployment model'):
        model.forward_train([], {})
