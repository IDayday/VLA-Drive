import torch
from navsim.agents.EpisodeDrive.layers.planning_registers import InternVLPlanningRegisters


def inputs():
    torch.manual_seed(29)
    adapter = InternVLPlanningRegisters(32, 16, 256, tile_aggregation='global_local_8_8')
    visual = torch.randn(3, 16, 256, requires_grad=True)
    metadata = torch.tensor([[.25,.5,.5,1,0],[.75,.5,.5,1,0],[.5,.5,1,1,1]])
    return adapter, visual, metadata


def test_shape_permutation_geometry_and_visual_content_response():
    model, visual, metadata = inputs()
    output = model._aggregate_tiles(visual, [3], metadata)
    assert output.shape == (1,16,256)
    order = torch.tensor([2,1,0])
    permuted = model._aggregate_tiles(visual[order], [3], metadata[order])
    torch.testing.assert_close(output, permuted, atol=2e-6, rtol=1e-5)
    wrong = metadata.clone()
    wrong[:2] = metadata[torch.tensor([1,0])]
    assert (model._aggregate_tiles(visual, [3], wrong)[:,8:] - output[:,8:]).abs().max() > 1e-6
    changed = visual.detach().clone()
    changed[0] += torch.randn_like(changed[0])
    new = model._aggregate_tiles(changed, [3], metadata)
    assert not torch.allclose(new[:,8:], output[:,8:])
    assert torch.equal(new[:,:8], output[:,:8])
    output.square().mean().backward()
    assert visual.grad[:2].norm() > 0


def test_position_or_query_identity_cannot_create_visual_rank():
    model, visual, metadata = inputs()
    constant = torch.randn(1,1,256).expand_as(visual)
    output = model._aggregate_tiles(constant, [3], metadata)
    for group in (output[:,:8], output[:,8:]):
        assert (group - group.mean(1, keepdim=True)).abs().max() < 2e-6


def test_scorer_input_responds_to_local_visual_changes():
    from test_scene_fusion import _action_config
    from omegaconf import OmegaConf
    from navsim.agents.EpisodeDrive.action_decoder import ActionDecoder
    model, visual, metadata = inputs()
    action = ActionDecoder(_action_config(), OmegaConf.create({'mode':'planning_primary_semantic_xattn'})).eval()
    hidden = torch.randn(1,16,256)
    scene, _ = action.fuse_scene_features(hidden, model._aggregate_tiles(visual, [3], metadata))
    altered = visual.clone()
    altered[0] *= -1
    other, _ = action.fuse_scene_features(hidden, model._aggregate_tiles(altered, [3], metadata))
    assert not torch.allclose(scene, other)
    proposals = torch.randn(1,64,8,3, requires_grad=True)
    embedded = action.pos_embed(proposals.detach().flatten(-2))
    logits = action.scorer_attention(embedded, scene)
    logits.square().mean().backward()
    assert visual.grad[:2].norm() > 0 and proposals.grad is None
