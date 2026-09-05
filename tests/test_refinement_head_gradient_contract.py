import copy
import torch
from test_scene_fusion import _action_config
from navsim.agents.EpisodeDrive.action_decoder import ActionDecoder
from navsim.agents.EpisodeDrive.refinement_supervision import intermediate_trajectory_loss


def test_freeze_unused_heads_preserves_rng_outputs_and_no_decay_updates():
    cfg = _action_config()
    cfg.ref_num = 4
    model = ActionDecoder(cfg).train()
    updated = copy.deepcopy(model)
    updated._config.refinement_training_policy = 'final_only'
    updated.apply_refinement_training_policy()
    scene, ego = torch.randn(2,16,256), torch.randn(2,1,256)
    rng = torch.random.get_rng_state()
    reference = model._decode_scene(scene, ego)
    after = torch.random.get_rng_state()
    torch.random.set_rng_state(rng)
    actual = updated._decode_scene(scene, ego)
    assert torch.equal(actual['proposals'], reference['proposals'])
    assert torch.equal(after, torch.random.get_rng_state())
    actual['proposals'].square().mean().backward()
    for index, head in enumerate(updated.traj_head):
        assert all(p.requires_grad == (index == 4) for p in head.parameters())
        if index < 4:
            assert all(p.grad is None for p in head.parameters())


def test_auxiliary_excludes_head0_and_averages_only_1_to_3():
    proposals = [torch.randn(2,64,8,3, requires_grad=True) for _ in range(5)]
    targets = {'trajectory': torch.zeros(2,8,3)}
    loss = intermediate_trajectory_loss(proposals, targets)
    expected = .2 * sum(p.abs().sum(-1).mean(-1).amin(1).mean() for p in proposals[1:4])/3
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert proposals[0].grad is None and proposals[4].grad is None
    assert all(p.grad.norm() > 0 for p in proposals[1:4])
