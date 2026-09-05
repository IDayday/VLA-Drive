import torch
from test_scene_fusion import _action_config
from navsim.agents.EpisodeDrive.action_decoder import ActionDecoder
from navsim.agents.EpisodeDrive.layers.planning_registers.content_diagnostics import semantic_content_diagnostics


def test_new_query_initialization_and_load_does_not_reinitialize():
    config = _action_config()
    config.semantic_query_init_std = 0.02
    decoder = ActionDecoder(config)
    assert 0.018 < decoder.scene_embeds.std() < 0.022
    saved = decoder.state_dict()
    restored = ActionDecoder(config)
    restored.load_state_dict(saved)
    assert torch.equal(decoder.scene_embeds, restored.scene_embeds)
    hidden = torch.randn(3, 13, 1536)
    outputs = decoder.q_former(decoder.scene_embeds, hidden)
    diag = semantic_content_diagnostics(outputs)
    assert diag['semantic_cross_scene_slot_content_rms'] > 1e-5
    constant_identity = torch.randn(1, 16, 256).expand(3, -1, -1)
    assert semantic_content_diagnostics(constant_identity)['semantic_cross_scene_content_rms'] < 1e-6
