import copy
import types
import pytest
import torch
from omegaconf import OmegaConf
from starVLA.model.modules.vehicle_joint.action_head import VehicleJointActionHead, modeled_mask
from starVLA.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead


def config():
    return OmegaConf.create({"framework": {"qwenvl": {"vl_hidden_dim": 24}, "action_model": {
        "hidden_size": 32, "action_dim": 4, "action_horizon": 8,
        "num_inference_timesteps": 3, "noise_beta_alpha": 1.5, "noise_beta_beta": 1.,
        "noise_s": .999, "num_timestep_buckets": 1000, "add_pos_embed": True, "max_seq_len": 1024,
        "DiTConfig": {"input_embedding_dim": 32, "num_layers": 4, "num_attention_heads": 4,
                      "attention_head_dim": 8},
        "diffusion_model_cfg": {"cross_attention_dim": 32, "dropout": 0., "final_dropout": False,
          "interleave_self_attention": True, "norm_type": "ada_norm", "output_dim": 32,
          "positional_embeddings": None}}}})


def fixture():
    torch.manual_seed(42)
    model = VehicleJointActionHead(config()).eval()
    active = torch.tensor([[True, True, False]])
    boxes = torch.ones(1, 3, 8)
    queries = torch.randn(1, 3, 24)
    actions = torch.randn(1, 8, 24)
    noise = torch.randn(1, 3, 8, 4)
    return model, (actions, queries, boxes, active, noise)


def test_original_ddp_shared_tensors_match_joint_initialization():
    torch.manual_seed(42); base = FlowmatchingActionHead(config())
    torch.manual_seed(42); joint = VehicleJointActionHead(config())
    assert all(torch.equal(value, joint.state_dict()[key]) for key, value in base.state_dict().items())


def test_unmodeled_yaw_cannot_change_generated_ego_or_vehicle_xy():
    model, args = fixture()
    out = model.sample(*args)
    changed = args[-1].clone(); changed[:, 1:, :, 2:] = float("nan")
    assert torch.equal(out, model.sample(*args[:-1], changed))
    altered = copy.deepcopy(model)
    original = altered.velocity
    def poisoned(self, *a, **kw):
        v = original(*a, **kw); v[:, 1:, :, 2:] = 1e30; return v
    altered.velocity = types.MethodType(poisoned, altered)
    assert torch.equal(out, altered.sample(*args))
    assert out[:, 1:, :, 2:].count_nonzero() == 0
    assert model.executed_ego(out).data_ptr() == out[:, 0].data_ptr()


def test_padding_nan_and_large_values_isolate_forward_backward():
    model, args = fixture()
    action, query, boxes, active, noise = args
    result = []; grads = []
    for poison in (0., float("nan"), 1e30):
        model.zero_grad(set_to_none=True)
        q, b, n = query.clone(), boxes.clone(), noise.clone()
        q[:, 2] = poison; b[:, 2] = poison; n[:, 2] = poison
        v = model.velocity(n, torch.tensor([.4]), action, q, b, active)
        v.square().sum().backward()
        result.append(v.detach()); grads.append({k:p.grad.clone() for k,p in model.named_parameters() if p.grad is not None})
        assert all(torch.isfinite(g).all() for g in grads[-1].values())
    assert all(torch.equal(result[0], r) for r in result[1:])
    assert all(torch.equal(grads[0][k], g[k]) for g in grads[1:] for k in g)
    bad = boxes.clone(); bad[:, 1, 0] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        model.velocity(noise, torch.tensor([.4]), action, query, bad, active)


def test_known_clamped_each_step_hidden_values_unused_and_ego_yaw_gradient():
    model, args = fixture()
    action, query, boxes, active, noise = args
    valid = modeled_mask(active)
    known = torch.zeros_like(valid); known[:, 1, :4, :2] = True
    values = torch.randn_like(noise)
    out, history = model.sample(*args, known_values=values, known_mask=known, return_history=True)
    assert all(torch.equal(h[known], values[known]) for h in history)
    poisoned = values.clone(); poisoned[~known] = float("nan")
    assert torch.equal(out, model.sample(*args, known_values=poisoned, known_mask=known))
    target = torch.randn_like(noise)
    target[~valid] = float("nan")
    loss, counts = model.loss(action, query, boxes, active, target, valid, noise, torch.tensor([.6]))
    loss.backward()
    assert counts["vehicle_coordinates"] == 16
    assert model.action_decoder.layer2.weight.grad[2:].abs().sum() > 0
    assert model.model.transformer_blocks[1].attn1.to_q.weight.grad.abs().sum() > 0


def test_checkpointed_dit_and_state_roundtrip():
    model, args = fixture()
    model.train(); model.model.gradient_checkpointing = True
    action, query, boxes, active, noise = args
    v = model.velocity(noise, torch.tensor([.4]), action, query, boxes, active)
    v.square().sum().backward()
    restored = VehicleJointActionHead(config()).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    model.eval()
    assert torch.equal(model.sample(*args), restored.sample(*args))
