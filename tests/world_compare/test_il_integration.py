"""Independent IL entry/factory/precision/whole-update integration audit.

Only tests are injected: no production factory or trainer is patched here.
Small encoders replace generic weight loading in forward/backward arithmetic
tests; full native Qwen dimensions are separately constructed on meta device.
"""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import Qwen3VLConfig, Qwen3VLModel

from starVLA.model.modules.qwen_world_compare.model import FourFrameworkVLA
from starVLA.model.modules.qwen_world_compare.factory import load_config, build_model
from starVLA.model.modules.qwen_world_compare.contracts import LOSS_WEIGHTS
from starVLA.model.modules.qwen_world_compare.heads import EvaluationReadout
from starVLA.rl.qwen_world_compare.batching import update_denominators, post_positions
from tools.qwen_world_compare.train_il import parameter_groups, lr_scales
from tools.qwen_world_postcot.train_il import initialize_joint_optimizer


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads();torch.set_num_threads(2);torch.manual_seed(42)
    yield
    torch.set_num_threads(old)


class MockEncoder(nn.Module):
    def __init__(self):
        super().__init__();self.model = nn.Module()
        self.model.visual = nn.Module()
        self.model.visual.backbone = nn.Linear(32, 32)
        self.model.visual.merger = nn.Linear(32, 32)
        self.model.language_model = nn.Sequential(nn.Linear(32, 32), nn.SiLU(), nn.Linear(32, 32))

    def forward(self, batch):
        h = self.model.visual.merger(self.model.visual.backbone(batch['features']))
        h = self.model.language_model(h)
        b = len(h);indices = torch.arange(9)[None].expand(b, -1)
        return dict(selected_hidden_states={8:h, 16:h, 28:h}, final_hidden=h,
            visual_indices=indices, visual_mask=torch.ones(b, 9, dtype=torch.bool),
            visual_positions=torch.stack((torch.zeros(9), torch.zeros(9), torch.arange(9)), -1).long()[None].expand(b, -1, -1),
            camera_ids=torch.arange(3).repeat_interleave(3)[None].expand(b, -1),
            text_indices=torch.arange(9, 12)[None].expand(b, -1), text_mask=torch.ones(b, 3, dtype=torch.bool),
            merged_visual_features=h[:, :9])


class MiniFeature(nn.Module):
    def __init__(self, channels, dino=False):
        super().__init__();self.output = nn.Linear(16, channels);self.dino = dino
    def forward(self, p, indices=None):
        x = self.output(p.mean(1))
        if self.dino:return x[:, None, None, None].expand(-1, 3, 9, 16, -1)
        return x[:, None].expand(-1, indices.shape[1], -1)


class MiniObjects(nn.Module):
    def __init__(self):
        super().__init__();self.output = nn.Linear(16, 32*40)
    def forward(self, p):
        raw = self.output(p.mean(1)).reshape(len(p), 32, 40)
        return dict(existence_logits=raw[..., 0], class_logits=raw[..., 1:8],
                    boxes=raw[..., 8:16], future_states=raw[..., 16:].reshape(len(p), 32, 4, 6))


class MiniReadouts(nn.Module):
    def __init__(self, width):
        super().__init__();self.bridge = nn.Linear(width, 16)
        self.dino = MiniFeature(1024, True);self.wa = MiniFeature(512)
        self.objects = MiniObjects();self.evaluation = EvaluationReadout(16)


def tiny_model(architecture):
    if architecture == 'D':
        cfg = Qwen3VLConfig(text_config=dict(vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            attention_dropout=0., rope_scaling={'rope_type':'default', 'mrope_section':[1, 1, 2]}),
            vision_config=dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=4,
                out_hidden_size=32, patch_size=16, spatial_merge_size=2, temporal_patch_size=2,
                num_position_embeddings=16, deepstack_visual_indexes=[0, 1]),
            image_token_id=60, video_token_id=61, vision_start_token_id=62, vision_end_token_id=63)
        native = Qwen3VLModel._from_config(cfg, dtype=torch.float32, attn_implementation='sdpa')
        model = FourFrameworkVLA('D', config={'d':{'qualification':True, 'activation_checkpointing':True}},
                                 qwen_encoder=SimpleNamespace(model=native))
        width = 32
    else:
        actor = dict(hidden_dim=16, num_layers=2, num_heads=4, ffn_dim=64, activation_checkpointing=True)
        world = dict(hidden_dim=12, num_layers=2, num_heads=3, ffn_dim=48,
                     bridge_groups=1, activation_checkpointing=True)
        config = dict(scene=dict(input_dim=32, hidden_dim=16, num_layers=1, num_heads=4))
        config['decision'] = actor if architecture == 'B' else dict(actor_config=actor, world_config=world)
        model = FourFrameworkVLA(architecture, config=config, qwen_encoder=MockEncoder())
        width = 16 if architecture == 'B' else 12
    model.readouts = MiniReadouts(width)
    return model


def supervised_batch(architecture, count=2):
    if architecture == 'D':
        current = dict(input_ids=torch.tensor([[3, 62, 60, 60, 60, 60, 63, 7]]).expand(count, -1),
            attention_mask=torch.ones(count, 8, dtype=torch.long),
            pixel_values=torch.randn(count*16, 1536), image_grid_thw=torch.tensor([[1, 4, 4]]).expand(count, -1))
    else:current = dict(features=torch.randn(count, 12, 32))
    objects = dict(boxes=torch.zeros(count, 32, 8), classes=torch.zeros(count, 32, dtype=torch.long),
        valid=torch.zeros(count, 32, dtype=torch.bool), coverage_valid=torch.ones(count, dtype=torch.bool),
        future_states=torch.zeros(count, 32, 4, 6), future_valid=torch.zeros(count, 32, 4, dtype=torch.bool))
    return dict(tokens=[f's{i}' for i in range(count)], positions=list(range(count)), current_inputs=current,
        ego=torch.randn(count, 8), target=torch.randn(count, 8, 3), noise=torch.randn(count, 8, 3),
        sigma=torch.rand(count), action_valid=torch.ones(count, 8, dtype=torch.bool),
        dino_target=torch.zeros(count, 3, 9, 16, 1024), dino_valid=torch.ones(count, 3, 9, 16, dtype=torch.bool),
        future_queries=torch.zeros(count, 128, 4, dtype=torch.long), future_target=torch.zeros(count, 128, 512),
        future_valid=torch.ones(count, 128, dtype=torch.bool), objects=objects,
        aux_scene_indices=torch.tensor([0]), post_trajectories=torch.randn(1, 3, 8, 3),
        post_labels=torch.full((1, 3, 6), .5), post_valid=torch.ones(1, 3, 6, dtype=torch.bool),
        post_score=torch.tensor([[.3, .8, .4]]), post_score_valid=torch.ones(1, 3, dtype=torch.bool),
        post_events=torch.full((1, 3, 3), 8, dtype=torch.long), post_event_valid=torch.ones(1, 3, 3, dtype=torch.bool))


def microbatch(batch, lo, hi):
    total = len(batch['tokens']);selected = batch['aux_scene_indices']
    keep = ((selected >= lo)&(selected < hi)).nonzero().flatten()
    out = {}
    for key, value in batch.items():
        if key.startswith('post_'):
            out[key] = value[keep]
        elif key == 'aux_scene_indices':out[key] = selected[keep]-lo
        elif key == 'current_inputs':
            out[key] = {name:(v[lo*16:hi*16] if name == 'pixel_values' else v[lo:hi]) for name, v in value.items()}
        elif isinstance(value, dict):out[key] = {name:v[lo:hi] for name, v in value.items()}
        else:out[key] = value[lo:hi]
    return out


@pytest.mark.parametrize('architecture', ['A', 'B', 'C', 'D'])
@pytest.mark.parametrize('amp', [False, True])
def test_main_model_IL_forward_FP32_losses_and_whole_update_micro_reduction(architecture, amp):
    model = tiny_model(architecture).train();model.configure_stage(False)
    batch = supervised_batch(architecture);den = update_denominators([batch], 'cpu')
    with torch.autocast('cpu', dtype=torch.bfloat16, enabled=amp):full = model(batch, denominators=den)
    assert full['loss'].dtype == torch.float32
    assert all(value.dtype == torch.float32 for value in full['losses'].values())
    full['loss'].backward()
    ref = {name:p.grad.detach().clone() for name, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True);loss = 0.
    for i in range(2):
        with torch.autocast('cpu', dtype=torch.bfloat16, enabled=amp):part = model(microbatch(batch, i, i+1), denominators=den)
        loss += float(part['loss'].detach());part['loss'].backward()
    assert loss == pytest.approx(float(full['loss']), rel=1e-3, abs=2e-3)
    current, reference = [], []
    for name, p in model.named_parameters():
        if name in ref:
            assert p.grad is not None, name
            current.append(p.grad.flatten().double());reference.append(ref[name].flatten().double())
            if not amp:
                torch.testing.assert_close(p.grad, ref[name], atol=3e-5, rtol=3e-4, msg=name)
    if amp:
        # BF16 backward reduction trees vary with physical micro layout. Check
        # the complete gradient direction/relative norm, not near-zero entries.
        # The FP32 branch above independently checks the loss arithmetic.
        v, r = torch.cat(current), torch.cat(reference)
        assert (v-r).norm()/r.norm() < .02
        assert torch.nn.functional.cosine_similarity(v, r, dim=0) > .999


@pytest.mark.parametrize('architecture', ['A', 'B', 'C', 'D'])
def test_all_invalid_supervision_markers_do_not_make_model_loss_nan(architecture):
    model = tiny_model(architecture).train();batch = supervised_batch(architecture)
    batch['action_valid'].fill_(False);batch['target'].fill_(float('nan'))
    for values, valid in (('dino_target', 'dino_valid'), ('future_target', 'future_valid'),
                          ('post_labels', 'post_valid'), ('post_score', 'post_score_valid')):
        batch[values].fill_(float('nan'));batch[valid].fill_(False)
    batch['objects']['coverage_valid'].fill_(False)
    batch['objects']['future_states'].fill_(float('nan'))
    batch['post_event_valid'].fill_(False)
    den = update_denominators([batch], 'cpu')
    with torch.autocast('cpu', dtype=torch.bfloat16):result = model(batch, denominators=den)
    assert result['loss'].dtype == torch.float32
    assert float(result['loss']) == 0.
    assert all(float(v) == 0. for v in result['losses'].values())


def test_D_interface_gradient_and_explicit_new_parameter_group():
    model = tiny_model('D');cfg = load_config('configs/qwen_world_compare/arch_d.yaml')
    groups = parameter_groups(model, cfg);schedule = lr_scales(cfg, groups)
    membership = {id(p):group['name'] for group in groups for p in group['params']}
    for name in ('action_input.weight', 'p_embeddings', 'time_films.0.0.weight'):
        assert membership[id(model.decision.get_parameter(name))] == 'new_modules'
    assert [fn(0) for fn in schedule] == [0., 0., pytest.approx(1/3000)]
    assert [fn(999) for fn in schedule][:2] == [0., 0.]
    assert all(fn(1000) == pytest.approx(1001/3000) for fn in schedule)
    model.configure_stage(True);model.train()
    with torch.autocast('cpu', dtype=torch.bfloat16):result = model(supervised_batch('D'))
    result['loss'].backward()
    assert model.decision.action_input.weight.grad.abs().sum() > 0
    assert model.decision.p_embeddings.grad.abs().sum() > 0


def test_joint_initializer_resets_only_pretrained_moments_not_new_interfaces():
    model = tiny_model('D');cfg = load_config('configs/qwen_world_compare/arch_d.yaml')
    groups = parameter_groups(model, cfg);opt = torch.optim.AdamW(groups)
    for group in opt.param_groups:
        for p in group['params']:
            opt.state[p] = dict(step=torch.tensor(1000.), exp_avg=torch.ones_like(p), exp_avg_sq=torch.ones_like(p)*2)
    state = dict(updates=1000, qwen_optimizer_initialized=False)
    initialize_joint_optimizer(SimpleNamespace(optimizer=opt), state)
    assert state['qwen_optimizer_initialized']
    for group in opt.param_groups:
        for p in group['params']:
            assert opt.state[p]['step'] == (1000 if group['name'] == 'new_modules' else 0)
            assert opt.state[p]['exp_avg'].sum() == (p.numel() if group['name'] == 'new_modules' else 0)


def test_post_selection_is_whole_update_and_independent_of_micro_layout():
    for update in (0, 1, 999, 1000):
        expected = post_positions(update)
        assert len(expected) == len(set(expected)) == 8
        for micro in (1, 2, 4):
            selected = []
            for rank in range(8):
                rank_selected = []
                for start in range(rank*4, rank*4+4, micro):
                    rank_selected += [i for i in range(start, start+micro) if i in expected]
                assert len(rank_selected) == 1
                selected += rank_selected
            assert tuple(selected) == expected


def test_full_default_factory_meta_and_D_role_partition():
    config = json.loads(Path('/mnt/project/DriveDreamer-Policy/models/Qwen3-VL-2B-Instruct/config.json').read_text())
    for architecture in 'ABCD':
        cfg = load_config(f'configs/qwen_world_compare/arch_{architecture.lower()}.yaml')
        with torch.device('meta'):model = build_model(cfg, checkpoint_config=config)
        groups = parameter_groups(model, cfg)
        assert all(p.dtype == torch.float32 for p in model.parameters())
        assert sum(p.numel() for p in groups[0]['params']) == 1821289472
        assert sum(p.numel() for p in groups[1]['params']) == 306242560
        ids = [id(p) for group in groups for p in group['params']]
        assert len(ids) == len(set(ids)) == len(list(model.parameters()))
        if architecture == 'D':
            assert len(model.decision.language_model.layers) == 28
            assert model.decision.hidden_dim == 2048
            assert model.parameter_roles()['decision.time_films.0.0.weight'] == 'new_time_modulation'
        else:
            actor = model.decision if architecture == 'B' else model.decision.actor
            assert actor.num_layers == 20 and actor.hidden_dim == 1024
            if architecture != 'B':
                assert model.decision.world_stream.num_layers == 20
                assert model.decision.world_stream.hidden_dim == 384


@pytest.mark.parametrize('script', ['train_il.py', 'export.py', 'evaluate.py', 'profile.py'])
def test_documented_bare_script_help_without_PYTHONPATH_or_stdlib_profile_shadow(script):
    environment = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    environment.pop('PYTHONPATH', None)
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run([sys.executable, str(root/'tools/qwen_world_compare'/script), '--help'],
                            cwd=root, env=environment, capture_output=True, text=True, timeout=40)
    assert result.returncode == 0, result.stdout+'\n'+result.stderr
    assert 'usage:' in result.stdout
