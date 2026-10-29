"""Strict sealed native/standalone boundaries on tiny actual A/B/C/D modules."""
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from test_rl_adapter import TinyDeployment
from starVLA.model.modules.qwen_world_compare.contracts import ArchitecturePrefix, CurrentContext
from starVLA.model.modules.qwen_world_postcot.flow import clean_x_step
from starVLA.model.modules.qwen_sourceflow.action_codec import ActionCodec
from starVLA.rl.qwen_world_compare import loading, deployment
from starVLA.rl.qwen_world_compare.contracts import fingerprint, file_sha
from starVLA.rl.qwen_world_compare.policy_adapter import FourFrameworkRecipeActor
from starVLA.rl.qwen_world_postcot.rl_runtime import expected_cleanx_recipe


class TinyModel(TinyDeployment):
    def __init__(self, architecture, train_heads=True):
        super().__init__(architecture)
        self.train_heads = train_heads
        if train_heads:self.readouts = nn.Linear(4, 3)

    @torch.no_grad()
    def predict(self, inputs, ego, noise):
        with torch.autocast('cpu', dtype=torch.bfloat16):
            if self.architecture_id == 'D':
                encoded = self.decision.encode_current(inputs, ego)
                context = CurrentContext(None, None, ego, encoded, ('s0',), 'test', 'cpu', 'native')
            else:
                scene = inputs['scene']
                context = CurrentContext(scene, torch.ones(scene.shape[:2], dtype=torch.bool), ego,
                    None, ('s0',), 'test', 'cpu', 'scene')
            p = self.prepare_prefix(context, auxiliary=False)
            state = noise.float()
            u = torch.linspace(1, 0, 11);grid = 5*u/(1+4*u)
            for sigma, nxt in zip(grid[:-1], grid[1:]):
                x = self.denoise_x0(state, sigma.expand(len(state)), p)
                state = clean_x_step(state, x, sigma, nxt)
        return dict(normalized_action=state, requested_trajectory=ActionCodec.denorm_odo(state))

    def deployment_components(self):
        return dict(architecture_id=self.architecture_id, post_reasoning=False,
                    readouts=False, world_prefix_retained=self.architecture_id == 'C')


@pytest.fixture(autouse=True)
def fixture_runtime(monkeypatch):
    old = torch.get_num_threads();torch.set_num_threads(2);torch.manual_seed(42)
    make = lambda identity, train_heads=True:TinyModel(identity['architecture_id'], train_heads)
    monkeypatch.setattr(loading, 'model_from_identity', make)
    monkeypatch.setattr(deployment, 'model_from_identity', make)
    monkeypatch.setattr(deployment, '_source_runtime', lambda:{'tiny_test_runtime': '0'*64})
    from starVLA.rl.qwen_sourceflow import export
    def processor(path):
        path = Path(path);path.mkdir()
        for name in export.PROCESSOR_FILES:(path/name).write_text('{}')
        return {name:file_sha(path/name) for name in export.PROCESSOR_FILES}
    monkeypatch.setattr(export, 'package_processor', processor)
    yield
    torch.set_num_threads(old)


def save_IL(path, architecture='A', updates=1000):
    path = Path(path);(path/'native').mkdir(parents=True)
    model = TinyModel(architecture)
    torch.save(dict(module=model.state_dict()), path/'native/mp_rank_00_model_states.pt')
    torch.save({}, path/'native/zero_pp_rank_0_mp_rank_00_optim_states.pt')
    torch.save({}, path/'rank00_state.pt')
    identity = dict(schema=loading.IL_SCHEMA, architecture_id=architecture, qualification=True,
        world=1, uses_rl=False, model_config={}, qwen_config={}, config={}, source_files={})
    (path/'identity.json').write_text(json.dumps(identity))
    reseal_IL(path, identity, updates)
    return model


def reseal_IL(path, identity=None, updates=1000):
    identity = json.loads((path/'identity.json').read_text()) if identity is None else identity
    files = {str(p.relative_to(path)):file_sha(p) for p in path.rglob('*')
             if p.is_file() and p.name not in ('files.json', 'identity.json', 'COMPLETE')}
    (path/'files.json').write_text(json.dumps(files))
    (path/'COMPLETE').write_text(json.dumps(dict(status='COMPLETE', updates=updates,
        cursor=updates*32, phase='joint', model_file='native/mp_rank_00_model_states.pt',
        identity_sha256=fingerprint(identity), files_sha256=file_sha(path/'files.json'))))


def inputs(architecture):
    if architecture == 'D':
        return dict(input_ids=torch.tensor([[3, 5, 7, 9]]), attention_mask=torch.ones(1, 4, dtype=torch.long))
    return dict(scene=torch.randn(1, 160, 16))


@pytest.mark.parametrize('architecture', ['A', 'B', 'C', 'D'])
def test_IL_native_export_strict_reload_complete_K10_parity(tmp_path, architecture):
    source = tmp_path/'native';original = save_IL(source, architecture)
    features, ego, noise = inputs(architecture), torch.randn(1, 8), torch.randn(1, 8, 3)
    before = original.eval().predict(features, ego, noise)
    out = tmp_path/'standalone'
    meta = deployment.export_policy(source, out)
    restored, again = deployment.load_export(out)
    assert meta == again and meta['architecture_id'] == architecture
    assert meta['qualification'] and meta['LoRA_merge'] is False
    assert not hasattr(restored, 'readouts')
    assert hasattr(original, 'readouts')  # no live/source object was pruned.
    after = restored.predict(features, ego, noise)
    for key in before:torch.testing.assert_close(before[key], after[key], atol=0, rtol=0)
    with pytest.raises(FileExistsError):deployment.export_policy(source, out)


@pytest.mark.parametrize('architecture', ['A', 'B', 'C', 'D'])
def test_prune_copy_keeps_training_model_and_exact_deployment_forward(architecture):
    original = TinyModel(architecture).eval()
    features, ego, noise = inputs(architecture), torch.randn(1, 8), torch.randn(1, 8, 3)
    model = deployment.prune_deployment(original)
    before, after = original.predict(features, ego, noise), model.predict(features, ego, noise)
    assert hasattr(original, 'readouts') and not hasattr(model, 'readouts')
    for key in before:torch.testing.assert_close(before[key], after[key], atol=0, rtol=0)


def save_RL(path, actor, parent):
    path.mkdir()
    identity = dict(schema=loading.RL_SCHEMA, architecture_id=actor.action_expert.architecture_id,
        world=1, purpose='qualification', IL_parent=parent, recipe=expected_cleanx_recipe())
    identity['sha256'] = fingerprint(identity)
    adapters = {name:p.detach().clone() for name, p in actor.named_parameters() if p.requires_grad}
    torch.save(dict(adapters=adapters), path/'optimizer.pt')
    torch.save({}, path/'rank0.pt')
    (path/'identity.json').write_text(json.dumps(identity))
    row = dict(phase='READY', updates=4, collections=1, scene_cursor=1,
               next_inner=0, run=identity['sha256'], pending=False)
    (path/'transaction.json').write_text(json.dumps(dict(ranks=[row])))
    files = {p.name:file_sha(p) for p in path.iterdir() if p.is_file()}
    (path/'files.json').write_text(json.dumps(files))
    (path/'COMPLETE').write_text(json.dumps(dict(schema=2, phase='READY', world=1,
        updates=4, next_inner=0, files_sha256=file_sha(path/'files.json'))))


@pytest.mark.parametrize('architecture', ['A', 'B', 'C', 'D'])
def test_RL_export_retains_exact_unmerged_scope_and_K10(tmp_path, architecture):
    source = tmp_path/'IL';save_IL(source, architecture)
    model, parent = loading.load_IL_checkpoint(source, deployment=True)
    actor = FourFrameworkRecipeActor(model, expected_cleanx_recipe())
    with torch.no_grad():
        for name, p in actor.named_parameters():
            if name.endswith('.lora_B'):p.add_(torch.randn_like(p)*0.001)
    rl = tmp_path/'RL';save_RL(rl, actor, parent)
    out = tmp_path/'RL_export';deployment.export_policy(source, out, rl)
    restored, info = deployment.load_export(out)
    features, ego, noise = inputs(architecture), torch.randn(1, 8), torch.randn(1, 8, 3)
    before = actor.action_expert.predict(features, ego, noise)
    after = restored.predict(features, ego, noise)
    assert info['RL_updates'] == 4 and info['RL_parent']['schema'] == loading.RL_SCHEMA
    assert any('lora_A' in name for name in restored.state_dict())
    for key in before:torch.testing.assert_close(before[key], after[key], atol=0, rtol=0)


def test_reject_changed_native_inventory_wrong_schema_dtype_and_extra_keys(tmp_path):
    root = tmp_path/'IL';model = save_IL(root)
    raw = torch.load(root/'native/mp_rank_00_model_states.pt', weights_only=False)
    raw['module']['surprise.weight'] = torch.ones(1)
    torch.save(raw, root/'native/mp_rank_00_model_states.pt')
    with pytest.raises(ValueError, match='changed checkpoint'):
        loading.load_IL_checkpoint(root)
    reseal_IL(root)
    with pytest.raises(ValueError, match='Strict deployment tensor keys'):
        loading.load_IL_checkpoint(root)
    raw['module'] = model.state_dict()
    key = next(iter(raw['module']));raw['module'][key] = raw['module'][key].bfloat16()
    torch.save(raw, root/'native/mp_rank_00_model_states.pt');reseal_IL(root)
    with pytest.raises(ValueError, match='shape/dtype'):
        loading.load_IL_checkpoint(root)
    identity = json.loads((root/'identity.json').read_text());identity['schema'] = 'world_postcot_full_IL_v1'
    (root/'identity.json').write_text(json.dumps(identity));reseal_IL(root, identity)
    with pytest.raises(ValueError, match='four-framework IL'):
        loading.inspect_IL_checkpoint(root)


def test_export_processor_hash_and_runtime_are_enforced(tmp_path, monkeypatch):
    source = tmp_path/'IL';save_IL(source)
    out = tmp_path/'export';deployment.export_policy(source, out)
    monkeypatch.setattr(deployment, '_source_runtime', lambda:{'tiny_test_runtime': '1'*64})
    with pytest.raises(ValueError, match='runtime'):
        deployment.load_export(out)
    monkeypatch.setattr(deployment, '_source_runtime', lambda:{'tiny_test_runtime': '0'*64})
    (out/'processor/tokenizer.json').write_text('changed')
    with pytest.raises(ValueError, match='processor'):
        deployment.inspect_export(out)
