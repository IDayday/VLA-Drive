"""Three bounded CPU tests; real tiny native Qwen/LoRA/FPO graph included."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from starVLA.rl.d44_native_rl.execution_cache import install_invariant_cache, invariant_cache
from starVLA.rl.d44_native_rl.fpo import velocity_energy
from starVLA.rl.d44_native_rl.gaussian_objective import clipped_loss
from starVLA.rl.qwen_world_compare.policy_adapter import FourFrameworkRecipeActor
from starVLA.rl.qwen_world_compare.recovery_state import tensor_digest


spec = importlib.util.spec_from_file_location('invariant_native_oracle',
    Path(__file__).parents[1] / 'world_compare/test_rl_adapter.py')
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)


@pytest.fixture(autouse=True)
def cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    yield
    torch.set_num_threads(previous)


def test_real_checkpointed_tiny_FPO_chain_loss_LoRA_gradient_Adam_and_RNG_bitexact():
    model = oracle.TinyDeployment('D')
    left = FourFrameworkRecipeActor(model, oracle.expected_cleanx_recipe())
    # Exercise both LoRA matrices rather than only the zero-B initialization.
    with torch.no_grad():
        for name, parameter in left.named_parameters():
            if name.endswith('lora_B'):
                parameter.normal_(std=.0003)
    right = copy.deepcopy(left)
    cond = oracle.condition(left.action_expert, group=16)
    flags = [install_invariant_cache(right.action_expert.decision, rotary=True, additive=True),
             install_invariant_cache(right.ref_action_expert.decision, rotary=True, additive=True)]
    source_keys = tuple(left.state_dict())
    assert tuple(right.state_dict()) == source_keys
    generators = [torch.Generator().manual_seed(133), torch.Generator().manual_seed(133)]
    global_rng = torch.get_rng_state().clone()
    chains, losses, gradients, final = [], [], [], []
    try:
        for actor, generator in zip((left, right), generators):
            with torch.no_grad():
                chain, times, deltas, indices = actor.sample_action_chain(cond, 8, generator=generator)
                chains.append((chain, times, deltas, indices))
                sigmas = .2 + .6 * torch.rand(4, 16, generator=generator)
                eps = torch.randn(4, 16, 8, 3, generator=generator)
                clean = chain[:, -1]
                states = (1 - sigmas[..., None, None]) * clean[None] + sigmas[..., None, None] * eps
                targets = eps - clean[None]
            parameters = [p for p in actor.parameters() if p.requires_grad]
            opt = torch.optim.AdamW(parameters, lr=1e-5)
            advantage = torch.linspace(-1, 1, 16)
            old_energy = None
            arm_losses, arm_gradients = [], []
            for inner in range(4):
                opt.zero_grad(set_to_none=True)
                with actor._context_call(cond) as condition:
                    prefix = actor._prepare(condition, False)
                    with torch.no_grad():
                        ref = actor._prepare(condition, True)
                    energies, keeps = [], []
                    for x, sigma, target in zip(states, sigmas, targets):
                        with torch.autocast('cpu', dtype=torch.bfloat16):
                            prediction = actor.action_expert.denoise_x0(x, sigma, prefix)
                        with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16):
                            reference = actor.ref_action_expert.denoise_x0(x, sigma, ref)
                        energies.append(velocity_energy(x, prediction, target, sigma))
                        keeps.append((prediction.float() - reference.float()).square().mean())
                    energy = torch.stack(energies).mean(0)
                    if old_energy is None:
                        old_energy = energy.detach()
                    loss = clipped_loss((old_energy - energy)[:, None], advantage, .1)[0]
                    loss = loss + .001 * torch.stack(keeps).mean()
                loss.backward()
                arm_losses.append(loss.detach().clone())
                arm_gradients.append(tensor_digest([p.grad for p in parameters]))
                opt.step()
            losses.append(arm_losses)
            gradients.append(arm_gradients)
            final.append(tensor_digest(dict(weights=actor.state_dict(), Adam=opt.state_dict())))
            assert torch.equal(global_rng, torch.get_rng_state())
        assert tensor_digest(chains[0]) == tensor_digest(chains[1])
        assert tensor_digest(losses[0]) == tensor_digest(losses[1])
        assert gradients[0] == gradients[1] and final[0] == final[1]
        assert torch.equal(generators[0].get_state(), generators[1].get_state())
        assert flags[0].stats['rotary_hits'] > 0 and flags[0].stats['additive_hits'] > 0
        # Checkpoint backward recomputes via original methods outside _run.
        assert flags[0].stats['rotary_bypass'] > 0 and not flags[0].active
    finally:
        for flag in flags:
            flag.close()


class ProbeRotary(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    @torch.no_grad()
    def forward(self, hidden, positions):
        self.calls += 1
        value = positions.float().transpose(0, 1).contiguous().to(hidden.dtype)
        return value.cos(), value.sin()


class ProbeDecision:
    def __init__(self):
        self.language_model = SimpleNamespace(rotary_emb=ProbeRotary())
        self.mask_calls = 0

    def _additive(self, allowed, dtype):
        self.mask_calls += 1
        return torch.zeros(allowed.shape, dtype=dtype).masked_fill(~allowed, torch.finfo(dtype).min)[:, None]

    def _run(self, hidden, positions, allowed, *, edit=False, nested=False, fail=False):
        outputs = []
        for index in range(4):
            if edit and index == 2:
                positions.add_(1)
                allowed[0, 0, 0] = ~allowed[0, 0, 0]
            if nested and index == 1:
                self._run(hidden, positions.clone(), allowed.clone())
            if fail and index == 2:
                raise RuntimeError('original run failure')
            value = hidden if index % 2 == 0 else hidden.to(torch.bfloat16)
            rotation = self.language_model.rotary_emb(value, positions)
            mask = self._additive(allowed, value.dtype)
            outputs.append((rotation, mask))
        return outputs


def probe_inputs():
    return torch.zeros(1, 4, 8), torch.arange(4).reshape(1, 4), torch.ones(1, 4, 4, dtype=torch.bool)


def layout(outputs):
    return [[(tuple(x.shape), x.stride(), x.dtype, x.grad_fn, x.requires_grad)
             for x in (*rotation, mask)] for rotation, mask in outputs]


def test_default_off_then_identity_version_dtype_layout_guards_and_no_cross_run_cache():
    plain, cached = ProbeDecision(), ProbeDecision()
    disabled = install_invariant_cache(cached)
    assert '_run' not in cached.__dict__ and 'forward' not in cached.language_model.rotary_emb.__dict__
    disabled.close()
    with invariant_cache(cached, rotary=True, additive=True) as handle:
        for edit in (False, True):
            first = plain._run(*probe_inputs(), edit=edit)
            second = cached._run(*probe_inputs(), edit=edit)
            assert tensor_digest(first) == tensor_digest(second)
            assert layout(first) == layout(second)
        assert handle.stats['rotary_hits'] == handle.stats['additive_hits'] == 2
        # Two dtypes per run; mutations in second run force fresh computations.
        assert cached.language_model.rotary_emb.calls == cached.mask_calls == 6
        assert not handle.active
    assert '_run' not in cached.__dict__ and '_additive' not in cached.__dict__
    assert 'forward' not in cached.language_model.rotary_emb.__dict__
    assert tensor_digest(plain._run(*probe_inputs())) == tensor_digest(cached._run(*probe_inputs()))


def test_nested_and_exception_scopes_bypass_clear_and_restore_original_calls():
    plain, cached = ProbeDecision(), ProbeDecision()
    with invariant_cache(cached, rotary=True, additive=True) as handle:
        assert tensor_digest(plain._run(*probe_inputs(), nested=True)) == tensor_digest(cached._run(*probe_inputs(), nested=True))
        assert handle.stats['nested_bypass'] == 1 and not handle.active
        with pytest.raises(RuntimeError, match='original run failure'):
            cached._run(*probe_inputs(), fail=True)
        assert not handle.active
        hits = handle.stats['rotary_hits']
        hidden, positions, allowed = probe_inputs()
        # Outside _run (including checkpoint replay) always executes originals.
        for _ in range(2):
            cached.language_model.rotary_emb(hidden, positions)
            cached._additive(allowed, hidden.dtype)
        assert handle.stats['rotary_hits'] == hits
        assert tensor_digest(plain._run(*probe_inputs())) == tensor_digest(cached._run(*probe_inputs()))
    with pytest.raises(RuntimeError, match='original run failure'):
        with invariant_cache(cached, rotary=True):
            cached._run(*probe_inputs(), fail=True)
    assert '_run' not in cached.__dict__ and '_d46_invariant_cache' not in cached.__dict__
