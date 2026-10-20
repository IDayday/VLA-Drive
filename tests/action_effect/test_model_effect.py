"""Behavior tests use explicit synthetic tensors, not real integration claims."""
from dataclasses import replace
import hashlib
import io
import json
import torch
import pytest

from starVLA.model.modules.action_effect.effect_model import (
    ActionEffectModel, EffectModelConfig, FeedbackLabelSpec, MODES,
)
from starVLA.model.modules.action_effect.effect_losses import (
    effect_model_loss, effect_loss_contributions, ranking_contribution,
    combine_contributions, reduce_contributions, scene_contribution,
)
from starVLA.model.modules.st_world.consequence import METRICS


torch.set_num_threads(2)


def spec():
    # The constructor consumes adapter-supplied names/units, not output_dim=11.
    names = tuple(METRICS) + ("progress_raw_meters", "first_at_fault_time_seconds",
                              "first_ttc_state_time_seconds", "first_road_time_seconds")
    return FeedbackLabelSpec.from_label_arrays({"names": names, "units": ("0..1",) * len(METRICS) + ("m", "s", "s", "s")})


def cfg(mode="delta_feedback"):
    return EffectModelConfig(mode=mode, context_dim=12, world_tokens=3, action_tokens=2,
                             hidden_dim=16, layers=2, heads=4)


def batch(size=3):
    g = torch.Generator().manual_seed(4)
    context = {"memory": torch.randn(size, 5, 12, generator=g),
               "ego": torch.randn(size, 1, 4, generator=g),
               "mask": torch.ones(size, 5, dtype=torch.bool)}
    a = torch.zeros(size, 8, 3); a[..., 0] = torch.arange(1, 9) * 2.5
    b = a.clone(); b[..., 1] = torch.linspace(0., .3, 8)
    return context, a, b


def state_hash(model):
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode()); h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def supervision(model, size):
    before = torch.full((size, len(model.label_spec.names)), .4)
    after = torch.full_like(before, .6)
    valid = torch.ones_like(before, dtype=torch.bool)
    return dict(before=before, after=after, before_valid=valid, after_valid=valid,
                label_spec=model.label_spec, mode=model.config.mode,
                event_targets=torch.tensor([[0, 1, 2]]).repeat(size, 1),
                event_valid=torch.ones(size, 3, dtype=torch.bool),
                temporal_targets=torch.zeros(size, 41, 3),
                temporal_valid=torch.ones(size, 41, 3, dtype=torch.bool),
                repair_targets=torch.ones(size), repair_valid=torch.ones(size, dtype=torch.bool))


def test_initialization_reproducible_without_consuming_caller_rng():
    torch.manual_seed(87); before = torch.get_rng_state().clone()
    first = ActionEffectModel(cfg(), spec())
    assert torch.equal(before, torch.get_rng_state())
    second = ActionEffectModel(cfg(), spec())
    third = ActionEffectModel(replace(cfg(), init_seed=18), spec())
    assert state_hash(first) == state_hash(second) != state_hash(third)
    assert all(p.data_ptr() != q.data_ptr() for p, q in zip(first.parameters(), second.parameters()))


def test_same_parameter_capacity_for_all_six_controls():
    states = [ActionEffectModel(cfg(mode), spec()).state_dict() for mode in MODES]
    first = states[0]
    assert all(list(state) == list(first) for state in states)
    assert all(torch.equal(state[key], first[key]) for state in states for key in first)


def test_delta_zero_swap_and_no_rng_consumption_in_train():
    model = ActionEffectModel(cfg(), spec()).train()
    context, a, b = batch()
    rng = torch.get_rng_state().clone()
    ab, ba, aa = model(context, a, b), model(context, b, a), model(context, a, a)
    assert torch.equal(aa["mean_delta"], torch.zeros_like(aa["mean_delta"]))
    torch.testing.assert_close(ab["mean_delta"], -ba["mean_delta"], rtol=0, atol=0)
    torch.testing.assert_close(ab["temporal_delta"], -ba["temporal_delta"], rtol=0, atol=0)
    torch.testing.assert_close(ab["event_logits"], ba["event_logits"].flip(-1), rtol=0, atol=0)
    torch.testing.assert_close(ab["logvar_delta"], ba["logvar_delta"], rtol=0, atol=0)
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize("mode", ["absolute_score", "terminal_pdms", "absolute_feedback"])
def test_absolute_controls_are_single_trajectory_potentials(mode):
    model = ActionEffectModel(cfg(mode), spec())
    context, a, b = batch()
    out = model(context, a, b)
    torch.testing.assert_close(out["mean_delta"], out["absolute_after"] - out["absolute_before"], rtol=0, atol=0)
    shuffled = model(context, b, a)
    torch.testing.assert_close(out["absolute_before"], shuffled["absolute_after"], rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["absolute_feedback", "delta_feedback", "trajectory_only"])
def test_repair_head_observes_both_trajectories_with_direction(mode):
    model = ActionEffectModel(cfg(mode), spec())
    context, a, b = batch()
    proposed = b.clone().requires_grad_()
    changed = b.clone(); changed[..., 1] += .7
    first = model(context, a, proposed)["repair_logit"]
    second = model(context, a, changed)["repair_logit"]
    assert not torch.equal(first, second)
    gradient, = torch.autograd.grad(first.sum(), proposed)
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    reverse = model(context, b, a)["repair_logit"]
    identity = model(context, a, a)["repair_logit"]
    torch.testing.assert_close(first + reverse, 2 * identity, rtol=1e-5, atol=1e-7)


def test_padding_and_candidate_batch_permutation_are_invariant():
    model = ActionEffectModel(cfg(), spec())
    context, a, b = batch()
    context["mask"][:, -1] = False
    before = model(context, a, b)["mean_delta"]
    changed = {**context, "memory": context["memory"].clone()}
    changed["memory"][:, -1] += 1e5
    torch.testing.assert_close(before, model(changed, a, b)["mean_delta"], rtol=0, atol=0)
    order = torch.tensor([2, 0, 1])
    shuffled = model({k: v[order] for k, v in context.items()}, a[order], b[order])["mean_delta"]
    torch.testing.assert_close(before[order], shuffled, rtol=1e-5, atol=1e-7)


def test_context_is_detached_but_proposed_trajectory_retains_gradient():
    model = ActionEffectModel(cfg(), spec())
    context, a, b = batch()
    context["memory"].requires_grad_(); context["ego"].requires_grad_(); b.requires_grad_()
    model(context, a, b)["mean_delta"].square().sum().backward()
    assert context["memory"].grad is None and context["ego"].grad is None
    assert b.grad is not None and torch.isfinite(b.grad).all() and b.grad.abs().sum() > 0


def test_scene_projection_is_computed_once_for_both_pair_directions():
    model = ActionEffectModel(cfg(), spec()); context, a, b = batch()
    calls = []
    hook = model.backbone.context_projection.register_forward_hook(lambda *args: calls.append(1))
    try:
        model(context, a, b)
    finally:
        hook.remove()
    assert len(calls) == 1


def test_trajectory_only_and_explicit_shuffle_controls():
    context, a, b = batch()
    model = ActionEffectModel(cfg("trajectory_only"), spec())
    reference = model(context, a, b)["mean_delta"]
    changed = {**context, "memory": context["memory"] * 7., "ego": context["ego"] * -3.}
    torch.testing.assert_close(reference, model(changed, a, b)["mean_delta"], rtol=0, atol=0)
    shuffle = ActionEffectModel(cfg("shuffled_context"), spec())
    with pytest.raises(ValueError, match="recorded permutation"):
        shuffle(context, a, b)
    shuffled = shuffle(context, a, b, context_permutation=torch.tensor([2, 0, 1]))
    assert torch.isfinite(shuffled["mean_delta"]).all()
    with pytest.raises(ValueError, match="full batch permutation"):
        shuffle(context, a, b, context_permutation=torch.tensor([1, 1, 1]))
    with pytest.raises(ValueError, match="BLOCKED_PRIVILEGED_ASSET"):
        ActionEffectModel(cfg("privileged"), spec())


def test_labels_cannot_enter_model_context_and_unknown_schema_rejects():
    model = ActionEffectModel(cfg(), spec()); context, a, b = batch()
    for key in ("score", "gt", "future", "scene_id", "expert_id", "metric_cache"):
        with pytest.raises(ValueError, match="only frozen current"):
            model({**context, key: torch.ones(1)}, a, b)
    with pytest.raises(ValueError, match="No registered"):
        FeedbackLabelSpec.from_label_arrays({"names": ["score"], "units": ["mystery"]})
    with pytest.raises(ValueError, match="dropout=0"):
        ActionEffectModel(replace(cfg(), dropout=.1), spec())


@pytest.mark.parametrize("mode", MODES)
def test_actual_backward_and_update_all_modes(mode):
    model = ActionEffectModel(cfg(mode), spec()).train(); context, a, b = batch()
    previous = state_hash(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    output = model(context, a, b, **({"context_permutation": torch.tensor([2, 0, 1])} if mode == "shuffled_context" else {}))
    result = effect_model_loss(output, **supervision(model, len(a)))
    assert set(result.terms) == {"value", "ranking", "events", "temporal", "uncertainty", "repair"}
    assert "mean_delta" not in result.terms and torch.isfinite(result.total)
    result.total.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert sum(g.abs().sum() for g in gradients) > 0
    optimizer.step(); assert previous != state_hash(model)


def test_single_small_gap_uses_exact_distribution_mass_not_unit_clamp():
    prediction = torch.tensor([.03], requires_grad=True)
    loss = ranking_contribution(prediction, torch.tensor([.02]), torch.tensor([True])).mean()
    expected = torch.nn.functional.softplus(-prediction[0] / .1)
    torch.testing.assert_close(loss, expected, rtol=0, atol=1e-7)
    observed_gradient, = torch.autograd.grad(loss, prediction, retain_graph=True)
    expected_gradient, = torch.autograd.grad(expected, prediction)
    torch.testing.assert_close(observed_gradient, expected_gradient, rtol=0, atol=1e-6)


def test_gap_is_not_cancelled_and_gap_sampled_stream_is_not_squared():
    prediction = torch.tensor([.2, .2], requires_grad=True)
    gap = torch.tensor([.95, .02]); valid = torch.ones(2, dtype=torch.bool)
    term = ranking_contribution(prediction, gap, valid)
    gradient, = torch.autograd.grad(term.mean(), prediction)
    assert (gradient[0] / gradient[1]).item() == pytest.approx(.95 / .02, rel=1e-6)
    sampled = ranking_contribution(prediction, gap, valid, sampling="gap_sampled")
    sampled_gradient, = torch.autograd.grad(sampled.mean(), prediction)
    assert (sampled_gradient[0] / sampled_gradient[1]).item() == pytest.approx(1.)
    # Weighting per-draw gradients by q restores the desired population ratio.
    q = gap / gap.sum()
    assert (q[0] * sampled_gradient[0] / (q[1] * sampled_gradient[1])).item() == pytest.approx(.95 / .02, rel=1e-6)
    assert (gradient < 0).all()  # Positive true gain must increase predicted gain.


def test_ties_missing_zero_denominator_and_invalid_predictions():
    p = torch.tensor([.2, .4], requires_grad=True)
    empty = ranking_contribution(p, torch.tensor([0., .01]), torch.ones(2, dtype=torch.bool))
    empty.mean().backward(); assert torch.equal(p.grad, torch.zeros_like(p))
    assert empty.denominator == 0 and empty.mean() == 0
    model = ActionEffectModel(cfg(), spec()); context, a, b = batch(); out = model(context, a, b)
    targets = supervision(model, 3); targets["before_valid"][:] = False; targets["after_valid"][:] = False
    targets["event_valid"][:] = False; targets["temporal_valid"][:] = False; targets["repair_valid"][:] = False
    result = effect_model_loss(out, **targets); assert result.total == 0
    result.total.backward()
    out = model(context, a, b); out["mean_delta"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite effect prediction"):
        effect_model_loss(out, **targets)


def test_physical_unit_scales_and_scene_mask_denominators():
    s = spec()
    assert s.scales[-4:] == (20., 4., 4., 4.)
    values = torch.tensor([[1., 9., 100.], [4., 100., 100.], [100., 100., 100.]], requires_grad=True)
    mask = torch.tensor([[True, True, False], [True, False, False], [False, False, False]])
    term = scene_contribution(values, mask)
    assert term.mean().item() == pytest.approx((5. + 4.) / 2)
    term.mean().backward()
    torch.testing.assert_close(values.grad, torch.tensor([[.25, .25, 0.], [.5, 0., 0.], [0., 0., 0.]]))


def test_terminal_only_control_cannot_receive_fine_grained_supervision():
    model = ActionEffectModel(cfg("terminal_pdms"), spec()); context, a, b = batch()
    out = model(context, a, b); targets = supervision(model, 3)
    first = effect_model_loss(out, **targets)
    # Same score, completely changed non-score labels/events: B must not see it.
    targets["before"][:, 1:] += 50.; targets["after"][:, 1:] -= 20.
    targets["temporal_targets"][:] = 1.; targets["event_targets"][:] = 2
    targets["repair_targets"][:] = 0.
    second = effect_model_loss(out, **targets)
    torch.testing.assert_close(first.total, second.total, rtol=0, atol=0)
    assert all(first.contributions[k].denominator == 0 for k in ("events", "temporal", "repair"))


def test_valid_nonfinite_temporal_and_repair_targets_are_not_silently_cleaned():
    model = ActionEffectModel(cfg(), spec()); context, a, b = batch()
    for name in ("temporal_targets", "repair_targets"):
        targets = supervision(model, 3); targets[name].reshape(-1)[0] = float("nan")
        with pytest.raises(ValueError, match="finite"):
            effect_model_loss(model(context, a, b), **targets)


def test_accumulation_uses_sums_before_division():
    p = torch.tensor([.2, -.1, .5], requires_grad=True)
    gaps = torch.tensor([.95, -.02, .4]); valid = torch.tensor([True, True, False])
    one = ranking_contribution(p, gaps, valid)
    pieces = [{"rank": ranking_contribution(p[:1], gaps[:1], valid[:1])},
              {"rank": ranking_contribution(p[1:], gaps[1:], valid[1:])}]
    combined = combine_contributions(pieces)["rank"]
    torch.testing.assert_close(one.mean(), combined.mean())
    g1, = torch.autograd.grad(one.mean(), p, retain_graph=True)
    g2, = torch.autograd.grad(combined.mean(), p)
    torch.testing.assert_close(g1, g2)


def test_model_and_optimizer_roundtrip_preserves_next_update():
    model = ActionEffectModel(cfg(), spec()); context, a, b = batch()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    effect_model_loss(model(context, a, b), **supervision(model, 3)).total.backward(); optimizer.step(); optimizer.zero_grad()
    buffer = io.BytesIO(); torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng": torch.get_rng_state()}, buffer)
    buffer.seek(0); saved = torch.load(buffer, weights_only=False)
    clone = ActionEffectModel(cfg(), spec()); clone.load_state_dict(saved["model"], strict=True)
    clone_optimizer = torch.optim.AdamW(clone.parameters(), lr=1e-4); clone_optimizer.load_state_dict(saved["optimizer"])
    for module, opt in ((model, optimizer), (clone, clone_optimizer)):
        torch.set_rng_state(saved["rng"])
        effect_model_loss(module(context, a, b), **supervision(module, 3)).total.backward(); opt.step()
    assert state_hash(model) == state_hash(clone)


def test_default_registered_model_runs_its_real_dimensions():
    config = EffectModelConfig(); model = ActionEffectModel(config, spec()).eval()
    context = {"memory": torch.zeros(1, 152, 2048), "ego": torch.zeros(1, 1, 4), "mask": torch.ones(1, 152, dtype=torch.bool)}
    _, a, b = batch(1)
    with torch.no_grad():
        output = model(context, a, b)
    assert output["mean_delta"].shape == (1, len(spec().names))
    assert torch.isfinite(output["mean_delta"]).all()


def _gloo_empty_rank_worker(rank, init_file, output_root):
    from pathlib import Path
    import datetime
    from torch import distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + init_file, world_size=2,
                            rank=rank, timeout=datetime.timedelta(seconds=25))
    try:
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad(): model.weight.fill_(1.)
        ddp = DistributedDataParallel(model)
        x = torch.tensor([[1.], [2.]])
        error = ddp(x).squeeze(-1).square()
        mask = torch.tensor([True, False]) if rank == 1 else torch.zeros(2, dtype=torch.bool)
        term = scene_contribution(error, mask)
        term.mean(distributed=True).backward()
        torch.optim.SGD(ddp.parameters(), lr=.1).step()
        Path(output_root, f"rank_{rank}.json").write_text(json.dumps({
            "gradient": model.weight.grad.item(), "updated_weight": model.weight.item()}))
    finally:
        dist.destroy_process_group()


def test_real_cpu_gloo_empty_rank_global_denominator_matches_single_process(tmp_path):
    import multiprocessing
    context = multiprocessing.get_context("spawn")
    processes = [context.Process(target=_gloo_empty_rank_worker,
                 args=(rank, str(tmp_path / "gloo_init"), str(tmp_path))) for rank in range(2)]
    try:
        for process in processes: process.start()
        for process in processes:
            process.join(35)
            assert not process.is_alive() and process.exitcode == 0
        for rank in range(2):
            actual = json.loads((tmp_path / f"rank_{rank}.json").read_text())
            assert actual["gradient"] == pytest.approx(2.)
            assert actual["updated_weight"] == pytest.approx(.8)
    finally:
        for process in processes:
            if process.is_alive(): process.terminate(); process.join(5)
