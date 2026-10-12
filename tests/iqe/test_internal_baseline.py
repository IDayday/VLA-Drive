from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from iqe.query_base import source_trajectory_contract
from iqe.internal_baseline.scorer import InternalScorer, initialization_rng
from iqe.internal_baseline.sampling import TwoStreamSampler, ExpertSampler
from iqe.internal_baseline import objectives
from iqe.internal_baseline.evaluation import selection_metrics, argmax_base_tie
from iqe.internal_baseline.features import ScoreContextBundle, encode_joint
from iqe.contracts import FeatureBundle


def context(b=3):
    return {"world": torch.randn(b, 6, 8), "action": torch.randn(b, 4, 8), "current_ego": torch.randn(b, 4),
            "world_valid": torch.ones(b, 6, dtype=torch.bool), "action_valid": torch.ones(b, 4, dtype=torch.bool)}


def model(seed=42):
    with initialization_rng(seed):
        return InternalScorer(8, 8, source_trajectory_contract(), hidden=16, dropout=0.)


def batch(n=3):
    return {"scores": torch.tensor([[0.02, .97], [.5, .52], [.4, .4]])[:n],
            "valid": torch.ones(n, 2, dtype=torch.bool), "trajectories": torch.randn(n, 2, 8, 3),
            "components": torch.full((n, 2, 6), .5), "component_valid": torch.ones(n, 2, 6, dtype=torch.bool),
            "natural": torch.ones(n, dtype=torch.bool), "comparison": torch.ones(n, dtype=torch.bool)}


def test_gap_cost_survives_k2_normalization():
    p = torch.tensor([[.4, .6], [.4, .6]], requires_grad=True)
    y = torch.tensor([[0., .95], [.4, .42]])
    term = objectives.weighted_population_ranking(p, y, torch.ones_like(y, dtype=torch.bool))
    (term.numerator/term.denominator).backward()
    assert p.grad[0, 1] < 0 and p.grad[0, 0] > 0
    assert float(p.grad[0, 1]/p.grad[1, 1]) == pytest.approx(.95/.02, rel=1e-5)


def test_comparison_stream_no_second_weight_and_no_pointwise_leakage():
    b = batch(2); b["natural"][:] = False
    p = torch.tensor([[.4, .6], [.4, .6]], requires_grad=True)
    result = objectives.terms({"values": p, "component_logits": torch.zeros(2, 2, 6)}, b)
    assert result["value"].denominator == result["components"].denominator == 0
    result["ranking"].numerator.backward()
    torch.testing.assert_close(p.grad[0], p.grad[1], atol=1e-7, rtol=1e-6)


def test_global_microbatch_denominators_equal_full_gradient():
    b = batch(); b["natural"][1] = False; b["valid"][0, 0] = False
    b["component_valid"][2, 0, :3] = False
    p = torch.rand(3, 2, requires_grad=True); c = torch.rand(3, 2, 6, requires_grad=True)
    den = objectives.denominators(b)
    t = objectives.terms({"values": p, "component_logits": c}, b)
    sum(t[k].numerator/den[k].clamp_min(1) for k in t).backward()
    expected = [p.grad.clone(), c.grad.clone()]
    p.grad = c.grad = None
    for start, end in ((0, 1), (1, 3)):
        small = {k: v[start:end] for k, v in b.items()}
        terms = objectives.terms({"values": p[start:end], "component_logits": c[start:end]}, small)
        sum(terms[k].numerator/den[k].clamp_min(1) for k in terms).backward()
    torch.testing.assert_close(p.grad, expected[0]); torch.testing.assert_close(c.grad, expected[1])


def test_ties_invalid_zero_denominator_graph_connected():
    b = batch(); b["valid"][:] = False
    p = torch.rand(3, 2, requires_grad=True); c = torch.randn(3, 2, 6, requires_grad=True)
    t = objectives.terms({"values": p, "component_logits": c}, b)
    loss = sum(x.numerator/x.denominator.clamp_min(1) for x in t.values())
    loss.backward(); assert loss == 0 and p.grad is not None and c.grad is not None
    assert objectives.comparison_weights(torch.tensor([[.4, .405], [.2, .2]]), torch.ones(2, 2, dtype=torch.bool)).sum() == 0


def test_natural_value_uses_smooth_l1_not_unscaled_huber():
    b = batch(1); b["comparison"][:] = False
    p = torch.tensor([[.4, .6]])
    t = objectives.terms({"values": p, "component_logits": torch.zeros(1, 2, 6)}, b)
    expected = torch.nn.functional.smooth_l1_loss(p, b["scores"], beta=.1, reduction="sum")
    torch.testing.assert_close(t["value"].numerator, expected)


def test_two_stream_resume_keeps_consumed_not_prefetch_cursor():
    s = TwoStreamSampler(["a", "b", "c"], [.95, .02, 0.], natural_seed=4, comparison_seed=7, pointwise_updates=1, bank_hash="bank")
    first = s.peek(128); assert all(e["natural"] for e in first)
    s.consume(128); state = deepcopy(s.state_dict()); following = s.peek(128)
    restored = TwoStreamSampler(["a", "b", "c"], [.95, .02, 0.], natural_seed=4, comparison_seed=7, pointwise_updates=1, bank_hash="bank")
    restored.load_state_dict(state)
    assert restored.peek(128) == following
    assert sum(e["natural"] for e in following) == sum(e["comparison"] for e in following) == 64
    assert all(e["scene_id"] != "c" for e in following if e["comparison"])
    changed = TwoStreamSampler(["a", "b", "c"], [.8, .02, 0.], natural_seed=4, comparison_seed=7, pointwise_updates=1, bank_hash="bank")
    with pytest.raises(ValueError, match="distribution"):
        changed.load_state_dict(state)


def test_comparison_q_expected_gradient_ratio_and_absent_signal():
    s = TwoStreamSampler(["large", "small"], [.95, .02], natural_seed=5, comparison_seed=6, pointwise_updates=0, bank_hash="x")
    assert s.q[0]/s.q[1] == pytest.approx(.95/.02)
    z = TwoStreamSampler(["tie"], [0], natural_seed=1, comparison_seed=2, pointwise_updates=0, bank_hash="z")
    assert all(e["natural"] and not e["comparison"] for e in z.peek(128))


def test_expert_90_10_cumulative_and_resume():
    s = ExpertSampler(["h1", "h2"], ["n1"], 1, "m")
    counts = {"hard": 0, "ordinary": 0}
    for _ in range(10):
        for e in s.peek(128): counts[e["bucket"]] += 1
        s.consume(128)
    assert counts == {"hard": 1152, "ordinary": 128}
    r = ExpertSampler(["h1", "h2"], ["n1"], 1, "m"); r.load_state_dict(s.state_dict())
    assert r.peek(128) == s.peek(128)


def test_scorer_permutation_duplicate_append_and_context_projection_once():
    m = model().eval(); c = context(); t = torch.randn(3, 2, 8, 3)
    calls = []
    hook = m.world_projection.register_forward_hook(lambda *args: calls.append(1))
    with torch.no_grad():
        p = m(c, t)["values"]
    assert len(calls) == 1; hook.remove()
    with torch.no_grad():
        swap = m(c, t.flip(1))["values"]
        append = m(c, torch.cat((t, t[:, :1]), 1))["values"]
        one = m(c, t[:, :1])["values"]
    torch.testing.assert_close(p, swap.flip(1), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(p, append[:, :2], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(append[:, 0], append[:, 2], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(one[:, 0], p[:, 0], atol=1e-6, rtol=1e-6)


def test_generator_detached_and_scorer_receives_gradients():
    m = model(); c = context(); c["world"].requires_grad_(); c["action"].requires_grad_()
    t = torch.randn(3, 2, 8, 3, requires_grad=True)
    out = m(c, t)
    (out["values"].sum()+out["component_logits"].sum()).backward()
    assert c["world"].grad is None and c["action"].grad is None and t.grad is None
    for name in ("world_projection.1.weight", "action_projection.1.weight", "ego_projection.0.weight", "trajectory_encoder.0.weight", "value_head.weight", "component_head.weight"):
        assert dict(m.named_parameters())[name].grad.norm() > 0


def test_init_reproducible_external_rng_preserved():
    state = torch.get_rng_state().clone()
    a, b, c = model(42), model(42), model(43)
    assert torch.equal(state, torch.get_rng_state())
    assert all(torch.equal(a.state_dict()[k], b.state_dict()[k]) for k in a.state_dict())
    assert any(not torch.equal(a.state_dict()[k], c.state_dict()[k]) for k in a.state_dict())


def test_score_context_rejects_targets_and_nonfinite_or_trainable_inputs():
    m = model(); c = context()
    with pytest.raises(ValueError, match="whitelist"):
        m(c | {"target": torch.zeros(1)}, torch.zeros(3, 2, 8, 3))
    c["world"].requires_grad_()
    with pytest.raises(ValueError, match="frozen"):
        ScoreContextBundle(**c, scene_ids=("a", "b", "c"), identity="x")


def test_invalid_candidate_and_base_tie_metrics():
    m = model().eval(); t = torch.zeros(3, 2, 8, 3); t[0, 1, 0, 0] = float("nan")
    out = m(context(), t); assert not out["valid"][0, 1] and torch.isfinite(out["values"]).all()
    ix = argmax_base_tie(np.array([[.5, .5], [.4, .5]])); assert ix.tolist() == [0, 1]
    metrics = selection_metrics(np.array([[.8, .8], [1., 0.]]), np.ones((2, 2, 6)), ix, ["a", "b"], bootstrap_repetitions=20)
    assert metrics["high_to_zero"] == 1 and metrics["negative_switch_cost_total"] == 1.
    assert metrics["oracle_capture"] is None
