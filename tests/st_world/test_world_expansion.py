"""CPU migration invariants. Real S0 inference is a separate integration test."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from starVLA.model.modules.foresight.queries import SpatialViewEncoding
from tools.st_world_policy.world_expansion import expand_world_queries


class Tokenizer:
    def __init__(self):
        self.vocab = {f"<foresight_{i}>": i for i in range(144)}

    def get_vocab(self):
        return self.vocab.copy()

    def add_special_tokens(self, spec, replace_additional_special_tokens=False):
        for token in spec["additional_special_tokens"]:
            self.vocab[token] = len(self.vocab)

    def convert_tokens_to_ids(self, tokens):
        return self.vocab[tokens] if isinstance(tokens, str) else [self.vocab[t] for t in tokens]


class VLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(320, 8)
        self.config = SimpleNamespace(text_config=SimpleNamespace(initializer_range=.02))

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return self.embedding


def model():
    vlm = VLM()
    return SimpleNamespace(
        foresight_queries=nn.Parameter(torch.randn(144, 8)),
        foresight_tokens=[f"<foresight_{i}>" for i in range(144)],
        query_geometry=SpatialViewEncoding(8, (6, 8)),
        _special_token_ids={"foresight": tuple(range(144))},
        qwen_vl_interface=SimpleNamespace(model=vlm, processor=SimpleNamespace(tokenizer=Tokenizer())))


def test_w48_does_not_change_values_or_rng():
    m = model()
    original = m.query_geometry(m.foresight_queries).detach().clone()
    rng = torch.get_rng_state().clone()
    receipt = expand_world_queries(m, 48, seed=42)
    assert torch.equal(rng, torch.get_rng_state())
    assert torch.equal(m.query_geometry(m.foresight_queries), original)
    assert receipt["changed"] is False


def test_w96_keeps_original_values_ids_and_offsets_with_explicit_rows():
    m = model()
    original = m.query_geometry(m.foresight_queries).detach().clone()
    embeddings = m.qwen_vl_interface.model.embedding.weight.detach().clone()
    rng = torch.get_rng_state().clone()
    receipt = expand_world_queries(m, 96, seed=42)
    assert torch.equal(rng, torch.get_rng_state())
    assert torch.equal(m.query_geometry(m.foresight_queries)[receipt["old_to_new_query_rows"]], original)
    assert torch.equal(m.qwen_vl_interface.model.embedding.weight[:144], embeddings[:144])
    assert torch.equal(m.qwen_vl_interface.model.embedding.weight[288:], embeddings[288:])
    assert len(set(m._special_token_ids["foresight"])) == 288
    assert receipt["initial_function_equivalence"] is False
    m.query_geometry(m.foresight_queries).square().mean().backward()
    assert torch.isfinite(m.foresight_queries.grad).all()
    assert m.foresight_queries.grad[receipt["new_query_rows"]].abs().sum() > 0
    with pytest.raises(ValueError, match="inventory"):
        expand_world_queries(m, 96, seed=42)


def test_new_slots_reproducible_independent_of_outer_rng():
    m, n = model(), model()
    a = expand_world_queries(m, 96, seed=42)
    torch.randn(1000)
    b = expand_world_queries(n, 96, seed=42)
    assert torch.equal(m.foresight_queries[a["new_query_rows"]], n.foresight_queries[b["new_query_rows"]])
    assert a["new_token_initialization"]["embedding_sha256"] == b["new_token_initialization"]["embedding_sha256"]


def test_unregistered_capacity_rejected():
    with pytest.raises(ValueError, match="W48 versus W96"):
        expand_world_queries(model(), 192, seed=42)
