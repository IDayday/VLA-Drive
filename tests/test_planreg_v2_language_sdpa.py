"""Native production Qwen2 SDPA, not a reimplemented attention reference."""
import copy
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from navsim.agents.EpisodeDrive.planreg_v2.language import (
    configure_language_attention, inject_language_lora, SoftTaskQueries)


def model():
    config = Qwen2Config(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=64,
        attention_dropout=0., max_position_embeddings=512)
    config._attn_implementation = 'eager'
    result = Qwen2ForCausalLM(config).eval().requires_grad_(False)
    inject_language_lora(result, rank=4, alpha=8)
    return result


@pytest.mark.parametrize('seed', [0, 19])
def test_native_sdpa_query_forward_gradient_padding_and_two_updates(seed):
    torch.manual_seed(seed)
    eager = model(); sdpa = copy.deepcopy(eager)
    original_keys = list(sdpa.state_dict())
    original_ids = {n: id(p) for n,p in sdpa.named_parameters()}
    audit = configure_language_attention(sdpa, 'sdpa')
    assert audit['layers'] == 2 and list(sdpa.state_dict()) == original_keys
    assert original_ids == {n:id(p) for n,p in sdpa.named_parameters()}
    eq = SoftTaskQueries(64); sq = copy.deepcopy(eq)
    prefix = torch.randn(2,11,64)
    valid = torch.tensor([[1]*11, [0,0,1,1,1,1,1,0,0,0,0]], dtype=torch.bool)
    parameters = lambda m,q: [p for p in list(m.parameters())+list(q.parameters()) if p.requires_grad]
    eo = torch.optim.AdamW(parameters(eager,eq), lr=1e-3)
    so = torch.optim.AdamW(parameters(sdpa,sq), lr=1e-3)
    old_a = {n:p.detach().clone() for n,p in sdpa.named_parameters() if 'lora_a' in n}
    for step in range(2):
        eo.zero_grad(); so.zero_grad()
        x=prefix.clone().requires_grad_(); y=prefix.clone().requires_grad_()
        left=eq(eager,x,valid); right=sq(sdpa,y,valid)
        torch.testing.assert_close(left,right,atol=2e-6,rtol=1e-4)
        left.square().mean().backward(); right.square().mean().backward()
        torch.testing.assert_close(x.grad,y.grad,atol=2e-6,rtol=1e-3)
        assert y.grad[~valid].count_nonzero() == 0 and y.grad[valid].norm() > 0
        for a,b in zip(parameters(eager,eq),parameters(sdpa,sq)):
            torch.testing.assert_close(a.grad,b.grad,atol=2e-6,rtol=1e-3)
            assert b.dtype == torch.float32 and torch.isfinite(b.grad).all()
        eo.step(); so.step()
    assert all(not torch.equal(old_a[n],p) for n,p in sdpa.named_parameters() if n in old_a)
    assert all(p.grad is None for n,p in sdpa.named_parameters() if not p.requires_grad)
    changed = prefix.clone(); changed[~valid] = 10000
    torch.testing.assert_close(sq(sdpa,prefix,valid),sq(sdpa,changed,valid),atol=0,rtol=0)


def test_native_sdpa_remains_causal():
    torch.manual_seed(5); llm=model(); configure_language_attention(llm,'sdpa')
    prefix=torch.randn(2,13,64); changed=prefix.clone();changed[:,8:] *= -17
    def forward(x):
        return llm.model(inputs_embeds=x,attention_mask=torch.ones(2,13,dtype=torch.bool),
                         use_cache=False,return_dict=True).last_hidden_state
    torch.testing.assert_close(forward(prefix)[:,:8],forward(changed)[:,:8],atol=0,rtol=0)


def test_language_backend_refuses_unknown_or_wrong_topology():
    with pytest.raises(ValueError,match='backend'):
        configure_language_attention(torch.nn.Linear(2,2),'unknown')
    with pytest.raises(TypeError,match='Qwen2ForCausalLM'):
        configure_language_attention(torch.nn.Linear(2,2),'sdpa')
