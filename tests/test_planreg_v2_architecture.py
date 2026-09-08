import torch
from navsim.agents.EpisodeDrive.planreg_v2.language import append_task_queries, LanguageAttentionLoRA
from navsim.agents.EpisodeDrive.planreg_v2.memory import pad_tile_registers, RichSceneMemory
from navsim.agents.EpisodeDrive.planreg_v2.action import V2ActionDecoder
from navsim.agents.EpisodeDrive.planreg_v2.normalizers import measured_statistics


def test_query_prefix_compaction_and_lora_two_steps():
    prefix = torch.randn(2,7,12); mask = torch.tensor([[0,0,1,1,1,1,1],[1,1,1,0,0,0,0]]).bool()
    queries = torch.randn(16,12,requires_grad=True)
    embeds, valid, positions, indices = append_task_queries(prefix,mask,queries)
    assert indices[0,0] == 5 and indices[1,0] == 3
    assert valid.sum(-1).tolist() == [21,19]
    embeds.sum().backward(); assert queries.grad.abs().sum() > 0
    layer = LanguageAttentionLoRA(torch.nn.Linear(12,8),rank=4,alpha=8)
    optimizer = torch.optim.AdamW([p for p in layer.parameters() if p.requires_grad],lr=.01)
    x = torch.randn(3,12)
    torch.testing.assert_close(layer(x),layer.base_layer(x))
    before = layer.lora_a.weight.detach().clone()
    for i in range(2):
        optimizer.zero_grad(); layer(x).square().mean().backward()
        if i == 0: assert layer.lora_a.weight.grad.count_nonzero() == 0
        else: assert layer.lora_a.weight.grad.norm() > 0
        optimizer.step()
    assert not torch.equal(before,layer.lora_a.weight)


def test_rich_padding_and_scorer_gradient_permutation():
    torch.manual_seed(2)
    registers = torch.randn(3,16,256,requires_grad=True)
    geometry = torch.rand(3,5)
    content, geom, valid = pad_tile_registers(registers,[1,2],geometry)
    memory_module = RichSceneMemory().eval()
    semantic=torch.randn(2,16,256)
    memory, valid = memory_module(content,geom,valid,semantic)
    joint=torch.randperm(32)
    permuted,_=memory_module(content[:,joint],geom[:,joint],valid[:,joint],semantic)
    torch.testing.assert_close(permuted,memory[:,joint])
    n = measured_statistics([('unit',torch.randn(8,3),torch.ones(8,dtype=torch.bool))],'train','unit')
    action = V2ActionDecoder(n).eval()
    output = action(memory,valid,torch.randn(2,8))
    assert output['stage_proposals'].shape == (4,2,64,8,3)
    assert len([name for name,_ in action.named_modules() if name == 'trajectory_head']) == 1
    proposals = torch.randn(2,64,8,3,requires_grad=True)
    ego = torch.randn(2,256)
    logits, scores = action.score(proposals,memory,valid,ego)
    perm = torch.randperm(32)
    _, shuffled = action.score(proposals,memory[:,perm],valid[:,perm],ego)
    torch.testing.assert_close(scores,shuffled,atol=1e-6,rtol=1e-5)
    candidate_permutation=torch.randperm(64)
    reordered_logits,reordered_scores=action.score(proposals[:,candidate_permutation],memory,valid,ego)
    torch.testing.assert_close(reordered_scores,scores[:,candidate_permutation],atol=2e-6,rtol=1e-5)
    for name,value in logits.items():
        torch.testing.assert_close(reordered_logits[name],value[:,candidate_permutation],atol=2e-6,rtol=1e-5)
    sum(v.square().mean() for v in logits.values()).backward()
    assert proposals.grad is None and registers.grad.norm() > 0
    assert all(p.grad is None for p in action.attention.parameters())
