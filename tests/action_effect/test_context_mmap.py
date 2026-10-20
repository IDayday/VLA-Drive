import torch


def test_lossless_mmap_records_and_private_pages(tmp_path):
    path=tmp_path/'shard.pt';x=[{'scene_id':str(i),'memory':torch.randn(152,2048),'raw':torch.randn(8,4)} for i in range(4)]
    torch.save(x,path);ordinary=torch.load(path,weights_only=False);mapped=torch.load(path,weights_only=False,mmap=True)
    for a,b in zip(ordinary,mapped):
        assert a['scene_id']==b['scene_id']
        assert torch.equal(a['memory'],b['memory']) and torch.equal(a['raw'],b['raw'])
    # A consumer cannot corrupt the immutable file through MAP_PRIVATE storage.
    mapped[0]['memory'].add_(7)
    restored=torch.load(path,weights_only=False,mmap=True)
    assert torch.equal(restored[0]['memory'],ordinary[0]['memory'])
