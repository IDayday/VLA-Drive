"""Independent bounded CPU/metadata checks; no real image/native checkpoint."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from tools.st_world_policy import spatial_probe as p


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True))
    return dict(path=str(path), sha256=p.file_sha256(path))


def metadata_protocol(tmp_path, *, leak=False):
    populations = {}
    for split,n in (('train',64),('dev',16)):
        log = 'train' if split=='train' or leak else 'dev'
        populations[split] = write(tmp_path/(split+'.json'), dict(schema='st_fixed_scene_list_v1',
            split=split, scenes=[dict(token=f'{split}_{i}',log=log) for i in range(n)]))
    labels=write(tmp_path/'labels.json',dict(label_schema=p.LABEL_SCHEMA,registration='approved'))
    protocol=p.seal(dict(schema='st_learned_w_probe_protocol_v1',recipe=p.RECIPE,
        populations=populations,labels=labels,label_schema=p.LABEL_SCHEMA,
        interpretation='synthetic metadata',common_hidden_dim=2048))
    return write(tmp_path/'protocol.json',protocol),protocol


def test_protocol_rejects_train_dev_log_leakage(tmp_path):
    ref,_=metadata_protocol(tmp_path,leak=True)
    with pytest.raises(ValueError,match='log leakage'):p.load_protocol(ref)


def test_protocol_rejects_changed_recipe_even_when_new_identity_is_valid(tmp_path):
    ref,protocol=metadata_protocol(tmp_path)
    changed=deepcopy(protocol);changed.pop('identity');changed['recipe']['updates']=301
    ref=write(Path(ref['path']),p.seal(changed))
    with pytest.raises(ValueError,match='recipe'):p.load_protocol(ref)


def test_sampled_target_order_matches_each_query_scene_view_and_pixel():
    value=torch.arange(3*3*18*24,dtype=torch.float32).reshape(3,3,18,24)
    scene=torch.tensor([2,0]);patch=torch.tensor([[[0,431],[24,407],[23,408]],[[431,0],[407,24],[408,23]]])
    actual=p.sampled_labels({'depth':value},scene,patch,torch.device('cpu'))['depth']
    expected=torch.stack([torch.stack([value[s,v,int(k)//24,int(k)%24]
        for v in range(3) for k in patch[b,v]]) for b,s in enumerate(scene)])
    assert torch.equal(actual,expected)
    views,xy,times=p.queries(patch)
    assert views.tolist()==[[0,0,1,1,2,2]]*2 and torch.count_nonzero(times)==0
    column=((xy[...,0]+1)*12-.5).round().long()
    row=((xy[...,1]+1)*9-.5).round().long()
    assert torch.equal((24*row+column),patch.flatten(1))


def test_empty_exposure_after_real_adam_state_does_not_decay_or_advance_moments():
    torch.set_num_threads(2)
    model=p.make_probe(8);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    world=torch.zeros(2,144,8);memory=torch.ones(2,144,dtype=torch.bool)
    depth=torch.full((2,3,18,24),10.);small=torch.zeros_like(depth);small[...,0,0]=1
    labels=dict(depth=depth,depth_valid=torch.ones_like(depth,dtype=torch.bool),
        small_vehicle=small,small_valid=torch.ones_like(small,dtype=torch.bool))
    stats=p.train_statistics(labels);stream=p.PatchStream(2)
    assert p.update_probe(model,opt,world,memory,labels,stats,stream)['optimizer_step']
    weights=deepcopy(model.state_dict());state=deepcopy(opt.state_dict())
    labels['depth_valid'].zero_();labels['small_valid'].zero_()
    outcome=p.update_probe(model,opt,world,memory,labels,stats,stream)
    assert stream.completed==2 and outcome['optimizer_step'] is False
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in weights.items())
    assert opt.state_dict()['param_groups']==state['param_groups']
    for key,values in state['state'].items():
        assert all(torch.equal(value,opt.state_dict()['state'][key][name]) for name,value in values.items())


def test_protocol_bytes_parsed_are_the_bytes_whose_sha_was_approved(tmp_path,monkeypatch):
    ref,approved=metadata_protocol(tmp_path)
    replacement=deepcopy(approved);replacement.pop('identity')
    replacement['labels']=write(tmp_path/'other_labels.json',dict(label_schema=p.LABEL_SCHEMA,registration='unapproved'))
    replacement=p.seal(replacement)
    path=Path(ref['path']);original=Path.open;replaced=False
    class SwapAfterRead:
        def __init__(self,stream):self.stream=stream
        def __enter__(self):self.stream.__enter__();return self
        def __exit__(self,*args):return self.stream.__exit__(*args)
        def __getattr__(self,name):return getattr(self.stream,name)
        def read(self,*args,**kwargs):
            nonlocal replaced
            content=self.stream.read(*args,**kwargs)
            if not replaced:
                replaced=True
                temp=path.with_suffix('.replacement');temp.write_text(json.dumps(replacement,sort_keys=True));temp.replace(path)
            return content
    def opening(candidate,*args,**kwargs):
        stream=original(candidate,*args,**kwargs)
        mode=args[0] if args else kwargs.get('mode','r')
        return SwapAfterRead(stream) if Path(candidate)==path and 'r' in mode else stream
    monkeypatch.setattr(Path,'open',opening)
    try:
        protocol,_,labels=p.load_protocol(ref)
    except ValueError as error:
        assert any(word in str(error).lower() for word in ('changed','pinned','hash','checksum'))
    else:
        assert protocol==approved, 'Parsed protocol was not the approved byte snapshot'
        assert labels['registration']=='approved'
    assert replaced


@pytest.mark.parametrize('excess',[0,1])
def test_pinned_control_size_limit_includes_exact_boundary(tmp_path,excess):
    path=tmp_path/'limit.json'
    path.write_bytes(b'{}'+b' '*((1<<20)+excess-2))
    ref=dict(path=str(path),sha256=p.file_sha256(path))
    if excess:
        with pytest.raises(ValueError,match='1MiB'):p.pinned(ref)
    else:
        assert p.pinned(ref)=={}


def test_pinned_control_growth_during_single_read_rejects(tmp_path,monkeypatch):
    path=tmp_path/'control.json';ref=write(path,{'approved':True})
    original=Path.open;reads=[]
    class GrowAfterRead:
        def __init__(self,stream):self.stream=stream
        def __enter__(self):self.stream.__enter__();return self
        def __exit__(self,*args):return self.stream.__exit__(*args)
        def __getattr__(self,name):return getattr(self.stream,name)
        def read(self,*args):
            reads.append(args);raw=self.stream.read(*args)
            with original(path,'ab') as out:out.write(b' '*(1<<20))
            return raw
    def opening(candidate,*args,**kwargs):
        f=original(candidate,*args,**kwargs)
        return GrowAfterRead(f) if Path(candidate)==path and args==('rb',) else f
    monkeypatch.setattr(Path,'open',opening)
    with pytest.raises(ValueError,match='Changed pinned'):p.pinned(ref)
    assert reads==[((1<<20)+1,)]


def test_pinned_control_parses_one_approved_read_without_reopening(tmp_path,monkeypatch):
    path=tmp_path/'control.json';value={'approved':True};ref=write(path,value)
    original=Path.open;calls=[]
    def opening(candidate,*args,**kwargs):
        if Path(candidate)==path:calls.append(args)
        return original(candidate,*args,**kwargs)
    monkeypatch.setattr(Path,'open',opening)
    assert p.pinned(ref)==value
    assert calls==[('rb',)]
