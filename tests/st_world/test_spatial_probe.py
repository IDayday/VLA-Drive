"""CPU tensor/interface tests only; no fake images presented as real extraction."""
import copy
import json
import subprocess
import sys

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from tools.st_world_policy import spatial_probe as p
from starVLA.dataloader.st_world_schema import ModelInputs


def labels(n=3):
    depth = torch.linspace(1., 80., 3*18*24).reshape(3,18,24).repeat(n,1,1,1)
    dv = torch.ones_like(depth, dtype=torch.bool); dv[..., ::3] = False
    small = torch.zeros_like(depth); small[..., ::7] = 1
    sv = torch.ones_like(dv); sv[..., 1::4] = False
    return dict(depth=depth, depth_valid=dv, small_vehicle=small, small_valid=sv)


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads(); torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def test_observation_only_encode_seam_has_no_target_or_forward():
    class TensorEncoder(torch.nn.Module):
        """Explicit synthetic tensor seam, never a real image/model test."""
        def __init__(self):
            super().__init__(); self.weight = torch.nn.Parameter(torch.ones(1))
        def encode_inputs(self, inputs):
            return {'W': torch.ones(len(inputs), 144, 8) * self.weight}
        def forward(self, *args):
            raise AssertionError('Training loss forward forbidden')
    obs = ModelInputs(dict(image=[object(), object(), object()], state=torch.zeros(1,4), lang='', token='a'))
    model = TensorEncoder().eval().requires_grad_(False)
    before = p.encode_world(model, [obs], tokens=144, hidden=8)
    target = labels(); target['depth'].fill_(float('nan')); target.clear()
    assert torch.equal(before, p.encode_world(model, [obs], tokens=144, hidden=8))
    with pytest.raises(TypeError):
        p.encode_world(model, [obs], targets=target, tokens=144, hidden=8)
    with pytest.raises(ValueError, match='Current input permits'):
        ModelInputs({**obs.current, 'labels': target})
    with pytest.raises(ValueError, match='ModelInputs'):
        p.encode_world(model, [dict(obs.current)], tokens=144, hidden=8)
    model.train()
    with pytest.raises(ValueError, match='frozen'):
        p.encode_world(model, [obs], tokens=144, hidden=8)


def test_queries_are_native_centers_view_order_and_current_time():
    ids = torch.tensor([[[0,431],[23,408],[24,407]]])
    view, xy, seconds = p.queries(ids)
    assert view.tolist() == [[0,0,1,1,2,2]]
    assert torch.allclose(xy[0,0], torch.tensor([-1+1/24, -1+1/18]))
    assert torch.allclose(xy[0,1], torch.tensor([1-1/24, 1-1/18]))
    model = p.make_probe(8)
    w = torch.randn(1,144,8)
    pred = model(w,view,xy,seconds)
    assert pred.shape == (1,6,2)
    with pytest.raises(ValueError, match='physical time'):
        model(w,view,xy,seconds+.5)
    with pytest.raises(ValueError, match='view identity'):
        model(w,view+3,xy,seconds)
    with pytest.raises(ValueError, match='coordinates'):
        model(w,view,xy+3,seconds)
    with pytest.raises(ValueError, match='outside'):
        p.queries(ids+432)


def test_w48_w96_same_parameters_and_isolated_initialization():
    torch.manual_seed(902)
    state = torch.get_rng_state().clone()
    a = p.make_probe(8)
    assert torch.equal(state, torch.get_rng_state())
    torch.rand(9)
    b = p.make_probe(8)
    assert sum(x.numel() for x in a.parameters()) == sum(x.numel() for x in b.parameters())
    assert set(a.state_dict()) == set(b.state_dict())
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    q = p.queries(torch.zeros(1,3,2,dtype=torch.long))
    assert a(torch.zeros(1,144,8),*q).shape == b(torch.zeros(1,288,8),*q).shape
    assert not any('288' in k or '144' in k for k in a.state_dict())


def test_task_denominators_and_zero_valid_have_finite_graph_zero():
    stats = dict(log_mean=0., log_std=1., pos_weight=3.)
    pred = torch.zeros(1,4,2,requires_grad=True)
    target = dict(depth=torch.tensor([[1.,np.e,float('nan'),float('nan')]]),
                  depth_valid=torch.tensor([[True,True,False,False]]),
                  small_vehicle=torch.tensor([[1.,0.,0.,float('nan')]]),
                  small_valid=torch.tensor([[True,True,True,False]]))
    total, losses, counts = p.task_loss(pred,target,stats)
    assert counts == dict(depth=2,small=3)
    assert losses['depth'].item() == pytest.approx(.475)
    assert losses['small'].item() == pytest.approx(5*np.log(2)/3)
    assert total.item() == pytest.approx(.475+5*np.log(2)/3)
    target['depth_valid'].zero_(); target['small_valid'].zero_()
    zero, _, counts = p.task_loss(pred,target,stats)
    zero.backward()
    assert zero.item() == 0 and torch.equal(pred.grad,torch.zeros_like(pred))
    assert counts == dict(depth=0,small=0)


def test_empty_update_keeps_probe_and_W_unchanged_and_tracks_exposure():
    model = p.make_probe(8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001,weight_decay=.001)
    w = torch.randn(3,144,8); m = torch.ones(3,144,dtype=torch.bool)
    y = labels(); stats = p.train_statistics(y)
    y['depth_valid'].zero_(); y['small_valid'].zero_()
    original = copy.deepcopy(model.state_dict()); original_w=w.clone()
    result = p.update_probe(model,optimizer,w,m,y,stats,p.PatchStream(3))
    assert result['queried_patches'] == 192 and not result['optimizer_step']
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in original.items())
    assert torch.equal(w,original_w)


def test_train_only_stats_and_label_validation():
    y=labels(); stats=p.train_statistics(y)
    expected=y['depth'][y['depth_valid']].double().log()
    assert stats['log_mean']==expected.mean().item()
    assert stats['log_std']==expected.std(unbiased=False).item()
    other=labels(); other['depth'].fill_(70)
    assert stats==p.train_statistics(y)
    y['small_valid']=y['small_valid'].float()
    with pytest.raises(ValueError,match='Boolean'):p.train_statistics(y)


def test_actual_probe_updates_frozen_W_and_exact_cpu_resume(tmp_path):
    world = torch.randn(3,144,8); original_w=world.clone()
    masks = torch.ones(3,144,dtype=torch.bool); y=labels(); stats=p.train_statistics(y)
    model = p.make_probe(8); initial=copy.deepcopy(model.state_dict())
    opt = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001); stream=p.PatchStream(3)
    exposure=[p.update_probe(model,opt,world,masks,y,stats,stream)]
    path=tmp_path/'probe.pt'; torch.save(p.checkpoint_state(model,opt,stream,'run',exposure),path)
    expected=[p.update_probe(model,opt,world,masks,y,stats,stream) for _ in range(2)]
    final=copy.deepcopy(model.state_dict()); opt_final=copy.deepcopy(opt.state_dict())
    restored=p.make_probe(8); opt2=torch.optim.AdamW(restored.parameters(),lr=.001,weight_decay=.001); stream2=p.PatchStream(3)
    state=torch.load(path,weights_only=False)
    loaded=p.restore_state(state,restored,opt2,stream2,'run')
    assert loaded==exposure
    actual=[p.update_probe(restored,opt2,world,masks,y,stats,stream2) for _ in range(2)]
    assert actual==expected
    assert all(torch.equal(v,restored.state_dict()[k]) for k,v in final.items())
    for key,values in opt_final['state'].items():
        assert all(torch.equal(v,opt2.state_dict()['state'][key][k]) for k,v in values.items())
    assert any(not torch.equal(v,final[k]) for k,v in initial.items())
    assert torch.equal(world,original_w) and world.grad is None
    with pytest.raises(ValueError,match='identity'):
        p.restore_state(state,restored,opt2,stream2,'other')


def test_stream_is_same_across_arms_and_each_view_samples_without_replacement():
    a,b=p.PatchStream(64),p.PatchStream(64)
    for _ in range(3):
        sa,pa=a.next(); sb,pb=b.next()
        assert torch.equal(sa,sb) and torch.equal(pa,pb)
        assert all(len(torch.unique(x))==32 for x in pa.flatten(0,1))


def test_cache_scene_hash_checkpoint_mismatch_rejected(tmp_path):
    sig=p.seal(dict(binding=dict(memory_tokens=144,hidden_dim=8),checkpoint='native'))
    row=dict(token='one',log='log')
    path=tmp_path/'one.safetensors'
    save_file(dict(W=torch.ones(144,8),memory_valid=torch.ones(144,dtype=torch.bool)),str(path))
    receipt=dict(identity=sig['identity'],token='one',log='log',sha256=p.file_sha256(path))
    path.with_suffix('.json').write_text(json.dumps(receipt))
    assert p.read_world(tmp_path,sig,row)[0].shape==(144,8)
    for changed in ({**sig,'identity':'other checkpoint'},{**sig,'identity':'other runtime'}):
        with pytest.raises(ValueError,match='identity'):p.read_world(tmp_path,changed,row)
    with pytest.raises(ValueError,match='identity'):p.read_world(tmp_path,sig,{**row,'log':'foreign'})
    path.write_bytes(b'changed payload')
    with pytest.raises(ValueError,match='hash'):p.read_world(tmp_path,sig,row)


def test_metrics_keep_zero_populations_and_report_scene_view_counts():
    model=p.make_probe(8); world=torch.ones(2,144,8); valid=torch.ones(2,144,dtype=torch.bool)
    y=labels(2); stats=p.train_statistics(y)
    y['depth_valid'][0].zero_(); y['small_valid'][0].zero_()
    result=p.evaluate(model,world,valid,y,stats,[dict(token='a',log='l1'),dict(token='b',log='l2')])
    assert result['scenes'][0]['depth']['valid']==0
    assert result['scenes'][0]['depth']['MAE_m'] is None
    assert result['scenes'][0]['small']['PR_AUC'] is None
    assert result['overall']['depth']['valid']==int(y['depth_valid'].sum())
    assert sum(v['small']['valid'] for v in result['views'].values())==int(y['small_valid'].sum())
    assert p.binary_metrics([0,1],[1,1])['ROC_AUC'] is None


def test_bounded_storage_estimate_and_real_cli_help():
    sizes=[p.cache_requirements(80,n,2048) for n in (144,144,144,288)]
    assert sum(x['output_bytes']+x['staging_bytes']+x['metadata_bytes'] for x in sizes)<1<<30
    result=subprocess.run([sys.executable,'-m','tools.st_world_policy.spatial_probe','--help'],capture_output=True,text=True,check=True)
    assert 'plan-cache' in result.stdout and 'extract' in result.stdout and 'fit' in result.stdout
    with pytest.raises(SystemExit):p.parser().parse_args(['extract','--binding','r','--output','o'])


def test_fit_two_slots_preserve_committed_state_when_pointer_write_is_interrupted(tmp_path, monkeypatch):
    """Synthetic tensor-only fit harness; no native reader or real image claim."""
    y=labels(); world=torch.randn(3,144,8); mask=torch.ones(3,144,dtype=torch.bool)
    rows={s:[dict(token=str(i),log=str(i)) for i in range(3)] for s in ('train','dev')}
    sets={s:(world.clone(),mask.clone(),copy.deepcopy(y)) for s in ('train','dev')}
    monkeypatch.setattr(p,'load_fit_data',lambda _:({'identity':'protocol','interpretation':'synthetic tensors'},rows,sets,{}))
    monkeypatch.setattr(p,'source_runtime',lambda *_:None)
    clock=[0.]
    monkeypatch.setattr(p.time,'monotonic',lambda:clock[0])
    original_update=p.update_probe
    def one_update(*args):
        result=original_update(*args); clock[0]+=1.; return result
    monkeypatch.setattr(p,'update_probe',one_update)
    reg=p.seal(dict(schema='st_spatial_probe_fit_v1',protocol={},caches={'train':{},'dev':{}},arm='S-P0',
                   source_sha='fixture',environment={},device='cpu',max_seconds=1.,max_output_bytes=64<<20))
    first=p.fit(reg,tmp_path)
    assert first['completed']==1
    receipt=json.loads((tmp_path/'STATE.json').read_text())
    assert receipt['file']=='probe_a.pt'
    clock[0]=0.
    second=p.fit(reg,tmp_path,resume=True)
    assert second['completed']==2
    receipt=json.loads((tmp_path/'STATE.json').read_text())
    assert receipt['file']=='probe_b.pt'
    previous_bytes=(tmp_path/'STATE.json').read_bytes()
    replace=p.os.replace
    def interrupt_pointer(src,dst):
        if str(dst).endswith('STATE.json'):raise OSError('simulated interrupted pointer publication')
        return replace(src,dst)
    monkeypatch.setattr(p.os,'replace',interrupt_pointer)
    clock[0]=0.
    with pytest.raises(OSError,match='interrupted'):
        p.fit(reg,tmp_path,resume=True)
    assert (tmp_path/'STATE.json').read_bytes()==previous_bytes
    assert p.file_sha256(tmp_path/receipt['file'])==receipt['sha256']
    uncommitted=torch.load(tmp_path/'probe_a.pt',weights_only=False)
    monkeypatch.setattr(p.os,'replace',replace)
    clock[0]=0.
    third=p.fit(reg,tmp_path,resume=True)
    assert third['completed']==3
    committed=torch.load(tmp_path/'probe_a.pt',weights_only=False)
    assert committed['exposure']==uncommitted['exposure']
    assert all(torch.equal(v,committed['model'][k]) for k,v in uncommitted['model'].items())


def test_binding_enforces_fixed_population_hash_role_and_shape(tmp_path):
    manifest=tmp_path/'manifest.json'
    payload=dict(schema='st_fixed_scene_list_v1',split='train',scenes=[dict(token=str(i),log='log') for i in range(64)])
    manifest.write_text(json.dumps(payload))
    ref=dict(path=str(manifest),sha256=p.file_sha256(manifest))
    binding=p.seal(dict(schema='st_spatial_w_binding_v1',source_sha='fixture',environment={},device='cpu',
        training_registration={},checkpoint_tag='step005000',checkpoint_identity='a'*64,arm='S-P0',
        observations=dict(current_root=str(tmp_path),manifest=ref,identity='b'*64,split='train'),
        hidden_dim=2048,memory_tokens=144,max_seconds=30.))
    assert p.validate_binding(binding)==binding
    wrong=copy.deepcopy(binding); wrong['memory_tokens']=288
    wrong=p.seal({k:v for k,v in wrong.items() if k!='identity'})
    with pytest.raises(ValueError,match='shape'):p.validate_binding(wrong)
    wrong=copy.deepcopy(binding); wrong['observations']['split']='navtest'
    wrong=p.seal({k:v for k,v in wrong.items() if k!='identity'})
    with pytest.raises(ValueError,match='train/dev'):p.validate_binding(wrong)
    payload['scenes'][0]['token']='foreign'
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError,match='pinned'):p.validate_binding(binding)
    with pytest.raises(ValueError,match='population'):p.checked_rows({**payload,'scenes':payload['scenes'][:32]},'train',64)
    with pytest.raises(ValueError,match='missing/unknown'):p.pinned({**ref,'unreviewed':True})


def test_plan_cache_metadata_never_calls_cuda_and_extract_activates_before_model(tmp_path, monkeypatch, capsys):
    """Synthetic metadata seams only; no images, native model or W cache is fabricated."""
    from contextlib import nullcontext
    from starVLA.dataloader import st_world_observation
    from tools.st_world_policy import checkpoints
    rows=[dict(token=str(i),log='log') for i in range(64)]
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps(dict(schema='st_fixed_scene_list_v1',split='train',scenes=rows)))
    training=tmp_path/'registration.json'; training.write_text('{}')
    checkpoint=dict(sha256='a'*64,completed=5000,scope='FORMAL')
    cfg=dict(space_time=dict(cell='S-P0'),data=dict(history_root=None))
    base=dict(parent_identity=dict(config=dict(framework=dict(qwenvl=dict(vl_hidden_dim=2048)))))
    class ObservationMetadata:
        def __init__(self,*_):
            self.rows=rows; self.identity=dict(identity='b'*64)
        def __len__(self):return len(self.rows)
        def __getitem__(self,_):raise AssertionError('Synthetic metadata has no real images')
    monkeypatch.setattr(st_world_observation,'STObservationDataset',ObservationMetadata)
    monkeypatch.setattr(checkpoints,'checkpoint_identity',lambda *_:(cfg,base,checkpoint))
    monkeypatch.setattr(p.subprocess,'check_output',lambda args,**_:b'' if '--porcelain' in args else 'fixture')
    def forbidden(*_,**__):raise AssertionError('Planning touched a CUDA/cuDNN API')
    for name in ('is_available','set_device','init','_lazy_init','is_initialized','current_device',
                 'device_count','get_device_properties','get_device_capability','get_rng_state',
                 'get_rng_state_all','synchronize','current_stream','manual_seed_all'):
        monkeypatch.setattr(torch.cuda,name,forbidden)
    monkeypatch.setattr(torch.backends.cudnn,'version',forbidden)
    environment=p.runtime_environment(metadata_only=True)
    environment['base']['cudnn']=12345  # Expected runtime; metadata plan must defer verification.
    binding=p.seal(dict(schema='st_spatial_w_binding_v1',source_sha='fixture',environment=environment,device='cuda:7',
        training_registration=dict(path=str(training),sha256=p.file_sha256(training)),checkpoint_tag='step005000',
        checkpoint_identity='a'*64,arm='S-P0',observations=dict(current_root=str(tmp_path),
        manifest=dict(path=str(manifest),sha256=p.file_sha256(manifest)),identity='b'*64,split='train'),
        hidden_dim=2048,memory_tokens=144,max_seconds=30.))
    signature,data,required=p.cache_plan(binding)
    assert signature['binding']['environment']==environment and len(data)==64
    assert required['output_bytes']>0 and torch.get_num_threads()==2
    binding_path=tmp_path/'binding.json'; binding_path.write_text(json.dumps(binding))
    monkeypatch.setattr(sys,'argv',['spatial_probe','plan-cache','--binding',str(binding_path),
                                   '--output',str(tmp_path/'never_created')])
    assert p.main()==0
    result=json.loads(capsys.readouterr().out)
    assert result['runtime_check']=='metadata_only'
    assert result['runtime_checks_deferred']==['cuDNN','device']
    assert result['no_cache_written'] and not (tmp_path/'never_created').exists()
    changed=copy.deepcopy(binding); changed['environment']['base']['packages']['torch']='foreign'
    changed=p.seal({k:v for k,v in changed.items() if k!='identity'})
    with pytest.raises(ValueError,match='source/runtime'):p.cache_plan(changed)
    # Actual extraction retains full runtime and device checks before invoking
    # the native loader. Only metadata files are written in this test seam.
    calls=[]
    monkeypatch.setattr(torch.backends.cudnn,'version',lambda:calls.append('cudnn_mismatch') or 99999)
    with pytest.raises(ValueError,match='source/runtime'):
        p.source_runtime('fixture',environment,'cuda:7')
    assert calls==['cudnn_mismatch']  # No device activation under a foreign runtime.
    calls.clear()
    monkeypatch.setattr(torch.backends.cudnn,'version',lambda:calls.append('cudnn') or 12345)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:calls.append('available') or True)
    monkeypatch.setattr(torch.cuda,'set_device',lambda device:calls.append(('activate',device)))
    class BudgetMetadata:
        def __init__(self,*_):pass
        def reserve(self,*_,**__):return self
        def writing(self,*_):return nullcontext()
    monkeypatch.setattr(p,'StorageBudget',BudgetMetadata)
    def no_model(*_,**__):
        calls.append('native_loader')
        raise RuntimeError('Stop before native model allocation')
    monkeypatch.setattr(checkpoints,'load_world_checkpoint',no_model)
    with pytest.raises(RuntimeError,match='before native model'):
        p.extract(binding,tmp_path/'synthetic_metadata_only',tmp_path/'unused_policy.json')
    assert calls==['cudnn','available',('activate','cuda:7'),'native_loader']
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.allow_tf32 is False
    assert torch.backends.cudnn.benchmark is False
    assert torch.are_deterministic_algorithms_enabled()
