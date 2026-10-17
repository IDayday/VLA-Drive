"""CPU fixtures test batch policy, not real DINO numerical equivalence."""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import random

import numpy as np
import pytest
import torch
from PIL import Image

from starVLA.dataloader.st_world_history import image_reference
from starVLA.dataloader.st_world_ondemand import BoundedTeacherProvider
from starVLA.dataloader.st_world_schema import identity_hash, TeacherFeatureContract
from starVLA.dataloader.st_world_targets import payload_key
from tools.st_world_policy.fixed_batch_teacher import (
    FixedBatchDINOExecutor, FIXED_BATCH_POLICY, pad_teacher_pixels, require_matching_qualification)
from tools.st_world_policy.ondemand_teacher import FrozenDINOExecutor
from test_ondemand_teacher import fixture


@pytest.fixture
def fixed_fixture(tmp_path, monkeypatch):
    from tools.st_world_policy import cache_teacher
    from starVLA.model.modules.foresight import dinov3_target_encoder as dino
    c, manifest = fixture(tmp_path)
    monkeypatch.setattr(cache_teacher, 'make_contract', lambda *a, **k:(c, {'fixture':True}, {'fixture':True}))
    class BatchSensitiveFixture(torch.nn.Module):
        """Deliberate batch-shaped arithmetic; not an approximation of DINO."""
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.p = torch.nn.Parameter(torch.rand(1), requires_grad=False)
            self.register_buffer('b', torch.zeros(1))
            self.patch_size = (16,16); self.feature_dim = 2; self.prefix = 5
            self.calls = []; self.fail = False
            random.random(); np.random.normal(); self.eval()
        def forward(self, pixels):
            assert not torch.is_autocast_enabled('cpu')
            assert pixels.dtype == torch.float32
            self.calls.append(pixels.detach().clone())
            random.random(); np.random.normal(); torch.rand(1)
            if self.fail:
                raise TimeoutError('fixture timeout after consuming RNG')
            # Image contribution is per-image. Only compute batch length is
            # intentionally different to reproduce the old partial-LRU defect.
            base = pixels[:,0,0,0] + len(pixels)
            return base[:,None,None,None].expand(-1,2,2,2).clone()
    monkeypatch.setattr(dino, 'DINOv3TargetEncoder', BatchSensitiveFixture)
    monkeypatch.setattr(torch.backends.cuda.matmul, 'allow_tf32', False)
    monkeypatch.setattr(torch.backends.cudnn, 'allow_tf32', False)
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    old = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled())
    torch.use_deterministic_algorithms(True, warn_only=False)
    try:
        yield c, manifest
    finally:
        torch.use_deterministic_algorithms(old[0], warn_only=old[1])


def rows(c, m):
    refs = [f['views'][0] for f in m['scenes'][0]['frames']]
    return [{'key':payload_key(ref,c), 'image':ref} for ref in refs]


def provider(c, m, executor):
    return BoundedTeacherProvider(m, c, executor=executor,
        execution_identity=executor.execution_identity, max_cached_bytes=72,
        max_scene_bytes=10000, max_batch_images=8, max_manifest_bytes=100000)


def rng_state():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())


def assert_rng_same(a,b):
    assert a[0] == b[0] and a[1][0] == b[1][0] and a[1][2:] == b[1][2:]
    assert np.array_equal(a[1][1], b[1][1]) and torch.equal(a[2], b[2])


def test_fixed_batches_return_only_actual_images_and_preserve_rng(fixed_fixture, tmp_path):
    c,m = fixed_fixture
    before = rng_state()
    e = FixedBatchDINOExecutor(tmp_path, contract=c, variant='SP2', device='cpu')
    assert_rng_same(before, rng_state())
    all_rows = rows(c,m)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        one = e.compute_batch(all_rows[:1], contract=c, execution_identity=e.execution_identity)
        four = e.compute_batch(all_rows[:4], contract=c, execution_identity=e.execution_identity)
    assert one.keys == (all_rows[0]['key'],) and len(one.features) == 1
    assert len(four.features) == 4 and four.valid.all()
    assert torch.equal(one.features[0], four.features[0])
    assert one.features.dtype == torch.float32 and not one.features.requires_grad
    assert all(len(x) == 8 for x in e.model.calls)
    assert torch.equal(e.model.calls[0], e.model.calls[0][:1].expand(8,-1,-1,-1))
    assert torch.equal(e.model.calls[1][4:], e.model.calls[1][3:4].expand(4,-1,-1,-1))
    assert_rng_same(before, rng_state())
    assert e.compute_stats == {'calls':2, 'requested_images':5, 'compute_image_slots':16, 'padding_image_slots':11}


def test_partial_lru_changes_legacy_targets_but_fixed8_restores_exactly(fixed_fixture, tmp_path):
    c,m = fixed_fixture
    old = FrozenDINOExecutor(tmp_path, contract=c, variant='SP2', device='cpu', max_batch_images=8)
    p = provider(c,m,old)
    first = p.read_scene('one')
    partial = p.read_scene('one')
    state = p.state_dict(); p.load_state_dict(state)
    resumed = p.read_scene('one')
    assert p.stats['ram_hits'] > 0 and p.stats['evictions'] > 0
    assert not torch.equal(first['current_dense_features'], partial['current_dense_features'])
    assert torch.equal(first['current_dense_features'], resumed['current_dense_features'])
    fixed = FixedBatchDINOExecutor(tmp_path, contract=c, variant='SP2', device='cpu')
    p = provider(c,m,fixed)
    before = rng_state()
    first = p.read_scene('one'); partial = p.read_scene('one')
    p.load_state_dict(p.state_dict()); resumed = p.read_scene('one')
    for name in ('current_dense_features','future_dense_features','current_valid','future_valid'):
        assert torch.equal(first[name], partial[name]) and torch.equal(first[name], resumed[name])
    assert p.stats['ram_hits'] > 0 and p.stats['evictions'] > 0
    assert_rng_same(before, rng_state())
    assert fixed.execution_identity != old.execution_identity
    assert fixed.execution['batch_policy'] == FIXED_BATCH_POLICY
    assert fixed.compute_stats['compute_image_slots'] == 24  # 5,3,5 actual records, always 8 compute slots.
    with pytest.raises(ValueError,match='identity'):
        p.load_state_dict({'schema':'st_bounded_teacher_state_v1', 'identity':'0'*64})


def test_padding_is_tensor_only_and_rejects_wrong_precision_or_size():
    x = torch.randn(8,3,2,2)
    assert pad_teacher_pixels(x) is x
    for n in (1,4,7):
        out = pad_teacher_pixels(x[:n])
        assert out.shape == x.shape and torch.equal(out[:n],x[:n])
        assert torch.equal(out[n:],x[n-1:n].expand(8-n,-1,-1,-1))
    for bad in (x[:0],torch.randn(9,3,2,2),x.bfloat16(),x.clone().requires_grad_(),x*float('nan')):
        with pytest.raises(ValueError,match='finite detached FP32'):
            pad_teacher_pixels(bad)


def test_error_rng_frozen_and_bad_identity_guards(fixed_fixture, tmp_path):
    c,m = fixed_fixture; e = FixedBatchDINOExecutor(tmp_path, contract=c, variant='SP2', device='cpu')
    group = rows(c,m)[:1]
    for wrong in ((),group*9):
        with pytest.raises(ValueError,match='budget'):
            e.compute_batch(wrong, contract=c, execution_identity=e.execution_identity)
    with pytest.raises(ValueError,match='budget'):
        e.compute_batch(group, contract=c, execution_identity='0'*64)
    with pytest.raises(ValueError,match='keys'):
        e.compute_batch(group*2, contract=c, execution_identity=e.execution_identity)
    assert e.compute_stats['calls'] == 0
    before = rng_state(); e.model.fail = True
    with pytest.raises(TimeoutError):
        e.compute_batch(group, contract=c, execution_identity=e.execution_identity)
    assert_rng_same(before,rng_state())
    assert e.compute_stats['compute_image_slots'] == 8  # Failed actual forward still costs a batch.
    e.model.fail = False
    with torch.no_grad(): e.model.b.add_(1)
    with pytest.raises(ValueError,match='mutated'):
        e.compute_batch(group, contract=c, execution_identity=e.execution_identity)


def test_fixed_policy_refuses_backend_drift_and_worker(fixed_fixture, tmp_path, monkeypatch):
    c,m = fixed_fixture; e = FixedBatchDINOExecutor(tmp_path, contract=c, variant='SP2', device='cpu')
    group = rows(c,m)[:1]
    with ThreadPoolExecutor(1) as pool:
        with pytest.raises(RuntimeError,match='owning'):
            pool.submit(e.compute_batch,group,contract=c,execution_identity=e.execution_identity).result()
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':16:8')
    with pytest.raises(ValueError,match='backend changed'):
        e.compute_batch(group, contract=c, execution_identity=e.execution_identity)
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True, warn_only=True)
    with pytest.raises(ValueError,match='backend changed'):
        e.compute_batch(group, contract=c, execution_identity=e.execution_identity)


def test_real_record_order_batch8_and_repeated_sample_are_preserved(fixed_fixture,tmp_path):
    c,m=fixed_fixture;group=rows(c,m)
    for i in range(5,8):
        path=tmp_path/f'{i}.png';Image.new('RGB',(32,18),(i,0,0)).save(path)
        image=image_reference(path);group.append({'key':payload_key(image,c),'image':image})
    e=FixedBatchDINOExecutor(tmp_path,contract=c,variant='SP2',device='cpu')
    original=e.compute_batch(group,contract=c,execution_identity=e.execution_identity)
    reverse=e.compute_batch(list(reversed(group)),contract=c,execution_identity=e.execution_identity)
    assert original.keys==tuple(reversed(reverse.keys))
    assert torch.equal(original.features,reverse.features.flip(0))
    repeated=e.compute_batch(group[:1],contract=c,execution_identity=e.execution_identity)
    assert torch.equal(repeated.features[0],original.features[0])
    assert e.compute_stats['requested_images']==17 and e.compute_stats['compute_image_slots']==24


def test_precision_nondeterminism_and_cross_image_operator_are_rejected(fixed_fixture,tmp_path,monkeypatch):
    c,_=fixed_fixture
    bad=c.to_dict();bad['storage_dtype']='float16'
    with pytest.raises(ValueError,match='FP32'):
        FixedBatchDINOExecutor(tmp_path,contract=TeacherFeatureContract.from_dict(bad),variant='SP2',device='cpu')
    torch.use_deterministic_algorithms(False)
    with pytest.raises(ValueError,match='deterministic'):
        FixedBatchDINOExecutor(tmp_path,contract=c,variant='SP2',device='cpu')
    torch.use_deterministic_algorithms(True)
    from starVLA.model.modules.foresight import dinov3_target_encoder as dino
    old=dino.DINOv3TargetEncoder
    class WithBatchNorm(old):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw);self.extra=torch.nn.BatchNorm2d(2).eval().requires_grad_(False)
    monkeypatch.setattr(dino,'DINOv3TargetEncoder',WithBatchNorm)
    with pytest.raises(ValueError,match='BatchNorm'):
        FixedBatchDINOExecutor(tmp_path,contract=c,variant='SP2',device='cpu')


@pytest.mark.parametrize('batch', [1,4,16,True])
def test_fixed_policy_has_no_silent_batch_override(fixed_fixture,tmp_path,batch):
    c,_ = fixed_fixture
    with pytest.raises(ValueError,match='exactly 8'):
        FixedBatchDINOExecutor(tmp_path,contract=c,variant='SP2',device='cpu',max_batch_images=batch)


def test_qualification_requires_exact_new_policy_identity():
    value = {'schema':'st_fixed_teacher_qualification_v1', 'execution_identity':'a'*64,
             'batch_policy':deepcopy(FIXED_BATCH_POLICY), 'fixed_policy_repeatability':True,
             'completed_all_cases':True, 'backend':{'deterministic_algorithms':True},
             'legacy_reference_compatibility':False}
    def signed(v):
        v['identity'] = identity_hash({k:x for k,x in v.items() if k!='identity'}); return v
    require_matching_qualification(signed(value),'a'*64)
    # This is only fixed-policy qualification, explicitly NOT legacy cache approval.
    for field,bad in [('execution_identity','b'*64),('completed_all_cases',False),
                      ('fixed_policy_repeatability',False),('batch_policy',{'compute_batch_images':4}),
                      ('schema','st_ondemand_teacher_qualification_v1')]:
        changed=deepcopy(value);changed[field]=bad
        with pytest.raises(ValueError,match='qualification'):
            require_matching_qualification(signed(changed),'a'*64)
    changed=deepcopy(value);changed['legacy_reference_compatibility']=True
    with pytest.raises(ValueError,match='qualification'):
        require_matching_qualification(changed,'a'*64)
