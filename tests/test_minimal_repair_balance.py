"""Fixed training-exposure weighting cannot depend on microbatch composition."""
from types import SimpleNamespace
import json
import pytest
import torch
from tools.decision_local.minimal_repair_training import repair_trajectory_loss,load_repair_class_balance,label_semantics
from tools.decision_local.common import digest,identity
from starVLA.model.modules.structured_world.losses import final_trajectory_loss


def target(gt, valid, status):
    return SimpleNamespace(repair_target=gt,repair_target_valid=valid,offline_case_status=status)


def test_fixed_class_weighted_reduction_and_gradient_across_microbatches():
    generator=torch.Generator().manual_seed(501)
    x=torch.randn(7,8,4,generator=generator,requires_grad=True)
    gt=torch.randn(7,8,4,generator=generator)
    valid=torch.ones(7,8,dtype=torch.bool);valid[2]=False;valid[4,4:]=False
    status=torch.tensor([0,1,2,0,1,0,0]);weights=(.51,32.5)
    eligible=int(valid.any(-1).sum())
    expected=sum(final_trajectory_loss(x[i:i+1],gt[i:i+1],valid[i:i+1])[0]*
                 (weights[int(status[i])] if status[i]<2 else 0.) for i in range(7))/eligible
    expected_gradient=torch.autograd.grad(expected,x,retain_graph=True)[0]
    split=sum(repair_trajectory_loss(x[a:b],target(gt[a:b],valid[a:b],status[a:b]),weights)['total']*
              (b-a)/eligible for a,b in ((0,2),(2,3),(3,6),(6,7)))
    actual_gradient=torch.autograd.grad(split,x)[0]
    torch.testing.assert_close(split,expected,rtol=2e-6,atol=1e-6)
    torch.testing.assert_close(actual_gradient,expected_gradient,rtol=2e-6,atol=1e-6)


def test_default_loss_and_empty_class_keep_native_behavior():
    x=torch.tensor([[[1.,2.,0.,1.]],[[3.,4.,1.,0.]]],requires_grad=True)
    gt=torch.zeros_like(x);valid=torch.ones(2,1,dtype=torch.bool)
    t=target(gt,valid,torch.zeros(2,dtype=torch.long))
    original=final_trajectory_loss(x,gt,valid)[0]
    torch.testing.assert_close(repair_trajectory_loss(x,t)['total'],original,rtol=0,atol=0)
    weighted=repair_trajectory_loss(x,t,(.5,30.))
    torch.testing.assert_close(weighted['total'],original*.5,rtol=0,atol=0)
    assert weighted['trajectory_repair'].item()==0
    empty=repair_trajectory_loss(x,target(gt,torch.zeros_like(valid),t.offline_case_status),(.5,30.))['total']
    assert empty.item()==0
    assert torch.count_nonzero(torch.autograd.grad(empty,x)[0])==0


def test_receipt_rejects_foreign_fold_modified_counts_and_log_arm(tmp_path):
    meta={k:{} for k in ('schema','recovery','constraints','solver','query_config','observed_free_support','source_files')}
    meta['identity']='labels';fold={'fold':1};index=[{'token':'training-only'}]
    train=SimpleNamespace(label_identity=meta,fold_identity=fold,index=index)
    evidence=tmp_path/'COMPLETE.json';counts={'KEEP':90,'REPAIR':10,'target_eligible':100}
    evidence.write_text(json.dumps({'exposure_counts':counts}))
    receipt={'schema':'minimal_repair_class_balance_v1','labels':'labels',
        'label_semantics':label_semantics(meta),'fold':fold,'training_index_sha256':identity(index),
        'training_exposure_source':str(evidence),'training_exposure_source_sha256':digest(evidence),
        'counts':counts,'weights':{'KEEP':100/180,'REPAIR':5.}}
    path=tmp_path/'balance.json';path.write_text(json.dumps(receipt))
    assert load_repair_class_balance(path,'MIN_FUT',train)[1]==(100/180,5.)
    with pytest.raises(ValueError,match='MIN follow-up'):load_repair_class_balance(path,'LOG_FUT',train)
    train.fold_identity={'fold':0}
    with pytest.raises(ValueError,match='population'):load_repair_class_balance(path,'MIN_FUT',train)
    train.fold_identity=fold;receipt['weights']['REPAIR']=1000.;path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='weights'):load_repair_class_balance(path,'MIN_FUT',train)
    receipt['weights']['REPAIR']=5.;path.write_text(json.dumps(receipt));evidence.write_text('{}')
    with pytest.raises(ValueError,match='evidence changed'):load_repair_class_balance(path,'MIN_FUT',train)
