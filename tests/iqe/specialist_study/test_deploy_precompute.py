from dataclasses import asdict
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from conftest import scene
from iqe.io import atomic_json,digest,file_hash
from iqe.specialist_study.rich_precompute import batch as rich_batch
from iqe.specialist_study.deploy_precompute import write_batch,read_batch,verified_rich_values


def setup(tmp_path):
    config=SimpleNamespace(root=tmp_path,identity='fixture');scenes=[scene(i) for i in range(4)]
    atomic_json(tmp_path/'BASE_BINDING.json',{'contract_hash':'fixture_contract','bound_at_utc':'fixture_time'})
    atomic_json(tmp_path/'OFFLINE_BASE_CONTRACT.json',{'trajectory':{'horizon':8,'dt':.5,
                'raw_representation':'normalized_xy_sincos','frame':'ego_relative'}})
    cp=tmp_path/'checkpoint';cp.write_bytes(b'fixture weights')
    source={'expert_id':'EXT_interaction_raw_3000','model_hash':'fixture_weights','checkpoint':str(cp),'checksum':file_hash(cp)}
    item={**source,'expert_id':'expert_2','reuse_rich_source':source}
    plan={'hash':'deployment_plan','rich_plan_hash':'rich_plan','scene_ids':[s.scene_id for s in scenes]}
    class Model:
        calls=0
        def eval(self):return self
        def __call__(self,x):
            self.calls+=1
            return SimpleNamespace(physical=torch.arange(len(x)*8*3,dtype=torch.float32).reshape(len(x),8,3))
    model=Model();cache=SimpleNamespace(batch=lambda sub,device:sub)
    rows,_=rich_batch(config,{'hash':'rich_plan','scene_ids':plan['scene_ids']},source,scenes,0,model,'cpu',cache)
    return config,scenes,item,plan,model,cache,rows


def test_selected_raw_endpoint_reuses_exact_bytes_without_model_call(tmp_path):
    config,scenes,item,plan,model,cache,rich=setup(tmp_path)
    rows,reused=write_batch(config,plan,item,scenes,0,None,'cpu',None)
    assert not reused and model.calls==1
    assert [r.trajectory_hash for r in rows]==[r.trajectory_hash for r in rich]
    assert all(r.expert_id=='expert_2' for r in rows)
    assert [r.key for r in rows]!=[r.key for r in rich]
    again,reused=write_batch(config,plan,item,scenes,0,None,'cpu',None)
    assert reused and rows==again
    with pytest.raises(Exception,match='layout'):read_batch(config,plan,item,list(reversed(scenes)),0)
    bad={**item,'reuse_rich_source':{**item['reuse_rich_source'],'model_hash':'wrong'}}
    with pytest.raises(Exception,match='same checkpoint'):verified_rich_values(config,plan,bad,scenes,0)


def test_new_ema_column_computes_once_and_retains_original_partial_records(tmp_path):
    config,scenes,item,plan,model,cache,_=setup(tmp_path)
    item={**item,'expert_id':'expert_1','reuse_rich_source':None}
    rows,reused=write_batch(config,plan,item,scenes,0,model,'cpu',cache)
    assert model.calls==2 and not reused
    assert read_batch(config,plan,item,scenes,0)==rows
    from iqe.specialist_study.deploy_precompute import receipt_path
    receipt_path(config,item,0).unlink()
    rows2,_=write_batch(config,plan,item,scenes,0,model,'cpu',cache)
    assert rows2==rows
    (tmp_path/'candidates'/item['expert_id']/item['model_hash']/(scenes[0].scene_id+'.npz')).write_bytes(b'corrupt')
    with pytest.raises(Exception,match='corrupt'):read_batch(config,plan,item,scenes,0)
