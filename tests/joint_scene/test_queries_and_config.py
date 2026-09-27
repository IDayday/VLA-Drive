import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from starVLA.model.modules.joint_scene.contracts import AnnotatedLocalScene
from tools.joint_local_scene_v3.runtime import build_queries,evaluate
from tools.joint_local_scene_v3.train_mechanism import validate_config,validate_arguments,forward_weights
from tools.joint_local_scene_v3.synthetic import write_synthetic
from tools.joint_local_scene_v3.data import AnnotatedCorpus,annotate
from starVLA.model.modules.structured_world.contracts import WorldTargets
from test_joint_scene import graph,model,labels


def corpus():
    g=graph();y,v=labels(g);v[:,2,-1,:2]=False
    return [AnnotatedLocalScene('scene','log',g,y,v,('ego','a','b'),{'source_current_objects':4})]


def config():
    return json.loads((Path(__file__).parents[2]/'configs/joint_local_scene_v3/mechanism.json').read_text())


def test_fixed_query_cohorts_enumerate_both_neighbors_and_same_valid_points(tmp_path):
    c=corpus();q=build_queries(c)
    assert len(q['queries'])==3 and [r['valid_xy_points'] for r in q['queries']]==[3,3,2]
    report=evaluate(model(),c,q,seed=99,sampling_steps=2,device='cpu',output=tmp_path)
    assert report['summary']['aggregate_valid'] and report['summary']['evaluated_queries']==3
    assert report['summary']['population']['source_current_objects']==4
    assert report['summary']['population']['selected_neighbors']==2
    assert (tmp_path/'queries.json').exists() and (tmp_path/'conditional_queries.csv').exists()
    for row in report['query_rows']:
        assert row['all_hidden_valid_xy_points']==row['conditional_valid_xy_points']==row['valid_xy_points']
        assert row['all_hidden_CV_available']==(row['slot']==0)
        assert row['all_hidden_valid_yaw_points']==(3 if row['slot']==0 else 0)
    assert report['query_rows'][2]['conditional_xy_FDE_m'] is None
    changed=copy.deepcopy(q);changed['queries'][1]['slot']=2
    with pytest.raises(ValueError,match='Query'):evaluate(model(),c,changed,device='cpu')


def test_failed_predictions_keep_every_query_and_invalidate_aggregate():
    m=model();m.sample=lambda noise,*a,**kw:torch.full_like(noise,float('nan'))
    result=evaluate(m,corpus(),sampling_steps=1,device='cpu')
    assert len(result['query_rows'])==3 and result['summary']['failed_queries']==3
    assert result['summary']['failed_scenes']==1 and not result['summary']['aggregate_valid']
    assert result['summary']['groups']['neighbor_all']['all_hidden_ADE_m'] is None


@pytest.mark.parametrize('key,value',[('forwards_per_batch',3),('sampling_steps',0),('role_loss_weight',-1),('training_seed',-1),('warmup_steps',0),('initial_lr',float('nan')),('unrecognized',1)])
def test_invalid_or_ignored_config_is_rejected(key,value):
    cfg=config();cfg[key]=value
    with pytest.raises(ValueError):validate_config(cfg)


def test_explicit_role_weight_formula():
    assert forward_weights('all',3)==[.5,.5]
    assert forward_weights('mask',3)==[.25,.75]
    assert sum(x*w for x,w in zip([2.,6.],forward_weights('mask',3)))==5.


@pytest.mark.parametrize('key,value',[('updates',0),('schedule_updates',0),('batch',0),('eval_every',0),('limit',0)])
def test_bad_training_arguments_before_updates(tmp_path,key,value):
    args=SimpleNamespace(updates=4,schedule_updates=4,batch=1,eval_every=2,limit=None,stop_after=None,output=str(tmp_path/'new'),resume=False,acknowledge_stop=False)
    setattr(args,key,value)
    with pytest.raises(ValueError):validate_arguments(args)


def test_nonempty_output_and_stop_are_not_silently_overwritten(tmp_path):
    args=SimpleNamespace(updates=4,schedule_updates=4,batch=1,eval_every=2,limit=None,stop_after=None,output=str(tmp_path),resume=False,acknowledge_stop=False)
    (tmp_path/'existing.log').write_text('history')
    with pytest.raises(FileExistsError):validate_arguments(args)
    args.resume=True;(tmp_path/'checkpoint.pt').write_text('fixture');(tmp_path/'STOP_REQUESTED').write_text('stop')
    with pytest.raises(ValueError,match='acknowledge'):validate_arguments(args)


def test_schema_graph_identity_and_horizon_recorded(tmp_path):
    cfg=config();cfg['model'].update(dim=32,heads=4,layers=1,steps=3,condition_dim=12)
    root=tmp_path/'data';write_synthetic(root,cfg);data=AnnotatedCorpus(root)
    assert data[0].graph.schema_version==4 and data[1].future.shape[2]==3
    manifest=json.loads((root/'manifest.json').read_text());manifest['schema_version']=3
    (root/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='mismatched'):AnnotatedCorpus(root)
    manifest['schema_version']=4;manifest['graph_config']['radius_m']=99
    (root/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='identity'):AnnotatedCorpus(root)


def test_track_supervision_mapping_and_hidden_future_cannot_rebuild_graph():
    g=graph();y,v=labels(g);boxes=g.boxes[0,1:].clone();future=boxes[:,:2,None].transpose(-1,-2).expand(2,3,2).clone()
    future[0,:,0]+=3;future[1,:,1]+=7
    target=WorldTargets(boxes,torch.zeros(2,dtype=torch.long),('track-a','track-b'),future,torch.ones(2,3,dtype=torch.bool),torch.ones(2,dtype=torch.bool),torch.tensor(True),torch.ones(2,8,dtype=torch.bool),torch.tensor([0.,-20.,50.,20.]))
    first=annotate('a','log',target,y[0,0],g)
    target.future_xy_in_ego_t0.fill_(999.)
    second=annotate('a','log',target,y[0,0],g)
    assert torch.equal(first.graph.boxes,second.graph.boxes) and torch.equal(first.graph.edge_features,second.graph.edge_features)
    assert first.track_ids[1:] == ('track-a','track-b')
    assert torch.equal(first.future[0,1,:,0],torch.full((3,),11.))
