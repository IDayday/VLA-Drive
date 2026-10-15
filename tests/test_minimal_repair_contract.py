import numpy as np
import torch
from shapely.geometry import box
from starVLA.model.modules.decision_local.minimal_repair_contract import CurrentFeatures,PlanInputs,MinimalRepairRefiner,CaseStatus
from starVLA.model.modules.decision_local.residual_refiner import ResidualRefiner
from starVLA.model.modules.decision_local.consequence_contract import ModelInputs,PredictionQueries
from starVLA.model.modules.decision_local.recovery_plans import PlanCurve,build_plan_pool,select_plan
from starVLA.model.modules.decision_local.local_constraint_checker import LocalConstraintChecker,body_polygons
from starVLA.model.modules.decision_local.minimal_repair_targets import solve_minimal_repair,SolverConfig
from starVLA.dataloader.minimal_repair_scene import AnnotatedScene
from starVLA.dataloader.foresight_dataset import encode_ego


def scene(objects=None):
    frames=[objects or {} for _ in range(9)]
    return AnnotatedScene(road=box(-20,-5,80,5),body={'length':5.176,'width':2.297,'rear_axle_to_center':1.461},
        frames=frames,observed=np.ones((9,160,200),bool),road_height_known=True)


def straight():return np.column_stack((np.arange(1,9)*2.5,np.zeros((8,2)))).astype(np.float32)


def test_actual_plan_and_legacy_path_equal():
    torch.manual_seed(8);old=ResidualRefiner(hidden_dim=16,dim=32)
    new=MinimalRepairRefiner(hidden_dim=16,dim=32);new.load_state_dict(old.state_dict(),strict=True)
    # Exercise nonzero output too; zero initialization alone is not a parity test.
    torch.nn.init.normal_(old.output.weight,std=.01);new.load_state_dict(old.state_dict())
    q=torch.from_numpy(straight())[None];enc=torch.from_numpy(encode_ego(q[0].numpy()))[None]
    c=CurrentFeatures(torch.randn(1,8,16),torch.randn(1,12,16),torch.randn(1,4),torch.tensor([[5.176,2.297,1.461]]))
    p=PlanInputs(q,enc,torch.arange(1,9)*.5)
    query=PredictionQueries(q[:,None].expand(1,7,8,3),p.physical_times,torch.ones(1,7,8,dtype=torch.bool),torch.ones(1,7,dtype=torch.bool))
    legacy=ModelInputs(c.H_A,c.W,enc,q,c.state,c.body)
    assert torch.equal(old.camera_only(legacy,query),new.camera_only(c,p,query))
    try:PlanInputs.from_mapping({**p.__dict__,'q0':q})
    except ValueError:pass
    else:raise AssertionError('Clean answer bypass accepted')
    try:new.camera_only(c,p,query,{'GT':'forbidden'})
    except TypeError:pass
    else:raise AssertionError('GT dictionary accepted')


def test_warp_boundaries_and_plan_pool():
    q=straight();pool=build_plan_pool(q,[5.,0.],'test')
    assert np.array_equal(pool['plans'][0],q)
    assert not np.array_equal(pool['plans'][1],q)
    curve=PlanCurve(q,[5.,0.]);t=np.linspace(0,4,401)
    for c in ([.7,.2,.4,.6],[-.8,0,-.5,0]):
        phi,derivative=curve.warp(t,c)
        assert phi[0]==0 and np.all(np.diff(phi)>0) and phi[-1]<=4 and derivative.min()>0
        assert np.allclose(curve.xy(np.array([0.]),c),0)
        assert np.allclose(curve.xy(np.array([1e-5]),c)/1e-5,[[5.,0.]],atol=1e-4)
    selected=[select_plan('abc',e)[1] for e in range(24)]
    assert [selected.count(i) for i in range(5)]==[12,3,3,3,3]


def test_keep_is_input_not_any_logged_gt():
    q=build_plan_pool(straight(),[5.,0.],'non_gt')['plans'][1]
    result=solve_minimal_repair(q,[5.,0.],LocalConstraintChecker(scene()))
    assert result['status']==CaseStatus.KEEP
    assert np.array_equal(result['target'],q)
    assert result['starts']==[]
    assert not np.array_equal(result['target'],straight())


def test_unknown_never_becomes_keep():
    s=scene();s.frames[4]=None
    result=solve_minimal_repair(straight(),[5.,0.],LocalConstraintChecker(s))
    assert result['status']==CaseStatus.UNCERTAIN and result['target'] is None


def test_annotated_geometry_is_not_free_support():
    s=scene();s.observed[:]=False
    values,valid,support=LocalConstraintChecker(s).relations(straight(),np.arange(1,9)*.5)
    assert valid.all() and (values[:,1]==10).all()
    assert not support['observed_free_support_fraction'].any()


def test_rotating_box_interpolation_does_not_shrink_or_release():
    from shapely import affinity
    a=box(-2,-1,2,1);b=affinity.rotate(a,60,origin=(0,0))
    aa=np.asarray(a.exterior.coords)[:4];bb=np.asarray(b.exterior.coords)[:4]
    middle=AnnotatedScene.interpolate_rectangle(aa,bb,.5)
    assert abs(middle.area-a.area)<1e-10
    flipped=AnnotatedScene.interpolate_rectangle(aa,np.roll(aa,2,axis=0),.5)
    assert flipped.symmetric_difference(a).area<1e-10
    obj={'corners':aa,'velocity':np.zeros(2),'reliable':True,'stationary':True}
    s=scene({'track':obj});s.frames[1]={};s=AnnotatedScene(road=s.road,body=s.body,frames=s.frames,observed=s.observed,road_height_known=True)
    known,unknown,valid=s.at_times(np.array([.5]),'FUTURE')
    assert unknown[0].intersects(a)  # absent frame is not a release target


def test_polygon_full_body_and_repair_validation():
    q=straight();b=scene().body
    p=body_polygons(np.zeros((1,3)),b)[0]
    assert np.allclose(p.bounds,[-1.127,-1.1485,4.049,1.1485])
    obstacle={'one':{'corners':np.array([[9.5,1.0],[10.5,1.0],[10.5,2.0],[9.5,2.0]]),
                     'velocity':np.zeros(2),'reliable':True}}
    checker=LocalConstraintChecker(scene(obstacle));before=checker.check(q,[5.,0.])
    assert before['annotated_overlap']
    out=solve_minimal_repair(q,[5.,0.],checker,SolverConfig(max_seconds=8.))
    assert out['status'] in (CaseStatus.REPAIR,CaseStatus.UNSOLVED,CaseStatus.AMBIGUOUS)
    if out['target_valid']:
        assert checker.check(out['target'],[5.,0.],reference=q)['feasible']
        assert not np.array_equal(out['target'],q)
