"""Proactive old-version history budget, dominated upgrade and native material."""
import copy,io
import pytest,torch
from test_world_performance_nominal import actual,Service,primary
from starVLA.rl.qwen_world_performance import ode_target as O,nominal_improvement as N,persistent_history as H


def test_actual_B1_history_floor_regression_current_memory_and_native_upgrade():
 torch.set_num_threads(1);actor,conditions,c=actual();service=Service(nominal_good=True);r=primary(actor,conditions,service);version=r['behavior_version']
 teacher=O.probe_nominal(actor,conditions,'t0','TRAIN_log',service,version);bank=H.new_history_state(version['run'],0);H.admit_nominal(bank,teacher)
 unchanged=torch.get_rng_state().clone();empty=H.attach_history_window(bank,actor,conditions,service,version)
 assert not empty['enabled']and empty['actual_B1_queries']==0;H.validate_history_material(empty,bank_after=bank,allowed_tokens={'t0':'TRAIN_log'})
 nextv=dict(version,update=1204);floor=H.attach_history_window(bank,actor,conditions,service,nextv)
 assert floor['hshare']==.05 and not floor['regression']and floor['teacher']['source_version']['update']==1200
 H.validate_history_material(floor,bank_after=bank);loss,metrics=H.history_loss(actor,floor,N.new_calibration_state())
 assert metrics['history_scale_fallback']==1 and loss.requires_grad
 with torch.no_grad():next(actor.scene_provider.parameters()).add_(.02)
 service.nominal_good=False;gate=H.attach_history_window(bank,actor,conditions,service,dict(version,update=1208))
 assert gate['hshare']==.25 and gate['regression']and gate['actual_B1_queries']==1
 H.validate_history_material(gate,bank_after=bank)
 state=N.new_calibration_state();state.update(effective_updates=1,calibration_updates=1,measured_updates=1,scale_cfm=2.,scale_ode=.5,last_norms=dict(pg=1.,IL=1.,cfm=1.,ode=1.))
 before=N.digest(state);builds=actor.current_memory_builds;history,metrics=H.history_loss(actor,gate,state)
 assert N.digest(state)==before and actor.current_memory_builds-builds==2
 torch.testing.assert_close(history,.75*2*metrics['history_raw_cfm']+.25*.5*metrics['history_raw_ode'])
 gradients=torch.autograd.grad(history,[p for p in actor.parameters()if p.requires_grad],retain_graph=True,allow_unused=True)
 names=[n for n,p in actor.named_parameters()if p.requires_grad]
 for prefix in ('action_expert.','scene_provider.','predictive_provider.'):
  assert sum(float(g.norm())for n,g in zip(names,gradients)if g is not None and n.startswith(prefix))>0
 service.nominal_good=True;service.nominal_ep=1.
 dominated=H.attach_history_window(bank,actor,conditions,service,dict(version,update=1212))
 assert dominated['dominated_teacher']and dominated['hshare']==0 and dominated['teacher']['source_version']['update']==1200
 assert bank['entries'][0]['source_version']['update']==1212 and bank['dominated_teacher_count']==1
 assert H.history_loss(actor,dominated,state)[0]is None
 H.validate_history_state(bank,version['run'],0,allowed_tokens={'t0':'TRAIN_log'});H.validate_history_material(dominated,bank_after=bank)
 data=io.BytesIO();torch.save((bank,dominated,state),data);data.seek(0);bb,mm,ss=torch.load(data,weights_only=False)
 H.validate_history_state(bb,version['run'],0,allowed_tokens={'t0':'TRAIN_log'});H.validate_history_material(mm,bb);N.validate_calibration_state(ss)
 assert N.digest(bank)==N.digest(bb)and torch.equal(unchanged,torch.get_rng_state())
 bad=copy.deepcopy(mm);bad['hshare']=.05;bad=N.sealed({k:v for k,v in bad.items()if k!='material_sha256'},'material_sha256')
 with pytest.raises(ValueError):H.validate_history_material(bad,bb)
 with pytest.raises(ValueError):H.validate_history_state(bb,'different_run',0)
 with pytest.raises(ValueError):H.validate_history_state(bb,version['run'],0,allowed_tokens={'NAV_other'})


def test_512_reservoir_private_rng_reproducible_counter_fixture():
 torch.set_num_threads(1);actor,conditions,c=actual();service=Service(nominal_good=True);r=primary(actor,conditions,service)
 teacher=O.probe_nominal(actor,conditions,'t0','TRAIN_log',service,r['behavior_version'])
 bank=H.new_history_state(r['behavior_version']['run'],0)
 #Only the reservoir selection/count boundary is synthetic: these512 distinct
 #IDs are not scored model witnesses and never passed to training or validators.
 bank['entries']=[dict(token='counter_fixture_'+str(i))for i in range(512)];bank['admissions']=512
 twin=copy.deepcopy(bank);main_rng=torch.get_rng_state().clone();H.admit_nominal(bank,teacher);H.admit_nominal(twin,teacher)
 assert N.digest(bank)==N.digest(twin)and len(bank['entries'])==512 and bank['admissions']==513
 assert bank['replacements']+bank['not_retained']==1 and torch.equal(main_rng,torch.get_rng_state())
