import sys,os,copy,json
from pathlib import Path
W=Path('/mnt/project/VLA-Drive-qwen-world-performance-rl-20261103');sys.path[:0]=[str(W),str(W/'tests')]
import torch,torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from test_world_performance_nominal import actual,primary,Service,Capture
from starVLA.rl.qwen_world_performance import nominal_improvement as N,ode_target as O,persistent_history as H
from starVLA.rl.qwen_world_performance.batch_runtime import PerformanceRuntime,HistoricalBudgetMixin
rank=int(os.environ['RANK']);torch.set_num_threads(1);dist.init_process_group('gloo')
try:
 a,conditions,c=actual();service=Service(nominal_good=True);v=dict(run='own_run_CPU',update=1196,adapter_sha256='old')
 bank=H.new_history_state(v['run'],rank)
 if rank==0:H.admit_nominal(bank,O.probe_nominal(a,conditions,'t0','TRAIN_log',service,v))
 service.nominal_good=bool(rank);r=primary(a,conditions,service);N.attach_material(a,r,conditions,service,exposure_epoch=.5,rank=rank)
 material=H.attach_history_window(bank,a,conditions,service,r['behavior_version']);h=material['hshare'];assert h==(.25 if rank==0 else 0.)
 pad=copy.deepcopy(r);pad['valid_scene']=False;pad.pop('nominal_improvement',None);r['valid_scene']=True
 cls=type('ReviewLoss',(HistoricalBudgetMixin,N.NominalLossMixin,Capture),{});a._loss=cls(c);a._loss.auxiliary_state=N.new_calibration_state()
 ref,cc,_=actual();ref._loss=cls(c);ref._loss.auxiliary_state=N.new_calibration_state();ref._loss.history_share=h
 ddp=DDP(a,broadcast_buffers=False,find_unused_parameters=True);opt=torch.optim.AdamW([p for p in a.parameters()if p.requires_grad],lr=1e-5);ro=torch.optim.AdamW([p for p in ref.parameters()if p.requires_grad],lr=1e-5)
 rt=PerformanceRuntime();rt.group_size=16;rt.spec=dict(population=103288);window=dict(samples=[r,pad],performance_history=material,valid_global_scenes=16,B=16,group_size=16)
 maxerr=0.
 for inner in range(2):
  opt.zero_grad(set_to_none=True);ro.zero_grad(set_to_none=True)
  ref._loss.history_share=h;fresh,_=ref(r);padded,_=ref(pad);hist,_=H.history_loss(ref,material,ref._loss.auxiliary_state)
  expected=.5*fresh+0*padded
  if hist is not None:expected=expected+.5*50*h*hist
  expected.backward()
  rt.backward_update(ddp,a,window,dict(next_inner=inner,valid_scene_exposures=16))
  for p,q in zip((p for p in a.parameters()if p.requires_grad),(p for p in ref.parameters()if p.requires_grad)):
   assert p.grad is not None and torch.isfinite(p.grad).all();dist.all_reduce(q.grad);q.grad.div_(2);maxerr=max(maxerr,float((p.grad-q.grad).abs().max()));torch.testing.assert_close(p.grad,q.grad,atol=1e-6,rtol=3e-5)
  torch.nn.utils.clip_grad_norm_([p for p in a.parameters()if p.requires_grad],1)
  for p,q in zip((p for p in a.parameters()if p.requires_grad),(p for p in ref.parameters()if p.requires_grad)):q.grad=p.grad.clone()
  opt.step();ro.step()
 out=Path('/tmp/performance_s5_runtime_review');out.mkdir(exist_ok=True);(out/f'rank{rank}.json').write_text(json.dumps(dict(status='PASS',world=2,rank=rank,history_h=h,gradient_max_abs_error=maxerr,steps=2,tests='actualRootbackward_update, realTinyScope, valid+padding, rank0historyactive/rank1dummy, fullcalibration insideforward, exactexplicit IL/rescue/history objective gradient',limitations='2-rank control-flow/gradient math; not global8rankpopulation orphysicalscore proof')))
finally:dist.destroy_process_group()
