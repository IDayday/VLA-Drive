"""Two actual8-CPU/Gloo native roundtrips, tiny FULL/M1 and LoRA/M2 only.

Synthetic TRAIN rewards, no real IL weights/PDM/GPU; not a formal admission.
"""
import copy
import importlib.util
import json
from pathlib import Path
import random
import socket
import time
from datetime import timedelta
import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from starVLA.rl.d44_native_rl.recovery import checkpoint,material_digest
from starVLA.rl.d46_augustus.runtime import AugustusExtension,AugustusBoundary,FixedKL,credit,advantages
from starVLA.rl.d46_augustus.config import SCHEMA
from starVLA.rl.qwen_world_compare.contracts import fingerprint
from tools.d44_native_rl.train import SceneScheduler
from tools.d46_world16.qualify_native import SyntheticService,exact


def factory(experiment,m):
    path=Path(__file__).with_name('test_d46_augustus_actor.py')
    spec=importlib.util.spec_from_file_location('augustus_native_test_model',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    torch.manual_seed(42);actor,_,cfg,_,condition=module.tiny(experiment)
    cfg.update(global_scenes=64,inner_updates=m);actor.study_config=copy.deepcopy(cfg)
    opt=torch.optim.AdamW(AugustusExtension.optimizer_groups(actor,cfg),betas=(.9,.95),eps=1e-8,weight_decay=.01)
    return actor,opt,SceneScheduler(opt),cfg,condition


def worker(rank,port,root,experiment,m):
    torch.set_num_threads(1);random.seed(42);np.random.seed(42)
    dist.init_process_group('gloo',init_method=f'tcp://127.0.0.1:{port}',rank=rank,world_size=8,timeout=timedelta(seconds=120))
    checkpoint.init_control_group();root=Path(root)
    actor,opt,scheduler,cfg,condition=factory(experiment,m)
    identity=dict(schema='d44_native_rl_v1',world=8,architecture_id='D',recipe=cfg,algorithm_schema=SCHEMA,
        augustus_protocol_sha256=fingerprint(cfg['augustus']),scope=actor.closure,
        qualification=True,qualification_fixture='AUGUSTUS_TINY_CPU_NOT_REAL_IL46',
        parent_model_sha256='tiny_native_qualification_random_weights_NOT_real_IL46',source_files={},fresh_Adam=True)
    identity['sha256']=fingerprint(identity)
    ddp=DDP(actor,find_unused_parameters=True,broadcast_buffers=False)
    generator=torch.Generator().manual_seed(420000+rank);scenes=[]
    from starVLA.rl.d46_augustus.collector import collect_scene
    for j in range(8):
        r=collect_scene(actor,condition,f'TRAIN_CPU_{rank}_{j}','SYNTHETIC_CPU_TRAIN',SyntheticService(),generator,cfg,0.,
            dict(run=identity['sha256'],update=0,collection=1))
        r.update(padding=False,beta=cfg['beta'],expect_fresh=True);scenes.append(r)
    from starVLA.rl.d46_augustus.runtime import recipe_credit
    raw,_=recipe_credit(scenes,None,0.,cfg);a,std=advantages(raw,torch.ones(8,dtype=torch.bool),8,8)
    for r,value in zip(scenes,a):r['advantages']=value
    batch=dict(scenes=scenes,behavior_update=0,global_std=float(std),valid_scenes=64,epoch=0.)
    batch['digest']=material_digest(batch);dual=FixedKL(cfg)
    state=dict(updates=0,next_inner=0,phase='ROLLOUT_COMMITTED',collections=1,scene_cursor=64,raw_scene_exposures=64,
        valid_scene_exposures=64,padding_scene_exposures=0,new_candidates=512,valid_PG_candidates=512,
        extra_B1_queries=0,adaptive_G32_queries=0,adaptive_kl=dual.state_dict(),evaluation_gates=[])
    opt.zero_grad(set_to_none=True);totalKL=0.
    for j,r in enumerate(scenes):
        import contextlib
        with ddp.no_sync()if j<7 else contextlib.nullcontext():
            loss,metrics=ddp(r);(loss/8).backward()
        assert torch.isfinite(loss)and metrics['fresh_ratio_error']==0
        totalKL+=float(metrics['reference_KL_joint'])
    norm=torch.nn.utils.clip_grad_norm_([p for p in actor.parameters()if p.requires_grad],1.);assert torch.isfinite(norm)
    opt.step();scheduler.step();state.update(updates=1,next_inner=1,phase='INNER_COMMITTED')
    if m==1:
        kval=torch.tensor(totalKL/64,dtype=torch.float64);dist.all_reduce(kval);dual.update(float(kval));state['adaptive_kl']=dual.state_dict()
        state.update(next_inner=0,phase='READY');batch=None
    boundary=AugustusBoundary();cp=checkpoint.save(root,actor,opt,scheduler,generator,state,batch,identity,runtime=boundary)
    # Fresh model/Adam/generator, then the unchanged audited native restore.
    second,opt2,scheduler2,cfg2,_=factory(experiment,m);g2=torch.Generator().manual_seed(999)
    restored,pending=checkpoint.restore(cp,second,opt2,scheduler2,g2,identity,runtime=boundary)
    exact(state,restored,'state');exact(opt.state_dict(),opt2.state_dict(),'Adam');exact(scheduler.state_dict(),scheduler2.state_dict(),'scheduler')
    expected=torch.load(Path(cp)/f'rank{rank}.pt',map_location='cpu',weights_only=False)
    assert torch.equal(torch.get_rng_state(),expected['rng']['torch'])and torch.equal(g2.get_state(),expected['rng']['rollout_generator'])
    exact(random.getstate(),expected['rng']['python'],'python_RNG');exact(np.random.get_state(),expected['rng']['numpy'],'numpy_RNG')
    params={n:p for n,p in actor.named_parameters()if p.requires_grad};other={n:p for n,p in second.named_parameters()if p.requires_grad}
    assert params.keys()==other.keys()and all(torch.equal(params[n],other[n])for n in params)
    if m==2:
        assert pending is not None and material_digest(pending)==batch['digest']and restored['next_inner']==1
        # The same pending update target and saved A/old probabilities survive.
        pending['scenes'][0]['expect_fresh']=False;batch['scenes'][0]['expect_fresh']=False
        left=actor(batch['scenes'][0])[0];right=second(pending['scenes'][0])[0]
        torch.testing.assert_close(left,right,atol=0,rtol=0)
    else:assert pending is None and restored['phase']=='READY'
    (root/f'CPU_RESTORE_rank{rank}.json').write_text(json.dumps(dict(status='PASS',rank=rank,experiment=experiment,inner=m,
        actual_world8_Gloo=True,updates=1,pending=m==2,Adam_RNG_material_exact=True,real_IL46_or_GPU=False)))
    dist.destroy_process_group()


@pytest.mark.parametrize('experiment,m',[('AUG1_FULL',1),('AUG2_LORA',2),('AUG3_FULL_ALIGNED',2),('AUG4_LORA_ALIGNED',2)])
def test_actual8CPU_native_save_restore_without_reset(tmp_path,experiment,m):
    if torch.cuda.is_available():pytest.skip('This bounded fixture is CPU-only')
    with socket.socket()as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    root=tmp_path/experiment
    context=mp.start_processes(worker,args=(port,str(root),experiment,m),nprocs=8,join=False,start_method='fork')
    deadline=time.monotonic()+120
    try:
        while not context.join(timeout=1):
            if time.monotonic()>deadline:raise TimeoutError('Bounded CPU native roundtrip120s')
    except BaseException:
        for process in context.processes:
            if process.is_alive():process.terminate()
        for process in context.processes:process.join(timeout=5)
        raise
    rows=[json.loads((root/f'CPU_RESTORE_rank{rank}.json').read_text())for rank in range(8)]
    assert all(x['status']=='PASS'and x['Adam_RNG_material_exact']for x in rows)
