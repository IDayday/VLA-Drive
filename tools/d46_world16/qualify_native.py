"""Bounded actual Tiny-native CPU/Gloo8->16 engine qualification.

No real IL46 model, official PDM scoring, GPUs, SSH or resource locks. Actual
native-Qwen sampler/current/ref loss, DDP, Adam and checkpoint code are used.
Full IL46/CUDA/NCCL/two-node throughput qualification remains mandatory.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import socket
import subprocess
import time
from datetime import timedelta
from types import SimpleNamespace
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from starVLA.rl.d44_native_rl.actor import D44Actor
from starVLA.rl.d44_native_rl.config import defaults46,source_config
from starVLA.rl.d44_native_rl.expanded_scope import groups
from starVLA.rl.d44_native_rl.batch_runtime import collect_scene
from starVLA.rl.d44_native_rl.safety_credit import raw_credit
from starVLA.rl.d44_native_rl.adaptive_kl import AdaptiveKL
from starVLA.rl.d44_native_rl.recovery import Boundary,checkpoint,material_digest
from starVLA.rl.qwen_world_compare.contracts import fingerprint
from starVLA.rl.qwen_world_compare.policy_adapter import CONDITION_SCHEMA
from starVLA.model.modules.qwen_world_compare.model import FourFrameworkVLA
from starVLA.rl.qwen_world_compare.deployment import prune_deployment
from starVLA.rl.d46_world16.native import BASE,prepare,source_inventory,World16Boundary
from starVLA.rl.d46_world16.runtime import World16Extension
from tools.d44_native_rl.train import SceneScheduler


class TinyActor(D44Actor):
    def scope_audit(self):
        current={n:p for n,p in self.named_parameters()if p.requires_grad}
        expected={'action_expert.'+p+'.'+suffix for p in self._scope_paths for suffix in('lora_A','lora_B')}
        if set(current)!=expected or any(p.requires_grad for p in self.ref_action_expert.parameters()):raise ValueError('Tiny closure/reference')
        return dict(trainable_names=sorted(current),trainable_numel=sum(p.numel()for p in current.values()),qualification_tiny=True)


def make_actor(cfg):
    file=Path(__file__).resolve().parents[2]/'tests/world_compare/test_d_qwen_flow_adapter.py'
    spec=importlib.util.spec_from_file_location('world16_native_oracle',file);oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)
    qwen=oracle.model().qwen_model
    model=FourFrameworkVLA('D',config={'d':{'qualification':True,'activation_checkpointing':False}},train_heads=False,qwen_encoder=SimpleNamespace(model=qwen))
    return TinyActor(prune_deployment(model,copy_model=False),source_config(cfg),cfg)


def condition(actor,slot):
    inputs=dict(input_ids=torch.tensor([[3,5,7,9+(slot%3)]]),attention_mask=torch.ones(1,4,dtype=torch.long))
    current=actor.action_expert.encode_current(inputs,torch.zeros(1,8),(f'TRAIN_CPU_{slot}',))
    return dict(schema=CONDITION_SCHEMA,context=torch.zeros(16,1,1),current=current)


class SyntheticService:
    def score(self,request):
        token,log,requested=request
        return [dict(status='OK',requested_trajectory=p,metrics=dict(no_at_fault_collisions=1.,drivable_area_compliance=1.,time_to_collision_within_bound=1.,
            ego_progress=.2+i/20,comfort=1.,driving_direction_compliance=1.,score=(5*(.2+i/20)+7)/12),
            non_drivable_body_by_time=[False]*41,official_first_at_fault_collision_seconds=None,official_first_ttc_infraction_seconds=None)
            for i,p in enumerate(requested)]


def _initial_state():
    return dict(updates=0,collections=0,scene_cursor=0,next_inner=0,phase='READY',valid_scene_exposures=0,raw_scene_exposures=0,new_candidates=0,
      valid_PG_candidates=0,extra_B1_queries=0,adaptive_G32_queries=0,padding_scene_exposures=0,adaptive_kl=AdaptiveKL().state_dict(),evaluation_gates=[])


def _initial_identity(actor,cfg):
    sources=source_inventory();sources={p:s for p,s in sources.items()if '/d46_world16/'not in p}
    raw=subprocess.run(['git','show',BASE+':tools/d44_native_rl/train.py'],check=True,stdout=subprocess.PIPE).stdout
    sources['tools/d44_native_rl/train.py']=hashlib.sha256(raw).hexdigest()
    value=dict(schema='d44_native_rl_v1',architecture_id='D',world=8,recipe=cfg,scope=actor.closure,source_files=sources,
      qualification=True,qualification_fixture='d46_tiny_native_world_transition_CPU_v1',
      parent_model_sha256='tiny_native_qualification:random_Qwen32_3layers_NOT_IL46_weights',fresh_Adam=True)
    value['sha256']=fingerprint(value);return value


def _worker(rank,world,port,root,stop_update,resume,arm):
    root=Path(root);torch.set_num_threads(1);random.seed(42);np.random.seed(42);torch.manual_seed(42)
    torch.cuda.synchronize=lambda *_,**__:None
    dist.init_process_group('gloo',init_method=f'tcp://127.0.0.1:{port}',rank=rank,world_size=world,timeout=timedelta(seconds=120))
    if world==8:checkpoint.init_control_group()
    else:World16Extension.init_control_group()
    cfg=defaults46(arm);cfg['checkpoint_policy']='all_off'
    actor=make_actor(cfg);opt=torch.optim.AdamW(groups(actor,cfg),betas=(.9,.95),eps=1e-8,weight_decay=.01);sched=SceneScheduler(opt);generator=torch.Generator().manual_seed(420000+rank)
    boundary=Boundary();extension=None
    if world==8:
        state=_initial_state();batch=None;identity=_initial_identity(actor,cfg)
    else:
        identity=json.loads((root/'identity.json').read_text());boundary=World16Boundary()
        state,batch=checkpoint.restore(resume,actor,opt,sched,generator,identity,runtime=boundary)
        extension=World16Extension();extension.saved_identity=identity
        # Actual READY roundtrip before any update; source Adam and RNG retained.
        if state['phase']=='READY':checkpoint.save(root,actor,opt,sched,generator,state,None,identity,runtime=boundary,tag='READY_world16_roundtrip')
    if extension:
        extension.args=SimpleNamespace(output=str(root),resume=str(resume))
        class CPUData:
            def observation(self,token):return {'proprio':[0.]*8}
        class CPUStream:
            data=CPUData()
            def at(self,cursor,rank,j,world):return f'TRAIN_CPU_{cursor+rank}',False
        processor=lambda observations:dict(input_ids=torch.tensor([[3,5,7,9]]),attention_mask=torch.ones(1,4,dtype=torch.long))
        extension.after_restore(actor,SimpleNamespace(processor=processor),CPUStream(),state,identity)
    ddp=DDP(actor,find_unused_parameters=True,broadcast_buffers=False,bucket_cap_mb=25)
    dual=AdaptiveKL();dual.load_state_dict(state['adaptive_kl'])
    while state['updates']<stop_update:
        if batch is None:
            scenes=[];version=dict(run=identity['sha256'],update=state['updates'],collection=state['collections']+1)
            if extension:extension.before_collection(state,generator,rank,world)
            for j in range(64//world):
                slot=state['scene_cursor']+(j*8+rank if world==8 else(j+4*(rank//8))*8+rank%8)
                if extension:extension.before_scene(generator,j)
                r=collect_scene(actor,condition(actor,slot),f'TRAIN_CPU_{slot}','physical_train_cpu',SyntheticService(),generator,cfg,0.,version)
                if extension:extension.after_scene(generator,j)
                r.update(padding=False,beta=dual.beta,expect_fresh=True);scenes.append(r)
            if extension:extension.after_collection(state,generator,rank,world)
            raw,_=raw_credit(torch.stack([r['factors']for r in scenes]));moments=torch.stack((raw.sum(),raw.square().sum(),torch.tensor(raw.numel(),dtype=torch.float64)))
            dist.all_reduce(moments);std=(moments[1]/moments[2]-(moments[0]/moments[2]).square()).sqrt()
            for r,adv in zip(scenes,(raw/(std+1e-8)).clamp(-5,5).float()):r['advantages']=adv.detach()
            batch=dict(scenes=scenes,behavior_update=state['updates'],global_std=float(std),valid_scenes=64,epoch=0.)
            batch['digest']=material_digest(batch);state['collections']+=1;state['scene_cursor']+=64;state['raw_scene_exposures']+=64;state['valid_scene_exposures']+=64
            state['new_candidates']+=1024;state['valid_PG_candidates']+=1024;state.update(phase='ROLLOUT_COMMITTED',next_inner=0)
        opt.zero_grad(set_to_none=True);kval=0.
        for j,r in enumerate(batch['scenes']):
            import contextlib
            r['expect_fresh']=state['next_inner']==0
            with ddp.no_sync()if j<len(batch['scenes'])-1 else contextlib.nullcontext():
                loss,metrics=ddp(r)
                if arm=='D3_FPO'and r['expect_fresh']:
                    error=max(abs(float(metrics['fpo_joint_logratio_min'])),abs(float(metrics['fpo_joint_logratio_max'])))
                    if error>1e-4:raise FloatingPointError('Actual fresh FPO energy logratio exceeds1e-4: '+str(error))
                (loss*(world/64)).backward()
            kval+=float(metrics['reference_KL_joint'])
        norm=torch.nn.utils.clip_grad_norm_([p for p in actor.parameters()if p.requires_grad],1.)
        if not torch.isfinite(norm):raise FloatingPointError('Tiny nonfinite')
        opt.step();sched.step();state['updates']+=1;state['next_inner']+=1;state['phase']='INNER_COMMITTED'
        if state['next_inner']==2:
            value=torch.tensor(kval/64,dtype=torch.float64);dist.all_reduce(value);dual.update(float(value));state['adaptive_kl']=dual.state_dict();batch=None;state.update(next_inner=0,phase='READY')
        checkpoint.save(root,actor,opt,sched,generator,state,batch,identity,runtime=boundary)
    # Exercise the actual CLI terminal JSON path as well as native save/restore.
    from starVLA.rl.flow_grpo.transactions import atomic_json
    result=dict(state,status='QUALIFICATION_STOPPED',rank=rank,checkpoint=str(root/'checkpoints'/f"update{state['updates']:06}_inner{state['next_inner']}"))
    public=result if extension is None else extension.json_result(result)
    json.dumps(public,allow_nan=False);atomic_json(root/f'result_rank{rank}.json',public)
    if extension and not all(torch.is_tensor(x)for x in state['world16_virtual_sampler']['lanes']):
        raise ValueError('JSON result mutated native RNG state')
    dist.destroy_process_group()


def run(world,root,stop,resume=None,arm='D0'):
    with socket.socket()as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    context=mp.start_processes(_worker,args=(world,port,str(root),stop,str(resume)if resume else None,arm),nprocs=world,join=False,start_method='fork')
    deadline=time.monotonic()+240
    try:
        while not context.join(timeout=1):
            if time.monotonic()>deadline:raise TimeoutError('Bounded CPU leg exceeded240s')
    except BaseException:
        for process in context.processes:
            if process.is_alive():process.terminate()
        for process in context.processes:process.join(timeout=10)
        raise


def exact(a,b,path='root'):
    if type(a)is not type(b):raise ValueError('Type '+path)
    if torch.is_tensor(a):
        if a.dtype!=b.dtype or a.shape!=b.shape or not torch.equal(a,b):raise ValueError('Tensor '+path)
    elif isinstance(a,np.ndarray):
        if not np.array_equal(a,b):raise ValueError('Array '+path)
    elif isinstance(a,dict):
        if a.keys()!=b.keys():raise ValueError('Keys '+path)
        for k in a:exact(a[k],b[k],path+'.'+str(k))
    elif isinstance(a,(list,tuple)):
        if len(a)!=len(b):raise ValueError('Len '+path)
        for i,(x,y)in enumerate(zip(a,b)):exact(x,y,path+'.'+str(i))
    elif a!=b:raise ValueError('Value '+path)


def qualify(output,arm='D0'):
    if arm not in('D0','D3_FPO'):raise ValueError('Bounded CPU fixture supports D0 or FPO only')
    if torch.cuda.is_available():raise ValueError('CPU engineering qualification requires CUDA_VISIBLE_DEVICES empty')
    output=Path(output)
    if output.exists():raise FileExistsError('Keep qualification immutable')
    output.mkdir(parents=True);tick=time.monotonic();run(8,output/'old8',2,arm=arm)
    old=output/'old8/checkpoints/update000002_inner0'
    a=prepare(old,output/'continuous16',qualification=True);b=prepare(old,output/'split16',qualification=True)
    run(16,output/'continuous16',4,a['checkpoint'],arm=arm);run(16,output/'split16',3,b['checkpoint'],arm=arm)
    run(16,output/'split16',4,output/'split16/checkpoints/update000003_inner1',arm=arm)
    cp='checkpoints/update000004_inner0'
    # Distinct target identities are actually identical because root path is not part of identity.
    for name in ['optimizer.pt']+[f'rank{r}.pt'for r in range(16)]:
        exact(torch.load(output/'continuous16'/cp/name,map_location='cpu',weights_only=False),torch.load(output/'split16'/cp/name,map_location='cpu',weights_only=False),name)
    original=torch.load(old/'optimizer.pt',map_location='cpu',weights_only=False)
    ready=torch.load(output/'continuous16/checkpoints/READY_world16_roundtrip/optimizer.pt',map_location='cpu',weights_only=False);exact(original,ready,'READY_common')
    result=dict(schema='d46_world16_Tiny_native_CPU_engine_qualification_v1',status='PASS',source_world=8,target_world=16,B=64,G=16,M=2,arm=arm,mc_samples=4 if arm=='D3_FPO'else None,actual_fresh_FPO_energy_ratio_checked=arm=='D3_FPO',
        actual_distributed_backend='gloo',world8_control_timeout_seconds=90,world16_control_timeout_seconds=600,GPU_used=False,canonical_PDM_queries=0,actual_model='native TinyQwen32x3, NOT full IL46 weights',
        optimizer_origin_updates=2,READY16_parameters_Adam_scheduler_exact=True,continuous_final_update=4,restored_inner_update=3,restored_next_inner=1,
        all16_adapters_Adam_scheduler_RNG_scene_stream_virtual8_and_material_bitexact=True,terminal_JSON_all16=True,post_native_B1_parity_all16=True,seconds=time.monotonic()-tick,
        limitation='CPU engineering only. Full IL46/CUDA/NCCL/two-host B1/ratio/throughput qualification required; test labels synthetic TRAIN fixtures.')
    (output/'COMPLETE.json').write_text(json.dumps(result,indent=2)+'\n');return result


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',required=True);p.add_argument('--arm',choices=('D0','D3_FPO'),default='D0');a=p.parse_args();print(json.dumps(qualify(a.output,a.arm),indent=2))


if __name__=='__main__':main()
