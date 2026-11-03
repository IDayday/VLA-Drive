"""Prepare/run/compare read-only D FPO execution microbenchmarks.

Prepare and compare are CPU-only. Run requires Root to allocate an existing GPU
worker; this tool never dispatches, scores, creates Adam or saves policy weights.
External tool re-executes inside the selected frozen baseline/candidate source.
"""
import argparse,copy,hashlib,json,os,statistics,subprocess,sys,time
from pathlib import Path

MODES={'baseline':(False,False),'prefix':(True,False),'fusion':(False,True),'both':(True,True)}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def require(x,m):
    if not x:raise ValueError(m)
def put(p,x):
    with Path(p).open('x')as stream:stream.write(json.dumps(x,indent=2,allow_nan=False)+'\n')
def head(source):return subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()


def prepare(a):
    import torch
    from starVLA.rl.d44_native_rl.recovery import material_digest
    from starVLA.rl.qwen_world_compare.contracts import fingerprint
    cp=Path(a.checkpoint).resolve();batchpath=Path(a.behavior).resolve();identity=read(cp/'identity.json');seal=read(cp/'COMPLETE');files=read(cp/'files.json')
    body=dict(identity);saved=body.pop('sha256',None)
    require(fingerprint(body)==saved,'Native identity self-seal changed')
    require(identity['recipe']['arm']=='D3_FPO'and identity['world']==16 and seal['world']==16,'Use actual own FPO16 native')
    require(seal['phase']=='READY'and seal['next_inner']==0 and seal['files_sha256']==sha(cp/'files.json'),'Sealed READY checkpoint required')
    for name,digest in files.items():require(sha(cp/name)==digest,'Changed native file '+name)
    batch=torch.load(batchpath,map_location='cpu',weights_only=False)
    require(len(batch['scenes'])==4 and batch['behavior_update']==seal['updates']and batch['digest']==material_digest(batch),'Need exact next-collection four-scene material from this theta')
    require(len({s['token']for s in batch['scenes']})==4,'Four real distinct TRAIN scenes required')
    for s in batch['scenes']:require(s['behavior_version']['run']==identity['sha256']and s['behavior_version']['update']==seal['updates'],'Foreign behavior version')
    sources={k:dict(path=str(Path(v).resolve()),commit=head(v))for k,v in [('baseline',a.baseline_source),('candidate',a.candidate_source)]}
    model_sources={name:sha(Path(sources['baseline']['path'])/name)for name in identity.get('source_files',{})if name.startswith('starVLA/model/')}
    require(all(sha(Path(sources['candidate']['path'])/name)==value for name,value in model_sources.items()),'Deployment model source changed')
    for row in sources.values():require(not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=row['path'],text=True).strip(),'Freeze source before benchmarking')
    directory=Path(a.result_dir).resolve();directory.mkdir(parents=True,exist_ok=True)
    value=dict(schema='d46_FPO_execution_microbenchmark_plan_v1',checkpoint=str(cp),native_files_sha256=sha(cp/'files.json'),identity_sha256=identity['sha256'],behavior=str(batchpath),behavior_sha256=sha(batchpath),unchanged_model_sources=model_sources,updates=seal['updates'],tokens=[s['token']for s in batch['scenes']],sources=sources,result_dir=str(directory),modes=MODES,iterations=a.iterations,warmup=a.warmup,no_Adam=True,no_new_rollout_or_reward=True,no_benchmark_weights_exported=True)
    put(a.output,value);print(json.dumps(value,indent=2))


def run(a):
    plan=read(a.plan)
    require(0<=plan['warmup']<=10 and 1<=plan['iterations']<=20,'Finite benchmark iterations only')
    if a.expected_plan_sha256:require(sha(a.plan)==a.expected_plan_sha256,'Pinned benchmark plan changed')
    source=plan['sources']['baseline'if a.mode=='baseline'else'candidate']
    if not a._in_source:
        environment=os.environ.copy();environment.pop('PYTHONPATH',None)
        bootstrap="import runpy,sys;sys.path.insert(0,sys.argv.pop(1));runpy.run_path(sys.argv.pop(1),run_name='__main__')"
        raise SystemExit(subprocess.run([sys.executable,'-c',bootstrap,source['path'],str(Path(__file__).resolve())]+sys.argv[1:]+['--_in-source'],cwd=source['path'],env=environment,timeout=1800).returncode)
    import torch
    from starVLA.rl.d44_native_rl import parent
    from starVLA.rl.d44_native_rl.actor import D44Actor
    from starVLA.rl.d44_native_rl.config import source_config
    from starVLA.rl.d44_native_rl.conditions import DConditions
    from starVLA.rl.d44_native_rl.data import DataStream
    from starVLA.rl.d44_native_rl.recovery import material_digest,checkpoint
    from starVLA.rl.simwam_recipe.lora_adapter import restore_adapters
    from starVLA.rl.flow_grpo.reproducibility import configure_numerics
    require(torch.cuda.is_available(),'Run only after Root allocates a GPU; prepare/compare need no GPU')
    require(head(source['path'])==source['commit'],'Benchmark source changed')
    cp=Path(plan['checkpoint']);require(sha(cp/'files.json')==plan['native_files_sha256']and sha(plan['behavior'])==plan['behavior_sha256'],'Input changed')
    identity=read(cp/'identity.json');files=read(cp/'files.json')
    require(identity['sha256']==plan['identity_sha256']and sha(cp/'identity.json')==files['identity.json'],'Native identity changed')
    require(all(sha(Path(source['path'])/name)==value for name,value in plan['unchanged_model_sources'].items()),'Model source changed after preparation')
    require(sha(cp/'optimizer.pt')==files['optimizer.pt'],'Native parameters changed')
    configure_numerics();torch.set_num_threads(2);torch.cuda.set_device(0);device=torch.device('cuda',0)
    cfg=identity['recipe'];model,info=parent.load(cfg['parent_checkpoint'],device,readouts=False,binding=cfg.get('parent_registration'))
    actor=D44Actor(model,source_config(cfg),cfg)
    native=torch.load(cp/'optimizer.pt',map_location='cpu',weights_only=False)
    require(set(native['adapters'])==set(actor.closure['trainable_names']),'Different actual parameter closure')
    restore_adapters(actor,native['adapters']);del native
    fast,fused=MODES[a.mode];actor.execution_fast_prefix=fast;actor.execution_fuse_fm_mc=fused
    stream=DataStream(cfg,cfg['global_scenes']);conditions=DConditions(model,stream.data,info,group=cfg['group'],capacity=8)
    object.__setattr__(actor,'conditions',conditions)
    raw=torch.load(plan['behavior'],map_location='cpu',weights_only=False);require(raw['digest']==material_digest(raw),'Changed behavior material')
    batch=checkpoint.device_tree(raw,device);rebindings=[]
    for scene in batch['scenes']:
        saved=scene['condition'];current=conditions.get(scene['token'])
        require({k:v for k,v in saved.items()if k!='identity'}=={k:v for k,v in current.items()if k!='identity'},'Only same-token frozen-source identity may be rebound')
        rebindings.append(dict(token=scene['token'],old=saved,new=current));scene['condition']=current;scene['expect_fresh']=True
        conditions.materialize(current,device)
    params={n:p for n,p in actor.named_parameters()if p.requires_grad};last_metrics=[];loss_values=[]
    def step():
        nonlocal last_metrics,loss_values
        actor.zero_grad(set_to_none=True);last_metrics=[];loss_values=[]
        for scene in batch['scenes']:
            loss,metrics=actor(scene);(loss/4).backward()
            loss_values.append(loss.detach());last_metrics.append({k:float(v.detach())for k,v in metrics.items()})
    for _ in range(plan['warmup']):step()
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();seconds=[]
    for _ in range(plan['iterations']):
        torch.cuda.synchronize();tick=time.perf_counter();step();torch.cuda.synchronize();seconds.append(time.perf_counter()-tick)
    gradients={n:p.grad.detach().cpu()if p.grad is not None else None for n,p in params.items()}
    energy_values=[]
    from starVLA.rl.d44_native_rl.fpo import velocity_energy
    with torch.no_grad():
        for scene in batch['scenes']:
            with actor._context_call(scene['condition'])as cond:
                prefix=actor._prepare(cond,False)
                if fused:
                    from starVLA.rl.d44_native_rl.fpo import mc_predictions_energy
                    energies,_=mc_predictions_energy(actor.action_expert,prefix,scene['fm_state'],scene['fm_sigma'],scene['fm_target'],fuse=True)
                else:
                    energies=[]
                    for state,sigma,target in zip(scene['fm_state'],scene['fm_sigma'],scene['fm_target']):
                        with torch.autocast(device.type,dtype=torch.bfloat16):pred=actor.action_expert.denoise_x0(state,sigma,prefix)
                        energies.append(velocity_energy(state,pred,target,sigma))
                    energies=torch.stack(energies)
                energy_values.append(energies.cpu())
    payload=dict(losses=torch.stack(loss_values).cpu(),energies=torch.stack(energy_values),gradients=gradients)
    require(all(x is None or torch.isfinite(x).all()for x in gradients.values()),'Nonfinite gradients')
    peak=torch.cuda.max_memory_allocated();allocated=torch.cuda.memory_allocated()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA])as prof:step();torch.cuda.synchronize()
    syncs={e.key:dict(count=e.count,cpu_time_total_us=e.cpu_time_total)for e in prof.key_averages()if e.key in('aten::_local_scalar_dense','aten::item','aten::is_nonzero')or'synchronize'in e.key.lower()}
    dest=Path(plan['result_dir']);data=dest/(a.mode+'_gradients.pt');require(not data.exists(),'Never overwrite prior benchmark')
    torch.save(payload,data)
    value=dict(schema='d46_FPO_execution_microbenchmark_result_v1',status='MEASURED_NOT_FORMAL_ADMISSION',mode=a.mode,plan_sha256=sha(a.plan),source=source,checkpoint=plan['checkpoint'],native_files_sha256=plan['native_files_sha256'],behavior_sha256=plan['behavior_sha256'],tokens=plan['tokens'],flags=dict(execution_fast_prefix=fast,execution_fuse_fm_mc=fused),wall_seconds=seconds,median_seconds=statistics.median(seconds),peak_GPU_allocated_bytes=peak,steady_GPU_allocated_bytes=allocated,profiler_scalar_sync_events=syncs,per_scene_metrics=last_metrics,losses=[float(x)for x in payload['losses']],gradient_payload=str(data),gradient_payload_sha256=sha(data),descriptor_rebindings=rebindings,model_work=dict(prefills=conditions.prefill_calls,current_prefix_builds=actor.prepared_context_builds['current'],reference_prefix_builds=actor.prepared_context_builds['reference']),no_Adam_updates=True,new_reward_queries=0,policy_weights_not_exported=True,limits=['Single-GPU forward/backward microbenchmark on four saved TRAIN scenes, no optimizer, not end-to-end/scoring speed.','Fused MC matrix shape may change BF16 outputs/gradients; only native same-execution collection/recompute may be used formally.','GPU timing includes cached frozen inputs and fresh current C/P each loss. No trainable prefix survives between passes.'])
    put(dest/(a.mode+'.json'),value);print(json.dumps(value,indent=2))


def compare(a):
    import torch
    base=read(Path(a.result_dir)/'baseline.json');require(sha(base['gradient_payload'])==base['gradient_payload_sha256'],'Changed baseline gradient payload');data=torch.load(base['gradient_payload'],map_location='cpu',weights_only=False);rows=[]
    for mode in a.modes:
        value=read(Path(a.result_dir)/(mode+'.json'));require(value['plan_sha256']==base['plan_sha256']and value['native_files_sha256']==base['native_files_sha256']and value['behavior_sha256']==base['behavior_sha256'],'Different benchmark inputs')
        require(sha(value['gradient_payload'])==value['gradient_payload_sha256'],'Changed gradient payload')
        other=torch.load(value['gradient_payload'],map_location='cpu',weights_only=False)
        require(other['gradients'].keys()==data['gradients'].keys(),'Different gradient closure')
        maximum=0.;squared_diff=0.;squared_base=0.;dot=0.;squared_new=0.;bitexact=torch.equal(data['losses'],other['losses'])and torch.equal(data['energies'],other['energies']);close=True
        for name,x in data['gradients'].items():
            y=other['gradients'][name]
            require((x is None)==(y is None),'Different unused parameter '+name)
            if x is None:continue
            require(x.shape==y.shape and x.dtype==y.dtype,'Different gradient layout')
            delta=(x-y).double();maximum=max(maximum,float(delta.abs().max()));squared_diff+=float(delta.square().sum());squared_base+=float(x.double().square().sum());dot+=float((x.double()*y.double()).sum());squared_new+=float(y.double().square().sum())
            bitexact=bitexact and torch.equal(x,y);close=close and torch.allclose(x,y,atol=a.atol,rtol=a.rtol)
        rows.append(dict(mode=mode,energy_max_abs=float((data['energies']-other['energies']).abs().max()),loss_max_abs=float((data['losses']-other['losses']).abs().max()),gradient_max_abs=maximum,gradient_relative_L2=(squared_diff/max(squared_base,1e-30))**.5,gradient_cosine=dot/max((squared_base*squared_new)**.5,1e-30),bitexact=bitexact,all_gradients_elementwise_close=bool(close),comparison_tolerance=dict(atol=a.atol,rtol=a.rtol),speedup_vs_baseline=base['median_seconds']/value['median_seconds'],peak_memory_delta_bytes=value['peak_GPU_allocated_bytes']-base['peak_GPU_allocated_bytes'],profiler_scalar_sync_events=value['profiler_scalar_sync_events']))
    result=dict(schema='d46_FPO_execution_microbenchmark_comparison_v1',status='MEASURED_NOT_FORMAL_ADMISSION',baseline=base['source'],comparisons=rows,limits=['No weights/Adam state changed or returned to training.','Tolerance failures remain failures; this comparison does not authorize a typed upgrade or relax any fresh-ratio guard.'])
    if a.output:put(a.output,result)
    print(json.dumps(result,indent=2))


def main():
    p=argparse.ArgumentParser(__doc__);sub=p.add_subparsers(dest='action',required=True)
    q=sub.add_parser('prepare');q.add_argument('--checkpoint',required=True);q.add_argument('--behavior',required=True);q.add_argument('--baseline-source',required=True);q.add_argument('--candidate-source',required=True);q.add_argument('--result-dir',required=True);q.add_argument('--output',required=True);q.add_argument('--warmup',type=int,default=2);q.add_argument('--iterations',type=int,default=5)
    q=sub.add_parser('run');q.add_argument('--plan',required=True);q.add_argument('--mode',choices=MODES,required=True);q.add_argument('--expected-plan-sha256');q.add_argument('--_in-source',action='store_true',help=argparse.SUPPRESS)
    q=sub.add_parser('compare');q.add_argument('--result-dir',required=True);q.add_argument('--modes',nargs='+',default=['prefix','fusion','both']);q.add_argument('--atol',type=float,default=1e-5);q.add_argument('--rtol',type=float,default=1e-3);q.add_argument('--output')
    a=p.parse_args()
    if a.action=='prepare':
        require(0<=a.warmup<=10 and 1<=a.iterations<=20,'Finite benchmark iterations only');prepare(a)
    elif a.action=='run':run(a)
    else:compare(a)


if __name__=='__main__':main()
