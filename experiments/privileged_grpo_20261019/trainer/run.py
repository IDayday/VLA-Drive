"""One independent experiment replica. All outputs go to a new run directory."""
import argparse,copy,hashlib,json,os,random,signal,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
for key,value in {'PYTHONDONTWRITEBYTECODE':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false','NO_ALBUMENTATIONS_UPDATE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}.items():os.environ.setdefault(key,value)
import numpy as np
import torch
from asset_adapter import config,rows,scene,mapped,validate,digest
import engine
sys.path.insert(0,str(ROOT/'vendor/grpo_smoke_v1'))
import smoke as core
from common import decode
from pipeline import load_images
from score_trace import score_one
STOP=False

def stop(signum,frame):
    global STOP
    STOP=True
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)

def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp-'+str(os.getpid()))
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)
def read(path):return json.loads(Path(path).read_text())
def identity(cfg,c,prepared):
    source=Path(c['source_root']);return {'training':cfg,'checkpoint':c['checkpoint_sha256'],'train':digest(c['train_pairs']),'test':digest(c['test_pairs']),'prepared':digest(prepared/'PREPARATION.json'),'code':{str(p.relative_to(ROOT)):digest(p) for p in sorted(list(ROOT.glob('*.py'))+list((ROOT/'vendor').rglob('*.py'))) if '__pycache__' not in p.parts},'model_source':{str(p.relative_to(source)):digest(p) for p in sorted((source/'starVLA').rglob('*.py'))},'model_config':digest(c['model_config']),'tokenizer':{p.name:digest(p) for p in sorted(Path(c['tokenizer_root']).glob('*.json'))}}
def validate_plan(cfg,prepared,mode):
    expected_arms={'A','B','C','F','E','A_VLM','C_VLM','E_VLM','A_DiT','C_DiT','E_DiT'}
    if set(cfg['arms'])!=expected_arms or len(cfg['arms'])!=11 or cfg['evaluation_replicates']!=3:raise ValueError('Expected all 11 arms and three paired evaluation replicates')
    if cfg['checkpoint_every_updates']<0 or min(cfg['vlm_lr'],cfg['action_lr'],cfg['gradient_clip'])<=0 or cfg['beta']<0:raise ValueError('Invalid optimizer/checkpoint bounds')
    p=read(prepared/'PREPARATION.json')
    for name,h in p['files'].items():
        if digest(prepared/name)!=h:raise ValueError('Prepared file changed: '+name)
    if cfg['group_size']!=8 or cfg['accumulation_scenes']!=4 or cfg['steps']!=10 or cfg['eta']<=0:raise ValueError('Changed G/accumulation/flow contract')
    if mode!='smoke':
        if not cfg.get('formal_protocol_locked') or not isinstance(cfg.get('epochs'),int) or cfg['epochs']<=0 or not cfg.get('training_seeds') or not cfg.get('wall_seconds'):raise ValueError('Formal epochs, seeds, wall budget and protocol lock required')
        if cfg['wall_seconds']<=0 or len(set(cfg['training_seeds']))!=len(cfg['training_seeds']) or not all(isinstance(s,int) for s in cfg['training_seeds']):raise ValueError('Invalid budget or duplicate/noninteger training seeds')
        if p['status']!='PASS':raise ValueError('Full F coverage not accepted: '+p['status'])

def load_model(c):
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('Each replica requires exactly one visible accelerator')
    sys.path.insert(0,c['source_root'])
    from omegaconf import OmegaConf
    from starVLA.model.framework.QwenOFT import Qwenvl_OFT
    mc=OmegaConf.load(c['model_config']);mc.framework.qwenvl.base_vlm=c['tokenizer_root'];mc.framework.qwenvl.attn_implementation='sdpa'
    model=Qwenvl_OFT(mc,infer_not_load_wan=1);state=torch.load(c['checkpoint'],map_location='cpu',weights_only=True,mmap=True);model.load_state_dict(state,strict=True);del state
    return model.to('cuda:0').float().eval()

def snapshot(path,model,opt,ident,arm,training_seed,completed,elapsed,parent,pending_monitor=None):
    if path.exists():raise FileExistsError(path)
    save_started=time.monotonic()
    state={'schema':'grpo-resume-v1','model':model.state_dict(),'optimizer':opt.state_dict(),'identity':ident,'arm':arm,'seed':training_seed,'completed_updates':completed,'elapsed_s':elapsed,'parent_checkpoint':parent,'pending_monitor':pending_monitor,'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()}
    temp=path.with_suffix('.partial');torch.save(state,temp);os.replace(temp,path)
    save_json(path.with_suffix('.json'),{'checkpoint':str(path),'sha256':digest(path),'completed_updates':completed,'elapsed_s':elapsed+time.monotonic()-save_started,'arm':arm,'seed':training_seed,'parent_checkpoint':parent,'pending_monitor':pending_monitor})

def restore(path,model,opt,ident,arm,seed):
    side=read(path.with_suffix('.json'))
    if digest(path)!=side['sha256']:raise ValueError('Resume checkpoint hash mismatch')
    saved=torch.load(path,map_location='cpu',weights_only=False) # trusted task-generated full optimizer/RNG checkpoint
    if saved['identity']!=ident or saved['arm']!=arm or saved['seed']!=seed:raise ValueError('Resume identity mismatch; cannot change experiment/budget')
    model.load_state_dict(saved['model'],strict=True);opt.load_state_dict(saved['optimizer']);random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng']);torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
    return saved['completed_updates'],max(saved['elapsed_s'],side['elapsed_s'])

def main(a):
    global STOP
    started=time.monotonic();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False);c=config(a.assets);cfg=read(a.training);prepared=Path(a.prepared).resolve();mode=a.mode
    for key,name in [('HF_HOME','hf'),('TORCH_HOME','torch'),('XDG_CACHE_HOME','xdg'),('TRITON_CACHE_DIR','triton'),('NUMBA_CACHE_DIR','numba'),('MPLCONFIGDIR','mpl')]:
        path=out/'cache'/name;path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path)
    def status(stage,**extra):save_json(out/'STATUS.json',{'stage':stage,'pid':os.getpid(),'elapsed_current_s':time.monotonic()-started,**extra})
    try:
        validate_plan(cfg,prepared,mode)
        if a.arm not in cfg['arms']:raise ValueError('Unknown arm')
        if mode!='smoke' and a.seed not in cfg['training_seeds'] and mode!='initial':raise ValueError('Unapproved training seed')
        if mode=='smoke' and (not 1<=a.smoke_updates<=2 or not 0<a.smoke_wall_seconds<=1800):raise ValueError('Smoke is limited to <=2 updates and <=1800 seconds')
        status('asset_validation');validation=validate(c,full_hash=True)
        if validation['status']!='PASS':raise ValueError(json.dumps(validation))
        ident=identity(cfg,c,prepared);save_json(out/'IDENTITY.json',ident)
        for name,h in ident['code'].items():
            src=ROOT/name;dst=out/'code'/name;dst.parent.mkdir(parents=True,exist_ok=True);data_bytes=src.read_bytes()
            if hashlib.sha256(data_bytes).hexdigest()!=h:raise RuntimeError('Code changed while snapshotting')
            dst.write_bytes(data_bytes)
        save_json(out/'INVOCATION.json',vars(a));splits=read(prepared/'SPLITS.json');prompts={r['token']:r for r in read(prepared/'PROMPTS.json')['scenes']};data={r['token']:r for s in ['train','test'] for r in rows(c,s)}
        status('loading_model');model=load_model(c);model.requires_grad_(False);frozen_base_hash=engine.state_hash(model,'all')
        input_arm=engine.input_arm(a.arm);device='cuda:0';completed=0;carry=0.;parent=str(Path(a.resume).resolve()) if a.resume else None
        def batch_for(token):
            row=data[token];sc=scene(row,c);images=load_images(sc);prompt=prompts[token]['arms'][input_arm]['text'];batch=core.build_inputs(model,sc,prompt,images)
            # Text-only matching is insufficient; check actual multimodal IDs/special positions.
            if input_arm=='F':
                other=core.build_inputs(model,sc,prompts[token]['arms']['C']['text'],images)
                if batch['input_ids'].shape!=other['input_ids'].shape:raise ValueError('C/F multimodal token length mismatch')
                for kind in ['history','action']:
                    if not torch.equal(model._find_token_positions(batch['input_ids'],model._special_token_ids[kind]),model._find_token_positions(other['input_ids'],model._special_token_ids[kind])):raise ValueError('C/F insertion positions mismatch')
            return sc,batch
        def scored(token,raw,role,rep,noise):
            r=data[token];trajectories=[];metrics=[];candidates=[]
            for i,x in enumerate(raw.detach().cpu().numpy()):
                try:p=decode(x);s=score_one(p,mapped(r['source_paths']['metric'],c),c)
                except Exception as exc:
                    save_json(out/'FAILED_CANDIDATE.json',{'token':token,'candidate':i,'role':role,'replicate':rep,'seed':noise,'type':type(exc).__name__,'message':str(exc)});raise
                trajectories.append(p);metrics.append(s['metrics']);candidates.append({'candidate':i,'trajectory':p.tolist(),'metrics':s['metrics']})
            return [m['PDMS'] for m in metrics],{'token':token,'log':r['log'],'drive':r['log'].rsplit('_',2)[0],'arm':a.arm,'input':input_arm,'role':role,'replicate':rep,'candidates':candidates,'seed':noise,'base_pdms':r['base_score']['metrics']['PDMS'],'exploration':engine.exploration(trajectories,metrics,r['base_score']['metrics']['PDMS'])}
        def evaluate(tokens,stage,reps=1):
            directory=out/'evaluation'/stage;directory.mkdir(parents=True,exist_ok=False);expected=[]
            for token in tokens:
                sc,batch=batch_for(token)
                with torch.no_grad():cond=core.context(model,batch,sc['state'])
                for rep in range(reps):
                    if STOP:raise InterruptedError('Evaluation interrupted; incomplete coverage')
                    if time.monotonic()-started+carry>budget:raise TimeoutError('Cumulative evaluation budget exceeded')
                    value=engine.seed('eval',token,rep);noise=engine.make_noise(device,value)
                    with torch.no_grad():raw,_=engine.sampling(model,cond,core.velocity,noise,cfg['eta'])
                    _,record=scored(token,raw,stage,rep,value);record['noise_sha256']=engine.noise_hash(noise);name=f'{token}_{rep}.json';save_json(directory/name,record);expected.append(name)
                    status('evaluation',stage_name=stage,completed_groups=len(expected),expected_groups=len(tokens)*reps)
            actual=sorted(p.name for p in directory.glob('*.json'))
            if actual!=sorted(expected) or len(expected)!=len(tokens)*reps:raise RuntimeError('Evaluation coverage mismatch')
            save_json(directory/'ACCEPTANCE.json',{'status':'PASS','groups':len(expected),'candidates':len(expected)*8,'token_count':len(tokens),'files':expected})
        # Replay official Base score on a training scene before any rollout.
        replay_token=splits['train'][0];r=data[replay_token]
        if score_one(r['input_trajectory'],mapped(r['source_paths']['metric'],c),c)['metrics']!=r['base_score']['metrics']:raise ValueError('Official Base score replay differs')
        save_json(out/'BASE_SCORE_REPLAY.json',{'status':'PASS','token':replay_token})
        if mode=='smoke':
            base_scene=scene(r,c);base_images=load_images(base_scene);base_batch=core.build_inputs(model,base_scene,base_scene['instruction'],base_images)
            model._inference_qwen_forward_mode='optimized'
            with torch.inference_mode():
                torch.manual_seed(719);torch.cuda.manual_seed_all(719)
                original=model.predict_action_infer_1d([dict(image=base_images,state=base_scene['state'],lang=base_scene['instruction'],token=replay_token)])['normalized_actions']
                torch.manual_seed(719);torch.cuda.manual_seed_all(719)
                q=core.queries(model,base_batch,base_scene['state'])
                with torch.autocast('cuda',enabled=False):local=model.action_model.predict_action(q).cpu().numpy()
            error=float(np.max(np.abs(original-local)))
            if not np.allclose(original,local,rtol=1e-5,atol=1e-5):raise ValueError('Original model forward differs from training path')
            save_json(out/'BASE_FORWARD_REPLAY.json',{'status':'PASS','max_abs_error':error,'token':replay_token})
            del original,local,q,base_batch,base_images
        budget=a.smoke_wall_seconds if mode=='smoke' else cfg['wall_seconds']
        if mode in ['initial','final']:
            if mode=='initial' and a.arm not in ['A','B','C','F','E']:raise ValueError('Only five unique initial inputs')
            if mode=='final':
                if not a.resume:raise ValueError('Final evaluation needs checkpoint')
                cp=Path(a.resume);side=read(cp.with_suffix('.json'))
                if digest(cp)!=side['sha256']:raise ValueError('Final checkpoint hash mismatch')
                state=torch.load(cp,map_location='cpu',weights_only=False)
                if state['identity']!=ident or state['arm']!=a.arm or state['seed']!=a.seed or state['completed_updates']!=cfg['epochs']*316:raise ValueError('Final checkpoint not at fixed approved endpoint')
                model.load_state_dict(state['model'],strict=True);del state
            evaluate(splits['test'],mode,cfg['evaluation_replicates']);save_json(out/'RESULT.json',{'status':'COMPLETE','mode':mode,'arm':a.arm,'checkpoint':a.resume,'elapsed_s':time.monotonic()-started});return
        reference=copy.deepcopy(model).eval().requires_grad_(False)
        inventory=engine.scope(model,engine.arm_scope(a.arm));save_json(out/'PARAMETERS.json',inventory);frozen_hash=engine.state_hash(model);opt=engine.optimizer(model,cfg)
        if a.resume:completed,carry=restore(Path(a.resume),model,opt,ident,a.arm,a.seed)
        # Check initial function under all three freeze ranges on a training-side scene.
        if completed==0:
            sc,batch=batch_for(replay_token);noises=engine.make_noise(device,engine.seed('scope-replay',replay_token));outputs={}
            with torch.no_grad():
                for scope_name in engine.SCOPES:
                    engine.scope(model,scope_name);cond=core.context(model,batch,sc['state']);raw,_=engine.sampling(model,cond,core.velocity,noises,cfg['eta']);outputs[scope_name]=raw
            if not all(torch.equal(outputs['joint'],x) for x in outputs.values()):raise ValueError('Freeze scope changed initial outputs')
            engine.scope(model,engine.arm_scope(a.arm));save_json(out/'SCOPE_REPLAY.json',{'status':'PASS','input':input_arm,'three_scopes_exact':True,'noise_sha256':engine.noise_hash(noises)})
        total=a.smoke_updates if mode=='smoke' else cfg['epochs']*316
        if completed>total:raise ValueError('Resume past target')
        ckpts=out/'checkpoints';ckpts.mkdir();initial=engine.witnesses(model);all_updates=[]
        def interrupted(reason,pending_monitor=None):
            snapshot(ckpts/f'update_{completed:06d}_interrupt.pt',model,opt,ident,a.arm,a.seed,completed,time.monotonic()-started+carry,parent,pending_monitor)
            save_json(out/'RESULT.json',{'status':'INTERRUPTED','completed_updates':completed,'reason':reason,'formal_final':False})
        if mode!='smoke' and completed==0:
            try:evaluate(splits['monitor'],'monitor_initial')
            except (TimeoutError,InterruptedError) as exc:interrupted(str(exc),'monitor_initial');return
        if a.resume and read(Path(a.resume).with_suffix('.json')).get('pending_monitor'):
            pending=read(Path(a.resume).with_suffix('.json'))['pending_monitor']
            try:evaluate(splits['monitor'],pending)
            except (TimeoutError,InterruptedError) as exc:interrupted(str(exc),pending);return
        while completed<total:
            if STOP or time.monotonic()-started+carry>=budget:
                snapshot(ckpts/f'update_{completed:06d}_interrupt.pt',model,opt,ident,a.arm,a.seed,completed,time.monotonic()-started+carry,parent)
                save_json(out/'RESULT.json',{'status':'INTERRUPTED','completed_updates':completed,'reason':'signal' if STOP else 'cumulative_budget','formal_final':False});return
            epoch=completed//316;window=completed%316;tokens=engine.order(splits['train'],a.seed,epoch)[window*4:(window+1)*4]
            if len(tokens)!=4:raise ValueError('Incomplete accumulation window')
            opt.zero_grad(set_to_none=True);records=[];before=engine.witnesses(model)
            for token in tokens:
                status('training',completed_updates=completed,epoch=epoch,token=token)
                sc,batch=batch_for(token);noise_seed=engine.seed('train',a.seed,epoch,token);noise=engine.make_noise(device,noise_seed)
                condition=lambda policy:core.context(policy,batch,sc['state'])
                reward=lambda raw:scored(token,raw,'train',epoch,noise_seed)
                record=engine.backward_scene(model,reference,condition,core.velocity,noise,reward,cfg,4);records.append(record)
            stats=engine.step(model,opt,cfg['gradient_clip']);completed+=1
            update={'update':completed,'epoch':epoch,'tokens':tokens,'groups':records,'parameters':engine.changes(before,engine.witnesses(model)),**stats,'elapsed_s':time.monotonic()-started+carry,'peak_memory_gib':torch.cuda.max_memory_allocated()/2**30}
            save_json(out/'updates'/f'{completed:06d}.json',update);all_updates.append({'update':completed,**stats,'changes':update['parameters']})
            if mode!='smoke' and completed%316==0:
                try:evaluate(splits['monitor'],f'monitor_epoch_{completed//316}')
                except (TimeoutError,InterruptedError) as exc:interrupted(str(exc),f'monitor_epoch_{completed//316}');return
            save_every=cfg['checkpoint_every_updates']
            if completed==total or completed%316==0 or (save_every and completed%save_every==0):
                status('checkpoint_validation',completed_updates=completed)
                if engine.state_hash(model)!=frozen_hash:raise RuntimeError('Frozen parameters/buffers changed')
                if engine.state_hash(reference,'all')!=frozen_base_hash:raise RuntimeError('Reference changed')
                status('saving_checkpoint',completed_updates=completed)
                snapshot(ckpts/f'update_{completed:06d}.pt',model,opt,ident,a.arm,a.seed,completed,time.monotonic()-started+carry,parent)
        if not (ckpts/f'update_{completed:06d}.pt').exists():
            if engine.state_hash(model)!=frozen_hash or engine.state_hash(reference,'all')!=frozen_base_hash:raise RuntimeError('Frozen/reference state changed')
            snapshot(ckpts/f'update_{completed:06d}.pt',model,opt,ident,a.arm,a.seed,completed,time.monotonic()-started+carry,parent)
        changed=engine.changes(initial,engine.witnesses(model))
        if mode=='smoke':
            required={'joint':['vlm','action'],'vlm_only':['vlm'],'dit_only':['action']}[engine.arm_scope(a.arm)]
            if not all(changed[k]>0 and any(u['gradient_norms'][k]>0 for u in all_updates) for k in required):raise RuntimeError('Smoke did not demonstrate updates in every trainable group')
        save_json(out/'RESULT.json',{'status':'COMPLETE','mode':mode,'arm':a.arm,'seed':a.seed,'completed_updates':completed,'checkpoint':str(ckpts/f'update_{completed:06d}.pt'),'frozen_parameters_buffers_exact':True,'reference_exact':True,'parameter_changes':changed,'elapsed_s':time.monotonic()-started+carry,'formal_training':mode=='train'})
        status('complete',completed_updates=completed)
    except BaseException as exc:
        save_json(out/'ERROR.json',{'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()});status('failed');raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--assets',required=True);p.add_argument('--training',required=True);p.add_argument('--prepared',required=True);p.add_argument('--out',required=True);p.add_argument('--arm',required=True);p.add_argument('--seed',type=int,default=20261019);p.add_argument('--mode',choices=['smoke','initial','train','final'],required=True);p.add_argument('--resume');p.add_argument('--smoke-updates',type=int,default=2);p.add_argument('--smoke-wall-seconds',type=int,default=1800);main(p.parse_args())
