"""Four from-scratch internal Scorers, with exact two-stream consumption state."""
from pathlib import Path
import numpy as np
import torch
from iqe.contracts import require
from iqe.io import atomic_json,atomic_torch,read_json,file_hash,digest,lock
from iqe.evaluation.retention import module_hash,_tensor_map_hash
from iqe.training.trainer import train
from iqe.internal_baseline.scorer import initialization_rng
from .binding import scenes_by_role
from .models import StudyScorer,load_bridge
from .objectives import StudySampler,terms,denominators,pair_population
from .data import ReusedContext
from .context import SpatialCache
from .training import runtime,trainer_config,preload
from .bank import load_bank,pool
from .metrics import metrics,select

CONTEXT_KEYS=('world','action','current_ego','world_valid','action_valid')
CELLS={'S00':{'spatial':False,'rich':False},'S01':{'spatial':False,'rich':True},
       'S10':{'spatial':True,'rich':False},'S11':{'spatial':True,'rich':True}}


def create_scorer(config,cell,*,diagnostic=False):
    require(cell in CELLS,'unknown Scorer cell')
    from .recipes import load
    recipe=load(config,'scorer_2x2');require(recipe['cells']==CELLS and recipe['dropout']==0.,'fixed Scorer 2x2/dropout definition')
    item=recipe['cells'][cell];scene=scenes_by_role(config)['scorer_fit'][0];x=ReusedContext(config,capacity=1).read(scene)
    visual_dim=None;views=len(read_json(config.root/'OFFLINE_BASE_CONTRACT.json')['input']['views'])
    if item['spatial']:
        v=SpatialCache(config,1).read(scene);visual_dim=v['visual'].shape[-1]
        require(v['visual'].shape[0]==views,'visual view count mismatch')
    contract=load_bridge(config).trajectory_contract
    with initialization_rng(config.seed):
        module=StudyScorer(x['world'].shape[-1],x['action'].shape[-1],contract,
                           spatial=item['spatial'],visual_dim=visual_dim,views=views)
    identity={'schema':'random_internal_study_scorer_v1','campaign':config.campaign_id,'cell':cell,
         'diagnostic':diagnostic,'init_seed':config.seed,'initial_state_hash':module_hash(module),
         'external_scorer_weights_loaded':False,'loaded_scorer_weights':[],
         'allowed_resume_lineage':config.campaign_id+'/'+cell+('/diagnostic' if diagnostic else ''),
         'parameters':{name:{'shape':list(p.shape),'count':p.numel()} for name,p in module.named_parameters()},
         'parameter_count':sum(p.numel() for p in module.parameters()),
         'Base_binding_hash':file_hash(config.root/'BASE_BINDING.json'),'pool_hash':pool(config)['hash'],
         'context':item,'common_initial_hash':_tensor_map_hash((k,v) for k,v in module.state_dict().items() if not k.startswith('spatial_reader.'))}
    module._iqe_initialization_identity=identity
    return module,identity


def fetch_context(config,scenes,device,cache,spatial_cache=None):
    # Explicit whitelist: target/score/future tensors never enter the model.
    x=cache.batch(scenes,device);context={k:x[k].detach() for k in CONTEXT_KEYS}
    if spatial_cache is not None:context.update(spatial_cache.batch(scenes,device))
    return context


@torch.no_grad()
def predict(config,module,role,device,*,diagnostic=None):
    meta,bank=load_bank(config,'deploy',role);scenes=scenes_by_role(config)[role]
    require(meta['scene_ids']==[s.scene_id for s in scenes],'Scorer bank scene order')
    cache=ReusedContext(config,128);spatial=SpatialCache(config,32) if module.spatial else None
    module.eval();pred=[]
    for at in range(0,len(scenes),16):
        sub=scenes[at:at+16];context=fetch_context(config,sub,device,cache,spatial)
        trajectory=torch.from_numpy(bank['trajectories'][at:at+len(sub)]).to(device)
        if diagnostic=='scene_mismatch':
            # Fixed cyclic offset over the full role, not just within a lucky batch.
            wrong=[scenes[(i+1)%len(scenes)] for i in range(at,at+len(sub))]
            context=fetch_context(config,wrong,device,cache,spatial)
        elif diagnostic=='trajectory_time_shuffle':
            order=torch.tensor([*range(trajectory.shape[2]-1,-1,-1)],device=device);trajectory=trajectory[:,:,order]
        elif diagnostic=='remove_W':context['world']=torch.zeros_like(context['world'])
        elif diagnostic=='shuffle_V':
            require(spatial is not None,'visual diagnosis on semantic model')
            wrong=[scenes[(i+1)%len(scenes)] for i in range(at,at+len(sub))];v=spatial.batch(wrong,device);context.update(v)
        elif diagnostic=='mask_V':
            require(spatial is not None,'visual diagnosis on semantic model');context['visual']=torch.zeros_like(context['visual'])
        elif diagnostic is not None:raise ValueError('unsupported diagnostic')
        result=module(context,trajectory)['values'];require(torch.isfinite(result).all().item(),'nonfinite Scorer inference')
        pred.append(result.cpu().numpy())
    values=np.concatenate(pred)
    return meta,bank,values


def train_scorer(config,cell,device='cuda:0',resume=None,stop_after=None):
    diagnostic=cell=='S00_DIAGNOSTIC';actual='S00' if diagnostic else cell
    require(actual in CELLS,'unknown Scorer cell')
    if diagnostic:require(stop_after is not None and 0<stop_after<=300,'diagnostic capped300 updates')
    from .recipes import load
    recipe=load(config,'scorer_2x2')
    require(recipe['natural_phase2']==recipe['comparison_phase2']==64,'fixed two-flow global exposure128')
    device=runtime(device);folder=config.root/('diagnostics' if diagnostic else 'scorers')/cell
    with lock(folder/'RUN.lock'):
        existing=list((config.root/'scorers').glob('*/SCORER_INITIALIZATION.json'))
        require(diagnostic or len(existing)<4 or (folder/'SCORER_INITIALIZATION.json').exists(),'four Scorer budget exhausted')
        meta,deploy=load_bank(config,'deploy','scorer_fit');roles=scenes_by_role(config)
        scenes=roles['scorer_fit'];require(meta['scene_ids']==[s.scene_id for s in scenes],'D_S scene order')
        rich_meta,rich=load_bank(config,'rich','scorer_fit') if CELLS[actual]['rich'] else (meta,deploy)
        require(rich_meta['scene_ids']==meta['scene_ids'],'natural distribution changed across rich bank')
        if diagnostic:
            pair,weights=pair_population(deploy['scores'],deploy['comparison_valid'])
            selected=list(dict.fromkeys(int(row[0]) for row in pair))[:64]
            if not selected:selected=list(range(min(64,len(scenes))))
            scenes=[scenes[i] for i in selected];deploy={k:v[selected] for k,v in deploy.items()};rich=deploy
        lookup={s.scene_id:s for s in scenes};index={s.scene_id:i for i,s in enumerate(scenes)}
        cache=preload(config,scenes);spatial=SpatialCache(config,64) if CELLS[actual]['spatial'] else None
        module,identity=create_scorer(config,actual,diagnostic=diagnostic);module.to(device)
        atomic_json(folder/'SCORER_INITIALIZATION.json',identity,immutable=True)
        sampler=StudySampler([s.scene_id for s in scenes],deploy['scores'],deploy['comparison_valid'],seed=config.seed+10,
                             bank_hash=digest([meta['hash'],rich_meta['hash'],[s.scene_id for s in scenes]]),phase_steps=0 if diagnostic else recipe['phase_switch_step'])
        atomic_json(folder/'COMPARISON_DISTRIBUTION.json',{'bank_hash':meta['hash'],'sampler_identity':sampler.identity,
            'pairs':sampler.pairs.tolist(),'weights':sampler.weights.tolist(),'probabilities':sampler.q.tolist() if sampler.q is not None else [],
            'definition':'q(x,i,j) proportional to non_tie*abs(gap)/all_valid_unordered_pair_count; sampled loss is unweighted',
            'status':'PRESENT' if sampler.q is not None else 'SCORER_COMPARISON_SIGNAL_ABSENT'},immutable=True)
        def fetch(entries,dev):
            batch_scenes=[lookup[e['scene_id']] for e in entries]
            context=fetch_context(config,batch_scenes,dev,cache,spatial)
            active=rich if entries[0]['phase']==1 and CELLS[actual]['rich'] else deploy
            width=max(active['scores'].shape[1],2)
            names=('trajectories','scores','valid','components','component_valid');rows={k:[] for k in names}
            natural=[];comparison=[]
            for e in entries:
                source=rich if e['phase']==1 and CELLS[actual]['rich'] else deploy
                n=index[e['scene_id']]
                candidate_indices=[e['pair_i'],e['pair_j']] if e['comparison'] else list(range(source['scores'].shape[1]))
                for key in names:
                    shape=(width,*source[key].shape[2:]);value=np.zeros(shape,dtype=source[key].dtype)
                    chosen=source[key][n,candidate_indices];value[:len(chosen)]=chosen;rows[key].append(value)
                is_padding=e.get('_padding',False)
                natural.append(e['natural'] and not is_padding);comparison.append(e['comparison'] and not is_padding)
            b={k:torch.from_numpy(np.stack(v)).to(dev) for k,v in rows.items()}
            b.update(context=context,natural=torch.tensor(natural,device=dev),comparison=torch.tensor(comparison,device=dev))
            return b
        def loss(model,b):return terms(model(b['context'],b['trajectories']),b)
        def validation(model,step):
            vm,vb,pred=predict(config,model,'stage_val',device)
            result=metrics(vb['scores'],vb['components'],select(pred,vb['valid']),vm['source_logs'],seed=config.bootstrap_seed,repetitions=config.bootstrap_repetitions)
            result['selection_key']=[result['selected_01'],-sum(result['safety'][c]['new_failures'] for c in ('no_at_fault_collisions','drivable_area_compliance'))]
            result.update(checkpoint_kind='raw_argmax',scorer_state_hash=module_hash(model),pool_hash=pool(config)['hash'])
            return result
        cfg=trainer_config(config,'scorer')
        if diagnostic:cfg.update(max_optimizer_steps=stop_after,warmup_steps=min(20,stop_after),validation_steps=[],checkpoint_every_steps=stop_after)
        deps={'campaign':file_hash(config.root/'CAMPAIGN_BINDING.json'),'cell':cell,'deployment_bank':meta['hash'],
              'pretraining_bank':rich_meta['hash'],'sampler':sampler.identity,'initialization':identity['initial_state_hash'],
              'source_code':file_hash(__file__),'diagnostic':diagnostic}
        atomic_json(folder/'RUN_IDENTITY.json',{'dependencies':deps,'training':cfg,'schema':'study_two_stream_v1'},immutable=True)
        train(module,loss,fetch,sampler,cfg,deps,folder,device=device,resume=resume,stop_after=stop_after,
              validate=None if diagnostic else validation,seed=config.seed+20,denominator_function=denominators,
              weights={'value':recipe['value_weight'],'components':recipe['components_weight'],'ranking':recipe['ranking_weight']})
        result=read_json(folder/'result.json')
        result.update(external_scorer_weights_loaded=False,training_scene_role='D_S_only',VLM_calls=0,
                      unique_scenes=len(scenes),natural_consumed=sampler.natural_consumed,comparison_consumed=sampler.comparison_consumed,
                      training_phase=1 if sampler.step<recipe['phase_switch_step'] else 2,sampler_identity=sampler.identity,
                      generation_parameters_present=False,scorer_new_parameters=identity['parameter_count'])
        atomic_json(folder/'SCORER_TRAINING_REPORT.json',result)
        if diagnostic:
            module.eval();x=fetch(sampler.peek(128),device)
            with torch.no_grad():pred=module(x['context'],x['trajectories'])['values'];truth=x['scores'];v=x['comparison']
            result['comparison_accuracy']=float((((pred[:,0]-pred[:,1])*(truth[:,0]-truth[:,1])>0)[v]).float().mean()) if v.any() else None
            atomic_json(folder/'OVERFIT_DIAGNOSIS.json',result)
        return result


def selected_scorer(config,cell,device):
    require(cell in CELLS,'Scorer identity')
    folder=config.root/'scorers'/cell;result=read_json(folder/'result.json')
    require(result['status']=='COMPLETE' and result['optimizer_steps']==6000,'Scorer fixed-budget training incomplete')
    require(file_hash(folder/'best.pt')==result['best_checkpoint_hash'],'selected Scorer corrupted')
    model,identity=create_scorer(config,cell)
    require(identity==read_json(folder/'SCORER_INITIALIZATION.json'),'Scorer initialization/lineage changed')
    model.load_state_dict(torch.load(folder/'best.pt',map_location='cpu',weights_only=True),strict=True)
    return model.to(device).eval().requires_grad_(False)
