"""Locked stage_val selection, heldout margin, final dev evidence, and online bundle."""
from itertools import combinations
from pathlib import Path
import csv,time
import numpy as np
import torch
from torch import nn
from iqe.contracts import require
from iqe.io import atomic_json,atomic_torch,atomic_bytes,complete,read_json,file_hash,digest
from iqe.data.sources import load_observation
from .binding import scenes_by_role
from .bank import load_bank,pool
from .experiments import load_endpoint
from .scorer_training import CELLS,selected_scorer,predict,CONTEXT_KEYS
from .metrics import report_metrics as metrics,select


def evaluate_scorers(config,device='cuda:0'):
    rows=[]
    for cell in CELLS:
        m=selected_scorer(config,cell,device);meta,bank,pred=predict(config,m,'stage_val',device)
        stats=metrics(bank['scores'],bank['components'],select(pred,bank['valid'],expert_ids=meta['provenance']['sources']),meta['source_logs'],seed=config.bootstrap_seed,repetitions=config.bootstrap_repetitions)
        stats.update(cell=cell,checkpoint=str(config.root/'scorers'/cell/'best.pt'),checkpoint_hash=file_hash(config.root/'scorers'/cell/'best.pt'))
        rows.append(stats);del m
    winner=max(rows,key=lambda r:(r['selected_01'],-sum(r['safety'][k]['new_failures'] for k in ('no_at_fault_collisions','drivable_area_compliance')),-list(CELLS).index(r['cell'])))
    result={'selection_role':'stage_val','pool_hash':pool(config)['hash'],'rows':rows,'selected':winner,'dev_used_for_selection':False}
    result['hash']=digest(result);atomic_json(config.root/'SCORER_MATRIX_SELECTION.json',result,immutable=True)
    write_csv(config.root/'SCORER_2X2_RESULTS.csv',rows)
    return result


def write_csv(path,rows):
    fields=['cell','scenes','logs','base_01','oracle_01','selected_01','oracle_gain_01','regret_01','selected_gain_01','selected_over_best_fixed_01','oracle_capture','switch_rate','high_to_zero','selected_zero']
    with open(path,'w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)


def calibrate(config,device='cuda:0'):
    from .recipes import load
    recipe=load(config,'final_evaluation')
    require(not recipe['automatic_ACTIVE'] and recipe['bootstrap_repetitions']==config.bootstrap_repetitions and recipe['bootstrap_seed']==config.bootstrap_seed,'fixed final evaluation/activation policy')
    selection=read_json(config.root/'SCORER_MATRIX_SELECTION.json');cell=selection['selected']['cell']
    rules={'role':'selector_cal','delta_grid':recipe['margin_grid'],
           'risk_budget':{'new_NC_hard_failures':recipe['margin_new_NC_budget'],'new_DAC_hard_failures':recipe['margin_new_DAC_budget'],'high_to_zero':recipe['margin_high_zero_budget']},
           'selection':'largest empirical selected_mean, ties fewer switches then larger margin',
           'unknown_scene_safety_guarantee':False,'pool_hash':pool(config)['hash'],'scorer_hash':selection['selected']['checkpoint_hash']}
    atomic_json(config.root/'MARGIN_RULES_BEFORE_CALIBRATION.json',rules,immutable=True)
    m=selected_scorer(config,cell,device);meta,bank,pred=predict(config,m,'selector_cal',device);rows=[]
    for delta in rules['delta_grid']:
        stats=metrics(bank['scores'],bank['components'],select(pred,bank['valid'],delta,expert_ids=meta['provenance']['sources']),meta['source_logs'],seed=config.bootstrap_seed,repetitions=config.bootstrap_repetitions)
        stats['delta']=delta;stats['feasible']=stats['high_to_zero']<=recipe['margin_high_zero_budget'] and stats['safety']['no_at_fault_collisions']['new_failures']<=recipe['margin_new_NC_budget'] and stats['safety']['drivable_area_compliance']['new_failures']<=recipe['margin_new_DAC_budget'];rows.append(stats)
    eligible=[r for r in rows if r['feasible']]
    best=max(eligible,key=lambda r:(r['selected_01'],-r['switch_count'],r['delta'])) if eligible else None
    result={'cell':cell,'selected_delta':best['delta'] if best else None,'always_Base':best is None or best['switch_count']==0,
            'rows':rows,'rules_hash':file_hash(config.root/'MARGIN_RULES_BEFORE_CALIBRATION.json'),
            'Scorer_success_claim':False,'pool_hash':pool(config)['hash'],'scorer_checkpoint_hash':selection['selected']['checkpoint_hash']}
    result['hash']=digest(result);atomic_json(config.root/'SELECTOR_MARGIN.json',result,immutable=True)
    return result


def evaluate_final(config,device='cuda:0'):
    selection=read_json(config.root/'SCORER_MATRIX_SELECTION.json');margin=read_json(config.root/'SELECTOR_MARGIN.json')
    frozen={'Base_binding':file_hash(config.root/'BASE_BINDING.json'),'pool_hash':pool(config)['hash'],
            'selection':file_hash(config.root/'SCORER_MATRIX_SELECTION.json'),'margin':file_hash(config.root/'SELECTOR_MARGIN.json'),
            'fixed_dev_ids_hash':digest([s.scene_id for s in scenes_by_role(config)['dev_report']])}
    atomic_json(config.root/'FINAL_EVALUATION_LOCK.json',frozen,immutable=True)
    meta,bank=load_bank(config,'deploy','dev_report');r=bank['scores'];rows=[];predictions={};ids=meta['provenance']['sources']
    for index,eid in enumerate(pool(config)['expert_ids']):
        out=metrics(r,bank['components'],np.full(len(r),index),meta['source_logs'],seed=config.bootstrap_seed,repetitions=config.bootstrap_repetitions);out['cell']=eid;rows.append(out)
    out=metrics(r,bank['components'],r.argmax(1),meta['source_logs'],seed=config.bootstrap_seed,repetitions=config.bootstrap_repetitions);out['cell']='Oracle';rows.append(out)
    for cell in CELLS:
        m=selected_scorer(config,cell,device);_,_,pred=predict(config,m,'dev_report',device);predictions[cell]=pred
        out=metrics(r,bank['components'],select(pred,bank['valid'],expert_ids=ids),meta['source_logs'],seed=config.bootstrap_seed,repetitions=config.bootstrap_repetitions);out['cell']=cell;rows.append(out);del m
    chosen=selection['selected']['cell'];choice=np.zeros(len(r),int) if margin['selected_delta'] is None else select(predictions[chosen],bank['valid'],margin['selected_delta'],expert_ids=ids)
    out=metrics(r,bank['components'],choice,meta['source_logs'],seed=config.bootstrap_seed,repetitions=config.bootstrap_repetitions);out['cell']=chosen+'_margin';rows.append(out)
    # Preserve the actual common-population predictions and choices. They are
    # report evidence only and never become training or selection inputs.
    import io
    arrays=io.BytesIO()
    ordered_cells=list(CELLS)
    np.savez(arrays,scene_ids=np.asarray(meta['scene_ids']),source_logs=np.asarray(meta['source_logs']),
             expert_ids=np.asarray(ids),true_scores=r,valid=bank['valid'],
             scorer_cells=np.asarray(ordered_cells),predicted_scores=np.stack([predictions[c] for c in ordered_cells]),
             raw_argmax=np.stack([select(predictions[c],bank['valid'],expert_ids=ids) for c in ordered_cells]),
             calibrated_margin_selected=choice)
    predictions_path=config.root/'FINAL_PREDICTIONS.npz'
    atomic_bytes(predictions_path,arrays.getvalue(),immutable=True)
    complete(config.root/'FINAL_PREDICTIONS.COMPLETE.json',
             {**frozen,'dev_bank_hash':meta['hash'],
              'scorer_checkpoints':{c:file_hash(config.root/'scorers'/c/'best.pt') for c in ordered_cells}},
             [predictions_path],requested_scenes=len(r),all_candidates_valid=bool(bank['valid'].all()),
             schema='locked_dev_candidate_predictions_v1',used_for_training=False)
    subsets=[]
    for count in range(r.shape[1]):
        for group in combinations(range(1,r.shape[1]),count):
            indices=[0,*group];truth=r[:,indices];comp=bank['components'][:,indices]
            for cell,pred in predictions.items():
                out=metrics(truth,comp,select(pred[:,indices],bank['valid'][:,indices],expert_ids=[ids[i] for i in indices]),meta['source_logs'],seed=config.bootstrap_seed,repetitions=config.bootstrap_repetitions)
                out.update(cell=cell,expert_ids=[pool(config)['expert_ids'][i] for i in indices],margin_recalibrated_for_subset=False);subsets.append(out)
    result={'scope':'entire_fixed_dev_report','requested_scenes':len(meta['scene_ids']),'results':rows,'subsets':subsets,
            'bindings':frozen,'safety_reference':'same Base checkpoint','units':'[0,1]','training_seed_count':1}
    atomic_json(config.root/'FINAL_SYSTEM_RESULTS.json',result,immutable=True);write_csv(config.root/'FINAL_SYSTEM_RESULTS.csv',rows)
    m=selected_scorer(config,chosen,device);diagnoses={}
    # Read the already fitted D_S only after choices and final dev predictions
    # are locked. This separates training fit from heldout selection failure;
    # it cannot change a checkpoint, pool, margin or optimizer.
    fit_meta,fit_bank,fit_pred=predict(config,m,'scorer_fit',device)
    fit_metrics=metrics(fit_bank['scores'],fit_bank['components'],
                        select(fit_pred,fit_bank['valid'],expert_ids=fit_meta['provenance']['sources']),
                        fit_meta['source_logs'],seed=config.bootstrap_seed,repetitions=config.bootstrap_repetitions)
    fit_diagnostic={'cell':chosen,'scope':'entire_fixed_D_S_after_selection_lock','metrics':fit_metrics,
                    'bank_hash':fit_meta['hash'],'checkpoint_hash':selection['selected']['checkpoint_hash'],
                    'selection_lock_hash':file_hash(config.root/'FINAL_EVALUATION_LOCK.json'),
                    'optimizer_updates':0,'used_for_checkpoint_or_recipe_selection':False}
    atomic_json(config.root/'SCORER_FIT_DIAGNOSTIC.json',fit_diagnostic,immutable=True)
    for kind in ['scene_mismatch','trajectory_time_shuffle','remove_W']+(['shuffle_V','mask_V'] if m.spatial else []):
        _,_,p=predict(config,m,'dev_report',device,diagnostic=kind)
        diagnoses[kind]=metrics(r,bank['components'],select(p,bank['valid'],expert_ids=ids),meta['source_logs'],seed=config.bootstrap_seed,repetitions=config.bootstrap_repetitions)
    atomic_json(config.root/'SCORER_INFORMATION_DIAGNOSES.json',diagnoses,immutable=True)
    export_bundle(config,chosen)
    raw=next(x for x in rows if x['cell']==chosen)
    conclusion='POSITIVE_BASELINE_SIGNAL' if raw['selected_over_best_fixed_01']>0 and raw['selected_gain_01']>0 else 'CANDIDATE_GAIN_NOT_REALIZED'
    if any(raw['safety'][k]['new_failures'] for k in raw['safety']) or raw['high_to_zero']:conclusion='SCORE_GAIN_WITH_SAFETY_REGRESSION' if raw['selected_gain_01']>0 else conclusion
    verdict={'IMPLEMENTATION':'PASS','DATA_AVAILABLE':'PASS','TRAINING_COMPLETE':'PASS','HELDOUT_GAIN':raw['oracle_gain_01']>0,
             'SELECTION_GAIN':raw['selected_gain_01'],'GAIN_OVER_BEST_FIXED':raw['selected_over_best_fixed_01'],'SAFETY_RESULT':raw['safety'],
             'SCIENTIFIC_CONCLUSION':conclusion,'MULTI_SEED':'UNTESTED','ACTIVE_UPDATED':False,'EXTERNAL_SCORER_WEIGHTS':False,
             'SCORER_TRAIN_FIT_GAIN':fit_metrics['selected_gain_01'],
             'SCORER_GENERALIZATION_GAP_SIGNAL':fit_metrics['selected_gain_01']>0 and raw['selected_gain_01']<=0}
    atomic_json(config.root/'VERDICT.json',verdict)
    return verdict


def export_bundle(config,cell):
    selected=pool(config);folder=config.root/'research_bundle';payload={'kind':'iqe_specialist_study_research_bundle_v1',
        'Base_binding':read_json(config.root/'BASE_BINDING.json'),'contract':read_json(config.root/'OFFLINE_BASE_CONTRACT.json'),
        'pool':selected,'expert_states':[torch.load(x['checkpoint'],map_location='cpu',weights_only=True) for x in selected['experts']],
        'scorer_cell':cell,'scorer':torch.load(config.root/'scorers'/cell/'best.pt',map_location='cpu',weights_only=True),
        'initialization':read_json(config.root/'scorers'/cell/'SCORER_INITIALIZATION.json'),
        'margin':read_json(config.root/'SELECTOR_MARGIN.json'),'selection_rule_default':'raw_argmax_Base_then_stable_expert_id',
        'inference_dependencies':['own_Base_checkpoint','own_expert_states','own_scorer_state'],
        'GT_required':False,'metric_cache_required':False,'ACTIVE':False}
    atomic_torch(folder/'bundle.pt',payload,immutable=True)
    atomic_json(folder/'BUNDLE.json',{'checksum':file_hash(folder/'bundle.pt'),'pool_hash':selected['hash'],'ACTIVE':False},immutable=True)
    return folder/'bundle.pt'


class ResearchSystem(nn.Module):
    """Deployment wrapper: one Base, N independent heads, one internal Scorer."""
    def __init__(self,adapter,experts,scorer,expert_ids=None):
        super().__init__();self.adapter=adapter;self.experts=nn.ModuleList(experts);self.scorer=scorer
        self.expert_ids=tuple(expert_ids or [f'expert_{i}' for i in range(len(experts)+1)])
        from .metrics import stable_order
        base,_=stable_order(len(experts)+1,self.expert_ids)
        require(base==0,'deployment Base branch must retain its registered identity')
    @torch.no_grad()
    def shared_forward(self,observations,*,expert_count=None,score=True):
        from .context import SharedCapture
        count=len(self.experts) if expert_count is None else expert_count
        calls={'visual':0,'language':0}
        if score and count and self.scorer.spatial:
            capture=SharedCapture(self.adapter)
            try:planner,x=capture.encode(observations);calls=capture.count
            finally:capture.close()
        else:
            from iqe.internal_baseline.features import encode_joint
            network=self.adapter.framework.qwen_vl_interface.model.model
            def hit(name):
                def callback(*_):calls[name]+=1
                return callback
            handles=[network.visual.register_forward_hook(hit('visual')),network.language_model.register_forward_hook(hit('language'))]
            try:
                planner,context=encode_joint(self.adapter,observations)
                x=context.tensors()|{'planner_scene':planner.scene,'planner_ego':planner.ego,'planner_mask':planner.valid_tokens}
            finally:
                for h in handles:h.remove()
        branch_calls={'Base_decoder':0,'expert_decoders':[0]*count,'score_context_projections':0,'spatial_reader':0}
        def branch(name,index=None):
            def callback(*_):
                if index is None:branch_calls[name]+=1
                else:branch_calls[name][index]+=1
            return callback
        handles=[self.adapter.framework.action_model.expert.register_forward_hook(branch('Base_decoder'))]
        handles += [m.expert.register_forward_hook(branch('expert_decoders',i)) for i,m in enumerate(self.experts[:count])]
        try:
            candidates=[self.adapter.predict_s0(planner).physical]
            candidates.extend(m(x).physical for m in self.experts[:count]);trajectories=torch.stack(candidates,1)
        finally:
            for handle in handles:handle.remove()
        require(torch.isfinite(trajectories).all().item(),'invalid deployed candidate trajectory')
        predictions=None
        if score and count:
            context={k:x[k] for k in CONTEXT_KEYS}
            if self.scorer.spatial:context.update({k:x[k] for k in ('visual','visual_valid','visual_xy')})
            handles=[self.scorer.world_projection.register_forward_hook(branch('score_context_projections'))]
            if self.scorer.spatial:handles.append(self.scorer.spatial_reader.register_forward_hook(branch('spatial_reader')))
            try:predictions=self.scorer(context,trajectories)['values']
            finally:
                for handle in handles:handle.remove()
            require(torch.isfinite(predictions).all().item(),'nonfinite deployed Scorer output')
            from .metrics import stable_order
            _,order=stable_order(count+1,self.expert_ids[:count+1])
            ordered=torch.tensor(order,device=predictions.device)
            chosen=ordered[predictions[:,ordered].argmax(1)]
        else:chosen=torch.zeros(len(observations),device=trajectories.device,dtype=torch.long)
        return {'trajectory':trajectories[torch.arange(len(observations),device=trajectories.device),chosen],
                'candidates':trajectories,'predicted_scores':predictions,'selected':chosen,'calls':calls,'branch_calls':branch_calls}


def load_research_bundle(path,device):
    from iqe.s0_adapter import S0Adapter
    from .models import Specialist,StudyScorer
    from .scorer_training import CELLS
    path=Path(path);receipt=read_json(path.parent/'BUNDLE.json')
    require(file_hash(path)==receipt['checksum'],'research bundle checksum mismatch')
    payload=torch.load(path,map_location='cpu',weights_only=False)
    require(payload['kind']=='iqe_specialist_study_research_bundle_v1' and payload['GT_required']==False and payload['metric_cache_required']==False,'invalid research bundle')
    require(file_hash(payload['Base_binding']['checkpoint'])==payload['Base_binding']['checkpoint_hash'],'Base external weight dependency changed')
    require(payload['pool']['hash']==receipt['pool_hash'] and payload['pool']['hash']==digest({k:v for k,v in payload['pool'].items() if k!='hash'}),'bundle pool mismatch')
    require(not payload['initialization']['external_scorer_weights_loaded'] and not payload['initialization']['loaded_scorer_weights'],
            'bundle has an external Scorer initialization')
    require(digest(payload['contract'])==payload['Base_binding']['contract_hash'],'bundle Base contract changed')
    adapter=S0Adapter.load_and_validate_s0(payload['contract'],device)
    experts=[]
    for spec,state in zip(payload['pool']['experts'],payload['expert_states']):
        m=Specialist(adapter.framework.action_model,private_reader=spec['private_reader'],contract_hash=adapter.contract_hash,
              planner_mode=payload['contract']['source_config']['foresight']['planner_condition_mode']);m.load_state_dict(state,strict=True);experts.append(m.to(device).eval())
    state=payload['scorer'];wd=state['world_projection.1.weight'].shape[1];ad=state['action_projection.1.weight'].shape[1]
    spatial=CELLS[payload['scorer_cell']]['spatial'];vd=state['spatial_reader.projection.1.weight'].shape[1] if spatial else None
    scorer=StudyScorer(wd,ad,adapter.framework.action_model.trajectory_contract,spatial=spatial,visual_dim=vd,views=len(payload['contract']['input']['views']))
    scorer.load_state_dict(state,strict=True)
    return ResearchSystem(adapter,experts,scorer.to(device),payload['pool']['expert_ids']).to(device).eval().requires_grad_(False)


def benchmark(config,device='cuda:0'):
    from iqe.s0_adapter import S0Adapter
    from .qualifications import scorer_resume,online_cached_system
    scorer_resume(config,device)
    chosen=read_json(config.root/'SCORER_MATRIX_SELECTION.json')['selected']['cell'];scorer=selected_scorer(config,chosen,device)
    p=pool(config);adapter=S0Adapter.load_and_validate_s0(read_json(config.root/'OFFLINE_BASE_CONTRACT.json'),device)
    experts=[load_endpoint(config,item,device) for item in p['experts']];system=ResearchSystem(adapter,experts,scorer,p['expert_ids']).eval().requires_grad_(False)
    online_cached_system(config,system,device)
    scenes=scenes_by_role(config)['stage_val'][:32];rows=[]
    sync=lambda:torch.cuda.synchronize() if str(device).startswith('cuda') else None
    scorer_parameters={}
    modes=[(n,None) for n in range(len(experts)+1)]
    modes += [(n,cell) for cell in ('S00','S10') for n in range(1,len(experts)+1)]
    loaded_cell=chosen
    for n,cell in modes:
            score=cell is not None
            if score and cell!=loaded_cell:
                scorer=selected_scorer(config,cell,device);system.scorer=scorer;loaded_cell=cell
            if score:scorer_parameters[cell]=sum(parameter.numel() for parameter in scorer.parameters())
            elapsed=[];scorer_times=[];calls=[];start_event=[None];handles=[]
            def before(*_):sync();start_event[0]=time.perf_counter()
            def after(*_):sync();scorer_times.append(time.perf_counter()-start_event[0])
            handles=[scorer.register_forward_pre_hook(before),scorer.register_forward_hook(after)]
            if str(device).startswith('cuda'):torch.cuda.reset_peak_memory_stats()
            for i in range(35):
                sync();started=time.perf_counter();obs=[load_observation(scenes[i%len(scenes)],adapter.contract)]
                result=system.shared_forward(obs,expert_count=n,score=score);sync()
                if i>=3:elapsed.append(time.perf_counter()-started);calls.append(result['calls'])
                require(result['branch_calls']=={'Base_decoder':1,'expert_decoders':[1]*n,
                    'score_context_projections':int(score),'spatial_reader':int(score and scorer.spatial)},'independent branch/projection calls changed')
            for h in handles:h.remove()
            require(all(x=={'visual':1,'language':1} for x in calls),'online shared calls')
            rows.append({'K':n+1,'scorer':score,'scorer_cell':cell,'scoring_input':'spatial' if score and scorer.spatial else ('semantic' if score else None),
                'mean_seconds':float(np.mean(elapsed)),'p50_seconds':float(np.median(elapsed)),
                'p95_seconds':float(np.quantile(elapsed,.95)),'measurements':len(elapsed),'warmup':3,'batch':1,
                'scorer_p50_seconds':float(np.median(scorer_times[3:])) if scorer_times else 0.,
                'GPU_peak_bytes':torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else None,
                'visual_calls':1,'language_calls':1,'score_context_projection_calls':int(score),'independent_Decoder_calls':n+1,
                'device':str(device),'online_image_decode_included':True,'timing_hooks_enabled_in_all_conditions':True})
    bundle=export_bundle(config,chosen)
    # All private weights restored strictly; independent process command is in RUNBOOK.
    result={'rows':rows,'hardware':torch.cuda.get_device_name() if str(device).startswith('cuda') else 'CPU',
        'Base_parameters':sum(p.numel() for p in adapter.parameters()),'expert_parameters':[sum(p.numel() for p in m.parameters()) for m in experts],
        'scorer_parameters':scorer_parameters,'bundle':str(bundle),'runtime':'locked Base original inference precision'}
    atomic_json(config.root/'LATENCY_AND_PARAMETER_REPORT.json',result,immutable=True)
    # Check an independently loaded process against the current in-memory model.
    system.scorer=selected_scorer(config,chosen,device)
    observation=load_observation(scenes[0],adapter.contract)
    reference=system.shared_forward([observation])
    refpath=config.root/'research_bundle/ROUNDTRIP_REFERENCE.pt'
    atomic_torch(refpath,{k:reference[k].detach().cpu() for k in ('trajectory','candidates','predicted_scores','selected')},immutable=True)
    atomic_json(config.root/'research_bundle/ROUNDTRIP_INPUT.json',{'scene_id':scenes[0].scene_id,
        'observation_hash':scenes[0].observation_hash,'reference_hash':file_hash(refpath),
        'physical_tolerance':1e-4,'score_tolerance':1e-5},immutable=True)
    import os,subprocess,sys
    process=subprocess.run([sys.executable,'-m','iqe.specialist_study.qualifications','bundle-roundtrip','--config',
        str(config.root/'resolved_config.yaml') if (config.root/'resolved_config.yaml').exists() else str(_portable_config(config)),
        '--device',str(device)],cwd=Path(__file__).resolve().parents[2],env=dict(os.environ),capture_output=True,text=True)
    (config.root/'research_bundle/ROUNDTRIP_PROCESS.log').write_text(process.stdout+process.stderr)
    require(process.returncode==0,'independent process bundle restore failed; inspect ROUNDTRIP_PROCESS.log')
    from .reporting import generate
    generate(config)
    return result


def _portable_config(config):
    from dataclasses import asdict
    import yaml
    from iqe.io import atomic_bytes
    path=config.root/'resolved_config.yaml'
    atomic_bytes(path,yaml.safe_dump(asdict(config),sort_keys=True).encode(),immutable=True)
    return path
