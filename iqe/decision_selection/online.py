"""Real image/cache equivalence and fresh-process deployment checks."""
from pathlib import Path
import argparse
import json
import torch
from iqe.contracts import FeatureBundle,require
from iqe.io import read_json,atomic_json,atomic_torch,file_hash,digest
from iqe.data.sources import load_observation
from iqe.evaluation.retention import module_hash
from iqe.specialist_study.binding import scenes_by_role
from iqe.specialist_study.context_batch_layout import OriginalBatchReplay
from iqe.specialist_study.data import ReusedContext
from iqe.specialist_study.training import runtime
from iqe.specialist_study.bank import pool
from iqe.specialist_study.experiments import load_endpoint
from .config import load as load_config
from .training import create,resource,KINDS
from .data import read_split
from .deployment import DecisionSystem
from .models import CONTEXT_KEYS,choose

TOLERANCES=dict(context=1e-5,raw=1e-5,physical=1e-4,logit=1e-5)


def systems(config,device,initial=False):
    from iqe.s0_adapter import S0Adapter
    from .bundle import load,export
    source=config.source();adapter=S0Adapter.load_and_validate_s0(read_json(source.root/'OFFLINE_BASE_CONTRACT.json'),device)
    if initial:
        p=pool(source);experts={e['expert_id']:load_endpoint(source,e,device) for e in p['experts']}
        return {k:DecisionSystem(adapter,experts,create(config,k,'diagnostic')[0].to(device),k,p['expert_ids']) for k in KINDS}
    return {k:load(export(config,k),device,adapter=adapter) for k in KINDS}


def check_count(kind,result,batch):
    c=result['calls'];require(c['visual']==c['language']==c['context_projections']==1,'must use exactly one shared VLA and one context projection')
    if kind=='M_scene':
        expected={eid:result['primary_selected'].eq(j).sum().item() for j,eid in enumerate(result['calls']['decoder'])}
        expected['expert_0']+=int(result['invalid_fallback'].sum())
        require(c['decoder_samples']==expected,'router computed unselected branch or lost sample order')
    else:require(all(n==1 for n in c['decoder'].values()) and all(n==batch for n in c['decoder_samples'].values()),'set must compute each frozen branch once')


@torch.no_grad()
def online_check(config,device='cuda:0',initial=False):
    resource(config,'real-decision-image-qualification');device=runtime(device)
    folder=config.root/'qualification'/('initial_online' if initial else 'trained_online')
    identity=dict(config=config.identity,initial_diagnostic_only=initial,implementation=file_hash(__file__),
                  deployment=file_hash(Path(__file__).with_name('deployment.py')),tolerances=TOLERANCES)
    if not initial:identity['final_lock']=file_hash(config.root/'FINAL_EVALUATION_LOCK.json')
    if (folder/'REPORT.json').exists():
        r=read_json(folder/'REPORT.json');require(r['identity']==identity,'online qualification identity changed');return r
    allsystems=systems(config,device,initial);adapter=allsystems['M_scene'].adapter
    frozen_before=module_hash(adapter);expert_before={eid:module_hash(m) for eid,m in allsystems['M_scene'].experts.items()}
    split=read_split(config);wanted=set(split['fit']['scene_ids'][:4]);source=config.source()
    lookup={s.scene_id:s for s in scenes_by_role(source)['scorer_fit']}
    replay=OriginalBatchReplay(source);cache=ReusedContext(source,capacity=16)
    groups=list(dict.fromkeys(replay.batch_for(sid) for sid in sorted(wanted)));evidence=[]
    for group in groups:
        rows=[replay.scenes[sid] for sid in group]
        require(all(s.split_role!='final_test' for s in rows),'test input prohibited')
        observations=[load_observation(s,adapter.contract) for s in rows]
        x=cache.batch(rows,device)
        features=FeatureBundle(x['planner_scene'],x['planner_ego'],x['planner_mask'],adapter.contract_hash,tuple(group))
        base=adapter.predict_s0(features)
        cached_context={k:x[k] for k in CONTEXT_KEYS}
        private={k:x[k] for k in ('planner_scene','planner_ego','planner_mask','world','action','current_ego')}
        template=allsystems['M_set']
        outputs=[base]+[template.experts[eid](private) for eid in template.expert_ids[1:]]
        trajectories=torch.stack([r.physical for r in outputs],1);raw=torch.stack([r.raw for r in outputs],1)
        for kind,system in allsystems.items():
            out=system(observations,return_context=True);check_count(kind,out,len(rows))
            errors={}
            for k,v in cached_context.items():
                actual=out['context'][k]
                require(v.shape==actual.shape and v.dtype==actual.dtype,'context schema/dtype changed')
                if v.dtype==torch.bool:
                    require(torch.equal(v,actual),'context mask changed');errors[k]=0.
                else:
                    errors[k]=float((v-actual).abs().max());require(errors[k]<=TOLERANCES['context'],'online/cache context drift: '+k)
            if kind=='M_scene':decision=system.head(cached_context)
            else:
                role=torch.zeros(trajectories.shape[:2],device=device,dtype=torch.bool);role[:,0]=True
                decision=system.head(cached_context,trajectories,role)
                errors['all_candidates']=float((trajectories-out['candidates']).abs().max())
                require(errors['all_candidates']<=TOLERANCES['physical'],'frozen candidate path changed')
            logits=decision['decision_logits'];selected=choose(logits,system.expert_ids)
            require(torch.equal(selected,out['selected']),'online/cache selected ID mismatch')
            idx=torch.arange(len(rows),device=device)
            errors['selected_raw']=float((raw[idx,selected]-out['raw']).abs().max())
            errors['selected_physical']=float((trajectories[idx,selected]-out['trajectory']).abs().max())
            errors['logits']=float((logits-out['decision_logits']).abs().max())
            require(errors['selected_raw']<=TOLERANCES['raw'] and errors['selected_physical']<=TOLERANCES['physical'] and errors['logits']<=TOLERANCES['logit'],'online/cache decision output drift')
            polluted=[o|dict(GT=object(),future_actors=object(),metric_cache=object(),score=float('nan'),target_trajectory=object()) for o in observations]
            changed=system(polluted,return_context=True)
            require(torch.equal(out['raw'],changed['raw']) and torch.equal(out['trajectory'],changed['trajectory'])
                    and torch.equal(out['decision_logits'],changed['decision_logits'])
                    and all(torch.equal(v,changed['context'][k]) for k,v in out['context'].items()),'labels affect deployment inputs')
            base_online=system(observations,base_only=True)
            errors['Base_raw']=float((base.raw-base_online['raw']).abs().max())
            errors['Base_physical']=float((base.physical-base_online['trajectory']).abs().max())
            require(errors['Base_raw']<=TOLERANCES['raw'] and errors['Base_physical']<=TOLERANCES['physical'],'Base output changed')
            evidence.append(dict(kind=kind,original_batch=list(group),requested=[sid for sid in group if sid in wanted],
                observation_hashes=[s.observation_hash for s in rows],errors=errors,calls=out['calls'],label_independence=True,
                cached_head_VLA_calls=0,invalid_fallbacks=int(out['invalid_fallback'].sum())))
        if not initial and not (folder/'ROUNDTRIP_REFERENCE.pt').exists():
            # New process will use this exact original image batch, not hidden features or labels.
            reference={k:{name:v.detach().cpu() for name,v in system(observations).items() if name in ('raw','trajectory','selected','decision_logits')}
                       for k,system in allsystems.items()}
            atomic_torch(folder/'ROUNDTRIP_REFERENCE.pt',reference,immutable=True)
            atomic_json(folder/'ROUNDTRIP_INPUT.json',dict(scene_ids=list(group),observation_hashes=[s.observation_hash for s in rows],
                reference_hash=file_hash(folder/'ROUNDTRIP_REFERENCE.pt'),bundle_hashes={k:file_hash(config.root/'research_bundle'/k/'bundle.pt') for k in KINDS}),immutable=True)
    require(module_hash(adapter)==frozen_before,'Base parameters/buffers changed')
    require(all(module_hash(allsystems['M_scene'].experts[eid])==h for eid,h in expert_before.items()),'frozen expert changed')
    report=dict(status='PASS',identity=identity,real_requested_scenes=len(wanted),evidence=evidence,
        Base_parameters_buffers_unchanged=True,expert_parameters_buffers_unchanged=True,
        actual_original_batch_replay_hash=replay.plan_hash,batch_shape_bitwise_invariance_claimed=False,
        training_labels_accessed_by_model=False,GT_or_metric_cache_inference_reads=False,
        device=device,hardware=torch.cuda.get_device_name(),all_added_heads_random_initialization=initial)
    atomic_json(folder/'REPORT.json',report,immutable=True);return report


@torch.no_grad()
def fresh_process(config,kind,device='cuda:0'):
    from .bundle import load
    resource(config,'fresh-process-decision-bundle');device=runtime(device)
    folder=config.root/'qualification/trained_online';r=read_json(folder/'ROUNDTRIP_INPUT.json')
    require(file_hash(folder/'ROUNDTRIP_REFERENCE.pt')==r['reference_hash'],'roundtrip reference changed')
    path=config.root/'research_bundle'/kind/'bundle.pt'
    require(file_hash(path)==r['bundle_hashes'][kind],'roundtrip bundle changed')
    system=load(path,device);replay=OriginalBatchReplay(config.source());rows=[replay.scenes[sid] for sid in r['scene_ids']]
    require([s.observation_hash for s in rows]==r['observation_hashes'],'roundtrip observation changed')
    actual=system([load_observation(s,system.adapter.contract) for s in rows]);check_count(kind,actual,len(rows))
    expected=torch.load(folder/'ROUNDTRIP_REFERENCE.pt',map_location='cpu',weights_only=True)[kind];errors={}
    for name,tolerance in (('raw',TOLERANCES['raw']),('trajectory',TOLERANCES['physical']),('decision_logits',TOLERANCES['logit'])):
        errors[name]=float((actual[name].cpu()-expected[name]).abs().max());require(errors[name]<=tolerance,'new-process output drift: '+name)
    require(torch.equal(actual['selected'].cpu(),expected['selected']),'new-process selected identity changed')
    report=dict(status='PASS',kind=kind,bundle_hash=r['bundle_hashes'][kind],errors=errors,calls=actual['calls'],real_scenes=len(rows),new_process=True,GT_or_metric_cache_read=False)
    atomic_json(folder/('FRESH_PROCESS_'+kind+'.json'),report,immutable=True);return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--initial-heads',action='store_true');p.add_argument('--fresh-process',choices=KINDS)
    a=p.parse_args();c=load_config(a.config)
    result=fresh_process(c,a.fresh_process,a.device) if a.fresh_process else online_check(c,a.device,a.initial_heads)
    print(json.dumps({k:v for k,v in result.items() if k!='evidence'}))


if __name__=='__main__':main()
