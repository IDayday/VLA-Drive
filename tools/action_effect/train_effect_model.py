"""Train matched small G1 controls on immutable real intervention pairs."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from pathlib import Path
import json, subprocess
import numpy as np
import torch
from .common import digest,artifact,verify_artifact,verify_identity,read_json,write_json,load_config
from .context_data import ContextReader,batch_context
from .training import run_training,load_private
from starVLA.model.modules.action_effect.effect_model import ActionEffectModel,EffectModelConfig,FeedbackLabelSpec,MODES
from starVLA.model.modules.action_effect.effect_losses import effect_loss_contributions,combine_contributions,reduce_contributions

WEIGHTS={'value':1.,'ranking':.2,'events':.2,'temporal':.2,'uncertainty':.05,'repair':.2}


def load_effect_teacher(checkpoint, *, expected_base_binding_hash, identity_path=None, allowed_modes=("delta_feedback",)):
    p=Path(checkpoint);identity=read_json(identity_path or p.parent/'IDENTITY.json')
    receipt=read_json(p.with_suffix('.COMPLETE.json'))
    if verify_artifact(receipt['checkpoint']).resolve()!=p.resolve():raise ValueError('Wrong effect checkpoint receipt')
    ih=digest(identity)
    if identity.get('base_binding_hash')!=expected_base_binding_hash or receipt['run_identity']!=ih:
        raise ValueError('Effect teacher Base/run identity mismatch')
    if identity.get('arm') not in allowed_modes or any(m not in MODES for m in allowed_modes):
        raise ValueError('Effect guidance requires its explicitly declared own trained teacher mode')
    c=torch.load(p,map_location='cpu',weights_only=False)
    if c['run_identity']!=ih or c['optimizer_step']!=receipt['optimizer_step'] or c['optimizer_step']<1:
        raise ValueError('No compatible real trained effect checkpoint')
    spec=FeedbackLabelSpec(**{k:tuple(v) for k,v in identity['label_spec'].items()})
    model=ActionEffectModel(EffectModelConfig(**identity['model_config']),spec)
    load_private(model,c['model']);model.requires_grad_(False).eval()
    model.provenance={'checkpoint':artifact(p),'identity':artifact(identity_path or p.parent/'IDENTITY.json'),
                      'optimizer_step':c['optimizer_step'],'external_weights_loaded':False,'base_binding_hash':expected_base_binding_hash}
    return model


class PairData:
    def __init__(self,path,reader):
        self.path=Path(path);self.manifest=verify_identity(read_json(path))
        m=self.manifest
        if m.get('schema')!='action_effect_pair_dataset_v1' or m.get('status')!='COMPLETE' or m.get('split')!='train':
            raise ValueError('Complete real TRAIN effect pairs required')
        reg=verify_identity(read_json(verify_artifact(m['registration'])))
        completion=read_json(verify_artifact(m['population_completion']))
        if (reg['identity']!=m['registration_identity'] or completion['registration_identity']!=reg['identity'] or
            reg['binding_identity']!=reader.binding['identity'] or reg['base_checkpoint']!=reader.binding['base_checkpoint']):
            raise ValueError('Effect pair completion/registration/Base identity mismatch')
        # First-access full source chain validation; never rely on a manifest
        # existing while its native feedback or scene receipt has disappeared.
        from .verified_data import validate_compact_dataset
        self.validation_receipt=validate_compact_dataset(path,kind='pair')
        self.reader=reader;self.by_scene={};self.validation=[];self.names=None;groups={'fit':set(),'validation':set()}
        for r in m['entries']:
            verify_identity(r)
            if r['scene_id'] not in reader.rows or r['context_entry']!=reader.rows[r['scene_id']]:
                raise ValueError('Pair/feature observation identity mismatch')
            if (r['role'] not in groups or r['role']!=reader.rows[r['scene_id']]['role'] or
                r['source_log']!=reader.rows[r['scene_id']]['source_log']):
                raise ValueError('Pair data role leakage')
            if r['base_labels']['names']!=r['modified_labels']['names'] or r['base_labels']['units']!=r['modified_labels']['units']:
                raise ValueError('Pair label units/schema changed')
            signature=(r['base_labels']['names'],r['base_labels']['units'])
            if self.names is None:self.names=signature
            if signature!=self.names:raise ValueError('Inconsistent effect label channels')
            groups[r['role']].add(r['source_log'])
            if r['role']=='fit':
                # Joint longitudinal+lateral combinations are held out in G1.
                # G0 still simulates/retains them for supervised evaluation.
                if r['kind']!='joint':self.by_scene.setdefault(r['scene_id'],[]).append(r)
            else:self.validation.append(r)
        if groups['fit']&groups['validation']:raise ValueError('Log leakage across effect fit/validation')
        if not self.by_scene or not self.validation:raise ValueError('Need real fit AND held-out TRAIN logs')
        self.scenes=sorted(self.by_scene);self.validation.sort(key=lambda r:digest(['heldout_pair_order',r['identity']]))
        self.spec=FeedbackLabelSpec.from_label_arrays({'names':self.names[0],'units':self.names[1]})
        self.role_scenes={role:sorted({r['scene_id'] for r in m['entries'] if r['role']==role}) for role in groups}
        self.shuffle_map={}
        for role,scenes in self.role_scenes.items():
            if len(scenes)>1:
                ordered=sorted(scenes,key=lambda s:digest(['effect_shuffle',s]))
                self.shuffle_map.update({x:ordered[(i+1)%len(ordered)] for i,x in enumerate(ordered)})

    def targets(self,rows,device):
        def tensor(k,dtype=torch.float32):return torch.tensor([r[k] for r in rows],dtype=dtype,device=device)
        out={}
        for prefix,field in [('before','base_labels'),('after','modified_labels')]:
            out[prefix]=torch.tensor([r[field]['values'] for r in rows],dtype=torch.float32,device=device)
            out[prefix+'_valid']=torch.tensor([r[field]['valid'] for r in rows],dtype=torch.bool,device=device)
        ev=[]
        for r in rows:
            ev.append([0 if r['event_transition'][k]['repaired'] else 2 if r['event_transition'][k]['introduced'] else 1
                       for k in ('collision','road','ttc')])
        out.update(event_targets=torch.tensor(ev,dtype=torch.long,device=device),
                   event_valid=torch.ones((len(rows),3),dtype=torch.bool,device=device),
                   temporal_targets=tensor('temporal_event_delta'),temporal_valid=tensor('temporal_event_valid',torch.bool),
                   repair_targets=tensor('eligible_repair'),repair_valid=torch.ones(len(rows),dtype=torch.bool,device=device),
                   label_spec=self.spec)
        return out

    def predict(self,model,rows,device):
        if model.config.mode=='shuffled_context':
            if any(r['scene_id'] not in self.shuffle_map for r in rows):raise ValueError('No independent scene for negative control')
            # A fixed whole-role scene derangement, bound below. Identity internal
            # permutation acknowledges that the cache lookup was already shuffled.
            records=[self.reader[self.shuffle_map[r['scene_id']]] for r in rows]
            perm=torch.arange(len(rows),device=device)
        else:records=[self.reader[r['scene_id']] for r in rows];perm=None
        context=batch_context(records,device)
        before=torch.tensor([r['base_physical'] for r in rows],dtype=torch.float32,device=device)
        after=torch.tensor([r['modified_physical'] for r in rows],dtype=torch.float32,device=device)
        return model(context,before,after,context_permutation=perm)


def evaluate(model,data,device,microbatch=8):
    score=data.spec.names.index('score');pred=[];true=[];scenes=[];variances=[];events=[];event_true=[];kinds=[]
    squared=np.zeros(len(data.spec.names));counts=np.zeros_like(squared);time_absolute=[];temporal_error=[]
    for at in range(0,len(data.validation),microbatch):
        rows=data.validation[at:at+microbatch];p=data.predict(model,rows,device);t=data.targets(rows,device)
        d=t['after']-t['before'];valid=t['after_valid']&t['before_valid'];error=(p['mean_delta']-d).detach().cpu().numpy()
        v=valid.cpu().numpy();squared+=(np.where(v,error,0.)**2).sum(0);counts+=v.sum(0)
        pred.extend(p['mean_delta'][:,score].cpu().tolist());true.extend(d[:,score].cpu().tolist())
        scenes.extend(r['scene_id'] for r in rows);kinds.extend(r['kind'] for r in rows);variances.extend(p['logvar_delta'][:,score].exp().cpu().tolist())
        events.extend(p['event_logits'].argmax(-1).cpu().tolist());event_true.extend(t['event_targets'].cpu().tolist())
        tv=t['temporal_valid'];te=(p['temporal_delta']-t['temporal_targets']).abs();temporal_error.extend(te[tv].cpu().tolist())
    pred=np.array(pred);true=np.array(true);valid=abs(true)>.01
    improvements=[];random=[];oracle=[];by={}
    for scene,p,t in zip(scenes,pred,true):by.setdefault(scene,[]).append((p,t))
    for rows in by.values():
        x=np.array(rows);i=int(np.argmax(x[:,0]));improvements.append(float(x[i,1]) if x[i,0]>0 else 0.)
        random.append(float(x[:,1].mean()));oracle.append(float(max(0.,x[:,1].max())))
    return {'heldout_pairs':len(true),'heldout_scenes':len(by),'non_tie_pairs':int(valid.sum()),
        'direction_accuracy':float(np.mean(np.sign(pred[valid])==np.sign(true[valid]))) if valid.any() else None,
        'intervention_subsets':{k:{'pairs':int((np.array(kinds)==k).sum()),
            'non_tie_pairs':int(((np.array(kinds)==k)&valid).sum()),
            'direction_accuracy':float(np.mean(np.sign(pred[(np.array(kinds)==k)&valid])==np.sign(true[(np.array(kinds)==k)&valid])))
                if ((np.array(kinds)==k)&valid).any() else None} for k in sorted(set(kinds))},
        'score_delta_rmse':float(np.sqrt(np.mean((pred-true)**2))),
        'label_rmse':{n:float(np.sqrt(squared[i]/counts[i])) if counts[i] else None for i,n in enumerate(data.spec.names)},
        'label_valid_counts':dict(zip(data.spec.names,counts.astype(int).tolist())),
        'event_transition_accuracy':float(np.mean(np.array(events)==np.array(event_true))),
        'event_class_counts':np.bincount(np.array(event_true).ravel(),minlength=3).tolist(),
        'temporal_event_mae':float(np.mean(temporal_error)) if temporal_error else None,
        'top1_real_delta_with_identity':float(np.mean(improvements)),
        'random_candidate_real_delta':float(np.mean(random)),'oracle_with_identity_delta':float(np.mean(oracle)),
        'uncertainty_1sigma_coverage':float(np.mean(abs(pred-true)<=np.sqrt(np.array(variances))))}


def train(cfg,*,context_manifest,pair_dataset,arm,device,seed,max_updates,stop_after=None,resume=False,output=None):
    reader=ContextReader(context_manifest,Path(cfg['artifact_root'])/'CAMPAIGN_BINDING.json');data=PairData(pair_dataset,reader)
    t=cfg['training'];mc=EffectModelConfig(mode=arm,hidden_dim=t['hidden'],layers=t['layers'],heads=t['heads'],init_seed=seed)
    model=ActionEffectModel(mc,data.spec).to(device);pair_rng=torch.Generator().manual_seed(seed+202)
    identity={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'arm':arm,
        'model_config':asdict(mc),'label_spec':asdict(data.spec),'base_binding_hash':reader.binding['base_checkpoint_binding_hash'],
        'data_artifact':artifact(pair_dataset),'context_artifact':artifact(context_manifest),'loss_weights':WEIGHTS,
        'scene_shuffle_map_hash':digest(data.shuffle_map),'withheld_training_interventions':['joint'],
        'sampling':'uniform fit scene then uniform valid pair'}
    def step_fn(model,indices,step):
        rows=[]
        for i in indices:
            choices=data.by_scene[data.scenes[i]];j=int(torch.randint(len(choices),(1,),generator=pair_rng));rows.append(choices[j])
        contributions=[]
        for at in range(0,len(rows),t['microbatch_scenes']):
            r=rows[at:at+t['microbatch_scenes']];p=data.predict(model,r,device)
            contributions.append(effect_loss_contributions(p,mode=arm,**data.targets(r,device)))
        loss=reduce_contributions(combine_contributions(contributions),WEIGHTS);loss.total.backward()
        return {**{k:float(v.detach()) for k,v in loss.terms.items()},**loss.statistics}
    def validate_fn(model,step):return evaluate(model,data,device,t['microbatch_scenes'])
    return run_training(model,output=output or Path(cfg['artifact_root'])/'g1'/arm/f'seed_{seed}',identity=identity,
        dataset_size=len(data.scenes),step_fn=step_fn,validation_fn=validate_fn,cfg=cfg,device=device,seed=seed,
        max_updates=max_updates,stop_after=stop_after,resume=resume,stateful={'pair_rng':pair_rng})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--context-manifest',required=True);p.add_argument('--pair-dataset',required=True)
    p.add_argument('--arm',choices=MODES,required=True);p.add_argument('--device',required=True);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--max-updates',type=int,required=True);p.add_argument('--stop-after',type=int);p.add_argument('--output')
    p.add_argument('--resume',action='store_true');p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();cfg=load_config(a.config)
    if a.dry_run:print(json.dumps({'status':'NOT_RUN','arm':a.arm,'max_updates':a.max_updates}));return
    print(json.dumps(train(cfg,context_manifest=a.context_manifest,pair_dataset=a.pair_dataset,arm=a.arm,device=a.device,
        seed=a.seed,max_updates=a.max_updates,stop_after=a.stop_after,resume=a.resume,output=a.output)))


if __name__=='__main__':main()
