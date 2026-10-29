"""Whole-update deterministic sampling, independent of microbatch layout."""
from pathlib import Path
import hashlib
import torch
import torch.distributed as dist
from starVLA.model.modules.qwen_sourceflow.action_codec import ActionCodec
from starVLA.model.modules.qwen_world_postcot.readouts import gather_future_targets
from starVLA.model.modules.qwen_world_postcot.post_action import OFFICIAL_METRIC_NAMES
from starVLA.rl.qwen_world_postcot.batching import to_device

def scene_generator(update,position,purpose):
    digest=hashlib.sha256(f'42:{update}:{position}:{purpose}'.encode()).digest()
    return torch.Generator().manual_seed(int.from_bytes(digest[:8],'little')%(2**63-1))

def post_positions(update):
    # One of the four logical scenes on each rank; exactly eight per global32,
    # even when a rank uses micro1/accum4 instead of micro4.
    return tuple(4*rank+int(torch.randint(4,(),generator=scene_generator(update,rank,'post_slot')))for rank in range(8))

def sample_valid_future_queries(valid,count=128,*,generator):
    """No-replacement valid WA patches, with explicit invalid padding.

    One independent CPU RNG per scene preserves exact global-update draws when
    microbatch/accumulation layouts differ. Active camera/tubelet buckets are
    balanced until a bucket runs out; remaining budget goes to available ones.
    A padding coordinate can happen to point to a valid patch, so its separate
    query_valid bit must always be intersected with the gathered teacher mask.
    """
    if valid.dtype!=torch.bool or valid.ndim!=5 or tuple(valid.shape[1:])!=(3,4,8,16):
        raise ValueError('WA query validity must be bool[B,3,4,8,16]')
    if valid.device.type!='cpu' or not 1<=int(count)<=1536:
        raise ValueError('WA queries are sampled on CPU before target transfer')
    generators=[generator] if isinstance(generator,torch.Generator) else list(generator)
    if len(generators)!=len(valid) or any(g.device.type!='cpu'for g in generators):
        raise ValueError('One independent CPU teacher-query generator is required per scene')
    queries=torch.zeros(len(valid),count,4,dtype=torch.long)
    query_valid=torch.zeros(len(valid),count,dtype=torch.bool)
    for scene,(mask,rng)in enumerate(zip(valid,generators)):
        buckets=[part.nonzero().flatten()for part in mask.reshape(12,128)]
        capacity=torch.tensor([len(part)for part in buckets],dtype=torch.long)
        allocation=torch.zeros_like(capacity)
        wanted=min(int(count),int(capacity.sum()));remaining=wanted
        while remaining:
            active=(allocation<capacity).nonzero().flatten()
            share=remaining//len(active)
            if share:
                increments=torch.minimum(capacity[active]-allocation[active],
                                         torch.full_like(active,share))
                allocation[active]+=increments;remaining-=int(increments.sum())
            else:
                order=active[torch.randperm(len(active),generator=rng)]
                allocation[order[:remaining]]+=1;remaining=0
        items=[]
        for bucket,n in enumerate(allocation.tolist()):
            if not n:continue
            patches=buckets[bucket][torch.randperm(len(buckets[bucket]),generator=rng)[:n]]
            items.append(torch.stack((torch.full_like(patches,bucket//4),
                torch.full_like(patches,bucket%4),patches//16,patches%16),-1))
        if wanted:
            selected=torch.cat(items)[torch.randperm(wanted,generator=rng)]
            queries[scene,:wanted]=selected;query_valid[scene,:wanted]=True
    return queries,query_valid

class BatchBuilder:
    def __init__(self,data,processor,teachers,bank,tracks,evidence):
        self.data,self.processor,self.teachers,self.bank,self.tracks,self.evidence=data,processor,teachers,bank,tracks,evidence
        self.calls=0

    def state_dict(self):
        return dict(schema='four_framework_stateless_batches_v1',calls=self.calls,bank_hash=self.bank.bank_hash,
                    randomness='scene_generator(update,canonical_rank_scene_position,purpose)',candidate_refresh=False)

    def load_state_dict(self,state):
        if state['schema']!='four_framework_stateless_batches_v1'or state['bank_hash']!=self.bank.bank_hash or state['candidate_refresh']is not False:raise ValueError('Builder identity changed')
        self.calls=state['calls']

    def build(self,tokens,positions,update):
        observations=[self.data.observation(t)for t in tokens]
        teacher=[self.teachers.get(t,source_record_hash=self.data.records[t]['source_record_hash'])for t in tokens]
        teacher_valid=torch.stack([v['future_valid']for v in teacher])
        queries,query_valid=sample_valid_future_queries(teacher_valid,128,generator=[scene_generator(update,p,'teacher_query')for p in positions])
        future,fv=gather_future_targets(torch.stack([v['future']for v in teacher]),teacher_valid,queries)
        fv=fv&query_valid
        target=ActionCodec.norm_odo(torch.stack([self.data.target(t)for t in tokens]))
        valid=torch.isfinite(target).all(-1)
        for i,t in enumerate(tokens):
            actual=torch.tensor(self.data.records[t]['raw_future_dt_seconds']).cumsum(0)
            valid[i]&=(actual-torch.arange(1,9)*.5).abs()<.06
        objects=dict(boxes=torch.zeros(len(tokens),32,8),classes=torch.zeros(len(tokens),32,dtype=torch.long),valid=torch.zeros(len(tokens),32,dtype=torch.bool),coverage_valid=torch.zeros(len(tokens),dtype=torch.bool),future_states=torch.zeros(len(tokens),32,4,6),future_valid=torch.zeros(len(tokens),32,4,dtype=torch.bool))
        for i,t in enumerate(tokens):
            tr=self.tracks.get(t);obj=tr['current_objects'];n=tr['current_count']
            if n:
                objects['boxes'][i,:n]=torch.as_tensor(obj['boxes']);objects['classes'][i,:n]=torch.as_tensor(obj['classes']);objects['valid'][i,:n]=True
            objects['coverage_valid'][i]=tr['coverage_valid']
            objects['future_states'][i]=torch.tensor(tr['future_states']);objects['future_valid'][i]=torch.tensor(tr['future_valid'])
        selected=[i for i,p in enumerate(positions)if p in post_positions(update)]
        candidates=[];labels=[];masks=[];scores=[];score_masks=[];events=[];event_masks=[];record_ids=[]
        for i in selected:
            token=tokens[i];pair=self.bank.sample_pairs([token],generator=scene_generator(update,positions[i],'candidate_pair'))
            rows=self.bank.records(token)
            physical=self.data.target(token).float()
            factual=next((r for r in rows if torch.equal(torch.tensor(r.trajectory_fp32),physical)),None)
            records=[None if factual is None else self.evidence.get(factual.record_id),*[self.evidence.get(x)for x in pair.record_ids[0]]]
            candidates.append(torch.cat((target[i:i+1],ActionCodec.norm_odo(pair.trajectory[0])),0))
            lab=[];mask=[];sc=[];sm=[];ev=[];em=[]
            for record in records:
                if record is None:
                    lab.append([0.]*6);mask.append([False]*6);sc.append(0.);sm.append(False);ev.append([8]*3);em.append([False]*3)
                else:
                    lab.append([record['factor_labels'][k]for k in OFFICIAL_METRIC_NAMES]);mask.append([record['factor_valid'][k]for k in OFFICIAL_METRIC_NAMES]);sc.append(record['canonical_total_score']);sm.append(record['total_valid']);ev.append(record['event_class']);em.append(record['event_valid'])
            labels.append(lab);masks.append(mask);scores.append(sc);score_masks.append(sm);events.append(ev);event_masks.append(em);record_ids.append([None if factual is None else factual.record_id,*pair.record_ids[0]])
        ns=len(selected);self.calls+=1
        uniform=torch.tensor([float(torch.rand((),generator=scene_generator(update,p,'FM_sigma')))for p in positions])
        return dict(tokens=list(tokens),positions=list(positions),current_inputs=self.processor(observations),ego=torch.tensor([o['proprio']for o in observations]),target=target,action_valid=valid,
            noise=torch.stack([torch.randn(8,3,generator=scene_generator(update,p,'FM_noise'))for p in positions]),
            sigma=5*uniform/(1+4*uniform),
            dino_target=torch.stack([v['dino']for v in teacher]),dino_valid=torch.stack([v['dino_valid']for v in teacher]),future_queries=queries,future_target=future,future_valid=fv,objects=objects,
            aux_scene_indices=torch.tensor(selected,dtype=torch.long),post_trajectories=torch.stack(candidates)if ns else torch.empty(0,3,8,3),
            post_labels=torch.tensor(labels).reshape(ns,3,6),post_valid=torch.tensor(masks,dtype=torch.bool).reshape(ns,3,6),post_score=torch.tensor(scores).reshape(ns,3),post_score_valid=torch.tensor(score_masks,dtype=torch.bool).reshape(ns,3),post_events=torch.tensor(events,dtype=torch.long).reshape(ns,3,3),post_event_valid=torch.tensor(event_masks,dtype=torch.bool).reshape(ns,3,3),post_record_ids=record_ids)

def update_denominators(batches,device):
    keys=('action_elements','dino','wa_pre','object_existence','object_present','future_track','wa_post','factor','score','rank','event')
    counts=torch.zeros(len(keys),device=device,dtype=torch.float64)
    for b in batches:
        selected=b['aux_scene_indices'];obj=b['objects'];scores=b['post_score'];sv=b['post_score_valid'];pair=(scores[:,:,None]-scores[:,None,:]).abs()>=.02
        pair&=sv[:,:,None]&sv[:,None,:];pair&=torch.triu(torch.ones(3,3,dtype=torch.bool),1)[None]
        present=obj['valid'].any(1)
        factual_count=((b['future_valid'][selected].flatten(1).any(1)&b['action_valid'][selected].all(1)).sum()if len(selected)else 0)
        values=[b['action_valid'].sum()*3,b['dino_valid'].flatten(1).any(1).sum(),b['future_valid'].flatten(1).any(1).sum(),(obj['coverage_valid']|present).sum(),present.sum(),obj['future_valid'].flatten(1).any(1).sum(),factual_count,b['post_valid'].flatten(1).any(1).sum(),sv.any(1).sum(),pair.flatten(1).any(1).sum(),b['post_event_valid'].flatten(1).any(1).sum()]
        counts+=torch.tensor([float(v)for v in values],device=device)
    if dist.is_initialized():dist.all_reduce(counts)
    return dict(zip(keys,counts))
