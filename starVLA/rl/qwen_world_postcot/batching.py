"""Ordered live observations plus detached teacher/candidate targets."""
import torch
import torch.distributed as dist
from pathlib import Path
from torch.profiler import record_function
from starVLA.model.modules.qwen_sourceflow.action_codec import ActionCodec
from starVLA.model.modules.qwen_world_postcot.readouts import sample_future_queries,gather_future_targets
from starVLA.model.modules.qwen_world_postcot.candidate_bank import choose_aux_scenes,CandidateBank
from .contracts import fingerprint

def to_device(value,device):
    if isinstance(value,torch.Tensor):return value.to(device,non_blocking=True)
    if isinstance(value,dict):return {k:to_device(v,device) for k,v in value.items()}
    return value

class BatchBuilder:
    def __init__(self,data,processor,teachers,bank,seed,*,require_full_bank=True,max_candidates=16):
        self.data,self.processor,self.teachers,self.bank=data,processor,teachers,bank
        self.require_full_bank=require_full_bank;self.bank_binding=None
        self.max_candidates=max_candidates
        if any(not 2<=count<=max_candidates for count in bank.manifest['candidate_counts'].values()):
            raise ValueError('Candidate-bank per-scene count exceeds the configured budget')
        self.score_protocol=bank.manifest['identity']['score_protocol_sha256']
        self.aux_rng=torch.Generator().manual_seed(seed+101)
        self.future_rng=torch.Generator().manual_seed(seed+202)
        self.candidate_rng=torch.Generator().manual_seed(seed+303)
    def state_dict(self):
        return dict(schema='world_postcot_batch_builder_v2',
            bank_root=str(Path(self.bank.manifest_path).parent.resolve()),bank_hash=self.bank.bank_hash,
            bank_state=self.bank.state_dict(),bank_binding=self.bank_binding,
            rng={k:getattr(self,k).get_state() for k in ('aux_rng','future_rng','candidate_rng')})
    def adopt_bank(self,entry,binding):
        if entry['data_manifest_sha256']!=self.data.sha or entry['score_protocol_sha256']!=self.score_protocol:
            raise ValueError('Candidate-bank current-observation/score identity changed')
        target=Path(entry['bank_root']).resolve()
        if target!=Path(self.bank.manifest_path).parent or entry['bank_hash']!=self.bank.bank_hash:
            replacement=CandidateBank(target,allowed_splits=('TRAIN',),expected_identity=dict(
                source_data_manifest_sha256=self.data.sha,source_records_sha256=self.data.manifest['records_sha256'],
                score_protocol_sha256=self.score_protocol))
            if (replacement.bank_hash!=entry['bank_hash']
                    or any(not 2<=count<=self.max_candidates for count in replacement.manifest['candidate_counts'].values())
                    or (self.require_full_bank and set(replacement.manifest['scene_ids'])!=set(self.data.tokens))):
                replacement.close();raise ValueError('Refreshed bank hash/full TRAIN coverage mismatch')
            old=self.bank;replacement.generator.set_state(old.generator.get_state())
            replacement.sampled_scenes=old.sampled_scenes;replacement.sampled_records=old.sampled_records
            self.bank=replacement;old.close()
        self.bank_binding=binding
    def load_state_dict(self,state):
        if state['schema']!='world_postcot_batch_builder_v2':raise ValueError('Unrecognized explicit candidate-bank recovery schema')
        binding=state['bank_binding']
        if binding is not None:
            body=dict(binding);seal=body.pop('sha256',None)
            if fingerprint(body)!=seal:raise ValueError('Changed saved bank boundary binding')
        entry=(binding or {}).get('selected',dict(data_manifest_sha256=self.data.sha,
            score_protocol_sha256=self.score_protocol,bank_root=state['bank_root'],bank_hash=state['bank_hash']))
        if Path(entry['bank_root']).resolve()!=Path(state['bank_root']).resolve() or entry['bank_hash']!=state['bank_hash']:
            raise ValueError('Saved current bank differs from its fixed boundary binding')
        self.adopt_bank(entry,binding);self.bank.load_state_dict(state['bank_state'])
        if set(state['rng'])!={'aux_rng','future_rng','candidate_rng'}:raise ValueError('Incomplete three-stream recovery state')
        for k,v in state['rng'].items():getattr(self,k).set_state(v)
    def build(self,tokens):
        with record_function('world/data_current_image_decode'):
            observations=[self.data.observation(t) for t in tokens]
        with record_function('world/data_teacher_cache_get'):
            targets=[self.teachers.get(t,source_record_hash=self.data.records[t]['source_record_hash']) for t in tokens]
        future=torch.stack([v['future'] for v in targets]);fv=torch.stack([v['future_valid'] for v in targets])
        queries=sample_future_queries(len(tokens),128,generator=self.future_rng)
        future,fv=gather_future_targets(future,fv,queries)
        objects=dict(boxes=torch.zeros(len(tokens),32,8),classes=torch.zeros(len(tokens),32,dtype=torch.long),
                     valid=torch.zeros(len(tokens),32,dtype=torch.bool),coverage_valid=torch.zeros(len(tokens),dtype=torch.bool))
        for i,token in enumerate(tokens):
            obj=self.data.records[token]['objects'];n=len(obj['boxes'])
            if n:
                objects['boxes'][i,:n]=torch.tensor(obj['boxes']);objects['classes'][i,:n]=torch.tensor(obj['classes']);objects['valid'][i,:n]=True
            objects['coverage_valid'][i]=obj['coverage_valid']
        with record_function('world/data_candidate_bank_get'):
            selected=choose_aux_scenes(tokens,fraction=.5,generator=self.aux_rng)
            pair=self.bank.sample_pairs([tokens[i] for i in selected.tolist()],generator=self.candidate_rng)
        target=ActionCodec.norm_odo(torch.stack([self.data.target(t) for t in tokens]))
        action_valid=torch.isfinite(target).all(-1)
        # Raw logs occasionally skip timestamps. Do not relabel a pose at4.5s
        # as the fixed4.0s action point. Other scene supervision remains valid.
        for i,t in enumerate(tokens):
            actual=torch.tensor(self.data.records[t]['raw_future_dt_seconds']).cumsum(0)
            action_valid[i]&=(actual-torch.arange(1,9)*.5).abs()<.06
        with record_function('world/data_processor_tokenizer'):
            current_inputs=self.processor(observations)
        return dict(current_inputs=current_inputs,ego=torch.tensor([o['proprio'] for o in observations]),
            target=target,action_valid=action_valid,dino_target=torch.stack([v['dino'] for v in targets]),
            dino_valid=torch.stack([v['dino_valid'] for v in targets]),future_queries=queries,future_target=future,future_valid=fv,
            objects=objects,aux_scene_indices=selected,candidate_trajectory=pair.trajectory,
            candidate_labels=pair.labels,candidate_valid=pair.valid,candidate_pair_mask=pair.pair_mask)

def update_denominators(batches,device):
    counts=torch.zeros(11,device=device,dtype=torch.float64)
    for b in batches:
        labels,valid=b['candidate_labels'][:,:,:2],b['candidate_valid'][:,:,:2]
        pair=valid.all(1)&((labels==0)|(labels==1)).all(1)&(labels[:,0]!=labels[:,1])&b['candidate_pair_mask']
        obj=b['objects'];present=obj['valid'].any(1)
        values=[b['action_valid'].sum()*3,b['dino_valid'].flatten(1).any(1).sum(),b['future_valid'].flatten(1).any(1).sum(),
            (obj['coverage_valid']|present).sum(),present.sum(),present.sum(),b['candidate_valid'].flatten(1).any(1).sum(),pair.any(1).sum(),
            b['dino_valid'].sum(),b['future_valid'].sum(),b['candidate_valid'].sum()]
        counts+=torch.tensor([float(v) for v in values],device=device)
    if dist.is_initialized():dist.all_reduce(counts)
    return dict(action_elements=counts[0],dino=counts[1],future=counts[2],
        object=dict(existence=counts[3],classification=counts[4],box=counts[5]),post=dict(factor=counts[6],pair=counts[7]),
        audit_counts=dict(dino_valid_patches=counts[8],future_sampled_valid_patches=counts[9],post_valid_metric_labels=counts[10]))
