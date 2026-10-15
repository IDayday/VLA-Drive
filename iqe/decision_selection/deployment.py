"""One frozen VLA execution, with true pre-routing or actual K4 set selection."""
from contextlib import contextmanager
import torch
from torch import nn
from iqe.contracts import require
from .models import CONTEXT_KEYS, choose


class DecisionSystem(nn.Module):
    def __init__(self,adapter,experts,head,kind,expert_ids):
        super().__init__()
        require(kind in ('M_scene','M_set'),'deployment excludes privileged diagnostics')
        self.adapter=adapter;self.experts=nn.ModuleDict(experts);self.head=head;self.kind=kind
        self.expert_ids=tuple(expert_ids)
        require(self.expert_ids[0]=='expert_0' and set(self.expert_ids[1:])==set(experts),'canonical frozen registry')
        if kind=='M_scene':require(tuple(head.expert_ids)==self.expert_ids,'router logit registry mismatch')
        self.requires_grad_(False).eval()

    def train(self,mode=True):
        # The exported research system is inference-only, including frozen buffers/dropout.
        super().train(False)
        return self

    @contextmanager
    def count_calls(self):
        calls=dict(visual=0,language=0,context_projections=0,
                   reader={k:0 for k in self.expert_ids},decoder={k:0 for k in self.expert_ids},
                   decoder_samples={k:0 for k in self.expert_ids})
        handles=[]
        def hit(key):
            def callback(*_):calls[key]+=1
            return callback
        def branch(key,eid):
            def callback(module,args,output):
                calls[key][eid]+=1
                if key=='decoder':calls['decoder_samples'][eid]+=len(args[0].scene)
            return callback
        network=self.adapter.framework.qwen_vl_interface.model.model
        handles.extend([network.visual.register_forward_hook(hit('visual')),
                        network.language_model.register_forward_hook(hit('language')),
                        self.head.context.register_forward_hook(hit('context_projections'))])
        bridge=self.adapter.framework.action_model
        handles.extend([bridge.q_former.register_forward_hook(branch('reader','expert_0')),
                        bridge.expert.register_forward_hook(branch('decoder','expert_0'))])
        for eid,m in self.experts.items():
            handles.append(m.expert.register_forward_hook(branch('decoder',eid)))
            if m.private_reader:handles.append(m.q_former.register_forward_hook(branch('reader',eid)))
        try:yield calls
        finally:
            for h in handles:h.remove()

    def _encode(self,observations):
        allowed=('image','lang','state','token')
        # Targets, score labels and metric contexts never cross this boundary.
        current=[{k:r[k] for k in allowed} for r in observations]
        encoded=self.adapter.framework.encode_current(current)
        w=encoded['W'].detach();a=encoded['action_queries'].detach()
        ego=self.adapter.framework.action_model.ego_state.detach().reshape(len(a),4)
        context=dict(world=w,action=a,current_ego=ego,
                     world_valid=torch.ones(w.shape[:2],device=w.device,dtype=torch.bool),
                     action_valid=torch.ones(a.shape[:2],device=a.device,dtype=torch.bool))
        # Building the input tensor is not the trainable planner reader. P0/Pj run only in _decode.
        condition=self.adapter.framework.build_planner_condition(encoded).detach()
        return context,condition,[str(r['token']) for r in current]

    def _decode(self,index,rows,context,condition,ids):
        eid=self.expert_ids[index];bridge=self.adapter.framework.action_model
        subset={k:v[rows] for k,v in context.items()}
        if index==0 or not self.experts[eid].private_reader:
            f=bridge.encode_features(condition[rows],subset['current_ego'],[ids[i] for i in rows.tolist()],self.adapter.contract_hash)
            if index==0:return self.adapter.predict_s0(f)
            subset.update(planner_scene=f.scene,planner_ego=f.ego,planner_mask=f.valid_tokens)
        return self.experts[eid](subset)

    @torch.no_grad()
    def forward(self,observations,*,gamma=None,always_base=False,base_only=False,return_context=False):
        require(observations,'nonempty real observation batch')
        with self.count_calls() as calls:
            context,condition,ids=self._encode(observations)
            device=condition.device;n=len(ids);all_rows=torch.arange(n,device=device)
            trajectories=None;raw_candidates=None;logits=None
            if base_only:
                out=self._decode(0,all_rows,context,condition,ids)
                primary=torch.zeros(n,device=device,dtype=torch.long);selected=primary.clone()
                trajectory,raw=out.physical,out.raw;fallback=torch.zeros(n,device=device,dtype=torch.bool)
            elif self.kind=='M_scene':
                # All four IDs are available before decoding; no offline true/finite mask is visible.
                output=self.head(context);logits=output['decision_logits']
                primary=(torch.zeros(n,device=device,dtype=torch.long) if always_base
                         else choose(logits,self.expert_ids,gamma=gamma))
                selected=primary.clone();fallback=torch.zeros(n,device=device,dtype=torch.bool)
                contract=self.adapter.framework.action_model.trajectory_contract
                trajectory=torch.empty(n,contract.horizon,3,device=device,dtype=torch.float32)
                raw=torch.empty(n,contract.horizon,contract.raw_dim,device=device,dtype=torch.float32)
                for j in range(len(self.expert_ids)):
                    rows=torch.nonzero(primary==j,as_tuple=False).flatten()
                    if not len(rows):continue
                    out=self._decode(j,rows,context,condition,ids)
                    good=torch.isfinite(out.physical).all((-1,-2))&torch.isfinite(out.raw).all((-1,-2))
                    require(j!=0 or bool(good.all()),'Base produced invalid trajectory')
                    trajectory[rows]=out.physical;raw[rows]=out.raw
                    bad_rows=rows[~good]
                    if len(bad_rows):
                        rescued=self._decode(0,bad_rows,context,condition,ids)
                        require(torch.isfinite(rescued.physical).all() and torch.isfinite(rescued.raw).all(),'invalid Base fallback')
                        trajectory[bad_rows]=rescued.physical;raw[bad_rows]=rescued.raw
                        selected[bad_rows]=0;fallback[bad_rows]=True
            else:
                candidates=[self._decode(j,all_rows,context,condition,ids) for j in range(len(self.expert_ids))]
                trajectories=torch.stack([x.physical for x in candidates],1)
                raw_candidates=torch.stack([x.raw for x in candidates],1)
                available=torch.isfinite(trajectories).all((-1,-2))&torch.isfinite(raw_candidates).all((-1,-2))
                base_role=torch.zeros_like(available);base_role[:,0]=True
                output=self.head(context,trajectories,base_role,available);logits=output['decision_logits']
                primary=(torch.zeros(n,device=device,dtype=torch.long) if always_base
                         else choose(logits,self.expert_ids,available,gamma))
                require(available[all_rows,primary].all(),'selected unavailable candidate')
                selected=primary.clone();trajectory=trajectories[all_rows,selected];raw=raw_candidates[all_rows,selected]
                fallback=torch.zeros(n,device=device,dtype=torch.bool)
            require(torch.isfinite(trajectory).all() and torch.isfinite(raw).all(),'nonfinite final trajectory')
        result=dict(trajectory=trajectory,raw=raw,selected=selected,primary_selected=primary,
                    selected_candidate_ids=[self.expert_ids[i] for i in selected.tolist()],
                    decision_logits=logits,candidates=trajectories,raw_candidates=raw_candidates,
                    invalid_fallback=fallback,calls=calls,selection_rule='raw_argmax' if gamma is None else 'logit_margin')
        if return_context:result['context']=context
        return result
