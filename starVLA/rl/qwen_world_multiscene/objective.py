"""Vectorize probability forwards; retain exact original per-scene loss math.

No group is combined for normalization, PPO masking, or IL reference reduction.
Caller-supplied weights are the original scene/padding backward weights, NOT
candidate-mask weights. This prototype covers the base source objective only;
nominal/history auxiliary forwards remain independent B1/MC calculations.
"""
import torch


class _CachedSceneActor:
    def __init__(self,actor,scene,current,reference,audits):
        self.actor,self.scene,self.current,self.reference,self.audits=actor,scene,current,reference,audits
        self._probability_audit={}

    def __getattr__(self,name):return getattr(self.actor,name)

    def action_chain_logprobs(self,*args,use_ref=False,return_terms=False,**kwargs):
        key='reference' if use_ref else 'current'
        self._probability_audit[key]=self.audits[key]
        logp,terms=self.reference if use_ref else self.current
        return (logp,terms) if return_terms else logp


def source_probabilities(actor,rollouts,packed_condition,*,use_ref=False):
    if len(rollouts)!=len(packed_condition['scene_tokens']):raise ValueError('Scene order/shape changed')
    first=rollouts[0];g=packed_condition['groups']
    for token,r in zip(packed_condition['scene_tokens'],rollouts):
        if r['token']!=token or r['chain'].shape[0]!=g:raise ValueError('Scene/candidate order changed')
        for key in ('timesteps','deltas','stoch_idx'):
            if not torch.equal(first[key],r[key]):raise ValueError('Only the same source W3 schedule can be packed')
    chain=torch.cat([r['chain'] for r in rollouts],0)
    result=actor.action_chain_logprobs(packed_condition,chain,first['timesteps'],first['deltas'],stoch_idx=first['stoch_idx'],return_terms=True,use_ref=use_ref)
    audit=actor._probability_audit['reference' if use_ref else 'current']
    return result,audit


def source_loss(actor,rollouts,packed_condition,*,loss,weights=None):
    """Return weighted total, per-scene metrics, and diagnostic raw probabilities."""
    current,ca=source_probabilities(actor,rollouts,packed_condition)
    with torch.no_grad():reference,ra=source_probabilities(actor,rollouts,packed_condition,use_ref=True)
    g=packed_condition['groups'];n=len(rollouts)
    if weights is None:weights=[1/n]*n
    if len(weights)!=n:raise ValueError('One original backward weight per scene required')
    total=current[0].sum()*0;metrics=[]
    for i,(rollout,weight) in enumerate(zip(rollouts,weights)):
        sl=slice(i*g,(i+1)*g)
        def cut(pair):return pair[0][sl],{k:(v if k=='std' else v[sl])for k,v in pair[1].items()}
        def auditcut(a):
            result={k:(v[sl]if torch.is_tensor(v)and v.ndim>1 else v)for k,v in a.items()}
            result['source_clamp_hit_rate']=float((result['raw_gaussian_logprob']!=result['source_clamped_logprob']).float().mean())
            return result
        proxy=_CachedSceneActor(actor,i,cut(current),cut(reference),dict(current=auditcut(ca),reference=auditcut(ra)))
        value,metric=loss._policy_loss(proxy,rollout)
        total=total+float(weight)*value;metrics.append(metric)
    return total,metrics,dict(current=current,reference=reference,current_audit=ca,reference_audit=ra)
