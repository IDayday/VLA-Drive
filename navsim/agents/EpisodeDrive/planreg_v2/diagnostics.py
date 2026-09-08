"""Low-frequency read-only diagnostics; never an auxiliary optimization objective."""
import math
import torch
import torch.nn.functional as F


class AttentionReadoutAudit:
    def __init__(self,agent):
        self.enabled=False
        self.inputs={}
        self.handles=[]
        for label,decoder in (('generator',agent.action_head.attention),('scorer',agent.action_head.scorer_attention)):
            def capture(module,args,kwargs,_label=label):
                if self.enabled:
                    self.inputs[_label]=(module,kwargs['query'].detach(),kwargs['key'].detach(),kwargs.get('key_padding_mask'))
            self.handles.append(decoder.layers[-1].cross_attn.attn.register_forward_pre_hook(capture,with_kwargs=True))

    @torch.no_grad()
    def collect(self):
        results={}
        for name,(module,q,k,padding) in self.inputs.items():
            d=module.embed_dim;h=module.num_heads
            q=F.linear(q,module.in_proj_weight[:d],module.in_proj_bias[:d])
            k=F.linear(k,module.in_proj_weight[d:2*d],module.in_proj_bias[d:2*d])
            q=q.reshape(len(q),-1,h,d//h).transpose(1,2)
            k=k.reshape(len(k),-1,h,d//h).transpose(1,2)
            scores=q@k.transpose(-2,-1)/math.sqrt(d//h)
            if padding is not None:scores=scores.masked_fill(padding[:,None,None],float('-inf'))
            weights=scores.softmax(-1).mean((1,2))
            results[name+'_token_mass']=weights.cpu().tolist()
            results[name+'_attention_entropy']=float(-(weights*weights.clamp_min(1e-20).log()).sum(-1).mean())
            if weights.shape[-1]%16==0:
                results[name+'_tile_mass']=weights.reshape(len(weights),-1,16).sum(-1).cpu().tolist()
        self.inputs.clear()
        return results


@torch.no_grad()
def representation_summary(tokens,valid=None):
    tokens=tokens.detach().float()
    valid=torch.ones(tokens.shape[:2],device=tokens.device,dtype=torch.bool) if valid is None else valid
    centered=tokens-(tokens*valid[...,None]).sum(1,keepdim=True)/valid.sum(1).clamp_min(1)[:,None,None]
    singular=torch.linalg.svdvals(centered*valid[...,None])
    probability=singular/singular.sum(-1,keepdim=True).clamp_min(1e-12)
    report=dict(slot_centered_rms=float(((centered.square()*valid[...,None]).sum()/
        (valid.sum().clamp_min(1)*tokens.shape[-1])).sqrt()),effective_rank=float((-(probability*probability.clamp_min(1e-12).log()).sum(-1)).exp().mean()))
    if len(tokens)>1:
        # Remove each slot's across-scene mean, not just a fixed query identity.
        scene_mean=(tokens*valid[...,None]).sum(0)/valid.sum(0).clamp_min(1)[:,None]
        report['cross_scene_content_rms']=float((((tokens-scene_mean).square()*valid[...,None]).sum()/
            (valid.sum().clamp_min(1)*tokens.shape[-1])).sqrt())
    else:report['cross_scene_content_rms']='NOT_AVAILABLE_BATCH1'
    return report


@torch.no_grad()
def score_summary(prediction,scores):
    truth=scores[...,-1]
    selected=truth.gather(1,prediction['selected_indices'][:,None]).squeeze(1)
    oracle=truth.max(-1).values
    diagnostic=prediction['log_pdm_score'].exp()/12.
    result=dict(candidate_mean=float(truth.mean()),candidate_p10=float(torch.quantile(truth,.1,dim=1).mean()),
        candidate_p25=float(torch.quantile(truth,.25,dim=1).mean()),selected_pdms=float(selected.mean()),
        oracle64=float(oracle.mean()),regret=float((oracle-selected).mean()),
        catastrophic_fraction=float(((oracle>.9)&(selected<.5)).float().mean()),
        pdms_calibration_mse=float((diagnostic-truth).square().mean()))
    order=prediction['log_pdm_score'].argsort(-1,descending=True)
    for count in (2,4,8):
        found=truth.gather(1,order[:,:count]).max(-1).values
        result['oracle_recall_at_'+str(count)]=float((found>=oracle-1e-6).float().mean())
    from scipy.stats import spearmanr
    for label,index in (('ego_progress',2),('time_to_collision_within_bound',3)):
        pred=prediction['pred_logit'][label].sigmoid();target=scores[...,index]
        valid=target!=2. if index==3 else torch.ones_like(target,dtype=torch.bool)
        result[label+'_mse']=float(((pred-target).square()*valid).sum()/valid.sum().clamp_min(1))
        correlations=[]
        for p,t,m in zip(pred,target,valid):
            if m.sum()>1 and t[m].std()>0 and p[m].std()>0:
                correlations.append(float(spearmanr(p[m].cpu().numpy(),t[m].cpu().numpy()).statistic))
        result[label+'_scene_spearman']=sum(correlations)/len(correlations) if correlations else 'UNDEFINED_CONSTANT'
    return result
