"""Read-only strict-quality and actual clipped-surrogate credit diagnostics."""
import torch

PROTOCOL=dict(schema='world_strict_quality_credit_diagnostics_v1',
    strict_high='NC=DAC=TTC=Comfort=1 and canonical PDMS>=.9',
    direction_checked_high='strict_high and reported DDC=1; extra diagnostic, canonical score unchanged',
    quality_not_claimed='unextracted geometric clearance margins or real-world safety guarantees',
    credit='after cap5/source PPO sign clipping; surrogate log-ratio coefficient, not a parameter-gradient norm',
    effect='metrics only; no sampling, advantage, loss, reference or optimizer change')


class QualityLossMetrics:
    def _policy_loss(self,model,rollout):
        loss,metrics=super()._policy_loss(model,rollout)
        with torch.no_grad():
            records=rollout['records'];g=len(records);device=rollout['advantages'].device
            feasible=[];good=[];direction=[]
            for r in records:
                m=r['metrics'];f=all(m[k]==1 for k in ('no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound'))
                h=f and m['comfort']==1 and m['score']>=.9
                feasible.append(f);good.append(h);direction.append(h and m['driving_direction_compliance']==1)
            f=torch.tensor(feasible,device=device);h=torch.tensor(good,device=device);ddc=torch.tensor(direction,device=device)
            a=rollout['advantages'].detach().float().clamp(-5,5)
            keep=rollout['adv_mask'].detach().float()
            now=model._probability_audit['current']['source_clamped_logprob'].mean((-1,-2))
            rho=(now-rollout['old_logp']).exp();clip=float(self.ppo_clip_range)
            active=~(((a[:,None]>0)&(rho>1+clip))|((a[:,None]<0)&(rho<1-clip)))
            coeff=a[:,None]*rho*active*keep[:,None]/(keep.sum().clamp_min(1)*rho.shape[1])
            positive=coeff.clamp_min(0).sum(1);negative=(-coeff).clamp_min(0).sum(1)
            values=dict(quality_group_size=g,quality_strict_high_count=h.sum(),quality_direction_high_count=ddc.sum(),
                quality_has_strict_high=h.any(),quality_feasible_count=f.sum(),
                quality_positive_high_count=((a>0)&h).sum(),quality_negative_high_count=((a<0)&h).sum(),
                quality_active_positive_high_weight=positive[h].sum(),
                quality_active_negative_high_weight=negative[h].sum(),
                quality_active_positive_unsafe_weight=positive[~f].sum(),
                quality_active_negative_unsafe_weight=negative[~f].sum(),
                quality_high_weight_per_good=positive[h].sum()/h.sum().clamp_min(1),
                quality_high_positive_PPO_blocked=((a[:,None]>0)&h[:,None]&~active).float().sum()/((a>0)&h).sum().clamp_min(1)/rho.shape[1])
            metrics.update({k:torch.as_tensor(v,device=device,dtype=torch.float32)for k,v in values.items()})
        return loss,metrics


class QualityDiagnosticsMixin:
    def make_actor(self,cfg,args,device):
        actor,a,b=super().make_actor(cfg,args,device)
        base=type(actor._loss)
        class LoggedLoss(QualityLossMetrics,base):pass
        actor._loss=LoggedLoss(cfg)
        return actor,a,b

    def recipe_identity(self,cfg,args,condition_sha):
        value=super().recipe_identity(cfg,args,condition_sha)
        value['quality_credit_diagnostics']=PROTOCOL
        return value
