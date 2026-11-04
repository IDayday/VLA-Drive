"""Same per-candidate objectives using one shared-prefix denoiser call."""
from functools import partial
import torch
from starVLA.rl.d46_fpo_verified.runtime import VerifiedActor,VerifiedExtension
from starVLA.rl.d44_native_rl.fpo import mc_predictions_energy,velocity_energy
from starVLA.rl.d44_native_rl.native_cleanx_aux import positive_loss
from starVLA.rl.d44_native_rl.batch_runtime import collect_scene
from .prefix_cache import repeat_scene,collection_prefix_cache


class JointVerifiedActor(VerifiedActor):
    def flow_energies(self,r,prefix):
        if not getattr(self,'execution_joint_flow_aux',False):
            return mc_predictions_energy(self.action_expert,prefix,r['fm_state'],r['fm_sigma'],r['fm_target'],
                fuse=bool(getattr(self,'execution_fuse_fm_mc',False)))[0]
        mc,g=r['fm_sigma'].shape;n=mc*g
        states=[r['fm_state'].flatten(0,1)];sigmas=[r['fm_sigma'].flatten()]
        rescue='rescue_teacher'in r;last=rescue or'retention_target'in r
        if rescue:states.append(r['rescue_state']);sigmas.append(r['rescue_sigma'])
        if last:states.append(r['nominal_chain'][:,9]);sigmas.append(r['nominal_sigma_last'])
        state=torch.cat(states);sigma=torch.cat(sigmas);packed=repeat_scene(prefix,len(state))
        with torch.autocast(self.device.type,dtype=torch.bfloat16):pred=self.action_expert.denoise_x0(state,sigma,packed)
        energy=velocity_energy(state[:n],pred[:n],r['fm_target'].flatten(0,1),sigma[:n]).reshape(mc,g)
        zero=pred.sum()*0;cfm=ode=keep=zero
        if rescue:
            end=n+len(r['rescue_sigma'])
            cfm=velocity_energy(state[n:end],pred[n:end],r['rescue_velocity_target'],sigma[n:end]).mean()
        if last:
            if rescue:ode=positive_loss(pred[-1:],r['rescue_teacher'])
            if'retention_target'in r:keep=positive_loss(pred[-1:],r['retention_target'])
        self._joint_auxiliary=(cfm,ode,keep)
        return energy

    def additional_objective(self,r,prefix):
        if not getattr(self,'execution_joint_flow_aux',False):return super().additional_objective(r,prefix)
        cfm,ode,keep=self._joint_auxiliary;self._joint_auxiliary=None;c=self.study_config['verified_improvement']
        total=c['CFM_weight']*cfm+c['ODE_last_weight']*ode+c['retention_weight']*r.get('retention_share',0.)*keep
        zero=total.detach()*0;self._auxiliary_parts=dict(CFM=cfm,ODE_last=ode,retention=keep)
        return total,dict(verified_CFM=cfm.detach(),verified_ODE_last=ode.detach(),retention_B1=keep.detach(),
            rescue_active=zero+float('rescue_teacher'in r),retention_active=zero+float('retention_target'in r),
            retention_budget=zero+r.get('retention_share',0.),verified_auxiliary=total.detach())


class OptimizedVerifiedCollection(VerifiedExtension):
    """Collection hook only; no new trainer and no formal-resume authority."""
    actor_factory=JointVerifiedActor
    main_collector=staticmethod(partial(collect_scene,defer_flow_energy=True,validate_sample_mean=False))

    @torch.no_grad()
    def collect_scene(self,actor,*args,**kwargs):
        with collection_prefix_cache(actor):
            result=super().collect_scene(actor,*args,**kwargs)
            with actor._context_call(result['condition'])as material:
                energy=actor.flow_energies(result,actor._prepare(material,False))
            result['old_energy']=energy.mean(0).detach();actor._joint_auxiliary=None
            return result
