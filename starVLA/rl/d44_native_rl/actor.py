"""D-only actor on the audited native sampler and functional prefix lifecycle."""
import copy
from dataclasses import replace
from contextlib import nullcontext
import torch
from torch import nn
from starVLA.rl.qwen_world_compare.policy_adapter import FourFrameworkRecipeActor
from .expanded_scope import install, independent_trainable_module
from .gaussian_objective import gaussian_terms, clipped_loss
from .native_cleanx_aux import positive_loss
from .fpo import velocity_energy, awm_loss, mc_predictions_energy


class D44Actor(FourFrameworkRecipeActor):
    def __init__(self,model,backend,cfg):
        self.study_config=copy.deepcopy(cfg)
        super().__init__(model,backend)
        if cfg['train_action_io']:
            for path in ('decision.action_input','decision.action_output'):
                independent_trainable_module(self.action_expert,path)
            self.action_expert.decision.action_position=nn.Parameter(self.action_expert.decision.action_position.detach().clone())
        if cfg['post_feedback']:
            for path in ('readouts.bridge','readouts.evaluation'):
                independent_trainable_module(self.action_expert,path)
        self.closure=self.scope_audit()
        self.record_sampling_terms=True
        self.execution_fast_prefix=False
        self.execution_fuse_fm_mc=False

    def _configure_action_grpo(self,cfg):
        self.eval().requires_grad_(False)
        self.ref_action_expert=copy.deepcopy(self.action_expert,{id(p):p for p in self.action_expert.parameters()}).eval().requires_grad_(False)
        c=self.study_config;r=c['text_rank'];self.action_lora_enabled=True
        self.scope_manifest=install(self.action_expert,self._scope_paths,r,r*2)
        if c['film_last_layers']:
            paths=[f'decision.time_films.{i}.{j}'for i in range(28-c['film_last_layers'],28)for j in (0,1)]
            self.scope_manifest+=install(self.action_expert,paths,8,16)

    def scope_audit(self):
        current={n:p for n,p in self.action_expert.named_parameters()if p.requires_grad}
        allowed=lambda n:n.endswith(('.lora_A','.lora_B'))or (
            self.study_config['train_action_io']and n.startswith(('decision.action_input.','decision.action_output.','decision.action_position')))or (
            self.study_config['post_feedback']and n.startswith(('readouts.bridge.','readouts.evaluation.')))
        if not current or not all(allowed(n)for n in current):raise ValueError('Unexpected D44 trainable scope')
        reference_ids={id(p)for p in self.ref_action_expert.parameters()}
        if any(id(p)in reference_ids for p in current.values())or any(p.requires_grad for p in self.ref_action_expert.parameters()):
            raise ValueError('Current trainables alias the permanent IL44 reference')
        text=sum(p.numel()for n,p in current.items()if '.language_model.'in n)
        if text!=6422528*(self.study_config['text_rank']//16):raise ValueError('Wrong native D text LoRA closure')
        return dict(trainable_names=['action_expert.'+n for n in current],trainable_numel=sum(p.numel()for p in current.values()),
            text_lora_numel=text,adapters=self.scope_manifest,reference_frozen=True,base_frozen=True,
            current_prefix_recomputed_each_update=True,architecture_id='D',no_independent_DiT=True)

    def _prepare(self,condition,use_ref):
        if not getattr(self,'execution_fast_prefix',False):return super()._prepare(condition,use_ref)
        from .fast_prefix import fast_index_select
        key='reference'if use_ref else'current'
        if self._prepared_scope is not None and key in self._prepared_scope:return self._prepared_scope[key]
        policy=self.ref_action_expert if use_ref else self.action_expert
        current=condition.get('ref_current',condition['current'])if use_ref else condition['current']
        current=replace(current,policy_version='permanent_IL_REFERENCE'if use_ref else self._policy_version)
        with torch.autocast(self.device.type,dtype=torch.bfloat16):prefix=policy.prepare_prefix(current,auxiliary=False)
        indices=condition.get('candidate_to_scene')
        if indices is None:indices=torch.zeros(len(condition['context']),dtype=torch.long,device=self.device)
        prefix=fast_index_select(prefix,indices)
        self.prepared_context_builds[key]+=1
        if self._prepared_scope is not None:self._prepared_scope[key]=prefix
        return prefix

    def transition_terms(self,condition,chain,times,deltas,*,use_ref=False):
        # Reuses the exact source mean/std kernel, bypasses ALL source logprob clamps.
        scope=self._context_call(condition)if self._prepared_scope is None else nullcontext(self.materialize(condition))
        means=[];stds=[]
        with scope as cond:
            if self.study_config.get('probability_layout')=='flat_W3':
                policy=self.ref_action_expert if use_ref else self.action_expert
                prefix=self._prepare(cond,use_ref)
                indices=torch.zeros(len(chain)*3,device=self.device,dtype=torch.long)
                prefix=prefix.index_select(indices)
                states=chain[:,[7,8,9]].reshape(-1,8,3).contiguous().float()
                sigma=(times[[7,8,9]].float()/1000)[None].expand(len(chain),-1).reshape(-1).contiguous()
                with torch.autocast(self.device.type,dtype=torch.bfloat16):prediction=policy.denoise_x0(states,sigma,prefix)
                velocity=((states-prediction.float())/sigma[:,None,None]).reshape(len(chain),3,8,3)
                self._reference_calls+=int(use_ref);self._velocity_calls+=int(not use_ref)
                for j,k in enumerate((7,8,9)):
                    mean,std=self._sde_mean_std(chain[:,k],velocity[:,j],float(times[k])/1000,float(deltas[k]))
                    means.append(mean.float());stds.append(std)
                return torch.stack(means,1),torch.tensor(stds,device=chain.device,dtype=torch.float32)
            for k in (7,8,9):
                sigma=float(times[k])/1000
                velocity=self.action_velocity(chain[:,k],times[k].expand(len(chain)),cond,use_ref=use_ref)
                mean,std=self._sde_mean_std(chain[:,k],velocity,sigma,float(deltas[k]))
                means.append(mean.float());stds.append(std)
        return torch.stack(means,1),torch.tensor(stds,device=chain.device,dtype=torch.float32)

    def forward(self,r,*,return_components=False):
        c=self.study_config
        with self._context_call(r['condition'])as cond:
            mean,std=self.transition_terms(cond,r['chain'],r['timesteps'],r['deltas'])
            if not torch.equal(std,r['std']):raise ValueError('Sampling and logprob covariance differ')
            logr,kl=gaussian_terms(mean,r['old_mean'],r['ref_mean'],std,r['chain'][:,[8,9,10]])
            if r['expect_fresh']:
                error=logr.detach().expm1().abs().max()
                if not torch.isfinite(error)or float(error)>1e-4:raise FloatingPointError('Fresh native joint ratio exceeds1e-4: '+str(float(error)))
            pg,metrics=clipped_loss(logr,r['advantages'],c['clip'])
            anchor=kl.mean();total=pg+r['beta']*anchor
            prefix=None
            def current_prefix():
                nonlocal prefix
                if prefix is None:prefix=self._prepare(cond,False)
                return prefix
            if c['arm']in('D3_FPO','D3_AWM'):
                if c['arm']=='D3_FPO':
                    energies,_=mc_predictions_energy(self.action_expert,current_prefix(),r['fm_state'],r['fm_sigma'],r['fm_target'],
                        fuse=bool(getattr(self,'execution_fuse_fm_mc',False)))
                    energy=energies.mean(0)
                    pg,proxy=clipped_loss((r['old_energy']-energy)[:,None],r['advantages'],c['clip']);metrics.update({'fpo_'+k:v for k,v in proxy.items()})
                else:
                    energies=[];references=[];predictions=[]
                    for state,sigma,target in zip(r['fm_state'],r['fm_sigma'],r['fm_target']):
                        with torch.autocast(self.device.type,dtype=torch.bfloat16):
                            pred=self.action_expert.denoise_x0(state,sigma,current_prefix())
                        energies.append(velocity_energy(state,pred,target,sigma));predictions.append(pred)
                        with torch.no_grad(),torch.autocast(self.device.type,dtype=torch.bfloat16):
                            references.append(self.ref_action_expert.denoise_x0(state,sigma,self._prepare(cond,True)))
                    energy=torch.stack(energies).mean(0)
                    pg=awm_loss(energy,r['advantages'],torch.stack(references),torch.stack(predictions),c['reference_coefficient'])
                total=pg+r['beta']*anchor
            aux=mean.sum()*0
            if 'teacher' in r:
                one=current_prefix().index_select(torch.zeros(1,device=self.device,dtype=torch.long))
                with torch.autocast(self.device.type,dtype=torch.bfloat16):
                    pred=self.action_expert.denoise_x0(r['positive_state'],r['positive_sigma'],one)
                    last=self.action_expert.denoise_x0(r['nominal_chain'][:,9],r['nominal_sigma_last'],one)
                pos=positive_loss(pred,r['teacher']);lastloss=positive_loss(last,r['teacher'])
                aux=aux+c['positive_weight']*pos+c['last_weight']*lastloss
                metrics.update(positive_cleanx=pos.detach(),last_step=lastloss.detach())
            if c['post_feedback']and r.get('enable_post_feedback',r['epoch']>=c['post_start_epoch']):
                from .post_feedback import loss
                one=current_prefix().index_select(torch.zeros(1,device=self.device,dtype=torch.long))
                with torch.autocast(self.device.type,dtype=torch.bfloat16):
                    post=loss(self.action_expert,cond,r['chain'][:,-1],r['records'],one)
                aux=aux+c['post_weight']*post['loss'];metrics.update(post_loss=post['loss'].detach(),event_accuracy=post['event_accuracy'].detach())
            metrics.update(pg=pg.detach(),reference_KL_joint=anchor.detach(),reference_KL_per_coordinate=anchor.detach()/24,
                auxiliary_loss=aux.detach(),fresh_ratio_error=logr.detach().expm1().abs().max()if r['expect_fresh']else mean.new_tensor(0.))
            if return_components:return total+aux,metrics,dict(policy=pg,reference=r['beta']*anchor,auxiliary=aux)
            return total+aux,metrics
