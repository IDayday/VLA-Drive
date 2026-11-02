"""D-only actor on the audited native sampler and functional prefix lifecycle."""
import copy
from contextlib import nullcontext
import torch
from torch import nn
from starVLA.rl.qwen_world_compare.policy_adapter import FourFrameworkRecipeActor
from .expanded_scope import install, independent_trainable_module
from .gaussian_objective import gaussian_terms, clipped_loss
from .native_cleanx_aux import positive_loss
from .fpo import velocity_energy, awm_loss


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

    def transition_terms(self,condition,chain,times,deltas,*,use_ref=False):
        # Reuses the exact source mean/std kernel, bypasses ALL source logprob clamps.
        scope=self._context_call(condition)if self._prepared_scope is None else nullcontext(self.materialize(condition))
        means=[];stds=[]
        with scope as cond:
            for k in (7,8,9):
                sigma=float(times[k])/1000
                velocity=self.action_velocity(chain[:,k],times[k].expand(len(chain)),cond,use_ref=use_ref)
                mean,std=self._sde_mean_std(chain[:,k],velocity,sigma,float(deltas[k]))
                means.append(mean.float());stds.append(std)
        return torch.stack(means,1),torch.tensor(stds,device=chain.device,dtype=torch.float32)

    def forward(self,r):
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
                energies=[];references=[];predictions=[]
                for state,sigma,target in zip(r['fm_state'],r['fm_sigma'],r['fm_target']):
                    with torch.autocast(self.device.type,dtype=torch.bfloat16):
                        pred=self.action_expert.denoise_x0(state,sigma,current_prefix())
                    energies.append(velocity_energy(state,pred,target,sigma));predictions.append(pred)
                    if c['arm']=='D3_AWM':
                        with torch.no_grad(),torch.autocast(self.device.type,dtype=torch.bfloat16):
                            references.append(self.ref_action_expert.denoise_x0(state,sigma,self._prepare(cond,True)))
                energy=torch.stack(energies).mean(0)
                if c['arm']=='D3_FPO':pg,proxy=clipped_loss((r['old_energy']-energy)[:,None],r['advantages'],c['clip']);metrics.update({'fpo_'+k:v for k,v in proxy.items()})
                else:pg=awm_loss(energy,r['advantages'],torch.stack(references),torch.stack(predictions))
                total=pg+r['beta']*anchor
            aux=mean.sum()*0
            if 'teacher' in r:
                one=current_prefix().index_select(torch.zeros(1,device=self.device,dtype=torch.long))
                with torch.autocast(self.device.type,dtype=torch.bfloat16):
                    pred=self.action_expert.denoise_x0(r['positive_state'],r['positive_sigma'],one)
                    last=self.action_expert.denoise_x0(r['nominal_chain'][:,9],r['timesteps'][9:10]/1000,one)
                pos=positive_loss(pred,r['teacher']);lastloss=positive_loss(last,r['teacher'])
                aux=aux+c['positive_weight']*pos+c['last_weight']*lastloss
                metrics.update(positive_cleanx=pos.detach(),last_step=lastloss.detach())
            if c['post_feedback']and r['epoch']>=c['post_start_epoch']:
                from .post_feedback import loss
                one=current_prefix().index_select(torch.zeros(1,device=self.device,dtype=torch.long))
                with torch.autocast(self.device.type,dtype=torch.bfloat16):
                    post=loss(self.action_expert,cond,r['chain'][:,-1],r['records'],one)
                aux=aux+c['post_weight']*post['loss'];metrics.update(post_loss=post['loss'].detach(),event_accuracy=post['event_accuracy'].detach())
            metrics.update(pg=pg.detach(),reference_KL_joint=anchor.detach(),reference_KL_per_coordinate=anchor.detach()/24,
                auxiliary_loss=aux.detach(),fresh_ratio_error=logr.detach().expm1().abs().max()if r['expect_fresh']else mean.new_tensor(0.))
            return total+aux,metrics
