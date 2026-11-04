"""Native D action-only AUGUstus adaptation; original B1 deployment is intact.

The full route trains the transformer trunk and existing action interfaces.
Frozen visual/input embeddings remain compatible with the observation cache.
It does not introduce, or claim to reproduce, the paper's textual CoT branch.
"""
import copy
import torch
from torch import nn
from starVLA.rl.d44_native_rl.actor import D44Actor
from starVLA.rl.d44_native_rl.expanded_scope import independent_trainable_module,install
from starVLA.rl.simwam_recipe.vendor.lora import LoRALinear


class AugustusActor(D44Actor):
    def _configure_action_grpo(self,backend):
        self.eval().requires_grad_(False)
        self.ref_action_expert=copy.deepcopy(self.action_expert,{id(p):p for p in self.action_expert.parameters()}).eval().requires_grad_(False)
        cfg=self.study_config;self.action_lora_enabled=True
        self.scope_manifest=install(self.action_expert,self._scope_paths,cfg['text_rank'],2*cfg['text_rank'])
        if cfg['film_last_layers']:
            layers=self.action_expert.decision.num_layers
            paths=[f'decision.time_films.{i}.{j}'for i in range(max(0,layers-cfg['film_last_layers']),layers)for j in(0,1)]
            self.scope_manifest+=install(self.action_expert,paths,8,16)

    def __init__(self,model,backend,cfg):
        if cfg['augustus'].get('action_output_precision')=='float32_linear':
            from .output_precision import install_fp32_action_output
            install_fp32_action_output(model.decision)
        if cfg['augustus'].get('text_action_precision'):
            from .text_precision import install_text_action_precision
            install_text_action_precision(model.decision,cfg['augustus']['text_action_precision'])
        self._initializing_augustus=True
        super().__init__(model,backend,cfg)
        scope=cfg['augustus']['scope']
        if scope=='full_text_action':
            # Base initialization installs zero LoRA before its legacy scope
            # audit. Remove those wrappers before cloning the full text trunk.
            for path,module in list(self.action_expert.named_modules()):
                if isinstance(module,LoRALinear):
                    parent,_,child=path.rpartition('.')
                    setattr(self.action_expert.get_submodule(parent),child,module.base)
            decision=self.action_expert.decision
            paths=[f'decision.qwen_model.language_model.layers.{i}'for i in range(decision.num_layers)]
            paths+=['decision.qwen_model.language_model.norm','decision.action_input','decision.action_output',
                    'decision.time_embedding','decision.time_films','decision.role_embedding','decision.ego_input']
            for path in paths:independent_trainable_module(self.action_expert,path)
            for name in('action_position','p_embeddings'):
                setattr(decision,name,nn.Parameter(getattr(decision,name).detach().clone()))
            self.scope_manifest=[dict(path=path,mode='full_parameter')for path in paths]
            self.action_lora_enabled=self.lora_enabled=False
        elif scope!='lora_text_action':raise ValueError('Unknown AUGUstus trainable scope')
        self._initializing_augustus=False
        self.closure=self.scope_audit()

    def scope_audit(self):
        if getattr(self,'_initializing_augustus',False):
            return dict(trainable_names=['action_expert.'+n for n,p in self.action_expert.named_parameters()if p.requires_grad],
                        trainable_numel=sum(p.numel()for p in self.action_expert.parameters()if p.requires_grad),initialization_only=True)
        current={n:p for n,p in self.action_expert.named_parameters()if p.requires_grad}
        reference={id(p)for p in self.ref_action_expert.parameters()}
        if not current or any(id(p)in reference for p in current.values()):
            raise ValueError('Trainable native D parameters alias the permanent IL reference')
        if any(p.requires_grad for p in self.ref_action_expert.parameters()):raise ValueError('Reference must be frozen')
        if any(n.startswith('decision.qwen_model.visual.')or '.embed_tokens.'in n for n in current):
            raise ValueError('Cached observation providers require frozen vision/input embeddings')
        scope=self.study_config['augustus']['scope']
        if scope=='full_text_action'and any(isinstance(m,LoRALinear)for m in self.action_expert.modules()):
            raise ValueError('The full trunk must not silently retain LoRA topology')
        return dict(architecture_id='D',trainable_names=['action_expert.'+n for n in current],
                    trainable_numel=sum(p.numel()for p in current.values()),adapters=self.scope_manifest,
                    reference_frozen=True,base_frozen=scope!='full_text_action',scope=scope,
                    visual_frozen=True,input_embeddings_frozen=True,current_prefix_recomputed_each_update=True,
                    no_independent_DiT=True,no_text_CoT_reproduction=True)

    def predict_clean(self,state,sigma,prefix,*,use_ref=False):
        policy=self.ref_action_expert if use_ref else self.action_expert
        self._reference_calls+=int(use_ref);self._velocity_calls+=int(not use_ref)
        with torch.autocast(self.device.type,dtype=torch.bfloat16):
            return policy.denoise_x0(state.float(),sigma.float(),prefix).float()

    @torch.no_grad()
    def sample_action_chain(self,condition,*_,generator=None,deterministic=False,init_actions=None,
                            velocity_use_ref=False,**kwargs):
        """Three actual lambda_t transitions, shared initial state within G."""
        from .kernel import step
        group=self.study_config['group'];spec=self.study_config['augustus']
        grid=torch.tensor(spec['training_sigma_grid'],device=self.device,dtype=torch.float32)
        if init_actions is None:
            if spec.get('initial_root_scheme')=='deployment_token_seed_v1':
                raise ValueError('Token-bound recipe requires explicit scene initial actions from collector')
            root_generator=(torch.Generator(device='cpu').manual_seed(spec['initial_root_seed'])
                            if 'initial_root_seed'in spec else generator)
            root=torch.randn((1,8,3),generator=root_generator,device='cpu').to(self.device)
            state=root.expand(group,-1,-1).clone()
        else:
            state=init_actions.to(self.device).float().clone()
            if state.shape!=(group,8,3):raise ValueError('Initial D group shape differs')
        states=[state.clone()];predictions=[];means=[];stds=[]
        with self._context_call(condition)as material:
            prefix=self._prepare(material,velocity_use_ref)
            if 'deterministic_prefix_grid'in spec:
                from starVLA.model.modules.qwen_world_postcot.flow import clean_x_step
                # Shared root permits one exact native prefix, independent of G.
                shared=torch.equal(state,state[:1].expand_as(state))
                p=prefix.index_select(torch.zeros(1,dtype=torch.long,device=self.device))if shared else prefix
                z=state[:1]if shared else state
                for sigma,nxt in zip(spec['deterministic_prefix_grid'][:-1],spec['deterministic_prefix_grid'][1:]):
                    x0=self.predict_clean(z,z.new_full((len(z),),sigma),p,use_ref=velocity_use_ref)
                    z=clean_x_step(z,x0,sigma,nxt,validate=False)
                state=z.expand(group,-1,-1).clone()if shared else z
                states=[state.clone()]
            for sigma,nxt in zip(grid[:-1],grid[1:]):
                x0=self.predict_clean(state,sigma.expand(group),prefix,use_ref=velocity_use_ref)
                if deterministic:
                    from starVLA.model.modules.qwen_world_postcot.flow import clean_x_step
                    mean=clean_x_step(state,x0,sigma,nxt,validate=False)
                    std=state.new_zeros((group,1,3));following=mean
                else:
                    innovation=torch.randn((group,8,3),generator=generator,device='cpu').to(self.device)
                    following,mean,std=step(x0,state,sigma.expand(group),nxt.expand(group),
                                           noise_scale=spec['noise_scale'],noise=innovation)
                predictions.append(x0);means.append(mean);stds.append(std)
                state=following;states.append(state.clone())
        self.augustus_sampling=dict(old_clean_x=torch.stack(predictions,1).detach(),
                                   old_mean=torch.stack(means,1).detach(),std=torch.stack(stds,1).detach())
        return torch.stack(states,1),grid,grid[1:]-grid[:-1],torch.arange(3,device=self.device)

    def forward(self,r,*,return_components=False):
        from .kernel import stats,policy_loss
        spec=self.study_config['augustus'];logratios=[];kls=[]
        with self._context_call(r['condition'])as material:
            prefix=self._prepare(material,False)
            for k,(sigma,nxt)in enumerate(zip(r['timesteps'][:-1],r['timesteps'][1:])):
                current=self.predict_clean(r['chain'][:,k],sigma.expand(len(r['chain'])),prefix)
                logr,kl=stats(current,r['old_clean_x'][:,k],r['ref_clean_x'][:,k],
                              r['chain'][:,k],r['chain'][:,k+1],sigma.expand(len(current)),
                              nxt.expand(len(current)),noise_scale=spec['noise_scale'])
                logratios.append(logr);kls.append(kl)
        logr=torch.stack(logratios,1);kl=torch.stack(kls,1)
        fresh=logr.detach().expm1().abs().max()if r['expect_fresh']else logr.new_zeros(())
        if not torch.isfinite(fresh)or fresh>1e-4:raise FloatingPointError('Fresh JaT ratio differs: '+str(float(fresh)))
        total,metrics=policy_loss(logr,r['advantages'],kl,clip=spec['clip'],beta=spec['fixed_reference_beta'])
        metrics=dict(metrics);coordinate_kl=kl.mean()
        metrics.update(fresh_ratio_error=fresh,reference_KL_joint=24*coordinate_kl.detach(),
                       reference_KL_per_coordinate=coordinate_kl.detach(),auxiliary_loss=total.detach()*0)
        if return_components:
            reference=spec['fixed_reference_beta']*coordinate_kl
            return total,metrics,dict(policy=total-reference,reference=reference,auxiliary=total*0,
                                      logratio=logr,reference_KL_steps=kl)
        return total,metrics
