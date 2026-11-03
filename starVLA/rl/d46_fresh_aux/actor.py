"""Current-only fresh R/heads, reusing every native D4 loss and sampler kernel."""
from starVLA.rl.d44_native_rl.actor import D44Actor
from starVLA.rl.d44_native_rl.expanded_scope import groups as original_groups
from .modules import attach_current_only
from .protocol import EXPERIMENT,PROTOCOL_SHA256


class FreshAuxActor(D44Actor):
    def __init__(self,model,backend,cfg,*,qualification=False):
        if cfg.get('experiment_id')!=EXPERIMENT or cfg['arm']!='D4' or not cfg['post_feedback']:
            raise ValueError('Only explicitly declared D4_FRESH_AUX is implemented')
        self.fresh_aux_qualification=bool(qualification)
        if not qualification and(model.decision.hidden_dim,model.decision.num_layers)!=(2048,28):
            raise ValueError('Formal fresh auxiliary scope requires the full native D policy')
        super().__init__(model,backend,cfg)

    def _configure_action_grpo(self,cfg):
        # The permanent reference is still the original pruned IL46 policy.
        if hasattr(self.action_expert,'readouts')or self.action_expert.decision.r_embeddings is not None:
            raise ValueError('Fresh route cannot inherit old or fabricated auxiliary modules')
        super()._configure_action_grpo(cfg)
        if hasattr(self.ref_action_expert,'readouts')or self.ref_action_expert.decision.r_embeddings is not None:
            raise ValueError('The permanent reference must remain the pure IL46 policy')

    def _after_policy_initialization(self):
        self.fresh_auxiliary_initialization=attach_current_only(self.action_expert)

    def scope_audit(self):
        current={n:p for n,p in self.action_expert.named_parameters()if p.requires_grad}
        text={path+'.'+name for path in self._scope_paths for name in ('lora_A','lora_B')}
        fresh=set(self.fresh_auxiliary_initialization['new_parameter_names'])
        if set(current)!=text|fresh:raise ValueError('Fresh auxiliary trainable scope differs from its exact declared closure')
        reference={id(p)for p in self.ref_action_expert.parameters()}
        if any(id(p)in reference for p in current.values())or any(p.requires_grad for p in self.ref_action_expert.parameters()):
            raise ValueError('Current trainables alias the permanent IL46 reference')
        if hasattr(self.ref_action_expert,'readouts')or self.ref_action_expert.decision.r_embeddings is not None:
            raise ValueError('Fresh training-only modules leaked into the permanent reference')
        text_count=sum(current[k].numel()for k in text)
        if not self.fresh_aux_qualification and text_count!=6422528:
            raise ValueError('Formal text LoRA closure must remain the original D4 r16')
        return dict(trainable_names=['action_expert.'+n for n in current],trainable_numel=sum(p.numel()for p in current.values()),
            trainable_specs={'action_expert.'+n:dict(shape=list(p.shape),dtype=str(p.dtype))for n,p in current.items()},
            text_lora_numel=text_count,adapters=self.scope_manifest,reference_frozen=True,
            current_prefix_recomputed_each_update=True,architecture_id='D',no_independent_DiT=True,
            experiment_id=EXPERIMENT,auxiliary_origin='fresh_seed42_not_IL_restored',
            fresh_auxiliary_protocol_sha256=PROTOCOL_SHA256,
            fresh_parameter_names=['action_expert.'+n for n in sorted(fresh)],
            original_deployment_base_frozen=True,reference_has_no_R_or_readouts=True)


def optimizer_groups(actor,cfg):
    query=actor.action_expert.decision.r_embeddings
    rows=[]
    for row in original_groups(actor,cfg):
        params=[p for p in row['params']if p is not query]
        if params:rows.append(dict(row,params=params))
    rows.append(dict(name='fresh_post_queries',lr=2e-5,initial_lr=2e-5,params=[query]))
    actual=[p for row in rows for p in row['params']];wanted=[p for p in actor.parameters()if p.requires_grad]
    if len(actual)!=len(set(map(id,actual)))or set(map(id,actual))!=set(map(id,wanted)):
        raise ValueError('Optimizer does not contain the exact fresh auxiliary closure')
    if any(query is p for row in rows if row['name']=='text_lora'for p in row['params']):
        raise ValueError('Fresh R16 was incorrectly classified as text LoRA')
    return rows
