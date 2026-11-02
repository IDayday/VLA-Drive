"""Optional detached IL-only W3 reference terms, owned by a saved collection.

Never cache current-policy terms/context or any autograd graph. The default is
the original reference forward. Enabling this execution optimization requires
the existing explicit source-migration admission, not a scientific cfg change.
"""
import copy
import torch

from starVLA.rl.qwen_world_safety_exploration.retention_material import digest
from starVLA.rl.simwam_recipe import checkpoint

SCHEMA = 'world_frozen_IL_reference_W3_terms_cache_v1'
KEY = 'frozen_IL_reference_terms'
TERMS = ('mean', 'std', 'vel')
REF_MODULES = ('ref_action_expert', 'ref_scene_provider', 'ref_predictive_provider')
PROTOCOL = dict(schema=SCHEMA, enabled_by='explicit execution flag --frozen-reference-cache',
    cached='detached frozen IL-only W3 mean/std/velocity; same saved inputs and dtype',
    forbidden='current trainable memory/KV, current-policy terms, PG, advantages, old_logp, or any autograd graph',
    persistence='native per-leaf pending material; input/source/parent identity and tensor seal validated',
    old_pending='validate complete checkpoint first; absent cache may then be built from this same frozen IL only')


def context_for_identity(identity):
    parent = identity.get('IL_parent', {})
    model_sha = parent.get('model_sha256') or identity.get('SFT_sha256')
    if not model_sha or not identity.get('sha256') or not identity.get('recipe'):
        raise ValueError('Reference cache requires explicit same-run IL-parent and recipe identity')
    return dict(schema=SCHEMA, run=identity['sha256'], IL_parent_model_sha256=model_sha,
                recipe_sha256=digest(identity['recipe']),
                reference_source_sha256=digest(identity.get('source_files', {})),
                projected_condition_identity=identity.get('projected_condition_identity'))


def _frozen_versions(model):
    modules = [(name, getattr(model, name, None)) for name in REF_MODULES]
    if modules[0][1] is None:
        raise ValueError('Reference cache requires an actual frozen IL action expert')
    versions = []
    for name, module in modules:
        if module is None:continue
        if module.training or any(p.requires_grad for p in module.parameters()):
            raise ValueError('Only permanently frozen eval-mode IL modules may be cached')
        versions.extend((name+'.'+key, id(p), p._version) for key,p in module.named_parameters())
        versions.extend((name+'.buffer.'+key, id(v), v._version) for key,v in module.named_buffers())
    return tuple(versions)


def enable_reference_cache(model, identity):
    """Operator-only opt-in. Bind constants after IL/ref restoration, once/run."""
    context = context_for_identity(identity)
    conditions=getattr(model,'conditions',None)
    if conditions is not None and getattr(conditions, 'sha', context['projected_condition_identity']) != context['projected_condition_identity']:
        raise ValueError('Frozen condition provider differs from the saved run identity')
    object.__setattr__(model, '_frozen_IL_reference_context', context)
    object.__setattr__(model, '_frozen_IL_reference_versions', _frozen_versions(model))
    return context


def disable_reference_cache(model):
    object.__setattr__(model, '_frozen_IL_reference_context', None)


def _ancestry(rollout, context):
    descriptor = rollout['cond_bg']
    if descriptor.get('identity') != context['projected_condition_identity']:
        raise ValueError('Frozen reference condition identity changed')
    if rollout['behavior_version']['run'] != context['run']:
        raise ValueError('Frozen reference collection belongs to a different run')
    return dict(context=copy.deepcopy(context),
                saved_inputs_sha256=digest({k:rollout[k] for k in ('chain','timesteps','deltas','stoch_idx','cond_bg')}),
                behavior_version=copy.deepcopy(rollout['behavior_version']),
                token=rollout.get('token'), log=rollout.get('log'))


def _payload_digest(payload):
    return digest({k:v for k,v in payload.items() if k!='sha256'})


def validate_reference_cache(rollout, identity=None, *, context=None):
    """Pure CPU/device validation used by native pending-checkpoint admission.

    Absent old-pending cache is legitimate: it may be lazily built ONLY from the
    restored same-run frozen IL. A present cache must never be silently rebuilt.
    """
    payload = rollout.get(KEY)
    if payload is None:return None
    if context is None:context=context_for_identity(identity)
    if payload.get('schema')!=SCHEMA or payload.get('ancestry')!=_ancestry(rollout,context):
        raise ValueError('Frozen reference cache input/parent/source ancestry changed')
    if payload.get('sha256')!=_payload_digest(payload):
        raise ValueError('Frozen reference cache tensor/audit seal changed')
    terms=payload.get('terms',{});g=rollout['chain'].shape[0];steps=len(rollout['stoch_idx'])
    if set(terms)!=set(TERMS):raise ValueError('Frozen reference terms incomplete')
    for name,x in terms.items():
        shape=(steps,) if name=='std' else (g,steps,8,3)
        if (not torch.is_tensor(x) or tuple(x.shape)!=shape or x.requires_grad or x.grad_fn is not None
                or not torch.isfinite(x).all() or (name=='std' and not (x>0).all())):
            raise ValueError('Invalid detached frozen reference '+name)
    return payload


def _forward(model, rollout):
    with torch.no_grad():
        _,terms=model.action_chain_logprobs(rollout['cond_bg'],rollout['chain'],rollout['timesteps'],rollout['deltas'],
            stoch_idx=rollout.get('stoch_idx'),use_ref=True,return_terms=True)
    return terms


def reference_terms(model, rollout):
    """Shared ResearchLoss/AWFM/FPO anchor hook; return original raw constants."""
    context=getattr(model,'_frozen_IL_reference_context',None)
    if context is None:return _forward(model,rollout)
    if _frozen_versions(model)!=model._frozen_IL_reference_versions:
        raise ValueError('The frozen IL reference changed after cache binding')
    payload=validate_reference_cache(rollout,context=context)
    if payload is None:
        terms=_forward(model,rollout)
        audit=getattr(model,'_probability_audit',{}).get('reference')
        payload=dict(schema=SCHEMA,ancestry=_ancestry(rollout,context),
            terms={k:terms[k].detach().clone() for k in TERMS},
            reference_audit=checkpoint.cpu_tree(audit) if audit is not None else None)
        payload['sha256']=_payload_digest(payload)
        rollout[KEY]=payload
        validate_reference_cache(rollout,context=context)
    device=rollout['chain'].device
    # Save/restore already uses native cpu_tree/device_tree. This conversion is
    # a no-op for an existing device tensor and never changes dtype/arithmetic.
    terms={k:v.to(device=device) for k,v in payload['terms'].items()}
    if payload.get('reference_audit') is not None:
        model._probability_audit['reference']=checkpoint.device_tree(payload['reference_audit'],device)
    return terms


def populate_reference_caches(model, window):
    if getattr(model,'_frozen_IL_reference_context',None) is not None:
        for leaf in window['samples']:reference_terms(model,leaf)
