"""Independent canonical-inversion cancellation, sharing the protected engine.

Both signs are partially/fully cancelled using verified lower-score positive
mass. This differs from unconditionally zeroing every negative high candidate.
"""
import copy
import torch

from . import protected_credit as engine
from .inversion_cancellation import cancel_inversions, PROTOCOL as PAIR_PROTOCOL
from .credit import raw_advantage_parts
from .group_runtime import validate_recipe

SCHEMA = 'world_SafetyEP_canonical_inversion_pair_cancel_fixed_base_std_v1'


def protocol(g):
    p = engine.protocol(g)
    p.update(schema=SCHEMA, pair_math_protocol=PAIR_PROTOCOL,
        eligibility='strict4 canonical>=.9 originalnegative endpoint, and strictly feasible lower-canonical originalpositive endpoint',
        delta='deterministic pair transfer min(-A_high,A_lower); high increases, lower decreases; add delta to EP only; use returned updated_rawA',
        order='negative endpoints score descending/index ascending; positive endpoints score ascending/index ascending',
        group_mean='original group sum conserved in real arithmetic; FP64 roundoff recorded, never recentered',
        preserved='unchanged candidates raw/normalizedA bitexact; same original world*G scale; original entire group mask retained',
        mask='original mask retained even when all updated rawA become0; new truePG0 counted explicitly',
        unmatched='high negatives remain if no lower-score positive mass remains; no fabricated witness/positive advantage',
        sign_relation='custom canonical-dominance sign cancellation, not Sign/GRPO theorem replication',
        exclusivity='mutually exclusive with Protected/Saturated/DenseDAC; separate independently registered schema')
    return p


def validate_inversion_recipe(cfg):
    validate_recipe(cfg)
    if cfg['research']['structured_advantage'] != SCHEMA:
        raise ValueError('Inversion cancellation requires its independent credit schema')
    return cfg


@torch.no_grad()
def decision_from_material(components, quality, original_raw):
    info = engine.decision_from_material(components, quality, original_raw)
    result = cancel_inversions(info['original_raw_A'], info['quality_material'][:,:,1],
        components.detach().cpu().double(), info['quality_material'][:,:,0])
    changed = result['updated_rawA'].ne(info['original_raw_A'])
    info.update(schema=SCHEMA, status='APPLIED_CANONICAL_INVERSION_CANCELLATION' if changed.any() else 'UNCHANGED_NO_CANONICAL_INVERSION',
        activated=bool(changed.any()), eligible=changed, delta=result['delta'], updated_raw_A=result['updated_rawA'],
        pairs=result['pairs'], counts=result['counts'], group_sum_before=result['group_sum_before'],
        group_sum_after=result['group_sum_after'], group_sum_roundoff=result['max_group_sum_roundoff'],
        pair_math_protocol=result['protocol'])
    return info


def updated_parts(components, info):
    old = raw_advantage_parts(components)
    return old, dict(old, A_EP=old['A_EP']+info['delta'].to(components.device),
                      A_raw=info['updated_raw_A'].to(components.device))


class InversionCredit(engine.ProtectedCredit):
    payload_key = 'inversion_credit'
    metrics_prefix = 'inversion_credit_'
    tensor_fields = dict(engine.ProtectedCredit.tensor_fields,
        updated_raw_A=torch.float64, group_sum_before=torch.float64, group_sum_after=torch.float64)
    expected_protocol = staticmethod(protocol)
    validate_recipe = staticmethod(validate_inversion_recipe)
    decision_from_material = staticmethod(decision_from_material)
    update_parts = staticmethod(updated_parts)

    def normalize_changed(self, base, parts, info):
        value = base['advantages'].clone()
        changed = info['eligible'].reshape(-1).to(value.device)
        replacement = (parts['A_raw']/(base['normalization_std']+1e-8)).reshape(-1).float()
        value[changed] = replacement[changed]
        return value

    def statistics(self, base, parts, info, out):
        meta = dict(base['statistics']);counts=info['counts']
        post_zero = not bool(parts['A_raw'].ne(0).any())
        meta.update(structured_advantage_schema=SCHEMA,normalization=self.protocol['normalization'],
            structured_zero_groups=int(post_zero),inversion_credit_status=info['status'],
            inversion_credit_activated_groups=int(info['activated']),
            inversion_credit_quality_candidates=int(info['quality_flags'].sum()),
            inversion_credit_activated_candidates=counts['changed_candidates'],
            inversion_credit_original_negative_high=counts['original_negative_high'],
            inversion_credit_original_qualifying_pairs=counts['original_qualifying_pairs'],
            inversion_credit_transfers=counts['transfers'],
            inversion_credit_negative_high_cleared=counts['negative_high_cleared'],
            inversion_credit_remaining_negative_high=counts['remaining_negative_high'],
            inversion_credit_remaining_qualifying_pairs=counts['remaining_qualifying_pairs'],
            inversion_credit_new_PG0_groups=counts['newly_allzero_groups'],
            inversion_credit_post_raw_PG0_groups=int(post_zero),
            inversion_credit_original_group_mask_fraction=float(base['adv_mask'].mean()),
            inversion_credit_true_nonzero_adv_fraction=float(out['advantages'].ne(0).float().mean()),
            inversion_credit_group_sum_roundoff=info['group_sum_roundoff'],
            inversion_credit_transferred_mass=sum(p['transfer'] for p in info['pairs']),
            inversion_credit_raw_group_mean_before=float(info['original_raw_A'].mean()),
            inversion_credit_raw_group_mean_after=float(parts['A_raw'].mean()),
            inversion_credit_original_normalization_std=float(base['normalization_std']),
            inversion_credit_normalized_removed_mass=float((out['advantages']-base['advantages']).detach().cpu().double().sum()),
            inversion_credit_adv_cap5_candidates=int(out['advantages'].abs().gt(5).sum()))
        return meta


# Same transaction/quality-material engine, dispatched by this credit's protocol
# and deterministic recomputation hooks; no duplicated native validators.
validate_inversion_boundary = engine.validate_protected_boundary
validate_inversion_ranks = engine.validate_protected_ranks


class InversionLoss(engine.ProtectedLoss):
    metrics_prefix = 'inversion_credit_'


class InversionRuntimeMixin:
    _credit_mixin_variant = 'inversion'

    def make_credit(self,cfg):
        from .dense_credit import DenseRuntimeMixin
        from .saturated_credit import SaturatedRuntimeMixin
        variants={c.__dict__['_credit_mixin_variant'] for c in type(self).__mro__ if '_credit_mixin_variant' in c.__dict__}
        if len(variants)>1 or isinstance(self,(engine.ProtectedRuntimeMixin,DenseRuntimeMixin,SaturatedRuntimeMixin)):
            raise ValueError('Inversion and other credit mixins are mutually exclusive')
        validate_inversion_recipe(cfg)
        return InversionCredit(cfg['grpo']['sample']['group_size'])
    def make_actor(self,cfg,args,device):
        actor,oldcfg,sft=super().make_actor(cfg,args,device);actor._loss=InversionLoss(cfg)
        return actor,oldcfg,sft
    def finalize_advantages(self,rollout):return self.credit.finalize(rollout)
    def validate_boundary(self,*args,**kwargs):return validate_inversion_boundary(*args,credit=self.credit,**kwargs)
    def validate_ranks(self,summaries):return validate_inversion_ranks(summaries,credit=self.credit)
    def recipe_identity(self,cfg,args,condition_sha):
        value=super().recipe_identity(cfg,args,condition_sha)
        value['canonical_inversion_credit_protocol']=copy.deepcopy(self.credit.protocol)
        return value
