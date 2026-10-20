"""Distinct B0/B1 targets on the actual S0 noise-to-action FM direction."""
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PairedFlowSample:
    minus: torch.Tensor
    plus: torch.Tensor
    noise: torch.Tensor
    times: torch.Tensor

    def validate(self):
        if self.minus.ndim != 3 or self.minus.shape[1:] != (8, 4):
            raise ValueError('Original raw8x4 trajectory required')
        if self.plus.shape != self.minus.shape or self.noise.shape != self.minus.shape or self.times.shape != (len(self.minus),):
            raise ValueError('Paired noise/time/trajectory shapes differ')
        if any(x.requires_grad or not x.is_floating_point() or not torch.isfinite(x).all()
               for x in (self.minus, self.plus, self.noise, self.times)):
            raise ValueError('Paired supervision must be finite detached tensors')
        if any(x.device != self.minus.device for x in (self.plus, self.noise, self.times)):
            raise ValueError('Paired tensors must share a device')
        # Preserve source s=.999 Beta time support, including its small negative tail.
        if ((self.times < -1./999 - 1e-7) | (self.times > 1)).any():
            raise ValueError('Time outside original unclamped Beta support')
        return self

    def state(self, endpoint):
        self.validate()
        if endpoint not in ('minus', 'plus'):
            raise ValueError('Explicit shared-path endpoint required')
        a = self.minus if endpoint == 'minus' else self.plus
        u = self.times[:, None, None]
        return (1-u)*self.noise + u*a

    def target_velocity(self):
        self.validate()
        return self.plus-self.noise


def sample_original_pair(minus, plus, *, noise_generator, time_generator):
    if noise_generator is time_generator:
        raise ValueError('Noise and Beta time need independent resumable streams')
    noise = torch.randn(minus.shape, device=minus.device, dtype=minus.dtype, generator=noise_generator)
    uniform = torch.rand((len(minus),), device=minus.device, dtype=minus.dtype, generator=time_generator)
    times = (.999-uniform.pow(1./1.5))/.999
    return PairedFlowSample(minus.detach(), plus.detach(), noise, times).validate()


def corrected_residual_target(pair, frozen_base_velocity):
    """At x_minus the residual is v_plus* - v_base, not merely plus-minus."""
    pair.validate()
    if frozen_base_velocity.shape != pair.minus.shape or not torch.isfinite(frozen_base_velocity).all():
        raise ValueError('Invalid frozen Base velocity')
    return pair.target_velocity()-frozen_base_velocity.detach()


def masked_mean(values, valid, *, global_denominator=None, world_size=1):
    """Sum/valid-count; caller supplies one global denominator across microbatches.

    DDP averages gradients, hence multiplication by world_size. Do not average
    local rank means. All-invalid ranks retain graph-connected zero loss.
    """
    if valid.dtype != torch.bool or valid.shape != values.shape:
        raise ValueError('Exact value mask required')
    if type(world_size) is not int or world_size < 1:
        raise ValueError('Positive world_size required')
    if world_size > 1 and global_denominator is None:
        raise ValueError('DDP requires an explicit global valid denominator')
    if not torch.isfinite(values).all():
        raise FloatingPointError('Nonfinite model output cannot be masked away')
    local = int(valid.sum())
    denominator = local if global_denominator is None else float(global_denominator)
    if not math.isfinite(denominator) or denominator < local or denominator < 0:
        raise ValueError('Invalid global loss denominator')
    if denominator == 0:
        return values.sum()*0
    return (values.float()*valid).sum()*(world_size/denominator)


def velocity_loss(prediction, target, valid_scenes, **reduction):
    if prediction.shape != target.shape or valid_scenes.shape != (len(prediction),):
        raise ValueError('FM prediction/target/scene-mask mismatch')
    return masked_mean((prediction.float()-target.detach().float()).square(),
                       valid_scenes[:, None, None].expand_as(prediction), **reduction)


def effect_consistency_loss(teacher, effect_context, pair, predicted_velocity, true_delta,
                            effect_valid, effect_scales, *, global_denominator=None,
                            world_size=1, direction_temperature=.1,
                            direction_global_denominator=None):
    """Training-only frozen teacher; no teacher is registered in deployable model.

    epsilon+v is a velocity-derived endpoint proxy, not the completed ODE output.
    Its effect loss is an auxiliary surrogate; acceptance requires Euler10+NAVSIM.
    Teacher signature: teacher(effect_context, minus_raw, intervention_raw)->[B,E].
    It must preserve input derivatives while all its own parameters stay frozen.
    """
    if any(p.requires_grad for p in teacher.parameters()) or any(m.training for m in teacher.modules()):
        raise ValueError('Effect teacher must be frozen eval; never silently freeze it')
    if not isinstance(effect_context, dict) or set(effect_context) != {'memory', 'ego', 'mask'}:
        raise ValueError('Effect context permits only current memory/ego/mask, never GT or feedback')
    memory, ego, mask = (effect_context[k] for k in ('memory', 'ego', 'mask'))
    if (not all(isinstance(x, torch.Tensor) for x in (memory, ego, mask))
            or memory.ndim != 3 or memory.shape[0] != len(pair.minus)
            or ego.shape != (len(pair.minus), 1, 4) or mask.shape != memory.shape[:2]
            or mask.dtype != torch.bool or not mask.any(-1).all()
            or not torch.isfinite(memory).all() or not torch.isfinite(ego).all()):
        raise ValueError('Malformed current-only W/A/ego effect context')
    context = {k: x.detach() for k, x in effect_context.items()}
    pair.validate()
    if predicted_velocity.shape != pair.minus.shape or direction_temperature <= 0:
        raise ValueError('Effect velocity/temperature contract mismatch')
    endpoint_proxy = pair.noise + predicted_velocity
    predicted_delta = teacher(context, pair.minus, endpoint_proxy-pair.minus)
    if predicted_delta.shape != true_delta.shape or effect_valid.shape != true_delta.shape:
        raise ValueError('Effect target dimensions differ from teacher schema')
    scales = torch.as_tensor(effect_scales, device=true_delta.device, dtype=torch.float32)
    if scales.shape != (true_delta.shape[-1],) or not torch.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError('Predeclared positive per-task effect scales required')
    if not torch.isfinite(true_delta).all() or not torch.isfinite(predicted_delta).all():
        raise FloatingPointError('Nonfinite physical delta or teacher prediction')
    target = true_delta.detach().float()/scales
    pred = predicted_delta.float()/scales
    consistency = masked_mean((pred-target).square(), effect_valid,
                              global_denominator=global_denominator, world_size=world_size)
    # Compare predicted effect with no intervention. Actual effect magnitude
    # remains in the global objective; never normalize it away per scene.
    weight = target.abs()*effect_valid*(target.abs() > 1e-6)
    local_weight = float(weight.sum().detach())
    if world_size != 1 and direction_global_denominator is None:
        raise ValueError('DDP effect direction needs the global weight denominator')
    denom = local_weight if direction_global_denominator is None else float(direction_global_denominator)
    if not math.isfinite(denom) or denom < local_weight:
        raise ValueError('Invalid global effect direction weight denominator')
    direction = ((F.softplus(-target.sign()*pred/direction_temperature)*weight).sum()*world_size/denom
                 if denom > 0 else pred.sum()*0)
    return {'effect_consistency': consistency, 'effect_direction': direction,
            'predicted_effect': predicted_delta, 'endpoint_proxy': endpoint_proxy}


def verified_pair_effect_loss(student_delta,student_embedding,*,true_delta,teacher_embedding,
        effect_valid,repair_valid,effect_scales,global_denominator=None,direction_global_denominator=None,
        representation_global_denominator=None,world_size=1,direction_temperature=.1):
    """Native-delta supervision and frozen verified-pair representation targets.

No learned teacher query occurs here, and targets never enter the student
forward. This is auxiliary representation distillation, not a velocity gradient
from a simulator and not a query at an unsupported eps+v endpoint.
    """
    if (student_delta.ndim!=2 or student_delta.shape!=true_delta.shape or effect_valid.shape!=true_delta.shape
            or effect_valid.dtype!=torch.bool or repair_valid.shape!=(len(student_delta),) or repair_valid.dtype!=torch.bool
            or student_embedding.ndim!=2 or student_embedding.shape!=teacher_embedding.shape
            or len(student_embedding)!=len(student_delta) or direction_temperature<=0):
        raise ValueError('Verified effect/representation target shapes or masks differ')
    if not all(torch.isfinite(x).all() for x in (student_delta,student_embedding,true_delta,teacher_embedding)):
        raise FloatingPointError('Nonfinite verified effect target or student prediction')
    scales=torch.as_tensor(effect_scales,device=student_delta.device,dtype=torch.float32)
    if scales.shape!=(student_delta.shape[1],) or not torch.isfinite(scales).all() or (scales<=0).any():
        raise ValueError('Declared positive native physical scales required')
    valid=effect_valid & repair_valid[:,None]
    target=true_delta.detach().float()/scales;prediction=student_delta.float()/scales
    consistency=masked_mean((prediction-target).square(),valid,global_denominator=global_denominator,world_size=world_size)
    weight=target.abs()*valid*(target.abs()>1e-6);local_weight=float(weight.sum().detach())
    if world_size!=1 and direction_global_denominator is None:raise ValueError('DDP direction requires a global weight sum')
    denom=local_weight if direction_global_denominator is None else float(direction_global_denominator)
    if not math.isfinite(denom) or denom<local_weight:raise ValueError('Invalid verified direction denominator')
    direction=((F.softplus(-target.sign()*prediction/direction_temperature)*weight).sum()*world_size/denom
        if denom>0 else prediction.sum()*0)
    rep_valid=repair_valid[:,None].expand_as(student_embedding)
    representation=masked_mean((student_embedding.float()-teacher_embedding.detach().float()).square(),rep_valid,
        global_denominator=representation_global_denominator,world_size=world_size)
    return {'effect_consistency':consistency,'effect_direction':direction,'effect_representation':representation}
