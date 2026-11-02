"""Genuine B1 clean-x deployment states for TRAIN-only nominal improvement."""
import copy
import math
import time
import torch
from starVLA.model.modules.qwen_sourceflow.action_codec import ActionCodec
from starVLA.model.modules.qwen_world_postcot.flow import shift_sigma, clean_x_step
from starVLA.rl.simwam_recipe import checkpoint
from starVLA.rl.simwam_recipe.contracts import seed_for
from starVLA.rl.qwen_world_safety_exploration.retention_material import digest, sealed, verify_seal, validate_feedback

SCHEMA = 'world_performance_true_B1_nominal_ODE_target_v1'
STRICT = ('no_at_fault_collisions', 'drivable_area_compliance', 'time_to_collision_within_bound', 'comfort')
PROTOCOL = dict(schema=SCHEMA, steps=10, shift=5., seed=2101, batch=1,
    root="seed_for('native_qwen_sourceflow_ODE',2101,token); CPUtorch.randn((1,8,3),float32)",
    final_step='direct current clean-x head prediction.float(); no velocity reconstruction/Euler rounding',
    differentiation='saved last-state is detached; current Memory rebuilt for the one final head invocation',
    teacher_scope='actual canonical TRAIN feedback, own run/version only; never NAV or another run')


def strict_high(record):
    return all(record['metrics'][k] == 1 for k in STRICT) and float(record['metrics']['score']) >= .9


def condition_for_batch(actor, descriptor, batch):
    if type(batch) is not int or batch <= 0:
        raise ValueError('Positive real current-condition batch required')
    value = actor.materialize(descriptor)
    if value['base_memory'].shape[0] != 1:
        raise ValueError('Exactly one genuine scene, not G copied deployment predictions')
    return dict(value, context=value['base_memory'].expand(batch, -1, -1),
        valid=value['base_valid'].expand(batch, -1), ego=value['base_ego'].expand(batch, -1))


def predict_clean_x(actor, descriptor, state, sigma):
    """Exact head invocation; this always creates a fresh current-Memory graph."""
    raw = condition_for_batch(actor, descriptor, len(state))
    with actor._context_call(raw) as condition:
        prepared = actor._prepare(condition, False)
        with torch.autocast(actor.device.type, dtype=torch.bfloat16):
            value = actor.action_expert(state.detach().float(), sigma.detach().float(), prepared)
    return value.float()


@torch.no_grad()
def probe_nominal(actor, conditions, token, log, service, version):
    started = time.monotonic()
    descriptor = conditions.get(token, actor.device)
    root = torch.randn((1, 8, 3), generator=torch.Generator(device='cpu').manual_seed(
        seed_for('native_qwen_sourceflow_ODE', 2101, token)), dtype=torch.float32).to(actor.device)
    schedule = shift_sigma(torch.linspace(1, 0, 11, dtype=torch.float32), 5.)
    state = root; states = [root.clone()]
    #Use one prepared current B1 Memory for all10 no-grad deployment steps.
    raw = condition_for_batch(actor, descriptor, 1)
    with actor._context_call(raw) as condition:
        prepared = actor._prepare(condition, False)
        with torch.autocast(actor.device.type, dtype=torch.bfloat16):
            for sigma, nxt in zip(schedule[:-1].tolist(), schedule[1:].tolist()):
                time_value = torch.full((1,), sigma, device=actor.device, dtype=torch.float32)
                prediction = actor.action_expert(state, time_value, prepared)
                state = prediction.float() if nxt == 0 else clean_x_step(state, prediction, time_value, nxt, validate=False)
                states.append(state.clone())
    chain = torch.stack(states, 1)
    if chain.shape != (1, 11, 8, 3) or not torch.isfinite(chain).all():
        raise FloatingPointError('Invalid genuineB1 deployment chain')
    requested = ActionCodec.denorm_odo(chain[:, -1]).float().detach().cpu()
    forward_seconds = time.monotonic()-started; started = time.monotonic()
    records = service.score((token, log, requested.numpy()))
    if len(records) != 1:
        raise ValueError('Nominal must actually score exactly one B1 trajectory')
    feedback = validate_feedback(records[0], token, log, requested[0], service.protocol['sha256'])
    value = dict(schema=SCHEMA, protocol=copy.deepcopy(PROTOCOL), token=token, log=log,
        condition=copy.deepcopy(descriptor), source_version=copy.deepcopy(version),
        protocol_sha256=service.protocol['sha256'], chain=chain.detach().cpu(),
        last_state=chain[:, -2].detach().cpu().clone(), last_sigma=schedule[-2:-1].clone(),
        clean=chain[:, -1].detach().cpu().clone(), requested=requested[0],
        feedback=checkpoint.cpu_tree(feedback), root_sha256=digest(root),
        seconds=dict(forward=forward_seconds, scoring=time.monotonic()-started),
        scored_candidates=1, PG_eligible=False)
    return sealed(value, 'material_sha256')


def validate_nominal(value, *, token, log, version, protocol_sha256, condition_identity):
    verify_seal(value, 'material_sha256')
    if (value.get('schema') != SCHEMA or value.get('protocol') != PROTOCOL
            or value.get('token') != token or value.get('log') != log
            or value.get('source_version') != version or value.get('protocol_sha256') != protocol_sha256
            or value.get('scored_candidates') != 1 or value.get('PG_eligible') is not False
            or value.get('condition', {}).get('identity') != condition_identity
            or value['condition'].get('token') != token):
        raise ValueError('Wrong current-run/scene/protocol nominalB1 ancestry')
    for name, shape in [('chain', (1,11,8,3)), ('last_state',(1,8,3)), ('last_sigma',(1,)), ('clean',(1,8,3)), ('requested',(8,3))]:
        x=value[name]
        if not torch.is_tensor(x) or x.shape != shape or x.dtype != torch.float32 or x.requires_grad or not torch.isfinite(x).all():
            raise ValueError('Invalid detached B1 '+name)
    root = torch.randn((1,8,3), generator=torch.Generator().manual_seed(seed_for('native_qwen_sourceflow_ODE',2101,token)), dtype=torch.float32)
    sigma = shift_sigma(torch.linspace(1,0,11,dtype=torch.float32),5.)[-2:-1]
    if (not torch.equal(value['chain'][:,0].cpu(),root) or digest(root)!=value['root_sha256']
            or not torch.equal(value['chain'][:,-2].cpu(),value['last_state'].cpu())
            or not torch.equal(value['chain'][:,-1].cpu(),value['clean'].cpu())
            or not torch.equal(value['last_sigma'].cpu(),sigma)):
        raise ValueError('Saved B1 root/last-state/finalclean-x changed')
    #Scored FP32 physical bits are authoritative. Cross-device element arithmetic
    #can round differently; the bound is not a guarantee of metric invariance.
    decoded=ActionCodec.denorm_odo(value['clean'].cpu())[0];actual=value['requested'].cpu()
    bound=8*torch.finfo(torch.float32).eps*(decoded.abs()+decoded.new_tensor([66.74,42.,3.53]))
    if bool(((decoded-actual).abs()>bound).any()):
        raise ValueError('B1 normalized endpoint is not the actually scored physical action')
    validate_feedback(value['feedback'],token,log,actual,protocol_sha256)
    if any(not math.isfinite(float(x)) or x<0 for x in value['seconds'].values()):
        raise ValueError('B1 actual cost counters changed')
    return value
