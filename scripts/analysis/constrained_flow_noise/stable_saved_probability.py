"""Bounded B1 recomputation for transitions exceeding the existing layout guard.

This is a numerical execution revision, not an old-logprob correction. The
real sampled chain and every recorded old density remain immutable. Selection
is made once before the first inner update, without reward/advantage inputs,
then remains fixed through both inner epochs and native interruption recovery.
"""
import copy

import torch

from starVLA.rl.constrained_flow_noise.contracts import fingerprint
from starVLA.rl.constrained_flow_noise.transition import density_views


TOLERANCE = 1e-4
MAX_FALLBACK_TRANSITIONS = 8


def replace_transitions(actor, raw, chain, flat, indices):
    """Differentiate the actual original B1 kernel at selected saved states."""
    if not indices:
        return flat
    if len(indices) > MAX_FALLBACK_TRANSITIONS:
        raise ValueError('Too many layout fallbacks; stop instead of unbounded memory use')
    b, g, kp, h, d = chain.shape
    k = kp - 1
    if b != 1 or (g, k, h, d) != (16, 10, 8, 4):
        raise ValueError('Qualified fallback requires the original one-scene G16/K10 layout')
    if sorted(set(indices)) != indices or min(indices) < 0 or max(indices) >= b*g*k:
        raise ValueError('Invalid transition indices')
    x = chain[:, :, :-1].detach().reshape(b*g*k, h, d)
    xn = chain[:, :, 1:].detach().reshape_as(x)
    # Same B1 projection as rollout; never cache a trainable projection result.
    condition = actor.condition(raw)
    values = {name: [] for name in ('mean', 'std', 'elementwise_logp')}
    for index in indices:
        step = torch.tensor([index % k], device=x.device)
        buckets = torch.tensor([int((index % k)/k*actor.policy.action_model.num_timestep_buckets)], device=x.device)
        transition = actor.distribution(x[index:index+1], buckets, condition, step)
        values['mean'].append(transition.mean)
        values['std'].append(transition.std.expand_as(x[index:index+1]))
        values['elementwise_logp'].append(transition.logprob(xn[index:index+1]))
    selected = torch.tensor(indices, device=x.device, dtype=torch.long)
    result = {}
    for name, replacement in values.items():
        result[name] = flat[name].reshape(b*g*k, h, d).index_copy(
            0, selected, torch.cat(replacement)).reshape(b, g, k, h, d)
    return dict(mean=result['mean'], std=result['std'], **density_views(result['elementwise_logp']))


class StableSavedProbability:
    def __init__(self):
        self.collection = None
        self.entries = {}
        self.slot = None

    def bind(self, behavior_identity, slot):
        identity = fingerprint(behavior_identity)
        if identity != self.collection:
            if any(entry['calls'] != 2 for entry in self.entries.values()):
                raise ValueError('Cannot discard layout choices before both inner epochs finish')
            self.collection, self.entries = identity, {}
        self.slot = str(slot)

    def state_dict(self):
        return copy.deepcopy(dict(schema='cfn_saved_transition_layout_v1',
                                  collection=self.collection, entries=self.entries))

    def load_state_dict(self, state, *, pending_inner):
        if state is None:
            if pending_inner:
                raise ValueError('Cannot infer layout choices after an inner update')
            return
        if state.get('schema') != 'cfn_saved_transition_layout_v1':
            raise ValueError('Unknown numerical layout state')
        if pending_inner and (not state['entries'] or any(e['calls'] != 1 for e in state['entries'].values())):
            raise ValueError('Incomplete first-inner layout decisions')
        self.collection = state['collection']
        self.entries = copy.deepcopy(state['entries'])
        self.slot = None

    def statistics(self, actor, original, pg):
        if self.collection is None or self.slot is None:
            raise ValueError('Missing immutable behavior/slot identity')
        flat = original(pg['raw'], pg['chain'])
        old = pg['old_elementwise_logp'].detach().mean((-2, -1))
        fresh = self.slot not in self.entries
        if fresh:
            error = ((flat['mean_transition_logp'].detach() - old).exp() - 1).abs()
            if not torch.isfinite(error).all():
                raise FloatingPointError('Nonfinite fresh probability; no layout fallback')
            indices = (error.reshape(-1) > TOLERANCE).nonzero().flatten().tolist()
            entry = dict(scene_key=pg['scene_key'], indices=indices, calls=0,
                         flat_fresh_ratio_error=float(error.max()))
        else:
            entry = self.entries[self.slot]
            if entry['scene_key'] != pg['scene_key'] or entry['calls'] != 1:
                raise ValueError('Layout selection crossed a scene or inner-epoch boundary')
        result = replace_transitions(actor, pg['raw'], pg['chain'], flat, entry['indices'])
        if fresh:
            error = ((result['mean_transition_logp'].detach() - old).exp() - 1).abs()
            if not torch.isfinite(error).all() or float(error.max()) > TOLERANCE:
                raise ValueError('Original B1 layout still violates the unchanged probability guard')
            entry['corrected_fresh_ratio_error'] = float(error.max())
            self.entries[self.slot] = entry
        entry['calls'] += 1
        return result, entry

    def objective(self, native_objective, actor, **arguments):
        if arguments.get('chunk') is not None:
            raise ValueError('Explicit chunk cannot be combined with automatic numerical layout')
        pg = arguments['pg']
        original = actor.saved_statistics
        calls = 0
        entry = None
        def statistics(raw, chain, *, chunk=None):
            nonlocal calls, entry
            calls += 1
            if calls == 1:
                if raw is not pg['raw'] or chain is not pg['chain'] or chunk is not None:
                    raise ValueError('Native objective changed its PG recomputation interface')
                value, entry = self.statistics(actor, original, pg)
                return value
            return original(raw, chain, chunk=chunk)
        actor.saved_statistics = statistics
        try:
            loss, metrics = native_objective(actor, **arguments)
        finally:
            del actor.saved_statistics
        metrics.update(layout_fallback_transitions=loss.detach().new_tensor(len(entry['indices'])),
                       layout_flat_fresh_ratio_error=loss.detach().new_tensor(entry['flat_fresh_ratio_error']),
                       layout_corrected_fresh_ratio_error=loss.detach().new_tensor(entry['corrected_fresh_ratio_error']))
        return loss, metrics
