"""Opt-in rank-local Safety-credit curriculum, with complete native state.

This is an online signal-priority heuristic, not an estimator of expected
learning progress. It changes scene allocation only. No evaluation list,
teacher trajectory, model feature, optimizer or global RNG enters the bank.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
import random

SCHEMA = 'world_online_Safety_credit_priority_stream_v1'
DEFAULT_CONFIG = dict(schema=SCHEMA, enabled=True, warmup_collections=8,
    priority_every_collections=4, uniform_floor=.75, capacity=256,
    max_priority_repeats=3, cooldown_collections=8, max_age_collections=128,
    min_rawA_variance=1e-6, min_EP_headroom=1e-4, priority_clip=1., seed=42)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def validate_config(config):
    c = copy.deepcopy(dict(config))
    if set(c) != set(DEFAULT_CONFIG) or c['schema'] != SCHEMA or c['enabled'] is not True:
        raise ValueError('Explicit complete online curriculum config required')
    if (type(c['warmup_collections']) is not int or c['warmup_collections'] not in (0, 4, 8, 12, 16)
            or c['priority_every_collections'] != 4 or c['uniform_floor'] != .75 or c['seed'] != 42):
        raise ValueError('Curriculum requires <=16 warmup, 3uniform/1priority, seed42')
    for key in ('capacity', 'max_priority_repeats', 'cooldown_collections', 'max_age_collections'):
        if type(c[key]) is not int or c[key] <= 0:
            raise ValueError('Positive bounded curriculum setting: ' + key)
    for key in ('min_rawA_variance', 'min_EP_headroom', 'priority_clip'):
        if not math.isfinite(c[key]) or c[key] <= 0:
            raise ValueError('Positive finite curriculum setting: ' + key)
    return c


def _tolist(value):
    if hasattr(value, 'detach'):
        return value.detach().cpu().tolist()
    return value.tolist() if hasattr(value, 'tolist') else value


def signal_statistics(rollout, config):
    """Use actual structured rawA, not canonical reward/Comfort variance."""
    s = rollout['structured_advantage']
    components, raw = _tolist(s['components']), _tolist(s['A_raw'])
    if len(components) == 1 and isinstance(components[0][0], list):
        components, raw = components[0], raw[0]
    g = len(raw)
    if g not in (8, 16) or len(components) != g:
        raise ValueError('Curriculum expects actual G8/G16 Safety material')
    if any(len(v) != 4 or any(not math.isfinite(x) or not 0 <= x <= 1 for x in v) for v in components):
        raise ValueError('Invalid saved Safety components')
    if any(not math.isfinite(x) for x in raw):
        raise ValueError('Invalid saved Safety rawA')
    feasible = [all(x == 1 for x in v[:3]) for v in components]
    count = sum(feasible)
    safe_ep = [v[3] for v, f in zip(components, feasible) if f]
    ep_mean = sum(safe_ep)/count if count else None
    mean = sum(raw)/g
    variance = sum((x-mean)**2 for x in raw)/g
    saturated = count == g and all(v[3] == 1 for v in components)
    mixed = 0 < count < g
    progress_room = count == g and 1-ep_mean >= config['min_EP_headroom']
    eligible = variance >= config['min_rawA_variance'] and (mixed or progress_room)
    weight = variance*(1+4*(count/g)*(1-count/g)) if mixed else variance*max(0., 1-(ep_mean or 0.))
    return dict(group_size=g, rawA_variance=variance, feasible_count=count,
                feasible_fraction=count/g, feasible_EP_mean=ep_mean, saturated=saturated,
                mixed=mixed, eligible=eligible,
                weight=min(config['priority_clip'], weight) if eligible else 0.,
                reason='eligible' if eligible else 'success_saturated' if saturated else
                       'no_success_witness' if not count else 'no_material_credit_variance')


class OnlinePriorityStream:
    def __init__(self, base_stream, config):
        self.base = base_stream
        self.config = validate_config(config)
        if base_stream.mode != 'uniform':
            raise ValueError('Online curriculum must wrap immutable uniform SceneStream')
        self.world = base_stream.world
        self.tokens, self.logs, self.sha = base_stream.tokens, base_stream.logs, base_stream.sha
        self.legal = set(self.tokens)
        self.mode = 'online_priority'
        self.identity = dict(schema=SCHEMA, base_stream=copy.deepcopy(base_stream.identity),
            config=copy.deepcopy(self.config), config_sha256=fingerprint(self.config),
            uniform_order='base.at(uniform_draws*world,rank); priority never consumes uniform position',
            allocation='rank-local bounded online bank; TRAIN rollout feedback only; no evaluation bank',
            rawA_priority='variance*(1+4p(1-p)) for mixed; variance*(1-mean_feasible_EP) for allfeasible',
            allfail='not automatically prioritized without a success witness; uniform exploration remains',
            learning_progress='not claimed; repeat feedback deltas are stochastic observations only',
            RNG='private Python Random(seed42+1000003*rank), full getstate stored',
            importance_correction=False)
        self.state = None

    def bind(self, state):
        if 'curriculum' not in state:
            if state.get('collections', 0) or state.get('scene_cursor', 0):
                raise ValueError('Nonfresh resume lacks complete curriculum state')
            state['curriculum'] = dict(schema=SCHEMA, identity_sha256=fingerprint(self.identity),
                rank=None, uniform_draws=0, committed_collections=0, rng_state=None,
                bank={}, priority_counts={}, last_priority_collection={},
                seen_uniform=set(), seen_priority=set(), selection_plan=None,
                stats=dict(total_exposures=0, uniform_exposures=0, priority_exposures=0,
                    priority_slots=0, priority_fallbacks=0, repeated_priority_exposures=0,
                    saturated_observations=0, allfail_observations=0, admissions=0,
                    exits_saturated=0, exits_uninformative=0, exits_expired=0,
                    exits_capacity=0, exits_repeat_cap=0, commit_calls=0))
        self.state = state
        self.validate(state)
        return self

    @property
    def current(self):
        if self.state is None:
            raise RuntimeError('Bind native training state before online stream use')
        return self.state['curriculum']

    def _prune(self, collection):
        c = self.current
        for token, entry in list(c['bank'].items()):
            if collection-entry['last_collection'] > self.config['max_age_collections']:
                del c['bank'][token]
                c['stats']['exits_expired'] += 1

    def at(self, global_cursor, rank):
        c = self.current
        if (global_cursor < 0 or global_cursor % self.world or not 0 <= rank < self.world
                or global_cursor != c['committed_collections']*self.world):
            raise ValueError('Curriculum cursor/rank differs from native boundary')
        if c['rank'] is None:
            c['rank'] = rank
            c['rng_state'] = random.Random(self.config['seed']+1000003*rank).getstate()
        if c['rank'] != rank:
            raise ValueError('Rank-local curriculum state used by another rank')
        plan = c['selection_plan']
        if plan and plan['cursor'] == global_cursor:
            if plan['status'] != 'prepared':
                raise ValueError('Committed selection cannot be collected twice')
            return plan['token'], copy.deepcopy(plan['position'])
        collection = c['committed_collections']+1
        self._prune(collection)
        slot = collection > self.config['warmup_collections'] and collection % 4 == 0
        candidates = [t for t, e in c['bank'].items()
            if c['priority_counts'].get(t, 0) < self.config['max_priority_repeats']
            and collection-c['last_priority_collection'].get(t, -10**12) >= self.config['cooldown_collections']]
        source = 'uniform'
        if slot and candidates:
            candidates.sort()
            rng = random.Random()
            rng.setstate(c['rng_state'])
            draw = rng.random()*sum(c['bank'][t]['weight'] for t in candidates)
            token = candidates[-1]
            for candidate in candidates:
                draw -= c['bank'][candidate]['weight']
                if draw <= 0:
                    token = candidate
                    break
            c['rng_state'] = rng.getstate()
            position = copy.deepcopy(c['bank'][token]['uniform_position'])
            source = 'priority'
        else:
            token, position = self.base.at(c['uniform_draws']*self.world, rank)
        selection_id = fingerprint(dict(schema=SCHEMA, stream_sha256=self.sha, rank=rank,
            cursor=global_cursor, token=token, source=source, uniform_draws=c['uniform_draws']))
        position.update(curriculum_selection_id=selection_id, selection_source=source,
                        curriculum_cursor=global_cursor, uniform_draws=c['uniform_draws'])
        c['selection_plan'] = dict(cursor=global_cursor, rank=rank, collection=collection,
            token=token, source=source, priority_slot=slot, id=selection_id,
            position=copy.deepcopy(position), status='prepared')
        return token, copy.deepcopy(position)

    def commit(self, rollout, state):
        if state is not self.state:
            raise ValueError('Curriculum committed to unbound native state')
        c = self.current
        plan = c['selection_plan']
        if (not plan or rollout['token'] != plan['token']
                or rollout['stream_position'].get('curriculum_selection_id') != plan['id']):
            raise ValueError('Rollout does not match saved curriculum selection')
        if (state['scene_cursor'] != plan['cursor']+self.world
                or state['collections'] != plan['collection']):
            raise ValueError('Commit hook requires newly advanced native collection/cursor')
        if plan['status'] == 'committed':
            return  # exact idempotent native callback, never repeat bank/RNG updates
        info = signal_statistics(rollout, self.config)
        token, collection = plan['token'], plan['collection']
        stats = c['stats']
        stats['total_exposures'] += 1
        stats['commit_calls'] += 1
        stats['priority_slots'] += int(plan['priority_slot'])
        if plan['source'] == 'priority':
            stats['priority_exposures'] += 1
            stats['repeated_priority_exposures'] += int(token in c['seen_priority'])
            c['seen_priority'].add(token)
            c['priority_counts'][token] = c['priority_counts'].get(token, 0)+1
            c['last_priority_collection'][token] = collection
        else:
            c['uniform_draws'] += 1
            stats['uniform_exposures'] += 1
            stats['priority_fallbacks'] += int(plan['priority_slot'])
            c['seen_uniform'].add(token)
        stats['saturated_observations'] += int(info['saturated'])
        stats['allfail_observations'] += int(info['feasible_count'] == 0)
        previous = c['bank'].get(token)
        if c['priority_counts'].get(token, 0) >= self.config['max_priority_repeats']:
            if previous is not None:
                del c['bank'][token]
                stats['exits_repeat_cap'] += 1
        elif info['eligible']:
            uniform_position = copy.deepcopy(previous['uniform_position'] if previous else plan['position'])
            for key in ('curriculum_selection_id', 'selection_source', 'curriculum_cursor', 'uniform_draws'):
                uniform_position.pop(key, None)
            entry = dict(info, token=token, owner_rank=c['rank'], last_collection=collection,
                observations=previous['observations']+1 if previous else 1,
                uniform_position=uniform_position,
                stochastic_feedback_delta=dict(feasible_fraction=info['feasible_fraction']-previous['feasible_fraction'],
                    feasible_EP_mean=info['feasible_EP_mean']-previous['feasible_EP_mean']) if previous else None)
            c['bank'][token] = entry
            stats['admissions'] += int(previous is None)
        elif previous is not None:
            del c['bank'][token]
            stats['exits_saturated' if info['saturated'] else 'exits_uninformative'] += 1
        while len(c['bank']) > self.config['capacity']:
            discard = min(c['bank'], key=lambda t:(c['bank'][t]['weight'], c['bank'][t]['last_collection'], t))
            del c['bank'][discard]
            stats['exits_capacity'] += 1
        c['committed_collections'] = collection
        plan.update(status='committed', signal=info)
        self.validate(state, rollout)

    def validate(self, state, rollout=None):
        c = state['curriculum']
        if c['schema'] != SCHEMA or c['identity_sha256'] != fingerprint(self.identity):
            raise ValueError('Curriculum identity/config changed across native recovery')
        if (state.get('collections', 0) != c['committed_collections']
                or state.get('scene_cursor', 0) != c['committed_collections']*self.world
                or c['uniform_draws']+c['stats']['priority_exposures'] != c['committed_collections']
                or c['stats']['total_exposures'] != c['committed_collections']):
            raise ValueError('Curriculum/native counters disagree')
        if len(c['bank']) > self.config['capacity'] or not set(c['bank']) <= self.legal:
            raise ValueError('Curriculum pool overflow or foreign/evaluation scene')
        if not set(c['priority_counts']) <= self.legal or any(not 0 < n <= self.config['max_priority_repeats'] for n in c['priority_counts'].values()):
            raise ValueError('Lifetime priority cap or scene eligibility changed')
        if sum(c['priority_counts'].values()) != c['stats']['priority_exposures']:
            raise ValueError('Priority lifetime exposure receipt changed')
        for token, e in c['bank'].items():
            if (e['owner_rank'] != c['rank'] or e['token'] != token or not e['eligible']
                    or not math.isfinite(e['weight']) or e['weight'] <= 0
                    or c['priority_counts'].get(token, 0) >= self.config['max_priority_repeats']):
                raise ValueError('Invalid rank-local curriculum entry')
        if c['stats']['uniform_exposures'] != c['uniform_draws'] or len(c['seen_uniform']) > c['uniform_draws']:
            raise ValueError('Uniform coverage counters disagree')
        if c['stats']['priority_exposures']*4 > c['committed_collections']:
            raise ValueError('75percent uniform floor violated')
        plan = c['selection_plan']
        if plan is not None:
            expected = fingerprint(dict(schema=SCHEMA, stream_sha256=self.sha,
                rank=plan['rank'], cursor=plan['cursor'], token=plan['token'],
                source=plan['source'], uniform_draws=plan['position']['uniform_draws']))
            committed = plan['status'] == 'committed'
            if (plan['status'] not in ('prepared', 'committed')
                    or plan['token'] not in self.legal or plan['rank'] != c['rank']
                    or plan['source'] not in ('uniform', 'priority') or plan['id'] != expected
                    or plan['position']['curriculum_selection_id'] != expected
                    or plan['cursor'] != (c['committed_collections']-int(committed))*self.world
                    or plan['collection'] != c['committed_collections']+int(not committed)):
                raise ValueError('Native curriculum selection plan changed')
        if rollout is not None:
            if (not plan or plan['status'] != 'committed' or plan['token'] != rollout['token']
                    or plan['collection'] != c['committed_collections']
                    or rollout['stream_position'].get('curriculum_selection_id') != plan['id']):
                raise ValueError('Pending native rollout/curriculum selection changed')
        if c['rng_state'] is not None:
            private = random.Random()
            private.setstate(c['rng_state'])
        return True

    def summary(self):
        c = self.current
        distinct = len(c['seen_uniform'] | c['seen_priority'])
        return dict(schema=SCHEMA, rank=c['rank'], bank_size=len(c['bank']),
            committed_collections=c['committed_collections'], uniform_draws=c['uniform_draws'],
            distinct_uniform_scenes=len(c['seen_uniform']), distinct_priority_scenes=len(c['seen_priority']),
            distinct_seen_scenes=distinct,
            repeat_scene_exposures=c['stats']['total_exposures']-distinct,
            priority_replay_exposures=c['stats']['priority_exposures'],
            stats=copy.deepcopy(c['stats']), selection_plan=copy.deepcopy(c['selection_plan']))


class CurriculumRuntimeMixin:
    """Compose before GroupRuntime; caller keeps original PG/G/noise/reference."""
    def make_scene_stream(self, cfg, args, world):
        base = super().make_scene_stream(cfg, args, world)
        if cfg['research']['data_policy'] == 'online_priority':
            self.stream = OnlinePriorityStream(base, cfg['research']['curriculum'])
        return self.stream

    def bind_training_state(self, state, rollout, identity):
        super().bind_training_state(state, rollout, identity)
        if isinstance(self.stream, OnlinePriorityStream):
            self.stream.bind(state)
            self.stream.validate(state, rollout)

    def on_collection_committed(self, rollout, state):
        super().on_collection_committed(rollout, state)
        if isinstance(self.stream, OnlinePriorityStream):
            self.stream.commit(rollout, state)

    def validate_boundary(self, state, rollout, *args, **kwargs):
        value = super().validate_boundary(state, rollout, *args, **kwargs)
        if isinstance(self.stream, OnlinePriorityStream):
            self.stream.validate(state, rollout)
            c = state['curriculum']
            value['curriculum_boundary'] = dict(schema=SCHEMA,
                identity_sha256=c['identity_sha256'], rank=c['rank'],
                committed_collections=c['committed_collections'],
                uniform_draws=c['uniform_draws'], priority_exposures=c['stats']['priority_exposures'])
        return value

    def validate_ranks(self, summaries):
        value = super().validate_ranks(summaries)
        if isinstance(self.stream, OnlinePriorityStream):
            expected = fingerprint(self.stream.identity)
            for rank, summary in enumerate(summaries):
                c = summary.get('curriculum_boundary', {})
                if (c.get('identity_sha256') != expected or c.get('schema') != SCHEMA
                        or c.get('committed_collections') != summary['collections']
                        or c.get('rank') not in ([None, rank] if summary['collections'] == 0 else [rank])
                        or c['uniform_draws']+c['priority_exposures'] != c['committed_collections']
                        or c['priority_exposures']*4 > c['committed_collections']):
                    raise ValueError('Cross-rank curriculum identity/residue/coverage mismatch')
        return value

    def public_training_state(self, state):
        value = super().public_training_state(state)
        if isinstance(self.stream, OnlinePriorityStream) and 'curriculum' in state:
            try:
                value['curriculum'] = self.stream.summary()
            except Exception as exc:
                # Failure reporting must not hide the original strict native error.
                value['curriculum'] = dict(schema=SCHEMA, summary_error=type(exc).__name__)
        return value

    def source_files(self):
        files = super().source_files()
        name = 'starVLA/rl/qwen_world_safety_exploration/curriculum.py'
        files[name] = hashlib.sha256(Path(name).read_bytes()).hexdigest()
        return dict(sorted(files.items()))
