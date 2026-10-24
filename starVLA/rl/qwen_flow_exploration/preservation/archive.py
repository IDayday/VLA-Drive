"""Fixed TRAIN-only success archive, shared eligibility for P2/P3/P5.

The planned/requested trajectory is the supervised label. Simulated states are
feedback only. No old-policy probability is created for archived trajectories.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Mapping, Sequence
import torch
from starVLA.model.modules.qwen_sourceflow.action_codec import ActionCodec


SAFETY = ('no_at_fault_collisions', 'drivable_area_compliance', 'time_to_collision_within_bound')


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class ArchiveConfig:
    capacity: int = 4
    minimum_reward: float = .9
    near_best_tolerance: float = .01
    duplicate_ADE_m: float = .15
    duplicate_heading_rms_rad: float = .03
    reward_temperature: float = .1
    feasibility_metrics: tuple = SAFETY

    def __post_init__(self):
        if not (1 <= self.capacity <= 4 and 0 <= self.minimum_reward <= 1 and
                0 <= self.near_best_tolerance <= 1 and self.duplicate_ADE_m > 0 and
                self.duplicate_heading_rms_rad > 0 and self.reward_temperature > 0):
            raise ValueError('Invalid preregistered fixed archive settings')
        if not self.feasibility_metrics or any(k not in SAFETY for k in self.feasibility_metrics):
            raise ValueError('Archive eligibility must use explicit official safety fields')


def trajectory_distance(a, b):
    a, b = torch.as_tensor(a, dtype=torch.float64), torch.as_tensor(b, dtype=torch.float64)
    if a.shape != (8, 3) or b.shape != (8, 3) or not (a.isfinite().all() and b.isfinite().all()):
        raise ValueError('Native physical requested 8x3 trajectories required')
    ade = torch.linalg.vector_norm(a[:, :2]-b[:, :2], dim=-1).mean()
    heading = a[:, 2]-b[:, 2]
    heading = torch.atan2(torch.sin(heading), torch.cos(heading))
    return float(ade), float(heading.square().mean().sqrt())


def _validated(row, train_tokens, token_logs, config):
    token = str(row['scene_token'])
    if row['split'] not in ('TRAIN', 'RL_TRAIN') or token not in train_tokens or row['log_id'] != token_logs[token]:
        raise ValueError('Archive cannot consume DEV/test or misassociated logs')
    if row.get('status', 'OK') != 'OK':
        raise ValueError('Archive input must be successfully scored; no failure-as-zero')
    if str(row['proposal_source']).lower() in ('gt', 'ground_truth', 'expert_gt'):
        raise ValueError('Initial archive must come from the shared policy candidate pool')
    reward = float(row['raw_reward'])
    metrics = row['native_subscores']
    if not math.isfinite(reward) or not 0 <= reward <= 1:
        raise ValueError('PDMS must remain finite in the original 0-1 units')
    if any(not math.isfinite(float(metrics[k])) or not 0 <= float(metrics[k]) <= 1 for k in config.feasibility_metrics):
        raise ValueError('Missing/nonfinite official safety feedback')
    physical = torch.as_tensor(row['planned_trajectory'], dtype=torch.float64)
    if physical.shape != (8, 3) or not physical.isfinite().all():
        raise ValueError('Supervision must use the finite requested trajectory, not simulation states')
    normalized = ActionCodec.norm_odo(physical).float()
    if 'normalized_action' in row:
        original = torch.as_tensor(row['normalized_action'], dtype=torch.float32)
        if original.shape != (8, 3) or not original.isfinite().all() or not torch.allclose(original, normalized, atol=2e-5, rtol=1e-5):
            raise ValueError('Requested physical trajectory and original native action encoding disagree')
        normalized = original
    # Store only the information needed for fixed supervision and provenance.
    entry = dict(candidate_id=str(row['candidate_id']), scene_token=token, log_id=str(row['log_id']),
                 reward=reward, planned_trajectory=physical.tolist(), normalized_action=normalized.tolist(),
                 scorer_config_hash=str(row['scorer_config_hash']), checkpoint_id=str(row['checkpoint_id']),
                 proposal_source=str(row['proposal_source']), on_policy_probability=None)
    return entry, reward >= config.minimum_reward and all(float(metrics[k]) == 1 for k in config.feasibility_metrics)


class FrozenArchive:
    def __init__(self, entries, config, provenance):
        self.config = config
        self.entries = entries
        self.provenance = provenance
        self.scenes = sorted(entries)
        self._initial_hash = self.digest()

    @classmethod
    def build(cls, records: Sequence[Mapping], *, train_tokens, token_logs,
              config=ArchiveConfig(), reference_candidate_ids=None):
        train_tokens = set(train_tokens)
        refs = reference_candidate_ids or {}
        grouped, rejected = {}, dict(below_quality_or_infeasible=0, duplicate=0)
        ids = set()
        protocol, checkpoint = set(), set()
        for row in records:
            entry, valid = _validated(row, train_tokens, token_logs, config)
            key = (entry['scene_token'], entry['candidate_id'])
            if key in ids:
                raise ValueError('Duplicate candidate identity in fixed shared pool')
            ids.add(key);protocol.add(entry['scorer_config_hash']);checkpoint.add(entry['checkpoint_id'])
            grouped.setdefault(entry['scene_token'], [])
            if valid:
                grouped[entry['scene_token']].append(entry)
            else:
                rejected['below_quality_or_infeasible'] += 1
        if len(protocol) > 1 or len(checkpoint) > 1:
            raise ValueError('Fixed initial archive must share one scorer and base policy identity')
        result = {}
        for token, candidates in grouped.items():
            candidates.sort(key=lambda x: (-x['reward'], x['candidate_id']))
            if not candidates:
                continue  # No manufactured success target or GT substitution.
            best = candidates[0]
            kept = [best]
            ref = next((r for r in candidates if r['candidate_id'] == refs.get(token)), None)
            ordered = ([ref] if ref is not None and ref != best else [])
            ordered += [r for r in candidates[1:] if r != ref and r['reward'] >= best['reward']-config.near_best_tolerance]
            while ordered and len(kept) < config.capacity:
                # A registered reference has first priority. Otherwise use
                # deterministic farthest-first coverage in physical units.
                if ref in ordered:
                    chosen = ordered.pop(ordered.index(ref))
                else:
                    def distance(row):
                        return min(max(trajectory_distance(row['planned_trajectory'], k['planned_trajectory'])[0]/config.duplicate_ADE_m,
                                       trajectory_distance(row['planned_trajectory'], k['planned_trajectory'])[1]/config.duplicate_heading_rms_rad) for k in kept)
                    chosen = max(ordered, key=lambda row: (distance(row), row['reward'], row['candidate_id']))
                    ordered.remove(chosen)
                if any(all(a <= threshold for a, threshold in zip(trajectory_distance(chosen['planned_trajectory'], k['planned_trajectory']),
                                                                 (config.duplicate_ADE_m, config.duplicate_heading_rms_rad))) for k in kept):
                    rejected['duplicate'] += 1
                    continue
                kept.append(chosen)
            result[token] = kept
        provenance = dict(schema='native_fixed_success_archive_v1', input_pool_hash=fingerprint(list(records)),
                          input_candidates=len(records), input_scenes=len(grouped), covered_scenes=len(result),
                          stored_candidates=sum(map(len, result.values())), rejected=rejected,
                          scorer_config_hash=next(iter(protocol), None), checkpoint_id=next(iter(checkpoint), None),
                          labels='requested physical trajectory reencoded with native source codec', dynamic_updates=False)
        return cls(result, config, provenance)

    def digest(self):
        return fingerprint(dict(config=asdict(self.config), entries=self.entries, provenance=self.provenance))

    def assert_fixed(self):
        if self.digest() != self._initial_hash:
            raise ValueError('Initial archive changed during the registered comparison')

    def sample(self, tokens, *, generator, single_best=False, device='cpu'):
        self.assert_fixed()
        targets, selected = [], []
        for token in tokens:
            values = self.entries[token]
            if single_best:
                index = 0
            else:
                weights = torch.softmax(torch.tensor([r['reward'] for r in values], dtype=torch.float64)/self.config.reward_temperature, 0)
                index = int(torch.multinomial(weights, 1, generator=generator))
            targets.append(values[index]['normalized_action'])
            selected.append(values[index]['candidate_id'])
        return torch.tensor(targets, dtype=torch.float32, device=device), selected

    def sample_scenes(self, batch_size, *, generator):
        if not self.scenes or batch_size < 1:
            raise ValueError('No eligible TRAIN replay scene; do not fabricate a successful archive')
        indices = torch.randint(len(self.scenes), (batch_size,), generator=generator).tolist()
        return [self.scenes[i] for i in indices]

    def state_dict(self):
        self.assert_fixed()
        return dict(config=asdict(self.config), provenance=self.provenance, entries=self.entries, sha256=self._initial_hash)

    @classmethod
    def from_state_dict(cls, state):
        config = dict(state['config']);config['feasibility_metrics'] = tuple(config['feasibility_metrics'])
        archive = cls(state['entries'], ArchiveConfig(**config), state['provenance'])
        if archive.digest() != state['sha256']:
            raise ValueError('Fixed archive payload hash changed')
        return archive
