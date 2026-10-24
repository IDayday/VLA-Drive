"""Small route hook around the existing source actor, no second PPO platform.

The main owner collects fixed fresh chains using NativeEnvironment and owns
atomic checkpoints/dispatch. This hook only supplies route-specific advantages,
positive rehearsal, and one true optimizer step per source inner epoch.
"""
import math
import time
import torch
from starVLA.rl.flow_grpo.math import centered_group_rewards
from starVLA.rl.simwam_recipe.contracts import seed_for
from starVLA.rl.simwam_recipe.recovery_state import tensor_digest
from starVLA.rl.qwen_flow_exploration.protocol import group_advantages
from .archive import SAFETY
from .objective import ProjectedCostDual, official_costs, cost_scalarized_advantage


ARMS = ('P0', 'P1', 'P2', 'P3', 'P4', 'P5')


class PreservationUpdate:
    def __init__(self, arm, *, seed, archive=None, replay_coefficient=0., budgets=None,
                 reward_scale_floor=.01, dual_lr=.01, dual_initial=1., metrics=SAFETY):
        if arm not in ARMS or not math.isfinite(replay_coefficient) or replay_coefficient < 0 or reward_scale_floor != .01:
            raise ValueError('Explicit common-scale positive-rehearsal P0-P5 contract required')
        self.arm, self.seed = arm, int(seed)
        self.archive = archive
        self.replay_coefficient, self.reward_scale_floor = float(replay_coefficient), float(reward_scale_floor)
        self.metrics = tuple(metrics)
        self.collections = 0
        replay = arm in ('P2', 'P3', 'P5')
        constrained = arm in ('P4', 'P5')
        if replay and (archive is None or replay_coefficient <= 0 or not archive.scenes):
            raise ValueError('Replay arm requires a qualified fixed initial TRAIN archive and locked positive coefficient')
        if not replay and (archive is not None or replay_coefficient != 0):
            raise ValueError('Baseline/control arm cannot silently receive rehearsal')
        if constrained != (budgets is not None):
            raise ValueError('Only P4/P5 consume frozen reference expected costs')
        self.dual = ProjectedCostDual(budgets, learning_rate=dual_lr, initial=dual_initial) if constrained else None
        if self.dual is not None and len(budgets) != len(self.metrics):
            raise ValueError('One independent multiplier per registered cost')
        # Separate RNGs keep replay SCENE exposure equal even though P2 needs
        # no target-mixture draw; FM noise never reuses root rollout noise.
        self.replay_scene_rng = torch.Generator().manual_seed(seed_for('preservation_archive_scene', seed, 'shared_population'))
        self.replay_target_rng = torch.Generator().manual_seed(seed_for('preservation_archive_target', seed, 'shared_population'))
        self.replay_fm_rng = torch.Generator().manual_seed(seed_for('preservation_archive_FM', seed, 'shared_population'))

    @torch.no_grad()
    def prepare(self, rollouts, record_groups, *, collection):
        if collection != self.collections or not rollouts or len(rollouts) != len(record_groups):
            raise ValueError('Prepare exactly once after each complete fresh unfiltered collection')
        group = len(record_groups[0])
        if group < 2 or any(len(rows) != group for rows in record_groups):
            raise ValueError('Scene groups cannot mix G or be filtered by outcomes')
        rewards = torch.stack([rollout['rewards'].detach().cpu().reshape(-1).double() for rollout in rollouts])
        if rewards.shape != (len(rollouts), group) or not rewards.isfinite().all() or (rewards < 0).any() or (rewards > 1).any():
            raise ValueError('True canonical reward matrix [scene,G] in 0-1 units required')
        for index, records in enumerate(record_groups):
            raw = torch.tensor([row['raw_reward'] for row in records], dtype=torch.float64)
            torch.testing.assert_close(rewards[index], raw, atol=1e-7, rtol=1e-6)
        _, raw_group_std, _ = centered_group_rewards(rewards)
        tied = raw_group_std[:, 0] == 0
        common = group_advantages(rewards, floor=self.reward_scale_floor)
        raw_global_std = common['global_std']
        scale = common['scale']
        advantage = common['advantages'].double()
        costs = None
        snapshot = None
        if self.dual is not None:
            costs = torch.stack([official_costs([row['native_subscores'] for row in rows], self.metrics) for rows in record_groups])
            snapshot = self.dual.snapshot()
            advantage = cost_scalarized_advantage(advantage, costs, snapshot, common_scale=scale).double()
            # Costs use the representative unfiltered BxG population; lambda
            # is updated once, while all four inners keep snapshot unchanged.
            self.dual.update(costs.mean((0, 1)), collection)
        advantage = advantage.float().detach()
        active = advantage.ne(0).any(-1) if self.dual is not None else common['mask'].ne(0).any(-1)
        mask = active[:, None].expand_as(advantage).float()
        self.collections += 1
        result = dict(collection=collection, advantages=advantage, mask=mask, reward_global_std=raw_global_std,
                      common_scale=scale, raw_group_std=raw_group_std.cpu(), raw_reward_tied=tied.cpu(),
                      lambda_snapshot=snapshot, cost_means=costs.mean((0,1)) if costs is not None else None,
                      old_chain_identity=tensor_digest([(r['chain'], r['old_logp'], r['rewards']) for r in rollouts]),
                      effective_scene_fraction=float(active.float().mean()),
                      diagnostic_if_historical_adv_clip5=float((advantage.abs() > 5).float().mean()),
                      no_post_scalarization_whitening=True)
        return result

    def update(self, environment, rollouts, prepared, optimizer, *, scheduler=None, inner_epochs=4,
               start_inner=0, after_inner=None):
        if inner_epochs != 4 or not 0 <= start_inner < 4 or prepared['collection'] != self.collections-1:
            raise ValueError('Fixed source inner4 update or exact pending-inner continuation required')
        if environment.actor._loss.adv_clip_max is not None:
            raise ValueError('All P arms share the explicitly registered untruncated final advantage')
        identity = tensor_digest([(r['chain'], r['old_logp'], r['rewards']) for r in rollouts])
        if identity != prepared['old_chain_identity']:
            raise ValueError('Behavior chain/old probability/reward changed before inner updates')
        batch = len(rollouts)
        logs = []
        for inner in range(start_inner, inner_epochs):
            start = time.monotonic()
            current_calls = environment.actor._velocity_calls
            reference_calls = environment.actor._reference_calls
            backwards = 0
            optimizer.zero_grad(set_to_none=True)
            policy_metric, total_loss = {}, 0.
            for scene, rollout in enumerate(rollouts):
                condition = environment.condition(rollout['token'])
                for loss, metrics, weight in environment.policy_chunks(
                        rollout, prepared['advantages'][scene], prepared['mask'][scene], condition):
                    factor = float(weight)/batch
                    (loss*factor).backward()
                    backwards += 1
                    total_loss += float(loss.detach())*factor
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)) or (torch.is_tensor(value) and value.numel() == 1):
                            policy_metric[key] = policy_metric.get(key, 0.)+float(value)*factor
                del condition
            replay_ids, replay_scenes = [], []
            replay_mean = 0.
            untracked_FM_forwards = 0
            if self.archive is not None:
                replay_scenes = self.archive.sample_scenes(batch, generator=self.replay_scene_rng)
                for token in replay_scenes:
                    targets, chosen = self.archive.sample([token], generator=self.replay_target_rng,
                                                          single_best=self.arm == 'P2', device=environment.device)
                    condition = environment.condition(token)
                    before_fm = environment.actor._velocity_calls
                    loss, _ = environment.source_fm_loss(condition, targets, generator=self.replay_fm_rng)
                    delta_fm = environment.actor._velocity_calls-before_fm
                    if delta_fm not in (0, 1):
                        raise ValueError('One source FM velocity call per replay scene required')
                    untracked_FM_forwards += int(delta_fm == 0)
                    (loss*self.replay_coefficient/batch).backward()
                    backwards += 1
                    replay_mean += float(loss.detach())/batch
                    replay_ids.extend(chosen)
                    del condition
                total_loss += self.replay_coefficient*replay_mean
            trainable = [p for p in environment.actor.parameters() if p.requires_grad]
            if any(p.grad is None for p in trainable):
                raise FloatingPointError('Expected native LoRA tensor has no gradient')
            grads = [p.grad for p in trainable]
            if not grads or any(not g.isfinite().all() for g in grads):
                raise FloatingPointError('No/invalid actor gradients; affected batch must stop before optimizer commit')
            grad_norm = math.sqrt(sum(float(g.detach().double().square().sum()) for g in grads))
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            row = dict(collection=prepared['collection'], inner_completed=inner+1, loss=total_loss,
                       gradient_norm=grad_norm, policy_metrics=policy_metric, replay_loss=replay_mean,
                       replay_coefficient=self.replay_coefficient, replay_scenes=replay_scenes,
                       replay_candidate_ids=replay_ids, supervised_scene_exposures=len(replay_scenes),
                       seconds=time.monotonic()-start, true_optimizer_steps=1,
                       velocity_forward_current=environment.actor._velocity_calls-current_calls+untracked_FM_forwards,
                       velocity_forward_reference=environment.actor._reference_calls-reference_calls,
                       backward_calls=backwards,
                       effective_PG_scenes=int(prepared['mask'].any(-1).sum()))
            logs.append(row)
            if tensor_digest([(r['chain'], r['old_logp'], r['rewards']) for r in rollouts]) != identity:
                raise ValueError('Saved behavior material mutated during inner updates')
            if after_inner is not None:
                after_inner(inner+1, row, self.state_dict())
        return logs

    def state_dict(self):
        return dict(arm=self.arm, seed=self.seed, collections=self.collections, metrics=self.metrics,
                    replay_coefficient=self.replay_coefficient, reward_scale_floor=self.reward_scale_floor,
                    archive_hash=self.archive.digest() if self.archive is not None else None,
                    dual=self.dual.state_dict() if self.dual is not None else None,
                    replay_scene_rng=self.replay_scene_rng.get_state(), replay_target_rng=self.replay_target_rng.get_state(),
                    replay_fm_rng=self.replay_fm_rng.get_state())

    def load_state_dict(self, state):
        expected = self.state_dict()
        for key in ('arm','seed','metrics','replay_coefficient','reward_scale_floor','archive_hash'):
            if state[key] != expected[key]:
                raise ValueError('Preservation resume changed '+key)
        self.collections = int(state['collections'])
        if self.dual is not None:
            self.dual.load_state_dict(state['dual'])
            if self.dual.collections != self.collections:
                raise ValueError('Dual/raw fresh collection progress mismatch')
        elif state['dual'] is not None:
            raise ValueError('Baseline cannot acquire a dual on resume')
        for name in ('replay_scene_rng','replay_target_rng','replay_fm_rng'):
            getattr(self,name).set_state(state[name].cpu())
