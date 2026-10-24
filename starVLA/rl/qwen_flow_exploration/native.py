"""Small study adapter around the unchanged native model/source loss/scorer.

The mature original trainer remains intact. Only the research population,
collection size and independently named objectives live in this namespace.
"""
from collections import OrderedDict
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch
from starVLA.rl.qwen_sourceflow.contracts import configure_native_numerics, fingerprint
from starVLA.rl.qwen_sourceflow.export import load_standalone
from starVLA.rl.qwen_sourceflow.data import RawSceneData
from starVLA.rl.qwen_sourceflow.actor import NativeRecipeActor
from starVLA.rl.qwen_sourceflow.recipe import expected_recipe
from starVLA.model.modules.qwen_sourceflow.action_codec import ActionCodec
from starVLA.model.modules.qwen_sourceflow.cached_action_runner import expand_condition, condition_fp32
from starVLA.rl.simwam_recipe.reward_readouts import RewardService
from scripts.analysis.qwen_sourceflow.evaluate_policy import evaluation_caches, json_feedback
from .protocol import read_protocol, require_dispatch, require_scene, Ledger


def move(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move(item, device) for item in value]
    return value


def tensor_bytes(value):
    if torch.is_tensor(value):
        return value.numel()*value.element_size()
    if isinstance(value, dict):
        return sum(map(tensor_bytes, value.values()))
    if isinstance(value, list):
        return sum(map(tensor_bytes, value))
    return 0


class NativeEnvironment:
    def __init__(self, protocol=None, device='cuda', trainable='frozen', scorer_workers=1,
                 ledger_path=None, experiment_id='qualification', cache_scenes=32):
        require_dispatch(); configure_native_numerics(); torch.set_num_threads(4)
        self.protocol = read_protocol(protocol) if not isinstance(protocol, dict) else protocol
        self.device = torch.device(device)
        self.experiment_id = experiment_id
        self.behavior_identity = dict(base='initial', version=0)
        self.ledger = Ledger(ledger_path)
        self.codec = ActionCodec
        self.model, self.processor, self.base = load_standalone(self.protocol['base_export'], device=self.device)
        if self.base['model_sha256'] != self.protocol['base_model_sha256']:
            raise ValueError('Wrong common complete base')
        self.model.requires_grad_(False).eval()
        self.actor = None
        if trainable == 'lora':
            cfg = expected_recipe('R_CANONICAL')
            cfg['grpo']['sample']['group_size'] = self.protocol['screen']['G']
            cfg['grpo']['train']['adv_clip_max'] = None
            self.actor = NativeRecipeActor(self.model.policy.action_expert, cfg)
            self.actor.record_sampling_terms = True
        elif trainable != 'frozen':
            raise ValueError('Explicit frozen or native attention-LoRA scope required')
        self.data = {}; self.conditions = OrderedDict(); self.cache_scenes = cache_scenes
        self.condition_sizes = []
        self.service = RewardService(Path(__file__).resolve().parents[3], workers=scorer_workers,
                                     cache_roots=list(self.protocol['metric_cache_roots']))
        if self.service.protocol['sha256'] != self.protocol['scorer']['sha256']:
            raise ValueError('Canonical protocol changed')
        self.scorer_config_hash = self.service.protocol['sha256']

    def dataset(self, split):
        family = 'TRAIN' if split == 'TRAIN' else 'NAVTEST' if split == 'NAVTEST' else 'DEV'
        if family not in self.data:
            self.data[family] = RawSceneData(self.protocol['data'][family], cohort=family)
        return self.data[family]

    def observation(self, token, split='TRAIN'):
        require_scene(self.protocol, token, split)
        return self.dataset(split).observation(token)

    def normalized_gt(self, token):
        require_scene(self.protocol,token,'TRAIN',update=True)
        return self.codec.norm_odo(self.dataset('TRAIN').target(token))

    @torch.no_grad()
    def condition(self, token, split='TRAIN'):
        require_scene(self.protocol, token, split)
        key = (split, token)
        if key not in self.conditions:
            begin = time.monotonic(); obs = self.observation(token, split)
            state = torch.tensor([obs['proprio']], device=self.device, dtype=torch.float32)
            with torch.autocast(self.device.type, dtype=torch.bfloat16):
                value = self.model.encode_condition(self.processor([obs]), state)
            value = condition_fp32(value)
            self.conditions[key] = move(value, 'cpu')
            self.condition_sizes.append(tensor_bytes(value))
            self.ledger.add(qwen_prefills=1, condition_seconds=time.monotonic()-begin)
            while len(self.conditions) > self.cache_scenes:
                self.conditions.popitem(last=False)
        self.conditions.move_to_end(key)
        return move(self.conditions[key], self.device)

    @staticmethod
    def repeat_condition(condition, repeats):
        return expand_condition(condition, repeats)

    @torch.no_grad()
    def sample_ode(self, condition, roots):
        begin = time.monotonic(); result = []
        chunk = self.protocol['execution']['candidate_chunk']
        for start in range(0, len(roots), chunk):
            z = roots[start:start+chunk].detach().to(self.device).float()
            result.append(self.model.policy.sample_ode(expand_condition(condition, len(z)), z))
            self.ledger.add(model_forward_calls=10)
        self.ledger.add(generation_seconds=time.monotonic()-begin)
        return torch.cat(result)

    def score_actions(self, token, normalized_actions, split='TRAIN', proposal_source='policy',
                      latent_or_proposal_parameters=None, requested_trajectories=None,
                      parent_candidate_ids=None, seed=None):
        require_scene(self.protocol, token, split)
        normalized = torch.as_tensor(normalized_actions).detach().float().cpu()
        decoded = self.codec.denorm_odo(normalized).numpy()
        requested = decoded if requested_trajectories is None else np.asarray(requested_trajectories, dtype=np.float64)
        if requested.shape != decoded.shape or not np.isfinite(requested).all():
            raise ValueError('Invalid physical requests; do not manufacture scores')
        roundtrip = float(np.max(np.abs(decoded-requested)))
        if roundtrip > 1e-5:
            raise ValueError('External physical request does not match source action label')
        data = self.dataset(split); log = data.records[token]['log']
        begin = time.monotonic(); raw = self.service.score((token, log, requested))
        seconds = time.monotonic()-begin
        self.ledger.add(candidate_attempts=len(requested), scored_candidates=len(raw), scorer_batch_calls=1,
                        logical_scoring_cost=len(raw), scoring_wait_seconds=seconds)
        result = []
        for index, feedback in enumerate(raw):
            if feedback['status'] != 'OK':
                raise ValueError('Infrastructure failure retained, not reward zero')
            feedback = json_feedback(feedback)
            metrics = feedback['metrics']; states = feedback['simulated_states']
            parameters = (latent_or_proposal_parameters[index] if isinstance(latent_or_proposal_parameters, list)
                          else latent_or_proposal_parameters)
            candidate = dict(scene_token=token, token=token, log_id=log, log=log, split=split,
                             checkpoint_id=(self.base['model_sha256'] if self.behavior_identity==dict(base='initial',version=0)
                                            else fingerprint(dict(base=self.base['model_sha256'],behavior=self.behavior_identity))),
                             base_checkpoint_id=self.base['model_sha256'],behavior_identity=self.behavior_identity,
                             experiment_id=self.experiment_id,
                             seed=(self.protocol['training_seed'] if seed is None else seed[index] if isinstance(seed,list) else seed), status='OK',
                             sampler_config_hash=fingerprint(self.protocol['samplers']),
                             scorer_config_hash=self.scorer_config_hash, proposal_source=proposal_source,
                             parent_candidate_id=None if parent_candidate_ids is None else parent_candidate_ids[index],
                             normalized_action=normalized[index].tolist(), planned_trajectory=requested[index].tolist(),
                             simulated_trajectory=states, raw_reward=float(metrics['score']), native_subscores=metrics,
                             feasibility_flags=dict(official_NC_DAC_TTC_all_pass=all(metrics[k] == 1 for k in self.protocol['safety_metrics'])),
                             physical_feedback_available_mask=dict(non_drivable_by_time=True, simulated_states=True,
                                 first_collision_time=True, first_TTC_rule_time=True, collision_by_time=False,
                                 TTC_by_time=False, road_margin=False, object_margin=False),
                             physical_feedback=dict(non_drivable_body_by_time=feedback['non_drivable_body_by_time'],
                                 official_first_at_fault_collision_seconds=feedback['official_first_at_fault_collision_seconds'],
                                 official_first_ttc_infraction_seconds=feedback['official_first_ttc_infraction_seconds'],
                                 raw_progress_meters=feedback['raw_progress_meters']),
                             feedback=feedback, latent_or_proposal_parameters=parameters,
                             candidate_attempts=1, scorer_calls=1, scorer_batch_calls_amortized=1/len(raw),
                             wall_time=seconds/len(raw), action_roundtrip_max_abs=roundtrip)
            candidate['candidate_id'] = fingerprint(dict(token=token, checkpoint=candidate['checkpoint_id'],
                     sampler=candidate['sampler_config_hash'], scorer=candidate['scorer_config_hash'],
                     requested=candidate['planned_trajectory']))
            result.append(candidate)
        return result

    def set_behavior_identity(self, identity):
        self.behavior_identity=copy.deepcopy(identity)

    @torch.no_grad()
    def sample_source_sde(self, condition, G, generator, *, token=None, split='TRAIN', proposal_source='source_sde'):
        if self.actor is None:
            raise ValueError('Native actor required for source SDE')
        if torch.device(generator.device).type!='cpu':
            raise ValueError('Original source _randn draws on CPU before transfer; CPU generator required')
        chunk = self.protocol['execution']['candidate_chunk']; chains=[]; old=[]
        begin = time.monotonic()
        for start in range(0, G, chunk):
            expanded = expand_condition(condition, min(chunk, G-start))
            chain, times, deltas, indices = self.actor.sample_action_chain(expanded, 8, generator=generator)
            probability = self.actor.action_chain_logprobs(expanded, chain, times, deltas, stoch_idx=indices)
            for i, (mean, std) in enumerate(self.actor.sampling_terms):
                terms = self.actor._probability_audit['current']
                if not torch.equal(mean, terms['mean'][:, i]) or std != float(terms['std'][i]):
                    # std is computed in Python double then represented FP32.
                    if not torch.equal(mean, terms['mean'][:, i]) or abs(std-float(terms['std'][i])) > 1e-7:
                        raise ValueError('Actual sampling/recompute mean/std mismatch')
            chains.append(chain.detach().cpu()); old.append(probability.detach().cpu())
            self.ledger.add(model_forward_calls=13)
        chain = torch.cat(chains); old_logp=torch.cat(old)
        self.ledger.add(generation_seconds=time.monotonic()-begin)
        rollout = dict(cond_bg=None, chain=chain, timesteps=times.cpu(), deltas=deltas.cpu(), stoch_idx=indices.cpu(),
                       old_logp=old_logp, token=token, split=split, sampling_chunk=chunk)
        records = self.score_actions(token, chain[:, -1], split, proposal_source=proposal_source)
        rollout['rewards'] = torch.tensor([[r['raw_reward'] for r in records]], dtype=torch.float32)
        return rollout, records

    def policy_loss(self, rollout, advantages, mask, condition):
        sub = move(rollout, self.device)
        sub.update(cond_bg=expand_condition(condition, len(sub['chain'])),
                   advantages=torch.as_tensor(advantages, device=self.device).float().reshape(-1).detach(),
                   adv_mask=torch.as_tensor(mask, device=self.device).float().reshape(-1).detach())
        before = self.actor._velocity_calls+self.actor._reference_calls
        loss, metrics = self.actor(sub)
        self.ledger.add(model_forward_calls=self.actor._velocity_calls+self.actor._reference_calls-before)
        return loss, metrics

    def policy_chunks(self, rollout, advantages, mask, condition):
        G = len(rollout['chain']); chunk=self.protocol['execution']['candidate_chunk']
        for start in range(0, G, chunk):
            end=min(G,start+chunk)
            sub=dict(rollout, chain=rollout['chain'][start:end], old_logp=rollout['old_logp'][start:end])
            loss,metrics=self.policy_loss(sub, advantages[start:end], mask[start:end], condition)
            yield loss,metrics,(end-start)/G

    def source_fm_loss(self, condition, normalized_targets, noise=None, times=None, generator=None):
        target=torch.as_tensor(normalized_targets,device=self.device).float()
        if target.ndim==2:target=target[None]
        if noise is None:
            if generator is None:raise ValueError('Independent FM generator or explicit noise required')
            noise=torch.randn(target.shape,device=generator.device,generator=generator,dtype=torch.float32).to(self.device)
        else:noise=torch.as_tensor(noise,device=self.device).float()
        if times is None:
            if generator is None:raise ValueError('Independent explicit FM generator required')
            seed=int(torch.randint(0,2**31-1,(),device=generator.device,generator=generator))
            with torch.random.fork_rng(devices=[self.device.index or 0]):
                torch.random.default_generator.manual_seed(seed)
                torch.cuda.default_generators[self.device.index or 0].manual_seed(seed)
                times=self.model.policy.scheduler.sample_training_t(len(target),self.device,torch.float32)
        else:times=torch.as_tensor(times,device=self.device).float()
        self.ledger.add(model_forward_calls=1)
        return self.model.policy(expand_condition(condition,len(target)),target,noise=noise,timestep=times)

    def close(self):
        self.service.close(); self.ledger.save()
