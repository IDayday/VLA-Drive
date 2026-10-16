"""Qualify the existing upstream window switch, without scoring or training DDP.

The real eight-rank continuation test remains a separate dispatch requirement.
This finite test does not claim that a cleaner candidate improves deployment.
"""
import argparse
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

from omegaconf import OmegaConf
import torch

from scripts.analysis.simwam_recipe.qualify_math import original_class, actor_class
from scripts.analysis.simwam_recipe.registered_followup import require_registration
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.simwam_recipe.contracts import ROOT, file_sha, fingerprint, load_recipe
from starVLA.rl.simwam_recipe.source_advantages import advantages
from starVLA.rl.simwam_recipe.trainer_adapter import SourceLoss
from starVLA.rl.simwam_recipe.vendor import action_math, trainer_math


def run(source):
    repo = Path(__file__).resolve().parents[3]
    configs = repo / 'configs/flow_rl'
    control = load_recipe(configs / 'simwam_canonical_ddp.yaml')
    candidate = load_recipe(configs / 'simwam_canonical_last_ode_ddp.yaml')
    expected = copy.deepcopy(control)
    expected['variant'] = 'R_CANONICAL_LAST_ODE'
    expected['grpo']['sample']['sde_exclude_last_step'] = True
    assert candidate == expected
    require_registration(candidate['variant'])
    rejected = []
    mutations = [('learning_rate', 1e-4), ('seed', 43), ('max_steps', 100),
                 ('gradient_clipping_effective', 1.), ('DDP_static_graph', True)]
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'invalid.yaml'
        changes = []
        for key, value in mutations:
            changed = copy.deepcopy(candidate); changed[key] = value
            changes.append((key, changed))
        for key, value in [('kl_dim_weights', [1., 1., .5, .5]), ('kl_beta', 10)]:
            changed = copy.deepcopy(candidate); changed['grpo']['train'][key] = value
            changes.append((key, changed))
        changed = copy.deepcopy(candidate); changed['grpo']['sample']['sde_step_indices'] = [6, 7, 8]
        changes.append(('explicit_indices_not_registered', changed))
        changed = copy.deepcopy(candidate); changed['variant'] = 'R_CANONICAL'
        changes.append(('historical_canonical_window', changed))
        changed = copy.deepcopy(candidate); changed['variant'] = 'R_CANONICAL_X1'
        changed['grpo']['train']['kl_dim_weights'] = [1., 1., .5, .5]
        changes.append(('X1_window_mutation', changed))
        for name, changed in changes:
            OmegaConf.save(OmegaConf.create(changed), path)
            try:
                load_recipe(path)
            except ValueError:
                rejected.append(name)
            else:
                raise AssertionError('Unregistered change accepted: ' + name)

    lock = json.loads((repo / 'reports/ddp_simwam_recipe_20261015/upstream_lock.json').read_text())
    for rel, sha in lock['simwam']['files'].items():
        assert file_sha(source / rel) == sha
    for rel, sha in lock['vendor'].items():
        assert file_sha(repo / rel) == sha
    oracle_action = original_class(source / 'src/simwam/models/wan22/simwam_grpo.py',
        'ActionGRPOMixin', vars(action_math), lock['extraction']['action_methods'])
    oracle_loss = original_class(source / 'src/simwam/trainer_grpo.py',
        'SimWAMGRPOTrainer', vars(trainer_math), lock['extraction']['trainer_methods'])
    checks = []

    def equal(name, actual, reference, atol=0):
        torch.testing.assert_close(actual, reference, rtol=0, atol=atol)
        checks.append(dict(check=name, max_abs=float((actual.float()-reference.float()).abs().max()), atol=atol))

    for dim in (3, 4):
        for batch in (8, 16):
            cfg = copy.deepcopy(candidate)
            if dim == 3:
                cfg['grpo']['train']['kl_dim_weights'] = [.2, 1, 1]
            actor = actor_class(action_math.ActionGRPOMixin)(cfg, dim=dim)
            oracle = actor_class(oracle_action)(cfg, dim=dim)
            cond = {'context': torch.randn(batch, 8, dim)}
            noise = torch.Generator().manual_seed(123)
            other_noise = torch.Generator().manual_seed(123)
            recorded = []
            kernel = actor._sde_mean_std

            def capture(x, v, sigma, dt):
                mean, std = kernel(x, v, sigma, dt)
                recorded.append((mean.detach().clone(), std))
                return mean, std

            actor._sde_mean_std = capture
            try:
                chain, ts, dt, idx = actor.sample_action_chain(cond, 8, generator=noise)
            finally:
                del actor._sde_mean_std
            other, ots, odt, oi = oracle.sample_action_chain(cond, 8, generator=other_noise)
            tag = f'D{dim}_B{batch}_'
            assert idx.tolist() == oi.tolist() == [6, 7, 8]
            assert chain.shape == (batch, 11, 8, dim) and len(recorded) == 3
            equal(tag+'chain_upstream', chain, other)
            equal(tag+'times_upstream', ts, ots)
            equal(tag+'deltas_upstream', dt, odt)
            equal(tag+'RNG_upstream', noise.get_state(), other_noise.get_state())
            with torch.no_grad():
                # The live x is contiguous. Stacking the saved chain changes
                # the batch stride of chain[:,9], which can change FP32 GEMM
                # rounding; reproduce the actual live-layout arithmetic here.
                x9 = chain[:, 9].contiguous()
                v = actor.action_velocity(x9, ts[9].expand(batch), cond)
                last = actor.infer_action_scheduler.step(v, dt[9], x9)
                equal(tag+'terminal_is_Euler_ODE', chain[:, 10], last)
                sde_mean, _ = actor._sde_mean_std(chain[:, 9], v, float(ts[9])/1000, float(dt[9]))
                assert not torch.equal(last, sde_mean), 'ODE suffix was confused with conditional SDE mean'
            old, terms = actor.action_chain_logprobs(cond, chain, ts, dt, stoch_idx=idx, return_terms=True)
            reference_old = oracle.action_chain_logprobs(cond, other, ots, odt, stoch_idx=oi)
            equal(tag+'old_logprob_upstream', old, reference_old)
            assert old.shape == (batch, 3)
            for j, k in enumerate(idx.tolist()):
                # Saved-state recomputation uses a different legal batch
                # stride; retain the existing actual-DDP 1e-6 qualification.
                equal(tag+f'sampling_recompute_mean{k}', recorded[j][0], terms['mean'][:, j], atol=1e-6)
                assert abs(recorded[j][1]-float(terms['std'][j])) < 1e-8
            replay = actor.action_chain_logprobs(cond, chain, ts, dt, stoch_idx=idx)
            equal(tag+'on_policy_ratio', (replay-old.detach()).exp(), torch.ones_like(old))
            changed_suffix = chain.clone(); changed_suffix[:, 10] += 1
            equal(tag+'deterministic_suffix_has_no_density',
                  actor.action_chain_logprobs(cond, changed_suffix, ts, dt, stoch_idx=idx), old)

            rewards = torch.linspace(.05, .95, 8).expand(batch//8, -1).clone()
            if batch == 16:
                rewards[1] = .1
            stats = advantages(rewards)
            rollout = dict(chain=chain, timesteps=ts, deltas=dt, stoch_idx=idx,
                           cond_bg=cond, old_logp=old.detach(), **stats)
            with torch.no_grad():
                for model in (actor, oracle):
                    for name, param in model.named_parameters():
                        if name.endswith('lora_B'):
                            param.add_(.0001)
            loss = SourceLoss(cfg)
            source_loss = oracle_loss()
            source_loss.__dict__.update(loss.__dict__)
            source_loss.accelerator = SimpleNamespace(gather=lambda value:value)
            actual, _ = loss._policy_loss(actor, rollout)
            expected_loss, _ = source_loss._policy_loss(oracle, rollout)
            equal(tag+'total_loss_upstream', actual, expected_loss)
            actual.backward(); expected_loss.backward()
            grads_a, grads_b = [], []
            for (name, param), (oname, other_param) in zip(actor.named_parameters(), oracle.named_parameters()):
                assert name == oname
                if param.requires_grad:
                    assert param.grad is not None and torch.isfinite(param.grad).all()
                    grads_a.append(param.grad.flatten()); grads_b.append(other_param.grad.flatten())
                else:
                    assert param.grad is None
            equal(tag+'LoRA_gradients_upstream', torch.cat(grads_a), torch.cat(grads_b))
            assert torch.cat(grads_a).abs().sum() > 0
    return dict(status='PASS', upstream_commit=lock['simwam']['commit'],
        control_identity=fingerprint(control), candidate_identity=fingerprint(candidate),
        only_registered_diff=True, rejected_mutations=rejected, checks=checks,
        stochastic_steps=[6,7,8], terminal_step='Euler ODE; excluded from Gaussian proxy',
        estimator_caveat='Shared adapters affect deterministic prefix/suffix; only saved stochastic transitions optimized; source empirical proxy, not full-chain unbiased gradient',
        new_scoring_calls=0, real_DDP_optimizer_updates=0,
        pending='Real eight-rank learning/recovery and zero/nonzero DDP sampling instrumentation')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--source', default=ROOT+'/upstream/SimWAM')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    result = run(Path(args.source)); output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, result)
    print(json.dumps(dict(status=result['status'], checks=len(result['checks']), output=str(output))))


if __name__ == '__main__':
    main()
