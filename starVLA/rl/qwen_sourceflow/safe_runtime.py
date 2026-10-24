"""Minimal safety hooks into the existing native on-policy/transaction trainer."""
import json
from pathlib import Path
import torch
import torch.distributed as dist
from .rl_runtime import NativeRuntime
from .contracts import configure_native_numerics, fingerprint, file_sha
from .loading import inspect_il
from .selection import validate_selection
from .safe_recipe import load_recipe, require_dispatch, COSTS
from .safe_advantages import component_advantages, event_costs, EventMultipliers, factorized_advantages
from starVLA.rl.simwam_recipe import checkpoint
from starVLA.rl.simwam_recipe.collection import finalize_advantages as source_finalize
from starVLA.rl.simwam_recipe.recovery_state import tensor_digest


class SafeRuntime(NativeRuntime):
    require_dispatch = staticmethod(require_dispatch)

    @staticmethod
    def source_files():
        files=NativeRuntime.source_files()
        for p in ['starVLA/rl/constrained_flow_noise/vendor/omnisafe_lagrange.py',
                  'starVLA/rl/flow_grpo/math.py']:
            files[p]=file_sha(p)
        return files

    @staticmethod
    def add_arguments(parser):
        NativeRuntime.add_arguments(parser)
        parser.add_argument('--safety-budget', help='Sealed SFT stochastic TRAIN calibration; SCADV3 only')

    def load_recipe(self, args):
        configure_native_numerics()
        self.cfg = cfg = load_recipe(args.config)
        self.parent = inspect_il(args.il_checkpoint, qualification=args.qualification, parent_selection=args.parent_selection)
        self.selection = validate_selection(args.parent_selection, self.parent) if not args.qualification else None
        if not args.qualification:
            from .safe_recipe import ROOT
            registration=json.loads((ROOT/'experiment_registration.json').read_text())
            active={v['name']:v for v in registration['variants'].values()}
            if (cfg['variant'] not in active or registration['parent_model_sha256']!=self.parent['files']['model.pt'] or
                    registration['config_sha256'][active[cfg['variant']]['key']]!=file_sha(args.config)):
                raise ValueError('Unregistered formal safety method/configuration')
        self.budget = None
        if cfg['variant'] == 'R_CANONICAL_SCADV3':
            if not args.safety_budget: raise ValueError('SCADV3 requires a sealed calibration, not a default zero budget')
            b = json.loads(Path(args.safety_budget).read_text()); body = dict(b); sha = body.pop('sha256')
            if (fingerprint(body) != sha or b['status'] != 'COMPLETE' or b['scenes'] != 512 or b['candidates'] != 4096 or
                    b['SFT_sha256'] != self.parent['files']['model.pt'] or b['cost_metrics'] != list(COSTS) or
                    b['hard_manifest_sha256'] != file_sha(args.manifest) or b['population'] != 'TRAIN_only_public_hard'):
                raise ValueError('Safety calibration does not bind the original IL and TRAIN population')
            self.budget = b
        elif args.safety_budget:
            raise ValueError('X1 does not consume safety costs or budgets')
        return cfg

    def recipe_identity(self, cfg, args, condition_sha):
        info = super().recipe_identity(cfg, args, condition_sha)
        info['safety_research'] = dict(variant=cfg['variant'], baseline='R_CANONICAL',
            baseline_training_source='434cba8e47aef3e5ab8319f6b017ab5b0f860db1',
            calibration=self.budget, hard_budget_unchanged=True,
            interpretation='separately named algorithm adaptation, not the unmodified SimWAM recipe')
        return info

    def initialize_state(self, state):
        if self.budget is None: return
        self.dual = EventMultipliers(self.budget['budgets'])
        if 'safety_dual' not in state:
            if state['updates'] or state['collections']: raise ValueError('Missing dual during exact resume')
            state['safety_dual'] = self.dual.state_dict()
            state['safety_budget_sha256'] = self.budget['sha256']
        if state['safety_budget_sha256'] != self.budget['sha256']: raise ValueError('Changed safety budget on resume')
        self.dual.load_state_dict(state['safety_dual'])
        if self.dual.collections != state['collections']: raise ValueError('Dual/fresh collection count mismatch')

    def finalize_advantages(self, rollout, state):
        source_finalize(rollout)  # RAW global std still logged; never the SCADV denominator.
        if self.cfg['variant']=='R_CANONICAL_WLA_ADV':
            a,stats=factorized_advantages(rollout['rewards'],rollout['records'])
            rollout['advantages']=a
            rollout['adv_mask']=torch.full_like(a,float(bool((a!=0).any())))
            rollout['factorized_advantage']=stats
            rollout['advantage_statistics'].update(normalization='DriveWLA_v1 component z; combined scene std; PDMS variance shrinkage',
                effective_PG_candidate_fraction=float(rollout['adv_mask'].mean()),
                shrinkage=float(stats['shrinkage']),PDMS_variance=float(stats['PDMS_variance']))
            return
        if self.budget is None: return
        self.dual.load_state_dict(state['safety_dual'])
        before = self.dual.state_dict(); snapshot = self.dual.snapshot()
        costs = event_costs(rollout['records']).to(rollout['rewards'].device)
        a, stats = component_advantages(rollout['rewards'], costs, snapshot.to(costs.device))
        rollout['advantages'] = a.reshape(-1)
        rollout['adv_mask'] = (a != 0).any(-1).float().expand_as(a).reshape(-1)
        local = costs.sum((0, 1)); count = torch.tensor(float(costs.shape[0] * 8), device=costs.device, dtype=torch.float64)
        if dist.is_initialized():
            dist.all_reduce(local); dist.all_reduce(count)
        means = (local / count).cpu()
        if not dist.is_initialized() or dist.get_rank() == 0:
            self.dual.update(means, state['collections']); after = self.dual.state_dict()
        else: after = None
        after = checkpoint.gather(after)[0]
        self.dual.load_state_dict(after)
        state['safety_dual'] = after
        rollout['safety'] = dict(schema='native_scadv3_collection_v1', costs=costs.detach(),
            lambda_snapshot=snapshot, budgets=list(self.dual.budgets),
            budget_sha256=self.budget['sha256'], dual_before=before, dual_after=after,
            global_unfiltered_cost_mean=means, component_z=stats['component_z'],
            component_std=stats['component_std'])
        rollout['advantage_statistics'].update(normalization='scene FP64 component std with fixed floors; detached lambda; no post whitening',
            effective_PG_candidate_fraction=float(rollout['adv_mask'].mean()),
            dual_collection=self.dual.collections,
            lambda_snapshot=snapshot.tolist(), lambda_next=self.dual.snapshot().tolist(),
            global_unfiltered_event_cost_mean=means.tolist(), fixed_event_cost_budgets=list(self.dual.budgets))


def boundary_summary(state, rollout, identity):
    """Recompute costs and component advantages at every published boundary."""
    safety = identity.get('safety_research', {})
    if safety.get('variant') != 'R_CANONICAL_SCADV3': return None
    budget = safety['calibration']; dual = state['safety_dual']
    if (state['safety_budget_sha256'] != budget['sha256'] or dual['collections'] != state['collections'] or
            tuple(dual['budgets']) != tuple(budget['budgets'])):
        raise ValueError('Uncommitted or changed dual state/budget')
    summary = dict(dual_state_digest=tensor_digest(dual), budget_sha256=budget['sha256'])
    if rollout is None: return summary
    s = rollout['safety']
    costs = event_costs(rollout['records']).to(s['costs'].device)
    torch.testing.assert_close(costs, s['costs'], atol=0, rtol=0)
    if (s['budget_sha256'] != budget['sha256'] or s['dual_after']['collections'] != state['collections'] or
            s['dual_before']['collections'] != state['collections'] - 1 or
            tensor_digest(s['dual_after']) != tensor_digest(dual)):
        raise ValueError('Frozen behavior advantage/dual ancestry mismatch')
    torch.testing.assert_close(s['lambda_snapshot'], torch.tensor(s['dual_before']['values'],
        device=s['lambda_snapshot'].device,dtype=s['lambda_snapshot'].dtype), atol=0, rtol=0)
    a, _ = component_advantages(rollout['rewards'], costs, s['lambda_snapshot'].to(costs.device))
    torch.testing.assert_close(a.reshape(-1), rollout['advantages'], atol=0, rtol=0)
    mask = (a != 0).any(-1).float().expand_as(a).reshape(-1)
    torch.testing.assert_close(mask, rollout['adv_mask'], atol=0, rtol=0)
    summary.update(costs=costs.cpu().tolist(), lambda_snapshot=s['lambda_snapshot'].tolist(),
                   global_unfiltered_cost_mean=s['global_unfiltered_cost_mean'].tolist())
    return summary


def validate_factorized_boundary(rollout):
    a,stats=factorized_advantages(rollout['rewards'],rollout['records'])
    torch.testing.assert_close(a,rollout['advantages'],atol=0,rtol=0)
    mask=torch.full_like(a,float(bool((a!=0).any())))
    torch.testing.assert_close(mask,rollout['adv_mask'],atol=0,rtol=0)
    # Restore checks CPU tensors against CUDA-produced FP64 diagnostics. Real
    # eight-rank evidence shows <=3.56e-15 reduction/sqrt roundoff here. Saved
    # FP32 advantages above must still match bitwise, and payload hashes remain
    # exact. This tolerance changes validation only, never sampling or loss.
    for k,v in stats.items():torch.testing.assert_close(v,rollout['factorized_advantage'][k],atol=1e-12,rtol=1e-12)
    return dict(formula='DriveWLA_v1_8_9',digest=tensor_digest(rollout['factorized_advantage']))
