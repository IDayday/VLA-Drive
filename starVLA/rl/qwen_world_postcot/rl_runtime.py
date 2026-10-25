"""Thin full-framework runtime for the proven real-PDM source GRPO trainer."""
from pathlib import Path
import copy
import json

import torch
import yaml

from starVLA.rl.qwen_sourceflow.collection import collect
from starVLA.rl.qwen_sourceflow.recipe import expected_recipe
from starVLA.rl.flow_grpo.reproducibility import configure_numerics
from scripts.analysis.simwam_recipe.train import parameter_hash
from .contracts import ROOT, boundary_stop_due, file_sha, require_dispatch
from .rl_actor import CleanXRecipeActor
from .rl_conditions import FrozenWorldConditions


def expected_cleanx_recipe():
    """Canonical GRPO mathematics, explicitly new architecture/compute/clipping."""
    cfg = copy.deepcopy(expected_recipe('R_CANONICAL'))
    cfg.update(model_family='QwenWorldPostCoT', prediction_type='x0',
        action_forward_dtype='bfloat16', autocast=True, action_autocast=True,
        data_order='physical-log-disjoint TRAIN hard-list intersection; source sampler and padding',
        observation='current_Qwen_spatial_global_predictive_student_memory',
        deployment='single_candidate_cleanx_shift5_ODE_K10',
        gradient_clipping_effective=1.0,
        gradient_clipping_reason='New clean-x recipe explicitly applies clip1; original source effective clip0 is not claimed unchanged',
        source_velocity_conversion='(FP32_state-FP32_pred_x0)/FP32_sigma; exactly once; sigma>0',
        context_precision='Preserve actual FP32 residual memory; BF16 projected KV; never condition_fp32 recast',
        context_lifetime='Raw frozen memory may persist; trainable policy KV rebuilt each inner update',
        deployment_lora='preserve attention LoRA wrappers for BF16 parity; merging is not default')
    return cfg


def load_cleanx_recipe(path):
    cfg = yaml.safe_load(Path(path).read_text())
    if cfg != expected_cleanx_recipe():
        raise ValueError('Full-framework clean-x canonical recipe differs from its registered complete contract')
    return cfg


def training_dispatch():
    """Request this trainer's own native checkpoint before the finite lease ends."""
    host = require_dispatch()
    if boundary_stop_due(host):
        from scripts.analysis.simwam_recipe.train import stop
        stop()
    return host


def source_files():
    paths = []
    for root in ['starVLA/model/modules/qwen_world_postcot', 'starVLA/model/modules/qwen_sourceflow',
                 'starVLA/rl/qwen_world_postcot', 'starVLA/rl/qwen_sourceflow',
                 'starVLA/rl/simwam_recipe', 'starVLA/rl/fm_transfer']:
        paths.extend(Path(root).rglob('*.py'))
    paths += [Path(name) for name in ['tools/qwen_world_postcot/train_rl.py',
        'scripts/analysis/simwam_recipe/train.py', 'starVLA/rl/flow_grpo/transactions.py',
        'starVLA/rl/flow_grpo/reproducibility.py', 'starVLA/rl/safe_flow_improvement/v1_reward.py',
        'scripts/analysis/constrained_flow_noise/batched_canonical_score.py']]
    return {str(path): file_sha(path) for path in sorted(set(paths))}


class CleanXRuntime:
    require_dispatch = staticmethod(training_dispatch)
    source_files = staticmethod(source_files)
    collect = staticmethod(collect)

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('--il-checkpoint', required=True,
                            help='Complete full-framework clean-x IL deployment checkpoint')
        parser.add_argument('--parent-selection', help='Sealed full100k IL and complete DEV selection; required formally')
        parser.add_argument('--data', default=str(ROOT/'data/TRAIN'))

    def load_recipe(self, args):
        configure_numerics()
        cfg = load_cleanx_recipe(args.config)
        from .loading import inspect_checkpoint, validate_rl_parent_selection
        self.parent = inspect_checkpoint(args.il_checkpoint)
        identity = self.parent['identity']
        if identity.get('uses_rl') is not False or identity.get('uses_metric_supervision') is not True:
            raise ValueError('RL parent must be the identified IL+representation+metric-supervision full framework')
        if args.qualification:
            self.selection = None
        else:
            if not args.parent_selection:
                raise ValueError('Formal RL waits for full100k IL and its locked DEV-selected checkpoint')
            self.selection = validate_rl_parent_selection(args.parent_selection, self.parent)
            if (self.selection['schema'] != 'world_postcot_DEV_selection_v1'
                    or self.selection['completed_IL_updates'] != 100000
                    or self.selection['Navtest_used'] is not False
                    or self.selection['selected']['model_sha256'] != self.parent['model_sha256']
                    or Path(self.selection['selected']['checkpoint']).resolve() != Path(args.il_checkpoint).resolve()):
                raise ValueError('Formal RL cannot precede complete IL/DEV selection or use Navtest selection')
        return cfg

    def make_scene_stream(self, cfg, args, world):
        from .data import WorldRLTokenStream
        expected = ROOT/'data/RL_HARD.json'
        if Path(args.manifest).resolve() != expected.resolve():
            raise ValueError('Use the locked physical-log-disjoint full-framework RL TRAIN manifest')
        return WorldRLTokenStream(args.manifest, world)

    def make_actor(self, cfg, args, device):
        from .loading import load_deployment_checkpoint
        from .data import WorldSceneData
        model, info = load_deployment_checkpoint(args.il_checkpoint, device=device)
        if info['model_sha256'] != self.parent['model_sha256']:
            raise ValueError('RL parent changed between eligibility verification and strict weight loading')
        self.parent = info
        head = model.actor
        if head.hidden_dim != 1024 or head.num_layers != 20 or head.action_dim != 3:
            raise ValueError('Formal/real qualification uses the complete20-layer clean-x actor, not a reduced head')
        data = WorldSceneData(args.data, role='TRAIN')
        self.conditions = FrozenWorldConditions(model, data, info['model_sha256'])
        actor = CleanXRecipeActor(head, cfg, self.conditions)
        self.actor = actor
        if args.qualification:
            actor.record_sampling_terms = True
            self.observation_before = {name: parameter_hash(getattr(self.conditions, name), trainable=False)
                                       for name in ['qwen', 'scene_provider', 'predictive_provider']}
        return actor, None, None

    def make_conditions(self, actor):
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        return self.conditions

    def recipe_identity(self, cfg, args, condition_sha):
        return dict(schema='world_postcot_cleanx_canonical_RL_v1', recipe=cfg,
            upstream_commit='1433ee7e06a62bd8b89da74455df0d82bda7b726',
            SFT_sha256=self.parent['model_sha256'], IL_parent=self.parent,
            parent_selection=self.selection,
            parent_selection_path=str(Path(args.parent_selection).resolve()) if self.selection is not None else None,
            hard_manifest_sha256=file_sha(args.manifest), projected_condition_identity=condition_sha,
            policy_ratio='source relative-clamped FP32 dimension-mean stochastic-transition surrogate',
            deployment='current-only student memory; clean-x single candidate; shift5 ODE10; frozen LoRA wrappers',
            prediction_type='x0', uses_metric_supervision=True, uses_rl=True,
            full_framework_IL_precedes_RL=True,
            x_to_v='(x-pred_x0)/sigma once at positive sigma; source SDE/probability math unchanged',
            trainable_scope='action self/cross-attention LoRA only; frozen Qwen/all student memory/IL reference',
            data_role='physical-log-disjoint TRAIN hard-list intersection; embargo preserved',
            execution_changes_vs_old_native='BF16 clean-x actor; smaller memory/actor; explicit clip1; different legal TRAIN population')

    def finish_qualification(self, result):
        after = {name: parameter_hash(getattr(self.conditions, name), trainable=False)
                 for name in ['qwen', 'scene_provider', 'predictive_provider']}
        if after != self.observation_before:
            raise ValueError('Frozen Qwen/current/predictive memory changed during real clean-x RL qualification')
        result.update(frozen_observation_before=self.observation_before, frozen_observation_after=after,
            prefill_calls_this_invocation=self.conditions.prefill_calls,
            prefill_seconds_this_invocation=self.conditions.prefill_seconds,
            actual_frozen_memory_dtype=self.conditions.observed_memory_dtype,
            prefill_cost_included_in_rollout=True,
            prepared_context_builds_this_invocation=dict(self.actor.prepared_context_builds),
            recent_prepared_context_source_batches=list(self.actor.prepared_context_source_batches),
            prediction_type='x0', teachers_or_post_head_used_in_RL=False)

    def validate_collection(self, rollout):
        check = rollout['qualification_sampling_check']
        if check and check['status'] != 'PASS':
            raise ValueError('Actual clean-x sampling/recompute means differ; retain the failed real chain: '+repr(check))


WorldPostCoTRuntime = CleanXRuntime
