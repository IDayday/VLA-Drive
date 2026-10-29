"""New architecture-neutral lifecycle on the proven real-PDM source GRPO math."""
import copy
from dataclasses import replace
from pathlib import Path
import time

import torch
import yaml

from starVLA.rl.qwen_world_postcot.rl_runtime import expected_cleanx_recipe
from starVLA.rl.qwen_world_postcot.data import WorldSceneData, WorldRLTokenStream
from starVLA.rl.qwen_sourceflow.collection import collect
from starVLA.rl.qwen_sourceflow.contracts import QWEN_PATH
from starVLA.model.modules.qwen_sourceflow.qwen_encoder import QwenObservationProcessor
from starVLA.rl.qwen_sourceflow.export import PROCESSOR_FILES
from starVLA.rl.flow_grpo.reproducibility import configure_numerics
from scripts.analysis.simwam_recipe.train import parameter_hash
from .contracts import ROOT, file_sha, fingerprint, require_dispatch, boundary_stop_due
from .loading import inspect_IL_checkpoint, load_IL_checkpoint
from .parent_selection import validate_il_parent_selection
from .policy_adapter import FourFrameworkRecipeActor, CONDITION_SCHEMA, DESCRIPTOR_SCHEMA


def expected_recipe(architecture_id=None):
    if architecture_id not in (None, 'A', 'B', 'C', 'D'):
        raise ValueError('Unknown four-framework architecture')
    cfg = copy.deepcopy(expected_cleanx_recipe())
    cfg.update(model_family='QwenWorldFourFrameworks', architecture_id=architecture_id,
        observation='current_only_frozen_C160_or_native_Qwen_visual_embedding_inputs',
        context_lifetime='frozen current inputs only; B/C/D policy prefixes rebuilt with gradients per logprob invocation',
        context_precision='preserve actual frozen residual dtype; BF16 prefix/actor compute, FP32 action state/sigma/probability',
        trainable_scope='registered own-architecture deployment attention and C world-KV bridges; ordinary r16 LoRA')
    return cfg


def load_recipe(path, architecture_id):
    cfg = yaml.safe_load(Path(path).read_text())
    # A single frozen common recipe file plus an explicit architecture argument.
    if cfg != expected_recipe(None):
        raise ValueError('RL recipe differs from the locked common source-math configuration')
    return expected_recipe(architecture_id)


def source_files():
    paths = []
    for folder in ('starVLA/model/modules/qwen_world_compare', 'starVLA/rl/qwen_world_compare',
                   'starVLA/model/modules/qwen_world_postcot', 'starVLA/model/modules/qwen_sourceflow',
                   'starVLA/rl/qwen_sourceflow', 'starVLA/rl/simwam_recipe'):
        paths += list(Path(folder).rglob('*.py'))
    for name in ('tools/qwen_world_compare/train_rl.py', 'tools/qwen_world_compare/evaluate.py',
                 'scripts/analysis/simwam_recipe/train.py', 'starVLA/rl/flow_grpo/transactions.py',
                 'starVLA/rl/flow_grpo/reproducibility.py', 'starVLA/rl/safe_flow_improvement/v1_reward.py',
                 'scripts/analysis/constrained_flow_noise/batched_canonical_score.py'):
        if Path(name).is_file():paths.append(Path(name))
    return {str(path):file_sha(path) for path in sorted(set(paths))}


class FrozenCurrentConditions:
    """One-scene frozen input cache; trainable P/W/text-prefix is never cached.

    Checkpoints contain only the four-field descriptor returned by ``get``.
    Re-materialization after resume re-encodes the same current observations.
    The cache identity binds architecture, own IL/reference, data, processor,
    numerical execution and mask/source contract.
    """
    def __init__(self, model, data, parent, *, processor=None):
        self.model, self.data = model.eval(), data
        self.architecture_id = model.architecture_id
        self.tokens = frozenset(data.tokens)
        self.processor = processor if processor is not None else QwenObservationProcessor(QWEN_PATH)
        if self.architecture_id != 'D':
            self.model.qwen.eval().requires_grad_(False)
            self.model.scene_provider.eval().requires_grad_(False)
        else:
            # Never freeze the complete language model here: it contains the
            # current policy's newly installed text attention LoRA.
            self.model.decision.qwen_model.visual.eval().requires_grad_(False)
            self.model.decision.qwen_model.get_input_embeddings().requires_grad_(False)
        identity = dict(schema='qwen_world_compare_frozen_current_inputs_v1',
            architecture_id=self.architecture_id, IL_parent_model_sha256=parent['model_sha256'],
            permanent_reference_identity_sha256=fingerprint(parent['identity']),
            data_manifest_sha256=data.sha, group_size=8,
            processor_files={name:file_sha(QWEN_PATH/name) for name in PROCESSOR_FILES},
            precision='FP32 parameters; BF16 frozen current forward; preserve residual dtype; no policy KV cache',
            image_contract='current left/front/right; official processor fixed384x672 target; native missing-view masks',
            prefix_mask=('native_C_causal/P_bidir/A_bidir_R_isolated' if self.architecture_id == 'D'
                         else 'current_C160_only; architecture_P_W_no_actions_or_sigma'),
            source_files={path:sha for path, sha in source_files().items()
                          if path.startswith('starVLA/model/modules/qwen_world_compare/')
                          or path in ('starVLA/model/modules/qwen_world_postcot/qwen_encoder.py',
                                      'starVLA/model/modules/qwen_world_postcot/memory.py',
                                      'starVLA/model/modules/qwen_sourceflow/qwen_encoder.py')},
            teachers_or_future_or_scores_as_inputs=False,
            frozen_cache_boundary='pre-language visual/embedding' if self.architecture_id == 'D' else 'current_C160')
        self.identity = identity
        self.sha = self.identity_sha256 = fingerprint(identity)
        self.cached = self.token = None
        self.prefill_calls = 0;self.prefill_seconds = 0.;self.observed_memory_dtype = None

    def get(self, token, device=None):
        if token not in self.tokens:
            raise ValueError('Current observation lies outside locked RL TRAIN')
        return dict(schema=DESCRIPTOR_SCHEMA, token=token, identity=self.sha, repeats=8)

    @torch.no_grad()
    def materialize(self, descriptor, device):
        if descriptor != self.get(descriptor['token'], device):
            raise ValueError('Frozen observation/IL/data/processor descriptor identity changed')
        if self.token != descriptor['token']:
            self.cached = None;start = time.monotonic()
            observation = self.data.observation(descriptor['token'])
            batch = self.processor([observation])
            ego = torch.tensor([observation['proprio']], device=device, dtype=torch.float32)
            with torch.autocast(device.type, dtype=torch.bfloat16):
                current = self.model.encode_current(batch, ego, (descriptor['token'],))
            if self.architecture_id == 'D':
                current = replace(current, encoded=current.encoded.detached(), ego=current.ego.detach())
                dtype = str(current.encoded.inputs_embeds.dtype)
            else:
                if current.scene.shape[1] != 160:
                    raise ValueError('Frozen A/B/C cache must be C160 without predictive tokens')
                current = replace(current, scene=current.scene.detach(), valid=current.valid.detach(),
                                  ego=current.ego.detach())
                dtype = str(current.scene.dtype)
            if self.observed_memory_dtype is not None and self.observed_memory_dtype != dtype:
                raise ValueError('Frozen current execution dtype changed during this run')
            self.observed_memory_dtype = dtype;self.cached = current;self.token = descriptor['token']
            self.prefill_calls += 1
            if device.type == 'cuda':torch.cuda.synchronize(device)
            self.prefill_seconds += time.monotonic()-start
        # Current and reference share only immutable frozen encoder inputs. The
        # actor invokes distinct current/reference prepare_prefix implementations.
        return dict(schema=CONDITION_SCHEMA, context=torch.zeros((8, 1, 1), device=device),
                    current=self.cached, ref_current=self.cached)


def training_dispatch():
    host = require_dispatch()
    if boundary_stop_due(host):
        # The new trainer installs the same graceful inner-boundary signal hook.
        from tools.qwen_world_compare.train_rl import stop
        stop()
    return host


class FourFrameworkRuntime:
    require_dispatch = staticmethod(training_dispatch)
    source_files = staticmethod(source_files)
    collect = staticmethod(collect)

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('--architecture', choices=('A', 'B', 'C', 'D'), required=True)
        parser.add_argument('--il-checkpoint', required=True)
        parser.add_argument('--parent-selection')
        parser.add_argument('--data', help='Exact same TRAIN root as this own IL parent')

    def load_recipe(self, args):
        configure_numerics()
        self.args = args;self.cfg = load_recipe(args.config, args.architecture)
        self.parent = inspect_IL_checkpoint(args.il_checkpoint, verify_files='model')
        identity = self.parent['identity']
        if identity['architecture_id'] != args.architecture:
            raise ValueError('Cannot initialize RL from a different architecture')
        parent_cfg = identity['config']
        expected_data = (Path(parent_cfg['training_data']) if 'training_data' in parent_cfg
                         else Path(parent_cfg['data_root'])/'TRAIN').resolve()
        if args.data is None:args.data = str(expected_data)
        if Path(args.data).resolve() != expected_data:
            raise ValueError('RL current observations must preserve the own IL TRAIN role')
        self.selection = None
        if not args.qualification:
            if not args.parent_selection:
                raise ValueError('Formal RL waits for full100k IL and complete DEV-only own-parent selection')
            self.selection = validate_il_parent_selection(args.parent_selection, self.parent)
        return self.cfg

    def make_scene_stream(self, cfg, args, world):
        stream = WorldRLTokenStream(args.manifest, world)
        if len(stream.tokens) != 15171:
            raise ValueError('The shared four-framework RL hard TRAIN population must contain15171 scenes')
        self.stream_identity = stream.identity
        return stream

    def make_actor(self, cfg, args, device):
        model, info = load_IL_checkpoint(args.il_checkpoint, device=device, deployment=True)
        if (info['model_sha256'] != self.parent['model_sha256']
                or info['identity'] != self.parent['identity']):
            raise ValueError('Own IL parent changed before strict RL initialization')
        self.parent = info
        actor = FourFrameworkRecipeActor(model, cfg)
        self.conditions = FrozenCurrentConditions(model, WorldSceneData(args.data, role='TRAIN'), info)
        object.__setattr__(actor, 'conditions', self.conditions)
        self.actor = actor
        if args.qualification:
            actor.record_sampling_terms = True
            self.frozen_before = parameter_hash(actor, trainable=False)
        return actor, None, None

    def make_conditions(self, actor):
        if torch.distributed.is_initialized():torch.distributed.barrier()
        return self.conditions

    def recipe_identity(self, cfg, args, condition_sha):
        return dict(schema='qwen_world_compare_RL_v1', architecture_id=args.architecture,
            recipe=cfg, upstream_commit='1433ee7e06a62bd8b89da74455df0d82bda7b726',
            SFT_sha256=self.parent['model_sha256'], IL_parent=self.parent,
            parent_selection=self.selection,
            parent_selection_path=str(Path(args.parent_selection).resolve()) if self.selection else None,
            hard_manifest_sha256=file_sha(args.manifest), projected_condition_identity=condition_sha,
            prediction_type='x0', uses_rl=True, uses_metric_supervision=True,
            full_framework_IL_precedes_RL=True, policy_ratio='source relative-clamped dimension-mean transition surrogate',
            deployment='single clean-x shift5 ODE K10; current images/navigation/ego; LoRA wrappers',
            x_to_v='one FP32 (state-pred_x0)/positive_sigma conversion outside autocast',
            trainable_scope=self.actor.closure,
            reference='own selected IL parent permanently frozen; independent full architecture copy',
            observation_cache=self.conditions.identity,
            training_data_role='same15171 shared physical-log-disjoint TRAIN hard list, original order/padding',
            active_auxiliary_losses=False, BC_or_GT_replay=False, learned_score_reward=False)

    def validate_collection(self, rollout):
        check = rollout.get('qualification_sampling_check')
        if check and check['status'] != 'PASS':
            raise ValueError('Actual source sampled mean/std disagree with fixed-chain recomputation: '+repr(check))
        if rollout['cond_bg'] != self.conditions.get(rollout['token']):
            raise ValueError('Rollout must retain a current-only recoverable token descriptor')

    def finish_qualification(self, result):
        after = parameter_hash(self.actor, trainable=False)
        if after != self.frozen_before:
            raise ValueError('A frozen base/observation/permanent reference parameter changed')
        result.update(frozen_parameters_before=self.frozen_before, frozen_parameters_after=after,
            prefill_calls_this_invocation=self.conditions.prefill_calls,
            prefill_seconds_this_invocation=self.conditions.prefill_seconds,
            actual_frozen_input_dtype=self.conditions.observed_memory_dtype,
            prepared_context_builds_this_invocation=dict(self.actor.prepared_context_builds),
            cache_boundary=self.conditions.identity['frozen_cache_boundary'],
            architecture_id=self.args.architecture, prediction_type='x0',
            teachers_or_post_head_used_in_RL=False)


ArchitectureRuntime = FourFrameworkRuntime
