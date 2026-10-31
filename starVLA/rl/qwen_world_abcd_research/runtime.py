"""Independent current-best-parent ABCD RL lifecycle; original campaign intact."""
import copy
from pathlib import Path

import yaml

from starVLA.rl.flow_grpo.reproducibility import configure_numerics
from starVLA.rl.qwen_world_compare.runtime import FourFrameworkRuntime, FrozenCurrentConditions, source_files as original_sources
from starVLA.rl.qwen_world_compare.recovery_state import validate_boundary, validate_ranks
from starVLA.rl.qwen_world_compare.contracts import ROOT, file_sha, boundary_stop_due, require_dispatch
from starVLA.rl.qwen_world_postcot.data import WorldSceneData
from .parents import load_parent_model, validate_best_selection
from .actor import ResearchActor
from .data import FullTRAINTokenStream


def training_dispatch():
    host = require_dispatch()
    if boundary_stop_due(host):
        from tools.qwen_world_abcd_research.train import stop
        stop()
    return host


class ABCDBestRuntime(FourFrameworkRuntime):
    require_dispatch = staticmethod(training_dispatch)
    validate_boundary = staticmethod(validate_boundary)
    validate_ranks = staticmethod(validate_ranks)

    def source_files(self):
        result = original_sources()
        for folder in ('starVLA/rl/qwen_world_abcd_research', 'tools/qwen_world_abcd_research'):
            for path in Path(folder).rglob('*.py'):
                result[str(path)] = file_sha(path)
        return dict(sorted(result.items()))

    def load_recipe(self, args):
        configure_numerics()
        cfg = yaml.safe_load(Path(args.config).read_text())
        if (cfg['architecture_id'] != args.architecture or cfg['seed'] != 42
                or cfg['research'].get('schema') != 'ABCD_current_best_RL_research_v1'):
            raise ValueError('Wrong registered architecture/research recipe')
        selected = validate_best_selection(args.parent_selection, args.il_checkpoint, args.architecture)
        self.selection = dict(path=str(Path(args.parent_selection).resolve()),
            file_sha256=file_sha(args.parent_selection), selected=selected,
            schema='user_current_NAVTEST_best_NOT_100k_DEV_selection', Test_used_for_selection=True)
        self.args, self.cfg = args, cfg
        expected = Path(selected['deployment_parent'])
        if Path(args.il_checkpoint).resolve() != expected.resolve():
            raise ValueError('Selected export parent changed')
        # Payload bytes are verified once in make_actor, not redundantly during
        # metadata/selection setup. The actual load remains strict.
        import json
        self.parent = copy.deepcopy(json.loads((expected/'export.json').read_text())['IL_parent'])
        expected_data = Path(self.parent['identity']['config']['data_root'])/'TRAIN'
        if args.data is None:
            args.data = str(expected_data)
        if Path(args.data).resolve() != expected_data.resolve():
            raise ValueError('Current-best RL must preserve own legal TRAIN role')
        if file_sha(args.manifest) != cfg['research']['training_manifest_sha256']:
            raise ValueError('Full TRAIN manifest changed')
        return cfg

    def make_scene_stream(self, cfg, args, world):
        stream = FullTRAINTokenStream(args.manifest, world)
        self.stream_identity = stream.identity
        return stream

    def make_reward_service(self):
        from starVLA.rl.simwam_recipe.reward_readouts import RewardService
        return RewardService(Path.cwd(), workers=1,
            cache_roots=self.cfg['research']['scoring_cache_roots'])

    def make_actor(self, cfg, args, device):
        model, parent = load_parent_model(args.il_checkpoint, device=device)
        if parent['model_sha256'] != self.selection['selected']['native_model_sha256']:
            raise ValueError('Loaded own IL model differs from registered best')
        self.parent = parent
        actor = ResearchActor(model, cfg)
        self.conditions = FrozenCurrentConditions(model, WorldSceneData(args.data, role='TRAIN'), parent)
        object.__setattr__(actor, 'conditions', self.conditions)
        self.actor = actor
        return actor, None, None

    def recipe_identity(self, cfg, args, condition_sha):
        value = super().recipe_identity(cfg, args, condition_sha)
        value.update(research_schema='ABCD_current_best_RL_research_v1',
            training_manifest_sha256=file_sha(args.manifest),
            training_data_role='full legal TRAIN90890/147logs; DEV1696, embargoed TRAIN and NAVTEST excluded',
            initialization='own currently best complete NAVTEST IL export; fresh Adam/update0/seed42',
            reference_storage='independent immutable reference module graph; frozen base tensors shared, current LoRA never shared',
            reference='own selected IL permanently frozen; independent module graph with shared immutable frozen parameters',
            parent_native_recovery_not_required=True,
            current_best_user_selection_supersedes_original100k_gate_for_this_new_run_only=True)
        return value
