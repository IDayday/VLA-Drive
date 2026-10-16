"""Explicit new-base dependencies for the unchanged source recipe train loop.

The original entry still rejects replacement bases. This entry binds its own
complete SFT, own TRAIN difficulty screen, and architecture to scientific
identity; all sampler, probability, advantage, loss and recovery code is shared.
"""
import copy
import json
from pathlib import Path
from omegaconf import OmegaConf
from .contracts import require_dispatch, source_files as transfer_sources
from .rl_adapter import make_transfer_actor, TransferConditions
from starVLA.rl.simwam_recipe.contracts import load_recipe, file_sha, UPSTREAM_COMMIT, POPULATION
from starVLA.rl.simwam_recipe.data_stream import SceneStream
from starVLA.rl.constrained_flow_noise.loading import DEFAULT_SOURCE


class TransferRuntime:
    require_dispatch = staticmethod(require_dispatch)

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('--sft-checkpoint', required=True,
                            help='Native new supervised endpoint; never a merged RL model')

    def load_recipe(self, args):
        cfg = load_recipe(args.config)
        if cfg['variant'] != 'R_SOURCE':
            raise ValueError('Both registered new bases use the identical source RL recipe')
        checkpoint = Path(args.sft_checkpoint)
        self.sft_checkpoint = checkpoint
        self.sft_seal = json.loads((checkpoint/'COMPLETE').read_text())
        self.sft_files = json.loads((checkpoint/'files.json').read_text())
        self.sft_identity = json.loads((checkpoint/'identity.json').read_text())
        if ('.incomplete' in checkpoint.name or self.sft_seal['phase'] != 'READY' or
                file_sha(checkpoint/'files.json') != self.sft_seal['files_sha256'] or
                file_sha(checkpoint/'identity.json') != self.sft_files['identity.json']):
            raise ValueError('Unpublished or changed supervised identity')
        # The large model content is checked again by load_supervised_policy.
        if not args.qualification and (self.sft_identity['qualification'] or
                self.sft_seal['updates'] != self.sft_identity['config']['max_updates']):
            raise ValueError('Formal RL requires a completed formal supervised endpoint')
        return cfg

    def make_scene_stream(self, cfg, args, world):
        stream = SceneStream(args.manifest, world, qualification=args.qualification)
        manifest = stream.manifest
        expected = dict(SFT_model_sha256=self.sft_files['model.pt'],
                        SFT_identity_sha256=self.sft_identity['sha256'],
                        architecture=self.sft_identity['config']['architecture'],
                        seed=2101, grid='native', threshold=.9,
                        statistic='one canonical v1.1 score per scene')
        if manifest.get('screen_contract') != expected:
            raise ValueError('Hard scenes must come from this new SFT, not the historical base or a G8 statistic')
        population=json.loads(Path(POPULATION).read_text())
        if not args.qualification and manifest.get('screened_scene_count') != len(population['train_tokens']):
            raise ValueError('Formal hard subset requires the complete locked TRAIN screen')
        stream.identity = dict(schema='fm_transfer_fixed_new_SFT_hard_population_v1',
            screen_contract=expected, manifest_sha256=file_sha(args.manifest),
            scene_count=len(stream.tokens), original_source_epoch_padding=True)
        return stream

    def make_actor(self, cfg, args, device):
        actor, self.loaded_sft = make_transfer_actor(args.sft_checkpoint, cfg,
            device=device, qualification=args.qualification)
        # The original four-channel encoding and reward decode contract remain.
        sft_config = OmegaConf.load(DEFAULT_SOURCE)
        return actor, cfg, sft_config

    def make_conditions(self, actor):
        return TransferConditions(actor, self.loaded_sft)

    def recipe_identity(self, cfg, args, condition_sha):
        resolved = copy.deepcopy(cfg)
        resolved['supervised_transfer'] = dict(
            variant=self.sft_identity['config']['variant'],
            architecture=self.sft_identity['config']['architecture'],
            SFT_identity=self.loaded_sft,
            action='original DDP normalized eight-by-four',
            condition='same frozen original Qwen normalized action queries',
            source_action_forward='standalone ActionDiT.forward; no Wan/MoT',
            permanent_reference='this new frozen SFT, cloned before LoRA insertion')
        return dict(schema='fm_transfer_source_RL_v1', recipe=resolved,
            upstream_commit=UPSTREAM_COMMIT, SFT_sha256=self.sft_files['model.pt'],
            hard_manifest_sha256=file_sha(args.manifest),
            projected_condition_identity=condition_sha,
            policy_ratio='unchanged source relative-clamped dimension-mean stochastic-transition surrogate',
            deployment='same current-observation VLAAgent; own native10 and shift5 ODE, separately reported')

    @staticmethod
    def source_files():
        # Bind both the reused trainer and all new interfaces. Existing source
        # identity checks and explicit migration receipts are not bypassed.
        from scripts.analysis.simwam_recipe.train import source_files
        return {**source_files(), **transfer_sources()}
