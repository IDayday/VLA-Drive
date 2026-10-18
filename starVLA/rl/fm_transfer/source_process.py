"""Registered source-data/README-budget replication; original contracts stay intact.

The actor, sampler, source loss, AdamW, four-inner transaction and deployment
implementations are unchanged. This runtime alone admits the source token list
and the separately authorized 170000-update budget.
"""
import copy
import json
from pathlib import Path
from omegaconf import OmegaConf
from .rl_runtime import TransferRuntime
from .contracts import require_dispatch
from starVLA.rl.simwam_recipe.contracts import load_recipe, file_sha, fingerprint, POPULATION
from starVLA.rl.simwam_recipe.vendor.samplers import ResumableEpochSampler

ROOT = Path('/mnt/project/simwam-source-process-20261018')
UPSTREAM = Path('/mnt/project/simwam-recipe-20261015/upstream/SimWAM')
SOURCE_LIST = 'navsim/navsim/planning/script/config/common/train_test_split/scene_filter/navtrain_pdm_score_below0p9.yaml'
B_SFT = Path('/mnt/project/fm-transfer-20261016/runs/B_SOURCE_ACTION_DIT_SFT_seed42_v1/checkpoints/update100000')
OWN_MANIFEST = Path('/mnt/project/fm-transfer-20261016/hard_manifests/B_SOURCE_ACTION_DIT_SFT_update100000.json')
MODES = ('UPSTREAM_DATA', 'OWN_DATA_CONTROL')
MILESTONES = [0, 5000, 20000, 50000, 100000, 170000]


def expected_config(mode):
    if mode not in MODES:
        raise ValueError('Only the two registered data populations')
    repo = Path(__file__).resolve().parents[3]
    cfg = copy.deepcopy(load_recipe(repo/'configs/flow_rl/simwam_source_ddp.yaml'))
    cfg.update(max_steps=170000, evaluate_updates=MILESTONES,
        source_process=dict(schema='simwam_source_process_v1', data_mode=mode,
            budget_source='locked README max_steps=170000; not claimed to be the released 20k weight budget',
            original_default_updates=20000, new_candidates=2720000,
            fresh_collections=42500, scene_exposures=340000,
            source_data_exclusions='locked RL_DEV only; never Navtest; no reward-based filtering',
            seed=42, evaluation_seed=2101))
    return cfg


def load_process_recipe(path):
    cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if cfg != expected_config(cfg.get('source_process', {}).get('data_mode')):
        raise ValueError('Changed complete registered source-process recipe')
    return cfg


def authenticate_manifest(path, population=None):
    obj = json.loads(Path(path).read_text())
    body = dict(obj); sha = body.pop('sha256')
    pop = population or json.loads(Path(POPULATION).read_text())
    if fingerprint(body) != sha or obj['status'] != 'COMPLETE_UPSTREAM_DATA_TRAIN_ONLY':
        raise ValueError('Unsealed upstream-data population')
    tokens = obj['tokens']
    if (not tokens or len(set(tokens)) != len(tokens) or
            not set(tokens) <= set(pop['train_tokens']) or
            set(tokens) & (set(pop['dev_tokens']) | set(pop['navtest_tokens'])) or
            obj['population_identity'] != pop['identity_sha256']):
        raise ValueError('Invalid or leaking source training population')
    if (obj['upstream_commit'] != '1433ee7e06a62bd8b89da74455df0d82bda7b726' or
            len(tokens) != obj['retained_scenes'] or
            obj['source_scenes'] != obj['retained_scenes'] + obj['excluded_DEV_scenes'] or
            obj['missing_source_scenes'] != 0 or obj['source_Navtest_overlap'] != 0):
        raise ValueError('Source membership accounting changed')
    return obj, pop


class SourceTokenStream:
    """Source loader order, then unchanged source epoch permutation and padding."""
    def __init__(self, manifest, world):
        self.manifest, pop = authenticate_manifest(manifest)
        self.sha = file_sha(manifest); self.logs = pop['token_logs']
        self.tokens = self.manifest['tokens']; self.world = world
        self.sampler = ResumableEpochSampler(self.tokens, seed=42, batch_size=1, num_processes=world)
        self.epoch = -1; self.order = []
        self.epoch_size = ((len(self.tokens)+world-1)//world)*world
        self.identity = dict(schema='source_hard_tokens_TRAIN_only_stream_v1',
            manifest_sha256=self.sha, scene_count=len(self.tokens),
            source_yaml_sha256=self.manifest['source_yaml_sha256'],
            source_loader_sha256=self.manifest['source_loader_sha256'],
            original_source_epoch_padding=True, ordering='source SceneLoader on recorded local log enumeration')

    def at(self, global_cursor, rank):
        if global_cursor < 0 or global_cursor % self.world or not 0 <= rank < self.world:
            raise ValueError('Invalid global scene cursor or rank')
        epoch, offset = divmod(global_cursor, self.epoch_size)
        if epoch != self.epoch:
            self.sampler.set_epoch(epoch); order = list(self.sampler)
            self.order = (order*((self.epoch_size+len(order)-1)//len(order)))[:self.epoch_size]
            self.epoch = epoch
        index = offset+rank
        return self.tokens[self.order[index]], dict(epoch=epoch, index=index, padding=index>=len(self.tokens))


class SourceProcessRuntime(TransferRuntime):
    def load_recipe(self, args):
        cfg = load_process_recipe(args.config)
        self.sft_checkpoint = Path(args.sft_checkpoint)
        if self.sft_checkpoint.resolve() != B_SFT.resolve():
            raise ValueError('Both process replicas use the fixed complete B100k SFT')
        self.sft_seal = json.loads((self.sft_checkpoint/'COMPLETE').read_text())
        self.sft_files = json.loads((self.sft_checkpoint/'files.json').read_text())
        self.sft_identity = json.loads((self.sft_checkpoint/'identity.json').read_text())
        if (self.sft_seal['phase'] != 'READY' or self.sft_seal['updates'] != 100000 or
                self.sft_identity['qualification'] or
                file_sha(self.sft_checkpoint/'files.json') != self.sft_seal['files_sha256'] or
                file_sha(self.sft_checkpoint/'identity.json') != self.sft_files['identity.json'] or
                self.sft_files['model.pt'] != '4e71b1fcf963d38456174de1f8d17762a9a4ccf1597c22b7cec262026d9201b7'):
            raise ValueError('Changed formal SFT parent')
        self.process_mode = cfg['source_process']['data_mode']
        return cfg

    def make_scene_stream(self, cfg, args, world):
        if self.process_mode == 'OWN_DATA_CONTROL':
            if Path(args.manifest).resolve() != OWN_MANIFEST.resolve():
                raise ValueError('The control retains the original full-screen hard manifest')
            return super().make_scene_stream(cfg, args, world)
        if Path(args.manifest).resolve() != (ROOT/'source_manifest.json').resolve():
            raise ValueError('Wrong registered upstream manifest')
        return SourceTokenStream(args.manifest, world)

