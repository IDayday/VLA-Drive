"""Current-only scene reader and stateless, cursor-recoverable sampling streams."""
import json
import math
from pathlib import Path
import pickle
import random

from PIL import Image

from .manifest import file_sha, fingerprint, read_sealed, physical, SCHEMA

class ResearchSceneData:
    """Compatible with FrozenWorldConditions/MemoryLearningConditions interfaces."""
    def __init__(self, root, role='NAVTRAIN'):
        root = Path(root)
        self.manifest = json.loads((root/'manifest.json').read_text())
        complete = json.loads((root/'COMPLETE').read_text())
        if (self.manifest['schema']!='world_rl_research_current_observations_v1'
                or self.manifest['cohort'] != role
                or complete['manifest_sha256'] != file_sha(root/'manifest.json')
                or self.manifest['records_sha256'] != file_sha(root/'records.pkl')):
            raise ValueError('Research current observations changed or incomplete')
        pop_path = Path(self.manifest['source_population_path'])
        if file_sha(pop_path) != self.manifest['population_sha256']:
            raise ValueError('Source population changed')
        population = json.loads(pop_path.read_text())
        legal = set(population['train_tokens']+population['dev_tokens'])
        tokens = self.manifest['tokens']
        if set(tokens) != legal or len(tokens) != len(legal):
            raise ValueError('Research data must contain the complete explicit NAVTRAIN population')
        forbidden_physical = {physical(population['token_logs'][t]) for t in population['navtest_tokens']}
        if any(physical(population['token_logs'][t]) in forbidden_physical for t in tokens):
            raise ValueError('NAVTEST physical log entered NAVTRAIN observations')
        with (root/'records.pkl').open('rb') as stream:
            self.records = pickle.load(stream)
        if set(self.records) != set(tokens):
            raise ValueError('Research current records have incomplete population')
        self.tokens = tokens
        self.sha = complete['manifest_sha256']
        self.logs = dict(self.manifest['token_logs'])
        self.epoch = -1

    def observation(self, token):
        row = self.records[token]
        images = {}
        for camera in ('cam_l0','cam_f0','cam_r0'):
            path = row['images'].get(camera)
            if path is None:
                images[camera] = None
            else:
                with Image.open(path) as image:
                    images[camera] = image.convert('RGB')
        return dict(images=images, proprio=row['proprio'], navigation=row['navigation'])

    def target(self, token):
        import torch
        return torch.tensor(self.records[token]['requested_GT'],dtype=torch.float32)

    def index(self, position):
        if position < 0:
            raise ValueError('Negative scene position')
        epoch, offset = divmod(position,len(self.tokens))
        if epoch != self.epoch:
            self.order = list(range(len(self.tokens)))
            random.Random(42+epoch).shuffle(self.order)
            self.epoch = epoch
        return self.order[offset]

class SceneStream:
    """Full/hard sampling with exact source epoch order, or sealed stage censuses.

    `at(global_cursor, rank)` is a pure function of immutable manifest/stage data.
    Absolute scene cursor is the only state needed at a saved optimizer boundary.
    A future adaptive stage must be sealed before admission; missing stages fail
    explicitly rather than falling back to a changed population.
    """
    def __init__(self, manifest, world):
        self.path = Path(manifest)
        self.manifest = read_sealed(self.path)
        m = self.manifest
        self.world = int(world)
        if self.world <= 0 or m['schema'] != SCHEMA:
            raise ValueError('Invalid research stream schema or world size')
        self.tokens = m['tokens']
        if not self.tokens or len(self.tokens) != len(set(self.tokens)) or len(self.tokens) != m['count']:
            raise ValueError('Invalid stream token census')
        data_path = Path(m['data_root'])/'manifest.json'
        if file_sha(data_path) != m['data_manifest_sha256']:
            raise ValueError('Stream observation identity changed')
        data = json.loads(data_path.read_text())
        if not set(self.tokens) <= set(data['tokens']):
            raise ValueError('Stream has scenes without legal current observations')
        pop_path = Path(m['population_manifest'])
        if file_sha(pop_path) != m['population_sha256']:
            raise ValueError('Source population identity changed')
        pop = json.loads(pop_path.read_text())
        legal = set(pop['train_tokens']+pop['dev_tokens'])
        forbidden = set(pop['navtest_tokens'])
        test_physical = {physical(pop['token_logs'][t]) for t in forbidden}
        if (not set(self.tokens) <= legal or set(self.tokens)&forbidden
                or any(physical(pop['token_logs'][t]) in test_physical for t in self.tokens)):
            raise ValueError('NAVTEST scene or physical log entered research RL')
        self.logs = pop['token_logs']; self.sha = file_sha(self.path)
        self.mode = m['mode']; self.epoch = -1; self.order = []
        self.epoch_size = math.ceil(len(self.tokens)/self.world)*self.world
        self.stage_exposures = m.get('stage_exposures')
        if self.mode == 'adaptive' and (not self.stage_exposures or self.stage_exposures % self.world):
            raise ValueError('Adaptive stage exposures must be a positive multiple of world size')
        self.identity = dict(schema=SCHEMA,manifest_sha256=self.sha,mode=self.mode,
                             scene_count=len(self.tokens),world=self.world,
                             physical_log_count=len({physical(self.logs[t]) for t in self.tokens}),
                             old_DEV_is_train_diagnostic=m['old_DEV_is_train_diagnostic'],
                             NAVTEST_token_and_physical_disjoint=True,
                             ordering='source ResumableEpochSampler seed and epoch padding' if self.mode=='uniform' else 'append-only sealed per-stage ordered census',
                             stage_exposures=self.stage_exposures, stage_registry=m.get('stage_registry'))
        self._stage_cache = {}

    def _uniform_order(self, epoch):
        if epoch != self.epoch:
            # Preserve hard-control ordering exactly; only explicitly selected
            # population membership changes between uniform experiment arms.
            from starVLA.rl.simwam_recipe.vendor.samplers import ResumableEpochSampler
            sampler = ResumableEpochSampler(self.tokens, seed=self.manifest['seed'],
                                             batch_size=1,num_processes=self.world)
            sampler.set_epoch(epoch)
            order = list(sampler)
            self.order = (order*math.ceil(self.epoch_size/len(order)))[:self.epoch_size]
            self.epoch = epoch
        return self.order

    def _stage(self, stage_index):
        if stage_index not in self._stage_cache:
            path = Path(self.manifest['stage_registry'])/f'stage{stage_index:06}.json'
            if not path.exists():
                raise FileNotFoundError('Next adaptive stage not sealed; controller must pause/admit it: '+str(path))
            stage = read_sealed(path)
            if (stage['schema']!='world_rl_research_stage_census_v1'
                    or stage['stream_manifest_sha256']!=self.sha
                    or stage['stage_index']!=stage_index
                    or stage['start_cursor']!=stage_index*self.stage_exposures
                    or len(stage['ordered_tokens'])!=self.stage_exposures
                    or not set(stage['ordered_tokens']) <= set(self.tokens)):
                raise ValueError('Changed or invalid adaptive stage census')
            self._stage_cache[stage_index] = stage
        return self._stage_cache[stage_index]

    def at(self, global_cursor, rank):
        if global_cursor < 0 or global_cursor % self.world or not 0 <= rank < self.world:
            raise ValueError('Invalid research global scene cursor or rank')
        position = global_cursor+rank
        if self.mode == 'uniform':
            epoch, offset = divmod(position,self.epoch_size)
            token = self.tokens[self._uniform_order(epoch)[offset]]
            return token,dict(epoch=epoch,index=offset,padding=offset>=len(self.tokens))
        stage_index, offset = divmod(position,self.stage_exposures)
        stage = self._stage(stage_index)
        return stage['ordered_tokens'][offset],dict(stage=stage_index,index=offset,
                   stage_sha256=stage['sha256'],padding=False,mode=stage['mode'])

    def token_at(self, position):
        if position < 0:
            raise ValueError('Negative scene position')
        cursor, rank = divmod(position,self.world)
        return self.at(cursor*self.world,rank)[0]

    def token(self, update, rank):
        return self.at(update*self.world,rank)[0]
