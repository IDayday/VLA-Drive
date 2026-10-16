"""Frozen TRAIN-only membership with the source epoch permutation."""
import json
import math
from pathlib import Path
from .contracts import POPULATION,file_sha,fingerprint,SAFETY25_VARIANT,SAFETY25_SAMPLING
from .vendor.samplers import ResumableEpochSampler


class SceneStream:
    def __init__(self,manifest,world,*,qualification=False):
        self.manifest=json.loads(Path(manifest).read_text());self.sha=file_sha(manifest)
        pop=json.loads(Path(POPULATION).read_text());self.logs=pop['token_logs']
        self.tokens=self.manifest['tokens'];self.world=world
        if not self.tokens or len(self.tokens)!=len(set(self.tokens)) or not set(self.tokens)<=set(pop['train_tokens']):
            raise ValueError('Only unique RL_TRAIN members may enter an update')
        if self.manifest['population_identity']!=pop['identity_sha256']:raise ValueError('Split identity changed')
        if not qualification and self.manifest.get('status')!='COMPLETE_FIXED_SFT_SINGLE_NATIVE_ODE_HARD_SUBSET':
            raise ValueError('Formal training needs the complete registered hard screen')
        self.sampler=ResumableEpochSampler(self.tokens,seed=42,batch_size=1,num_processes=world)
        self.epoch=-1;self.order=[]
        self.epoch_size=((len(self.tokens)+world-1)//world)*world

    def at(self,global_cursor,rank):
        if global_cursor%self.world:raise ValueError('Scene cursor must be at a fresh global-rollout boundary')
        epoch,offset=divmod(global_cursor,self.epoch_size)
        if epoch!=self.epoch:
            self.sampler.set_epoch(epoch);order=list(self.sampler)
            # Source Accelerate BatchSamplerShard defaults to even_batches:
            # repeat the initial samples only to complete the final global batch.
            self.order=(order*((self.epoch_size+len(order)-1)//len(order)))[:self.epoch_size]
            self.epoch=epoch
        i=offset+rank
        return self.tokens[self.order[i]],dict(epoch=epoch,index=i,padding=i>=len(self.tokens))


class FixedSafetyMixtureStream(SceneStream):
    """Static SFT strata, fresh current-policy trajectories, no reward filtering.

    A global cursor fully determines both stratum cursors and rank assignment.
    Source permutations use private CPU generators, never actor/sampling RNG.
    """
    def __init__(self, manifest, world, *, qualification=False):
        super().__init__(manifest, world, qualification=qualification)
        if world != 8:
            raise ValueError('The registered 6+2 exposure control requires world8')
        rows = self.manifest.get('rows', [])
        by_token = {r['token']: r for r in rows}
        if len(rows) != len(by_token) or set(by_token) != set(self.tokens):
            raise ValueError('Fixed SFT strata require one authenticated score row per TRAIN token')
        self.strata = {'safe': [], 'unsafe': []}
        keys = SAFETY25_SAMPLING['reference_metrics']
        for token in self.tokens:
            values = [float(by_token[token]['metrics'][k]) for k in keys]
            if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
                raise ValueError('Unknown/invalid SFT safety labels are not a stratum')
            label = 'safe' if all(v == 1 for v in values) else 'unsafe'
            self.strata[label].append(token)
        if len(self.strata['safe']) < 6 or len(self.strata['unsafe']) < 2:
            raise ValueError('Insufficient real TRAIN members for the fixed quota')
        self._samplers = {k: ResumableEpochSampler(v,seed=42,batch_size=1,num_processes=1)
                          for k,v in self.strata.items()}
        self._epochs = {}; self._orders = {}
        self.identity = dict(schema=SAFETY25_SAMPLING['schema'],manifest_sha256=self.sha,
            reference_population_identity=self.manifest['population_identity'],
            reference_screen_identity=self.manifest.get('screen_identity'),
            cohort_sizes={k:len(v) for k,v in self.strata.items()},
            cohort_token_sha256={k:fingerprint(v) for k,v in self.strata.items()},
            safe_groups=6,unsafe_groups=2,rank_rotation='slot=(rank-collection_zero_based)%8',
            permutation='unchanged source ResumableEpochSampler seed42+stratum_epoch',
            cursor='stratum cursor = collection*quota + within-stratum slot',
            padding='none; each stratum continues across its own permutation boundary',
            actor_input='unchanged; stratum and score are not actor features',
            importance_correction=False)

    def at(self, global_cursor, rank):
        if global_cursor < 0 or global_cursor % 8 or not 0 <= rank < 8:
            raise ValueError('Invalid fresh global cursor/rank')
        collection = global_cursor // 8
        slot = (rank - collection) % 8
        label = 'unsafe' if slot < 2 else 'safe'
        quota = 2 if label == 'unsafe' else 6
        within = slot if label == 'unsafe' else slot - 2
        cursor = collection * quota + within
        epoch, index = divmod(cursor, len(self.strata[label]))
        if self._epochs.get(label) != epoch:
            sampler = self._samplers[label]; sampler.set_epoch(epoch)
            self._orders[label] = list(sampler); self._epochs[label] = epoch
        token = self.strata[label][self._orders[label][index]]
        return token, dict(scheme=SAFETY25_SAMPLING['schema'],stratum=label,
            stratum_cursor=cursor,epoch=epoch,index=index,padding=False,
            collection=collection,rank_slot=slot)


def make_scene_stream(cfg, manifest, world, *, qualification=False):
    if cfg['variant'] == SAFETY25_VARIANT:
        if cfg.get('scene_sampling') != SAFETY25_SAMPLING:
            raise ValueError('Changed registered exposure contract')
        return FixedSafetyMixtureStream(manifest, world, qualification=qualification)
    if cfg['variant'] not in ('R_SOURCE','R_CANONICAL','R_CANONICAL_X1','R_CANONICAL_LAST_ODE'):
        raise ValueError('Unregistered scene stream')
    return SceneStream(manifest,world,qualification=qualification)
