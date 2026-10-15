"""Frozen TRAIN-only membership with the source epoch permutation."""
import json
from pathlib import Path
from .contracts import POPULATION,file_sha,fingerprint
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
