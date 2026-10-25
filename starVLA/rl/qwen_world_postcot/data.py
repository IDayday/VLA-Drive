"""Frozen data roles; only current legal fields are exposed to the actor."""
import json
from pathlib import Path
import pickle
import re
from PIL import Image
import torch
from .contracts import file_sha, fingerprint
from starVLA.rl.simwam_recipe.vendor.samplers import ResumableEpochSampler

POPULATION = Path('/mnt/project/flow-safe-improvement-20261013/population/manifest.json')
SOURCE_HARD_MANIFEST = Path('/mnt/project/simwam-source-process-20261018/source_manifest.json')

class WorldSceneData:
    def __init__(self, root, role='TRAIN'):
        root=Path(root);self.manifest=json.loads((root/'manifest.json').read_text());seal=json.loads((root/'COMPLETE').read_text())
        if self.manifest['cohort']!=role or seal['manifest_sha256']!=file_sha(root/'manifest.json') or file_sha(root/'records.pkl')!=self.manifest['records_sha256']:raise ValueError('Unsealed or changed current data')
        self.records=pickle.load(open(root/'records.pkl','rb'));self.tokens=self.manifest['tokens'];self.sha=seal['manifest_sha256'];self.epoch=-1
        if set(self.tokens)!=set(self.records):raise ValueError('Incomplete data coverage')
    def observation(self, token):
        row=self.records[token];images={}
        for camera,path in row['images'].items():
            if path is None:images[camera]=None
            else:
                with Image.open(path) as im:images[camera]=im.convert('RGB')
        return dict(images=images,proprio=row['proprio'],navigation=row['navigation'])
    def target(self, token):return torch.tensor(self.records[token]['requested_GT'],dtype=torch.float32)
    def index(self, position):
        epoch,offset=divmod(position,len(self.tokens))
        if epoch!=self.epoch:self.order=torch.randperm(len(self.tokens),generator=torch.Generator().manual_seed(42+epoch)).tolist();self.epoch=epoch
        return self.order[offset]

class WorldRLTokenStream:
    def __init__(self, manifest, world):
        path=Path(manifest);self.manifest=json.loads(path.read_text());self.tokens=self.manifest['tokens'];self.sha=file_sha(path);self.world=int(world);self.epoch=-1
        if (self.world <= 0 or self.manifest['schema']!='world_postcot_rl_tokens_v1'
                or not self.tokens or len(self.tokens)!=len(set(self.tokens))
                or len(self.tokens)!=self.manifest['count']):raise ValueError('Wrong RL data identity')
        roles=json.loads((path.parent/'DATA_PARTITION.json').read_text())
        train_path=path.parent/'TRAIN/manifest.json';train=json.loads(train_path.read_text())
        train_complete=json.loads((path.parent/'TRAIN/COMPLETE').read_text())
        pop=json.loads(POPULATION.read_text());original=json.loads(SOURCE_HARD_MANIFEST.read_text())
        if (self.manifest['role_partition_sha256']!=fingerprint(roles)
                or train['role_partition_sha256']!=fingerprint(roles)
                or file_sha(train_path)!=self.manifest['train_manifest_sha256']
                or train_complete['manifest_sha256']!=file_sha(train_path)
                or roles['population_sha256']!=file_sha(POPULATION)
                or self.manifest['parent_manifest_sha256']!=file_sha(SOURCE_HARD_MANIFEST)):
            raise ValueError('RL source, role partition or sealed TRAIN manifest changed')
        legal=set(roles['train_tokens'])
        if train['cohort']!='TRAIN' or set(train['tokens'])!=legal:
            raise ValueError('RL must use the exact locked full-framework TRAIN role')
        expected=[token for token in original['tokens'] if token in legal]
        if self.tokens!=expected or set(self.tokens)&(set(roles['dev_tokens'])|set(roles['navtest_tokens'])):
            raise ValueError('RL membership/order differs from the TRAIN-only source intersection')
        physical=lambda token:re.sub(r'_\d+_\d+$','',pop['token_logs'][token])
        train_logs={physical(token) for token in legal}
        heldout_logs={physical(token) for token in roles['dev_tokens']+roles['navtest_tokens']}
        if train_logs&heldout_logs:
            raise ValueError('Physical log leakage into full-framework RL TRAIN')
        self.logs=pop['token_logs'];self.order=[]
        self.sampler=ResumableEpochSampler(self.tokens,seed=42,batch_size=1,num_processes=self.world)
        self.epoch_size=((len(self.tokens)+self.world-1)//self.world)*self.world
        self.identity=dict(schema='world_postcot_RL_hard_TRAIN_only_stream_v1',manifest_sha256=self.sha,
            scene_count=len(self.tokens),parent_manifest_sha256=self.manifest['parent_manifest_sha256'],
            role_partition_sha256=self.manifest['role_partition_sha256'],train_manifest_sha256=self.manifest['train_manifest_sha256'],
            original_source_epoch_padding=True,physical_log_disjoint=True,physical_log_count=len({physical(t) for t in self.tokens}),
            ordering='original source retained order; unchanged ResumableEpochSampler seed42 and epoch padding')
    def at(self, global_cursor, rank):
        if global_cursor<0 or global_cursor%self.world or not 0<=rank<self.world:
            raise ValueError('Invalid global scene cursor or rank')
        epoch,offset=divmod(global_cursor,self.epoch_size)
        if epoch!=self.epoch:
            self.sampler.set_epoch(epoch);order=list(self.sampler)
            self.order=(order*((self.epoch_size+len(order)-1)//len(order)))[:self.epoch_size];self.epoch=epoch
        index=offset+rank
        return self.tokens[self.order[index]],dict(epoch=epoch,index=index,padding=index>=len(self.tokens))
    def token_at(self, position):
        if position<0:raise ValueError('Scene position must be nonnegative')
        epoch,offset=divmod(position,self.epoch_size)
        if epoch!=self.epoch:
            self.sampler.set_epoch(epoch);order=list(self.sampler)
            self.order=(order*((self.epoch_size+len(order)-1)//len(order)))[:self.epoch_size];self.epoch=epoch
        return self.tokens[self.order[offset]]
    def token(self, update, rank):return self.at(update*self.world,rank)[0]
