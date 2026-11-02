"""Full legal NAVTRAIN; former DEV is explicitly training, never NAVTEST."""
import json
import math
import re
from pathlib import Path
import torch
from starVLA.rl.qwen_world_postcot.data import WorldSceneData
from starVLA.rl.qwen_world_compare.contracts import file_sha


class DataStream:
    def __init__(self,cfg,global_batch):
        self.data=WorldSceneData(cfg['data'],role='NAVTRAIN')
        self.tokens=self.data.tokens;self.n=len(self.tokens);self.global_batch=global_batch
        manifest=self.data.manifest;self.logs=manifest['token_logs']
        population=json.loads(Path(manifest['source_population_path']).read_text())
        testtokens=set(population['navtest_tokens'])
        logmap=population['token_logs']
        physical=lambda token:re.sub(r'_\d+_\d+$','',logmap[token])
        trainlogs={physical(t)for t in self.tokens};testlogs={physical(t)for t in testtokens}
        if self.n!=103288 or len(set(self.tokens))!=self.n or set(self.tokens)&testtokens or trainlogs&testlogs:
            raise ValueError('Full legal NAVTRAIN/NAVTEST population or physical-log isolation failed')
        self.padded=math.ceil(self.n/global_batch)*global_batch;self.cache={}
        self.identity=dict(count=self.n,manifest_sha256=self.data.sha,full_population_sha256=file_sha(cfg['population_manifest']),
            TRAIN_physical_logs=len(trainlogs),NAVTEST_physical_logs=len(testlogs),NAVTEST_overlap=0,
            former_DEV_is_TRAIN=True,padding_per_epoch=self.padded-self.n)

    def at(self,cursor,rank,local_scene,world):
        position=cursor+local_scene*world+rank;epoch,index=divmod(position,self.padded)
        if epoch not in self.cache:
            self.cache={epoch:torch.randperm(self.n,generator=torch.Generator().manual_seed(42+epoch)).tolist()}
        order=self.cache[epoch];return self.tokens[order[index%self.n]],index>=self.n
