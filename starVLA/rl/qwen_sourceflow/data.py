"""Sealed raw observations; targets are kept out of actor input dictionaries."""
import json
from pathlib import Path
import pickle
from PIL import Image
import torch
from .contracts import file_sha, POPULATION


class RawSceneData:
    def __init__(self, root, *, cohort):
        root=Path(root);self.manifest=json.loads((root/'manifest.json').read_text())
        seal=json.loads((root/'COMPLETE').read_text())
        if file_sha(root/'manifest.json')!=seal['manifest_sha256'] or file_sha(root/'records.pkl')!=self.manifest['records_sha256']:
            raise ValueError('Unsealed raw scene data')
        if self.manifest['cohort']!=cohort or self.manifest['population_sha256']!=file_sha(POPULATION):
            raise ValueError('Wrong or changed data population')
        with (root/'records.pkl').open('rb') as f:self.records=pickle.load(f)
        self.tokens=self.manifest['tokens'];self.sha=seal['manifest_sha256']
        self.epoch=-1;self.order=[]

    def index(self, position):
        epoch,offset=divmod(position,len(self.tokens))
        if epoch!=self.epoch:
            self.order=torch.randperm(len(self.tokens),generator=torch.Generator().manual_seed(42+epoch)).tolist();self.epoch=epoch
        return self.order[offset]

    def observation(self, token):
        row=self.records[token];images={}
        for name,path in row['images'].items():
            if path is None:images[name]=None
            else:
                with Image.open(path) as im:images[name]=im.convert('RGB')
        return dict(images=images,proprio=row['proprio'],navigation=row['navigation'])

    def target(self, token):
        return torch.tensor(self.records[token]['requested_GT'],dtype=torch.float32)
