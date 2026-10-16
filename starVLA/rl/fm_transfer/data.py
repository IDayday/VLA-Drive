"""Fixed whole-TRAIN stream; supervision is separate from actor condition."""
import json
from pathlib import Path
import torch
from starVLA.rl.constrained_flow_noise.condition_cache import ConditionStore
from starVLA.rl.simwam_recipe.contracts import RAW_CONDITION, CONDITION_SHA, POPULATION, file_sha, fingerprint


class TrainingData:
    def __init__(self, root, *, seed=42):
        root = Path(root)
        self.manifest = json.loads((root/'manifest.json').read_text())
        seal = json.loads((root/'COMPLETE').read_text())
        if seal['manifest_sha256'] != file_sha(root/'manifest.json'):
            raise ValueError('Unsealed supervised data')
        if self.manifest['labels_sha256'] != file_sha(root/'labels.pt'):
            raise ValueError('Supervised label identity changed')
        pop = json.loads(Path(POPULATION).read_text())
        self.tokens = self.manifest['tokens']
        if (self.tokens != pop['train_tokens'] or
            self.manifest['population_identity'] != pop['identity_sha256']):
            raise ValueError('Only the complete locked RL_TRAIN population is allowed')
        self.labels = torch.load(root/'labels.pt',map_location='cpu',weights_only=True,mmap=True)
        if self.labels.shape != (len(self.tokens),8,4) or not torch.isfinite(self.labels).all():
            raise ValueError('Invalid original supervised labels')
        self.conditions = ConditionStore(RAW_CONDITION)
        if self.conditions.identity_sha != CONDITION_SHA:
            raise ValueError('Changed frozen observation execution')
        self.seed = seed; self.epoch = None; self.order = None
        self.sha = fingerprint(self.manifest)

    def index(self, position):
        epoch, offset = divmod(position, len(self.tokens))
        if epoch != self.epoch:
            self.order = torch.randperm(len(self.tokens),generator=torch.Generator().manual_seed(self.seed+epoch)).tolist()
            self.epoch = epoch
        return self.order[offset]

    def batch(self, cursor, rank, world, per_rank, device):
        # No epoch padding or dropped scene: the next epoch simply continues.
        indices=[self.index(cursor+i*world+rank) for i in range(per_rank)]
        raw=torch.cat([self.conditions.get(self.tokens[i]) for i in indices]).to(device=device,dtype=torch.float32)
        target=self.labels[indices].to(device=device,dtype=torch.float32)
        return raw,target,[self.tokens[i] for i in indices]
