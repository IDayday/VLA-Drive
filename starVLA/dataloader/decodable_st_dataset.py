"""Explicit input/target split on top of the original current-frame adapter."""
from pathlib import Path
import json
import torch
from torch.utils.data import Dataset
from starVLA.dataloader.foresight_dataset import ForesightCurrentDataset
from starVLA.model.modules.decodable_st.contracts import AllowedObservation,RepresentationTargets,SampleIdentity


class DecodableDataset(Dataset):
    def __init__(self,current_root,target_cache=None,indices=None):
        self.current=ForesightCurrentDataset(current_root)
        self.indices=list(range(len(self.current))) if indices is None else list(indices)
        if len(set(self.indices))!=len(self.indices) or any(i<0 or i>=len(self.current) for i in self.indices):
            raise ValueError("Invalid or duplicate scene selection")
        self.target_cache=target_cache

    def __len__(self):return len(self.indices)

    def __getitem__(self,i):
        j=self.indices[i];record=self.current.index[j];source=self.current[j]
        identity=SampleIdentity(record["token"],record["log"],self.current.identity["identity"])
        # Construct from exactly permitted fields; identity is returned separately.
        obs=AllowedObservation((tuple(source["image"]),),(source["lang"],),
                torch.as_tensor(source["state"],dtype=torch.float32)[None],("CAM_F0","CAM_L0","CAM_R0"))
        target=self.target_cache.read(identity) if self.target_cache else None
        return obs,target,identity


def collate_observations(rows):
    obs,targets,identities=zip(*rows)
    batch=AllowedObservation(tuple(x.images[0] for x in obs),tuple(x.navigation[0] for x in obs),
                             torch.cat([x.ego_state for x in obs]),obs[0].camera_ids)
    if all(t is None for t in targets):return batch,None,identities
    if any(t is None for t in targets):raise ValueError("Missing target record; task absence must use explicit validity")
    keys=set(targets[0].values)
    if any(set(t.values)!=keys for t in targets):raise ValueError("Task schema mismatch")
    target=RepresentationTargets({k:torch.stack([t.values[k] for t in targets]) for k in keys},
          {k:torch.stack([t.valid[k] for t in targets]) for k in targets[0].valid},targets[0].metadata)
    return batch,target,identities
