"""Transfer detached per-scene scalars once, preserving Python reduction order."""
import torch


class DeviceMetrics:
    def __init__(self):self.rows=[]
    def add(self,metrics,*,valid=True):
        keys=tuple(metrics)
        if any(not torch.is_tensor(v)or v.numel()!=1 for v in metrics.values()):
            raise ValueError('Metrics must be scalar tensors')
        self.rows.append((keys,[v.detach().reshape(())for v in metrics.values()],valid))
    def drain(self):
        if not self.rows:return {},0.
        # Promotion is lossless for the native FP32/FP64 scalar values.
        values=torch.stack([v for _,row,_ in self.rows for v in row]).cpu().tolist()
        result={};offset=0;kl=0.;n=len(self.rows)
        for keys,row,valid in self.rows:
            entries=values[offset:offset+len(row)];offset+=len(row)
            for key,value in zip(keys,entries):result[key]=result.get(key,0.)+value/n
            if valid:kl+=entries[keys.index('reference_KL_joint')]
        self.rows.clear();return result,kl
