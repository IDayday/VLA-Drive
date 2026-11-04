"""Reproduce the complete first6400 TRAIN credit-weight counterfactual.

CPU only. No optimizer, scoring, NAVTEST labels or active recipe changes.
The original audit JSON is preserved; recomputation uses a separate output.
"""
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from starVLA.rl.d44_native_rl.safety_credit import raw_credit


def main():
    root=Path('/mnt/project/d46-priority-rl-20261103');torch.set_num_threads(1);start=time.monotonic();result={}
    for exp,collections in [('AUG3_FULL_ALIGNED',50),('AUG4_LORA_ALIGNED',100)]:
        buckets={};tokens=set();files=0
        for f in sorted((root/'runs'/(exp+'_v2')/'behavior').glob('*/collection*/rank*.pt')):
            if int(f.parent.name.removeprefix('collection'))>collections:continue
            batch=torch.load(f,map_location='cpu',weights_only=False,mmap=True);files+=1
            for r in batch['scenes']:
                if r['padding']:continue
                assert r['token']not in tokens;tokens.add(r['token'])
                raw,feasible=raw_credit(r['factors'][None]);raw=raw[0];mask=feasible[0]
                category='all_feasible'if mask.all()else'mixed_safety'if mask.any()else'all_infeasible'
                a=r['advantages'].double();alternative=(raw/(batch['global_std']+1e-8)).clamp(-5,5)
                b=buckets.setdefault(category,dict(groups=0,raw_stds=[],current_mass=0.,batchstd_mass=0.,effective=0))
                b['groups']+=1;b['raw_stds'].append(float(raw.std(unbiased=True)))
                b['current_mass']+=float(a.abs().sum());b['batchstd_mass']+=float(alternative.abs().sum())
                b['effective']+=int(float(a.abs().max())>1e-6)
        assert len(tokens)==6400
        current=sum(b['current_mass']for b in buckets.values());alternative=sum(b['batchstd_mass']for b in buckets.values())
        for b in buckets.values():
            b['raw_std_median']=float(np.median(b.pop('raw_stds')))
            b['current_advantage_mass_share']=b['current_mass']/current
            b['batchstd_counterfactual_mass_share']=b['batchstd_mass']/alternative
        result[exp]=dict(unique_TRAIN_scenes=len(tokens),collections=collections,rank_material_files=files,buckets=buckets)
    value=dict(schema='d46_first6400_all_material_normalization_audit_v1',status='COMPLETE_CPU_COUNTERFACTUAL',results=result,seconds=time.monotonic()-start,
        limits='Advantage L1 mass is a coefficient proxy, not parameter-gradient contribution or proof of B1 improvement. Not enabled in current runs.')
    path=root/'analysis/AUGUSTUS_REGRESSION_20261103/NORMALIZATION_AUDIT_RECOMPUTED.json'
    if path.exists():raise FileExistsError('Preserve existing analysis evidence')
    path.write_text(json.dumps(value,indent=2))


if __name__=='__main__':main()
