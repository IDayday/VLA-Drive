"""Fixed-protocol original-DiT PDMS comparison, retaining every requested scene."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from tools.structured_world.summarize import read, compare
from tools.structured_world_v1p1.reaudit_metrics import write_csv


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifacts',required=True);p.add_argument('--output',required=True)
    p.add_argument('--runs',nargs='+',required=True)
    p.add_argument('--reference',default='planner_dev1696_baseline')
    a=p.parse_args();root=Path(a.artifacts);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    factors=['score','no_at_fault_collisions','drivable_area_compliance','ego_progress',
             'time_to_collision_within_bound','comfort','driving_direction_compliance']
    banks={};summaries=[];all_rows=[];identities={};protocol=None
    for name in a.runs:
        source=root/(name+'_pdms');score=json.loads((source/'summary.json').read_text())
        identity=json.loads((root/name/'manifest_0.json').read_text())
        bank=read(source/'scenes.csv');banks[name]=bank
        current={k:identity[k] for k in ['cache_identity','original_checkpoint_sha256','candidates','sampling_steps','scorer']}
        current.update(seed=identity['arguments']['seed'],evaluator_sha256=score['evaluator_sha256'],
                       cache_adapter_sha256=score['cache_adapter_sha256'],protocol=score['protocol'])
        if protocol is None:protocol=current
        elif current!=protocol:raise ValueError('Different evaluation protocols')
        if identity['world_targets_opened'] or len(bank)!=identity['samples'] or len(bank)!=score['requested']:
            raise ValueError('Invalid current-only/scene completeness identity')
        failed=sum(r['status']!='ok' for r in bank.values())
        result={'run':name,'scenes':len(bank),'failed':failed,'PDMS_percent':100*sum(float(r.get('score') or 0) for r in bank.values())/len(bank),
                'zero_fraction':sum(float(r.get('score') or 0)==0 for r in bank.values())/len(bank),
                'peak_gpu_bytes':identity['peak_gpu_bytes']}
        for metric in factors[1:]:result[metric]=sum(float(r.get(metric) or 0) for r in bank.values())/len(bank)
        latency=read(root/name/'scenes_0.csv');values=[float(r['cached_policy_latency_seconds']) for r in latency.values() if r['status']=='ok']
        result.update(cached_policy_seconds_mean=float(np.mean(values)),cached_policy_seconds_p95=float(np.quantile(values,.95)))
        summaries.append(result);identities[name]={'inference':identity,'score':score}
        for row in bank.values():all_rows.append(dict(run=name,**row))
        dest=out/name;dest.mkdir(exist_ok=True)
        for f in ['summary.json','scenes.csv']:shutil.copy2(source/f,dest/f)
        shutil.copy2(root/name/'manifest_0.json',dest/'inference_manifest.json')
    pairs={};paired=[]
    if a.reference not in banks:raise ValueError('Missing reference')
    comparisons=[(n,a.reference) for n in a.runs if n!=a.reference]
    for x,y in [('planner_dev1696_randommask','planner_dev1696_allmask'),('planner_dev1696_bev_tasks','planner_dev1696_bev_control')]:
        if x in banks and y in banks:comparisons.append((x,y))
    for first,second in comparisons:
        if set(banks[first])!=set(banks[second]):raise ValueError('Unpaired scene sets')
        key=first+'__vs__'+second
        failed=any(r['status']!='ok' for b in [banks[first],banks[second]] for r in b.values())
        pairs[key]={'status':'INVALID_RETAINED_FAILURES'} if failed else compare(banks[first],banks[second])
        for token in sorted(banks[first]):
            x,y=banks[first][token],banks[second][token]
            row={'comparison':key,'token':token,'log':x['log'],'first_status':x['status'],'second_status':y['status']}
            for factor in factors:row[factor+'_delta']=float(x.get(factor) or 0)-float(y.get(factor) or 0)
            paired.append(row)
    write_csv(out/'summary.csv',summaries);write_csv(out/'scene_metrics.csv',all_rows);write_csv(out/'paired_scenes.csv',paired)
    report={'runs':summaries,'paired_comparisons':pairs,'protocol':protocol,
            'selection':'Fixed final checkpoints; PDMS never enters training or model selection.',
            'uncertainty':'Log-cluster intervals describe scene sampling for one training seed, not training-seed stability.',
            'cost_limit':'Cached policy latency excludes current image/Qwen/BEV provider extraction. See online measurements separately.'}
    (out/'SUMMARY.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
