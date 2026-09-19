"""Compare real eager/graph optimizer tensors, without claiming exact resume.

Code/config identities deliberately differ between these executions. Numerical
state and fixed-chain/scene semantics are checked independently at zero tolerance.
"""
import argparse
import json
from pathlib import Path
import torch
from scripts.flow_grpo.compare_boundaries import flatten
from scripts.cluster_flow_grpo.boundary_evidence import exact_statistics
from starVLA.rl.flow_grpo.loading import file_sha


def compare(left,right,output):
    left,right,out=Path(left),Path(right),Path(output)
    if out.exists():raise FileExistsError(out)
    result={'status':'PASS','scope':'real eager versus graph updates, all model/optimizer tensors, zero tolerance; NOT cross-code exact resume',
            'left':str(left),'right':str(right),'script_sha256':file_sha(__file__),'updates':{}}
    rows_a=[json.loads(x) for x in (left/'training.jsonl').read_text().splitlines()]
    rows_b=[json.loads(x) for x in (right/'training.jsonl').read_text().splitlines()]
    assert [r['update'] for r in rows_a]==[1,2]==[r['update'] for r in rows_b]
    for update in (1,2):
        a,b=[p/f'checkpoints/update_{update:06}' for p in (left,right)]
        assert all((p/'COMPLETE').is_file() for p in (a,b))
        ra,rb=rows_a[update-1],rows_b[update-1]
        semantic={}
        for key in ('scene_count','candidate_count','replay_scene_count','inner_epoch','lr','lr_used','scheduler_completed_updates','stability'):
            semantic[key]=ra[key]==rb[key]
        for rank,(x,y) in enumerate(zip(ra['ranks'],rb['ranks'])):
            for key in ('scene_tokens','replay_tokens','pre_update_ratio_min','pre_update_ratio_max','gradient_tensors','local_backward_contribution_l2'):
                semantic[f'rank{rank}/{key}']=x[key]==y[key]
        # Behavior hashes include intentionally changed provenance. Compare chain
        # tensor payloads in rank state files instead of requiring that hash equal.
        names={str(p.relative_to(a)) for p in a.rglob('*') if p.suffix in ('.pt','.bin','.pkl')}
        assert names=={str(p.relative_to(b)) for p in b.rglob('*') if p.suffix in ('.pt','.bin','.pkl')}
        files={}
        for name in sorted(names):
            aa,bb=[dict(flatten(torch.load(p/name,map_location='cpu',weights_only=False,mmap=False))) for p in (a,b)]
            tensors={k for k,v in aa.items() if isinstance(v,torch.Tensor)}
            assert tensors=={k for k,v in bb.items() if isinstance(v,torch.Tensor)}
            checks={k:exact_statistics(aa[k],bb[k]) for k in sorted(tensors)}
            files[name]={'tensors':checks,'tensor_count':len(checks),
                         'equal':all(c['allclose'] for c in checks.values())}
            del aa,bb
        passed=all(semantic.values()) and all(v['equal'] for v in files.values())
        result['updates'][str(update)]={'status':'PASS' if passed else 'FAIL','semantics':semantic,'files':files}
        if not passed:result['status']='FAIL'
        out.write_text(json.dumps(result,indent=2))
        print(update,result['updates'][str(update)]['status'],flush=True)
    if result['status']!='PASS':raise SystemExit(1)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--left',required=True);p.add_argument('--right',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();torch.set_num_threads(8);compare(a.left,a.right,a.output)
