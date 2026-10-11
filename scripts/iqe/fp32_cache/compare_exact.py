"""Exact online/cache full-state comparison; no 0.1 relative-error escape hatch."""
import argparse
from pathlib import Path
import torch
from iqe.io import atomic_json,read_json,digest
from iqe.performance.compare import compare_resume,flatten_values


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('online',type=Path);p.add_argument('cached',type=Path)
    p.add_argument('--output',required=True);a=p.parse_args()
    torch.set_num_threads(2)
    result=compare_resume(a.online,a.cached)
    result['full_state_rng_cursor_equal']=result['status']=='PASS'
    gradient=[];values=[]
    for rank in range(8):
        online=read_json(a.online/f'rank_{rank}.json');cached=read_json(a.cached/f'rank_{rank}.json')
        x=online['production_cache']['gradient_audit'];y=cached['production_cache']['gradient_audit']
        gradient.append({'rank':rank,'full_gradient_bitwise':x['full_named_gradient_sha256']==y['full_named_gradient_sha256'],
                         'unused_equal':x['unused_names']==y['unused_names']})
        first=torch.load(a.online/f'probe_rank{rank}.pt',map_location='cpu',weights_only=True,mmap=True)
        second=torch.load(a.cached/f'probe_rank{rank}.pt',map_location='cpu',weights_only=True,mmap=True)
        v1,v2=flatten_values(first['values']),flatten_values(second['values'])
        assert v1.keys()==v2.keys(),'missing intermediate comparison (not silently intersected)'
        for key in v1:
            lhs,rhs=v1[key],v2[key]
            values.append({'rank':rank,'name':key,'bitwise':torch.equal(lhs,rhs),
                'max_abs':float((lhs.double()-rhs.double()).abs().max()) if lhs.numel() else 0.})
    result['full_gradient_audit']=gradient;result['intermediates']=values
    passed=result['full_state_rng_cursor_equal'] and all(x['full_gradient_bitwise'] and x['unused_equal'] for x in gradient) and all(x['bitwise'] for x in values)
    result.update(status='PASS' if passed else 'FAIL',criterion='exact full state/optimizer/RNG/cursor/full gradients plus every captured intermediate')
    atomic_json(a.output,result);print({k:v for k,v in result.items() if k not in ['ranks','intermediates','full_gradient_audit']})
    return 0 if passed else 2

if __name__=='__main__':raise SystemExit(main())
