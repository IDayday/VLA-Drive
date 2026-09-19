"""Check BF16 gradients against explicitly rounded FP32 leaf oracle.

This tests the representable BF16 result, not relaxed float allclose. The input
oracle must preserve the source BF16 timestep input and have identical forward
statistics. It does not turn historical serial/batch failures into PASS.
"""
import argparse
import json
from pathlib import Path
import torch
from starVLA.rl.flow_grpo.transactions import atomic_json


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--oracle',required=True);p.add_argument('--actual',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output)
    if out.exists(): raise FileExistsError(out)
    torch.set_num_threads(4)
    meta=json.loads((Path(a.oracle)/'report.json').read_text())
    if meta.get('head_storage')!='fp32' or not meta.get('preserve_bf16_time_input'):raise ValueError('wrong oracle')
    result={'status':'PASS','scope':__doc__,'scenes':{}}
    for scene in (1,2):
        ref,actual=[torch.load(Path(x)/f'stats_rank{scene}.pt',weights_only=True) for x in (a.oracle,a.actual)]
        equal=set(ref)==set(actual) and all(torch.equal(v,actual[k]) for k,v in ref.items())
        if not equal:raise ValueError('forward differs; cannot isolate leaf rounding')
        ref,actual=[torch.load(Path(x)/f'grads_rank{scene}.pt',weights_only=True) for x in (a.oracle,a.actual)]
        assert len(ref)==359 and ref.keys()==actual.keys()
        rows={}
        for name,x in ref.items():
            y=actual[name];assert x.dtype==torch.float32 and y.dtype==torch.bfloat16
            expected=x.to(y.dtype);diff=expected!=y
            rows[name]={'equal':not bool(diff.any()),'nonidentical':int(diff.sum()),'numel':x.numel()}
        passed=all(r['equal'] for r in rows.values())
        result['scenes'][scene]={'status':'PASS' if passed else 'FAIL','forward_exact':equal,'parameters':rows,'failed_tensors':sum(not r['equal'] for r in rows.values())}
        if not passed:result['status']='FAIL'
        atomic_json(out,result);print(scene,result['scenes'][scene]['failed_tensors'],flush=True)
    if result['status']!='PASS':raise SystemExit(1)

if __name__=='__main__':main()
