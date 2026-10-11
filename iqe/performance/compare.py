"""Compare bounded real probes and checkpoint fingerprints; explicit sampled scope."""
import argparse,json
from pathlib import Path
import torch
from ..contracts import require
from ..io import atomic_json,read_json


def metrics(a,b):
    if a.shape!=b.shape:return {'shape_mismatch':[list(a.shape),list(b.shape)]}
    a,b=a.detach().reshape(-1).float(),b.detach().reshape(-1).float()
    if not len(a):return {'elements':0}
    aa=float(torch.dot(a,a));bb=float(torch.dot(b,b));ab=float(torch.dot(a,b));err=b-a
    return {'elements':a.numel(),'finite':bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
        'exact':torch.equal(a,b),'reference_norm':aa**.5,'candidate_norm':bb**.5,
        'cosine':ab/max((aa*bb)**.5,1e-30) if aa and bb else (1. if not aa and not bb else 0.),
        'relative_l2':float(err.norm())/max(aa**.5,1e-30),'max_abs':float(err.abs().max()),'mean_abs':float(err.abs().mean())}


def flatten_values(value,path=''):
    if torch.is_tensor(value):return {path:value}
    result={}
    if isinstance(value,dict):
        for k,v in value.items():result.update(flatten_values(v,path+'/'+k))
    return result


def compare_probes(reference,candidate):
    records=[]
    for rank in range(8):
        a=torch.load(reference/f'probe_rank{rank}.pt',map_location='cpu',mmap=True,weights_only=True)
        b=torch.load(candidate/f'probe_rank{rank}.pt',map_location='cpu',mmap=True,weights_only=True)
        values_a,values_b=flatten_values(a['values']),flatten_values(b['values'])
        record={'rank':rank,'values':{k:metrics(v,values_b[k]) for k,v in values_a.items() if k in values_b}}
        for key in ['gradient_samples','update_samples']:
            require(a[key].keys()==b[key].keys(),'gradient/update parameter names differ')
            names=sorted(a[key]);x=torch.cat([a[key][n] for n in names]);y=torch.cat([b[key][n] for n in names])
            record[key]=metrics(x,y)
        records.append(record)
    passed=all(r[k]['finite'] and r[k]['cosine']>=.99 and r[k]['relative_l2']<=.10 for r in records for k in ['gradient_samples','update_samples'])
    return {'reference':str(reference),'candidate':str(candidate),'sampled_numerical_criteria_passed':passed,'ranks':records,
        'criterion':'predeclared sampled gradient/update cosine>=0.99, relativeL2<=0.10; all finite',
        'limitation':'up to4096 evenly spaced elements of each trainable tensor; no full-gradient equivalence claim; scientific quality untested'}


def compare_resume(reference,resumed):
    rows=[]
    for rank in range(8):
        a=read_json(reference/f'rank_{rank}.json');b=read_json(resumed/f'rank_{rank}.json')
        bystep={x['optimizer_step']:x for x in a['rows']}
        row={'rank':rank,'fingerprint_equal':bool(a['fingerprint']) and a['fingerprint']==b['fingerprint'],
            'python_rng_equal':a['rng_python_hash']==b['rng_python_hash'],'numpy_rng_equal':a['rng_numpy_hash']==b['rng_numpy_hash'],
            'scheduler_equal':a['scheduler_state']==b['scheduler_state'],'data_cursor_equal':a['data_cursor']==b['data_cursor'],
            'cpu_rng_equal':a['rng_cpu_hash']==b['rng_cpu_hash'],'cuda_rng_equal':a['rng_cuda_hashes']==b['rng_cuda_hashes'],
            'next_batches_equal':all(x['sample_ids']==bystep[x['optimizer_step']]['sample_ids'] for x in b['rows']),
            'loss_gradient_lr_equal':all(all(x[k]==bystep[x['optimizer_step']][k] for k in ['losses','gradient_norm','lr']) for x in b['rows'])}
        rows.append(row)
    return {'reference':str(reference),'resumed':str(resumed),'status':'PASS' if all(all(v for k,v in r.items() if k!='rank') for r in rows) else 'FAIL','ranks':rows}


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('reference',type=Path);p.add_argument('candidate',type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--resume',action='store_true');a=p.parse_args()
    torch.set_num_threads(2)
    result=(compare_resume if a.resume else compare_probes)(a.reference,a.candidate)
    atomic_json(a.output,result);print(json.dumps({k:v for k,v in result.items() if k!='ranks'}))
if __name__=='__main__':main()
