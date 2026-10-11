"""Full200-update state, per-rank sample/loss/RNG and shared IO comparison."""
import argparse
from pathlib import Path
import numpy as np
import torch
from iqe.io import read_json, atomic_json, file_hash, digest
from iqe.contracts import require


def equal_tree(a,b,path='',bad=None):
    bad=[] if bad is None else bad
    if torch.is_tensor(a):
        if not torch.is_tensor(b) or a.dtype!=b.dtype or not torch.equal(a,b):bad.append(path)
    elif isinstance(a,np.ndarray):
        if not isinstance(b,np.ndarray) or not np.array_equal(a,b):bad.append(path)
    elif isinstance(a,dict):
        if not isinstance(b,dict) or a.keys()!=b.keys():bad.append(path+'/keys')
        else:
            for k in a:equal_tree(a[k],b[k],path+'/'+str(k),bad)
    elif isinstance(a,(tuple,list)):
        if type(a)!=type(b) or len(a)!=len(b):bad.append(path+'/length')
        else:
            for i,(x,y) in enumerate(zip(a,b)):equal_tree(x,y,path+'/'+str(i),bad)
    elif a!=b:bad.append(path)
    return bad


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--max-memory-gib',type=float,default=64.);a=p.parse_args()
    torch.set_num_threads(2);r=a.root
    results=[read_json(r/name/'RESULT.json') for name in ('online200','cached200')]
    require(all(x['status']=='COMPLETE' and x['updates']==200 for x in results),'real200 comparisons incomplete')
    # Whole-state comparison scans every storage. Sequential eager reads avoid
    # pathological4KiB remote mmap faults; this CPU audit has an explicit RAM cap.
    required=sum((r/name/'step_033200.pt').stat().st_size for name in ('online200','cached200'))
    available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                       if line.startswith('MemAvailable:')))*1024
    require(required <= a.max_memory_gib*1024**3 and available > required*1.2,
            'BLOCKED_AUDIT_MEMORY: full comparison needs declared resident-state budget')
    states=[torch.load(r/name/'step_033200.pt',map_location='cpu',weights_only=False)
            for name in ('online200','cached200')]
    checks={k:not equal_tree(states[0][k],states[1][k]) for k in
            ('model','optimizer','scheduler','rng_by_rank','epoch','offset','optimizer_step','fit_scene_ids')}
    audits=[];cached=[];reference=[]
    fields=('optimizer_step','sample_ids','global_batch','losses','gradient_norm','lr','valid_denominators','used_parameter_names_hash')
    for rank in range(8):
        x=read_json(r/'online200'/f'rank_{rank}.json');y=read_json(r/'cached200'/f'rank_{rank}.json')
        reference.append(x);cached.append(y)
        audits.append({'rank':rank,'every_update_equal':all(not equal_tree({k:s[k] for k in fields},{k:t[k] for k in fields})
            for s,t in zip(x['rows'],y['rows'])) and len(x['rows'])==len(y['rows'])==200,
            'final_rng_equal':all(x[k]==y[k] for k in ('rng_python_hash','rng_numpy_hash','rng_cpu_hash','rng_cuda_hashes')),
            'cursor_equal':x['data_cursor']==y['data_cursor'], 'dtypes_equal':x['actual_dtype_elements']==y['actual_dtype_elements']})
    sequence=[]
    for index in range(200):
        mine=[rank['rows'][index]['sample_ids'] for rank in cached]
        sequence.append([mine[rank][offset] for offset in range(4) for rank in range(8)])
    plan=read_json(r/'REPRESENTATIVE_PLAN.json');require(sequence==plan['sample_sequence'],'actual sampler differs from locked representative list')
    hit=all(x['production_cache']['calls']=={'cached_encode':200,'online_encode':0,'visual_calls_on_hit':0} for x in cached)
    pilot=read_json(r/'PAYLOAD_REGROUP.json');small=read_json(r/'EXACT_ONLINE_CACHE_2.json')
    passed=all(checks.values()) and all(all(v for k,v in x.items() if k!='rank') for x in audits) and hit
    numeric={'status':'PASS' if passed else 'FAIL','full_state_rng_cursor_equal':all(checks.values()),
        'checks':checks,'per_rank':audits,'sample_sequence_sha256':digest(sequence),
        'storage_bitwise':pilot['storage_bitwise'],'deepstack_count':3,'no_images_or_visual_on_hit':hit,
        'pilot_full_gradient_bitwise':small['status']=='PASS','sample_exposures':6400,'optimizer_updates':200,
        'criterion':'bitwise full tensors/state; exact original samples, all losses, gradients norms, denominators, LR and RNG',
        'checkpoint_hashes':{name:read_json(r/name/'step_033200.pt.COMPLETE.json')['checksum'] for name in ('online200','cached200')},
        'source_changes_between_comparisons':'only distributed attachment error propagation, diagnostic gradient grouping, builder recovery and diagnostic binding; source hashes retained per launch'}
    atomic_json(r/'NUMERICAL_200.json',numeric)
    readers=[x['production_cache']['cache_reader'] for x in cached]
    total_bytes=sum(x['bytes'] for x in readers);count=sum(x['reads'] for x in readers)
    io={'status':'PASS' if passed else 'FAIL','shared_storage':True,'steady_updates':results[1]['slowest_rank_per_step']['n'],
        'actual_global_batch':32,'sample_sequence_sha256':digest(sequence),'unique_observations':len(set(t for b in sequence for t in b)),
        'cache_reads':count,'read_bytes':total_bytes,'bytes_per_sample':total_bytes/count,'misses':sum(x['misses'] for x in readers),
        'page_cache':'own-files cold-advised; physical cold not guaranteed','local_staging':False,
        'steady_equivalent_shared_bytes_per_second':32*total_bytes/count/results[1]['slowest_rank_per_step']['mean'],
        'per_rank_reader':readers,'max_RSS_KiB':max(x['production_cache']['max_rss_kib'] for x in cached),
        'max_FD':max(x['production_cache']['open_fds'] for x in cached),
        'pinned_memory_measurement':'bounded one producer+consumer batch; allocator peak not exposed by installed torch API',
        'timings':{n:{k:v for k,v in result.items() if k!='identity'} for n,result in zip(('online','cached'),results)},
        'original_checkpoint_interval':1000,
        'mean_with_checkpoint_amortization':{n:result['slowest_rank_per_step']['mean']+result['checkpoint_seconds']/1000
                                            for n,result in zip(('online','cached'),results)},
        'cache_only_speedup':results[0]['slowest_rank_per_step']['mean']/results[1]['slowest_rank_per_step']['mean']}
    atomic_json(r/'REPRESENTATIVE_IO.json',io)
    print({'status':numeric['status'],'state_checks':checks,'cache_only_speedup':io['cache_only_speedup'],
           'online':results[0]['slowest_rank_per_step'],'cached':results[1]['slowest_rank_per_step']})
    return 0 if passed else 2

if __name__=='__main__':raise SystemExit(main())
