"""Aggregate real kernels and CUDA intervals without summing nested/multistream work."""
import argparse,json,collections,statistics
from pathlib import Path
from ..io import atomic_json


def union_us(intervals):
    end=None;total=0
    for a,b in sorted(intervals):
        if end is None or a>end:total+=b-a;end=b
        elif b>end:total+=b-end;end=b
    return total


def analyze(root,rank):
    events=json.loads((root/f'trace_rank{rank}.json').read_text())['traceEvents']
    kernels=[e for e in events if e.get('cat')=='kernel' and 'dur' in e]
    streams=collections.defaultdict(list);work=collections.Counter();counts=collections.Counter()
    for e in kernels:
        work[e['name']]+=e['dur'];counts[e['name']]+=1;streams[str(e['args'].get('stream'))].append((e['ts'],e['ts']+e['dur']))
    all_intervals=[i for values in streams.values() for i in values]
    result={'rank':rank,'kernel_count':len(kernels),'kernel_work_ms':sum(work.values())/1000,
        'kernel_timeline_union_ms':union_us(all_intervals)/1000,
        'stream_busy_union_ms':{k:union_us(v)/1000 for k,v in streams.items()},
        'overlap_kernel_work_minus_union_ms':(sum(work.values())-union_us(all_intervals))/1000,
        'top_kernels':[{'name':k,'total_ms':v/1000,'calls':counts[k]} for k,v in work.most_common(25)],
        'attention_kernels':[{'name':k,'total_ms':v/1000,'calls':counts[k]} for k,v in work.items() if any(s in k.lower() for s in ('flash','fmha','attention'))],
        'nccl_work_ms':sum(v for k,v in work.items() if 'nccl' in k.lower())/1000,
        'cuda_launch_count':sum(e.get('name','').startswith(('cudaLaunchKernel','cuLaunchKernel')) for e in events),
        'graph_breaks':'NOT_APPLICABLE_EAGER_NO_COMPILE','source_trace':str(root/f'trace_rank{rank}.json')}
    mod=json.loads((root/f'modules_rank{rank}.json').read_text());groups=collections.defaultdict(list)
    for e in mod['intervals']:groups[e['name']].append(e['inclusive_cuda_ms'])
    result['inclusive_regions']={k:{'calls':len(v),'median_ms':statistics.median(v),'mean_ms':statistics.mean(v)} for k,v in groups.items()}
    ddp=json.loads((root/f'rank_{rank}.json').read_text())['DDP']
    result['DDP']={k:ddp.get(k) for k in ['total_parameter_size_bytes','avg_backward_comm_time','avg_backward_compute_comm_overlap_time','unused_parameter_size','num_parameter_tensors']}
    result['DDP']['exposed_comm_estimate_ms']=max(0,ddp.get('avg_backward_comm_time',0)-ddp.get('avg_backward_compute_comm_overlap_time',0))/1e6
    result['DDP']['ring_allreduce_logical_traffic_bytes_per_rank']=ddp['total_parameter_size_bytes']*2*(ddp['world_size']-1)/ddp['world_size']
    result['DDP']['traffic_note']='Logical ring formula, not measured network counters; excludes buffers and scalar control collectives.'
    return result


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('root',type=Path);p.add_argument('--rank',type=int,default=0);a=p.parse_args()
    atomic_json(a.root/f'ANALYSIS_rank{a.rank}.json',analyze(a.root,a.rank))
if __name__=='__main__':main()
