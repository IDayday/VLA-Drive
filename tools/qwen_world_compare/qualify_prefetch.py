"""Real TRAIN CPU-reader equivalence, including worker-local SQLite readers.

No policy forward, optimizer, simulator or extra training exposure. Native
micro1/micro2 padding, fixed-bank identities, target masks and all random
draws must match. This does not claim end-to-end throughput improvement.
"""
import argparse
from collections.abc import Mapping
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from starVLA.model.modules.qwen_world_postcot.teacher_targets import TeacherCacheReader
from starVLA.model.modules.qwen_world_postcot.candidate_bank import CandidateBank
from starVLA.model.modules.qwen_sourceflow.qwen_encoder import QwenObservationProcessor
from starVLA.rl.qwen_world_postcot.data import WorldSceneData
from starVLA.rl.qwen_world_compare.batching import BatchBuilder
from starVLA.rl.qwen_world_compare.track_targets import FutureTrackStore
from starVLA.rl.qwen_world_compare.evidence import EvidenceBank
from starVLA.rl.qwen_world_compare.prefetch import OrderedUpdatePrefetch


def equal(a,b):
    if isinstance(a,torch.Tensor):
        assert a.dtype==b.dtype and a.shape==b.shape
        torch.testing.assert_close(a,b,rtol=0,atol=0,equal_nan=True)
        return a.numel()*a.element_size()
    if isinstance(a,Mapping):
        assert set(a)==set(b)
        return sum(equal(a[k],b[k])for k in a)
    if isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        return sum(equal(x,y)for x,y in zip(a,b))
    assert a==b
    return 0


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--snapshot',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    cfg=json.loads(Path(args.snapshot).read_text())['identity']['config']
    torch.set_num_threads(4);torch.manual_seed(42);random.seed(42);np.random.seed(42)
    rng=(torch.get_rng_state().clone(),random.getstate(),np.random.get_state())
    def factory():
        return BatchBuilder(WorldSceneData(Path(cfg['data_root'])/'TRAIN',role='TRAIN'),
            QwenObservationProcessor(cfg['qwen_path']),TeacherCacheReader(cfg['teacher_cache'],expected_cohort='TRAIN'),
            CandidateBank(cfg['candidate_bank'],allowed_splits=('TRAIN',)),FutureTrackStore(Path(cfg['target_root'])/'TRAIN'),
            EvidenceBank(cfg['evidence_root']))
    rows=[];builder=factory();expected=factory();data=builder.data
    for micro in(2,1):
        def request(update,cursor):
            return [([data.tokens[data.index(cursor+i)]for i in range(m,m+micro)],list(range(m,m+micro)))
                    for m in range(0,4,micro)]
        tick=time.perf_counter();values={}
        for u in(20000,20001):values[u]=[expected.build(t,pos,u)for t,pos in request(u,u*32)]
        sequential=time.perf_counter()-tick;comparison_bytes=0;waits=[]
        # Close/recreate after update1: unconsumed preparation must not become
        # committed builder progress, and resume must reproduce update2.
        for u in(20000,20001):
            before=builder.calls
            with OrderedUpdatePrefetch(builder,request,start_update=u,start_cursor=u*32,builder_factory=factory)as pool:
                tick=time.perf_counter();actual=pool.next(u,u*32);waits.append(time.perf_counter()-tick)
                comparison_bytes+=equal(actual,values[u])
            assert builder.calls==before+4//micro
        assert builder.state_dict()==expected.state_dict()
        rows.append(dict(micro=micro,compared_updates=2,compared_tensor_bytes=comparison_bytes,
            committed_builder_calls=builder.calls,sequential_read_seconds=sequential,
            initialization_and_first_wait_seconds=waits,
            exact_tensors_masks_queries_noise_sigma_candidate_ids=True,unconsumed_progress_excluded=True))
    assert torch.equal(rng[0],torch.get_rng_state())and rng[1]==random.getstate()
    state=np.random.get_state();assert state[0]==rng[2][0]and np.array_equal(state[1],rng[2][1])and state[2:]==rng[2][2:]
    builder.close();expected.close();out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(dict(status='PASS',layouts=rows,global_RNG_unchanged=True,scorer_calls=0,
        optimizer_updates=0,caveat='CPU equivalence and ownership test, not a pipeline-overlap speed measurement.'),indent=2))
    print(out.read_text())


if __name__=='__main__':main()
