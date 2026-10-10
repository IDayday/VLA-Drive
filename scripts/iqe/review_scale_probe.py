"""Bounded real cache/retrieval timing, exact reference comparison, no full bank."""
import argparse
import time
from pathlib import Path
import numpy as np
import torch
from iqe.config import load_config
from iqe.pipeline import Pipeline
from iqe.data.support_retrieval import retrieve,RetrievalIndex
from iqe.io import atomic_json,read_json
from iqe.resources import qualify


def main():
    a=argparse.ArgumentParser(__doc__);a.add_argument('--config',required=True);args=a.parse_args()
    qualify('cpu');torch.set_num_threads(2)
    p=Pipeline(load_config(args.config),mode='smoke',max_samples=32,device='cpu');p.use_locked_base()
    rows=p.scenes(['incremental_fit']);start=time.perf_counter();features=[p.cached(s) for s in rows]
    read_seconds=time.perf_counter()-start
    vectors=[torch.cat((f.scene.mean(1)[0],f.ego.flatten())).numpy() for f in features]
    start=time.perf_counter();reference=[retrieve(s,v,rows,vectors,16,'real_features',42) for s,v in zip(rows,vectors)]
    original_seconds=time.perf_counter()-start
    start=time.perf_counter();index=RetrievalIndex(rows,vectors,'real_features',42);build_seconds=time.perf_counter()-start
    start=time.perf_counter();optimized=list(index.query_many(rows,vectors,16));query_seconds=time.perf_counter()-start
    assert all(a[0]==b[0] and a[1]['candidates']==b[1]['candidates'] for a,b in zip(reference,optimized))
    result={'status':'PASS','real_library_scenes':len(rows),'feature_dim':len(vectors[0]),'feature_read_seconds':read_seconds,
            'reference_seconds':original_seconds,'index_build_seconds':build_seconds,'indexed_queries_seconds':query_seconds,
            'exact_scene_IDs_and_distances':True,'index_identity':index.identity,'full_bank_started':False,
            'extrapolation':'O(H*N*D); 32-scene timing is not a full-scale throughput guarantee; source/log exclusions unchanged'}
    atomic_json(p.root/'SCALE_PROBE.json',result,immutable=True)
    print(result)


if __name__=='__main__':main()
