"""Bounded real frozen-context reader comparison; never selects a research model."""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import time
import numpy as np
import torch
from iqe.contracts import require
from iqe.io import atomic_json, file_hash
from iqe.evaluation.retention import module_hash, _tensor_map_hash
from iqe.training.trainer import slice_batch
from .config import load_config
from .binding import scenes_by_role
from .data import ReusedContext
from .context_reader import VerifiedSpatialReader
from .scorer_training import create_scorer, batch_fetch
from .objectives import terms, denominators
from .training import runtime


def run(config_path, device, samples):
    require(samples in (64, 128), 'reader qualification is capped at 128 real fit scenes')
    config=load_config(config_path)
    from .resource_guard import check
    check(config, 'qualify-spatial-reader')
    device=runtime(device)
    scenes=scenes_by_role(config)['scorer_fit'][:samples]
    with np.load(config.root/'banks/deploy/scorer_fit/arrays.npz', allow_pickle=False) as f:
        bank={key:f[key][:samples].copy() for key in f.files}
    entries=[{'scene_id':s.scene_id,'phase':1,'natural':True,'comparison':False} for s in scenes]
    model,identity=create_scorer(config,'S10')
    model=model.to(device)
    initial=deepcopy(model.state_dict())
    cache=ReusedContext(config,capacity=samples)
    contexts={};timings={};statistics={}
    rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state_all()
    for mode in ('legacy','mmap_v1'):
        os.environ['IQE_SPATIAL_READER_MODE']=mode
        os.environ['IQE_SPATIAL_MMAP_RECORDS']='32768'
        reader=VerifiedSpatialReader(config,64)
        fetch=batch_fetch(config,scenes,cache,reader,bank,bank,False)
        measurements=[]
        for repetition in range(3):
            torch.cuda.synchronize();start=time.perf_counter()
            batch=fetch(entries,device)
            torch.cuda.synchronize();measurements.append(time.perf_counter()-start)
            if repetition==2:contexts[mode]=batch
            else:del batch
        timings[mode]=measurements;statistics[mode]=dict(reader.statistics)
    def equal(a,b):
        if torch.is_tensor(a):return torch.equal(a,b) and a.dtype==b.dtype
        if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
        if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
        return a==b
    require(equal(contexts['legacy'],contexts['mmap_v1']), 'mmap changes real training input bytes')
    require(torch.equal(rng,torch.get_rng_state()) and all(torch.equal(a,b) for a,b in zip(cuda_rng,torch.cuda.get_rng_state_all())),
            'reader changed training RNG')
    updates={};states={};compute={}
    for mode in ('legacy','mmap_v1'):
        model.load_state_dict(initial,strict=True);model.train()
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
        batch=contexts[mode];den=denominators(batch);optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize();start=time.perf_counter();values=[]
        for at in range(0,samples,16):
            small=slice_batch(batch,at,min(at+16,samples))
            losses=terms(model(small['context'],small['trajectories']),small)
            value=sum(v.numerator/den[k].clamp_min(1)*{'value':1.,'components':.2,'ranking':.2}[k] for k,v in losses.items())
            value.backward();values.append(float(value.detach()))
        gradient={k:p.grad.detach().clone() for k,p in model.named_parameters()}
        require(all(torch.isfinite(v).all() for v in gradient.values()), 'nonfinite real qualification gradient')
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step();torch.cuda.synchronize();compute[mode]=time.perf_counter()-start
        states[mode]={'model':deepcopy(model.state_dict()),'optimizer':deepcopy(optimizer.state_dict()),'gradient':gradient}
        updates[mode]={'loss':sum(values),'gradient_norm':float(norm),'model_hash':module_hash(model),
                       'gradient_hash':_tensor_map_hash(gradient.items())}
    require(equal(states['legacy'],states['mmap_v1']), 'mmap changes gradient/Adam/model update')
    report={'status':'PASS','scene_ids':[s.scene_id for s in scenes],'actual_samples':samples,
            'Base_binding_hash':file_hash(config.root/'BASE_BINDING.json'),
            'deployment_bank_hash':file_hash(config.root/'banks/deploy/scorer_fit/BANK.json'),
            'reader_code_hash':file_hash(Path(__file__).with_name('context_reader.py')),
            'all_input_tensor_bytes_equal':True,'RNG_unchanged':True,'gradient_Adam_parameter_update_equal':True,
            'scorer_init_hash':identity['initial_state_hash'],'scorer_weights_loaded':False,
            'reader_fetch_seconds':timings,'forward_backward_clip_Adam_seconds':compute,'updates':updates,
            'reader_statistics':statistics,'GPU_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'production_throughput_claim':False,'timing_scope':'three diagnostic fetches, not steady training',
            'qualification_weights_not_used_for_training_selection':True}
    atomic_json(config.root/'qualifications/spatial_reader_mmap_v1/REPORT.json',report,immutable=True)
    print(report,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--samples',type=int,default=128)
    a=p.parse_args();run(a.config,a.device,a.samples)


if __name__=='__main__':main()
