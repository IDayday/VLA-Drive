"""Disposable real-context resume qualification, never a selected study model."""
import argparse
import json
from pathlib import Path
import torch
from iqe.io import atomic_json,read_json,file_hash,digest
from iqe.contracts import require
from iqe.specialist_study.objectives import SceneSampler
from iqe.specialist_study.qualifications import equal_tree
from iqe.specialist_study.training import runtime
from iqe.training.trainer import train
from .config import load
from .data import canonical_bank,read_split,FrozenContexts
from .training import create,make_fetch,forward,take,resource
from .objectives import cost_ce,denominators,fixed_normalizer


def real_resume(config,kind,device):
    resource(config,'qualify-decision-resume',training=True);device=runtime(device)
    folder=config.root/'qualification'/('resume_'+kind);done=folder/'REPORT.json'
    if done.exists():return read_json(done)
    split=read_split(config);meta,bank,scenes=canonical_bank(config,'scorer_fit')
    ids=sorted(split['fit']['scene_ids'],key=lambda s:digest([config.partition_seed,'resume32',s]))[:32]
    _,bank,scenes=take(meta,bank,scenes,ids);cache=FrozenContexts(config,scenes)
    fetch=make_fetch(config,scenes,bank,cache,kind)
    z=fixed_normalizer(torch.from_numpy(bank['scores']),torch.from_numpy(bank['valid']))
    dependencies=dict(kind=kind,qualification_only=True,bank=meta['hash'],ids=digest(ids),Z_train=z)
    cfg=dict(lr=config.lr,weight_decay=config.weight_decay,global_batch_size=128,micro_batch_size=config.micro_batch,
             max_optimizer_steps=4,warmup_steps=100,lr_min_ratio=1.,grad_clip_norm=1.,schedule='warmup_constant',
             validation_steps=[],checkpoint_every_steps=2,validation_every_steps=500,early_stop_patience=10**9)
    def execute(name,stop=None,resume=None):
        model,_=create(config,kind,'resume_audit',cache.read(scenes[0]));sampler=SceneSampler(ids,config.sampler_seed,dependencies)
        def loss(m,b):return {'decision':cost_ce(forward(m,kind,b)['decision_logits'],b['scores'],b['label_valid'],z,b['scene_valid'])}
        train(model,loss,fetch,sampler,cfg,dependencies,folder/name,device=device,seed=config.training_seed,
              resume=resume,stop_after=stop,denominator_function=denominators)
        return sampler.peek(128)
    next_a=execute('continuous');execute('interrupted',stop=2);next_b=execute('interrupted',resume='auto')
    paths=[folder/x/'step_000004.pt' for x in ('continuous','interrupted')]
    a,b=[torch.load(p,weights_only=False,map_location='cpu') for p in paths]
    equal={k:equal_tree(a[k],b[k]) for k in ('module','optimizer','scheduler','sampler','rng_by_rank','consumed_sample_ids')}
    equal['next_sample_ids']=next_a==next_b
    histories=[[json.loads(line) for line in (folder/n/'steps.jsonl').read_text().splitlines()] for n in ('continuous','interrupted')]
    equal['loss_sequence']=[x['loss'] for x in histories[0]]==[x['loss'] for x in histories[1]]
    equal['gradient_norm_sequence']=[x['gradient_norm'] for x in histories[0]]==[x['gradient_norm'] for x in histories[1]]
    require(all(equal.values()),'real-context exact resume failed')
    # Terminal restore must not consume or update anything, including checkpoint bytes.
    before=file_hash(paths[1]);count=len(histories[1]);next_terminal=execute('interrupted',resume='auto')
    require(file_hash(paths[1])==before and next_terminal==next_b and len((folder/'interrupted/steps.jsonl').read_text().splitlines())==count,
            'terminal resume unexpectedly trained or consumed samples')
    result=dict(status='PASS',kind=kind,real_scenes=len(scenes),optimizer_updates_total=8,
                equality=equal,terminal_new_optimizer_steps=0,checkpoint_hashes=[file_hash(p) for p in paths],
                production_weights_changed=False,device=device,hardware=torch.cuda.get_device_name() if device.startswith('cuda') else 'CPU')
    atomic_json(done,result,immutable=True);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--kind',required=True,choices=('M_scene','M_set'));p.add_argument('--device',default='cuda:0')
    a=p.parse_args();print(json.dumps(real_resume(load(a.config),a.kind,a.device)))


if __name__=='__main__':main()
