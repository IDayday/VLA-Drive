"""Retain strict failures, compare recovered outputs to the observed CUDA A/A."""
import argparse
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.decision_local.common import atomic_json,digest,read,source_sha
from tools.decision_local.verify_resume import equal
from tools.decision_local.train_small_model import build_model,forward
from starVLA.dataloader.decision_local import FrozenConsequenceData


def errors(x,y):
    a=x.detach().double().flatten();b=y.detach().double().flatten();d=b-a
    return {'max_absolute':float(d.abs().max()),'normalized_L2':float(d.norm()/a.norm().clamp_min(1e-15)),
            'cosine':float(torch.nn.functional.cosine_similarity(a,b,dim=0))}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('runs','features','labels','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2)
    data=FrozenConsequenceData(a.features,a.labels);batch=data.batch(list(range(min(32,len(data.index)))),'cpu',evaluation=True)
    output={}
    for kind in ('E_NONE','P_DIFF'):
        base=torch.load(a.runs/kind/'uninterrupted/latest.pt',weights_only=False,map_location='cpu')
        outputs={}
        names=('uninterrupted','interrupted','AA') if kind=='P_DIFF' else ('uninterrupted','interrupted')
        for name in names:
            path=a.runs/kind/name/'latest.pt';s=torch.load(path,weights_only=False,map_location='cpu')
            for k in ('identity','scheduler','rng','order','completed','epoch','offset','exposure'):equal(base[k],s[k],k)
            model=build_model(kind,s['identity']['seed'],s['identity']['dim']).eval();model.load_state_dict(s['model'],strict=True)
            with torch.no_grad():pred,losses,_=forward(model,kind,batch)
            outputs[name]=(pred,losses['total'])
        output[kind]={'E1_exact':True,'output_comparisons':{name:{'prediction':errors(outputs['uninterrupted'][0],outputs[name][0]),
            'absolute_loss_difference':float((outputs['uninterrupted'][1]-outputs[name][1]).abs())} for name in names[1:]}}
    atomic_json(a.output,{'source_sha':source_sha(ROOT),'tool_sha256':digest(__file__),'results':output,
        'note':'FP32 CPU fixed-input readout of real GPU-trained checkpoints; parameter/moment GPU A/A assessment separate. Strict P_DIFF failure retained.'})


if __name__=='__main__':main()
