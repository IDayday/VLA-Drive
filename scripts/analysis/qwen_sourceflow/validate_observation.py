"""Real current-image full-Qwen masking/dependency tests; no policy learning claim."""
import argparse
import copy
import json
from pathlib import Path
import time
import torch
from starVLA.rl.qwen_sourceflow.contracts import ROOT,QWEN_PATH,require_dispatch
from starVLA.rl.qwen_sourceflow.data import RawSceneData
from starVLA.model.modules.qwen_sourceflow.qwen_encoder import QwenEncoder,QwenObservationProcessor


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',required=True);a=p.parse_args()
    host=require_dispatch();torch.set_num_threads(4);torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False
    data=RawSceneData(ROOT/'data/TRAIN_v3',cohort='TRAIN')
    first=data.observation(data.tokens[0]);second=data.observation(data.tokens[1])
    second['images']['cam_r0']=None
    processor=QwenObservationProcessor(QWEN_PATH);model=QwenEncoder(QWEN_PATH).cuda().eval()
    def run(obs):return model(processor(obs))
    start=time.monotonic();one=run([first]);two=run([second]);batched=run([first,second]);errors=[]
    for i,single in enumerate([one,two]):
        for kind in ['visual','text']:
            indices=single[kind+'_indices'];valid=single[kind+'_mask']
            a_hidden=single['final_hidden'][0,indices[0,valid[0]]]
            bidx=batched[kind+'_indices'][i,batched[kind+'_mask'][i]]
            b_hidden=batched['final_hidden'][i,bidx]
            relative_rms=float((a_hidden-b_hidden).square().mean().sqrt()/a_hidden.square().mean().sqrt())
            # Native Qwen SDPA uses different GEMM layouts at B1/B2. This
            # diagnostic does not promise bitwise equality or change formal B1.
            # Detect masking/membership errors using whole-feature relative RMS.
            if not torch.isfinite(b_hidden).all() or relative_rms>1e-4:
                raise AssertionError('Qwen padded/single feature mismatch '+str(relative_rms))
            errors.append(dict(max_abs=float((a_hidden-b_hidden).abs().max()),relative_rms=relative_rms))
    assert set(batched['camera_ids'][1,batched['visual_mask'][1]].tolist())=={0,1}
    swapped=dict(first,images=dict(first['images']))
    swapped['images']['cam_l0'],swapped['images']['cam_r0']=swapped['images']['cam_r0'],swapped['images']['cam_l0']
    swap=run([swapped]);image_delta=float((one['final_hidden']-swap['final_hidden']).abs().mean())
    if image_delta<=1e-5:raise AssertionError('Images do not affect current hidden states')
    nav=dict(first,navigation='turn right' if first['navigation']!='turn right' else 'turn left',proprio=list(first['proprio']))
    nav['proprio'][4:]=[0.,0.,1.,0.] if nav['navigation']=='turn right' else [1.,0.,0.,0.]
    changed=run([nav]);aidx=one['visual_indices'][0];bidx=changed['visual_indices'][0]
    nav_delta=float((one['final_hidden'][0,aidx]-changed['final_hidden'][0,bidx]).abs().mean())
    if nav_delta<=1e-6:raise AssertionError('Navigation preceding images does not influence visual tokens')
    future=dict(first,requested_GT=torch.randn(8,3),reward=-999.,future_images='not an actor input')
    clean=processor([first]);dirty=processor([future])
    for k in clean:
        if torch.is_tensor(clean[k]):assert torch.equal(clean[k],dirty[k]),k
        else:assert clean[k]==dirty[k],k
    result=dict(status='PASS',qualification_only=True,host=host,precision='FP32 no autocast',
        scope='full generic Qwen observation; no driving policy score',batch_single_max_errors=errors,
        image_swap_mean_hidden_delta=image_delta,navigation_mean_visual_delta=nav_delta,
        missing_camera_mask='explicit; no image tokens for absent right view',future_labels_ignored=True,
        actual_visual_counts=batched['visual_mask'].sum(1).tolist(),wall_seconds=time.monotonic()-start)
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
