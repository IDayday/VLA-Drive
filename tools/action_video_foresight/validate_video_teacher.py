"""Strict loading and true eight-frame, rectangular temporal-path checks."""
import argparse,json,subprocess
from pathlib import Path
import torch
from PIL import Image
from starVLA.model.modules.foresight.video_target_encoder import VideoTargetEncoder
from starVLA.model.modules.vehicle_joint.initialization import file_sha256
from tools.ddpolicy_vehicle.prepare_data import atomic_json
from tools.ddpolicy_vehicle.run_meter import metered_run
from tools.action_video_foresight.build_clip_index import one_log


def main():
    p=argparse.ArgumentParser(__doc__)
    for k in ('source-root','source-sha','checkpoint','weight-sha256','train-data','raw-log-root','sensor-root','output','campaign-root','run-id'):p.add_argument('--'+k,required=True)
    p.add_argument('--fallback-sensor-root',action='append',default=[]);a=p.parse_args()
    if subprocess.check_output(['git','status','--porcelain']).strip():raise ValueError('Freeze validation source')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
    with metered_run(a.campaign_root,a.run_id,1,{'kind':'video_teacher_validation','real_optimizer_updates':0}) as (meter,_,save):
        model=VideoTargetEncoder(a.source_root,a.checkpoint,source_sha=a.source_sha,weight_sha256=a.weight_sha256).cuda()
        index=json.loads((Path(a.train_data)/'index.json').read_text());cfg=vars(a)|{'tolerance':.05}
        scenes,_=one_log((index[0]['log'],[r['token'] for r in index if r['log']==index[0]['log']][:4],'train',cfg))
        scene=next(r for r in scenes if all(r['clip_view_valid']))
        frames=[Image.open(r['path']).convert('RGB') for r in scene['frames_by_view'][0]]
        x=model.preprocess(frames)[None].cuda();z=model(x);again=model(x)
        rev=model(x.flip(2));order=[0,2,4,6,1,3,5,7];permuted=model(x[:,:,order])
        current=Image.open(scene['current_paths'][0]).convert('RGB');static=model(model.preprocess([current]*8)[None].cuda())
        # Same official encoder/weights, actual image tokenizer; limited diagnostic only.
        image_output=model.encoder(x[:,:,0:1])
        expected=model.preprocess(frames)
        result={'source':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'teacher':model.identity,
            'strict_load':True,'missing_keys':[],'unexpected_keys':[],'real_scene':scene,'output_shape':list(z.shape),
            'image_path_shape':list(image_output.shape),'deterministic_exact':torch.equal(z,again),
            'reverse_mse':float((z-rev).square().mean()),'time_permutation_mse':float((z-permuted).square().mean()),
            'static_current_clip_mse':float((z-static).square().mean()),'cross_time_variance':float(z.var(dim=1,unbiased=False).mean()),
            'feature_mean':float(z.mean()),'feature_variance':float(z.var(unbiased=False)),
            'fp16_quantization_mse':float((z-z.half().float()).square().mean()),'all_finite':bool(torch.isfinite(z).all()),
            'inputs_source_sha256':[file_sha256(r['path']) for r in scene['frames_by_view'][0]],
            'static_control':'repeated current image only for diagnostic, never used to fill missing clips',
            'real_optimizer_updates':0}
        if not result['deterministic_exact'] or not result['all_finite']:raise RuntimeError('Invalid frozen teacher')
        out=Path(a.output);out.mkdir(parents=True,exist_ok=False);atomic_json(out/'RESULTS.json',result)
        torch.save({'video':z.cpu().clone(),'static':static.cpu().clone(),'input':expected},out/'real_clip.pt')
        meter['inference_clips']=5;save()

if __name__=='__main__':main()
