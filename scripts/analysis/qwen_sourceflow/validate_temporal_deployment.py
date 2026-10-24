"""Actual nine-image ODE works after deleting auxiliary head; one TRAIN score."""
import argparse,json,time
from pathlib import Path
import torch
from starVLA.rl.qwen_sourceflow.temporal_contracts import ROOT,require_temporal_dispatch
from starVLA.rl.qwen_sourceflow.temporal_loading import load_temporal
from starVLA.rl.qwen_sourceflow.temporal_data import TemporalSceneData
from starVLA.rl.qwen_sourceflow.contracts import QWEN_PATH,configure_native_numerics,fingerprint
from starVLA.model.modules.qwen_sourceflow.temporal_dino import TemporalQwenProcessor
from starVLA.rl.simwam_recipe.reward_readouts import RewardService
from scripts.analysis.qwen_sourceflow.evaluate_policy import evaluation_caches,json_feedback


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();host=require_temporal_dispatch();configure_native_numerics();torch.set_num_threads(4);torch.cuda.set_device(0);start=time.monotonic()
    model,info=load_temporal(a.checkpoint,'cuda',qualification=True)
    data=TemporalSceneData(ROOT/'data/TRAIN_v3',cohort='TRAIN');token=data.tokens[0];obs=data.observation(token)
    processor=TemporalQwenProcessor(QWEN_PATH);batch=processor([obs]);state=torch.tensor([obs['proprio']],device='cuda',dtype=torch.float32)
    noise=torch.randn(1,8,3,generator=torch.Generator().manual_seed(2101),dtype=torch.float32).cuda()
    calls=model.qwen.forward_calls;out=model.predict(batch,state,noise);assert model.qwen.forward_calls==calls+1
    del model.current_dino_head
    without=model.predict(batch,state,noise)
    error=float((out['normalized_action']-without['normalized_action']).abs().max());assert error==0.,error
    changed=dict(obs,frames=[dict(f,images=dict(f['images'])) for f in obs['frames']])
    f=changed['frames'][0];f['images']['cam_l0'],f['images']['cam_r0']=f['images']['cam_r0'],f['images']['cam_l0']
    historical=model.predict(processor([changed]),state,noise)
    history_delta=float((out['normalized_action']-historical['normalized_action']).abs().mean())
    assert history_delta>0.,'Historical pixels do not affect the actor'
    assert out['normalized_action'].shape==(1,8,3) and torch.isfinite(out['requested_trajectory']).all()
    service=RewardService(Path.cwd(),workers=1,cache_roots=evaluation_caches('TRAIN'))
    try:feedback=service.score((token,data.records[token]['log'],out['requested_trajectory'].cpu().numpy()))
    finally:service.close()
    assert len(feedback)==1 and feedback[0]['status']=='OK'
    result=dict(status='PASS',qualification_only=True,host=host,scope='one true TRAIN ODE and canonical score; not planning efficacy',
        checkpoint=str(a.checkpoint),updates=info['seal']['updates'],actual_images=9,visual_tokens=int((batch['input_ids']==model.qwen.image_token_id).sum()),
        auxiliary_head_deleted_ODE_max_abs=error,history_image_swap_action_mean_abs=history_delta,
        normalized_action_shape=list(out['normalized_action'].shape),teacher_or_target_cache_used_by_inference=False,
        one_Qwen_prefill_per_prediction=True,canonical_train_score=feedback[0]['metrics']['score'],
        wall_seconds=time.monotonic()-start)
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
