"""Real raw-image ODE parity without generic weights, parent, teacher or scorer."""
import argparse
import gc
import json
import os
from pathlib import Path
from unittest.mock import patch
import torch
from starVLA.rl.qwen_sourceflow.contracts import ROOT,QWEN_PATH,ACTION_CONFIG,require_dispatch,configure_native_numerics,file_sha,runtime_identity
from starVLA.rl.qwen_sourceflow.loading import load_il
from starVLA.rl.qwen_sourceflow.data import RawSceneData
from starVLA.rl.qwen_sourceflow.export import package_processor,load_standalone
from starVLA.model.modules.qwen_sourceflow.qwen_encoder import QwenObservationProcessor
from starVLA.rl.flow_grpo.transactions import atomic_json


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--bundle',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    require_dispatch();configure_native_numerics();torch.set_num_threads(4);torch.cuda.set_device(0)
    data=RawSceneData(ROOT/'data/TRAIN_v3',cohort='TRAIN');token=data.tokens[0];obs=data.observation(token)
    del data
    state=torch.tensor([obs['proprio']],device='cuda',dtype=torch.float32)
    noise=torch.randn((1,8,3),generator=torch.Generator().manual_seed(2101)).cuda()
    model,parent=load_il(a.checkpoint,device='cuda',qualification=True)
    batch=QwenObservationProcessor(QWEN_PATH)([obs])
    with torch.autocast('cuda',dtype=torch.bfloat16):condition=model.encode_condition(batch,state)
    expected=model.policy.sample_ode(condition,noise).cpu();del condition
    bundle=Path(a.bundle);bundle.mkdir(parents=True,exist_ok=False)
    # Qualification bundle reuses the exact full IL payload; real RL exports
    # run the separately tested LoRA merge before publishing this same schema.
    os.link(Path(a.checkpoint)/'model.pt',bundle/'model.pt')
    record=dict(schema='native_Qwen_sourceflow_merged_deployment_v1',qualification=True,
        checkpoint=a.checkpoint,parent_model_sha256=parent['files']['model.pt'],updates=parent['seal']['updates'],
        model_sha256=file_sha(bundle/'model.pt'),action_config=ACTION_CONFIG,runtime=runtime_identity(),
        processor_files=package_processor(bundle/'processor'),action_shape=[8,3],
        inference='single_candidate_source_shift5_ODE_K10',test_scope='full IL standalone export, no RL merge claim')
    atomic_json(bundle/'export.json',record);atomic_json(bundle/'COMPLETE',dict(export_sha256=file_sha(bundle/'export.json')))
    del model;gc.collect();torch.cuda.empty_cache()
    # No model/label/scorer/parent loading function is available to this path.
    with patch('transformers.Qwen3VLForConditionalGeneration.from_pretrained',side_effect=AssertionError('Generic weights forbidden at deployment')):
        model,processor,_=load_standalone(bundle,device='cuda')
        with torch.autocast('cuda',dtype=torch.bfloat16):condition=model.encode_condition(processor([obs]),state)
        actual=model.policy.sample_ode(condition,noise).cpu()
    err=float((expected-actual).abs().max());torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-6)
    atomic_json(a.output,dict(status='PASS',token=token,checkpoint=a.checkpoint,bundle=str(bundle),
        standalone_ODE_max_abs=err,exact=torch.equal(actual,expected),no_generic_model_load=True,
        raw_images=True,full_visual_tokens=True,parent_teacher_score_cache_not_read_by_standalone=True,
        qualification_only=True,peak_GPU_bytes=torch.cuda.max_memory_allocated()))
    print(json.dumps(dict(status='PASS',ODE_max_abs=err)))


if __name__=='__main__':main()
