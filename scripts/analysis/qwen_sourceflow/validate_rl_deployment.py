"""Actual nonzero native LoRA versus standalone merged full raw-image ODE."""
import argparse
import gc
import json
from pathlib import Path
from unittest.mock import patch
import torch
from starVLA.rl.qwen_sourceflow.contracts import ROOT,QWEN_PATH,require_dispatch,configure_native_numerics,file_sha
from starVLA.rl.qwen_sourceflow.loading import load_il
from starVLA.rl.qwen_sourceflow.export import load_standalone,inspect_export
from starVLA.rl.qwen_sourceflow.actor import NativeRecipeActor
from starVLA.rl.qwen_sourceflow.data import RawSceneData
from starVLA.model.modules.qwen_sourceflow.qwen_encoder import QwenObservationProcessor
from starVLA.model.modules.qwen_sourceflow.action_codec import ActionCodec
from starVLA.rl.simwam_recipe.lora_adapter import restore_adapters
from starVLA.rl.simwam_recipe.contracts import seed_for
from starVLA.rl.flow_grpo.transactions import atomic_json


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--checkpoint',required=True)
    p.add_argument('--bundle',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    require_dispatch();configure_native_numerics();torch.set_num_threads(4);torch.cuda.set_device(0)
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise FileExistsError('Preserve the previous numerical validation')
    path=Path(a.checkpoint);info=json.loads((path/'identity.json').read_text())
    export=inspect_export(a.bundle)
    qualification=info['purpose']!='formal'
    if export['checkpoint']!=str(path.resolve()) or export['qualification']!=qualification:
        raise ValueError('Require the explicitly matched formal/qualification export')
    files=json.loads((path/'files.json').read_text());seal=json.loads((path/'COMPLETE').read_text())
    if file_sha(path/'files.json')!=seal['files_sha256'] or any(file_sha(path/n)!=files[n] for n in ('identity.json','optimizer.pt')):
        raise ValueError('Changed original adapter checkpoint')
    data=RawSceneData(ROOT/'data/TRAIN_v3',cohort='TRAIN');token=data.tokens[0];obs=data.observation(token)
    del data
    state=torch.tensor([obs['proprio']],device='cuda',dtype=torch.float32)
    noise=torch.randn((1,8,3),generator=torch.Generator().manual_seed(seed_for('native_qwen_sourceflow_ODE',2101,token))).cuda()
    model,parent=load_il(info['IL_parent']['path'],device='cuda',qualification=qualification)
    actor=NativeRecipeActor(model.policy.action_expert,info['recipe'])
    saved=torch.load(path/'optimizer.pt',map_location='cpu',weights_only=False)
    restore_adapters(actor,saved['adapters'])
    active=sum(bool(torch.count_nonzero(v)) for k,v in saved['adapters'].items() if k.endswith('lora_B'))
    if qualification and active!=len(actor.closure['adapters']):raise AssertionError('Require every real nonzero LoRA B projection in this learning qualification')
    batch=QwenObservationProcessor(QWEN_PATH)([obs])
    with torch.autocast('cuda',dtype=torch.bfloat16):condition=model.encode_condition(batch,state)
    expected=model.policy.sample_ode(condition,noise).cpu()
    del actor,model,condition,saved;gc.collect();torch.cuda.empty_cache()
    with patch('transformers.Qwen3VLForConditionalGeneration.from_pretrained',side_effect=AssertionError('Generic model unavailable')):
        model,processor,_=load_standalone(a.bundle,device='cuda')
        with torch.autocast('cuda',dtype=torch.bfloat16):condition=model.encode_condition(processor([obs]),state)
        actual=model.policy.sample_ode(condition,noise).cpu()
    error=float((actual-expected).abs().max())
    physical=(ActionCodec.denorm_odo(actual)-ActionCodec.denorm_odo(expected)).abs().amax((0,1)).tolist()
    report=dict(schema='native_actual_LoRA_merged_ODE_v1',status='PASS',checkpoint=str(path),bundle=a.bundle,
        updates=seal['updates'],nonzero_B_projections=active,normalized_max_abs=error,
        physical_max_abs_xy_m_heading_rad=physical,tolerance=dict(atol=1e-5,rtol=1e-5),
        raw_images=True,source_shift=5,steps=10,candidates=1,qualification_only=qualification,
        numerical_validation_only=True,
        deployment_reads_generic_weights=False,deployment_reads_training_labels_cache_teacher_scorer=False)
    try:torch.testing.assert_close(actual,expected,atol=1e-5,rtol=1e-5)
    except AssertionError:
        report['status']='FAIL';atomic_json(a.output,report);raise
    atomic_json(a.output,report);print(json.dumps(report))


if __name__=='__main__':main()
