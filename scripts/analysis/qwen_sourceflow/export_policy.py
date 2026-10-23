"""Merge native attention LoRA into a complete original-class Qwen-Flow model."""
import argparse
import json
import os
from pathlib import Path
import torch
from starVLA.rl.qwen_sourceflow.contracts import file_sha,require_dispatch,ACTION_CONFIG,runtime_identity
from starVLA.rl.qwen_sourceflow.export import package_processor
from starVLA.rl.qwen_sourceflow.loading import load_il
from starVLA.rl.qwen_sourceflow.actor import NativeRecipeActor
from starVLA.rl.simwam_recipe.lora_adapter import restore_adapters,merged_state_dict
from starVLA.rl.simwam_recipe.checkpoint import atomic_torch
from starVLA.rl.flow_grpo.transactions import atomic_json


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    require_dispatch();torch.set_num_threads(8);path=Path(a.checkpoint);out=Path(a.output)
    info=json.loads((path/'identity.json').read_text());seal=json.loads((path/'COMPLETE').read_text())
    files=json.loads((path/'files.json').read_text())
    if info['schema']!='native_qwen_sourceflow_RL_v1' or seal['files_sha256']!=file_sha(path/'files.json'):
        raise ValueError('Wrong/incomplete RL checkpoint')
    for name in ['optimizer.pt','identity.json']:
        if file_sha(path/name)!=files[name]:raise ValueError('Changed RL artifact '+name)
    parent_path=info['IL_parent']['path'];qualification=info['purpose']!='formal'
    model,parent=load_il(parent_path,qualification=qualification,parent_selection=info.get('parent_selection_path'))
    if parent['files']['model.pt']!=info['SFT_sha256']:raise ValueError('IL parent changed')
    actor=NativeRecipeActor(model.policy.action_expert,info['recipe'])
    saved=torch.load(path/'optimizer.pt',map_location='cpu',weights_only=False);restore_adapters(actor,saved['adapters'])
    merged=merged_state_dict(actor.action_expert)
    if set(merged)!=set(actor.ref_action_expert.state_dict()) or any('lora_' in k or '.base.' in k for k in merged):
        raise ValueError('Non-native merged action state')
    prefix='policy.runner.mixtures.action.'
    state={k:v for k,v in model.state_dict().items() if not k.startswith(prefix)}
    state.update({prefix+k:v for k,v in merged.items()})
    if out.exists():raise FileExistsError(out)
    attempt=out.with_name(out.name+'.incomplete');attempt.mkdir(parents=True,exist_ok=False)
    atomic_torch(attempt/'model.pt',state)
    processor_files=package_processor(attempt/'processor')
    record=dict(schema='native_Qwen_sourceflow_merged_deployment_v1',checkpoint=str(path.resolve()),
        run_identity=info['sha256'],parent_model_sha256=parent['files']['model.pt'],updates=seal['updates'],
        model_sha256=file_sha(attempt/'model.pt'),qualification=qualification,
        inference='single_candidate_source_shift5_ODE_K10',action_shape=[8,3],
        action_config=ACTION_CONFIG,runtime=runtime_identity(),processor_files=processor_files,
        no_training_cache_teacher_scorer_dependency=True,not_resume_use_native_unmerged_checkpoint=True,
        merge='FP32 B@A, frozen original base plus delta; full Qwen/provider included')
    atomic_json(attempt/'export.json',record);atomic_json(attempt/'COMPLETE',dict(export_sha256=file_sha(attempt/'export.json')))
    os.rename(attempt,out);print(json.dumps(record))


if __name__=='__main__':main()
