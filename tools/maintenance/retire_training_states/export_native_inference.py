"""Export a Qwen-Flow policy after explicitly authorized optimizer retirement.

Uses the original load/LoRA restore/FP32 merge/processor functions. This reads
the preserved adapter capsule, not an invented resumable optimizer checkpoint.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def read(path): return json.loads(Path(path).read_text())


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(16*1024**2),b''):digest.update(data)
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint',required=True);parser.add_argument('--source',required=True)
    destination=parser.add_mutually_exclusive_group(required=True)
    destination.add_argument('--output');destination.add_argument('--verify-against')
    parser.add_argument('--proof');args=parser.parse_args()
    if args.verify_against and not args.proof:parser.error('--verify-against requires --proof')
    source=Path(args.source).resolve();sys.path.insert(0,str(source));os.chdir(source)
    import torch
    from starVLA.rl.qwen_sourceflow.contracts import require_dispatch,ACTION_CONFIG,runtime_identity
    from starVLA.rl.qwen_sourceflow.loading import load_il
    from starVLA.rl.qwen_sourceflow.actor import NativeRecipeActor
    from starVLA.rl.qwen_sourceflow.export import package_processor
    from starVLA.rl.simwam_recipe.lora_adapter import restore_adapters,merged_state_dict
    from starVLA.rl.simwam_recipe.checkpoint import atomic_torch
    from starVLA.rl.flow_grpo.transactions import atomic_json
    require_dispatch();torch.set_num_threads(8)
    checkpoint=Path(args.checkpoint).resolve();output=Path(args.output).resolve() if args.output else None
    identity=read(checkpoint/'identity.json');seal=read(checkpoint/'COMPLETE');files=read(checkpoint/'files.json')
    if identity['schema']!='native_qwen_sourceflow_RL_v1' or sha(checkpoint/'files.json')!=seal['files_sha256']:
        raise ValueError('Wrong or changed original policy identity')
    if sha(checkpoint/'identity.json')!=files['identity.json']:raise ValueError('Changed original identity')
    receipt=read(checkpoint/'RECOVERY_REMOVED_BY_USER.json')
    entries=[r for r in receipt['files'] if r.get('name')=='optimizer.pt' and r.get('inference_payload_sha256')]
    if receipt['resume_available'] is not False or len(entries)!=1:raise ValueError('Missing explicit inference retirement receipt')
    entry=entries[0];capsule=checkpoint/'inference_adapters.pt'
    if sha(capsule)!=entry['inference_payload_sha256']:raise ValueError('Changed preserved inference weights')
    saved=torch.load(capsule,map_location='cpu',weights_only=False)
    if 'adapters' not in saved or 'optimizer' in saved:raise ValueError('Not an inference-only adapter archive')
    model,parent=load_il(identity['IL_parent']['path'],qualification=identity['purpose']!='formal',
        parent_selection=identity.get('parent_selection_path'))
    if parent['files']['model.pt']!=identity['SFT_sha256']:raise ValueError('Changed frozen IL parent')
    actor=NativeRecipeActor(model.policy.action_expert,identity['recipe']);restore_adapters(actor,saved['adapters'])
    merged=merged_state_dict(actor.action_expert)
    if set(merged)!=set(actor.ref_action_expert.state_dict()):raise ValueError('Incomplete native action weights')
    prefix='policy.runner.mixtures.action.'
    state={k:v for k,v in model.state_dict().items() if not k.startswith(prefix)}
    state.update({prefix+k:v for k,v in merged.items()})
    if args.verify_against:
        reference=Path(args.verify_against).resolve();meta=read(reference/'export.json')
        if meta['updates']!=seal['updates'] or meta['parent_model_sha256']!=parent['files']['model.pt']:
            raise ValueError('Different previously qualified inference policy')
        expected=torch.load(reference/'model.pt',mmap=True,map_location='cpu',weights_only=True)
        if set(expected)!=set(state):raise ValueError('Different deployment tensor inventory')
        for name,value in state.items():
            other=expected[name]
            if value.dtype!=other.dtype or value.shape!=other.shape or not torch.equal(value,other):
                raise ValueError('Preserved-adapter deployment differs: '+name)
        proof=dict(status='EXACT_ALL_DEPLOYMENT_TENSORS',checkpoint=str(checkpoint),reference=str(reference),
            updates=seal['updates'],tensors=len(state),adapter_sha256=entry['inference_payload_sha256'],
            parent_model_sha256=parent['files']['model.pt'],max_absolute_error=0,
            new_model_payload_written=False,optimizer_or_rng_reconstructed=False)
        atomic_json(Path(args.proof).resolve(),proof);print(json.dumps(proof));return
    if output.exists():raise FileExistsError(output)
    temporary=output.with_name(output.name+'.building');temporary.mkdir(parents=True,exist_ok=False)
    atomic_torch(temporary/'model.pt',state);processor=package_processor(temporary/'processor')
    record=dict(schema='native_Qwen_sourceflow_merged_deployment_v1',checkpoint=str(checkpoint),
        run_identity=identity['sha256'],parent_model_sha256=parent['files']['model.pt'],updates=seal['updates'],
        model_sha256=sha(temporary/'model.pt'),qualification=identity['purpose']!='formal',
        inference='single_candidate_source_shift5_ODE_K10',action_shape=[8,3],action_config=ACTION_CONFIG,
        runtime=runtime_identity(),processor_files=processor,no_training_cache_teacher_scorer_dependency=True,
        not_resume_use_native_unmerged_checkpoint=True,resume_available=False,
        inference_archive=dict(receipt=str(checkpoint/'RECOVERY_REMOVED_BY_USER.json'),
            adapter_sha256=entry['inference_payload_sha256'],original_optimizer_sha256=files['optimizer.pt']),
        merge='Original FP32 B@A and frozen native base; retired Adam/RNG are not reconstructed')
    atomic_json(temporary/'export.json',record);atomic_json(temporary/'COMPLETE',dict(export_sha256=sha(temporary/'export.json')))
    os.rename(temporary,output);print(json.dumps(dict(status='COMPLETE',updates=seal['updates'],model_sha256=record['model_sha256'])))


if __name__=='__main__':main()
