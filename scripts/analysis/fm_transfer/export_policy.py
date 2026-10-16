"""Export a sealed new SFT or merged RL head through the original VLAAgent.

The original frozen Qwen/history/input weights are copied exactly. Export is
never a native optimizer resume; use the unmerged checkpoint for that purpose.
"""
import argparse
import json
import os
from pathlib import Path
import uuid
import torch
from omegaconf import OmegaConf
from starVLA.rl.fm_transfer.contracts import require_dispatch
from starVLA.rl.fm_transfer.rl_adapter import load_supervised_policy, make_transfer_actor
from starVLA.rl.simwam_recipe.contracts import SFT_EXPORT, SFT_SHA256, file_sha, fingerprint
from starVLA.rl.simwam_recipe.lora_adapter import restore_adapters, merged_state_dict
from starVLA.rl.simwam_recipe.recovery_state import validate_boundary
from starVLA.rl.simwam_recipe.checkpoint import atomic_torch
from starVLA.rl.flow_grpo.transactions import atomic_json


def export(checkpoint, output, *, qualification=False):
    checkpoint=Path(checkpoint);output=Path(output)
    if output.exists():raise FileExistsError(output)
    identity=json.loads((checkpoint/'identity.json').read_text())
    seal=json.loads((checkpoint/'COMPLETE').read_text())
    if '.incomplete' in checkpoint.name or seal['files_sha256']!=file_sha(checkpoint/'files.json'):
        raise ValueError('Unpublished checkpoint')
    files=json.loads((checkpoint/'files.json').read_text())
    if file_sha(checkpoint/'identity.json')!=files['identity.json']:raise ValueError('Changed identity')
    if identity['schema']=='fm_transfer_SFT_identity_v1':
        policy,sft=load_supervised_policy(checkpoint,device='cpu',qualification=qualification,require_endpoint=False)
        head=policy.head;architecture=policy.architecture;merged=head.state_dict();stage='SFT'
        changed=None
    elif identity['schema']=='fm_transfer_source_RL_v1':
        if identity['purpose']!='formal' and not qualification:raise ValueError('Qualification is not a formal export')
        sft=identity['recipe']['supervised_transfer']['SFT_identity']
        actor,actual=make_transfer_actor(sft['checkpoint'],identity['recipe'],device='cpu',qualification=qualification)
        if sft!=actual:raise ValueError('Changed supervised parent')
        for name in ('optimizer.pt','rank0.pt'):
            if file_sha(checkpoint/name)!=files[name]:raise ValueError('Changed RL payload')
        common=torch.load(checkpoint/'optimizer.pt',map_location='cpu',weights_only=False)
        own=torch.load(checkpoint/'rank0.pt',map_location='cpu',weights_only=False)
        validate_boundary(own['state'],own['rollout'],identity,common['optimizer'],common['scheduler'])
        restore_adapters(actor,common['adapters'])
        merged=merged_state_dict(actor.action_expert);reference=actor.ref_action_expert.state_dict()
        attention={r['path'].removeprefix('action_model.')+'.weight' for r in actor.closure['adapters']}
        changed=[k for k in merged if not torch.equal(merged[k],reference[k])]
        if not set(changed)<=attention:raise ValueError('RL changed frozen non-attention weights')
        architecture=sft['config']['architecture'];stage='RL'
    else:raise ValueError('Not a registered new-base checkpoint')
    if any('lora_' in k or '.base.' in k for k in merged):raise ValueError('Unmerged deployment state')
    original=Path(SFT_EXPORT)
    if file_sha(original/'pytorch_model.pt')!=SFT_SHA256:raise ValueError('Original prefix identity changed')
    with torch.serialization.safe_globals([set]):
        old=torch.load(original/'pytorch_model.pt',map_location='cpu',weights_only=True,mmap=True)
    old=old.get('module',old)
    state={k:v for k,v in old.items() if not k.startswith('action_model.')}
    prefix='action_model.' if architecture=='ddp' else 'action_model.policy.head.'
    state.update({prefix+k:v.detach().cpu().float() for k,v in merged.items()})
    cfg=OmegaConf.load(original/'config.yaml')
    if architecture=='source_action_dit':
        OmegaConf.update(cfg,'framework.action_model.standalone_source_action',True,force_add=True)
    attempt=output.with_name(output.name+'.incomplete.'+uuid.uuid4().hex)
    attempt.mkdir(parents=True,exist_ok=False)
    atomic_torch(attempt/'pytorch_model.pt',state)
    OmegaConf.save(cfg,attempt/'config.yaml')
    report=dict(schema='fm_transfer_original_entry_export_v1',stage=stage,architecture=architecture,
        checkpoint=str(checkpoint),run_identity=identity['sha256'],updates=seal['updates'],
        new_SFT=sft,original_frozen_prefix_sha256=SFT_SHA256,qualification=qualification,
        weight_sha256=file_sha(attempt/'pytorch_model.pt'),config_sha256=file_sha(attempt/'config.yaml'),
        changed_attention_from_own_SFT=changed,frozen_upstream_exact=True,
        deployment='original VLAAgent observation/encoding/decoding; no cache, reference, scorer or LoRA wrapper',
        not_exact_resume_use_native_checkpoint=True)
    report['sha256']=fingerprint(report)
    atomic_json(attempt/'export.json',report);atomic_json(attempt/'COMPLETE',report)
    os.rename(attempt,output)
    return report


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True);p.add_argument('--qualification',action='store_true')
    args=p.parse_args();require_dispatch()
    print(json.dumps(export(args.checkpoint,args.output,qualification=args.qualification)))


if __name__=='__main__':main()
