"""CPU strict8rank native validation before the original deployment exporter."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

from starVLA.rl.qwen_world_verified_quality.recipe import ROOT,ARM,PROTOCOL,QUALITY_PROTOCOL_SHA256,validate_recipe


def read(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(2**20),b''):h.update(part)
    return h.hexdigest()


def load_validated(parent,checkpoint,*,qualification=False,allow_pending=False):
    import torch
    from starVLA.rl.qwen_world_verified_quality import recovery
    from starVLA.rl.qwen_world_postcot.loading import inspect_checkpoint
    from starVLA.rl.qwen_world_postcot.contracts import fingerprint
    root=Path(checkpoint).resolve();seal=read(root/'COMPLETE');identity=read(root/'identity.json');files=read(root/'files.json')
    if allow_pending and not qualification:raise ValueError('Pending payload review is qualification-only; never export it')
    phases=('READY','COMPLETE','INNER_COMMITTED','ROLLOUT_COMMITTED') if allow_pending else ('READY','COMPLETE')
    if (seal.get('schema'),seal.get('world'))!=(2,8) or (seal.get('next_inner')!=0 and not allow_pending) or seal['phase'] not in phases:
        raise ValueError('S6 periodic export requires an actual complete-window world8 native checkpoint')
    if sha(root/'files.json')!=seal['files_sha256']:raise ValueError('S6 native inventory seal changed')
    required={'optimizer.pt','identity.json','transaction.json'}|{f'rank{r}.pt' for r in range(8)}
    if not required.issubset(files) or any(sha(root/name)!=digest for name,digest in files.items()):raise ValueError('S6 native payload missing or changed')
    if identity['sha256']!=fingerprint({k:v for k,v in identity.items() if k!='sha256'}):raise ValueError('S6 identity seal changed')
    cfg=validate_recipe(identity['recipe'])
    if identity.get('world')!=8 or identity.get('parameter_contract',{}).get('trainable_numel')!=52_707_840:
        raise ValueError('S6 export requires the real complete world8/52.7M scope')
    if not qualification:
        if identity.get('purpose')!='formal':raise ValueError('Qualification weights cannot be silently exported as a formal model')
        runs=read(ROOT/'RUNS.json')['runs'];run=next((r for r in runs if r['id']==ARM),None)
        if not run or not root.is_relative_to(Path(run['root']).resolve()/'checkpoints') or sha(run['config'])!=run['config_sha256']:
            raise ValueError('S6 native export is not owned by its exact registered formal run')
        if read(Path(run['root'])/'identity.json')!=identity:raise ValueError('S6 export identity differs from the registered run')
    elif identity.get('purpose')!='qualification_not_formal_initialization':raise ValueError('Explicit qualification export requires qualification identity')
    parent_info=inspect_checkpoint(parent,verify_files='model')
    if parent_info['model_sha256']!=identity['IL_parent']['model_sha256'] or parent_info['model_sha256']!=cfg['research']['performance']['parent_model_sha256']:
        raise ValueError('S6 export parent model differs from the same complete IL70')
    common=torch.load(root/'optimizer.pt',map_location='cpu',weights_only=False);transaction=read(root/'transaction.json');summaries=[]
    for rank in range(8):
        own=torch.load(root/f'rank{rank}.pt',map_location='cpu',weights_only=False)
        if own['identity_sha256']!=identity['sha256']:raise ValueError('S6 rank identity changed')
        summary=recovery.validate_boundary(own['state'],own['rollout'],identity,common['optimizer'],common['scheduler'],expected_rank=rank)
        if summary!=transaction['ranks'][rank] or summary['phase']!=seal['phase']:raise ValueError('S6 native transaction summary changed')
        summaries.append(summary)
    recovery.validate_ranks(summaries)
    del common,own;gc.collect()
    first=summaries[0]
    return dict(schema='world_verified_quality_export_native_validation_v1',status='PASS',checkpoint=str(root),
        files_sha256=seal['files_sha256'],identity_sha256=identity['sha256'],quality_protocol_sha256=QUALITY_PROTOCOL_SHA256,
        counts={k:first[k] for k in ('updates','next_inner','collections','scene_cursor','valid_scene_exposures','new_candidates')},
        parent_model_sha256=parent_info['model_sha256'],qualification=qualification,world=8,
        source=str(Path.cwd()),source_file_sha256=sha(__file__),at_unix=time.time())


def strict_export(parent,output,checkpoint,*,qualification=False):
    from starVLA.rl.qwen_world_postcot.deployment import export_checkpoint,inspect_export
    from starVLA.rl.flow_grpo.transactions import atomic_json
    receipt=load_validated(parent,checkpoint,qualification=qualification)
    output=Path(output);atomic_json(output.with_name(output.name+'.native_validation.json'),receipt)
    if not (output/'COMPLETE').exists():export_checkpoint(parent,output,checkpoint)
    meta=inspect_export(output)
    if meta.get('RL_checkpoint_files_sha256')!=receipt['files_sha256'] or meta.get('RL_updates')!=receipt['counts']['updates']:
        raise ValueError('S6 deployment export is bound to another native checkpoint')
    return meta


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('action',choices=['export'])
    for name in ('parent-checkpoint','rl-checkpoint','output'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--qualification-native',action='store_true')
    args=parser.parse_args();strict_export(args.parent_checkpoint,args.output,args.rl_checkpoint,qualification=args.qualification_native)

if __name__=='__main__':main()
