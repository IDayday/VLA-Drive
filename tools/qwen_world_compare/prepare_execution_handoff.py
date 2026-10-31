"""Pin an original COMPLETE boundary and publish a qualified execution receipt.

Hardlinks preserve native model/Adam/RNG files without copying large tensors.
The original checkpoint, seal, identity and metrics are never rewritten.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
from starVLA.rl.qwen_world_compare.contracts import file_sha,fingerprint
from starVLA.rl.qwen_world_compare.execution_migration import SCHEMA,validate_patch
from starVLA.rl.qwen_world_compare.loading import inspect_IL_checkpoint
from starVLA.rl.qwen_world_postcot.checkpoint import atomic_json
from tools.qwen_world_compare.train_il import source_files


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--checkpoint-policy',choices=('short_off','all_off'),required=True)
    p.add_argument('--cpu-prefetch-depth',type=int,choices=(1,2),default=2)
    p.add_argument('--proof',action='append',required=True)
    a=p.parse_args();out=Path(a.output);source=Path(__file__).resolve().parents[2]
    if (out/'MIGRATION.json').exists():raise FileExistsError('Do not replace a published handoff')
    info=inspect_IL_checkpoint(a.checkpoint,verify_files='inventory');old=info['identity']
    if old['qualification']or old['world']!=8:raise ValueError('Only original formal native boundaries can migrate')
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=source,text=True).strip():
        raise ValueError('Execution source has uncommitted tracked changes')
    new=copy.deepcopy(old);new['source_files']=source_files()
    new['execution']['execution_tuning']=dict(checkpoint_policy=a.checkpoint_policy,cpu_prefetch_depth=a.cpu_prefetch_depth)
    changes=validate_patch(old,new);proofs=[]
    for path in a.proof:
        path=Path(path).resolve();value=json.loads(path.read_text())
        if value.get('status')!='PASS':raise ValueError('Actual qualification is incomplete/failed: '+str(path))
        proofs.append(dict(path=str(path),sha256=file_sha(path)))
    # A source-bound eight-rank continuation is mandatory, in addition to the
    # per-architecture real-model gradient and CPU data tests.
    native=[json.loads(Path(x['path']).read_text())for x in proofs]
    if not any(v.get('world')==8 and v.get('architecture_id')==old['architecture_id']and v.get('actual_optimizer_updates_and_native_states_compared')is True
               and v.get('source_files')==new['source_files']for v in native):
        raise ValueError('No eight-rank native continuation proof for this exact source')
    original=Path(info['path']);parent=out/'PARENT';parent.mkdir(parents=True,exist_ok=False)
    names=set(info['files'])|{'identity.json','files.json','COMPLETE'}
    for name in sorted(names):
        src=original/name;dst=parent/name
        if not src.resolve().is_relative_to(original.resolve()):raise ValueError('Escaping native path')
        dst.parent.mkdir(parents=True,exist_ok=True);os.link(src,dst)
    receipt=dict(schema=SCHEMA,status='PASS',architecture_id=old['architecture_id'],
        parent_checkpoint=str(parent.resolve()),original_parent_checkpoint=str(original),
        parent_COMPLETE_sha256=file_sha(parent/'COMPLETE'),parent_inventory_sha256=file_sha(parent/'files.json'),
        old_identity=old,new_identity=new,changes=changes,proofs=proofs,
        execution_source=str(source),execution_git_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip(),
        qualification_weights_used_as_formal_initialization=False,
        preserved=['parameters','Adam_FP32','scheduler','RNG_all_ranks','builder','scene_cursor','effective_updates'],
        budgets_and_objectives_unchanged=True)
    receipt['sha256']=fingerprint(receipt);atomic_json(out/'MIGRATION.json',receipt)
    print(json.dumps(dict(receipt=str(out/'MIGRATION.json'),parent=str(parent),updates=info['seal']['updates'],sha256=receipt['sha256'])))


if __name__=='__main__':main()
