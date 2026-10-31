"""Pin an original COMPLETE boundary and publish a qualified execution receipt.

Hardlinks preserve native model/Adam/RNG files without copying large tensors.
The original checkpoint, seal, identity and metrics are never rewritten.
"""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
if __package__ in(None,''):
    script_dir=Path(__file__).resolve().parent
    sys.path[:]=[x for x in sys.path if Path(x or os.curdir).resolve()!=script_dir]
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from starVLA.rl.qwen_world_compare.contracts import file_sha,fingerprint
from starVLA.rl.qwen_world_compare.execution_migration import SCHEMA,validate_patch
from starVLA.rl.qwen_world_compare.loading import inspect_IL_checkpoint
from starVLA.rl.qwen_world_postcot.checkpoint import atomic_json
from tools.qwen_world_compare.train_il import source_files


def adopt_campaign(root,source,proof_root,output,execution_migration=None):
    """Metadata-only owner adoption. Existing native workers are untouched."""
    from tools.qwen_world_compare.campaign import make_plan,expected_source_files
    root,source,output=map(Path,(root,source,output));directory=root/'campaign'
    with(directory/'owner.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        old=json.loads((root/'RUNS.json').read_text());state=json.loads((directory/'STATE.json').read_text())
        plan=make_plan(source/'configs/qwen_world_compare/campaign.yaml',source,execution_migration=execution_migration)
        full=expected_source_files(source);train=source_files();proofs={}
        for key in('root','common','data_paths','stages','IL_updates','RL_updates','IL_DEV_updates','RL_DEV_updates',
                    'inference','train_seed','evaluation_seed','DEV_count','NAVTEST_count','logical_IL_scene_exposures_per_arm',
                    'logical_RL_fresh_rollouts_per_arm','logical_RL_scene_exposures_per_arm','logical_RL_candidates_per_arm'):
            if plan[key]!=old['plan'][key]:raise ValueError('Campaign scientific plan changed: '+key)
        for arch in 'ABCD':
            proof=Path(proof_root)/arch/'COMPARISON.json';value=json.loads(proof.read_text())
            if(value.get('status')!='PASS'or value.get('architecture_id')!=arch or value.get('world')!=8
                or value.get('source_files')!=train or value.get('actual_optimizer_updates_and_native_states_compared')is not True):
                raise ValueError('Missing actual source-bound native continuation: '+arch)
            arm=plan['architectures'][arch];previous=old['plan']['architectures'][arch]
            for key in('config_sha256','IL','RL','selection_IL','selection_RL'):
                if arm[key]!=previous[key]:raise ValueError('Campaign arm changed: '+arch+'/'+key)
            prior=Path(previous['qualification']);accept=json.loads(prior.read_text())
            if accept['status']!='PASS'or accept['source_files']!=old['source_files']:raise ValueError('Original qualification changed')
            passed=dict(status='PASS',architecture_id=arch,source_files=full,config_sha256=arm['config_sha256'],
                formal_initialized_from_qualification=False,validation_kind='original_qualification_plus_exact_native_execution_continuation',
                original_qualification=dict(path=str(prior),sha256=file_sha(prior)),
                native_equivalence=dict(path=str(proof.resolve()),sha256=file_sha(proof)))
            target=Path(arm['qualification']);target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists()and json.loads(target.read_text())!=passed:raise ValueError('Refusing to overwrite qualification lineage')
            atomic_json(target,passed);proofs[arch]=passed['native_equivalence']
        if state['plan_sha256']!=old['plan']['sha256']or state['source_files']!=old['source_files']:
            raise ValueError('Old owner state is not the archived campaign')
        output.mkdir(parents=True,exist_ok=False)
        for name,path in [('RUNS',root/'RUNS.json'),('STATE',directory/'STATE.json'),('OWNER',directory/'OWNER.json')]:
            (output/(name+'_BEFORE.json')).write_bytes(path.read_bytes())
        state['plan_sha256']=plan['sha256'];state['source_files']=full
        registry=dict(schema='qwen_world_compare_IL_handoff_registry_v1',architectures={a:dict(status='HOLD')for a in 'ABCD'})
        atomic_json(directory/'IL_EXECUTION_HANDOFFS.json',registry)
        atomic_json(root/'RUNS.json',dict(plan=plan,source_files=full,source_files_sha256=fingerprint(full)))
        atomic_json(directory/'STATE.json',state)
        atomic_json(output/'COMPLETE.json',dict(status='PASS',kind='metadata_only_existing_native_workers_preserved',
            old_source=old['plan']['source'],new_source=str(source),proofs=proofs,plan_sha256=plan['sha256']))
        print(json.dumps(dict(status='PASS',existing_native_workers_preserved=True,receipt=str(output/'COMPLETE.json'))))


def activate(root,arch,receipt,host,lease_slot,minimum):
    from starVLA.rl.qwen_world_compare.execution_migration import validate_receipt
    receipt=Path(receipt).resolve();value=json.loads(receipt.read_text());validate_receipt(receipt,value['new_identity'])
    if value['architecture_id']!=arch:raise ValueError('Another architecture owns this receipt')
    from starVLA.rl.qwen_world_compare.resources import probe,allowed_hosts
    if host not in allowed_hosts(admit_long_job=True):raise PermissionError('Host permission withdrawn')
    devices=probe(host);registry=Path(root)/'campaign/IL_EXECUTION_HANDOFFS.json'
    with registry.with_suffix('.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        table=json.loads(registry.read_text())
        if table['architectures'][arch]['status']!='HOLD':raise ValueError('Do not replace an active handoff')
        table['architectures'][arch]=dict(status='ACTIVE',receipt=str(receipt),receipt_file_sha256=file_sha(receipt),
            host=host,uuids=[g['uuid']for g in devices['gpus']],lease_slot=lease_slot,share_min_free_mib=minimum)
        atomic_json(registry,table)
    print(json.dumps(dict(status='ACTIVE',architecture=arch,host=host,receipt=str(receipt))))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--checkpoint');p.add_argument('--output')
    p.add_argument('--checkpoint-policy',choices=('short_off','all_off'))
    p.add_argument('--cpu-prefetch-depth',type=int,choices=(1,2),default=2)
    p.add_argument('--proof',action='append')
    p.add_argument('--adopt-campaign');p.add_argument('--native-proof-root')
    p.add_argument('--training-source');p.add_argument('--controller-migration')
    p.add_argument('--activate-architecture',choices=list('ABCD'));p.add_argument('--root');p.add_argument('--receipt');p.add_argument('--host')
    p.add_argument('--lease-slot',choices=('primary','shared'),default='shared');p.add_argument('--share-min-free-mib',type=int,default=40000)
    a=p.parse_args();source=Path(__file__).resolve().parents[2]
    if a.adopt_campaign:
        if not a.native_proof_root or not a.output:raise ValueError('Adoption requires actual native proofs and an archive directory')
        return adopt_campaign(a.adopt_campaign,Path(a.training_source).resolve()if a.training_source else source,
            a.native_proof_root,a.output,a.controller_migration)
    if a.activate_architecture:
        return activate(a.root,a.activate_architecture,a.receipt,a.host,a.lease_slot,a.share_min_free_mib)
    if not(a.checkpoint and a.output and a.checkpoint_policy and a.proof):raise ValueError('Pinning needs parent, execution policy and actual equivalence proofs')
    out=Path(a.output)
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
