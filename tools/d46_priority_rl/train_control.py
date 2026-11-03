"""Administrative600s CPU-control startup on byte-identical original SCI.

This file lies OUTSIDE tools.d44_native_rl.train.code_identity roots. No actor,
loss, optimizer, checkpoint math, saved identity or RNG tensor is changed.
"""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
import time
import yaml
from starVLA.rl.qwen_world_compare.contracts import file_sha,fingerprint
from starVLA.rl.flow_grpo.transactions import atomic_json

ROOT=Path('/mnt/project/d46-priority-rl-20261103')
SOURCE=Path(__file__).resolve().parents[2]
WRAPPER='tools/d46_priority_rl/train_control.py'
MODEL_SHA='386a0a27bd197cb002501bcb90c343659001d2e6d4f7bad0cbd5e31a91f8477d'
SCHEMA='d46_world8_control_startup_execution_profile_v1'
PROTOCOL=dict(schema='d46_world8_CPU_control_timeout600_v1',world=8,backend='gloo',timeout_seconds=600,
    original_timeout_seconds=90,changes='CPU startup/control wait only',policy_optimizer_RNG_identity_unchanged=True)
ARMS=('D0','D1','D2','D3_FPO','D3_AWM','D5')


def read(path):return json.loads(Path(path).read_text())

def argument(argv,name):
    if argv.count(name)!=1:raise ValueError('Exactly one original '+name+' required')
    i=argv.index(name)
    if i+1>=len(argv):raise ValueError('Missing original argument '+name)
    return argv[i+1]


def original_module():
    from tools.d44_native_rl import train
    from starVLA.rl.simwam_recipe import checkpoint as source
    if Path.cwd().resolve()!=SOURCE or Path(train.__file__).resolve()!=SOURCE/'tools/d44_native_rl/train.py' or Path(source.__file__).resolve()!=SOURCE/'starVLA/rl/simwam_recipe/checkpoint.py':
        raise ValueError('Run from the frozen original-science+wrapper source, never mixed ops modules')
    return train,source


def inspect_resume(argv,train):
    if '--qualification'in argv:raise ValueError('Administrative wrapper supports own formal world8 resume only')
    output=Path(argument(argv,'--output')).resolve();cp=Path(argument(argv,'--resume')).resolve();config=Path(argument(argv,'--config')).resolve()
    identity=read(cp/'identity.json');body=dict(identity);saved=body.pop('sha256',None);seal=read(cp/'COMPLETE');files=read(cp/'files.json')
    arm=identity.get('recipe',{}).get('arm')
    if (fingerprint(body)!=saved or arm not in ARMS or output!=ROOT/'runs'/('D46_'+arm)or not cp.is_relative_to(output/'checkpoints')
        or identity.get('schema')!='d44_native_rl_v1'or identity.get('world')!=8 or identity.get('qualification')is not False
        or identity.get('parent_model_sha256')!=MODEL_SHA or identity['recipe'].get('parent')!='D_IL46K'
        or identity['recipe'].get('parent_checkpoint')!=str(ROOT/'IL46_PARENT')
        or seal.get('world')!=8 or seal.get('contract')!='qwen_world_abcd_research_checkpoint_v1'
        or seal.get('files_sha256')!=file_sha(cp/'files.json')or files.get('identity.json')!=file_sha(cp/'identity.json')):
        raise ValueError('Not a sealed original own IL46 world8 resume')
    if yaml.safe_load(config.read_text())!=identity['recipe']or read(output/'identity.json')!=identity:raise ValueError('Original recipe/run identity changed')
    current=train.code_identity()
    if current!=identity['source_files']:raise ValueError('Every original mathematical/runtime source must remain byte-identical')
    return dict(identity=identity,checkpoint=str(cp),files_sha256=seal['files_sha256'],updates=seal['updates'],next_inner=seal['next_inner'],
        output=str(output),config=str(config),config_sha256=file_sha(config),source_files_sha256=fingerprint(current))


def prepare(argv,path):
    train,_=original_module();value=inspect_resume(argv,train);out=Path(path)
    if out.exists():raise FileExistsError('Keep execution profiles immutable')
    profile=dict(schema=SCHEMA,protocol=PROTOCOL,science_source=str(SOURCE),wrapper_file=WRAPPER,wrapper_sha256=file_sha(SOURCE/WRAPPER),
        identity_sha256=value['identity']['sha256'],parent_model_sha256=MODEL_SHA,arm=value['identity']['recipe']['arm'],
        output=value['output'],config=value['config'],config_sha256=value['config_sha256'],source_files_sha256=value['source_files_sha256'],
        anchor_checkpoint=value['checkpoint'],anchor_files_sha256=value['files_sha256'],anchor_updates=value['updates'],
        same_identity_native_resume=True,CPU_control_only=True)
    profile['sha256']=fingerprint(profile);out.parent.mkdir(parents=True,exist_ok=True);atomic_json(out,profile)
    return dict(profile=str(out.resolve()),file_sha256=file_sha(out),source_files=len(value['identity']['source_files']))


def validate(profile_path,expected_sha,argv,train):
    if file_sha(profile_path)!=expected_sha:raise ValueError('Pinned startup execution profile changed')
    p=read(profile_path);body=dict(p);seal=body.pop('sha256',None)
    if (p.get('schema')!=SCHEMA or fingerprint(body)!=seal or p.get('protocol')!=PROTOCOL or p.get('wrapper_file')!=WRAPPER
        or p.get('science_source')!=str(SOURCE)or p.get('wrapper_sha256')!=file_sha(SOURCE/WRAPPER)or p.get('parent_model_sha256')!=MODEL_SHA):raise ValueError('Unregistered startup execution wrapper/protocol')
    value=inspect_resume(argv,train)
    for key in ('output','config','config_sha256','source_files_sha256'):
        if value[key]!=p[key]:raise ValueError('Startup profile resume mismatch '+key)
    if value['identity']['sha256']!=p['identity_sha256']or value['updates']<p['anchor_updates']or file_sha(Path(p['anchor_checkpoint'])/'files.json')!=p['anchor_files_sha256']:
        raise ValueError('Startup profile/native ancestry changed')
    return p,value


def run(argv):
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--control-profile',required=True);parser.add_argument('--control-profile-sha256',required=True)
    extra,original=parser.parse_known_args(argv)
    train,source=original_module();profile,value=validate(extra.control_profile,extra.control_profile_sha256,original,train)
    if int(os.environ.get('WORLD_SIZE','8'))!=8:raise ValueError('World8 control wrapper only')
    rank=int(os.environ.get('RANK','0'))
    if not 0<=rank<8:raise ValueError('Invalid world8 rank')
    old=train.checkpoint.init_control_group
    if old is not source.init_control_group:raise ValueError('Another runtime patch is already active')
    def initialize():
        if not source.dist.is_initialized()or source.dist.get_world_size()!=8:raise ValueError('Requires the original initialized world8')
        if source.CONTROL_GROUP is not None:raise ValueError('Do not overwrite another control group')
        source.CONTROL_GROUP=source.dist.new_group(backend='gloo',timeout=timedelta(seconds=600))
    # Administrative alias only, restored even when the original CLI fails.
    saved_argv=sys.argv[:]
    train.checkpoint.init_control_group=initialize
    try:
        receipt=dict(schema='d46_world8_control_startup_rank_receipt_v1',rank=rank,profile=str(Path(extra.control_profile).resolve()),
            profile_file_sha256=extra.control_profile_sha256,protocol=PROTOCOL,wrapper_sha256=profile['wrapper_sha256'],
            checkpoint=value['checkpoint'],identity_sha256=value['identity']['sha256'],original_source_inventory_exact=True,
            code_identity_sha256=value['source_files_sha256'],original_argv=original,CPU_control_timeout_only=True,
            optimizer_reset=False,native_state_modified=False,created=time.time())
        launch_id=os.environ.get('TORCHELASTIC_RUN_ID')or('local_'+str(os.getpid())+'_'+str(time.time_ns()))
        if not all(c.isalnum()or c in('-','_')for c in launch_id):raise ValueError('Invalid elastic invocation identifier')
        receipt_path=Path(value['output'])/'executions'/('control600_'+extra.control_profile_sha256)/launch_id/f'rank{rank}.json'
        receipt_path.parent.mkdir(parents=True,exist_ok=True);atomic_json(receipt_path,receipt)
        sys.argv=[sys.argv[0]]+original;return train.main()
    finally:sys.argv=saved_argv;train.checkpoint.init_control_group=old


def main(argv=None):
    argv=sys.argv[1:]if argv is None else argv
    if argv and argv[0]=='prepare':
        p=argparse.ArgumentParser(__doc__);p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--resume',required=True);p.add_argument('--profile',required=True);a=p.parse_args(argv[1:])
        print(json.dumps(prepare(['--config',a.config,'--output',a.output,'--resume',a.resume],a.profile),indent=2));return
    return run(argv)


if __name__=='__main__':main()
