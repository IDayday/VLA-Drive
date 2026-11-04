"""Own-job receipt/locks and finite-host cooperative checkpoint guard.

No unrelated PID is signalled. Algorithms must poll boundary_stop_due() at
their native recoverable boundary. Long tasks are handed back to the permanent
host queue after a guarded exit; this worker never starts a second lineage.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import traceback
from starVLA.rl.qwen_world_compare.contracts import authorization,require_dispatch,file_sha

def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));os.replace(tmp,path)


def task_ranks(state):
    """Verified descendants, including torchrun's separate rank sessions."""
    expected=state['command'];output=expected[expected.index('--output')+1] if '--output' in expected else None
    rows=[];processes={}
    known={row['pid']:row for row in state.get('known_rank_processes',[])+state.get('guard_rank_processes',[])}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            stat=(proc/'stat').read_text().split();processes[int(proc.name)]=(int(stat[3]),int(stat[21]),stat[2])
        except (OSError,ValueError,IndexError):continue
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            pid=int(proc.name)
            if pid not in processes or processes[pid][2] in ('Z','X'):continue
            old=known.get(pid);previous=old is not None and verified_process(pid,old['start_ticks'],old['pgid'])
            ancestor=pid;visited=set();descendant=False
            while ancestor in processes and ancestor not in visited:
                visited.add(ancestor)
                if ancestor==state['child_pid']:
                    descendant=processes[ancestor][1]==state['child_start_ticks'];break
                ancestor=processes[ancestor][0]
            if not descendant and not previous:continue
            args=[v for v in (proc/'cmdline').read_bytes().decode().split('\0') if v]
            entry=args[args.index('-m')+1] if '-m' in args else None
            if entry not in ('tools.qwen_world_postcot.train_il','tools.qwen_world_postcot.train_rl',
                             'tools.qwen_world_postcot.train_scope_rl','tools.qwen_world_compare.train_il','tools.qwen_world_compare.train_rl','tools.d44_native_rl.train','tools.d46_world16.train','tools.d46_world16.train_fast','tools.d46_augustus.train','tools.d46_fpo_verified.train','tools.d46_fpo_engineering.train','tools.d46_priority_rl.train_control'):continue
            if output is None or '--output' not in args or args[args.index('--output')+1]!=output:continue
            status=dict(line.split(':',1) for line in (proc/'status').read_text().splitlines() if ':' in line)
            rows.append(dict(pid=pid,start_ticks=int((proc/'stat').read_text().split()[21]),
                pgid=os.getpgid(pid),entry=entry,
                usr1_handler=bool(int(status['SigCgt'].strip(),16)&(1<<(signal.SIGUSR1-1)))))
        except (OSError,ValueError,IndexError,ProcessLookupError):continue
    return rows


def verified_process(pid,start_ticks,pgid):
    try:
        stat=(Path('/proc')/str(pid)/'stat').read_text().split()
        return stat[2] not in ('Z','X') and int(stat[21])==start_ticks and os.getpgid(pid)==pgid
    except (OSError,ValueError,ProcessLookupError):return False


def last_complete_checkpoint(state):
    command=state['command']
    if '--output' not in command:return None
    root=Path(command[command.index('--output')+1]);candidates=[]
    for name in ('progress.json','LATEST.json','latest.json'):
        try:
            value=json.loads((root/name).read_text());checkpoint=Path(value['checkpoint'])
            seal=json.loads((checkpoint/'COMPLETE').read_text());files=json.loads((checkpoint/'files.json').read_text())
            if seal['files_sha256']!=file_sha(checkpoint/'files.json'):continue
            if any(not (checkpoint/n).resolve().is_relative_to(checkpoint.resolve()) or not (checkpoint/n).is_file() for n in files):continue
            candidates.append(dict(path=str(checkpoint.resolve()),updates=seal['updates'],receipt=seal,
                inventory_seal_verified=True,full_tensor_hashes_deferred_to_strict_resume=True))
        except (OSError,ValueError,KeyError,TypeError):continue
    return max(candidates,key=lambda value:value['updates']) if candidates else None


def deadline_tick(state,child,root,stop,*,now=None):
    """Cooperative native stop, then explicit owned-group absolute-expiry fallback.

    Forced expiry preserves the last atomic recovery point but is never called
    lossless/exact. Injected times are solely for standalone child unit tests;
    production timestamps always come from the immutable task authorization.
    """
    if not state['temporary']:return
    now=time.time() if now is None else now
    if now>=state['guard_unix'] and not state.get('expiry_guard_requested'):
        stop.touch();state.update(expiry_guard_requested=True,stop_requested_unix=now)
        state['guard_rank_processes']=task_ranks(state)
        signalled=[]
        for row in state['guard_rank_processes']:
            if row['usr1_handler'] and verified_process(row['pid'],row['start_ticks'],row['pgid']):
                os.kill(row['pid'],signal.SIGUSR1);signalled.append(row)
        state['native_usr1_signals']=signalled
        write(root/'RUNNING.json',state)
    for seconds,label,sig in ((90,'SIGTERM',signal.SIGTERM),(30,'SIGKILL',signal.SIGKILL)):
        key='expiry_'+label
        if now<state['deadline_unix']-seconds or state.get(key):continue
        leader=verified_process(state['child_pid'],state['child_start_ticks'],state['child_pid'])
        ranks=[row for row in state.get('known_rank_processes',[])+state.get('guard_rank_processes',[]) if verified_process(row['pid'],row['start_ticks'],row['pgid'])]
        if not leader and not ranks:
            if child.poll() is None:raise RuntimeError('Owned process identity changed; refusing an unsafe expiry signal')
            continue
        state[key]=dict(at_unix=now,pgid=state['child_pid'],leader_identity_verified=leader,
            verified_rank_processes=ranks,last_atomic_checkpoint=last_complete_checkpoint(state))
        state['forced_expiry']=True
        state['recovery_contract']='FORCED_EXIT_REVIEW_REQUIRED; latest atomic checkpoint retained; uncommitted work may be lost'
        write(root/'FORCED_EXPIRY.json',state)
        groups=({state['child_pid']} if leader else set())|{row['pgid']for row in ranks}
        for pgid in groups:
            try:os.killpg(pgid,sig)
            except ProcessLookupError:pass


class ResourceRefused(RuntimeError):
    """A positively identified pre-child capacity/owner/lease change only."""


PRIMARY_LEASE_ROOT=Path('/mnt/project/runtime_gpu_leases')
SHARED_LEASE_ROOT=Path('/mnt/project/runtime_gpu_shared_workers')
PROFILED_THIRD_LEASE_ROOT=Path('/mnt/project/runtime_gpu_profiled_third_workers')
D44_ROOT=Path('/mnt/project/d44-native-rl-20261103')
D44_PARENT_SHA='9715b9c0cfbef7a7d3d94fb83e7b0e2f0838d21cf06d3277deef6733cccbc50e'
D46_ROOT=Path('/mnt/project/d46-priority-rl-20261103')
D46_PARENT_SHA='386a0a27bd197cb002501bcb90c343659001d2e6d4f7bad0cbd5e31a91f8477d'
D46_CONTRACT_SHA='584dbcf1103fb4f052baa843add861a33bca90023f47827e2a882ef5a66a9632'


def known_D46_output(output):
    """One exact IL46 deployment lineage; never accepts a path prefix alone."""
    if not output.is_relative_to(D46_ROOT):return False
    try:
        value=json.loads((output/'identity.json').read_text());body=dict(value);saved=body.pop('sha256')
        from starVLA.rl.qwen_world_compare.contracts import fingerprint
        contract=D46_ROOT/'PARENT_REGISTRATION_v2.json'
        return (fingerprint(body)==saved and value['schema']=='d44_native_rl_v1'
            and value['architecture_id']=='D'and value['recipe']['parent']=='D_IL46K'
            and value['recipe']['arm']in('D0','D1','D2','D3_FPO','D3_AWM','D5')
            and value['recipe']['parent_checkpoint']==str(D46_ROOT/'IL46_PARENT')
            and value['parent_model_sha256']==D46_PARENT_SHA
            and value['recipe']['parent_registration']==dict(path=str(contract),file_sha256=D46_CONTRACT_SHA)
            and value['recipe']['training_uses_navtest']is False and value['data']['NAVTEST_overlap']==0
            and file_sha(contract)==D46_CONTRACT_SHA)
    except (OSError,KeyError,TypeError,ValueError):return False


def known_D46_world16_output(output):
    """Only a typed continuation of its own sealed formal D0/D5/FPO world8 run."""
    if not known_D46_output(output):return False
    try:
        from starVLA.rl.qwen_world_compare.contracts import fingerprint
        value=json.loads((output/'identity.json').read_text())
        tx=value['native_world_transition'];arm=value['recipe']['arm']
        receipt=json.loads((output/'WORLD_TRANSITION.json').read_text())
        body=dict(receipt);digest=body.pop('sha256')
        source=Path(tx['source_checkpoint']).resolve()
        old=json.loads((source/'identity.json').read_text())
        oldseal=json.loads((source/'COMPLETE').read_text())
        oldbody=dict(old);oldsha=oldbody.pop('sha256')
        return (value['world']==16 and value['fresh_Adam']is False and arm in('D0','D5','D3_FPO')
            and tx['schema']=='d46_exact_native_READY_world8_to16_v1'
            and tx['source_world']==8 and tx['target_world']==16 and tx['same_algorithm']is True
            and tx['old_owner_may_adopt']is False and tx['virtual_sampler_lanes']==8
            and (tx['global_scenes'],tx['group'],tx['inner_updates'])==(64,16,2)
            and tx['parent_policy_unchanged']is True
            and source.is_relative_to(D46_ROOT/'runs'/('D46_'+arm)/'checkpoints')
            and known_D46_output(source) and fingerprint(oldbody)==oldsha
            and old['world']==8 and old['qualification']is False
            and old['recipe']==value['recipe'] and old['scope']==value['scope']
            and oldseal['world']==8 and oldseal['phase']=='READY' and oldseal['next_inner']==0
            and oldseal['updates']==tx['origin_updates']
            and oldseal['files_sha256']==file_sha(source/'files.json')==tx['source_files_sha256']
            and tx['common_optimizer_sha256']==json.loads((source/'files.json').read_text())['optimizer.pt']
            and tx['source_run_identity_sha256']==oldsha
            and fingerprint(body)==digest
            and receipt['schema']==tx['schema']
            and receipt['source_identity_sha256']==oldsha
            and Path(receipt['source_checkpoint']).resolve()==source
            and receipt['source_files_sha256']==oldseal['files_sha256']
            and receipt['origin_updates']==tx['origin_updates']
            and receipt['target_identity_sha256']==value['sha256']
            and receipt['common_optimizer_sha256']==tx['common_optimizer_sha256']
            and receipt['native_weights_and_Adam_inherited']is True
            and receipt['qualification']==value['qualification']==tx['target_qualification']
            and receipt['source_qualification']==tx['source_qualification']==old['qualification'])
    except(OSError,KeyError,TypeError,ValueError):return False



def known_D46_augustus_output(output):
    if not known_D46_output(output):return False
    try:
        from starVLA.rl.qwen_world_compare.contracts import fingerprint
        value=json.loads((output/'identity.json').read_text());cfg=value['recipe'];aug=cfg['augustus']
        exp=cfg['experiment_id'];aligned=exp in('AUG3_FULL_ALIGNED','AUG4_LORA_ALIGNED')
        return(value.get('algorithm_schema')=='d46_augustus_native_v1'
            and value.get('experiment_id')==exp in('AUG1_FULL','AUG2_LORA','AUG3_FULL_ALIGNED','AUG4_LORA_ALIGNED')
            and value.get('augustus_protocol_sha256')==fingerprint(aug)
            and value.get('world')in(8,16,32)and cfg['global_scenes']in(64,128)and cfg['global_scenes']%value['world']==0
            and cfg['group']==8 and cfg['inner_updates']in(1,2)and cfg['training_uses_navtest']is False
            and aug['training_steps']==3 and aug['noise_scale']==[.05,.05,.02]
            and aug['coordinate_reduction']=='mean24'and aug['fixed_reference_beta']==(.1 if aligned else .001)
            and (not aligned or(cfg['inner_updates']==2 and aug['schema']=='d46_augustus_deployment_suffix_v2'
                and aug['initial_root_seed']==2101 and aug['deployment']['training_deployment_grid_mismatch']is False))
            and aug['scope']==('full_text_action'if exp in('AUG1_FULL','AUG3_FULL_ALIGNED')else'lora_text_action'))
    except(OSError,KeyError,TypeError,ValueError):return False


def known_D46_fast16_output(output):
    """Same own IL46 native ancestry, plus a sealed execution-only upgrade."""
    if not known_D46_world16_output(output):return False
    try:
        from starVLA.rl.qwen_world_compare.contracts import fingerprint
        value=json.loads((output/'identity.json').read_text());meta=value['execution_upgrade']
        receipt=json.loads((output/'EXECUTION_UPGRADE.json').read_text());body=dict(receipt);digest=body.pop('sha256')
        transition=json.loads((output/'WORLD_TRANSITION.json').read_text())
        flags=meta['flags'];keys={'fast_prefix','fuse_fm_mc','async_scoring','rotary_cache','no_grad_prefix_cache'}
        if (meta.get('schema')!='d46_world16_execution_upgrade_v1' or set(flags)!=keys
            or any(type(v)is not bool for v in flags.values())or flags['async_scoring']or flags['no_grad_prefix_cache']
            or fingerprint(body)!=digest or receipt.get('execution_upgrade')!=meta
            or receipt.get('target_identity_sha256')!=value['sha256']or transition.get('execution_upgrade_sha256')!=digest
            or meta['new_inventory_sha256']!=fingerprint(value['source_files'])):return False
        cp=Path(meta['source_world16_checkpoint']).resolve();oldroot=cp.parent.parent
        if not cp.is_relative_to(D46_ROOT)or not known_D46_world16_output(oldroot):return False
        old=json.loads((cp/'identity.json').read_text());seal=json.loads((cp/'COMPLETE').read_text());files=json.loads((cp/'files.json').read_text())
        oldbody=dict(old);oldsha=oldbody.pop('sha256')
        if (fingerprint(oldbody)!=oldsha or oldsha!=meta['source_identity_sha256']
            or file_sha(cp/'identity.json')!=meta['source_checkpoint_identity_file_sha256']
            or files.get('identity.json')!=meta['source_checkpoint_identity_file_sha256']
            or file_sha(cp/'files.json')!=seal['files_sha256']or seal['files_sha256']!=meta['source_checkpoint_files_sha256']
            or files.get('optimizer.pt')!=meta['common_optimizer_sha256']or seal['world']!=16 or old['world']!=16
            or seal['phase']!='READY'or seal['next_inner']!=0 or seal['updates']!=meta['origin_updates']
            or meta['origin_updates']%value['recipe']['inner_updates']or old['qualification']is not meta['source_qualification']
            or value['qualification']is not meta['target_qualification']or(meta['source_qualification']and not meta['target_qualification'])
            or fingerprint(old['source_files'])!=meta['old_inventory_sha256']):return False
        src=json.loads((oldroot/'WORLD_TRANSITION.json').read_text());srcbody=dict(src);srcsha=srcbody.pop('sha256')
        if fingerprint(srcbody)!=srcsha or srcsha!=meta['source_world_transition_sha256']:return False
        import copy
        expected=copy.deepcopy(oldbody);expected.update(source_files=value['source_files'],qualification=value['qualification'],fresh_Adam=False,execution_upgrade=meta)
        expected['native_world_transition']['target_qualification']=value['qualification']
        srcbody.update(target_identity_sha256=value['sha256'],source_inventory=value['source_files'],qualification=value['qualification'],execution_upgrade_sha256=digest)
        return fingerprint(expected)==value['sha256']and fingerprint(srcbody)==transition['sha256']
    except(OSError,KeyError,TypeError,ValueError):return False


def known_D46_control_args(args):
    """Only fixed-profile600s wrapper around the same sealed formal8 policy."""
    try:
        output=Path(args[args.index('--output')+1]).resolve();profile=Path(args[args.index('--control-profile')+1]);sha=args[args.index('--control-profile-sha256')+1]
        resume=Path(args[args.index('--resume')+1]).resolve();cfg=Path(args[args.index('--config')+1])
        if file_sha(profile)!=sha or not known_D46_output(output)or not resume.is_relative_to(output/'checkpoints'):return False
        value=json.loads(profile.read_text());body=dict(value);digest=body.pop('sha256')
        identity=json.loads((output/'identity.json').read_text());native=json.loads((resume/'identity.json').read_text());seal=json.loads((resume/'COMPLETE').read_text())
        from starVLA.rl.qwen_world_compare.contracts import fingerprint
        protocol=dict(schema='d46_world8_CPU_control_timeout600_v1',world=8,backend='gloo',timeout_seconds=600,original_timeout_seconds=90,changes='CPU startup/control wait only',policy_optimizer_RNG_identity_unchanged=True)
        source=Path(value['science_source'])
        return(fingerprint(body)==digest and value['schema']=='d46_world8_control_startup_execution_profile_v1'and value['protocol']==protocol
            and value['parent_model_sha256']==D46_PARENT_SHA and value['output']==str(output)and value['identity_sha256']==identity['sha256']==native['sha256']
            and identity['world']==seal['world']==8 and identity['qualification']is False and value['arm']==identity['recipe']['arm']
            and value['config']==str(cfg.resolve())and value['config_sha256']==file_sha(cfg)
            and seal['files_sha256']==file_sha(resume/'files.json')and seal['updates']>=value['anchor_updates']
            and value['source_files_sha256']==fingerprint(identity['source_files'])
            and value['wrapper_file']=='tools/d46_priority_rl/train_control.py'and value['wrapper_sha256']==file_sha(source/value['wrapper_file']))
    except(OSError,KeyError,TypeError,ValueError,IndexError):return False

def known_owner_args(args):
    entry=args[args.index('-m')+1]if '-m'in args and args.index('-m')+1<len(args)else None
    if entry=='tools.d46_fpo_engineering.train':
        try:
            output=Path(args[args.index('--output')+1]).resolve()
            identity=json.loads((output/'identity.json').read_text())
            meta=identity.get('performance_execution',{})
            from starVLA.rl.qwen_world_compare.contracts import fingerprint
            body=dict(identity);sha=body.pop('sha256')
            return (output.is_relative_to(D46_ROOT/'runs')and known_D46_output(output)
                and fingerprint(body)==sha and meta.get('schema')=='d46_fpo_performance_execution_v1'
                and meta.get('profile')in('exact','fast')and identity['recipe']['arm']=='D3_FPO'
                and identity['scope']['trainable_numel']==6422528 and identity.get('qualification')is False)
        except(OSError,ValueError,IndexError,KeyError,TypeError):return False
    if entry=='tools.d46_fpo_verified.train':
        try:
            output=Path(args[args.index('--output')+1]).resolve()
            identity=json.loads((output/'identity.json').read_text())
            return (output==D46_ROOT/'runs/D46_FPO_VERIFIED_B1_v1'and known_D46_output(output)
                and identity.get('algorithm_schema')=='d46_verified_FPO_improvement_v1'
                and identity['recipe']['arm']=='D3_FPO'and identity['scope']['trainable_numel']==6422528
                and identity['recipe']['verified_improvement']['extra_G_as_PG']is False)
        except(OSError,ValueError,IndexError,KeyError,TypeError):return False
    if entry in('tools.d46_augustus.train','tools.d46_augustus.evaluate'):
        try:
            key='--output'if entry.endswith('.train')else'--checkpoint'
            return known_D46_augustus_output(Path(args[args.index(key)+1]).resolve())
        except(ValueError,IndexError):return False
    if entry=='tools.d46_priority_rl.train_control':return known_D46_control_args(args)
    if entry=='tools.d46_world16.train_fast':
        try:return known_D46_fast16_output(Path(args[args.index('--output')+1]).resolve())
        except(ValueError,IndexError):return False
    if entry=='tools.d46_world16.train':
        try:return known_D46_world16_output(Path(args[args.index('--output')+1]).resolve())
        except(ValueError,IndexError):return False
    if entry=='tools.d44_native_rl.evaluate':
        # A frozen D46 B1 evaluator may share its measured spare capacity.
        # Bind the exact loaded checkpoint, not an arbitrary output prefix.
        try:
            checkpoint=Path(args[args.index('--checkpoint')+1]).resolve()
            output=Path(args[args.index('--output')+1]).resolve()
            seal=json.loads((checkpoint/'COMPLETE').read_text())
            return ((output.is_relative_to(D46_ROOT/'evaluations')or output.is_relative_to(Path('/mnt/project/d46-fpo-two-routes-20261104/evaluations')))
                and known_D46_output(checkpoint)
                and seal['files_sha256']==file_sha(checkpoint/'files.json'))
        except(OSError,ValueError,IndexError,KeyError,TypeError):return False
    if entry=='tools.d44_native_rl.train':
        # Native D44 has no --il-checkpoint flag: bind the actual process
        # output to its already sealed scientific identity instead.
        if '--output'not in args or args.index('--output')+1>=len(args):return False
        output=Path(args[args.index('--output')+1]).resolve();base=D44_ROOT
        if output.is_relative_to(D46_ROOT):return known_D46_output(output)
        if not output.is_relative_to(base):return False
        try:
            value=json.loads((output/'identity.json').read_text())
            from starVLA.rl.qwen_world_compare.contracts import fingerprint
            body=dict(value);saved=body.pop('sha256')
            return (fingerprint(body)==saved and value['schema']=='d44_native_rl_v1'and value['architecture_id']=='D'
                and value['recipe']['parent_checkpoint']==str(base/'IL44_PARENT')
                and value['recipe']['training_uses_navtest']is False and value['data']['NAVTEST_overlap']==0
                and value['parent_model_sha256']==D44_PARENT_SHA)
        except (OSError,KeyError,ValueError,TypeError):return False
    legacy=('tools.qwen_world_postcot.train_il','tools.qwen_world_postcot.train_rl','tools.qwen_world_postcot.train_scope_rl',
            'tools.qwen_world_postcot.evaluate','tools.qwen_world_compare.train_il','tools.qwen_world_compare.train_rl','tools.qwen_world_compare.evaluate')
    if entry in legacy and any(arg.startswith(('/mnt/project/qwen-sourceflow-','/mnt/project/qwen-world-postcot-'))for arg in args):
        return True
    # The already running, separately authorized Safety F/G jobs were added
    # after this worker's legacy allowlist. Recognize their actual module AND
    # fixed IL ancestry; a new arbitrary path prefix alone is insufficient.
    modules=('tools.qwen_world_safety_exploration.train','tools.qwen_world_safety_exploration.evaluate','tools.qwen_world_performance.train',
        'tools.qwen_world_multiscene.train','tools.qwen_world_verified_quality.train')
    entry=args[args.index('-m')+1]if '-m'in args and args.index('-m')+1<len(args)else None
    if entry not in modules:return False
    parent_flag='--il-checkpoint'if entry.endswith('.train')else'--parent-checkpoint'
    if parent_flag not in args or args.index(parent_flag)+1>=len(args):return False
    if args[args.index(parent_flag)+1]!='/mnt/project/qwen-world-rl-optimization-20261030/IL_PARENT':return False
    return any(arg.startswith(('/mnt/project/qwen-world-safety-exploration-20261031/','/mnt/project/qwen-world-performance-rl-20261103/'))for arg in args)


def validate_profiled_d0_resume(command,minimum):
    """Opt-in third context only for the actually measured 25.7GiB D0 scope."""
    if ((minimum or 0)<40960 or '-m'not in command or command[command.index('-m')+1]!='torch.distributed.run'
            or 'tools.d44_native_rl.train'not in command or '--resume'not in command or '--output'not in command):
        raise ValueError('Profiled third slot requires native D0 resume and at least40GiB measured free memory')
    output=Path(command[command.index('--output')+1]);identity=json.loads((output/'identity.json').read_text());c=identity['recipe']
    required=dict(arm='D0',global_scenes=64,group=16,text_rank=16,film_last_layers=0,train_action_io=False,post_feedback=False,checkpoint_policy='all_off')
    if not known_owner_args(['python','-m','tools.d44_native_rl.train','--output',str(output)])or any(c.get(k)!=v for k,v in required.items()):
        raise ValueError('Only the actually profiled D0 native scope may use a third slot')


def validate_profiled_native_qualification(command,minimum):
    """Bounded actual D scopes beside frozen B1 evaluation, >=48GiB free."""
    if (minimum or 0)<49152:raise ValueError('Native qualification sharing requires48GiB free')
    if 'tools.d44_native_rl.validate_deployment'in command:
        cp=Path(command[command.index('--checkpoint')+1]);identity=json.loads((cp.parent.parent/'identity.json').read_text())
        if not cp.resolve().is_relative_to(D44_ROOT/'qualification')or not(cp/'COMPLETE').exists():raise ValueError('Unsealed qualification deployment checkpoint')
        from starVLA.rl.qwen_world_compare.contracts import fingerprint
        body=dict(identity);saved=body.pop('sha256')
        if fingerprint(body)!=saved or not identity['qualification']or identity['parent_model_sha256']!=D44_PARENT_SHA or identity['recipe']['arm']not in('D4','D5'):
            raise ValueError('Wrong native qualification asset')
        return
    if ('tools.d44_native_rl.train'not in command or '--qualification'not in command or '--config'not in command
            or '--stop-after-update'not in command or '--scenes-per-rank'in command
            or '--nproc_per_node=2'not in command or not 1<=int(command[command.index('--stop-after-update')+1])<=16):
        raise ValueError('Only bounded two-rank native qualification is profiled')
    output=Path(command[command.index('--output')+1]).resolve()
    if not output.is_relative_to(D44_ROOT/'qualification'):raise ValueError('Not a qualification output')
    # Use the already installed model environment's YAML parser, no GPU/import
    # of a model and no additional dependency in the stdlib operations worker.
    code='import json,yaml,sys;print(json.dumps(yaml.safe_load(open(sys.argv[1]))))'
    c=json.loads(subprocess.check_output([command[0],'-c',code,command[command.index('--config')+1]],text=True))
    expected=dict(global_scenes=32,group=16,text_rank=16,film_last_layers=0,train_action_io=False,checkpoint_policy='all_off',inner_updates=2)
    if c['arm']not in('D1','D3_FPO','D3_AWM','D4')or c['parent_checkpoint']!=str(D44_ROOT/'IL44_PARENT')or any(c.get(k)!=v for k,v in expected.items()):
        raise ValueError('Unprofiled native scope cannot enter third slot')


def frozen_evaluation_owner(args):
    if '-m'not in args or args.index('-m')+1>=len(args):return False
    entry=args[args.index('-m')+1]
    if entry not in('tools.qwen_world_compare.evaluate','tools.qwen_world_safety_exploration.evaluate')or not known_owner_args(args)or '--export'not in args:return False
    try:
        export=Path(args[args.index('--export')+1]).resolve()
        if entry=='tools.qwen_world_compare.evaluate':
            if not export.is_relative_to('/mnt/project/qwen-world-four-frameworks-20261029'):return False
        elif export!=Path('/mnt/project/qwen-world-performance-rl-20261103/diagnostics/IL70_export'):return False
        seal=json.loads((export/'COMPLETE').read_text());meta=json.loads((export/'export.json').read_text())
        return seal['export_sha256']==file_sha(export/'export.json')and meta['qualification']is False and(export/'model.pt').is_file()
    except(OSError,ValueError,KeyError,IndexError):return False


def _profiled_slot(args):
    return getattr(args,'profiled_d0_third_slot',False)or getattr(args,'profiled_native_qualification_third_slot',False)


def _known_owners(rows,mapping,indices):
    for row in rows:
        if row['uuid']not in {mapping[i]for i in indices}:continue
        try:args=(Path('/proc')/str(row['pid'])/'cmdline').read_bytes().decode().split('\0')
        except (FileNotFoundError,ProcessLookupError,PermissionError)as error:
            raise ResourceRefused('Owner visibility changed before child startup')from error
        if not known_owner_args(args):
            raise ResourceRefused('Unknown GPU owner must not be shared or evicted')


def _admission_sample():
    devices={}
    for line in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.free,utilization.gpu',
            '--format=csv,noheader,nounits'],text=True).splitlines():
        index,uuid,total,free,util=line.split(',')
        devices[int(index)]=dict(index=int(index),uuid=uuid.strip(),total_MiB=int(total),free_MiB=int(free),utilization=int(util))
    owners=[]
    for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory',
            '--format=csv,noheader,nounits'],text=True).splitlines():
        pid,uuid,memory=line.split(',')
        owners.append(dict(pid=int(pid),uuid=uuid.strip(),used_MiB=int(memory)if memory.strip().isdigit()else None))
    return devices,owners


def shared_training_host(host,grant):
    """Sharing is explicit; a bounded grant never becomes an unbounded one."""
    if host in grant['permanent_hosts']:return True
    explicit=grant.get('shared_training_hosts',[])
    bounded=(host in grant['temporary_hosts'] and grant['starts_unix']<=time.time()<grant['expires_unix']-grant['checkpoint_guard_seconds'])
    if host not in explicit or not (bounded or host in grant.get('unbounded_temporary_hosts',[])):return False
    original=json.loads(Path(grant['record']).read_text())
    if original.get('shared_training_hosts',[])!=explicit:
        raise PermissionError('Shared training host list differs from immutable task grant')
    return True


def _gpu_admission(args,host,grant,indices,expected,mapping,locks):
    if not indices:return
    from starVLA.rl.qwen_world_postcot.resources import gpu_activity_issue
    def verify(devices,owners,*,locked):
        physical=[devices[i]['uuid']for i in indices]
        if expected is not None and physical!=expected:
            raise ResourceRefused('Physical GPU mapping changed after scheduling')
        if mapping and physical!=[mapping[i]for i in indices]:
            raise ResourceRefused('Physical GPU mapping changed under admission lock')
        mapping.update({i:devices[i]['uuid']for i in indices})
        if args.share_min_free_mib:
            if not shared_training_host(host,grant)or args.share_min_free_mib<8192:
                raise ValueError('Sharing requires an explicitly authorized host and measured valid headroom')
            _known_owners(owners,mapping,indices)
            maximum=2 if _profiled_slot(args)else 1
            if getattr(args,'profiled_native_qualification_third_slot',False):
                for row in owners:
                    if row['uuid']not in physical:continue
                    argv=[a for a in(Path('/proc')/str(row['pid'])/'cmdline').read_bytes().decode().split('\0')if a]
                    if not frozen_evaluation_owner(argv):raise ResourceRefused('Profiled qualification sharing requires only known frozen B1 evaluation cohosts')
            if any(sum(r['uuid']==mapping[i]for r in owners)>maximum for i in indices):
                raise ResourceRefused('Two existing CUDA owners leave no qualified third slot')
        elif any(row['uuid']in physical for row in owners):
            raise ResourceRefused('GPU acquired another owner '+('under admission lock'if locked else'before dispatch'))
        for index in indices:
            issue=gpu_activity_issue(devices[index],owners)
            if issue:raise ResourceRefused(issue+(' under admission lock'if locked else' before dispatch'))
        required=args.share_min_free_mib or args.minimum_free_mib
        if required and any(devices[i]['free_MiB']<required for i in indices):
            raise ResourceRefused('Measured GPU headroom changed '+('under admission lock'if locked else'before dispatch'))
    devices,owners=_admission_sample();verify(devices,owners,locked=False)
    PRIMARY_LEASE_ROOT.mkdir(exist_ok=True);SHARED_LEASE_ROOT.mkdir(exist_ok=True)
    if _profiled_slot(args):PROFILED_THIRD_LEASE_ROOT.mkdir(exist_ok=True)
    # Slot ownership and whether this process shares CUDA capacity are distinct.
    # A previous learner can still hold the shared slot after its original
    # cohost exits. Reuse the vacant primary slot with the SAME owner/headroom
    # checks; never steal the shared lock or silently start a third slot.
    slot=getattr(args,'lease_slot',None) or ('shared' if args.share_min_free_mib else 'primary')
    if slot=='shared' and not args.share_min_free_mib:
        raise ValueError('The shared slot requires explicit measured sharing admission')
    choices=[slot]+(['primary']if slot=='shared'and args.share_min_free_mib else[])
    if _profiled_slot(args):choices+=['profiled_third']
    for candidate in choices:
        if candidate=='profiled_third'and any(sum(r['uuid']==mapping[i]for r in owners)!=2 for i in indices):continue
        acquired=[]
        try:
            for index in indices:
                lease=({'shared':SHARED_LEASE_ROOT,'primary':PRIMARY_LEASE_ROOT,'profiled_third':PROFILED_THIRD_LEASE_ROOT}[candidate]/(mapping[index]+'.lock')).open('a')
                acquired.append(lease);fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            for stream in acquired:stream.close()
            continue
        locks.extend(acquired);args.lease_slot=candidate;break
    else:raise ResourceRefused('Both qualified admission slots are busy before child startup')
    # A second actual total/free/utilization+per-process-memory sample under
    # the lock closes the invisible-container race before Popen or CUDA init.
    devices,owners=_admission_sample();verify(devices,owners,locked=True)


def admission_with_wait(args,host,indices,expected,mapping,locks,state,root):
    """Bounded pre-child wait; never retain partial leases or run CUDA."""
    deadline=time.monotonic()+args.admission_wait_seconds
    while True:
        try:require_dispatch(admit_long_job=True)
        except PermissionError as error:raise ResourceRefused('Host authorization expired or was withdrawn before child startup')from error
        grant,_=authorization()
        try:
            _gpu_admission(args,host,grant,indices,expected,mapping,locks)
            return grant
        except ResourceRefused as error:
            for stream in locks:stream.close()
            locks.clear();mapping.clear()
            if time.monotonic()>=deadline:raise
            state.update(status='WAITING_RESOURCE',reason=str(error),child_not_started=True,last_check_unix=time.time())
            write(root/'ADMISSION_WAIT.json',state)
            time.sleep(min(15,max(0,deadline-time.monotonic())))


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--receipt-root',required=True);parser.add_argument('--gpus',default='')
    parser.add_argument('--expected-gpu-uuids-json');parser.add_argument('--command-json',required=True);parser.add_argument('--cwd',required=True)
    parser.add_argument('--share-min-free-mib',type=int);parser.add_argument('--minimum-free-mib',type=int)
    parser.add_argument('--lease-slot',choices=('primary','shared'))
    parser.add_argument('--admission-wait-seconds',type=int,default=0)
    parser.add_argument('--profiled-d0-third-slot',action='store_true')
    parser.add_argument('--profiled-native-qualification-third-slot',action='store_true')
    args=parser.parse_args();root=Path(args.receipt_root);root.mkdir(parents=True,exist_ok=True)
    command=json.loads(args.command_json)
    if not isinstance(command,list)or not command or not all(isinstance(value,str)for value in command):raise ValueError('Exact argument vector required')
    indices=[]if not args.gpus else[int(value)for value in args.gpus.split(',')]
    if len(indices)!=len(set(indices)):raise ValueError('GPU indices must be unique')
    if args.minimum_free_mib is not None and args.minimum_free_mib<1:raise ValueError('Minimum headroom must be positive')
    if not 0<=args.admission_wait_seconds<=1800:raise ValueError('Pre-child resource wait must be finite')
    expected=None if args.expected_gpu_uuids_json is None else json.loads(args.expected_gpu_uuids_json)
    if expected is not None and(not isinstance(expected,list)or len(expected)!=len(indices)or not all(isinstance(v,str)for v in expected)):
        raise ValueError('Expected physical GPU identities must match requested indices')
    lock=(root/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if(root/'RESULT.json').exists():raise ValueError('Invocation already finalized; do not duplicate it')
    host=socket.gethostname().removesuffix('-worker-0');mapping={};locks=[]
    state=dict(status='ADMISSION',host=host,worker_pid=os.getpid(),worker_start_ticks=int(Path('/proc/self/stat').read_text().split()[21]),
        start_unix=time.time(),gpus=indices,requested_gpu_uuids=expected,
        command=command,cwd=str(Path(args.cwd).resolve()),shared_headroom_MiB=args.share_min_free_mib)
    state.update(lease_slot=args.lease_slot or ('shared' if args.share_min_free_mib else 'primary'),
        execution_worker=str(Path(__file__).resolve()),execution_worker_sha256=file_sha(__file__))
    try:
        if args.profiled_d0_third_slot:validate_profiled_d0_resume(command,args.share_min_free_mib)
        if args.profiled_native_qualification_third_slot:validate_profiled_native_qualification(command,args.share_min_free_mib)
        grant=admission_with_wait(args,host,indices,expected,mapping,locks,state,root)
    except ResourceRefused as error:
        state.update(status='REFUSED_RESOURCE',child_not_started=True,returncode=75,reason=str(error),end_unix=time.time(),
            gpu_uuids=[mapping[i]for i in indices]if all(i in mapping for i in indices)else[])
        if state['requested_gpu_uuids']is None:state['requested_gpu_uuids']=state['gpu_uuids']
        write(root/'REFUSED_RESOURCE.json',state);write(root/'RESULT.json',state)
        return 75
    except Exception:
        state.update(status='FAILED_BEFORE_CHILD',child_not_started=True,returncode=2,error=traceback.format_exc(),end_unix=time.time())
        write(root/'RESULT.json',state);return 2
    stop=root/'STOP_REQUESTED'
    env=dict(os.environ,WORLD_POSTCOT_STOP_FILE=str(stop),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',
        PYTHONPATH=str(Path(args.cwd).resolve())+(os.pathsep+os.environ['PYTHONPATH']if os.environ.get('PYTHONPATH')else''))
    if indices:env['CUDA_VISIBLE_DEVICES']=','.join(map(str,indices))
    state.update(status='RUNNING',lease_slot=args.lease_slot or state['lease_slot'],gpu_uuids=[mapping[i]for i in indices],temporary=host not in grant['permanent_hosts'] and host not in grant.get('unbounded_temporary_hosts', []),
        guard_unix=grant['expires_unix']-grant['checkpoint_guard_seconds'],deadline_unix=grant['expires_unix'])
    def request(*_):
        stop.touch();state['stop_requested_unix']=time.time();write(root/'RUNNING.json',state)
    signal.signal(signal.SIGTERM,request);signal.signal(signal.SIGINT,request)
    with(root/'stdout.log').open('ab',buffering=0)as log:
        try:child=subprocess.Popen(command,cwd=args.cwd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        except Exception:
            state.update(status='FAILED_BEFORE_CHILD',child_not_started=True,returncode=2,error=traceback.format_exc(),end_unix=time.time())
            write(root/'RESULT.json',state);return 2
        state['child_pid']=child.pid;state['child_start_ticks']=int(Path(f'/proc/{child.pid}/stat').read_text().split()[21]);write(root/'RUNNING.json',state)
        while child.poll()is None or task_ranks(state):
            state['known_rank_processes']=task_ranks(state);deadline_tick(state,child,root,stop);time.sleep(5)
    state.update(status='FORCED_EXIT_REVIEW_REQUIRED' if state.get('forced_expiry') else 'EXITED' if child.returncode==0 else 'FAILED',
        returncode=child.returncode,end_unix=time.time(),migration_required=bool(stop.exists()and state['temporary']))
    if state.get('forced_expiry'):
        state['last_atomic_checkpoint']=last_complete_checkpoint(state);write(root/'FORCED_EXPIRY.json',state)
    write(root/'RESULT.json',state)
    return child.returncode


if __name__=='__main__':raise SystemExit(main())
