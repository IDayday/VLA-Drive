"""One owner for four complete IL100k/RL50k lines and fixed full evaluations.

Scheduling only: native training, strict exports, canonical scoring, checkpoint
inspection and DEV selection are delegated to the existing new-family tools.
No qualification PASS, policy update or score is manufactured here.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback
import uuid

if __package__ in (None, ''):
    sys.path[:] = [x for x in sys.path if Path(x or os.curdir).resolve() != Path(__file__).resolve().parent]
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import yaml

from starVLA.rl.qwen_world_compare.contracts import ROOT, allowed_hosts, require_dispatch, fingerprint, file_sha
from starVLA.rl.qwen_world_postcot.checkpoint import atomic_json
from starVLA.rl.qwen_world_compare.parent_selection import IL_MILESTONES, RL_MILESTONES
from starVLA.rl.qwen_world_compare.loading import inspect_IL_checkpoint, inspect_RL_checkpoint

ARCHITECTURES = tuple('ABCD')
MODEL_PYTHON = '/mnt/project/flow-safe-improvement-20261013/venv/bin/python'
SCHEMA = 'qwen_world_compare_campaign_v1'
MAX_PRECHILD_ATTEMPTS = 3
STOP = False


def expected_source_files(source):
    """Union of the IL/RL scientific dependency declarations, without importing models."""
    source = Path(source)
    folders = ('starVLA/model/modules/qwen_world_compare', 'starVLA/rl/qwen_world_compare',
               'tools/qwen_world_compare', 'starVLA/model/modules/qwen_world_postcot',
               'starVLA/rl/qwen_world_postcot',
               'starVLA/model/modules/qwen_sourceflow', 'starVLA/rl/qwen_sourceflow',
               'starVLA/rl/simwam_recipe')
    paths = {p for folder in folders for p in (source/folder).rglob('*.py')}
    paths.update(source/name for name in (
        'tools/qwen_world_postcot/train_il.py', 'starVLA/rl/qwen_world_postcot/checkpoint.py',
        'starVLA/rl/qwen_world_postcot/loading.py', 'starVLA/rl/qwen_world_postcot/data.py',
        'scripts/analysis/simwam_recipe/train.py', 'starVLA/rl/flow_grpo/transactions.py',
        'starVLA/rl/flow_grpo/reproducibility.py', 'starVLA/rl/safe_flow_improvement/v1_reward.py',
        'scripts/analysis/constrained_flow_noise/batched_canonical_score.py'))
    if not paths or any(not p.is_file() for p in paths):
        raise ValueError('The immutable source lacks a declared scientific dependency')
    return {str(path.relative_to(source)):file_sha(path) for path in sorted(paths)}


def _read(path):
    return json.loads(Path(path).read_text())


def make_plan(config, source, *, execution_migration=None):
    from starVLA.model.modules.qwen_world_compare.factory import load_config
    source = Path(source).resolve();cfg = yaml.safe_load(Path(config).read_text())
    if (cfg.get('schema') != 'qwen_world_four_frameworks_v1'
            or cfg.get('no_short_training_elimination') is not True
            or cfg.get('training_seed') != 42 or cfg.get('evaluation_seed') != 2101
            or set(cfg['architectures']) != set(ARCHITECTURES)):
        raise ValueError('The campaign must retain all four complete fixed-seed experiments')
    root = Path(cfg['root']).resolve();arms = {};common = None
    for arch in ARCHITECTURES:
        rel = cfg['architectures'][arch];path = source/rel;resolved = load_config(path)
        if resolved['architecture_id'] != arch or Path(resolved['root']).resolve() != root:
            raise ValueError('Architecture recipe/root identity mismatch')
        shared = {k:resolved[k] for k in ('seed','evaluation_seed','qwen_path','qwen_identity','data_root',
            'teacher_cache','candidate_bank','evidence_root','target_root','candidate_refresh','il','deployment','precision')}
        if common is not None and shared != common:
            raise ValueError('Architectures changed the common data/IL/reward protocol')
        common = shared
        base = root/'runs'/f'ARCH_{arch}'
        qualification_root=Path(cfg.get('qualification_root',root/'qualification'))
        arms[arch] = dict(config=str(path), config_relative=rel, config_sha256=file_sha(path),
            IL=str(base/'IL_seed42'), RL=str(base/'RL_seed42'),
            selection_IL=str(root/'selections'/f'{arch}_IL_parent.json'),
            selection_RL=str(root/'selections'/f'{arch}_RL_policy.json'),
            qualification=str(qualification_root/arch/'ENGINEERING_ACCEPTANCE.json'))
    rl_path = source/'configs/qwen_world_compare/rl.yaml';rl = yaml.safe_load(rl_path.read_text())
    if (rl['max_steps'] != 50000 or rl['grpo']['sample']['group_size'] != 8
            or rl['grpo']['train']['num_inner_epochs'] != 4):
        raise ValueError('The common RL budget/group/inner reuse changed')
    value = dict(schema=SCHEMA, root=str(root), source=str(source),
        controller_source=str(Path(__file__).resolve().parents[2]),
        execution_migration=str(Path(execution_migration).resolve()) if execution_migration else None,
        execution_migration_sha256=file_sha(execution_migration) if execution_migration else None,
        architectures=arms, common=common,
        data_paths=dict(TRAIN=str(Path(common['data_root'])/'TRAIN'),DEV=str(Path(common['data_root'])/'DEV'),
                        NAVTEST=str(Path(cfg.get('navtest_data_root',Path(common['data_root'])/'NAVTEST')).resolve())),
        rl_config=str(rl_path), rl_config_sha256=file_sha(rl_path), stages=['il','rl','evaluate'],
        IL_updates=100000, RL_updates=50000, IL_DEV_updates=list(IL_MILESTONES),
        RL_DEV_updates=list(RL_MILESTONES), inference='single_candidate_cleanx_K10_shift5',
        train_seed=42, evaluation_seed=2101, DEV_count=1696, NAVTEST_count=12146,
        logical_RL_fresh_rollouts_per_arm=12500, logical_RL_scene_exposures_per_arm=100000,
        logical_RL_candidates_per_arm=800000, no_short_training_elimination=True,
        logical_IL_scene_exposures_per_arm=3200000,
        minimum_disk_free_GiB=cfg.get('minimum_disk_free_GiB',750),
        maximum_evaluation_workers=cfg.get('maximum_evaluation_workers',8))
    value['sha256'] = fingerprint(value)
    return value


def qualification_gate(arch, plan, source_files):
    p=Path(plan['architectures'][arch]['qualification'])
    if not p.exists():
        return False,'WAITING_QUALIFICATION','Engineering acceptance is not yet published'
    receipt=_read(p)
    if (receipt.get('status')!='PASS' or receipt.get('architecture_id')!=arch
            or receipt.get('formal_initialized_from_qualification') is not False):
        return False,'WAITING_QUALIFICATION','Missing real acceptance or qualification-weight isolation'
    if receipt.get('source_files')!=source_files or receipt.get('config_sha256')!=plan['architectures'][arch]['config_sha256']:
        return False,'BLOCKED_SOURCE_MISMATCH','Engineering acceptance does not bind this exact source/config'
    return True,'QUALIFIED',None


def preparation(plan):
    """Read sealed metadata, never rebuild a running target/evidence producer."""
    common=plan['common'];root=Path(plan['root']);missing=[]
    required = [Path(common['evidence_root'])/'COMPLETE.json',Path(common['target_root'])/'COMPLETE.json',
        Path(common['teacher_cache'])/'IDENTITY.json',Path(common['candidate_bank'])/'COMPLETE.json',
        Path(common['qwen_identity'])]
    for path in required:
        if not path.is_file():missing.append(str(path))
    paths={role:Path(path)for role,path in plan['data_paths'].items()}
    for role,count in [('TRAIN',90890),('DEV',1696),('NAVTEST',12146)]:
        manifest=paths[role]/'manifest.json';complete=paths[role]/'COMPLETE'
        if not manifest.exists()or not complete.exists():missing.extend([str(manifest),str(complete)]);continue
        value=_read(manifest);seal=_read(complete)
        if value.get('cohort')!=role or len(value['tokens'])!=count or seal['manifest_sha256']!=file_sha(manifest):
            raise ValueError('Common fixed data population/seal changed: '+role)
    if missing:return dict(status='WAITING_ASSETS',missing=missing)
    from starVLA.rl.qwen_world_compare.evidence import read_manifest
    evidence=read_manifest(common['evidence_root']);candidate=_read(Path(common['candidate_bank'])/'MANIFEST.json')
    if (evidence['counts']!={'scenes':90890,'records':90890*16}
            or evidence['identity']['source_bank_hash']!=candidate['sha256']):
        raise ValueError('Full shared fixed candidate/evidence coverage is incomplete')
    train=_read(paths['TRAIN']/'manifest.json');dev=_read(paths['DEV']/'manifest.json');test=_read(paths['NAVTEST']/'manifest.json')
    if set(evidence['scenes'])!=set(train['tokens']):raise ValueError('Evidence population differs from TRAIN')
    # Physical-log separation uses the existing canonical physical log mapping.
    from starVLA.rl.qwen_world_postcot.data import POPULATION
    from tools.qwen_world_postcot.summarize import physical_log
    pop=_read(POPULATION)
    logs=[{physical_log(pop['token_logs'][token])for token in x['tokens']}for x in(train,dev,test)]
    if any(logs[a]&logs[b]for a,b in[(0,1),(0,2),(1,2)]):raise ValueError('Physical log leakage')
    from starVLA.rl.qwen_world_compare.track_targets import FutureTrackStore
    tracks=[FutureTrackStore(Path(common['target_root'])/role)for role in('TRAIN','DEV')]
    if any(set(t.index)!=set(x['tokens'])for t,x in zip(tracks,(train,dev))):
        raise ValueError('Native-yaw future track population incomplete')
    return dict(status='READY',data_counts=[90890,1696,12146],physical_logs=[len(x)for x in logs],
        evidence_sha256=evidence['sha256'],candidate_bank_sha256=candidate['sha256'],
        required_asset_sha256={str(p):file_sha(p)for p in required})


def _checkpoint(run, stage, updates=None):
    root=Path(run);paths=[]
    if updates is None:
        for name in ('LATEST.json','latest.json','progress.json'):
            p=root/name
            if p.exists():
                value=_read(p)
                if value.get('checkpoint'):paths.append(Path(value['checkpoint']))
    else:
        paths=list((root/'checkpoints').glob(f'update{updates:06d}*'))
    found=[]
    for p in paths:
        if not(p/'COMPLETE').exists()or '.incomplete' in p.name:continue
        info=(inspect_IL_checkpoint(p,verify_files='inventory')if stage=='IL'
              else inspect_RL_checkpoint(p,verify_files='inventory'))
        if updates is None or info['seal']['updates']==updates:found.append(info)
    return max(found,key=lambda x:x['seal']['updates'])if found else None


class Campaign:
    def __init__(self,plan,source_files,*,driver=None):
        from tools.qwen_world_compare import dispatch
        self.plan,self.sources=plan,source_files;self.source=Path(plan['source']);self.root=Path(plan['root'])
        self.controller_source=Path(plan.get('controller_source',plan['source']))
        self.controller_sources=expected_source_files(self.controller_source) if self.controller_source!=self.source else source_files
        self.driver=driver or dispatch;self.directory=self.root/'campaign';self.directory.mkdir(parents=True,exist_ok=True)
        self.path=self.directory/'STATE.json'
        if self.path.exists():
            self.state=_read(self.path)
            if self.state['plan_sha256']!=plan['sha256']or self.state['source_files']!=source_files:
                raise ValueError('Campaign resume changed its frozen plan/source')
        else:
            self.state=dict(schema=SCHEMA,plan_sha256=plan['sha256'],source_files=source_files,
                jobs={},architectures={a:dict(status='PENDING',evaluations={})for a in ARCHITECTURES})
        self.persist()

    def persist(self):
        atomic_json(self.path,self.state)

    def _poll(self):
        for key,entry in self.state['jobs'].items():
            if entry['status'] in ('COMPLETE','FAILED','BLOCKED_REVIEW_REQUIRED','REFUSED_RESOURCE'):continue
            try:result=self.driver.job_state(entry['receipt'])
            except ValueError:
                entry.update(status='BLOCKED_DISPATCH_RECEIPT',reason='Dispatch intent has no worker receipt; do not duplicate');continue
            status=result.get('status')
            if status=='REFUSED_RESOURCE'and result.get('child_not_started')is True:
                entry.update(status='REFUSED_RESOURCE',result=result)
            elif status=='REFUSED_RESOURCE':
                entry.update(status='BLOCKED_REVIEW_REQUIRED',result=result,reason='No positive pre-child refusal proof')
                self.state['architectures'][entry['architecture']]['status']='BLOCKED_REVIEW_REQUIRED'
            elif status=='EXITED'and result.get('returncode')==0:
                entry.update(status='COMPLETE',result=result)
            elif status in ('FAILED','FORCED_EXIT_REVIEW_REQUIRED','EXITED'):
                entry.update(status='BLOCKED_REVIEW_REQUIRED'if status=='FORCED_EXIT_REVIEW_REQUIRED'else'FAILED',result=result)
                self.state['architectures'][entry['architecture']]['status']=entry['status']
            else:entry.update(status='RUNNING',result=result)

    def _history(self,logical):
        return [v for v in self.state['jobs'].values()if v['logical']==logical]

    def _last(self,logical):
        values=self._history(logical);return values[-1]if values else None

    def _last_family(self,family):
        values=[v for v in self.state['jobs'].values()if v.get('family',v['logical'])==family]
        return values[-1]if values else None

    def _can_launch(self,logical):
        history=self._history(logical)
        if not history:return True
        last=history[-1]
        if last['status']=='REFUSED_RESOURCE'and len(history)>=MAX_PRECHILD_ATTEMPTS:
            self.state['architectures'][last['architecture']].update(status='BLOCKED_RESOURCE_RETRY_LIMIT',
                reason='Finite pre-child resource retry budget exhausted: '+logical)
        return last['status']=='REFUSED_RESOURCE'and len(history)<MAX_PRECHILD_ATTEMPTS

    def reserved_uuids(self):
        return {u for job in self.state['jobs'].values()if job['status']not in('COMPLETE','REFUSED_RESOURCE','FAILED')for u in job.get('uuids',[])}

    def _launch(self,arch,logical,command,host,*,gpus=(),uuids=(),minimum=65000,family=None,kind=None,stage=None,execution=None):
        attempt=len(self._history(logical))+1;name=f'four_{self.plan["sha256"][:10]}_{logical}_attempt{attempt:03d}'
        # Intent is durable before any SSH/Popen. A gap is blocked for review,
        # rather than presumed harmless and automatically duplicated.
        execution_source=self.controller_source if kind=='rl_qualification_helper' else self.source
        entry=dict(architecture=arch,logical=logical,attempt=attempt,command=command,host=host,
            scientific_source=str(self.source),execution_source=str(execution_source),
            execution_migration_sha256=self.plan.get('execution_migration_sha256'),
            family=family or logical,kind=kind,stage=stage,
            uuids=list(uuids),gpus=list(gpus),receipt=str(self.root/'jobs'/name),status='LAUNCH_INTENT',at=time.time())
        self.state['jobs'][name]=entry;self.persist()
        sharing={}if execution is None else dict(share_minimum=execution['share_min_free_mib'],lease_slot=execution['lease_slot'])
        directory=self.driver.launch(host,execution_source,command,name,gpus=','.join(map(str,gpus)),
            expected_uuids=list(uuids)if gpus else None,minimum=minimum if gpus and not execution else None,**sharing)
        entry['receipt']=str(directory);entry['status']='RUNNING';self.persist();return entry

    def _training(self,arch,stage):
        arm=self.plan['architectures'][arch];run=Path(arm[stage]);limit=100000 if stage=='IL'else 50000
        endpoint=_checkpoint(run,stage,limit)
        if endpoint:return endpoint
        family=f'{arch}_{stage}';logical=family
        last=self._last_family(family)
        if last and last['status']not in('COMPLETE','REFUSED_RESOURCE'):
            return None
        execution=None;registry=self.directory/'IL_EXECUTION_HANDOFFS.json'
        if stage=='IL'and registry.exists():
            table=_read(registry)
            if table.get('schema')!='qwen_world_compare_IL_handoff_registry_v1':raise ValueError('Invalid execution registry')
            execution=table['architectures'][arch]
            if execution['status']=='HOLD':
                self.state['architectures'][arch]['status']='WAITING_EXECUTION_HANDOFF';return None
            if execution['status']!='ACTIVE':raise ValueError('Unapproved IL execution state')
            receipt=_read(execution['receipt'])
            if file_sha(execution['receipt'])!=execution['receipt_file_sha256']or receipt.get('status')!='PASS':
                raise ValueError('IL execution receipt changed')
        if last and last['status']=='REFUSED_RESOURCE':logical=last['logical']
        if last and last['status']=='COMPLETE':
            # A native cooperative stop is a continuation, never a new run.
            latest=_checkpoint(run,stage)
            if latest is None:raise ValueError('Training exited without endpoint or legal recovery point')
            if any(run.glob('FAILURE_rank*.json')):
                raise ValueError('Scientific failure requires review; no automatic training replay')
            logical=f'{arch}_{stage}_resume{latest["seal"]["updates"]:06d}'
        if not self._can_launch(logical):return None
        fs=os.statvfs(self.root)
        if fs.f_bavail*fs.f_frsize<self.plan['minimum_disk_free_GiB']*2**30:
            self.state['architectures'][arch]['status']='WAITING_STORAGE';return None
        if execution:
            from starVLA.rl.qwen_world_compare.resources import probe,allowed_hosts
            host=execution['host'];actual=probe(host)
            if host not in allowed_hosts(admit_long_job=True):raise PermissionError('Handoff host was withdrawn')
            if [g['uuid']for g in actual['gpus']]!=execution['uuids']:raise ValueError('Handoff physical GPUs changed')
            route=(host,actual)if min(g['free_MiB']for g in actual['gpus'])>=execution['share_min_free_mib']else None
        else:route=self.driver.choose_host(excluded_uuids=self.reserved_uuids(),minimum=65000)
        if route is None:self.state['architectures'][arch]['status']='WAITING_TRAINING_RESOURCES';return None
        host,probe=route
        command=[MODEL_PYTHON,'-u','-m','torch.distributed.run','--standalone','--nproc-per-node=8','-m',
            'tools.qwen_world_compare.train_il'if stage=='IL'else'tools.qwen_world_compare.train_rl']
        if stage=='IL':command+=['--config',arm['config'],'--output',str(run)]
        else:
            selected=_read(arm['selection_IL'])['selected']
            command+=['--architecture',arch,'--config',self.plan['rl_config'],
                '--manifest',str(Path(self.plan['common']['data_root'])/'RL_HARD.json'),
                '--data',str(Path(self.plan['common']['data_root'])/'TRAIN'),
                '--il-checkpoint',selected['checkpoint'],'--parent-selection',arm['selection_IL'],'--output',str(run)]
        latest=_checkpoint(run,stage)
        if execution:
            tuning=receipt['new_identity']['execution']['execution_tuning']
            command+=['--checkpoint-policy',tuning['checkpoint_policy'],'--cpu-prefetch-depth',str(tuning['cpu_prefetch_depth']),
                '--execution-receipt',execution['receipt']]
            parent=receipt['parent_checkpoint']if latest['identity']==receipt['old_identity']else latest['path']
            command+=['--resume',parent]
        elif latest:command+=['--resume',latest['path']]
        self._launch(arch,logical,command,host,gpus=range(8),uuids=[v['uuid']for v in probe['gpus']],family=family,kind='training',stage=stage,execution=execution)
        self.state['architectures'][arch]['status']=stage+'_RUNNING';return None

    def _qualify_rl(self,arch,parent):
        """Finite disposable real-RL chain/inner2 resume gate before formal RL."""
        from tools.qwen_world_compare.qualify_rl import validate_acceptance
        arm=self.plan['architectures'][arch];record=self.state['architectures'][arch]
        base=self.root/'qualification'/arch/'RL';baseline=base/'baseline4';resumed=base/'resume_inner2_to4'
        acceptance=base/'RL_ENGINEERING_ACCEPTANCE.json'
        if acceptance.exists():
            value=validate_acceptance(acceptance,arch,parent,arm['selection_IL'],self.sources,
                                      self.controller_sources,self.plan['rl_config'],arm['qualification'])
            record['RL_qualification']=dict(status='PASS',receipt=str(acceptance),sha256=value['sha256'],
                formal_initialized_from_qualification=False)
            return True
        record['status']='WAITING_RL_QUALIFICATION'
        prefix=f'{arch}_RL_qualification'
        def completed(step):
            last=self._last(prefix+'_'+step)
            return bool(last and last['status']=='COMPLETE')
        def launch_step(step,command,*,gpu):
            logical=prefix+'_'+step
            if not self._can_launch(logical):return
            if gpu:
                route=self.driver.choose_host(excluded_uuids=self.reserved_uuids(),minimum=65000)
                if route is None:return
                host,probe=route
                self._launch(arch,logical,command,host,gpus=range(8),uuids=[g['uuid']for g in probe['gpus']],
                    kind='rl_qualification',stage='RL')
            else:
                host=self._cpu_host()
                if host:self._launch(arch,logical,command,host,kind='rl_qualification_helper',stage='RL')
        train=[MODEL_PYTHON,'-u','-m','torch.distributed.run','--standalone','--nproc-per-node=8',
            '-m','tools.qwen_world_compare.train_rl','--architecture',arch,'--config',self.plan['rl_config'],
            '--manifest',str(Path(self.plan['common']['data_root'])/'RL_HARD.json'),
            '--data',self.plan['data_paths']['TRAIN'],'--il-checkpoint',parent['selected']['checkpoint'],
            '--parent-selection',arm['selection_IL'],'--qualification','--stop-after-update','4','--save-every','1']
        helper=[MODEL_PYTHON,'-u','-m','tools.qwen_world_compare.qualify_rl']
        if not completed('baseline'):
            launch_step('baseline',train+['--output',str(baseline)],gpu=True)
        elif not completed('prepare_resume'):
            launch_step('prepare_resume',helper+['prepare-resume','--baseline',str(baseline),'--output',str(resumed)],gpu=False)
        elif not completed('resume'):
            launch_step('resume',train+['--output',str(resumed),'--resume',str(baseline/'checkpoints/update000002_inner2')],gpu=True)
        elif not completed('verify'):
            launch_step('verify',helper+['verify','--architecture',arch,'--baseline',str(baseline),'--resumed',str(resumed),
                '--parent-checkpoint',parent['selected']['checkpoint'],'--selection-path',arm['selection_IL'],
                '--training-source',str(self.source),'--controller-source',str(self.controller_source),
                '--config',self.plan['rl_config'],'--il-acceptance',arm['qualification'],
                '--output',str(acceptance)],gpu=False)
        else:
            raise ValueError('Real RL qualification verifier exited without a sealed acceptance')
        record['RL_qualification']=dict(status='PENDING',receipt=str(acceptance),
            budget=dict(logical_updates=4,physical_updates=6,new_candidates=64),
            formal_initialized_from_qualification=False)
        return False

    def _cpu_host(self):
        from starVLA.rl.qwen_world_compare.resources import probe
        for host in allowed_hosts():
            try:
                row=probe(host)
                if row['load1']<row['CPU_affinity']*.85 and row['MemAvailable_bytes']>=160*2**30:return host
            except Exception:continue
        return None

    def _export(self,arch,stage,step,checkpoint,il_parent=None):
        directory=self.root/'exports'/f'ARCH_{arch}'/f'{stage}_update{step:06d}'
        if (directory/'COMPLETE').exists():
            meta=_read(directory/'export.json');seal=_read(directory/'COMPLETE')
            if (seal['export_sha256']!=file_sha(directory/'export.json')or meta['architecture_id']!=arch
                    or meta['qualification'] or not(directory/'model.pt').is_file()):
                raise ValueError('Standalone export metadata/architecture changed')
            return directory,meta
        logical=f'{arch}_export_{stage}_{step:06d}'
        if self._can_launch(logical):
            # Serialize multi-GB reconstruction globally; retain inference/learners.
            if any('_export_'in j['logical']and j['status']=='RUNNING'for j in self.state['jobs'].values()):return None
            host=self._cpu_host()
            if host:
                command=[MODEL_PYTHON,'-u','-m','tools.qwen_world_compare.export','--checkpoint',str(il_parent or checkpoint),'--output',str(directory)]
                if stage=='RL':command+=['--rl-checkpoint',str(checkpoint)]
                self._launch(arch,logical,command,host,gpus=(),uuids=(),kind='export',stage=stage)
        elif self._last(logical)['status']=='COMPLETE':
            raise ValueError('Exporter exited without a complete standalone export')
        return None

    def _gpu_routes(self):
        from starVLA.rl.qwen_world_compare.resources import inventory
        from tools.qwen_world_compare.dispatch import leases_available
        routes,observed=inventory(minimum_free_MiB=65000);reserved=self.reserved_uuids()
        values=[]
        for row in routes:
            host=observed['second'][row['host']]
            if (row['uuid']not in reserved and host['load1']<host['CPU_affinity']*.9
                    and host['MemAvailable_bytes']>=80*2**30 and leases_available([row['uuid']])):values.append(row)
        return values

    def _evaluation(self,arch,stage,step,checkpoint,*,cohort='DEV',il_parent=None,selection=None,role=None,dev=None):
        label=f'{stage}_update{step:06d}';output=self.root/'evaluations'/f'ARCH_{arch}'/f'{cohort}_{label}'
        summary=output/'SUMMARY.json'
        if summary.exists():
            value=_read(summary);body=dict(value);seal=body.pop('sha256')
            if (fingerprint(body)!=seal or value['scenes']!=(1696 if cohort=='DEV' else 12146)
                    or value['invalid_scenes']or value['architecture_id']!=arch or value['cohort']!=cohort):
                raise ValueError('Changed/incomplete evaluation summary')
            return dict(updates=step,checkpoint=str(checkpoint)if checkpoint else None,evaluation=str(output),summary=value)
        export=self._export(arch,stage,step,checkpoint,il_parent)
        if export is None:return None
        export_path,meta=export
        prefix=f'{arch}_{cohort}_{label}'
        completed=[]
        for rank in range(8):
            logical=f'{prefix}_rank{rank:02d}';last=self._last_family(logical)
            p=output/f'rank{rank}.json'
            complete=last and last['status']=='COMPLETE'and p.exists()and _read(p).get('status')=='COMPLETE'
            completed.append(bool(complete))
        if all(completed):
            from tools.qwen_world_compare.evaluate import summarize
            value=summarize(output);atomic_json(summary,value)
            atomic_json(output/'COMPLETE.json',dict(status='COMPLETE',all_owned_workers_exited=True,
                summary_sha256=value['sha256'],scenes=value['scenes'],model_sha256=value['model_sha256']))
            return dict(updates=step,checkpoint=str(checkpoint)if checkpoint else None,evaluation=str(output),summary=value)
        live=sum(j['status']=='RUNNING'and('_DEV_'in j['logical']or'_NAVTEST_'in j['logical'])for j in self.state['jobs'].values())
        if live>=self.plan['maximum_evaluation_workers']:return None
        routes=None
        for rank,complete in enumerate(completed):
            if complete:continue
            family=f'{prefix}_rank{rank:02d}';logical=family;last=self._last_family(family)
            if last and last['status']not in('COMPLETE','REFUSED_RESOURCE'):continue
            if last and last['status']=='REFUSED_RESOURCE':logical=last['logical']
            if last and last['status']=='COMPLETE':
                # Successful scene-boundary pause: same rank/world/identity and
                # outputs are resumed, never rerun successful scenes.
                logical=f'{family}_scene_resume{len([j for j in self.state["jobs"].values()if j.get("family")==family]):03d}'
            if not self._can_launch(logical):continue
            if routes is None:routes=self._gpu_routes()
            if not routes:break
            route=routes.pop(0)
            command=[MODEL_PYTHON,'-u','-m','tools.qwen_world_compare.evaluate','--export',str(export_path),
                '--output',str(output),'--cohort',cohort,'--data',self.plan['data_paths'][cohort],
                '--rank',str(rank),'--world','8','--seed','2101']
            if rank==0:command+=['--native-parity-output',str(output/'native_export_parity.json')]
            if cohort=='NAVTEST':command+=['--selection',str(selection),'--policy-role',role,'--dev-evaluation',str(dev)]
            self._launch(arch,logical,command,route['host'],gpus=[route['gpu']],uuids=[route['uuid']],family=family,kind='evaluation',stage=stage)
            live+=1
            if live>=self.plan['maximum_evaluation_workers']:break
        return None

    def _dev_window(self,arch,stage,points,*,il_parent=None):
        arm=self.plan['architectures'][arch];values=[]
        for step in points:
            if stage=='RL'and step==0:
                selected=_read(arm['selection_IL'])['selected'];values.append(dict(updates=0,checkpoint=None,evaluation=selected['evaluation']));continue
            info=_checkpoint(arm[stage],stage,step)
            if info is None:continue
            value=self._evaluation(arch,stage,step,info['path'],il_parent=il_parent)
            if value:values.append(value)
        self.state['architectures'][arch][stage+'_DEV_complete']=[row['updates']for row in values]
        return values

    def _inference_profile(self,arch,request):
        root=self.root/'profiles'/f'ARCH_{arch}'/'selected_policy'
        path=root/'INFERENCE_PROFILE.json'
        export=self._export(arch,request['stage'],request['updates'],request['checkpoint'],request['il_parent'])
        if export is None:return False
        if path.exists():
            value=_read(path)
            if value.get('model_sha256')!=export[1]['model_sha256']or value.get('architecture_id')!=arch:
                raise ValueError('Selected-policy inference profile changed its model')
            job=self._last(f'{arch}_profile_selected')
            if job and job['status']!='COMPLETE':return False
            self.state['architectures'][arch]['inference_profile']=str(path);return True
        logical=f'{arch}_profile_selected'
        if self._can_launch(logical):
            routes=self._gpu_routes()
            if routes:
                route=routes[0]
                command=[MODEL_PYTHON,'-u','-m','tools.qwen_world_compare.profile','--export',str(export[0]),
                    '--output',str(root),'--data',self.plan['data_paths']['TRAIN'],'--repeats','3']
                self._launch(arch,logical,command,route['host'],gpus=[route['gpu']],uuids=[route['uuid']],kind='profile')
        elif self._last(logical)['status']=='COMPLETE':
            raise ValueError('Profiler exited without a complete inference profile')
        return False

    def tick(self,architectures=ARCHITECTURES,stages=('il','rl','evaluate')):
        self._poll()
        for arch in architectures:
            try:self._advance_architecture(arch,stages)
            except Exception:
                self.state['architectures'][arch].update(status='FAILED',error=traceback.format_exc(),
                    retry_requires_review=True,other_architectures_continue=True)
        self.state['updated_unix']=time.time();self.persist();return self.state

    def _advance_architecture(self,arch,stages):
        arm=self.plan['architectures'][arch];record=self.state['architectures'][arch]
        if record['status']in('FAILED','BLOCKED_REVIEW_REQUIRED'):return
        ok,status,reason=qualification_gate(arch,self.plan,self.sources)
        if not ok:record.update(status=status,reason=reason);return
        record.pop('reason',None)
        if 'il'not in stages and not Path(arm['selection_IL']).exists():record['status']='WAITING_IL_STAGE';return
        il_end=self._training(arch,'IL')if'il'in stages else _checkpoint(arm['IL'],'IL',100000)
        il_rows=self._dev_window(arch,'IL',IL_MILESTONES)
        if il_end is None:return
        if len(il_rows)!=len(IL_MILESTONES):record['status']='WAITING_IL_DEV';return
        from starVLA.rl.qwen_world_compare.parent_selection import select_il_parent,select_rl_policy,validate_evaluation_selection
        from starVLA.rl.qwen_world_compare.parent_selection import validate_selection
        parent=(validate_selection(arm['selection_IL'],stage='IL')if Path(arm['selection_IL']).exists()
                else select_il_parent(il_rows,il_end['path'],arm['selection_IL']))
        record['IL_parent_update']=parent['selected']['updates']
        if 'rl'not in stages and'evaluate'not in stages:record['status']='IL_COMPLETE';return
        if 'rl'not in stages and not Path(arm['selection_RL']).exists():record['status']='WAITING_RL_STAGE';return
        if not self._qualify_rl(arch,parent):return
        rl_end=self._training(arch,'RL')if'rl'in stages else _checkpoint(arm['RL'],'RL',50000)
        rl_rows=self._dev_window(arch,'RL',RL_MILESTONES,il_parent=parent['selected']['checkpoint'])
        if rl_end is None:return
        if len(rl_rows)!=len(RL_MILESTONES):record['status']='WAITING_RL_DEV';return
        selected=(validate_selection(arm['selection_RL'],stage='RL')if Path(arm['selection_RL']).exists()
                  else select_rl_policy(rl_rows,rl_end['path'],arm['selection_IL'],arm['selection_RL']))
        record['RL_selected_update']=selected['selected']['updates']
        if 'evaluate'not in stages:record['status']='RL_COMPLETE';return
        requests=final_policy_requests(parent,selected,arm['selection_IL'],arm['selection_RL'])
        complete=True
        for request in requests:
            value=self._evaluation(arch,request['stage'],request['updates'],request['checkpoint'],
                cohort='NAVTEST',il_parent=request['il_parent'],selection=request['selection'],
                role=request['role'],dev=request['dev'])
            if value is None:complete=False;continue
            export=self._export(arch,request['stage'],request['updates'],request['checkpoint'],request['il_parent'])
            for role,binding in request['bindings'].items():
                validation=validate_evaluation_selection(binding['selection'],role,export[1])
                record['evaluations'][role]=dict(evaluation=value['evaluation'],PDMS=value['summary']['PDMS'],
                    summary_sha256=value['summary']['sha256'],role_binding=validation,
                    shared_model_evaluation=len(request['bindings'])>1)
        if complete and set(record['evaluations'])=={'il_endpoint','il_selected','rl_endpoint','rl_selected'}:
            profile_request=next(r for r in requests if'rl_selected'in r['bindings'])
            record['status']='COMPLETE'if self._inference_profile(arch,profile_request)else'WAITING_INFERENCE_PROFILE'
        else:record['status']='WAITING_NAVTEST'


def final_policy_requests(il_selection,rl_selection,il_selection_path,rl_selection_path):
    """One export/evaluation per unique policy; all logical role bindings retained."""
    requests={};parent=il_selection['selected']['checkpoint']
    for role,selection,path in [('il_endpoint',il_selection,il_selection_path),('il_selected',il_selection,il_selection_path),
                           ('rl_endpoint',rl_selection,rl_selection_path),('rl_selected',rl_selection,rl_selection_path)]:
        row=selection['evaluated_candidates'][-1]if role.endswith('endpoint')else selection['selected']
        stage='IL'if role.startswith('il_')or row['updates']==0 else'RL'
        step=row['updates']if stage=='RL'or role.startswith('il_')else il_selection['selected']['updates']
        checkpoint=row['checkpoint']if stage=='RL'or role.startswith('il_')else parent
        key=(stage,step);binding=dict(selection=selection,selection_path=path,DEV_evaluation=row['evaluation'])
        if key not in requests:
            requests[key]=dict(stage=stage,updates=step,checkpoint=checkpoint,
                il_parent=parent if stage=='RL'else None,selection=path,role=role,dev=row['evaluation'],bindings={})
        requests[key]['bindings'][role]=binding
    return list(requests.values())


def _freeze_runs(plan,sources):
    root=Path(plan['root']);path=root/'RUNS.json'
    value=dict(plan=plan,source_files=sources,source_files_sha256=fingerprint(sources))
    if path.exists():
        if _read(path)!=value:raise ValueError('Existing RUNS/source identity changed; explicit migration required')
    else:atomic_json(path,value)


def _source_clean(source):
    status=subprocess.check_output(['git','status','--porcelain'],cwd=source,text=True)
    if status:raise ValueError('Formal campaign requires an immutable clean source worktree')
    return subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()


def selected_stages_complete(state,architectures,stages):
    target='COMPLETE'if'evaluate'in stages else'RL_COMPLETE'if'rl'in stages else'IL_COMPLETE'
    return all(state['architectures'][arch]['status'] in ({target,'COMPLETE'}if target!='IL_COMPLETE'else{'IL_COMPLETE','RL_COMPLETE','COMPLETE'})for arch in architectures)


def run(plan,source_files,args):
    require_dispatch();root=Path(plan['root'])
    if root!=ROOT.resolve():raise ValueError('Campaign ROOT differs from the current task grant namespace')
    controller=Path(__file__).resolve().parents[2];training=Path(plan['source']).resolve()
    if controller!=Path(plan.get('controller_source',training)).resolve():raise ValueError('Controller source path changed')
    controller_files=expected_source_files(controller)
    def check_execution_sources():
        if expected_source_files(training)!=source_files or expected_source_files(controller)!=controller_files:
            raise ValueError('Immutable scientific/controller source files changed')
        if controller!=training:
            from tools.qwen_world_compare.qualify_rl import validate_execution_migration
            path=plan.get('execution_migration')
            if not path or file_sha(path)!=plan.get('execution_migration_sha256'):
                raise ValueError('External controller requires an explicit sealed operations-only migration')
            validate_execution_migration(path,training,controller,source_files,controller_files)
    _source_clean(training);_source_clean(controller);check_execution_sources();_freeze_runs(plan,source_files)
    directory=root/'campaign';directory.mkdir(parents=True,exist_ok=True)
    with (directory/'owner.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if(directory/'STATE.json').exists()and not args.resume:raise ValueError('Existing campaign requires --resume')
        owner=dict(pid=os.getpid(),start_ticks=int(Path('/proc/self/stat').read_text().split()[21]),
            started_unix=time.time(),source=plan['source'],source_sha=_source_clean(plan['source']),
            controller_source=str(controller),controller_sha=_source_clean(controller),
            execution_migration_sha256=plan.get('execution_migration_sha256'))
        atomic_json(directory/'OWNER.json',owner)
        instance=Campaign(plan,source_files)
        def stop(*_):
            global STOP
            STOP=True
        for sig in(signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
        try:
            while not STOP:
                check_execution_sources()
                ready=preparation(plan);atomic_json(directory/'PREPARATION.json',ready)
                if ready['status']=='READY':instance.tick(args.architectures,args.stages)
                else:
                    for arch in args.architectures:instance.state['architectures'][arch]['status']='WAITING_ASSETS'
                    instance.persist()
                if args.once or selected_stages_complete(instance.state,args.architectures,args.stages):break
                time.sleep(args.poll_seconds)
            status='COMPLETE'if all(instance.state['architectures'][a]['status']=='COMPLETE'for a in ARCHITECTURES)else'OWNER_STOPPED'if STOP else'PENDING'
            atomic_json(directory/f'OWNER_RESULT_{owner["pid"]}.json',dict(status=status,existing_workers_preserved=True,at=time.time()))
        except BaseException:
            atomic_json(directory/f'FAILURE_{owner["pid"]}.json',dict(status='FAILED',error=traceback.format_exc(),workers_not_signalled=True));raise
    return instance.state


def report(plan):
    root=Path(plan['root']);state=_read(root/'campaign/STATE.json')if(root/'campaign/STATE.json').exists()else{'architectures':{}}
    rows={}
    for arch in ARCHITECTURES:
        value=state['architectures'].get(arch,dict(status='PENDING'));ev=value.get('evaluations',{})
        rows[arch]=dict(status=value['status'],IL100k_Navtest=ev.get('il_endpoint',{}).get('PDMS'),
            IL_parent_update=value.get('IL_parent_update'),IL_parent_Navtest=ev.get('il_selected',{}).get('PDMS'),
            RL50k_Navtest=ev.get('rl_endpoint',{}).get('PDMS'),RL_selected_update=value.get('RL_selected_update'),
            RL_selected_Navtest=ev.get('rl_selected',{}).get('PDMS'))
        if rows[arch]['RL50k_Navtest']is not None and rows[arch]['IL_parent_Navtest']is not None:
            rows[arch]['RL50k_minus_own_IL_parent']=rows[arch]['RL50k_Navtest']-rows[arch]['IL_parent_Navtest']
        cost={}
        for job in state.get('jobs',{}).values():
            if job['architecture']!=arch:continue
            result=job.get('result',{});kind=job.get('kind')or'unknown';key=kind+('_'+job['stage']if job.get('stage')else'')
            bucket=cost.setdefault(key,dict(job_wall_seconds=0.,allocated_GPU_hours=0.,physical_invocations=0,prechild_refusals=0))
            if result.get('child_not_started')is True:
                bucket['prechild_refusals']+=1;continue
            if result.get('end_unix')is not None and result.get('start_unix')is not None:
                wall=result['end_unix']-result['start_unix'];bucket['job_wall_seconds']+=wall
                bucket['allocated_GPU_hours']+=wall*len(job.get('gpus',[]))/3600;bucket['physical_invocations']+=1
        rows[arch]['cost']=cost
        rows[arch]['cost_semantics']='Admission on idle disjoint GPUs; allocated worker occupancy is not busy GPU compute. Summed worker wall is not end-to-end wall. Pending job costs not yet final.'
        for stage in('IL','RL'):
            p=Path(plan['architectures'][arch][stage])/'metrics_rank0.jsonl'
            if p.exists():
                with p.open('rb')as stream:stream.seek(max(0,p.stat().st_size-65536));lines=stream.read().splitlines()
                for line in reversed(lines):
                    try:value=json.loads(line)
                    except json.JSONDecodeError:continue
                    rows[arch][stage+'_actual_progress']={k:value.get(k)for k in('updates','cursor','raw_scene_exposures','new_candidates','collection')};break
        p=state['architectures'].get(arch,{}).get('inference_profile')
        if p and Path(p).exists():rows[arch]['inference_profile']=_read(p)
    result=dict(schema=SCHEMA,as_of_unix=time.time(),architectures=rows,
        complete=all(row['status']=='COMPLETE'for row in rows.values()),training_seed=42,evaluation_seed=2101,
        limits='Single training seed; all scores require full canonical Navtest. Pending is not zero or completed.')
    (root/'campaign').mkdir(parents=True,exist_ok=True)
    atomic_json(root/'campaign/REPORT.json',result);return result


def main():
    parser=argparse.ArgumentParser(__doc__);sub=parser.add_subparsers(dest='mode',required=True)
    for mode in('plan','prepare','verify','run','report'):
        p=sub.add_parser(mode);p.add_argument('--config',required=True)
        p.add_argument('--source',default=str(Path(__file__).resolve().parents[2]))
        p.add_argument('--execution-migration',help='Explicit controller-only lineage when --source remains the original scientific snapshot')
        if mode=='run':
            p.add_argument('--architectures',nargs='+',choices=ARCHITECTURES,default=list(ARCHITECTURES))
            p.add_argument('--stages',nargs='+',choices=['il','rl','evaluate'],default=['il','rl','evaluate'])
            p.add_argument('--resume',action='store_true');p.add_argument('--once',action='store_true')
            p.add_argument('--poll-seconds',type=float,default=60)
    args=parser.parse_args();plan=make_plan(args.config,args.source,execution_migration=args.execution_migration);sources=expected_source_files(args.source)
    if args.mode=='plan':result=plan
    elif args.mode=='prepare':result=preparation(plan)
    elif args.mode=='verify':result=dict(preparation=preparation(plan),gates={a:qualification_gate(a,plan,sources)for a in ARCHITECTURES},source_files=sources)
    elif args.mode=='run':result=run(plan,sources,args)
    else:result=report(plan)
    print(json.dumps(result,indent=2,allow_nan=False));return 0


if __name__=='__main__':raise SystemExit(main())
