"""CPU-only, same-native-parent full-cycle execution benchmark review.

Reads two finite world16 qualification roots. Never dispatches, scores, loads a
model/optimizer, edits journals, or publishes partial timing as a completed run.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import time
import torch
from starVLA.rl.qwen_world_compare.contracts import fingerprint
from starVLA.rl.qwen_world_compare.recovery_state import tensor_digest

SCHEMA='d46_same_native_world16_full_cycle_execution_benchmark_v1'
FLAGS={'fast_prefix','fuse_fm_mc','async_scoring','rotary_cache','no_grad_prefix_cache'}
ORIGIN,END,WORLD=1500,1512,16
CONTRACT='qwen_world_abcd_research_checkpoint_v1'
MODEL_SHA='386a0a27bd197cb002501bcb90c343659001d2e6d4f7bad0cbd5e31a91f8477d'
_HASH_CACHE={}


def need(ok,message):
    if not ok:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    path=Path(path).resolve();st=path.stat();key=(str(path),st.st_size,st.st_mtime_ns)
    if key not in _HASH_CACHE:
        h=hashlib.sha256()
        with path.open('rb')as stream:
            for block in iter(lambda:stream.read(2**20),b''):h.update(block)
        _HASH_CACHE[key]=h.hexdigest()
    return _HASH_CACHE[key]


def sealed(value,label):
    body=dict(value);digest=body.pop('sha256',None);need(fingerprint(body)==digest,'Changed '+label+' seal');return value


def finite(value,label):
    if isinstance(value,float):need(math.isfinite(value),'Nonfinite '+label)
    elif isinstance(value,dict):
        for k,v in value.items():finite(v,label+'.'+str(k))
    elif isinstance(value,list):
        for i,v in enumerate(value):finite(v,label+'.'+str(i))


def stat(values):
    values=list(values);need(bool(values),'No complete steady timing window')
    return dict(n=len(values),mean=statistics.mean(values),median=statistics.median(values),min=min(values),max=max(values))


def native(cp,identity):
    cp=Path(cp).resolve();seal=read(cp/'COMPLETE');files=read(cp/'files.json')
    need('.incomplete'not in cp.name and seal.get('schema')==2 and seal.get('contract')==CONTRACT and seal.get('world')==WORLD,'Unsealed/foreign native16')
    need(seal['files_sha256']==sha(cp/'files.json')and read(cp/'identity.json')==identity,'Changed native inventory/identity')
    required={'identity.json','optimizer.pt','transaction.json'}|{f'rank{r}.pt'for r in range(WORLD)}
    need(required<=files.keys(),'Missing native16 payload')
    for name,digest in files.items():
        need(Path(name).name==name and(cp/name).resolve().is_relative_to(cp)and sha(cp/name)==digest,'Changed native bytes '+name)
    tx=read(cp/'transaction.json');need(tx.get('schema')==CONTRACT and len(tx.get('ranks',[]))==WORLD,'Incomplete native transaction')
    normalized=[];rng=[];virtual=[];counters=[]
    for rank in range(WORLD):
        own=torch.load(cp/f'rank{rank}.pt',map_location='cpu',weights_only=False);s=own['state'];row=tx['ranks'][rank]
        need(own['identity_sha256']==identity['sha256']and s['updates']==seal['updates']and s['next_inner']==seal['next_inner'],'Rank native identity/clock changed')
        need(s['updates']==2*s['collections']-(2-s['next_inner']if own['rollout']is not None else 0),'Native collection/inner clock changed')
        need(s['phase']==seal['phase']and(row['updates'],row['next_inner'],row['scene_cursor'],row['run'])==(s['updates'],s['next_inner'],s['scene_cursor'],identity['sha256']),'Native transaction counters differ')
        need(s['scene_cursor']==64*s['collections']and s['new_candidates']==16*s['scene_cursor'],'Native B64/G16 clock differs')
        lanes=s['world16_virtual_sampler'];need(lanes['committed_collections']==s['collections']and len(lanes['lanes'])==8,'Invalid virtual8 state')
        lane_hash=[]
        for lane in lanes['lanes']:
            need(lane.dtype==torch.uint8 and lane.ndim==1,'Invalid virtual8 RNG');torch.Generator().set_state(lane)
            lane_hash.append(hashlib.sha256(lane.numpy().tobytes()).hexdigest())
        virtual.append(fingerprint(lane_hash));need(row['virtual8_rng_sha256']==virtual[-1],'Virtual8 transaction differs')
        rng.append(tensor_digest(own['rng']));value=copy.deepcopy(own);value.pop('identity_sha256');value['state'].pop('journal_origin_updates',None)
        normalized.append(tensor_digest(value));counters.append({k:s[k]for k in('updates','scene_cursor','valid_scene_exposures','new_candidates','extra_B1_queries','adaptive_G32_queries')})
    need(len(set(virtual))==1 and all(x==counters[0]for x in counters),'Native rank consensus changed')
    return dict(path=str(cp),seal=seal,files=files,files_file_sha256=sha(cp/'files.json'),common_optimizer_sha256=files['optimizer.pt'],
        normalized_rank_payload_sha256=normalized,rank_RNG_sha256=rng,virtual8_sha256=virtual[0],counters=counters[0])


def git_head(source):return subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()


def source_plan(root,identity,path=None):
    if path is not None:paths=[Path(path)]
    elif(root/'BENCHMARK_PLAN.json').is_file():paths=[root/'BENCHMARK_PLAN.json']
    else:paths=sorted((root.parent.parent/'operations/DEDICATED_TWO_NODE_OPTIMIZATION_20261103').glob('*LAUNCHES.json'))
    plans=[];receipts=[]
    for p in paths:
        value=read(p);selected=[]
        for row in value.get('receipts',[]):
            command=row.get('command',[])
            if '--output'in command and Path(command[command.index('--output')+1]).resolve()==root:selected.append(row)
        if selected:plans.append(dict(path=str(p.resolve()),sha256=sha(p)));receipts+=selected
    need(bool(receipts),'Missing exact benchmark source/launch plan; supply --baseline-plan/--candidate-plan')
    layout={};source_values=set()
    for row in receipts:
        command=row['command'];source=Path(row['source']).resolve();source_values.add((str(source),row['source_sha']))
        need('tools.d46_world16.train_fast'in command and '--qualification'in command,'Not typed qualification entry')
        need(command[command.index('--stop-after-update')+1]in('1511','1512'),'Unexpected benchmark update budget')
        need('--nnodes=2'in command and'--nproc_per_node=8'in command,'Not actual two-node16 launch')
        resume=Path(command[command.index('--resume')+1]).resolve();need(resume.is_relative_to(root/'checkpoints')and resume.name in('update001500_inner0','update001511_inner1'),'Foreign benchmark resume')
        node=next((int(x.split('=')[1])for x in command if x.startswith('--node_rank=')),None)
        need(node in(0,1)and len(row.get('expected_gpu_uuids',[]))==8,'Missing node/UUID layout')
        item=dict(host=row['host'],GPU_UUIDs=row['expected_gpu_uuids']);need(node not in layout or layout[node]==item,'Layout changed between benchmark attempts');layout[node]=item
    need(set(layout)=={0,1}and len(source_values)==1,'Incomplete/mixed source plan')
    source,commit=next(iter(source_values));need(git_head(source)==commit,'Frozen benchmark source commit changed')
    for name,digest in identity['source_files'].items():
        p=(Path(source)/name).resolve();need(p.is_relative_to(Path(source))and sha(p)==digest,'Frozen source inventory changed '+name)
    return dict(plans=plans,layout=layout,source=source,source_commit=commit,inventory_sha256=fingerprint(identity['source_files']))


def one(root,plan=None):
    root=Path(root).resolve();identity=sealed(read(root/'identity.json'),'identity');meta=identity.get('execution_upgrade',{})
    cfg=identity['recipe'];profile=meta.get('flags',{})
    need(identity.get('world')==WORLD and identity.get('qualification')is True and identity.get('fresh_Adam')is False and identity.get('parent_model_sha256')==MODEL_SHA,'Benchmark must be native16 qualification, not formal/fresh Adam')
    need((cfg.get('arm'),cfg.get('global_scenes'),cfg.get('group'),cfg.get('inner_updates'),cfg.get('mc_samples'))==('D3_FPO',64,16,2,4),'Changed D3_FPO/B64/G16/M2/MC4')
    need(set(profile)==FLAGS and all(type(v)is bool for v in profile.values())and not profile['async_scoring']and not profile['no_grad_prefix_cache'],'Unsupported execution profile')
    need(meta.get('schema')=='d46_world16_execution_upgrade_v1'and meta.get('origin_updates')==ORIGIN and meta.get('source_qualification')is False and meta.get('target_qualification')is True,'Not own formal1500 upgrade')
    upgrade=sealed(read(root/'EXECUTION_UPGRADE.json'),'upgrade');transition=sealed(read(root/'WORLD_TRANSITION.json'),'transition')
    need(upgrade['execution_upgrade']==meta and upgrade['target_identity_sha256']==identity['sha256']and transition['target_identity_sha256']==identity['sha256']and transition['execution_upgrade_sha256']==upgrade['sha256'],'Upgrade/profile receipt mismatch')
    need(meta['new_inventory_sha256']==fingerprint(identity['source_files'])and transition['source_inventory']==identity['source_files'],'Source inventory/profile mismatch')
    source_cp=Path(meta['source_world16_checkpoint']);source_identity=sealed(read(source_cp/'identity.json'),'formal source identity')
    need(source_identity.get('qualification')is False and source_identity.get('world')==WORLD,'Qualification cannot be the formal native parent')
    expected=copy.deepcopy(source_identity);expected.pop('sha256');expected.update(source_files=identity['source_files'],qualification=True,fresh_Adam=False,execution_upgrade=meta);expected['native_world_transition']['target_qualification']=True
    need(fingerprint(expected)==identity['sha256'],'Changed recipe/scope/parent/reference/data in execution upgrade')
    need(source_identity['sha256']==meta['source_identity_sha256']and sha(source_cp/'identity.json')==meta['source_checkpoint_identity_file_sha256'],'Formal source identity binding changed')
    start=native(root/'checkpoints/update001500_inner0',identity);source=native(source_cp,source_identity)
    need(start['seal']['phase']==source['seal']['phase']=='READY'and start['seal']['next_inner']==source['seal']['next_inner']==0 and source['seal']['updates']==ORIGIN,'Not same formal1500 READY')
    need(source['files_file_sha256']==meta['source_checkpoint_files_sha256']and source['common_optimizer_sha256']==start['common_optimizer_sha256']==meta['common_optimizer_sha256'],'Common Adam bytes differ')
    need(start['normalized_rank_payload_sha256']==source['normalized_rank_payload_sha256']and start['rank_RNG_sha256']==upgrade['source_rank_RNG_sha256']and start['virtual8_sha256']==upgrade['source_virtual8_rng_sha256'],'Initial16 RNG/virtual8/state not inherited exactly')
    finish=native(root/'checkpoints/update001512_inner0',identity);need(finish['seal']['phase']=='READY'and finish['seal']['next_inner']==0,'Missing final1512 READY')
    journals=[];rank_rows=[];invocations={};transitions=[]
    for rank in range(WORLD):
        path=root/f'metrics_rank{rank}.jsonl';lines=path.read_bytes().splitlines();rows=[json.loads(x)for x in lines]
        need([x['updates']for x in rows]==list(range(ORIGIN+1,END+1)),f'Missing/duplicate/extra committed updates rank{rank}')
        digests=[]
        for line,row in zip(lines,rows):
            finite(row,f'rank{rank}u{row["updates"]}');u=row['updates'];inner=(u-1)%2+1;c=(u+1)//2
            need(row['run']==identity['sha256']and row['collection']==c and row['inner_completed']==inner and row['behavior_update']==2*(c-1),'Wrong journal/behavior/inner identity')
            need(row['active_batch']['scenes']==64 and row['active_batch']['group']==16 and row['scene_exposures']==64*c and row['PG_candidates']==1024*c,'Changed exposure/query clock')
            need(row['generated_PG_candidates_including_padding']==1024*c and row['extra_B1_queries']==row['adaptive_G32_queries']==0,'Unexpected benchmark queries')
            need(all(row[k]>=0 for k in('update_seconds','collection_seconds_generation','collection_seconds_scoring','peak_GPU_bytes')),'Negative benchmark timing/memory')
            if inner==1:
                need(abs(row['losses']['fresh_ratio_error'])<=1e-4 and max(abs(row['losses']['fpo_joint_logratio_min']),abs(row['losses']['fpo_joint_logratio_max']))<=1e-4,'Fresh FPO/Gaussian ratio error')
            digests.append(dict(updates=u,raw_line_sha256=hashlib.sha256(line).hexdigest(),canonical_row_sha256=fingerprint(row)))
            key=(rank,row['invocation'])
            if key not in invocations:
                ip=root/'invocations'/row['invocation']/f'rank{rank}.json';iv=read(ip)
                need(iv['run']==identity['sha256']and iv['rank']==rank and iv['start_update']in(ORIGIN,1511),'Foreign invocation ancestry')
                resume=Path(iv['resume']).resolve();need(resume.is_relative_to(root/'checkpoints')and resume.name in('update001500_inner0','update001511_inner1'),'Foreign invocation resume')
                invocations[key]=dict(path=str(ip),file_sha256=sha(ip),record=iv)
        for a,b in zip(rows,rows[1:]):
            if a['invocation']!=b['invocation']:
                need((a['updates'],b['updates'])==(1511,1512),'Unexpected cross-invocation boundary')
                transitions.append(dict(rank=rank,from_update=1511,to_update=1512,from_invocation=a['invocation'],to_invocation=b['invocation']))
        result=read(root/f'result_rank{rank}.json');need(result['updates']==END and result['next_inner']==0 and result['status']=='QUALIFICATION_STOPPED','Missing complete rank result')
        journals.append(dict(rank=rank,path=str(path),file_sha256=sha(path),rows=digests));rank_rows.append({x['updates']:x for x in rows})
    need(not transitions or len(transitions)==WORLD,'Ranks disagree on pending resume transition')
    if transitions:
        pending=native(root/'checkpoints/update001511_inner1',identity)
        need(pending['seal']['phase']=='INNER_COMMITTED'and pending['seal']['next_inner']==1,'Missing real pending1511 checkpoint')
    else:pending=None
    spans=[]
    for end in range(ORIGIN+2,END+1,2):
        begin=end-2;warmup=end<=ORIGIN+4;cross=any(len({b[u]['invocation']for u in(range(begin,end+1)if begin>ORIGIN else range(begin+1,end+1))})!=1 for b in rank_rows)
        span=dict(start_update=begin,end_update=end,warmup=warmup,cross_invocation=cross,rank0_seconds=None if begin==ORIGIN else rank_rows[0][end]['at']-rank_rows[0][begin]['at'])
        span['included']=not warmup and not cross
        if span['rank0_seconds']is not None:need(span['rank0_seconds']>0,'Non-monotonic rank0 clock')
        span['max_rank_update_seconds']=[max(b[u]['update_seconds']for b in rank_rows)for u in(end-1,end)]
        span['max_rank_generation_seconds']=max(b[end]['collection_seconds_generation']for b in rank_rows)
        span['max_rank_base_scoring_seconds']=max(b[end]['collection_seconds_scoring']for b in rank_rows)
        spans.append(span)
    kept=[s for s in spans if s['included']];seconds=sum(s['rank0_seconds']for s in kept);need(seconds>0,'No complete same-invocation steady cycles')
    actual_queries=dict(PG_candidates=finish['counters']['new_candidates']-start['counters']['new_candidates'],extra_B1=0,adaptive_G32=0)
    for key in('physical_candidates','scorer_candidate_attempts','generation_worker_seconds','scoring_wait_worker_seconds'):
        records=[x['record']for x in invocations.values()]
        actual_queries[key]=sum(x[key]for x in records)if all(key in x for x in records)else None
    actual_queries['physical_counter_scope']='All native invocations represented in committed rows; uncommitted prior attempts outside these rows are not silently estimated.'
    need(actual_queries['PG_candidates']==6*64*16,'Wrong total actual finite benchmark query count')
    return dict(root=str(root),identity_sha256=identity['sha256'],identity_file_sha256=sha(root/'identity.json'),flags=profile,recipe=cfg,scope=identity['scope'],
        parent_model_sha256=identity['parent_model_sha256'],source_plan=source_plan(root,identity,plan),source_meta=meta,
        initial_native=start,source_native=source,final_native=finish,pending_resume_native=pending,
        journals=journals,invocations=list(invocations.values()),invocation_transitions=transitions,timing_windows=spans,
        steady_collections=len(kept),steady_seconds=seconds,steady_seconds_per_collection=seconds/len(kept),steady_scenes_per_second=64*len(kept)/seconds,
        full_cycle_seconds=stat(s['rank0_seconds']for s in kept),max_rank_optimizer_seconds=stat(v for s in kept for v in s['max_rank_update_seconds']),
        max_rank_generation_seconds=stat(s['max_rank_generation_seconds']for s in kept),max_rank_base_scoring_seconds=stat(s['max_rank_base_scoring_seconds']for s in kept),
        peak_GPU_bytes=max(b[u]['peak_GPU_bytes']for b in rank_rows for u in b),actual_queries=actual_queries)


def review(baseline,candidate,*,baseline_plan=None,candidate_plan=None):
    left=one(baseline,baseline_plan);right=one(candidate,candidate_plan)
    need(not any(left['flags'].values())and any(right['flags'].values()),'Require all-false baseline and opt-in fast candidate')
    need(left['recipe']==right['recipe']and left['scope']==right['scope']and left['parent_model_sha256']==right['parent_model_sha256'],'Not same recipe/scope/IL46')
    for key in('source_world16_checkpoint','source_identity_sha256','source_checkpoint_files_sha256','common_optimizer_sha256'):
        need(left['source_meta'][key]==right['source_meta'][key],'Different formal1500 parent '+key)
    need(left['initial_native']['normalized_rank_payload_sha256']==right['initial_native']['normalized_rank_payload_sha256'],'Different initial16 RNG/state')
    need(left['source_plan']['layout']==right['source_plan']['layout'],'Different benchmark hosts/UUID layout')
    matched=sorted({s['end_update']for s in left['timing_windows']if s['included']}&{s['end_update']for s in right['timing_windows']if s['included']})
    def matched_mean(value):return statistics.mean(s['rank0_seconds']for s in value['timing_windows']if s['end_update']in matched)
    need(bool(matched),'No matched steady full cycles')
    value=dict(schema=SCHEMA,status='COMPLETE',at=time.time(),baseline=left,candidate=right,
        full_cycle_speedup=left['steady_seconds_per_collection']/right['steady_seconds_per_collection'],
        matched_cycle_end_updates=matched,matched_full_cycle_speedup=matched_mean(left)/matched_mean(right),
        limits=['Qualification benchmark only; no new PDMS or policy-quality claim.',
          'First2 collections excluded; cross-invocation intervals excluded. Each cycle uses rank0 timestamps on its own clock, including generation/scoring/2Adam/control/I/O.',
          'Generation/base scoring fields omit FPO old-energy preparation; only full-cycle wall captures it. Max-rank component times are diagnostics, not an additive critical-path decomposition.',
          'GPU peak is cumulative process peak including startup/parity. Short windows/layout parity do not remove all load/noise or guarantee long-run speedup.',
          'Fusion changes BF16 operation grouping; equal initial native/RNG does not claim cross-execution final parameter bitexactness.'])
    value['sha256']=fingerprint(value);return value


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--baseline',required=True);p.add_argument('--candidate',required=True);p.add_argument('--output',required=True)
    p.add_argument('--baseline-plan');p.add_argument('--candidate-plan');a=p.parse_args()
    value=review(a.baseline,a.candidate,baseline_plan=a.baseline_plan,candidate_plan=a.candidate_plan)
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x')as stream:stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(output),full_cycle_speedup=value['full_cycle_speedup'],matched_full_cycle_speedup=value['matched_full_cycle_speedup'])))


if __name__=='__main__':main()
