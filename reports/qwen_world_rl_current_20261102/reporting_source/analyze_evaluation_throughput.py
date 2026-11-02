#!/usr/bin/env python3
"""One read-only metadata snapshot; no model, scorer, GPU query or controller."""
from __future__ import annotations
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib,json,os
from pathlib import Path
import statistics,time

ROOT=Path('/mnt/project/qwen-world-rl-optimization-20261030')
evidence=[]
def read(path):
    path=Path(path)
    if not path.is_file():return {}
    raw=path.read_bytes();stat=path.stat()
    evidence.append(dict(path=str(path),mtime_unix=stat.st_mtime,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
    return json.loads(raw)
def ratio(a,b):return a/b if b else None
def median(values):return statistics.median(values) if values else None

def derive_conclusions(result):
    evaluations=result['evaluations'];by_label={v['label']:v for v in evaluations}
    navtest=[v for v in evaluations if v['cohort']=='NAVTEST']
    scoring_fractions=[v['scoring_share_of_loop_wall'] for v in navtest if v['scoring_share_of_loop_wall'] is not None]
    cispo=by_label.get('CISPO_FULL_u001000_NAVTEST',{})
    full=by_label.get('PPO_FULL_u001000_NAVTEST',{})
    adaptive=by_label.get('PPO_ADAPTIVE_u001000_NAVTEST',{})
    waiting=[]
    for arm,run in result['campaign_arm_statuses'].items():
        if run.get('status')=='GATE_EVALUATED':
            entry=by_label.get(arm+'_u001000_NAVTEST',{})
            stamp=entry.get('controller_completed_at')
            waiting.append(dict(arm=arm,status=run['status'],navtest_completed_at=stamp,
                seconds_since_navtest_complete_at_capture=(result['captured_unix']-datetime.fromisoformat(stamp).timestamp()) if stamp else None))
    for e in evaluations:
        for rank in e['ranks']:
            host=next((v for v in result['resource_inventory_hosts'] if v['host']==rank['host']),{})
            stamp=host.get('inventory_unix')
            start=rank['rank_receipt'].get('start_unix')
            end=rank['rank_receipt'].get('end_unix',result['captured_unix'])
            rank['inventory_timestamp_overlaps_recorded_loop']=bool(stamp and start and start<=stamp<=end)
    result['conclusions']=[
        dict(id='single_rank_navtest_tail',finding='CISPO 的完整 NAVTEST 在仅一个可用路由时被固定为 world=1，是当前最明显的跨臂等待风险。',
            evidence=dict(label=cispo.get('label'),world=cispo.get('world'),scenes=cispo.get('scene_files'),expected=cispo.get('expected_scenes'),
                projected_remaining_seconds_at_lifetime_rate=cispo.get('projected_critical_tail_seconds_at_lifetime_rates')),
            interpretation='该剩余时间只是当前累计速率外推，不是保证或真实队列 ETA；world 不会随着后来空闲资源自动扩大。'),
        dict(id='global_gate_wait',finding='已有全部完成1k评测的臂仍为 GATE_EVALUATED；将1k诊断与继续训练解耦有直接流程依据。',
            evidence=dict(current_gate=result['current_gate'],arms=waiting),
            interpretation='这是一次元数据快照，不把潜在等待折算为实际GPU空转成本。'),
        dict(id='scorer_not_dominant',finding='已记录的同步 canonical scorer 仅占 NAVTEST 循环墙钟约6%至7%；主要被计时阶段是模型前向，不能据此认定CPU scorer为主瓶颈。',
            evidence=dict(scoring_loop_fraction_min=min(scoring_fractions),scoring_loop_fraction_max=max(scoring_fractions),
                scoring_seconds_per_scene_range=[min(v['scoring_seconds_per_new_scene'] for v in navtest),max(v['scoring_seconds_per_new_scene'] for v in navtest)]),
            interpretation='约21%至26%的循环时间尚未细分，包含观察处理、序列化/写盘、校验和rank0 parity。CPU压力和IO仍可能影响这些阶段；现有字段无法分离。'),
        dict(id='static_shard_heterogeneity',finding='固定场景数近乎相等的rank仍出现明显完成长尾；PPO_FULL首末rank完成相差约9.2分钟。',
            evidence=dict(PPO_FULL_completion_spread_seconds=full.get('completion_spread_seconds'),
                PPO_FULL_host_timings=full.get('host_timings'),PPO_ADAPTIVE_progress_range=[adaptive.get('progress_min'),adaptive.get('progress_max')],
                PPO_ADAPTIVE_host_timings=adaptive.get('host_timings')),
            interpretation='低吞吐主机与已有IL/GPU共享及较高CPU load的库存记录相关；这是过程关联，不能单独归因于CPU、GPU或场景难度。'),
        dict(id='resource_receipt_age',finding='资源清点只证明库存时刻；已完成job仍可留在claims直至下次调度清理，不应据此认定资源泄漏。',
            evidence=dict(inventory_age_seconds=result['inventory_age_seconds'],claims=len(result['claim_receipts']),
                claims_with_terminal_receipts=sum(v['worker_has_terminal_receipt'] for v in result['claim_receipts'])),
            interpretation='现有 dispatcher 在下一次 _eval_routes 中先剔除 RESULT 及已退出worker的claims。未发起新探测，未声明哪些GPU现在空闲。')]
    result['recommendations']=[
        dict(priority='P0',action='1k只诊断：每臂合法TRAIN反馈完成即可继续5k，不等待本臂或其他臂NAVTEST；完整Test继续异步，只有5k起才执行既定性能筛选。',
            owner='root/data agent',status='requested_by_root_not_executed_by_this_analysis',expected_effect='移除诊断门槛的跨臂等待；不改变模型、优化器、训练样本合法边界。'),
        dict(priority='P0',action='CISPO world=1的现有Test不应阻塞后续训练。若要加速它，须由唯一owner显式在场景边界drain并登记layout迁移，保留封存模型、token、seed及既有scene；不能直接给活跃world=1目录追加不同world。',
            owner='root',status='recommendation_only',expected_effect='避免约4.8小时累计速率外推成为全局关键路径；精确收益未知。'),
        dict(priority='P1',action='后续完整Navtest优先预留稳定的8或16个可用路由，避免瞬时仅1路由就固化完整人口；异构共享设备的固定分片按实际吞吐和setup延迟安排，不能仅凭剩余显存。',
            owner='root',status='recommendation_only',expected_effect='减少world不足和慢rank尾部；不以缩小人口或报告partial均值换速度。'),
        dict(priority='P1',action='先处理全局屏障、固定world和设备共享长尾；暂不增加canonical scorer并发。若后续仍慢，只为未细分的观察处理/写盘/初始化添加轻量计时，保持评分数学与协议。',
            owner='root',status='recommendation_only',expected_effect='当前scorer只占约6%至7%循环，单独优化它的端到端上限有限。')]
    return result

def main():
    captured=time.time()
    campaign=read(ROOT/'owners/CAMPAIGN/STATE.json')
    inventory=read(ROOT/'resources/LATEST_EVALUATION_INVENTORY.json')
    claims=read(ROOT/'resources/EVALUATION_CLAIMS.json')
    specs={}
    def visit(node,arm=None):
        if isinstance(node,list):
            for v in node:visit(v,arm)
        elif isinstance(node,dict):
            arm=node.get('id',node.get('arm',arm))
            spec=node.get('spec',{})
            if spec.get('label') and spec.get('output'):
                specs[spec['label']]=dict(spec=spec,arm=arm,status=node.get('status'),completed_at=node.get('completed_at'))
            for k,v in node.items():
                if k=='runs' and isinstance(v,dict):
                    for name,run in v.items():visit(run,name)
                elif isinstance(v,(dict,list)):visit(v,arm)
    visit(campaign)
    evaluations=[]
    for label,entry in sorted(specs.items()):
        spec=entry['spec'];output=Path(spec['output']);manifest=read(spec['manifest'])
        tokens=manifest.get('tokens',[]);expected=len(tokens) or manifest.get('count')
        layout=read(output/'layout.json')
        world=layout.get('world',spec['world'])
        filenames={p.name[:-5] for p in os.scandir(output/'scenes') if p.name.endswith('.json')} if (output/'scenes').exists() else set()
        summary=read(output/'SUMMARY.json')
        jobs={}
        for job_path in spec.get('jobs',[]):
            job=Path(job_path);launch=read(job/'LAUNCH.json');running=read(job/'RUNNING.json');result=read(job/'RESULT.json')
            command=launch.get('command',[])
            rank=int(command[command.index('--rank')+1]) if '--rank' in command else None
            jobs[rank]=dict(path=str(job),launch=launch,running=running,result=result)
        ranks=[]
        for rank in range(world):
            path=output/f'rank{rank}.json';record=read(path);job=jobs.get(rank,{})
            assigned=tokens[rank::world];done=sum(t in filenames for t in assigned)
            r=dict(rank=rank,host=record.get('host',job.get('launch',{}).get('host')),
                expected_scenes=len(assigned),scene_files=done,rank_receipt=record,
                rank_receipt_path=str(path),job_path=job.get('path'),
                worker_status=job.get('result',{}).get('status') or job.get('running',{}).get('status') or 'NO_START_RECEIPT',
                worker_returncode=job.get('result',{}).get('returncode'))
            if record:
                modified=path.stat().st_mtime;start=record.get('start_unix',modified)
                elapsed=record.get('wall_seconds',max(0,min(captured,modified)-start))
                r.update(receipt_age_seconds=max(0,captured-modified),observed_loop_elapsed_seconds=elapsed,
                    lifetime_loop_new_scenes_per_second=ratio(record.get('new',0),elapsed),
                    mean_forward_seconds_per_new_scene=ratio(record.get('forward_seconds',0),record.get('physical_inference',0)),
                    mean_scoring_seconds_per_new_scene=ratio(record.get('scoring_seconds',0),record.get('new',0)),
                    scoring_share_of_forward_plus_scoring=ratio(record.get('scoring_seconds',0),record.get('forward_seconds',0)+record.get('scoring_seconds',0)))
                r['estimated_remaining_seconds_at_observed_lifetime_rate']=ratio(max(0,len(assigned)-done),r['lifetime_loop_new_scenes_per_second'])
            gpu_ids=job.get('launch',{}).get('gpu_uuids',[])
            host_inventory=inventory.get('hosts',{}).get(r['host'],{})
            r['gpu_uuids']=gpu_ids
            r['inventory_snapshot_gpu_utilizations']={g['uuid']:g.get('utilization') for g in host_inventory.get('gpus',[]) if g['uuid'] in gpu_ids}
            r['inventory_snapshot_other_owners']=[v for v in host_inventory.get('owners',[]) if v.get('uuid') in gpu_ids and v.get('output')!=str(output)]
            ranks.append(r)
        new=sum(v['rank_receipt'].get('new',0) for v in ranks)
        forward=sum(v['rank_receipt'].get('forward_seconds',0) for v in ranks)
        scoring=sum(v['rank_receipt'].get('scoring_seconds',0) for v in ranks)
        loop=sum(v.get('observed_loop_elapsed_seconds',0) for v in ranks)
        completed=[v for v in ranks if v['rank_receipt'].get('status')=='COMPLETE']
        end_times=[v['rank_receipt']['end_unix'] for v in completed]
        all_complete=len(completed)==world
        groups=defaultdict(list)
        for rank in ranks:groups[rank['host']].append(rank)
        host_timing=[]
        for host,rs in groups.items():
            count=sum(v['rank_receipt'].get('new',0) for v in rs)
            host_timing.append(dict(host=host,ranks=len(rs),new_scenes=count,
                forward_seconds_per_new_scene=ratio(sum(v['rank_receipt'].get('forward_seconds',0) for v in rs),count),
                scoring_seconds_per_new_scene=ratio(sum(v['rank_receipt'].get('scoring_seconds',0) for v in rs),count)))
        evaluations.append(dict(label=label,arm=entry['arm'],cohort=spec.get('cohort'),output=str(output),
            controller_stage_status=entry['status'],summary_status=summary.get('status'),expected_scenes=expected,
            scene_files=len(filenames),unexpected_scene_files=len(filenames-set(tokens)),world=world,
            partition=layout.get('partition'),ranks_complete=len(completed),worker_status_counts=dict(Counter(v['worker_status'] for v in ranks)),
            new_scene_observations=new,rank_forward_seconds_sum=forward,rank_scoring_seconds_sum=scoring,
            rank_loop_seconds_sum=loop,scoring_share_of_loop_wall=ratio(scoring,loop),
            scoring_share_of_forward_plus_scoring=ratio(scoring,forward+scoring),
            forward_seconds_per_new_scene=ratio(forward,new),scoring_seconds_per_new_scene=ratio(scoring,new),
            uninstrumented_loop_seconds_sum=loop-forward-scoring,
            complete_rank_loop_wall_median_seconds=median([v['rank_receipt']['wall_seconds'] for v in completed]),
            completion_spread_seconds=max(end_times)-min(end_times) if all_complete else None,
            completed_rank_wait_until_last_rank_seconds_sum=sum(max(end_times)-e for e in end_times) if all_complete else None,
            caveat_completed_rank_wait='Potential fixed-shard tail only; not measured GPU idle time, because released slots can run other work.',
            progress_min=min([v['scene_files'] for v in ranks],default=0),progress_max=max([v['scene_files'] for v in ranks],default=0),
            projected_critical_tail_seconds_at_lifetime_rates=max([v['estimated_remaining_seconds_at_observed_lifetime_rate'] for v in ranks if v.get('estimated_remaining_seconds_at_observed_lifetime_rate') is not None],default=None),
            controller_completed_at=entry.get('completed_at'),ranks=ranks,host_timings=host_timing,
            metadata_snapshot_not_transactional=True))
    hosts=[]
    for name,host in inventory.get('hosts',{}).items():
        owners=host.get('owners',[])
        hosts.append(dict(host=name,inventory_unix=host.get('at'),CPU_affinity=host.get('CPU_affinity'),load1=host.get('load1'),
            gpu_count=len(host.get('gpus',[])),owner_count=len(owners),
            evaluation_owners=sum(v.get('entry')=='tools.qwen_world_research.evaluate' for v in owners),
            other_owners=sum(v.get('entry')!='tools.qwen_world_research.evaluate' for v in owners),
            gpu_utilization_range=[min((g.get('utilization',0) for g in host.get('gpus',[])),default=0),max((g.get('utilization',0) for g in host.get('gpus',[])),default=0)]))
    active_claims=[]
    for gpu,claim in claims.items():
        result=read(Path(claim['receipt'])/'RESULT.json')
        active_claims.append(dict(gpu_uuid=gpu,**claim,worker_has_terminal_receipt=bool(result),worker_returncode=result.get('returncode')))
    result=dict(schema='world_rl_evaluation_throughput_snapshot_v1',captured_utc=datetime.fromtimestamp(captured,timezone.utc).isoformat(),
        captured_unix=captured,capture_finished_unix=time.time(),scope='One read-only metadata snapshot of existing evaluation outputs; no scene payload, model, scorer or GPU query.',
        campaign_status=campaign.get('status'),current_gate=campaign.get('current_gate'),
        campaign_arm_statuses={k:dict(status=v.get('status'),stage_target_updates=v.get('stage_target_updates')) for k,v in campaign.get('runs',{}).items()},
        inventory_age_seconds=captured-inventory.get('at_unix',captured),resource_inventory_hosts=hosts,claim_receipts=active_claims,
        evaluations=evaluations,evidence=evidence,
        timing_semantics=dict(rank_start_unix='After preflight, imports, model loading, scorer construction and identity setup.',
            model_loading_seconds='Pre-loop setup, reported separately; do not add it into loop fractions.',
            forward_seconds='model.predict plus torch.cuda.synchronize, excludes CPU observation/processor.',
            scoring_seconds='Synchronous canonical RewardService.score wall time; includes IPC and scorer work.',
            loop_other='Observation/processor, validation, JSON/scenes/fsync metadata writes, rank0 native parity and other overhead; not further instrumented.',
            progress='Filename census plus separately read atomic rank receipts; transient one-scene differences possible.',
            ETA='Descriptive projection at observed lifetime post-load rate, not a scheduler deadline or guaranteed completion time.',
            resources='A historical resource inventory; neither fresh device probe nor proof of current idleness.'),
        conclusions=[],recommendations=[],actions_taken=dict(training_source_changed=False,worker_relaunched=False,worker_signalled=False,report_watcher_changed=False,model_or_scoring_calls=0))
    destination=ROOT/'analysis/evaluation_throughput_20261030.json'
    from render_report import atomic_json
    atomic_json(destination,derive_conclusions(result))
    print(json.dumps({'output':str(destination),'evaluations':len(evaluations),'inventory_age_seconds':result['inventory_age_seconds']}))
    for e in evaluations:
        print(json.dumps({k:e[k] for k in ('label','world','scene_files','expected_scenes','ranks_complete','forward_seconds_per_new_scene','scoring_seconds_per_new_scene','scoring_share_of_loop_wall','completion_spread_seconds','progress_min','progress_max','projected_critical_tail_seconds_at_lifetime_rates')},ensure_ascii=False))

if __name__=='__main__':main()
