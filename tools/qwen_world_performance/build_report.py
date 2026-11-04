"""Read-only living performance report. No GPU imports, dispatch or score creation."""
from datetime import datetime, timezone
import argparse
import html
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import time

ROOT=Path('/mnt/project/qwen-world-performance-rl-20261103')
LEGACY_ROOT=Path('/mnt/project/qwen-world-safety-exploration-20261031/integrated_training_20261102')
D46_ROOT=Path('/mnt/project/d46-priority-rl-20261103')
esc=lambda x:html.escape(str(x),quote=True)


def read(path, default=None):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return default


def atomic_text(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.writing-report-'+str(os.getpid()));tmp.write_text(value,encoding='utf-8');os.replace(tmp,path)


def D46_qualification_status(registered):
    """Separate real eight-rank qualification from the fresh formal run."""
    directory=Path(registered['qualification']);jobs=registered.get('qualification_jobs',[])
    current=jobs[-1]if jobs else {};job=Path(current['job'])if current.get('job')else None
    result=read(job/'RESULT.json',{})if job else {}
    running=read(job/'RUNNING.json',{})if job else {}
    identity=read(directory/'identity.json',{})
    ranks=[read(directory/f'result_rank{rank}.json',{})for rank in range(8)]
    if (result.get('returncode')==0 and identity.get('world')==8 and identity.get('qualification')is True
            and all(r.get('rank')==rank and r.get('status')=='QUALIFICATION_STOPPED'
                and r.get('qualification')is True and r.get('updates',0)>=10
                and r.get('frozen_before')and r.get('frozen_before')==r.get('frozen_after')
                for rank,r in enumerate(ranks))):return 'PASS · 8rank × '+str(min(r['updates']for r in ranks))+'更新'
    if result:
        return '资源等待（prechild75）'if result.get('returncode')==75 and result.get('child_not_started')else '资格异常待复核（RESULT '+str(result.get('returncode'))+'）'
    if running:
        rows=[tail(directory/f'metrics_rank{rank}.jsonl')for rank in range(8)]
        if all(r.get('at',0)>=running.get('start_unix',float('inf'))and r.get('updates',0)>0 for r in rows):
            progress=str(min(r['updates']for r in rows))+'/10'
            return '资格 '+progress+' · 8rank实测'+('（日志过期）'if max(time.time()-r['at']for r in rows)>600 else '')
        return '资格加载中，等待八rank更新'
    return '资格尚未派发'


def D46_throughput_fragment():
    path=D46_ROOT/'analysis/D46_THROUGHPUT_REVIEW_20261103.json';value=read(path,{})
    body=dict(value);seal=body.pop('sha256',None)
    if (value.get('schema')!='D46_measured_eight_rank_throughput_review_v1'
            or value.get('status')!='COMPLETE_CURRENT_SNAPSHOT'
            or hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()!=seal):return ''
    rows=value.get('runs',[])
    if not rows:return ''
    body='<h3>D46 速度实测（带时间的窗口）</h3><p>实测快照 '+esc(value['generated_UTC'])+'；每条全八rank、同一 invocation 最近100个完整 collection / 200次 Adam，不跨保存暂停。实时 Adam / 曝光 epoch 以最前训练表的最新 PRIORITY_STATUS 为准，本窗口不代替实时进度。纯更新与端到端耗时分开；后者包括本窗口采样、评分等待、材料准备、更新和保存。</p>'
    body+=table(['路线','纯更新 秒/Adam','端到端 秒/Adam','新场景/秒','新PG候选/秒','到首评触发约需小时'],[
        (r['arm'],f'{r["optimizer_seconds_mean"]:.3f}',f'{r["end_to_end_seconds_per_Adam"]:.3f}',
            f'{r["fresh_scenes_per_second"]:.3f}',f'{r["fresh_main_candidates_per_second"]:.2f}',
            f'{r["first_NAVTEST"]["hours_to_trigger_if_recent_rate_persists"]:.2f}')for r in rows])
    gates=sorted({(r['first_NAVTEST']['epoch'],r['first_NAVTEST']['trigger_updates'])for r in rows})
    requested=sum(bool(r['first_NAVTEST'].get('requests_already_written'))for r in rows)
    complete=value.get('complete_D46_NAVTEST_results',[])
    body+='<p><b>该速度快照时的常规首评门：'+esc('；'.join(str(e)+' exposure epoch / '+str(u)+'次实际 Adam'for e,u in gates))+'；当时已写请求 '+esc(requested)+' 条，完整 D46 RL NAVTEST '+esc(len(complete))+' 个。</b>此后提前批次的实际进度见上方面板。IL46 的89.49552098是父策略分数。上表估时只在当时负载不变时成立，未包括全12146评测耗时；D1两阶段信用、D2救援激活后需重测吞吐。<a href="'+esc(path)+'">全八rank速度与首评审计</a></p>'
    historical_path=D46_ROOT/'analysis/D_THROUGHPUT_ORIGINAL_BASELINES_20261103.json'
    historical=read(historical_path,{})
    if historical.get('status')=='COMPLETE_READ_ONLY_ACTUAL_HISTORY':
        comparison=historical.get('comparison',{});baseline_rows=[]
        for r in historical.get('baselines',[]):
            c=r['configuration'];w=r['stable_100_Adam_window']
            if r['id']=='ABCD_D_JAT_ACTION_IL12K':
                update=comparison['earliest_pure_update_seconds'];elapsed=comparison['earliest_end_to_end_seconds_per_Adam_first_actual_invocation']
                scope='纯更新100步；端到端为首个完整1000步 invocation'
            else:
                update=comparison['D44_pure_update_seconds'];elapsed=w['end_to_end_seconds_per_Adam'];scope='连续100步同一 invocation'
            baseline_rows.append((r['id'],str(c['global_scenes_per_collection'])+'/'+str(c['group'])+'/'+str(c['inner_Adam_updates_per_collection']),
                f'{update:.3f}',f'{elapsed:.3f}',str(c['amortized_new_scenes_per_Adam'])+'/'+str(c['amortized_new_PG_candidates_per_Adam']),scope))
        body+='<details><summary>相比最早 D：每步更慢，但每步处理的数据更多</summary>'
        body+=table(['历史正式路线','B/G/M','纯更新 秒/Adam','端到端 秒/Adam','每Adam新场景/PG候选','实际计时窗口'],baseline_rows)
        current=next((r for r in rows if r['arm']=='D0'),None)
        if current:
            body+='<p>当前 D0 每Adam新增 '+esc(current['fresh_scenes_per_Adam'])+' 场景 / '+esc(current['fresh_main_candidates_per_Adam'])+' 候选，最早 D 为2/16。此 D0 窗口的端到端每步时长为最早的 '+esc(f'{current["end_to_end_seconds_per_Adam"]/comparison["earliest_end_to_end_seconds_per_Adam_first_actual_invocation"]:.2f}')+'×，场景/候选吞吐分别为 '+esc(f'{current["fresh_scenes_per_second"]/comparison["earliest_amortized_fresh_scene_rate"]:.2f}')+'× / '+esc(f'{current["fresh_main_candidates_per_second"]/comparison["earliest_amortized_fresh_query_rate"]:.2f}')+'×。</p>'
        body+='<p>这是不同实际工作负载的计时：B/G/M、父权重、信用、LR、checkpointing和共享GPU同住任务均不同，计时窗口也不同；不是严格同算力或纯执行加速消融。按新场景/候选摊销，不等于独占GPU算力或独立token覆盖率。<a href="'+esc(historical_path)+'">原始D与D44历史计时证据</a></p></details>'
    return body


def sealed_JSON(value):
    body=dict(value);seal=body.pop('sha256',None)
    return bool(seal)and hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()==seal


def D46_early_checkpoint_matches(case):
    """Rebind small sealed inventories/identity; never load optimizer tensors."""
    cp=Path(case['checkpoint']);inventory=cp/'files.json';identity_path=cp/'identity.json'
    seal=read(cp/'COMPLETE',{});files=read(inventory,{});identity=read(identity_path,{})
    if not inventory.is_file()or not identity_path.is_file():return False
    signature=dict(parent=identity.get('parent_model_sha256'),trainable_payload=files.get('optimizer.pt'),scope=identity.get('scope'))
    return (seal.get('files_sha256')==hashlib.sha256(inventory.read_bytes()).hexdigest()
        and files.get('identity.json')==hashlib.sha256(identity_path.read_bytes()).hexdigest()
        and sealed_JSON(identity)and identity.get('sha256')==case.get('identity_sha256')
        and identity.get('architecture_id')=='D'and identity.get('qualification')is False
        and identity.get('recipe',{}).get('parent')=='D_IL46K'
        and seal.get('world')in(8,16)and seal.get('phase')=='READY'and seal.get('next_inner')==0
        and seal.get('updates')==case.get('updates')and files.get('optimizer.pt')==case.get('payload_sha256')
        and hashlib.sha256(json.dumps(signature,sort_keys=True,separators=(',',':')).encode()).hexdigest()==case.get('policy_signature'))


def D46_early_complete(case,node,baseline):
    output=Path(case['output']);summary_path=output.parent/'SUMMARY.json'
    summary=read(summary_path,{});identity=read(output/'identity.json',{})
    if (node.get('summary')not in(None,str(summary_path))
            or not sealed_JSON(summary)or not sealed_JSON(identity)
            or identity.get('schema')!='qwen_world_compare_evaluation_v1'
            or identity.get('checkpoint')!=case['checkpoint']or identity.get('updates')!=case['updates']
            or identity.get('D44_training_identity')!=case['identity_sha256']
            or identity.get('model_sha256')!=case['policy_signature']
            or identity.get('inference')!='single_candidate_cleanx_K10_shift5'
            or identity.get('count')!=12146 or len(identity.get('tokens',[]))!=12146
            or len(set(identity['tokens']))!=12146
            or hashlib.sha256(json.dumps(identity['tokens'],sort_keys=True,separators=(',',':')).encode()).hexdigest()!=identity.get('token_sha256')):return None,None
    expected=dict(status='COMPLETE',architecture_id='D',cohort='NAVTEST',scenes=12146,scene_denominator=12146,
        invalid_scenes=0,physical_logs=44,qualification=False,seed=2101,evaluation=str(output),
        evaluation_identity_sha256=identity['sha256'],model_sha256=identity.get('model_sha256'),
        protocol_sha256=identity.get('protocol',{}).get('sha256'),token_sha256=identity.get('token_sha256'))
    if (any(summary.get(k)!=v for k,v in expected.items())or summary.get('qualification')is not False or identity.get('qualification')is not False
            or identity.get('cohort')!='NAVTEST'or identity.get('architecture_id')!='D'or identity.get('seed')!=2101
            or not sealed_JSON(baseline)or baseline.get('scenes')!=12146
            or baseline.get('token_sha256')!=summary['token_sha256']or baseline.get('protocol_sha256')!=summary['protocol_sha256']
            or not isinstance(summary.get('PDMS'),(int,float))or not 0<=summary['PDMS']<=100
            or abs(summary['PDMS']-100*summary.get('means',{}).get('score',-1))>1e-8):return None,None
    paired_path=output.parent/'PAIRED_ANALYSIS.json';paired=read(paired_path,{})
    if (node.get('paired_analysis')not in(None,str(paired_path))or not sealed_JSON(paired)
            or paired.get('analysis_schema')not in('D46_complete_B1_paired_gain_loss_v1','D44_complete_B1_paired_gain_loss_v1')
            or paired.get('source_summary_sha256')!=summary['sha256']
            or any(paired.get(k)!=v for k,v in expected.items())
            or paired.get('paired',{}).get('scene_denominator')!=12146
            or paired.get('paired',{}).get('physical_groups')!=44
            or len(paired.get('paired',{}).get('CI95',[]))!=2
            or paired.get('paired',{}).get('all_scores_units')!='0..1; PDMS_display_explicitly_times100'):paired=None
    return summary,paired


def D46_early_evaluation_fragment():
    directory=D46_ROOT/'operations/EARLY_EVAL_20261103';plan_path=directory/'PLAN.json'
    plan=read(plan_path,{})
    if not plan:return ''
    body='<h3>提前完整 NAVTEST：四个独立策略，分波评测</h3>'
    if plan.get('schema')!='D46_finite_early_full_NAVTEST_plan_v1'or not sealed_JSON(plan):
        return body+'<p>计划封印校验未通过，暂不发布进度或成绩。</p>'
    state_path=directory/'STATE.json';state=read(state_path,{})
    state_ok=state.get('schema')=='D46_finite_early_full_NAVTEST_state_v1'and state.get('plan_sha256')==plan['sha256']
    nodes=state.get('cases',{})if state_ok else {};world=plan['world'];entries=[];scores=[];paired_rows=[];valid={};scored_signatures=set()
    baseline=read(Path(plan['baseline']).parent/'SUMMARY.json',{})
    for case in plan['cases']:
        matched=D46_early_checkpoint_matches(case);valid[case['id']]=matched
        node=nodes.get(case['id'],{});output=Path(case['output']);counts=dict(complete=0,started=0,admitting=0,waiting=0,failed=0)
        if node.get('output')!=case['output']:node={}
        for rank in range(world):
            jobs=node.get('jobs',{}).get(str(rank),[])
            if not jobs:counts['waiting']+=1;continue
            job=Path(jobs[-1]['job']);result=read(job/'RESULT.json',{});running=read(job/'RUNNING.json',{})
            shard=read(output/f'rank{rank}.json',{})
            if result:
                if result.get('returncode')==0 and shard.get('status')=='COMPLETE'and shard.get('rank')==rank and shard.get('world')==world:counts['complete']+=1
                elif result.get('returncode')==75 and result.get('child_not_started'):counts['waiting']+=1
                else:counts['failed']+=1
            elif running.get('child_pid'):counts['started']+=1
            elif running or jobs[-1].get('pid'):counts['admitting']+=1
        try:
            with os.scandir(output/'scenes')as files:written=sum(1 for f in files if f.name.endswith('.json')and f.is_file())
        except OSError:written=0
        progress='已写场景 '+str(written)+'/12146；分片完成 {complete}/'+str(world)+'；已启动待返回 {started}；准入 {admitting}；资源待派 {waiting}；异常 {failed}'
        label=node.get('status','等待状态收据')if matched else'检查点封印/身份不匹配'
        summary,paired=D46_early_complete(case,node,baseline)if matched and counts['complete']==world else(None,None)
        if summary:
            scores.append((case['arm'],case['updates'],f'{summary["PDMS"]:.8f}',f'{summary["PDMS"]-baseline["PDMS"]:+.6f}',str(output.parent/'SUMMARY.json')))
            scored_signatures.add(case['policy_signature']);label='完整分数与配对已验证'if paired else'完整分数已验证；配对分析中'
        entries.append((case['arm'],case['updates'],case['scene_exposures'],f'{case["exposure_epoch"]:.5f}',progress.format(**counts),label))
        if paired:
            p=paired['paired'];cohorts=paired.get('gain_loss',{}).get('fixed_reference_cohorts',{});high=cohorts.get('IL46_high_ge_0.9',cohorts.get('IL44_high_ge_0.9',{}))
            paired_rows.append((case['arm']+' / '+str(case['updates']),f'{100*p["mean_delta"]:+.6f}',
                '['+', '.join(f'{100*x:+.6f}'for x in p['CI95'])+']',p['repaired_zero'],p['new_zero'],
                f'{100*high["population_contribution"]:+.6f}'if high.get('population_contribution')is not None else'未提供',
                high.get('new_zero','未提供')))
    # Registered points and this independent early batch share the same full-result gate.
    for node in read(D46_ROOT/'CAMPAIGN.json',{}).get('evaluations',{}).values():
        request=node.get('request',{});checkpoint=request.get('checkpoint');output_path=node.get('output')
        if not checkpoint or not output_path or node.get('world',world)!=world or not(Path(output_path).parent/'SUMMARY.json').is_file():continue
        cp=Path(checkpoint);identity=read(cp/'identity.json',{});inventory=read(cp/'files.json',{});seal=read(cp/'COMPLETE',{})
        signature=dict(parent=identity.get('parent_model_sha256'),trainable_payload=inventory.get('optimizer.pt'),scope=identity.get('scope'))
        signature=hashlib.sha256(json.dumps(signature,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if signature in scored_signatures:continue
        case=dict(arm=identity.get('recipe',{}).get('arm','未标注'),checkpoint=checkpoint,output=output_path,updates=seal.get('updates'),
            identity_sha256=identity.get('sha256'),payload_sha256=inventory.get('optimizer.pt'),policy_signature=signature)
        if not D46_early_checkpoint_matches(case):continue
        complete=True
        for rank in range(node.get('world',world)):
            jobs=node.get('jobs',{}).get(str(rank),[]);shard=read(Path(output_path)/f'rank{rank}.json',{})
            if (not jobs or read(Path(jobs[-1]['job'])/'RESULT.json',{}).get('returncode')!=0
                    or shard.get('status')!='COMPLETE'or shard.get('rank')!=rank or shard.get('world')!=node.get('world',world)):complete=False;break
        summary,paired=D46_early_complete(case,node,baseline)if complete else(None,None)
        if summary:
            scored_signatures.add(signature)
            scores.append((case['arm'],case['updates'],f'{summary["PDMS"]:.8f}',f'{summary["PDMS"]-baseline["PDMS"]:+.6f}',str(Path(output_path).parent/'SUMMARY.json')))
        if paired:
            p=paired['paired'];cohorts=paired.get('gain_loss',{}).get('fixed_reference_cohorts',{});high=cohorts.get('IL46_high_ge_0.9',cohorts.get('IL44_high_ge_0.9',{}))
            paired_rows.append((case['arm']+' / '+str(case['updates']),f'{100*p["mean_delta"]:+.6f}',
                '['+', '.join(f'{100*x:+.6f}'for x in p['CI95'])+']',p['repaired_zero'],p['new_zero'],
                f'{100*high["population_contribution"]:+.6f}'if high.get('population_contribution')is not None else'未提供',high.get('new_zero','未提供')))
    if scores:
        best=max(scores,key=lambda r:float(r[2]))
        body+='<p><b>旧六路线最高完整 RL：'+esc(best[2])+' · '+esc(best[0])+' / Adam '+esc(best[1])+'；相固定 IL46 '+esc(best[3])+'。</b>已验证的独立完整策略点 '+esc(len(scores))+' 个（合并提前批次与常规评测，等参数reference不另计）。<b>NAVTEST_SELECTED</b>：用于当前资源择优，不是独立未选择测试；历史 IL70 最高91.45359756另列，不是D46父基线。</p>'
    body+='<p>'+esc(len(plan['cases']))+' 个独立参数策略、每个 '+esc(world)+' 个逻辑分片，按策略轮转在 '+esc(' / '.join(plan['hosts']))+' 分波。逻辑分片数不是同时占用GPU数；“已启动待返回”依据最新worker收据，不能等同正在评分。进度与分数不读部分均值；训练继续，评测不导出训练标签。<a href="'+esc(plan_path)+'">封印计划</a>'
    if state_path.exists():body+=' · <a href="'+esc(state_path)+'">实际分片状态</a>'
    body+='</p>'+table(['独立策略','Adam','有效场景曝光','exposure epoch','最新分片收据','阶段状态'],entries)
    references=[];cases={c['id']:c for c in plan['cases']}
    for r in plan.get('equivalent_references',[]):
        reference=cases.get(r.get('reference_case'),{})
        if (D46_early_checkpoint_matches(r)and valid.get(r.get('reference_case'))
                and r.get('policy_signature')==reference.get('policy_signature')and r.get('payload_sha256')==reference.get('payload_sha256')):
            references.append(r['arm']+' / Adam '+str(r['updates'])+' → '+reference['arm'])
    body+='<p><b>参数一致的早期参考：</b>'+esc('；'.join(references)or'等待等参数身份校验')+'。完整父策略、scope与封存payload签名相同，D1阶段II@2E和D2 rescue@.5E尚未在这些检查点启用；不复制D0分数冒充独立评测，不据此淘汰尚未启用的机制。不同曝光预算的四策略也不是同预算方法排名。</p>'
    if scores:
        body+=table(['已完成的独立策略','Adam','完整PDMS','相IL46'],[(a,u,p,d)for a,u,p,d,path in scores])
        body+='<p>'+ ' · '.join('<a href="'+esc(path)+'">'+esc(a)+'完整结果</a>'for a,u,p,d,path in scores)+'</p>'
        if len(paired_rows)<len(scores):body+='<p>部分完整分数的配对分析仍在汇总；未通过配对封印校验的修复/退化统计暂不发布。</p>'
    else:body+='<p><b>该提前批次尚无通过全量封印校验的完整 D46 RL 分数。</b>需 sealed identity、12146场景/44log/invalid0、seed2101及精确检查点绑定。</p>'
    if paired_rows:body+=table(['完整配对 vs IL46 / Adam','ΔPDMS','44log CI95','修复零分','新增零分','固定IL46≥.9全局贡献','该高分群体新增零分'],paired_rows)+'<p>自动聚合器部分沿用D44/IL44字段名；这里已按相同IL46父策略/人口/protocol及完整摘要封印绑定解释，不改原artifact。</p>'
    return body


def D46_world16_qualification_fragment():
    path=D46_ROOT/'operations/MULTINODE_NETWORK_20261103/GPU16_QUALIFICATION_STATUS.json';value=read(path,{})
    if value.get('schema')!='D46_actual_world16_qualification_status_v1':
        return '<p>world16 的Tiny-native CPU跨world恢复通过；完整GPU/NCCL资格以实际收据为准，网络元数据不是多节点训练速度。</p>'
    stamp=value.get('updated_at')
    if isinstance(stamp,(int,float)):stamp=datetime.fromtimestamp(stamp,timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    flag=lambda key:'是'if value.get(key)is True else'否'if value.get(key)is False else'未记录'
    proof_path=Path(value.get('qualification_receipt',''));proof=read(proof_path,{})
    engineering=(value.get('engineering_qualified')is True and value.get('native_midinner_resume_passed')is True
        and proof_path.is_file()and hashlib.sha256(proof_path.read_bytes()).hexdigest()==value.get('qualification_receipt_file_sha256')
        and sealed_JSON(proof)and proof.get('schema')=='D46_actual_full_model_world16_pending_resume_qualification_v1'
        and proof.get('status')=='PASS'and proof.get('world')==16 and proof.get('native_midinner_restored')is True
        and proof.get('full_RNG_fields_preserved')is True and len(proof.get('ranks',[]))==16)
    body='<p><b>world16 动态资格状态：'+esc(value.get('status','待核对'))+'</b>（'+esc(stamp)+'）。工程恢复资格通过：'+('是'if engineering else'尚未核实通过')+'；长期性能资格通过：'+flag('performance_qualified')+'；正式world16启动标记：'+flag('formal_world16_started')+'。<a href="'+esc(path)+'">真实资格状态</a></p>'
    if engineering:
        body+='<p>真实D0全模型16rank从1403_inner1跨host恢复至1404_READY，两node正常EXIT0；Adam/scheduler均1404，16份RNG与原8虚拟采样流保留，pending复用未新增曝光或评分，16rank原生B1恢复零差。<a href="'+esc(proof_path)+'">封印的中途原生恢复证明</a>。实际GPU资格路线：'+esc(' / '.join(value.get('actually_GPU_qualified_arms',[])))+'；仅引擎支持、尚无该路线16卡GPU资格：'+esc(' / '.join(a for a in value.get('supported_engine_arms',[])if a not in value.get('actually_GPU_qualified_arms',[]))or'无')+'。</p>'
    body+='<details><summary>有界GPU资格过程（原失败保留）</summary>'
    for key,attempt in value.items():
        if not isinstance(attempt,dict)or not(key.endswith('_attempt')or key.startswith('resume_')):continue
        facts=[]
        if attempt.get('status'):facts.append(str(attempt['status']))
        if 'new_Adam_updates'in attempt:facts.append('新增Adam '+str(attempt['new_Adam_updates']))
        if attempt.get('all16_native_B1_restore_parity_zero')is True:facts.append('全部16rank真实B1恢复零差')
        if attempt.get('all16_gradient_norm_finite')is True:facts.append('16rank梯度有限')
        times=[x for x in attempt.get('update_seconds_critical_rank',[])if isinstance(x,(int,float))and math.isfinite(x)]
        if times:facts.append('纯更新max-rank '+f'{min(times):.2f}–{max(times):.2f}'+'秒（'+str(len(times))+'步短窗）')
        if attempt.get('checkpoint_sealed')is True:facts.append('原生检查点已封存，pending inner '+str(attempt.get('pending_inner','未记录')))
        if 'exit_code'in attempt:facts.append('退出码 '+str(attempt['exit_code']))
        error=attempt.get('error',attempt.get('first_error'))
        if error:facts.append('保留错误：'+str(error))
        body+='<p>'+esc(key)+'：'+esc('；'.join(facts))+'</p>'
    followup=('正式D0 world16已启动，最新进度与短窗口吞吐见前部；长期效果仍待评估。'if value.get('formal_world16_started')is True else'工程恢复已通过，下一步据完整分数选择并登记正式资源；长期吞吐仍需实测。')if engineering else'当前工程资格仍待完整收尾与中途恢复核对。'
    body+='</details><p>资格短窗本身不能证明长期稳定或稳定2倍加速；正式启动标记见上。'+followup+'现有六路线和早评资源是否改变，只以实际运行/资源收据为准。</p>'
    return body


def D46_optimization_health_fragment():
    path=D46_ROOT/'analysis/EARLY_OPTIMIZATION_HEALTH_20261103.json';value=read(path,{})
    if value.get('status')!='COMPLETE_BOUNDED_CPU_AUDIT'or not sealed_JSON(value):return ''
    compared=value['cross_arm_scale_comparison'];arms=value['arms'];window=value['matched_window']
    ratios=[]
    for arm in ('D3_FPO','D3_AWM'):
        c=compared[arm]
        ratios.append(arm+'：grad为D0的'+f'{100*c["gradient_mean_first800_relative_D0"]:.3f}'+'%，但相同LoRA_B实际位移仍为'+f'{100*c["text_LoRA_B_delta_L2_relative_D0"]:.1f}'+'%')
    first={a:r['equal_exposure_first800']for a,r in arms.items()}
    zeros=' / '.join(a+':'+str(int(first[a]['exploration_global_once_per_collection']['canonical_allzero_groups']))for a in ('D0','D5','D3_FPO','D3_AWM'))
    fpo_clip=first['D3_FPO']['local_losses_equal_rank_weight']['fpo_clip_fraction']['mean']
    body='<details><summary>为何等待完整评测择优：有界训练健康审计，不是部署收益</summary>'
    body+='<p>审计 '+esc(value['generated_utc'])+'：四路线全八rank前 '+esc(window['optimizer_updates'])+' 次 Adam、同 '+esc(window['valid_scene_exposures'])+' 个 TRAIN 场景曝光 / '+esc(window['physical_candidates'])+' 候选，均 loss/gradient finite、首inner fresh-ratio误差0；不把各自不同曝光的recent窗口当匹配对照。</p>'
    body+='<p>'+esc('；'.join(ratios))+'。小梯度不等于没有学习，Adam/epsilon尺度统计也不能直接证明PDMS损失。FPO真正energy-ratio clip为 '+esc(f'{100*fpo_clip:.1f}%')+'；第二inner普通Gaussian诊断约75%不是FPO更新被clip的比例，AWM也没有ratio-clipped目标。</p>'
    body+='<p>原始A有效组约96%；同25600组的canonical全零仅 '+esc(zeros)+'。这说明本窗口多数有训练信用，不代表信用方向已改善部署。D5每步触发norm1裁剪、action I/O主导且近期参考beta升高，仍需完整B1和高分保持证据；不能据单一loss、小grad或clip诊断淘汰，更不能据其推断clip/epsilon/anchor的提分因果。<a href="'+esc(path)+'">完整健康审计与边界</a></p>'
    body+=D46_world16_qualification_fragment()+'</details>'
    return body


def D46_formal_world16_fragment():
    status=read(D46_ROOT/'operations/MULTINODE_NETWORK_20261103/GPU16_QUALIFICATION_STATUS.json',{})
    path=Path(status.get('formal_world16_actual_proof',''));proof=read(path,{})
    if (status.get('formal_world16_started')is not True or status.get('formal_world16_stage_id')!='D0_WORLD16_PRIORITY_v2'
            or proof.get('schema')!='d46_formal_world16_actual_training_started_v1'
            or proof.get('status')!='ALL16_ACTUAL_FORMAL_TRAINING'or proof.get('world')!=16
            or proof.get('own_optimizer_inherited_without_reset')is not True or proof.get('actual_new_updates_min',0)<22
            or len(proof.get('ranks',[]))!=16):return ''
    body='<p><b>D0正式双节点16卡已实际训练，非资格任务。</b>当前v2阶段从自身1714_READY继承Adam/RNG与原6E预算；全部16rank已实际跨过原点至少22次Adam，原生B1恢复零差，无资格stop-after上限。实际主机/最新更新见实时表。<a href="'+esc(path)+'">正式16rank启动证明</a>。工程资格通过与长期效果分开，尚无该16卡阶段新增完整PDMS。</p>'
    speed_path=D46_ROOT/'analysis/D0_WORLD16_THROUGHPUT_WORKING.json';speed=read(speed_path,{})
    if speed.get('full_cycle_speedup')and speed.get('world8')and speed.get('world16'):
        a=speed['world8'];b=speed['world16'];stamp=datetime.fromtimestamp(speed['at'],timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        body+='<h3>D0实测双节点吞吐：完整cycle '+esc(f'{speed["full_cycle_speedup"]:.4f}')+'×</h3><p>'+esc(stamp)+'固定实测窗口，B64/G16/M2不变；16卡跳过前4次warm updates。</p>'
        body+=table(['实测指标','8卡独占窗口','正式16卡窗口'],[
            ('实际更新区间',str(a['updates']),str(b['updates'])),('完整collection数',a['steady_collections'],b['steady_collections']),
            ('完整cycle秒',f'{a["steady_collection_seconds"]:.5f}',f'{b["steady_collection_seconds"]:.5f}'),
            ('纯更新中位秒',f'{a["optimizer_seconds_median"]:.4f}',f'{b["optimizer_seconds_median"]:.4f}'),
            ('新场景/小时',f'{3600*a["steady_scenes_per_second"]:.0f}',f'{3600*b["steady_scenes_per_second"]:.0f}'),
            ('按当前吞吐外推1E小时',f'{103288/a["steady_scenes_per_second"]/3600:.1f}',f'{103288/b["steady_scenes_per_second"]/3600:.1f}')])
        body+='<p>这是已测短窗口完整cycle加速，外推不含启动和完整评测；不是长期保证，也不是新PDMS或算法胜出。<a href="'+esc(speed_path)+'">真实journal窗口输入</a></p>'
    return body


def D46_FPO_world16_fragment():
    current=next((r for r in read(D46_ROOT/'PRIORITY_STATUS.json',{}).get('runs',[])if r.get('id')=='D46_D3_FPO'),{})
    stage=current.get('execution_stage','FPO_WORLD16_PRIORITY_v1')
    directory=D46_ROOT/'operations/WORLD16_STAGES'/stage;plan=read(directory/'PLAN.json',{})
    audit_path=D46_ROOT/'operations/FPO_WORLD16_TSH4_20261103/GPU_QUALIFICATION_AUDIT.json';audit=read(audit_path,{})
    if (audit.get('status')!='PASS'or audit.get('world')!=16 or len(audit.get('ranks',[]))!=16
            or not audit_path.is_file()or hashlib.sha256(audit_path.read_bytes()).hexdigest()!=plan.get('engineering_qualification_sha256')):return ''
    proof_path=directory/'ACTUAL_FORMAL_TRAINING.json';proof=read(proof_path,{})
    actual=(proof.get('schema')=='d46_FPO_formal_world16_actual_training_started_v1'and proof.get('status')=='ALL16_ACTUAL_FORMAL_TRAINING'and proof.get('world')==16
        and len(proof.get('ranks',[]))==16 and proof.get('actual_new_updates_min',0)>0
        and proof.get('optimizer_reset')is False and proof.get('qualification_weights_used')is False
        and proof.get('identity_sha256')==plan.get('identity_sha256'))
    current_label='当前正式16rank训练'if current.get('status')in('TRAINING','TRAINING_WORLD16')else'当前 '+str(current.get('status','待核对'))+'；仅启动/恢复收据不计实际训练'
    body='<h3>FPO · 当前双节点阶段与历史吞吐</h3><p><b>'+esc(current_label)+'</b>。'+esc(stage)+'；当前主机 '+esc(current.get('host','待分配'))+'。自身Adam/RNG/MC4材料与原6E预算继承，无资格stop-after；不使用资格权重作为正式起点。<a href="'+esc(directory/'STATE.json')+'">当前阶段</a> · <a href="'+esc(audit_path)+'">原GPU资格与pending恢复</a>'
    if actual:body+=' · <a href="'+esc(proof_path)+'">正式16rank新更新实证</a>'
    recovery=plan.get('operator_recovery_review',{})
    if recovery.get('checkpoint'):body+=' · 从 '+esc(Path(recovery['checkpoint']).name)+' 原生恢复，科学源保持不变'
    body+='</p><p>原first3+pending resume1通过，16份RNG、虚拟8流/材料/Adam与B1零差，pending恢复未新增PG查询。早期资源75/peer rendezvous失败保留；按已知受保护IL共同驻留的标准显存准入，不触碰IL。资格前三步纯更新时间不当完整cycle；正式进度不是新分数。FPO Adam900的完整PDMS为90.12278094，后续分数只以前部完整结果表为准。</p>'
    speed_path=D46_ROOT/'analysis/FPO_WORLD16_THROUGHPUT_20261103.json';speed=read(speed_path,{})
    if speed.get('schema')=='d46_fixed_window_actual_journal_throughput_v1'and speed.get('arm')=='D3_FPO':
        a=speed['world8'];b=speed['world16']
        body+='<p><b>FPO历史TSH3+TSH4独立实测完整cycle '+esc(f'{speed["full_cycle_speedup"]:.4f}')+'×</b>，与D0的2.0727×分开。B64/G16/M2/MC4不变，原正式窗口跳前4次warm更新；不能套作当前新主机组合的速度。</p>'
        body+=table(['FPO实测指标','8卡窗口','正式16卡窗口'],[
            ('实际更新区间',str(a['updates']),str(b['updates'])),('完整collection数',a['steady_collections'],b['steady_collections']),
            ('完整cycle秒',f'{a["steady_collection_seconds"]:.5f}',f'{b["steady_collection_seconds"]:.5f}'),
            ('纯更新中位秒',f'{a["optimizer_seconds_median"]:.4f}',f'{b["optimizer_seconds_median"]:.4f}'),
            ('新场景/秒',f'{a["steady_scenes_per_second"]:.6f}',f'{b["steady_scenes_per_second"]:.6f}')])
        body+='<p>短窗口且允许受保护IL共同驻留；不含启动、完整评测或长期保证，不声明新PDMS。<a href="'+esc(speed_path)+'">FPO真实journal计时输入</a></p>'
    return body


def D46_current_interruption_fragment(value):
    row=next((r for r in value.get('runs',[])if r.get('id')=='D46_D3_FPO'),{})
    if not row.get('status','').startswith('FAILED'):return ''
    directory=D46_ROOT/'operations/WORLD16_STAGES/FPO_WORLD16_PRIORITY_v1'
    state=read(directory/'STATE.json',{});native=state.get('latest_native',{})
    interrupted=False
    for node in state.get('attempts',[{}])[-1].get('nodes',[]):
        path=Path(node['job'])/'stdout.log'
        try:
            with path.open('rb')as f:f.seek(max(0,path.stat().st_size-65536));text=f.read().decode(errors='replace')
            interrupted|='SignalException:'in text and'got signal: 15'in text
        except OSError:pass
    cause='节点1收到外部 SIGTERM，属于执行中断，不据此判数值不稳定。'if interrupted else'执行失败待复核，不能按旧启动收据计为健康在训。'
    return '<p style="background:#fff0d6;padding:12px"><b>FPO 当前中断、待恢复，不计实际在训。</b>'+esc(cause)+'最新完整原生状态 '+esc(native.get('updates',row.get('updates','待核对')))+' / inner'+esc(native.get('next_inner','待核对'))+'；恢复是否成功只据新的实际日志，不沿用旧16rank健康/五条56卡描述。<a href="'+esc(directory/'STATE.json')+'">当前stage与原失败收据</a></p>'


def D46_execution_benchmark_fragment():
    directory=D46_ROOT/'qualification/EXECUTION_BENCHMARK_20261103'
    rows=[];comparison=read(directory/'COMPARISON_0ab.json',{})
    checks={r['mode']:r for r in comparison.get('comparisons',[])if r.get('mode')}
    for mode,label in [('baseline','原执行'),('prefix','prefix检查优化'),('fusion','融合MC'),('both','两项组合')]:
        path=directory/(mode+'.json');value=read(path,{})
        if value.get('schema')!='d46_FPO_execution_microbenchmark_result_v1'or value.get('status')!='MEASURED_NOT_FORMAL_ADMISSION':continue
        checked=checks.get(mode,{})
        gradient='基准'if mode=='baseline'else'梯度对比待齐'if not checked else'逐位一致'if checked.get('bitexact')is True else'非逐位一致；梯度相对L2 '+format(checked.get('gradient_relative_L2',float('nan')),'.5f')
        rows.append((label,format(value['median_seconds'],'.4f'),format(value['peak_GPU_allocated_bytes']/2**30,'.2f'),gradient))
    if not rows:return ''
    return '<details><summary>执行微基准：checkpoint开启的局部测试，尚未采用正式训练</summary><p><b>重要边界：这四项原始微基准都开启activation checkpoint；正式训练使用all_off。</b>15.75→10.49秒是该同一checkpoint/同材料局部条件下的真实计时，不能当作正式all_off配置的绝对耗时或加速承诺；生产配置对照另行测量。</p>'+table(['同材料实测模式','中位秒','峰值allocated GiB','实际梯度比较'],rows)+'<p>不是完整训练cycle、分数改善或正式执行准入。融合模式改变BF16矩阵形状，梯度非逐位一致；不会因计时更快就自动采用。历史D0/FPO完整cycle约2.07×/1.90×属于双节点独立实测。<a href="'+esc(directory/'baseline.json')+'">原始计时</a> · <a href="'+esc(directory/'COMPARISON_0ab.json')+'">当前比较收据</a></p></details>'


def D46_registered_evaluation_progress_fragment():
    rows=[]
    for name,node in read(D46_ROOT/'CAMPAIGN.json',{}).get('evaluations',{}).items():
        if name.startswith('EARLY_'):continue
        if node.get('status','').startswith('EQUIVALENT_REFERENCE'):
            rows.append((name,'同参引用 '+str(node.get('reference_case','D0_epoch0.5')),'不独立计分','独立jobs '+str(sum(len(v)for v in node.get('jobs',{}).values())),node['status']));continue
        world=node.get('world',32);done=active=waiting=failed=0;output=Path(node['output'])
        for rank in range(world):
            jobs=node.get('jobs',{}).get(str(rank),[])
            if not jobs:waiting+=1;continue
            job=Path(jobs[-1]['job']);result=read(job/'RESULT.json',{});running=read(job/'RUNNING.json',{})
            if result:
                if result.get('returncode')==0 and read(output/f'rank{rank}.json',{}).get('status')=='COMPLETE':done+=1
                elif result.get('returncode')==75:waiting+=1
                else:failed+=1
            elif running.get('child_pid'):active+=1
            else:waiting+=1
        try:
            with os.scandir(output/'scenes')as files:scenes=sum(1 for f in files if f.name.endswith('.json')and f.is_file())
        except OSError:scenes=0
        rows.append((name,str(scenes)+'/12146',str(done)+'/'+str(world),'已启动未返回 '+str(active)+'；排队/准入 '+str(waiting)+'；异常 '+str(failed),'评测待齐'if node.get('status')=='RUNNING'else node.get('status','待确认')))
    return '<h3>常规0.5E全评测：实际分片与引用</h3>'+table(['检查点','已写场景/引用','完成分片','当前worker收据','状态'],rows)+'<p>case标记RUNNING不等于此刻有GPU评分；完成片、未返回worker与排队分开，不发布partial均值。D1/D2等参别名不复制独立分数。</p>'if rows else ''


def D46_full_result_figures_fragment():
    path=D46_ROOT/'analysis/plots/D46_EARLY_FULL_FACTS.json';value=read(path,{})
    if value.get('schema')!='D46_early_full_plot_facts_v1'or value.get('status')!='COMPLETE'or not sealed_JSON(value):return ''
    evidence={v['path']:v['file_sha256']for v in value.get('evidence',[])}
    for row in value.get('rows',[]):
        for key in ('summary_file','paired_analysis_file'):
            p=Path(row[key])
            if not p.is_file()or hashlib.sha256(p.read_bytes()).hexdigest()!=evidence.get(str(p)):return ''
    figures=[]
    for name in ('D46_zero_repair_and_gain_loss.svg','D46_safety_and_progress_delta.svg'):
        item=value.get('figures',{}).get(name,{});p=Path(item.get('path',''))
        if not p.is_file()or hashlib.sha256(p.read_bytes()).hexdigest()!=item.get('file_sha256'):return ''
        figures.append(p)
    rows={r['arm']:r for r in value['rows']};d=rows['D0'];f=rows['D3_FPO'];a=rows['D3_AWM']
    body='<details><summary>四个完整点的零分修复、增损与安全/进度两图</summary>'
    body+='<p>固定 IL46≥.9 的同一 '+esc(value['baseline']['fixed_high_ge09_count'])+' 场景队列。D0修复 '+esc(d['zero_repaired'])+' 个原零分，但新增 '+esc(d['new_zero'])+'；该高分队列全局贡献 '+esc(f'{d["fixed_IL46_high_ge09"]["population_net_contribution_PDMS"]:+.6f}')+' PDMS，EP '+esc(f'{d["metric_delta_pp"]["ego_progress"]:+.3f}')+'pp，抵消部分收益。FPO曝光较少而分数接近，高分净损失 '+esc(f'{f["fixed_IL46_high_ge09"]["population_net_contribution_PDMS"]:+.6f}')+'；AWM保持较多原高分、EP '+esc(f'{a["metric_delta_pp"]["ego_progress"]:+.3f}')+'pp，但NC '+esc(f'{a["metric_delta_pp"]["no_at_fault_collisions"]:+.3f}')+'pp，不能概括所有安全均改善。</p>'
    for p in figures:body+='<figure style="margin:14px 0"><img loading="lazy" style="display:block;width:100%;height:auto" src="'+esc(p)+'" alt="D46四个完整NAVTEST点的配对诊断"></figure>'
    body+='<p>图只复用封存完整结果和既有44log CI；各点曝光不同，不能因果排序或据EP推断不可调和的物理权衡。高分贡献与gross损失口径不同，不把抵消量当成可实现收益；NAV标签不入训练。<a href="'+esc(path)+'">完整事实表与口径</a></p></details>'
    return body


def D46_execution_recovery_fragment():
    directory=D46_ROOT/'operations/EVAL_DIRECTORY_RECOVERY_20261103';path=directory/'RECOVERY_AUDIT.json'
    audit=read(path,{})
    if audit.get('status')!='PASS'or not sealed_JSON(audit):return ''
    body='<details><summary>执行故障与恢复边界：保存状态、资源等待不等于算法无效</summary><p>'+esc(audit.get('finding_zh',''))+' 原生审计确认Adam/scheduler1614和八rank RNG完整，原code1与日志保留；当前是否恢复只看实时任务表。<a href="'+esc(path)+'">原生恢复审计</a></p>'
    backfill=directory/'DIRECTORIES_AND_GATE_BACKFILL.json'
    if read(backfill,{}).get('scientific_source_unchanged')is True:body+='<p>缺失目录已补、.5E请求已按真实封存状态回填；不丢gate、不重置Adam/RNG。<a href="'+esc(backfill)+'">目录与门槛回填收据</a></p>'
    retry=directory/'D2_STARTUP_TIMEOUT_REVIEW.json';review=read(retry,{})
    if review.get('schema')=='d46_exact_failed_job_recovery_review_v1':
        body+='<p>D2的旧attempt002在原生恢复前发生Gloo启动90秒超时，0新更新；经一次显式复核从同1614、同科学源作暖缓存资源重试。不是数值失败；实际重试进度以实时表为准。<a href="'+esc(retry)+'">精确失败与重试审查</a></p>'
    return body+'<p>资源hold保留代码、结果与原生状态，不是无效结论。服务器列来自实际当前/最近job，不由首选路由推断独占；授权不等于空闲，未知owner不回收。</p></details>'


def legacy_I_resource_hold():
    directory=D46_ROOT/'operations/OLD_I_RESOURCE_HOLD_20261103'
    request=read(directory/'RESOURCE_PRIORITY_PAUSE.json',{});path=directory/'NATIVE_SAVE_EXIT0.json';saved=read(path,{})
    job=Path(saved.get('job',''));result_path=job/'RESULT.json';result=read(result_path,{})
    fence=Path(request.get('resume_fence',''));cp=Path(saved.get('checkpoint',''))
    if (request.get('parent_and_budget_unchanged')is not True or not fence.is_file()
            or saved.get('status')!='NATIVE_COOPERATIVE_SAVE_EXIT0'or saved.get('resource_hold_not_algorithm_termination')is not True
            or saved.get('rank_result_agreement')!=8 or saved.get('job_returncode')!=0 or result.get('returncode')!=0
            or not result_path.is_file()or hashlib.sha256(result_path.read_bytes()).hexdigest()!=saved.get('job_result_sha256')
            or read(cp/'COMPLETE',{})!=saved.get('seal')):return {}
    return dict(path=str(path),request=str(directory/'RESOURCE_PRIORITY_PAUSE.json'),job=str(job),
        updates=saved['seal']['updates'],inner=saved['seal']['next_inner'],budget=saved['original_minimum_updates_preserved'])


def AUG_complete_points():
    path=Path(__file__).with_name('complete_events.py')
    if not path.is_file():return []
    spec=importlib.util.spec_from_file_location('performance_complete_events_reader',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.completed_points(D46_ROOT)


def AUG_complete_results_fragment(historical_peak=91.45359756166451):
    points=AUG_complete_points()
    if not points:return '<p>新AUG尚无通过完整封印校验的NAVTEST分数；不发布partial均值或将资格记为提分。</p>'
    best=max(points,key=lambda r:r['PDMS'])
    label=best['experiment']+' / Adam '+str(best['updates'])
    body='<h3>AUG最新完整单轨迹结果</h3><p><b>当前AUG最高完整 PDMS '+esc(f'{best["PDMS"]:.8f}')+' · '+esc(label)+'</b>；相固定IL46 '+esc(f'{best["PDMS"]-89.4955209758546:+.6f}')+'点。<a href="'+esc(best['summary'])+'">完整评测与SHA</a> · <a href="'+esc(Path(best['checkpoint'])/'COMPLETE')+'">最高点封印检查点</a></p>'
    if best['PDMS']>historical_peak:
        body+='<p><b>超过本项目既有最高完整分 '+esc(f'{historical_peak:.8f}')+'，属于新已观测项目高点。</b>不是公开SOTA声明；IL46与历史IL70父策略不同，不能把跨研究差值归因单一改动。</p>'
    else:body+='<p>本项目既有最高完整分 '+esc(f'{historical_peak:.8f}')+'仍单列；与IL70不同父策略，AUG相同阶段的改善不等于超过该最高分。不称公开SOTA。</p>'
    rows=sorted(points,key=lambda r:Path(r['summary']).stat().st_mtime,reverse=True)
    body+=table(['最新完整点','实际Adam','PDMS','相固定IL46','场景 / invalid'],[
        (r['experiment'],r['updates'],f'{r["PDMS"]:.8f}',f'{r["PDMS"]-89.4955209758546:+.6f}','12146 / 0')for r in rows])
    return body


def D46_augustus_formal_fragment(plan,directory):
    launch_path=directory/'FORMAL_LAUNCHES.json';launches=read(launch_path,{})
    if not launches.get('receipts'):return '',0,0
    entries=[];actual=starting=slots=0
    for route in plan.get('routes',[]):
        exp=route['experiment'];output=Path(route.get('formal_output',str(D46_ROOT/'runs'/(exp+'_v1'))));identity=read(output/'identity.json',{})
        jobs=[r for r in launches['receipts']if str(output)in r.get('command',[])]
        live=failed=0;starts=[]
        for receipt in jobs:
            job=Path(receipt['job']);result=read(job/'RESULT.json',{});running=read(job/'RUNNING.json',{})
            if result:
                if result.get('returncode')!=0:failed+=1
            elif running.get('child_pid'):live+=1;starts.append(running.get('start_unix',float('inf')))
        rows=[tail(output/f'metrics_rank{rank}.jsonl')for rank in range(route['world'])]
        fresh=(identity.get('algorithm_schema')=='d46_augustus_native_v1'and identity.get('qualification')is False
            and identity.get('experiment_id')==exp and len(jobs)==len(route['hosts'])and live==len(jobs)
            and all(r.get('updates',0)>0 and r.get('run')==identity.get('sha256')and 0<=time.time()-r.get('at',0)<600
                    and r.get('at',0)>=min(starts,default=float('inf'))for r in rows))
        common=min(r.get('updates',0)for r in rows)if all(r.get('updates',0)>0 for r in rows)else None
        exposure=min(r.get('scene_exposures',0)for r in rows)if common else 0
        if fresh:actual+=1;slots+=route['world']
        elif live:starting+=1
        status='正式全rank实际训练'if fresh else'执行异常待复核'if failed else'正式已派发/初始化，等待全rank更新'if live else'正式worker已结束/状态待核对'
        cfg=identity.get('recipe',{})
        entries.append((exp+' / world'+str(route['world']),str(identity.get('scope',{}).get('trainable_numel','待核对')),
            '/'.join(str(cfg.get(k,'待核对'))for k in('global_scenes','group','inner_updates')),
            ' + '.join(route['hosts']),str(common)if common else'尚无全rank正式Adam',
            format(exposure,',')+' / '+format(exposure/103288,'.5f')+'E',status))
    body='<h3>当前AUG正式实际训练 '+esc(actual)+' 条 / '+esc(slots)+' 逻辑卡位；正式初始化 '+esc(starting)+' 条</h3>'
    body+=table(['正式路线','实际trainable参数','B/G/M','实际派发主机','全rank共同Adam','有效场景 / epoch','实际状态'],entries)
    body+='<p>真实训练必须同时有当前正式job未返回、全部rank新日志和相同run身份；初始化/资格不冒充Adam进展。正式两条均独立IL46/Adam0，目标6E，不能继承资格权重。<a href="'+esc(launch_path)+'">六node正式派发收据</a></p>'
    proof_path=directory/'GPU_NATIVE_RESUME_RNG_AUDIT.json';proof=read(proof_path,{})
    if proof.get('status')=='PASS':body+='<p>资格1→2的两路线全部EXIT0，48rank私有RNG原生接续审计PASS；这是恢复证据，不是GPU不中断参数对照或PDMS提升。<a href="'+esc(proof_path)+'">实际RNG恢复审计</a></p>'
    return body,actual,slots


def D46_augustus_regression_fragment():
    path=D46_ROOT/'analysis/AUGUSTUS_REGRESSION_20261103/DIAGNOSIS.json';v=read(path,{})
    if v.get('schema')!='d46_augustus_regression_evidence_v1'or not sealed_JSON(v):return ''
    body='<section style="background:#fff7ed;border:2px solid #d19042;border-radius:12px;padding:18px"><h3>最新：D46适配代码已修改 · 正式重训与退化原因分析</h3>'
    best=max(AUG_complete_points(),key=lambda p:p['PDMS'],default={})
    highest=(best['experiment']+' '+format(best['PDMS'],'.4f'))if best.get('PDMS',0)>90.32032057511778 else'FPO 90.3203'
    body+='<p><b>当前 D 系最高完整成绩：'+esc(highest)+'；同父 IL46 为 89.4955。历史 IL70/SafetyEP 91.4536 属于另一父模型。</b>修正收益仅以完整官方评测确认。</p>'
    current_directory=D46_ROOT/'operations/AUGUSTUS_ALIGNED_20261103'
    current_plan=read(current_directory/'PLAN.json',{})
    current_formal,_,_=D46_augustus_formal_fragment(current_plan,current_directory)
    body+=current_formal
    evaluation=read(D46_ROOT/'operations/AUGUSTUS_CAMPAIGN_20261103/EVAL_STATE.json',{})
    verified={(p['experiment'],p['updates']):p for p in AUG_complete_points()}
    cases=[c for c in evaluation.get('cases',{}).values()if c.get('binding',{}).get('experiment')in('AUG3_FULL_ALIGNED','AUG4_LORA_ALIGNED')]
    if cases:
        body+='<h4>修正版完整 B1 评测队列</h4>'
        entries=[]
        for c in sorted(cases,key=lambda c:({'RUNNING':0,'PENDING':1,'COMPLETE':2}.get(c.get('status'),3),c['binding']['scene_budget_clock'],c['id'])):
            b=c['binding'];point=verified.get((b['experiment'],b['updates']))
            status={'RUNNING':'评测中','PENDING':'已排队','COMPLETE':'完整已发布'}.get(c.get('status'),c.get('status','待核对'))
            entries.append((b['experiment']+'/'+str(b['updates']),f'{b["scene_budget_clock"]/103288:.4f}E',
                str(c.get('completed_shards',0))+'/32',c.get('active_shards',0),status,
                f'{point["PDMS"]:.6f}'if point else'等待完整结果'))
        body+=table(['checkpoint','有效场景曝光','完成分片','运行分片','状态','完整PDMS'],entries)
        body+='<p>分片与排队状态来自后台账本；PDMS仅在完整12,146场景、44物理log、invalid=0并通过封印后显示。后续checkpoint会自动登记，完整结果立即触发报告更新。</p>'
    latest=read(path.parent/'FIRST_ALIGNED_COMPLETE_READOUT.json',{})
    if latest.get('schema')=='d46_aligned_first_complete_readout_v1'and latest.get('status')=='COMPLETE'and sealed_JSON(latest):
        body+='<h4>最新完整结论：尚无净提分，LoRA基本持平，全参数仍小幅退化</h4>'
        body+=table(['修正版首点','PDMS','相同IL46变化','44log差值CI95','原零分修复','新增零分'],[
            (p['experiment']+'/'+str(p['updates']),f'{p["PDMS"]:.6f}',f'{p["delta_IL46"]:+.6f}',
             '['+', '.join(f'{x:+.4f}'for x in p['CI95_delta_PDMS'])+']',p['repaired_zero'],p['new_zero'])for p in latest['points']])
        body+='<p>两点均为相同6,400场景曝光（0.06196E）、完整12,146场景/44log/invalid0。FULL：EP约+1.00pp，TTC约−1.42pp，NC/DAC也下降；旧高分场景损失仍大于零分修复收益。LoRA相对IL46仅−0.0370分、CI跨零，基本持平。当前不能宣称修正已提分，也不能据首个早期点判定最终上限；继续评测FULL200/LoRA400的0.12393E点。</p>'
        body+='<p><b>新的训练过程诊断：</b>已检查各自前6,400个TRAIN场景的全部rollout材料。当前组std归一化下，完全安全组的优势绝对系数约占66%–67%；用原生batch std反事实重算约占16%–17%，安全混合组约占83%。这提示小EP差异可能被组std过度放大；这里只是系数占比，不是参数梯度或新算法收益。该归一化改动尚未启用，当前两组继续既定训练与评测。</p>'
        body+='<p><a href="'+esc(path.parent/'FIRST_ALIGNED_COMPLETE_READOUT.json')+'">首轮完整证据与配对结论</a> · <a href="'+esc(path.parent/'NORMALIZATION_FULL_FIRST6400_AUDIT.json')+'">全部前6400组的归一化诊断</a></p>'
    rows=[]
    for p in v['complete_points']:
        rows.append((p['arm'],p['updates'],f'{p["exposure"]:.4f}E',f'{p["PDMS"]:.4f}',f'{p["delta_IL46"]:+.4f}',p['paired']['repaired_zero'],p['paired']['new_zero']))
    body+=table(['旧版完整 NAVTEST','Adam','有效曝光','PDMS','相对同 IL46','零分修复','新增零分'],rows)
    body+='<p>每点均为 12,146 场景 / 44 物理 log / invalid=0 / 真实 B1 K10 shift5。FULL 的碰撞、越界和 TTC 同时恶化。LoRA 400 次的下降 CI 也低于零，不能把所有下降都解释成随机波动。</p>'
    body+=table(['相同IL46基线的收益/损失分解','全场景正收益 PDMS','全场景负损失 PDMS','固定原高分队列≥.9贡献','原零分队列贡献'],[
        (p['arm']+'/'+str(p['updates']),f'{100*p["gain_loss"]["G_plus"]:+.4f}',f'{100*p["gain_loss"]["G_minus"]:+.4f}',
         f'{100*p["gain_loss"]["fixed_reference_cohorts"]["IL46_high_ge_0.9"]["population_contribution"]:+.4f}',
         f'{100*p["gain_loss"]["fixed_reference_cohorts"]["IL46_zero"]["population_contribution"]:+.4f}')for p in v['complete_points']])
    body+='<p>固定原高分队列为同一 8,722 场景，原零分为479场景。FULL100 虽获得 +4.1083 的 gross 改善，却损失 -12.3307；其中原高分队列贡献 -7.7508，远大于原零分队列 +1.0072。修正不能只追求零分恢复，还必须控制原高分能力流失。表中贡献都按完整12,146人口折算，不能将子集均值直接相加。</p>'
    body+='<p><b>已确认的工程错误：</b>FULL404、LoRA807 都已封存 0.5E native，随后首次写评测请求时缺少目录而退出。已修复目录创建；这个错误解释训练中断，不能解释更早的掉分。</p>'
    body+='<p><b>优化问题：</b>旧训练三步线性网格与部署十步 shift5 不一致；M1 在求梯度时 ratio=1，clipping 不限制这次 Adam 位移；FULL 的 IL 参考 KL 已明显增大。TRAIN 还发现存在安全候选时，TTC0 轨迹仍得到正 canonical 优势。这些是已观察风险，尚未逐项隔离证明掉分原因。</p>'
    body+='<p><b>已用于正式重训的修改：</b>沿用官方 JaT 核心，改为原生 K10/shift5 的七步确定性前缀＋末三步探索；共享实际部署 seed2101 初始噪声；安全分层优势；参考 KL 系数 0.001→0.1；M1→M2；FULL LR 1e-6→2e-7。两条均从同 IL46 / 空 Adam 开始。相关CPU与真实多卡原生恢复检查已通过；是否提分仍待完整B1评测。</p>'
    proof=read(D46_ROOT/'operations/AUGUSTUS_ALIGNED_20261103/QUALIFICATION_PROOFS.json',{})
    if proof.get('status')=='PASS':body+='<p><b>真实资格已完成：</b>FULL32卡、LoRA16卡均通过第1次更新中途保存→同rollout第2次更新；48rank初始原生B1逐值一致，Adam/RNG/冻结参考检查通过。正式两组从原IL46与空Adam另行重训，资格权重未带入。资格通过不代表PDMS已提升。</p>'
    native=read(path.parent/'NONZERO_NATIVE_DEPLOYMENT_AUDIT.json',{})
    if native.get('status')=='PASS'and sealed_JSON(native):body+='<p><b>训练/部署实现排查：</b>非零FULL100和LoRA400的原生恢复actor与部署加载model，全部参数及buffer逐值一致（CPU状态检查，不冒充GPU前向或新分数）；缓存仅冻结视觉/输入embedding，文本/P/KV在更新时重建。尚未发现漏载权重或优势符号硬错误。</p>'
    calibration=read(path.parent/'TRAIN103K_NOISE_CALIBRATION.json',{})
    if calibration.get('status')=='CPU_CALIBRATED_NOT_IN_RUNNING_RECIPES'and sealed_JSON(calibration):
        body+='<h4>新适配发现：相同噪声系数不等于相同物理探索</h4><p>作者使用每时刻动作q01/q99，D46使用固定全局坐标范围。沿用同样系数，在原生末次转移会给各航点约0.596m纵向、0.375m横向创新噪声。基于全部103,288合法TRAIN GT的标定候选，0.5s首点约0.056m/0.007m，4s末点约0.430m/0.238m；IL坐标协议不改，采样与概率使用同一8×3协方差。模块和独立Gaussian检查已完成，但<b>尚未接入当前两组</b>，需要验证覆盖/优势信号/B1，不能直接认定已解决退化。</p>'
        body+='<p><b>还需同时处理输出精度：</b>当前BF16 action projection之后转float，仍保留量化。在代表性归一化x≈−0.8处，一档约0.13m，可能大于标定候选的近期扰动。真正FP32动作投影模块已开发并通过参数/Adam/RNG测试，<b>未启用</b>；须训练、reference、部署一致，并重新测u0数值基线。不能只把噪声调小就认定实现更好。</p>'
    body+='<p><a href="/mnt/project/VLA-Drive-d46-augustus-native-20261103/reports/d46_augustus_regression_20261103/ALGORITHM_IMPLEMENTATION_AUDIT.md">算法实现审查：已改代码、排除范围、B1目标与全零组尚存缺口</a></p>'
    diagnostic=read(path.parent/'TRAIN64_COMPARISON.json',{})
    if diagnostic.get('count')==64 and diagnostic.get('NAVTEST')is False:
        body+='<h4>补充的固定 TRAIN64 对照：不能只归因于网格</h4>'
        body+=table(['同64个已训练场景，非NAVTEST','原生K10 PDMS','旧三步K3 PDMS'],[(k,f'{m["K10"]:.4f}',f'{m["K3"]:.4f}')for k,m in diagnostic['means'].items()])
        body+='<p>FULL100 的两种网格都退化，且 TTC 98.44%→82.81%、EP 81.85%→88.23%，表明仅改网格不足以解释或解决问题。LoRA400 的 TRAIN K10 改善而完整 NAVTEST 下降；该64场景已参与训练，不能作为泛化或新提分证据。更强约束与安全信用是修正的重点，组合有效性仍待完整评测。</p>'
    plan=read(D46_ROOT/'operations/AUGUSTUS_ALIGNED_20261103/PLAN.json',{});entries=[]
    for r in plan.get('routes',[]):
        q=Path(r['output']);jobs=list((D46_ROOT/'jobs/AUGUSTUS_ALIGNED_20261103'/r['experiment']).glob('*/node*/LAUNCH.json'))
        newest={}
        for j in jobs:
            receipt=read(j,{})
            host=receipt.get('host')
            if host and receipt.get('at',0)>newest.get(host,(0,None))[0]:newest[host]=(receipt.get('at',0),j)
        jobs=[item[1]for item in newest.values()]
        live=fail=done=0
        for j in jobs:
            result=read(j.parent/'RESULT.json',{});running=read(j.parent/'RUNNING.json',{})
            if result:done+=int(result.get('returncode')==0);fail+=int(result.get('returncode')!=0)
            elif running.get('child_pid'):live+=1
        native=read(q/'latest.json',{})
        entries.append((r['experiment'],r['world'],'128/8/2'if 'FULL'in r['experiment']else'64/8/2',
            f'{live} 活跃 worker / {done} EXIT0 / {fail} 异常',native.get('updates','尚无已封存更新'),
            '异常待修复'if fail else'真实资格/正式派发，见收据'if live else'等待派发或阶段衔接'))
    body+=table(['修正版','GPU计划','B/G/M','实际任务收据','资格 native Adam','当前状态'],entries)
    body+='<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:12px">'
    for name in ('deployment_regression.svg','trust_and_zero_regression.svg'):
        image=path.parent/name
        if image.is_file():body+='<figure style="margin:8px 0"><img style="width:100%;height:auto" src="'+esc(image)+'" alt="完整部署退化与训练漂移"></figure>'
    body+='</div><p><a href="'+esc(path)+'">完整证据与统计</a> · <a href="'+esc(D46_ROOT/'operations/AUGUSTUS_ALIGNED_20261103/PLAN.json')+'">修正版科学配置与资源计划</a></p></section>'
    return body


def D46_augustus_qualification_fragment(historical_peak=91.45359756166451):
    directory=D46_ROOT/'operations/AUGUSTUS_CAMPAIGN_20261103'
    plan=read(directory/'QUALIFICATION_PLAN.json',{});launches=read(directory/'QUALIFICATION_LAUNCHES.json',{})
    if plan.get('maximum_simultaneous_routes')!=2 or not plan.get('routes'):return '',0,0
    entries=[];active_routes=0;active_workers=0;uuids=set()
    for route in plan['routes']:
        if route.get('qualification')is not True:continue
        cfg=Path(route['config']);fields={}
        if cfg.is_file()and hashlib.sha256(cfg.read_bytes()).hexdigest()==route.get('config_sha256'):
            for line in cfg.read_text().splitlines():
                key,sep,v=line.partition(':')
                if sep and key in('global_scenes','group','inner_updates'):fields[key]=v.strip()
        jobs=[r for r in launches.get('receipts',[])if route['output']in r.get('command',[])]
        running=done=failed=0
        for job in jobs:
            path=Path(job['job']);result=read(path/'RESULT.json',{});worker=read(path/'RUNNING.json',{})
            if result:
                if result.get('returncode')==0:done+=1
                else:failed+=1
            elif worker.get('child_pid'):
                running+=1;uuids.update(job.get('expected_gpu_uuids',[]))
        active_workers+=running;active_routes+=int(running>0)
        rows=[tail(Path(route['output'])/f'metrics_rank{rank}.jsonl')for rank in range(route['world'])]
        common=min(r.get('updates',0)for r in rows)if all(r.get('updates',0)>0 for r in rows)else None
        progress='全rank实际Adam '+str(common)if common else'原生模型加载/初始化，等待全rank首更新'
        status='异常待复核'if failed else'初段结束，等待自身恢复资格'if jobs and done==len(jobs)else'资格进行中'if running else'资格已派发/等待worker'
        method='全28层dense text+norm+动作/时间/P/ego/role；ViT与embedding冻结'if route['experiment']=='AUG1_FULL'else'attention LoRA r16+last8 FiLM与动作I/O；冻结其余base'
        entries.append((route['experiment']+' / world'+str(route['world']),
            '/'.join(fields.get(k,'待核对')for k in('global_scenes','group','inner_updates')),method,
            ' + '.join(route['hosts']),progress,f'{running} 已启动未返回 / {done} EXIT0 / {failed}异常',status))
    formal,formal_actual,formal_slots=D46_augustus_formal_fragment(plan,directory)
    diagnosis=D46_augustus_regression_fragment()
    if diagnosis:
        current_directory=D46_ROOT/'operations/AUGUSTUS_ALIGNED_20261103'
        _,aligned_actual,aligned_slots=D46_augustus_formal_fragment(read(current_directory/'PLAN.json',{}),current_directory)
        formal_actual+=aligned_actual;formal_slots+=aligned_slots
    body=diagnosis+('<details><summary>旧版AUG正式任务，已退出并保留</summary>'+formal+'</details>'if diagnosis else formal)+AUG_complete_results_fragment(historical_peak)+('<details><summary>旧版原生资格与正式起点分开核对</summary>'if formal else'')
    body+='<h3>AUGUstus 当前资格任务：'+esc(active_routes)+' 组，'+esc(active_workers)+' 个CUDA子进程worker已启动，覆盖 '+esc(len(uuids))+' 个物理GPU UUID</h3>'
    body+='<p><b>此表仅资格，不是正式训练。</b>两条均从同IL46/Adam0起步，先1次真实Adam再自身原生恢复核对；正式阶段另从原IL46和新空Adam开始，绝不继承资格权重。资格设备、模型加载、真实更新与正式计数分开；最大同时两路线。</p>'
    body+=table(['路线/资格world','实际配置B/G/M','训练范围','已派发主机','真实共同进度','逐node收据','当前阶段'],entries)
    body+='<p>科学源 augustus_science_v1@78cea8f，运营源 operations_augustus_v1@81c997e；FULL B128、LORA B64也有batch差异，不能把之后成绩直接解释为纯scope消融。<a href="'+esc(directory/'QUALIFICATION_PLAN.json')+'">资格计划/配置SHA</a> · <a href="'+esc(directory/'QUALIFICATION_LAUNCHES.json')+'">实际六node派发收据</a></p>'
    return body+('</details>'if formal else''),formal_actual,formal_slots


def D46_dedicated_execution_fragment():
    path=D46_ROOT/'analysis/DEDICATED_TWO_NODE_EXECUTION_REVIEW_20261103.json';v=read(path,{})
    if v.get('schema')!='d46_same_native_world16_full_cycle_execution_benchmark_v1'or v.get('status')!='COMPLETE'or not sealed_JSON(v):return ''
    a=v['baseline'];b=v['candidate']
    return '<p><b>两节点真实完整cycle优化资格已完成：'+esc(f'{a["steady_seconds_per_collection"]:.3f}')+'→'+esc(f'{b["steady_seconds_per_collection"]:.3f}')+'秒，'+esc(f'{v["full_cycle_speedup"]:.5f}')+'×</b>（匹配cycle '+esc(f'{v["matched_full_cycle_speedup"]:.5f}')+'×）。同1500原生起点、实际采样/评分/2Adam/控制/I/O；fast的1511 pending→1512原生恢复通过。短资格不是长期吞吐承诺或PDMS提升；融合改变BF16运算分组，不宣称跨执行最终参数逐位一致。<a href="'+esc(path)+'">完整cycle与恢复收据</a>。旧15.75→10.49秒四模式均开启checkpoint，仅作历史微基准，不能当production all_off耗时。</p>'


def D46_priority_fragment(other_active=None,historical_peak=91.45359756166451):
    """A status receipt is required; a new directory or config proves no training."""
    path=D46_ROOT/'PRIORITY_STATUS.json';value=read(path,{})
    augustus_path=D46_ROOT/'operations/AUGUSTUS_PRIORITY_ALL_RESOURCES_20261103/PRIORITY_CHANGE_REQUEST.json'
    augustus=read(augustus_path,{})
    augustus_priority=augustus.get('maximum_concurrent_training')==2 and augustus.get('old_results_and_native_state_retained')is True
    aug_actual=aug_slots=0
    body='<section id="d46-priority" style="background:#edf7f5;border:2px solid #428b7a;border-radius:14px;padding:24px;margin:20px 0"><small>当前研究优先级</small><h2>'+('AUGUstus 主线 · IL46k fresh RL 资格与正式状态'if augustus_priority else'D 框架 · IL46k 父策略优先 RL')+'</h2>'
    if augustus_priority:
        qualification,aug_actual,aug_slots=D46_augustus_qualification_fragment(historical_peak)
        formal_receipts=read(D46_ROOT/'operations/AUGUSTUS_CAMPAIGN_20261103/FORMAL_LAUNCHES.json',{})
        phase=('两条AUG均已正式派发；真实更新与初始化按全rank新日志区分，见最前表。'if formal_receipts.get('receipts')else'AUG1_FULL/world32 与 AUG2_LORA/world16已实际派发资格，尚非正式训练。')if qualification else'候选FULL/LORA尚待实际派发收据，不据计划称已训练。'
        if (D46_ROOT/'analysis/AUGUSTUS_REGRESSION_20261103/DIAGNOSIS.json').is_file():phase='旧版AUG已退出保留；当前修正版、真实进度和退化分析见下方最前面板。'
        body+='<p style="background:#e2eefc;padding:12px"><b>用户新优先级：所有当前已授权且可用的RL资源集中到AUGUstus主线，最多两条正式训练。</b>'+esc(phase)+'旧D0/D2/D5按原生边界协作保存，旧结果、完整状态和预算保留，不判方法无效。<a href="'+esc(augustus_path)+'">实际优先级调整请求</a></p>'+qualification+D46_dedicated_execution_fragment()+'<p>旧D0/FPO是本地D原生条件Gaussian/energy代理适配，不能称官方AUGUstus/JaT完整复现。官方默认LoRA关闭、文本主干可训练、RL冻结视觉；本地旧D0仅attention-LoRA约642万参数，且sum24/W3与官方mean-coordinate/lambda_t/3步不同。</p>'
        scope_path=Path(__file__).resolve().parents[2]/'reports/d46_augustus_scope_audit_20261103/SCOPE_AUDIT.json'
        if scope_path.is_file():body+='<p><a href="'+esc(scope_path)+'">官方scope与本地精确路径只读审计</a>：FULL建议dense text layers/FFN/norm与动作/时间接口，LORA保同接口作范围对照；当前缓存embedding合同与永久参考参数别名必须处理，不能直接把旧共享base解冻。该建议未部署，不含性能提升声明。</p>'
    if not value:
        return body+'<p>等待 D46 正式状态收据；尚未确认启动。既有 IL70 路线保存让位属于研究优先级调整，不判算法无效。</p></section>'
    stamp=value.get('updated_at',value.get('at',value.get('generated_UTC','未记录')))
    if isinstance(stamp,(int,float)):stamp=datetime.fromtimestamp(stamp,timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    status=value.get('status','状态待核对')
    label={'PREPARING_EXACT_IL46_POLICY_PARENT_AND_FRESH_RUNS':'准备固定 IL46k 策略与独立 fresh RL，尚未启动',
        'D46_FRESH_IL46_PRIORITY_CAMPAIGN':'D46 六路线已登记，资格与正式训练按各行实际状态区分'}.get(status,status)
    body+='<p><b>最新状态：'+esc(label)+'</b>；收据时间 '+esc(stamp)+'。<a href="'+esc(path)+'">实际优先级与执行状态</a></p>'
    count=value.get('actual_training')
    if isinstance(count,int)and count>=0:body+='<p><b>旧D46六路线实际正式训练数量：'+esc(count)+' 条</b>（资格与新AUG不计入此旧路线数；AUG实际 '+esc(aug_actual)+' 条）。</p>'
    allocation=sum(r.get('world',8)for r in value.get('runs',[])if isinstance(r,dict)and r.get('status')in('TRAINING','TRAINING_WORLD16'))
    body+='<p>当前逻辑训练rank/卡位合计 '+esc(allocation+aug_slots)+'（含新AUG；不等于独占GPU利用率）。</p>'
    if isinstance(count,int)and isinstance(other_active,int):body+='<p><b>当前实际 RL 训练合计 '+esc(count+other_active+aug_actual)+' 条：AUG '+esc(aug_actual)+'，旧D46 '+esc(count)+'，既有 IL70 系 '+esc(other_active)+'。</b>按各路线world的实际任务与全rank新日志，不按注册数计。</p>'
    body+=D46_current_interruption_fragment(value)
    dedicated_path=D46_ROOT/'operations/DEDICATED_TWO_NODE_OPTIMIZATION_20261103/REQUEST.json'
    dedicated=read(dedicated_path,{})
    if not augustus_priority and dedicated.get('automatic_capacity_resume_disabled')is True and dedicated.get('all_original_training_budgets_preserved')is True:
        body+='<p style="background:#e2eefc;padding:12px"><b>当前优先腾出两节点，专做真实执行优化资格后再部署。</b>'+esc(' + '.join(dedicated.get('hosts',[])))+'；FPO/D1协作保存，AWM保留自身状态，原训练预算均保留，不判算法无效；自动capacity恢复已关闭。D0/D2/D5继续，实际数量与保存进度见本表。计划比较同1500权重的baseline/fast真实12次Adam及fast中途恢复，尚无该完整资格结果。原15.75→10.49秒微基准开启checkpoint，不能代表正式all_off配置。<a href="'+esc(dedicated_path)+'">用户要求与两节点资格安排</a></p>'
    hold=legacy_I_resource_hold()
    if hold:body+='<p>旧 IL70 综合 I 已协作完整保存于 '+esc(hold['updates'])+' / inner'+esc(hold['inner'])+'，原job EXIT0，现临时资源保留，为D0让位；原 '+esc(hold['budget'])+' 次实际更新承诺保留，未终止算法、未改父模型/Adam/RNG。<a href="'+esc(hold['path'])+'">实际保存退出收据</a> · <a href="'+esc(hold['request'])+'">资源保留与原预算</a></p>'
    additional=D46_ROOT/'operations/ADD_TSH4_20261103/AUTHORIZATION.json';grant=read(additional,{})
    if 'training-vlawm-tsinghua-zt4'in grant.get('unbounded_temporary_hosts',[]):body+='<p>新增TSH4八卡已获D46授权，供FPO双节点按实际准入使用；授权不代表空闲或独占，受保护IL不受影响。<a href="'+esc(additional)+'">新增主机不可变授权</a></p>'
    transition_path=D46_ROOT/'operations/CONTINUOUS_EVAL_20261103/OWNER_TRANSITION.json'
    policy_path=D46_ROOT/'operations/EXECUTION_POLICY.json'
    transition=read(transition_path,{});policy=read(policy_path,{})
    if (policy.get('schema')=='D46_evaluation_execution_policy_v1'and transition.get('control')==policy
            and transition.get('CPU_owner_only_signal')is True and transition.get('GPU_processes_signalled')==[]
            and policy.get('continue_training_during_evaluation')is True):
        operations=read(D46_ROOT/'CAMPAIGN.json',{}).get('evaluation_operations',{})
        deferred=operations.get('registered_dispatch_deferred')
        state='当前常规 gate 派发暂缓，早评继续'if deferred is True else'当前未暂缓常规 gate 派发'if deferred is False else'当前派发状态以控制收据为准'
        body+='<p><b>执行策略允许评测与训练并行。</b>该次CPU运营接管未改变GPU job、科学源或Adam/RNG；早评运行期间只暂缓重复的常规gate派发，不丢gate。此后资源hold、执行恢复及实际在训数量以实时表为准。'+esc(state)+'。<a href="'+esc(transition_path)+'">当次接管收据</a> · <a href="'+esc(policy_path)+'">执行策略</a></p>'
    restored_path=D46_ROOT/'operations/RESTORE_SIX_UNBOUNDED_20261103/APPLIED.json'
    restored=read(restored_path,{})
    if restored.get('finite_deadline_applies') is False:
        body+='<p><b>当前资源：六机已对 D46 重新开放，无到期截止。</b>'+esc(' / '.join(restored.get('hosts',[])))+'。<a href="'+esc(restored_path)+'">新授权实际应用</a>。下方原 IL70 的00:00到期和保存退出记录保留为历史，不代表本 D46 新授权已到期；其它任务权限不扩张。</p>'
    parent=value.get('parent',{})
    if parent.get('PDMS') is not None:
        body+='<p>固定 IL46k 父策略完整 NAVTEST：'+esc(f'{parent["PDMS"]:.8f}')+'。这是 IL 起点，不是新 RL 收益。'
        if parent.get('export_sha256'):body+='部署策略 SHA '+esc(parent['export_sha256'])+'。'
        body+='</p>'
    rows=value.get('runs',[])
    if isinstance(rows,dict):rows=[dict(v,id=k)for k,v in rows.items()]
    methods={
        'D3_FPO':('MC4 CFM-energy ratio PPO代理','非精确轨迹似然；clip .01，保留原生KL约束'),
        'D3_AWM':('正优势加权 matching + 参考保持','保守本地适配；非完整 signed AWM，matching不使用PPO clip'),
        'D5':('扩大 D 原生语义/动作适配范围','text LoRA r32 + last8 FiLM低秩 + action I/O/position'),
        'D1':('两阶段语义/终点质量信用','2 exposure epochs后引入Comfort与TRAIN GT endpoint质量；标签不进当前观测'),
        'D2':('真实 B1 对照的 verified teacher 吸收','0.5 exposure epoch后启用；严格安全且较B1高≥.02，clean-x +真实ODE-last'),
        'D0':('D 原生 joint24 Gaussian PPO 基线','clip .1 +自适应 native KL；保存状态条件转移密度，非最终轨迹边缘密度')}
    campaign=read(D46_ROOT/'CAMPAIGN.json',{}).get('runs',{})
    if rows:
        entries=[]
        states={'RUNNING_QUALIFICATION':'资格阶段，非正式训练','WAITING_RESOURCE':'等待资源',
            'WAITING_FOR_AUTHORIZED_CAPACITY':'等待有权限资源','RUNNING_OR_ADMITTING':'正式已派发/初始化，待八rank更新',
            'SAVED_AND_HELD_FOR_REGISTERED_STAGE':'已完整保存，资源保留/等待（非无效结论）',
            'FAILED_REQUIRES_REVIEW':'执行中断待复核/恢复，当前不计在训',
            'STOPPED_AT_NATIVE_BOUNDARY':'已原生边界保存，资源让位优化资格；预算保留',
            'TRAINING':'正式实际训练','TRAINING_WORLD16':'正式16rank实际训练','RUNNING':'运行中（以实际日志核对）'}
        for r in rows:
            arm=r.get('id','').removeprefix('D46_');registered=campaign.get(arm,{})
            history=registered.get('qualification_jobs',[])if r.get('status')=='RUNNING_QUALIFICATION'else registered.get('jobs',[])
            host=r.get('host')or(history[-1].get('host')if history else None)or'待分配'
            method,gate=methods.get(arm,('以登记配方为准',''))
            qualification=D46_qualification_status(registered)if registered.get('qualification')else'等待登记'
            exposure=r.get('scene_exposures');progress=str(r['updates'])if r.get('updates')is not None else'尚无实际日志'
            if isinstance(exposure,(int,float)):progress+=' / '+format(exposure,',')+'场景 / '+f'{exposure/103288:.5f}E'
            entries.append((r.get('id','')+' / world'+str(r.get('world',8)),method,gate,qualification,host,progress,states.get(r.get('status'),r.get('status','待确认'))))
        body+=table(['D46 路线','方法','阶段启用与边界','独立资格','实际当前/最近服务器','实时 Adam / 有效曝光 / epoch','正式阶段状态'],entries)
    body+=D46_registered_evaluation_progress_fragment()+D46_early_evaluation_fragment()+D46_full_result_figures_fragment()
    body+=D46_formal_world16_fragment()+D46_FPO_world16_fragment()+D46_execution_benchmark_fragment()
    body+=D46_execution_recovery_fragment()+D46_optimization_health_fragment()+D46_throughput_fragment()
    reviewed=[]
    for arm,registered in campaign.items():
        review=registered.get('qualification_resource_retry_review',{});review_path=Path(review.get('receipt',''))
        if not review_path.is_file():continue
        if hashlib.sha256(review_path.read_bytes()).hexdigest()!=review.get('receipt_sha256'):continue
        evidence=read(review_path,{})
        reviewed.append((arm,evidence.get('original_returncode'),evidence.get('reason',''),str(review_path)))
    if reviewed:
        body+='<details><summary>已复核的资格启动资源故障（保留原失败，不作为算法负结果）</summary><p>FPO/AWM 首次共用 WM4 时未预留冷加载显存，竞争触发 OOM；原 RESULT1 保留。仅调度准入升级：启动预留至当前八rank均有更新、冷加载需48GiB；科学源仍 formal_v1。各自从原资格 native0、保留其 Adam0/RNG 作一次显式资源重试，资格权重不进入正式训练。</p>'
        for arm,code,reason,path in reviewed:body+='<p>'+esc(arm)+' / 原RESULT '+esc(code)+'：'+esc(reason)+' <a href="'+esc(path)+'">精确复核收据</a></p>'
        body+='</details>'
    body+='<p>六条专用路线优先顺序为 D3_FPO / D3_AWM / D5 / D1 / D2 / D0。共同 B64 / G16 / inner2；各自先完成真实八卡10更新资格，再从相同完整 IL46 部署策略与新空 Adam 独立正式起步，绝不继承资格权重或 D44 RL 状态。它们适配 D 原生 Qwen 架构，不能写成把原 IL70 Safety 配方换父权重。<a href="'+esc(D46_ROOT/'CAMPAIGN.json')+'">六路线登记与当前阶段</a> · <a href="'+esc(D46_ROOT/'PARENT_CONTRACT_v2.json')+'">父策略合同v2</a></p>'
    saved_path=D46_ROOT/'operations/USER_PRIORITY_20261103/D44_ACTUAL_NATIVE_SAVED.json'
    saved=read(saved_path,{})
    if saved.get('old_campaign_closed') is True:
        body+='<p>旧 D44 已正常保存并退出：'+esc('；'.join(str(r['arm'])+' / '+str(r['updates'])+' / inner'+str(r['next_inner'])for r in saved.get('saved_runs',[])if r.get('returncode')==0))+'。<a href="'+esc(saved_path)+'">实际原生保存收据</a>；状态未转入 D46，不判算法无效。</p>'
    body+='<p>D4 / D6 依赖的原生 readout 缺失，仍暂缓。保留的部署策略和零差导出核对不等于原完整 IL native 仍存在。模型初始化、资格、已派发与实际正式八 rank 更新分开记录；未完整评测不发布新 RL 分数。既有 IL70 代码、结果与保存状态保留，资源让位不判方法无效。</p></section>'
    return body


def reviewed_I_fragment(root):
    """Consume only the complete sealed population audit and its exact fragment."""
    directory=Path(root)/'analysis/INTEGRATED_I_u002000_FULL_REVIEW'
    handoff=read(directory/'HTML_HANDOFF.json',{});summary=read(directory/'SUMMARY.json',{})
    body=dict(summary);digest=body.pop('sha256',None)
    if (summary.get('status')!='COMPLETE' or summary.get('population',{}).get('scenes')!=12146
            or summary.get('population',{}).get('physical_logs')!=44
            or hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()!=digest
            or handoff.get('summary_sha256')!=digest):return ''
    try:raw=(directory/'EMBED_SUMMARY.html').read_bytes()
    except OSError:return ''
    if hashlib.sha256(raw).hexdigest()!=handoff.get('fragment_sha256'):return ''
    text=raw.decode();start=text.find('<figure');end=text.rfind('</section>')
    if start>=0 and end>start:
        text=text[:start]+'<details><summary>查看完整分数分布、逐场景转移、增损及物理 log 三张图</summary>'+text[start:end]+'</details>'+text[end:]
    return '<style>.integrated-i2000-review svg{max-width:100%;height:auto}</style>'+text


def tail(path):
    try:
        with Path(path).open('rb') as f:
            f.seek(max(0,Path(path).stat().st_size-2**20))
            rows=f.read().splitlines()
        for row in reversed(rows):
            try:return json.loads(row)
            except ValueError:pass
    except OSError:pass
    return {}


def journal_rows(path, limit=1024, max_bytes=2**21):
    """Bounded metadata read; discard a possibly partial last JSON line."""
    try:
        with Path(path).open('rb') as f:
            start=max(0,Path(path).stat().st_size-max_bytes);f.seek(start)
            lines=f.read().splitlines()
        if start:lines=lines[1:]
        rows=[]
        for line in lines[-limit:]:
            try:
                row=json.loads(line)
                if isinstance(row,dict):rows.append(row)
            except ValueError:pass
        return rows
    except OSError:return []


def registered_plans(root, registry):
    """Read JSON identity/projection only; never import YAML, Torch or runtime."""
    projection=read(root/'reporting/registered_plans.json',{}).get('runs',{})
    plans={}
    for run in registry.get('runs',[]):
        arm=run['id'];identity_path=Path(run['root'])/'identity.json'
        identity=read(identity_path,{})
        cfg=identity.get('recipe',{})
        if cfg.get('research',{}).get('arm')!=arm:
            item=projection.get(arm,{})
            try:actual=hashlib.sha256(Path(run['config']).read_bytes()).hexdigest()
            except OSError:actual=None
            cfg=item.get('recipe',{}) if actual==run.get('config_sha256')==item.get('config_sha256') else {}
        spec=cfg.get('research',{}).get('performance')
        if spec:plans[arm]=dict(spec=spec,config=run['config'],config_sha256=run.get('config_sha256'),identity=str(identity_path) if identity else None)
    return plans


def verified_quality_evidence(root):
    """Compact sealed qualification metadata; never load native tensors."""
    directory=root/'operations/S6_NATIVE_QUALIFICATION_LANE_V2'
    receipt=read(directory/'COMPLETE_RECEIPT.json',{})
    proof_path=root/'qualification/S6_VERIFIED_QUALITY_NATIVE_V2/COMPLETE.json'
    gradient_path=directory/'GRADIENT_READOUT.json'
    proof=read(proof_path,{});gradient=read(gradient_path,{})
    try:
        proof_sha=hashlib.sha256(proof_path.read_bytes()).hexdigest()
        gradient_sha=hashlib.sha256(gradient_path.read_bytes()).hexdigest()
    except OSError:return {}
    if (receipt.get('status')!='PASS_QUALIFICATION_ONLY' or proof.get('status')!='PASS'
            or proof.get('schema')!='world_verified_quality_actual_world8_native_qualification_v1'
            or (proof.get('world'),proof.get('B'),proof.get('G'),proof.get('M'))!=(8,64,16,4)
            or not proof.get('native_bitexact_at_inner2_and4')
            or receipt.get('complete_sha256')!=proof_sha or receipt.get('gradient_readout_sha256')!=gradient_sha
            or gradient.get('qualification_complete_sha256')!=proof_sha
            or gradient.get('status')!='COMPLETE_ACTUAL_ALL8_RANKS_FOUR_UPDATES_NOT_PERFORMANCE'):
        return {}
    rank0=next((r for r in gradient.get('final_rank_readouts',[]) if r.get('rank')==0),{})
    return dict(proof=str(proof_path),proof_sha256=proof_sha,gradient=str(gradient_path),
        gradient_sha256=gradient_sha,source_commit=receipt.get('source_commit'),
        parameter_count=proof.get('trainable_numel'),teacher_classes=proof.get('actual_first_collection_teacher_classes',{}),
        summary=gradient.get('summary',{}),rank0=rank0)


def train_GT_evidence(root):
    """Only compact aggregate fields from the completed TRAIN-only CPU audit."""
    path=root/'analysis/VERIFIED_TRAIN_GT_TEACHER_AUDIT/RESEARCH_READOUT.json'
    value=read(path,{})
    if value.get('status')!='COMPLETE_DISCOVERY_ONLY_NOT_TRAINING':return {}
    population=value.get('population',{});budget=value.get('query_budget',{})
    if (population.get('input_count')!=2048 or population.get('valid_whole_trajectory_count')!=2035
            or population.get('NAVTEST_trajectory_or_label_read') is not False
            or budget.get('training_updates')!=0 or budget.get('GPU_model_forwards')!=0):return {}
    cohorts=read(path.with_name('SUMMARY.json'),{}).get('cohort_statistics',{})
    return dict(path=str(path),file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        population={k:population.get(k) for k in ('input_count','valid_whole_trajectory_count','invalid_time_count',
            'physical_log_count','source_TRAIN_count','old_DEV_in_existing_NAVTRAIN_count')},
        budget={k:budget.get(k) for k in ('actual','training_updates','GPU_model_forwards')},
        opportunity=value.get('verified_opportunity',{}),roundtrip=value.get('normalization_roundtrip',{}),
        GT_worse_all=cohorts.get('all',{}).get('GT_score_worse_lt_minus005'),
        GT_worse_strict_high=cohorts.get('B1_strict_high',{}).get('GT_score_worse_lt_minus005'),
        GT_comfort_not_one=population['valid_whole_trajectory_count']-value.get('quality',{}).get('comfort',{}).get('equal_one',0),
        summary=value.get('summary_zh',[]))


def train19_evidence(root):
    path=root/'analysis/IL70_TRAIN19_G16_G32_EXPLORATION/RESEARCH_REVIEW.json'
    value=read(path,{})
    if (value.get('status')!='COMPLETE_CPU_REVIEW_OF_REAL_GPU_PROBE'
            or value.get('protocol',{}).get('Adam_updates')!=0
            or value.get('actual_budget',{}).get('complete_groups')!=38):return {}
    return dict(path=str(path),file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        cohorts=value.get('cohorts',{}),budget=value.get('actual_budget',{}))


def S6_live_evidence(root):
    path=root/'analysis/S6_LIVE_ACTIVATION_REVIEW.json';value=read(path,{})
    if (value.get('schema')!='world_S6_live_activation_research_review_v1'
            or not value.get('status','').startswith(('INITIAL_PREFIX_','COMPLETE'))
            or value.get('rank_count')!=8 or value.get('common_updates',0)<1):return {}
    cases=[c for c in value.get('allzero_PG0_cases',[]) if c.get('rawA_PG0')]
    gradients=value.get('gradient_diagnostics',{})
    return dict(path=str(path),file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        status=value['status'],analysis_at=value.get('analysis_at'),common_updates=value['common_updates'],
        rank_count=value['rank_count'],rank_update_rows=value.get('rank_update_rows'),
        quality=value.get('window_quality_counts',{}),gates=value.get('teacher_gates',{}),
        alpha=value.get('alpha_budget',{}),queries=value.get('query_cost',{}),
        PG0_cases_observed=len(cases),PG0_B1_full_score=sum(c.get('nominal_score')==1. for c in cases),
        local_components=gradients.get('conditional_teacher_slot_component_norms',{}),
        DDP_total=gradients.get('actual_DDP_reduced_total_gradient_norm',{}),
        combined_aux_gradient_norm=gradients.get('combined_aux_gradient_norm'))


def first_window_progress_evidence(root):
    GT_path=root/'analysis/S6_FIRST_WINDOW_GT_FALLBACK_AUDIT/RESEARCH_READOUT.json'
    synthetic_path=root/'analysis/S6_FIRST_WINDOW_SYNTHETIC_PROGRESS_AUDIT/SUMMARY.json'
    GT=read(GT_path,{});synthetic=read(synthetic_path,{})
    complete=read(synthetic_path.with_name('COMPLETE.json'),{})
    if (GT.get('status')!='COMPLETE_CPU_DISCOVERY_ONLY'
            or synthetic.get('status')!='COMPLETE_CPU_SYNTHETIC_DIAGNOSTIC_ONLY'
            or synthetic.get('schema')!='world_S6_TRAIN7_explicit_synthetic_progress_teacher_diagnostic_v1'
            or synthetic.get('no_training_or_native_mutation') is not True
            or complete.get('status')!='COMPLETE'):return {}
    digest=hashlib.sha256(synthetic_path.read_bytes()).hexdigest()
    if complete.get('files_sha256',{}).get('SUMMARY.json')!=digest:return {}
    return dict(GT_path=str(GT_path),synthetic_path=str(synthetic_path),
        GT_file_sha256=hashlib.sha256(GT_path.read_bytes()).hexdigest(),synthetic_file_sha256=digest,
        fallback={k:v for k,v in GT.get('fallback7',{}).items() if k!='cases'},
        GT_budget=GT.get('budget',{}),GT_fallback=GT.get('cohort_statistics',{}).get('fallback7',{}),
        synthetic_cohorts=synthetic.get('cohorts',{}),synthetic_queries=complete.get('actual_CPU_PDM_queries'))


def FAST_review_evidence(root,complete_results):
    directory=root/'analysis/FAST_u000808_FULL_REVIEW';path=directory/'SUMMARY.json'
    value=read(path,{});body=dict(value);seal=body.pop('sha256',None)
    population=value.get('population',{})
    official=next((r for r in complete_results if r['arm']=='S0_FAST_PACK4' and r['updates']==808),None)
    if (not official or value.get('status')!='COMPLETE'
            or (population.get('scenes'),population.get('physical_logs'),population.get('invalid'))!=(12146,44,0)
            or hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()!=seal):return {}
    evaluations=value.get('evaluations',{})
    if abs(evaluations.get('FAST808',{}).get('PDMS',-1)-official['PDMS'])>1e-8:return {}
    # Recheck only sealed source metadata, not the12146 scene arrays/ledger.
    for item in evaluations.values():
        source=Path(item['source']);summary=source/'SUMMARY.json'
        if not summary.exists():summary=source.parent/'SUMMARY.json'
        for source_path,key in [(summary,'summary_file_sha256'),(source/'identity.json','identity_file_sha256')]:
            try:raw=source_path.read_bytes();metadata=json.loads(raw)
            except (OSError,ValueError):return {}
            expected=metadata.pop('sha256',None)
            if (hashlib.sha256(raw).hexdigest()!=item['provenance'].get(key)
                    or hashlib.sha256(json.dumps(metadata,sort_keys=True,separators=(',',':')).encode()).hexdigest()!=expected):return {}
    figures=[]
    for name in ('score_bands.svg','band_transitions.svg','gain_loss_and_logs.svg'):
        item=value.get('artifacts',{}).get(name,{})
        try:valid=hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256']
        except (OSError,KeyError):valid=False
        if valid:figures.append(item['path'])
    return dict(path=str(path),seal=seal,score=official['PDMS'],
        scores={k:v['PDMS'] for k,v in evaluations.items()},
        contrasts={k:{field:v[field] for field in ('PDMS_delta','zero','metrics')} for k,v in value['contrasts'].items()},
        fixed_IL70_high=value['fixed_IL70_high_between_arm'],budget=value['native_budget_metadata'],figures=figures)


def S7_evidence(root,registry):
    run=next((r for r in registry.get('runs',[]) if r['id']=='S7_STRUCTURED_PROGRESS'),{})
    registered=run.get('native_qualification',{});path=Path(registered.get('path',''))
    try:raw=path.read_bytes();value=json.loads(raw)
    except (OSError,ValueError):return {}
    if (hashlib.sha256(raw).hexdigest()!=registered.get('file_sha256') or value.get('status')!='PASS'
            or value.get('schema')!='world_structured_progress_actual_world8_native_qualification_v1'
            or (value.get('world'),value.get('B'),value.get('G'),value.get('M'))!=(8,64,16,4)
            or value.get('trainable_numel')!=52707840 or not value.get('native_bitexact_at_inner2_and4')):return {}
    return dict(path=str(path),file_sha256=registered['file_sha256'],source=run.get('source'),source_commit=run.get('source_commit'),
        teachers=value.get('actual_first_collection_teacher_classes',{}),
        structured_queries=value.get('strict_reviews',{}).get('continuous4',{}).get('actual_teacher_only_structured_queries'))


def zero2_evidence(root):
    directory=root/'analysis/TRAIN_ZERO2_STRUCTURED_PROGRESS_AUDIT';path=directory/'RESEARCH_READOUT.json'
    value=read(path,{});evidence=read(directory/'FINAL_EVIDENCE.json',{})
    try:valid=hashlib.sha256(path.read_bytes()).hexdigest()==evidence.get('files_sha256',{}).get('RESEARCH_READOUT.json')
    except OSError:valid=False
    if not valid or value.get('status')!='COMPLETE_DISCOVERY_ONLY_NOT_TRAINING':return {}
    return dict(path=str(path),cohorts=value.get('cohorts',{}),budget=value.get('budget',{}),
        cases=[dict(token=c['token'],best=c['synthetic_all']['best_qualified_score'],
            qualified=c['synthetic_all']['qualifying_teacher'],minimum_delta=c['minimum_successful_delta_path_coordinate_at4_m']) for c in value.get('cases',[])])


def credit_lineage_evidence(root):
    path=root/'analysis/SAFETYEP_CREDIT_LINEAGE_REVIEW.json';value=read(path,{})
    if value.get('status')!='COMPLETE_BOUNDED_CPU_SOURCE_AND_BEHAVIOR_REVIEW':return {}
    return dict(path=str(path),file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        groups=sum(v.get('groups',0)for v in value.get('bounded_TRAIN_windows',{}).values()),
        summary=value.get('summary_zh',[]),limits=value.get('limits',[]))


def S5_review_evidence(root,complete_results):
    directory=root/'analysis/S5_u001616_FULL_REVIEW';path=directory/'SUMMARY.json'
    value=read(path,{});body=dict(value);seal=body.pop('sha256',None)
    phase=read(directory/'PHASE_AND_BUDGET.json',{})
    point=next((r for r in complete_results if r['arm']=='S5_PERFORMANCE_MAX' and r['updates']==1616),None)
    if (not point or value.get('status')!='COMPLETE' or phase.get('status')!='COMPLETE'
            or phase.get('actual_Adam_updates')!=1616 or phase.get('actual_recipe_matches_registration') is not True
            or phase.get('checkpoint')!=point.get('checkpoint')
            or abs(value.get('evaluations',{}).get('S5_1616',{}).get('PDMS',-1)-point['PDMS'])>1e-8
            or hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()!=seal):return {}
    return dict(path=str(path),phase_path=str(directory/'PHASE_AND_BUDGET.json'),
        zero_IL=value['contrasts']['IL70']['zero'],zero_Control=value['contrasts']['Control10k']['zero'],
        fixed=value['fixed_IL70_high_between_arm']['target_vs_Control10k'],
        phase={k:phase[k]for k in ('phase','actual_exposure_epoch','extra_true_B1_queries','extra_teacher_SDE_queries','extra_history_queries','global_scene_exposures','global_main_PG_queries','matched_compute')})


def TRAIN2048_pair_evidence(root):
    directory=root/'analysis/TRAIN2048_S0u400_S5u800_REVIEW'
    path=directory/'SUMMARY.json';value=read(path,{})
    readout=read(directory/'RESEARCH_READOUT.json',{})
    body=dict(value);seal=body.pop('sha256',None);population=value.get('population',{})
    if (value.get('status')!='COMPLETE' or readout.get('status')!='COMPLETE'
            or population.get('scenes')!=2048 or population.get('invalid')!=0
            or population.get('cohort')!='TRAIN_DIAGNOSTIC'
            or population.get('same_tokens_logs_cache_protocol_numerics') is not True
            or hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()!=seal
            or hashlib.sha256(path.read_bytes()).hexdigest()!=readout.get('summary_file_sha256')):return {}
    return dict(path=str(path),readout_path=str(directory/'RESEARCH_READOUT.json'),
        population=population,
        evaluations={k:{field:v[field]for field in ('PDMS','bins','strict4_high')}for k,v in value['evaluations'].items()},
        cohorts={k:{field:v[field]for field in ('fixed_IL70_high','fixed_IL70_low')}for k,v in value['strict_high_and_fixed_IL_cohorts'].items()},
        zero=value['cross_arm_zero_sets'],headlines=readout.get('headline_zh',[]),
        budget={k:{field:v[field]for field in ('updates','scene_exposures','actual_PG_queries','extra_queries')}for k,v in value['native_budget_metadata'].items()},
        figures=[str(p)for p in sorted(directory.glob('*.svg'))])


def TRAIN2048_pair_fragment(s):
    value=s['TRAIN2048_pair']
    if not value:return ''
    rows=[]
    for label in ('IL70','S0_u400','S5_u800'):
        point=value['evaluations'][label];contrast=value['cohorts'].get(label.split('_')[0]+'_vs_IL70')
        rows.append((label,f'{point["PDMS"]:.5f}',point['bins']['zero'],point['strict4_high'],
            f'{contrast["fixed_IL70_low"]["net_global_PDMS"]:+.5f}'if contrast else '基准',
            f'{contrast["fixed_IL70_high"]["net_global_PDMS"]:+.5f}'if contrast else '基准'))
    body='<details id="performance-train2048-pair"><summary>TRAIN2048 配对诊断：低分修复与高分退化相抵，不是 NAVTEST 成绩</summary>'
    body+='<p>同 '+esc(value['population']['scenes'])+' 个 TRAIN 场景 / '+esc(value['population']['physical_logs'])+' 个物理 log，invalid0；相同 token、cache、scorer 与真实 B1。末两列是固定 IL 队列对整个 TRAIN2048 均分的贡献，不能与 NAVTEST 排行榜混用。</p>'
    body+=table(['检查点','TRAIN PDMS','零分','strict-high','IL低分432贡献','IL高分1616贡献'],rows)
    for line in value['headlines'][2:]:body+='<p>'+esc(line)+'</p>'
    body+='<p><b>12 个零分修复中 11 个尚未直接曝光，表明本 TRAIN 人口内存在跨场景参数迁移，不能解释为只记住已见 token。</b>这仍不是独立泛化验证；同一检查点只评一次 B1，旧 IL70 采样见证不是当前 RL 同 θ 材料。均值、EP 分解和曝光关联不能证明梯度因果或最终上限；两臂此时都未激活 rescue/history，且 B/Adam 不同。</p>'
    body+='<p><a href="'+esc(value['path'])+'">完整配对审计</a> · <a href="'+esc(value['readout_path'])+'">研究简报与限制</a></p>'
    if value['figures']:
        body+='<details><summary>已有 TRAIN 分布与转移图</summary>'
        for path in value['figures']:body+='<img loading="lazy" style="max-width:100%" src="'+esc(path)+'" alt="TRAIN2048 诊断图">'
        body+='</details>'
    return body+'</details>'


def current_runs(root, registry, owner):
    now=time.time();runs=[];legacy_hold=legacy_I_resource_hold()if Path(root)==LEGACY_ROOT else{}
    for registered in registry.get('runs',[]):
        arm=registered['id'];node=owner.get('runs',{}).get(arm,{})
        target=node.get('stage_target_updates')
        stage=node.get('stages',{}).get(str(target),{})
        job=stage.get('job')
        if not job and node.get('job'):
            # An earlier completed stage is not the current resource-waiting job.
            candidate=Path(node['job'])
            if target is not None and f'formal_to{int(target):06}' in candidate.name:job=str(candidate)
        directory=Path(job) if job else None
        result=read(directory/'RESULT.json') if directory else None
        running=read(directory/'RUNNING.json') if directory else None
        output=Path(registered['root']);latest=[];ages=[];current_invocations=0
        for rank in range(8):
            path=output/f'metrics_rank{rank}.jsonl'
            rows=journal_rows(path,limit=1,max_bytes=2**16)
            if rows:
                latest.append(rows[-1])
                invocation=read(output/'invocations'/str(rows[-1].get('invocation',''))/f'rank{rank}.json',{})
                if running and invocation.get('process_start_unix',0)>=running.get('start_unix',float('inf')):
                    current_invocations+=1
                try:ages.append(max(0,now-path.stat().st_mtime))
                except OSError:pass
        fresh=len(latest)==8 and len(ages)==8 and max(ages)<=600 and current_invocations==8
        if result:status='本阶段已结束' if result.get('returncode')==0 else '执行退出，待复核'
        elif running and fresh:status='实际训练'
        elif running and current_invocations==8:status='进度过期 / 当前运行未确认'
        elif running:status='启动 / 等待首条更新'
        elif 'WAIT' in node.get('status',''):status='等待资源'
        else:status=node.get('status','已登记，尚未启动')
        if not job and 'WAIT' in node.get('status',''):status='等待资源'
        reserved=registered.get('disposition')=='RESERVED_NATIVE_CHECKPOINT_FOR_HIGHER_PRIORITY_QUALITY_STUDY'
        if reserved and status!='实际训练':
            status='保留原生检查点，资源让位高优先级研究；未判无效'
        if legacy_hold and job==legacy_hold['job']and status!='实际训练':status='临时资源保留，原'+str(legacy_hold['budget'])+'更新预算未变（未终止）'
        epochs=[r.get('metrics',{}).get('exposure_epoch') for r in latest]
        epochs=[v for v in epochs if isinstance(v,(int,float))]
        runs.append(dict(id=arm,status=status,resource_note='已登记资源让位，仍以真实退出回执为准' if reserved and status=='实际训练' else None,
            owner_status=node.get('status'),host=(running or result or {}).get('host',node.get('host')),
            job=job,rank_count=len(latest),oldest_rank_age_seconds=max(ages) if ages else None,
            updates=min((r.get('updates',0) for r in latest),default=0),
            exposure_epoch=min(epochs) if epochs else None,target=target,output=str(output)))
    return runs


def snapshot(root=ROOT):
    root=Path(root)
    jobs=[]
    directories=set((root/'jobs').glob('QUAL_*'))|set((root/'jobs').glob('PROBE*'))|set((root/'jobs').glob('S*/formal_*'))
    for directory in sorted(directories):
        result=read(directory/'RESULT.json');running=read(directory/'RUNNING.json')
        value=result or running or read(directory/'DISPATCH.json')
        if value:
            cmd=value.get('command',[]);output=cmd[cmd.index('--output')+1] if '--output' in cmd else ''
            row=tail(Path(output)/'metrics_rank0.jsonl') if output else {}
            jobs.append(dict(id=str(directory.relative_to(root/'jobs')),status=value.get('status','DISPATCHED'),host=value.get('host'),
                qualification='QUAL_' in directory.name or 'PROBE' in directory.name,
                output=output,updates=row.get('updates',0),scene_exposures=row.get('raw_scene_exposures',0),
                exposure_epoch=row.get('metrics',{}).get('exposure_epoch'),returncode=value.get('returncode')))
    # Reuse the stdlib-only sealed SPEC/identity qualification, not bare summaries.
    spec=importlib.util.spec_from_file_location('performance_report_analysis',Path(__file__).with_name('analyze.py'))
    analyzer=importlib.util.module_from_spec(spec);spec.loader.exec_module(analyzer)
    analysis=analyzer.collect_results(root)
    complete=[r for r in analysis['complete_results'] if r['updates']>0 and r.get('physical_logs')==44]
    registry=read(root/'RUNS.json',{});owner=read(root/'owners/PERFORMANCE_NATIVE/STATE.json',{})
    previous_review=read(Path('/mnt/project/qwen-world-rl-optimization-20261030/analysis/REPORT_LATEST_COMPLETE_REVIEW_20261101/SUMMARY.json'),{})
    previous_peak=previous_review.get('observed_peak',{})
    previous_score=read(previous_peak.get('evidence',''),{})
    previous_body=dict(previous_score);previous_seal=previous_body.pop('sha256',None)
    if not (previous_score.get('status')=='COMPLETE' and previous_score.get('scenes')==12146
            and previous_score.get('invalid_scenes')==0 and previous_score.get('physical_logs')==44
            and previous_score.get('PDMS')==previous_peak.get('PDMS')
            and hashlib.sha256(json.dumps(previous_body,sort_keys=True,separators=(',',':')).encode()).hexdigest()==previous_seal):
        previous_peak={}
    current=current_runs(root,registry,owner)
    legacy_current=current_runs(LEGACY_ROOT,read(LEGACY_ROOT/'RUNS.json',{}),
        read(LEGACY_ROOT/'owners/INTEGRATED_NATIVE/STATE.json',{}))
    legacy_complete=analyzer.collect_results(LEGACY_ROOT)['complete_results']
    legacy_complete=[r for r in legacy_complete if r['updates']>0 and r.get('physical_logs')==44]
    journals={r['id']:journal_rows(Path(r['output'])/'metrics_rank0.jsonl') for r in current}
    return dict(generated_UTC=datetime.now(timezone.utc).isoformat(),jobs=jobs,complete_results=complete,
        active_formal=sum(j['status']=='实际训练' for j in current),current_runs=current,journals=journals,
        active_qualification=sum(j['status']=='RUNNING' and j['qualification'] for j in jobs),
        audit=read(root/'data/AUDIT.json',{}),registry=registry,native_owner=owner,previous_peak=previous_peak,
        legacy_current_runs=legacy_current,legacy_complete_results=legacy_complete,
        plans=registered_plans(root,registry),qualification=read(root/'performance_qualification.json',{}),
        result_review=dict(pending=analysis['pending'],rejected=analysis['rejected']),
        deadline=read(root/'operations/RESOURCES_UNTIL_20261103T000000Z/REQUEST.json',{}),
        resume=read(root/'operations/NATIVE_RESUME_QUALIFICATION.json',{}),
        nominal=read(root/'qualification/REAL_NOMINAL_IL70/COMPLETE.json',{}),
        training_diagnosis=read(root/'analysis/CURRENT_TRAINING_DIAGNOSIS.json',{}),
        execution_qualifications={name:read(root/relative,{}) for name,relative in (
            ('frozen_reference','qualification/REFERENCE_CACHE_FULL_IL70/COMPLETE.json'),
            ('PACK2','benchmarks/COHERENT_PACK2/COMPLETE.json'),
            ('PACK4','benchmarks/COHERENT_PACK4/COMPLETE.json'))},
        permanent_continuation=read(root/'operations/PERMANENT_PRIORITY_BINDING/S1_NOMINAL_RESCUE_NATIVE_RETURN.json',
            read(root/'operations/OPS_V2_HANDOFF/S1_NATIVE_RETURN.json',{})),
        cache_world8=read(root/'qualification/REFERENCE_CACHE_WORLD8_NATIVE/COMPLETE.json',{}),
        packed_world8=read(root/'benchmarks/NATIVE8_PACK4/REVIEWED_COMPLETE_V4.json',{}),
        TRAIN2048_quality=read(root/'diagnostics/IL70_TRAIN2048/QUALITY.json',{}),
        verified_quality=verified_quality_evidence(root),TRAIN_GT=train_GT_evidence(root),TRAIN19=train19_evidence(root),
        S6_fresh=read(root/'operations/S6_FORMAL_LAUNCH/FRESH_IL70_ADAM0_PROOF.json',{}),
        S6_live=S6_live_evidence(root),
        first_window_progress=first_window_progress_evidence(root),
        FAST_review=FAST_review_evidence(root,complete),
        S7=S7_evidence(root,registry),zero2=zero2_evidence(root),
        credit_lineage=credit_lineage_evidence(root),
        S5_review=S5_review_evidence(root,complete),
        TRAIN2048_pair=TRAIN2048_pair_evidence(root),
        resource_window=read(root/'operations/RESTORE_REQUESTED_15MIN_GUARD/WINDOW_CORRECTION.json',{}),
        resource_restore=read(root/'operations/RESTORE_REQUESTED_15MIN_GUARD/RESTORATION_RECEIPT.json',{}),
        S7_permanent_wait=read(root/'operations/S7_PERMANENT_CAPACITY_WAIT/STATE.json',{}),
        full_TRAIN_census=read(root/'diagnostics/FULL_TRAIN_REGISTRATION.json',{}))


def table(headers,rows):
    return '<div style="overflow:auto"><table style="width:100%;border-collapse:collapse"><thead><tr>'+''.join('<th>'+esc(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td style="padding:9px;border-bottom:1px solid #dce3eb">'+esc(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def verified_quality_fragment(s):
    proof=s['verified_quality']
    run=next((r for r in s['current_runs'] if r['id']=='S6_VERIFIED_QUALITY'),None)
    status=run['status'] if run else '尚未正式登记启动'
    body='<section id="performance-verified-quality"><h3>S6：质量门控教师与冷启动校准</h3>'
    body+='<p>正式状态：<b>'+esc(status)+'</b>。与当前任务表同源；资格权重不用于正式初始化。S6 从相同 IL70 与 Adam0 独立开始，B64/G16/四 inner；真实 B1 不达严格高分时要求教师增益至少 .02、使用 α=.05，已严格高分时允许至少 .005 的更优教师、使用较弱 α=.02。教师均须 NC/DAC/TTC/Comfort=1、score≥.9；只重新分配原 β50 预算，不叠加新 β。</p>'
    fresh=s['S6_fresh']
    if (fresh.get('status')=='PASS' and fresh.get('optimizer_state_entries')==0
            and fresh.get('from_other_RL_checkpoint') is False and fresh.get('qualification_weights_used') is False):
        body+='<p>正式 native0 已核对同一 IL70、空 Adam 状态与零场景曝光；不是从资格或其它 RL 权重继续。<a href="'+esc(ROOT/'operations/S6_FORMAL_LAUNCH/FRESH_IL70_ADAM0_PROOF.json')+'">fresh IL70 / Adam0 启动证据</a>。实际在训数量仍只由八 rank 新日志决定。</p>'
    live=s['S6_live']
    if live:
        quality=live['quality'];g=live['gates'];a=live['alpha'];q=live['queries']
        body+='<aside style="background:#edf7f4;border-left:4px solid #16816c;padding:12px 16px"><h4>正式 S6 已验证到什么</h4><p>分析截至共同 Adam '+esc(live['common_updates'])+'，覆盖八 rank 的 '+esc(live['rank_update_rows'])+' 条更新日志、'+esc(quality.get('valid_scenes'))+' 个去重新鲜 TRAIN 场景；分析时间 '+esc(live['analysis_at'])+'。</p>'
        body+=table(['实际机制','已观测结果'],[
            ('真实教师门控','修复 '+str(g.get('repair_enabled'))+'/'+str(g.get('repair_scenes'))+'，α='+str(a.get('alpha_repair'))+'；已高分精修 '+str(g.get('already_strict_high_enabled'))+'/'+str(g.get('already_strict_high_scenes'))+'，α='+str(a.get('alpha_already_strict_high'))),
            ('全场景平均原 β 预算',f'IL {100*a["reconstructed_IL_share_all_valid"]:.2f}% / 辅助 {100*a["mean_alpha_all_valid_scenes"]:.2f}%；是系数预算，不是梯度比例'),
            ('实际评分分账','PG '+str(q.get('main_fresh_PG'))+' + 真 B1 '+str(q.get('true_B1'))+' + 条件独立 G16 '+str(q.get('conditional_teacher_only_extra_G16'))+' = '+str(q.get('total_actual_candidate_queries'))+f'；额外/PG {100*q["additional_over_PG_fraction"]:.2f}%（不是 GPU 时间）'),
            ('两种零分别核对','候选官方全零组 '+str(g.get('main_canonical_allzero'))+'；rawA 全零 PG0 组 '+str(g.get('main_rawA_PG0'))+'；PG0 明细中 B1 满分 '+str(live['PG0_B1_full_score'])+'/'+str(live['PG0_cases_observed']))])
        local=live['local_components'];ddp=live['DDP_total']
        cfm=local.get('nominal_scaled_cfm_gradient_norm_last_measured',{}).get('mean')
        ode=local.get('nominal_scaled_ode_gradient_norm_last_measured',{}).get('mean')
        if cfm is not None and ode is not None and ddp.get('mean') is not None:
            body+='<p>已测条件局部分量范数均值：scaled CFM '+esc(f'{cfm:.6g}')+'、ODE '+esc(f'{ode:.6g}')+'；DDP 归约后总梯度范数均值 '+esc(f'{ddp["mean"]:.6g}')+'。两者归约与分母不同，不能直接相除；辅助联合向量范数和梯度夹角尚未记录。</p>'
        body+='<p><b>这里只证明门控与有效辅助路径已激活，未证明更新后的 B1 吸收或 NAVTEST 提升。</b>同一 collection 的四 inner 不重复计新评分；条件额外池是独立 G16，不是 TRAIN19 的真 batch32。无信号的采样组与部署 B1 失败也不能互相等同。<a href="'+esc(live['path'])+'">完整正式激活审核</a></p></aside>'
    if proof:
        classes=proof['teacher_classes'];summary=proof['summary'];r=proof['rank0']
        state=r.get('final_calibration_state',{});norms=state.get('last_norms',{})
        body+='<p><b>v2 真实八卡资格 PASS：</b>52,707,840 个可训练参数，连续四步与保存 inner2 后恢复至四步的参数、Adam、scheduler、校准 PG EMA、八 rank RNG 和数据流逐位一致。64 个真实 TRAIN 场景自然触发高分精修 '+esc(classes.get('.02'))+' 个、低分修复 '+esc(classes.get('.05'))+' 个；无教师 '+esc(classes.get('inactive'))+' 个，没有人工注入门控。<a href="'+esc(proof['proof'])+'">完整原生资格</a> · <a href="'+esc(proof['gradient'])+'">全八 rank 四步梯度读数</a></p>'
        body+='<p>v1 实测发现从 IL0 出发时 IL 锚定梯度很小，按它单独校准会让辅助预算过弱。v2 只将校准目标改为 <code>max(||gIL||, .04×EMA(||gPG||))</code>，PG EMA 衰减 .9、首次正 PG 初始化；当前 PG0 可使用已有 EMA。初始 PG 与 EMA 均为零时保留最小尺度，但不消耗可冻结校准次数。旧 S1 校准、教师门和原 PG/锚定定义保持。</p>'
        if norms:
            body+=table(['rank0 第四步最后有效教师','PG','未加权 IL','原始 CFM','原始 ODE-last'],[
                ('实际局部全参数梯度范数',*[f'{norms[k]:.6g}' for k in ('pg','IL','cfm','ode')])])
            body+='<p>该槽位 PG EMA='+esc(f'{state["PG_gradient_EMA"]:.6g}')+'，目标='+esc(f'{state["last_target_norm"]:.6g}')+'；CFM/ODE scale='+esc(f'{state["scale_cfm"]:.6g}')+' / '+esc(f'{state["scale_ode"]:.6g}')+'。八 rank 末有效槽位的目标/IL 比为 '+esc(f'{summary["last_target_to_IL"]["minimum"]:.0f}')+'–'+esc(f'{summary["last_target_to_IL"]["maximum"]:.0f}')+' 倍，已摆脱近零 IL-only 校准。权重乘以实际分量梯度范数可估计注入量级；它不等于直接测得的联合 BF16/DDP 梯度、参数更新归因或 PDMS 收益。</p>'
    else:
        body+='<p>完整资格收据尚未匹配；不从 CPU 就绪或部分日志推断通过。</p>'
    return body+'</section>'


def train_GT_fragment(s):
    value=s['TRAIN_GT']
    if not value:return ''
    p=value['population'];o=value['opportunity']
    body='<section id="performance-train-GT"><h3>真实 TRAIN GT 审核：有教师机会，也必须拒绝退化标签</h3>'
    body+='<p>既有 NAVTRAIN 诊断 '+esc(p['input_count'])+' 个场景中，'+esc(p['invalid_time_count'])+' 条原始时间有缺口，整条不评分、也不记零分。其余 '+esc(p['valid_whole_trajectory_count'])+' 条用原官方参考与原物理 FP32 GT 各评分一次，共 '+esc(value['budget']['actual'])+' 次新 CPU 查询、0 模型 forward、0 训练更新。<b>这是 TRAIN 诊断，GT 是特权离线标签，不是模型分数或 NAVTEST 收益。</b></p>'
    body+=table(['固定 B1 群体','真实审计结果','含义'],[
        ('原 B1 零分 '+str(o['zero_count']),str(o['zero_strict4'])+' 条 GT 严格四项通过；'+str(o['zero_strict_high'])+' 条 score≥.9','其余安全 GT 也可能未达高分门槛'),
        ('全部合法 '+str(p['valid_whole_trajectory_count']),str(o['all_gt005'])+' 条 strict-high 且 gain>.005；'+str(o['all_ge02'])+' 条 gain≥.02','存在可验证监督机会，不能盲目模仿所有 GT'),
        ('B1 已 strict-high '+str(o['B1_strict_high_count']),str(o['B1_strict_high_GT_strict_high_gain_gt005'])+' 条 GT 改善>.005；'+str(value['GT_worse_strict_high'])+' 条反而下降>.005',
            '全体有'+str(value['GT_worse_all'])+'条 GT 下降>.005，另有'+str(value['GT_comfort_not_one'])+'条 Comfort 未通过')])
    body+='<p>这里 gain=.005 对应 0.5 个 PDMS 点；score≥.9 是严格高分门。仅凭这份 GT 审核不能由 GT 好/B1 差断言 G16 探索失败；下方另列 19 场景的同父模型采样补证。GT 没有 on-policy log-prob，不进入 PPO 优势；本次未把 GT 加入 S6、任何 bank 或正式算法。归一化再解码会改变 FP32 端点位，未来若采用必须重新评分。输入沿既有池口径，含 '+esc(p['old_DEV_in_existing_NAVTRAIN_count'])+' 个原 DEV 场景；NAVTEST token/log 严格隔离。<a href="'+esc(value['path'])+'">完整发现性审核与限制</a></p></section>'
    panel=s['TRAIN19']
    if panel:
        zero=panel['cohorts']['B1_zero'];rows=[]
        for label,key in [('15个固定 B1 零分','B1_zero'),('4个固定高分对照','fixed_strict_high_control')]:
            cohort=panel['cohorts'][key]
            for group in ('16','32'):
                counts=cohort['groups'][group]
                rows.append((label+' / G'+group,str(counts['scenes_with_strict_high'])+'/'+str(counts['scenes']),
                    str(counts['unsafe_candidates'])+'/'+str(counts['actual_PDM_queries']),
                    str(counts['strict_high_negative_rawA'])+'/'+str(counts['strict_high_candidates'])))
        body+='<details><summary>追加真实 TRAIN19 探针：G16/G32 是否已找到好轨迹</summary>'
        body+=table(['定向群体 / 采样组','有 strict-high 见证的场景','strict3 未通过候选','strict-high 中 rawA<0'],rows)
        body+='<p>零分群体两组高见证并集 '+esc(zero['high_found_either'])+'/15；11 个 GT≥.9 场景两组全部采到同门槛候选。两组都无高见证的 2 个场景，其 GT 虽 strict4，却只有 .857625/.890610，因此本 panel 的 GT 独有 strict-high 为 0，不能外推 GT 对全部训练无用。38 组均非官方全零且均有非零 rawA；零分群体的 91 条 strict-high 均 rawA>0。高分对照里的相对负 rawA 不等于 unsafe 获正信用，后者实测为 0。</p>'
        body+='<p>这是 19 个定向 TRAIN 场景、'+esc(panel['budget']['new_PDM_candidate_queries'])+' 次真实候选评分、0 Adam；G16/G32 使用独立随机组，G32 有两倍查询且 BF16 batch shape 不同，不是同算力因果对照。采样好而 B1 差是当前单轨转换机会，不是训练后吸收失败，也不是 NAVTEST 提升。<a href="'+esc(panel['path'])+'">完整 TRAIN19 研究简报</a></p></details>'
    return body


def first_window_progress_fragment(s):
    value=s['first_window_progress']
    if not value:return ''
    fallback=value['fallback'];count=fallback['count'];cohorts=value['synthetic_cohorts']
    found=cohorts['statistics'];old=cohorts['old_extra_G16_statistics'];GT=value['GT_fallback']
    body='<section id="performance-progress-teacher-discovery"><h3>研究发现：安全通过但进度偏低，能否找到更好的教师</h3>'
    body+='<p>正式 S6 首窗口中 '+esc(count)+' 个主 G16 与额外 G16 都未找到高分教师的场景，真实 B1 与 GT 均 '+esc(count)+'/'+esc(count)+' 通过 NC/DAC/TTC/Comfort，<b>是安全但进度较低，不是 '+esc(count)+' 个安全失败。</b>GT score 范围 '+esc(f'{fallback["GT_score_min"]:.6f}')+'–'+esc(f'{fallback["GT_score_max"]:.6f}')+'，达到 .9 的仅 '+esc(fallback['GT_strict_high_count'])+' 个；GT 本身没有增加合格教师槽。</p>'
    rows=[('同场景原独立额外 G16（复用评分）',old['candidates'],old['strict4'],old['qualifying_teacher'],str(old['scenes_with_qualifying_teacher'])+'/'+str(count)),
        ('原 GT roundtrip（既有75次审核的子集）',GT['scored_count'],GT['GT_roundtrip_strict4'],GT['teacher_meets_actual_gate'],'0/'+str(count))]
    for label,key in [('沿 B1 路径的平滑进度候选','B1_path_statistics'),('沿 TRAIN GT 路径的平滑进度候选','GT_path_statistics')]:
        x=cohorts[key];rows.append((label,x['candidates'],x['strict4'],x['qualifying_teacher'],str(x['scenes_with_qualifying_teacher'])+'/'+str(count)))
    body+=table(['同一固定场景组的候选来源','已评分候选','严格四项通过','合格教师','有合格教师场景'],rows)
    body+='<p><b>新增合成候选 '+esc(found['qualifying_teacher'])+'/'+esc(found['candidates'])+' 满足 strict4、score≥.9 且相对真实 B1 gain≥.02，覆盖 '+esc(found['scenes_with_qualifying_teacher'])+'/'+esc(count)+'。</b>GT 审核实际新增 '+esc(value['GT_budget']['actual_new_PDM_candidate_queries'])+' 次 CPU 查询；合成诊断另新增 '+esc(value['synthetic_queries'])+' 次 CPU 查询，0 GPU forward、0 Adam。两类合成分支的场景覆盖有重叠，不能相加。</p>'
    body+='<p>仍有 '+esc(found['candidates']-found['strict4'])+'/'+esc(found['candidates'])+' 个合成候选未通过全部严格条件；平滑构造不保证安全，必须逐条经原官方评分验证。GT 分支使用离线特权 TRAIN 标签，生成机制和成本也不同于原 SDE，不能称公平的策略采样消融、best-of16 部署分数或已学会的能力。没有更新后 B1 重评，因此这是教师发现，不是 B1 吸收或 NAVTEST 提分。</p>'
    S7=next((r for r in s['current_runs'] if r['id'].startswith('S7_')),None)
    body+='<p>独立 S7：'+esc(S7['status'] if S7 else '尚无正式任务')+'；资格状态与真实训练见前部 S7 卡，未有八 rank 新日志不计实际训练。上述诊断不修改 S6 的教师或损失。<a href="'+esc(value['GT_path'])+'">GT 首窗审核</a> · <a href="'+esc(value['synthetic_path'])+'">完整合成进度诊断</a></p></section>'
    return body


def S7_fragment(s):
    proof=s['S7'];run=next((r for r in s['current_runs'] if r['id']=='S7_STRUCTURED_PROGRESS'),None)
    if not proof and not run:return ''
    body='<section id="performance-S7-structured-progress"><h3>S7：轨迹教师发现 → CFM 与真实 ODE-last 吸收</h3>'
    body+='<p>正式状态：<b>'+esc(run['status'] if run else '尚无正式任务')+'</b>，按当前 stage 与八 rank 新日志判定，注册或模型初始化不算已学习。独立从相同 IL70 / Adam0 开始，严禁继承资格权重；主 G16/W3 SafetyEP PG 不变，沿用 PG-EMA 校准和 α=.02/.05 的原 β50 重分配。</p>'
    body+='<p>仅在真实 B1 未达 strict-high、主 G16 和原额外 SDE16 都无 high 时，再构造最多16条 TRAIN GT/B1路径进度候选；逐条经官方评分，通过 strict4、score≥.9、gain≥.02 才作教师。它们仅进入原 .75CFM+.25真实 ODE-last 辅助，不混入 PPO/oldlogp/std 或部署选轨；GT 不进入真实单轨部署输入，额外查询单独分账。</p>'
    if proof:
        t=proof['teachers']
        body+='<p><b>真实八 rank 工程资格 PASS：</b>B64/G16/四 inner、52.7M 参数，连续4步/保存2步/原生恢复4步通过，参数、Adam、RNG/stream及辅助校准恢复一致。自然教师 '+esc(t.get('.02',0)+t.get('.05',0))+' 个：α.02×'+esc(t.get('.02'))+'、α.05×'+esc(t.get('.05'))+'，其中 '+esc(t.get('structured'))+' 个由结构化候选补出，实际新增 '+esc(proof['structured_queries'])+' 次教师候选评分。科学材料比较只排除资格协议已声明的计时/Invocation元数据。<a href="'+esc(proof['path'])+'">完整原生资格</a>。这不是 B1 吸收或 NAVTEST 收益证明。</p>'
    zero=s['zero2']
    if zero:
        counts=zero['cohorts'];total=counts['statistics'];GT=counts['GT_path_statistics'];B1=counts['B1_path_statistics']
        body+='<details><summary>新增独立 CPU 验证：两个真实 B1 零分且 G16/G32 无高见证场景</summary><p>'+esc(zero['budget']['actual_new_CPU_candidate_queries'])+' 次新候选评分，0 GPU / 0 Adam；'+esc(total['qualifying_teacher'])+'/'+esc(total['candidates'])+' 条合格、覆盖 '+esc(total['scenes_with_qualifying_teacher'])+'/2。全部来自 GT 路径族；B1 路径 '+esc(B1['DAC_failed'])+'/'+esc(B1['candidates'])+' 全部 DAC 失败，不能把给任意路径增加进度等同于安全修复。</p>'
        body+=table(['TRAIN 场景','合格教师','最佳教师 score（非模型得分）','最小成功附加4s路径坐标'],[(c['token'],c['qualified'],f'{c["best"]:.6f}',str(c['minimum_delta'])+'m') for c in zero['cases']])
        body+='<p>原 GT 安全但均低于 .9，原 G16/G32 都无 high；GT-path 合成后 '+esc(GT['qualifying_teacher'])+'/'+esc(GT['candidates'])+' 合格。合成轨迹并非原 GT 或模型输出；这是特权 TRAIN 教师发现，<b>不是两个 B1 已修复、部署提分或公平采样消融</b>。<a href="'+esc(zero['path'])+'">32查询完整证据</a></p></details>'
    return body+'</section>'


def FAST_review_fragment(s):
    value=s['FAST_review']
    if not value:return ''
    IL=value['contrasts']['IL70'];control=value['contrasts']['Control10k']
    fixed=value['fixed_IL70_high'];mass=fixed['target_vs_Control10k'];metrics=control['metrics'];budget=value['budget']
    body='<section id="performance-FAST808-review"><h3>FAST 首个全量结果：0.25 epoch / Adam808</h3>'
    body+='<p><b>PDMS '+esc(f'{value["score"]:.8f}')+'</b>，完整 12146 场景 / 44 physical logs / invalid0，真正单轨 B1 ODE K10/shift5/seed2101。相同 IL70 提高 '+esc(f'{IL["PDMS_delta"]:+.6f}')+' 点；低于已观测 Control10k '+esc(f'{control["PDMS_delta"]:.6f}')+' 点。<b>这是首个 0.25 epoch 观察，尚不能据此判定方法无效。</b></p>'
    body+=table(['逐场景对照','对照实际 PDMS','FAST差值','零分修复 / 新增 / 当前'],[
        ('相同 IL70',f'{value["scores"]["IL70"]:.8f}',f'{IL["PDMS_delta"]:+.6f}',str(IL['zero']['repaired'])+' / '+str(IL['zero']['new'])+' / '+str(IL['zero']['target'])),
        ('历史 Control10k',f'{value["scores"]["Control10k"]:.8f}',f'{control["PDMS_delta"]:+.6f}',str(control['zero']['repaired'])+' / '+str(control['zero']['new'])+' / '+str(control['zero']['target']))])
    body+='<p>相 IL70：原零分保留 '+esc(IL['zero']['retained'])+'、修复 '+esc(IL['zero']['repaired'])+'，同时新增 '+esc(IL['zero']['new'])+' 个零分。相 Control10k 的差额中，固定 IL70 高分 '+esc(fixed['scenes'])+' 场景贡献 '+esc(f'{mass["global_PDMS_contribution"]:+.6f}')+' 点，其余场景 '+esc(f'{mass["other_scenes_global_contribution"]:+.6f}')+' 点。EP 分项 '+esc(f'{metrics["ego_progress"]["delta_pp"]:+.6f}')+'pp、DAC '+esc(f'{metrics["drivable_area_compliance"]["delta_pp"]:+.6f}')+'pp，NC/TTC 分别 '+esc(f'{metrics["no_at_fault_collisions"]["delta_pp"]:+.6f}')+' / '+esc(f'{metrics["time_to_collision_within_bound"]["delta_pp"]:+.6f}')+'pp。它们是完整场景的描述性分解，不证明因果或必然安全效率冲突。</p>'
    body+='<p>实际曝光 '+esc(budget['fresh_valid_scenes'])+' 个场景、PG 候选 '+esc(budget['fresh_PG_candidates'])+'；Control10k 为 '+esc(budget['Control10k_fresh_scenes'])+' 场景 / '+esc(budget['Control10k_fresh_PG_candidates'])+' 候选，Adam 时钟也不同，<b>不是等算力或单因子对照</b>。<a href="'+esc(value['path'])+'">完整封存分析</a></p>'
    if value['figures']:
        body+='<details><summary>查看已生成的全量分档、转移与增损/log分布三图</summary>'
        for path in value['figures']:body+='<figure><img loading="lazy" style="width:100%;height:auto" src="'+esc(path)+'" alt="FAST808全量分析"/><figcaption><a href="'+esc(path)+'">原始 SVG</a></figcaption></figure>'
        body+='</details>'
    return body+'</section>'


COLORS=['#087f8c','#cb6d26','#7554a3','#397744','#b94755','#2d66af','#716236']


def svg_text(x,y,value,size=13,color='#253b50',anchor='start'):
    return f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" fill="{color}" text-anchor="{anchor}">{esc(value)}</text>'


def svg_document(title,content,width=1240,height=440):
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}"><title>{esc(title)}</title><rect width="100%" height="100%" fill="#fff"/><g font-family="system-ui,sans-serif">{content}</g></svg>'


def chart_panel(x,y,w,h,title,series,xlabel,ylabel,xmax=None,ymax=None):
    """Finite numeric data only; missing series remain visibly absent."""
    clean=[]
    for item in series:
        points=[(a,b) for a,b in item['points'] if isinstance(a,(int,float)) and isinstance(b,(int,float)) and math.isfinite(a) and math.isfinite(b)]
        if points:clean.append(dict(item,points=points))
    out=svg_text(x,y,title,16)
    if not clean:return out+svg_text(x+20,y+65,'尚无可绘制的实际观测',14,'#667788')
    left=x+64;top=y+27;pw=w-85;ph=h-113
    xmax=max(xmax or 0,max(a for item in clean for a,b in item['points']),1e-10)
    ymax=max(ymax or 0,max(b for item in clean for a,b in item['points'])*1.08,1e-15)
    for n in range(5):
        yy=top+ph*(1-n/4);value=ymax*n/4
        label=(f'{value:.1e}' if 0<ymax<.01 else f'{value:.3g}')
        out+=f'<path d="M {left} {yy} h {pw}" stroke="#e5ebf0"/>'+svg_text(left-8,yy+4,label,11,anchor='end')
        xx=left+pw*n/4
        out+=svg_text(xx,top+ph+20,f'{xmax*n/4:.3g}',11,anchor='middle')
    out+=svg_text(left,top-7,ylabel,11,'#627487')
    out+=svg_text(left+pw/2,top+ph+42,xlabel,12,anchor='middle')
    for index,item in enumerate(clean):
        points=item['points'];stride=max(1,len(points)//360);points=points[::stride]+([points[-1]] if points[-1]!=points[::stride][-1] else [])
        color=item.get('color',COLORS[index%len(COLORS)])
        coords=' '.join(f'{left+pw*a/xmax:.2f},{top+ph-ph*b/ymax:.2f}' for a,b in points)
        dashed=' stroke-dasharray="5 4"' if item.get('dashed') else ''
        out+=f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"{dashed}/>'
        if len(points)==1:
            a,b=points[0];out+=f'<circle cx="{left+pw*a/xmax:.2f}" cy="{top+ph-ph*b/ymax:.2f}" r="3" fill="{color}"/>'
        lx=x+(index%3)*(w/3);ly=y+h-18+(index//3)*17
        out+=f'<path d="M {lx} {ly-4} h 17" stroke="{color}" stroke-width="2"{dashed}/>'+svg_text(lx+22,ly,item['label'],10)
    return out


def figure(root,name,svg,caption):
    path=root/'reporting/figures'/name;atomic_text(path,svg)
    return f'<figure style="margin:22px 0"><img style="width:100%;height:auto" src="{esc(path)}" alt="{esc(caption)}"/><figcaption style="color:#53697c;font-size:.9em">{esc(caption)} · <a href="{esc(path)}">SVG</a></figcaption></figure>'


def engineering_figure(root,s):
    rows=s['qualification'].get('configurations',[])
    if not rows:return '<p>吞吐资格结果尚未发布。</p>'
    body=''
    for panel,key,title,unit in [(0,'gpu','实测窗口推算：每曝光 epoch GPUh','GPUh / epoch'),(1,'queries','每曝光 epoch：有效 PG 评分预算','百万次 PG queries')]:
        x=25+panel*610;body+=svg_text(x,28,title,17)
        values=[r['epoch_budget']['estimated_window_only_GPU_hours'] if key=='gpu' else r['epoch_budget']['valid_PG_candidates']/1e6 for r in rows]
        maximum=max(values)*1.16
        for i,(r,v) in enumerate(zip(rows,values)):
            yy=65+i*73;label=f'B{r["B"]} / G{r["G"]}'
            body+=svg_text(x,yy+17,label,13)
            body+=f'<rect x="{x+112}" y="{yy}" width="{360*v/maximum:.2f}" height="24" rx="3" fill="{COLORS[i]}"/>'
            body+=svg_text(x+119+360*v/maximum,yy+17,f'{v:.2f}',12)
            if key=='gpu':
                t=r['timing_seconds']['steady_window'];body+=svg_text(x+112,yy+43,f'{t:.2f} s / window · {r["B"]/t:.2f} scenes/s',11,'#627487')
        body+=svg_text(x+112,388,unit,12)
    return figure(root,'performance_throughput_budget.svg',svg_document('四种 B/G 资格窗口与每 epoch 评分预算',body,height=415),
        '每配置仅一个真实 world8/M4 首窗口：sampling + scoring wait + old-prob + 四次反传；启动约70秒另计。不同主机，非长程稳态或 B/G 因果速度对照；不代表 S2/S3 的 MC/M2 开销。额外 B1、导出和评测未计入。')


def schedule_multiplier(epoch,spec):
    warm=spec['warmup_epochs'];hold=spec['hold_until_epoch'];end=spec['cosine_decay_end_epoch'];floor=spec['minimum_lr_ratio']
    if epoch<warm:return .001+.999*epoch/warm
    if epoch<=hold:return 1.
    if epoch>=end:return floor
    return floor+(1-floor)*(1+math.cos(math.pi*(epoch-hold)/(end-hold)))/2


def planned_figure(root,s):
    pairs=[('S0_LARGE_SAFETYEP','S0 / S1',COLORS[0]),('S5_PERFORMANCE_MAX','S5',COLORS[1])]
    body='';stage_rows=[]
    for panel,key in enumerate(['action_lr','memory_lr']):
        series=[]
        for arm,label,color in pairs:
            plan=s['plans'].get(arm)
            if not plan:continue
            p=plan['spec'];stages=p['stages'];schedule=p['schedule']
            epochs=set(i/100 for i in range(821))|{float(schedule[k]) for k in ['warmup_epochs','hold_until_epoch','cosine_decay_end_epoch']}
            for stage in stages:
                e=stage['start_epoch'];epochs.update([max(0,e-1e-8),float(e)])
            effective=[];peaks=[]
            for e in sorted(epochs):
                stage=[st for st in stages if st['start_epoch']<=e][-1]
                effective.append((e,stage[key]*schedule_multiplier(e,schedule)));peaks.append((e,stage[key]))
            series.extend([dict(label=label+' 有效计划LR',points=effective,color=color),dict(label=label+' 阶段峰值',points=peaks,color=color,dashed=True)])
            if panel==0:
                for st in stages:
                    f=schedule_multiplier(st['start_epoch'],schedule)
                    stage_rows.append((label,st['name'],st['start_epoch'],f'{st["action_lr"]:.1e} / {st["memory_lr"]:.1e}',f'{f:.4g}',f'{st["action_lr"]*f:.2e} / {st["memory_lr"]*f:.2e}'))
        body+=chart_panel(20+panel*610,28,590,365,'Action LoRA' if panel==0 else 'Scene / predictive memory',series,'有效 fresh scene exposure epoch','计划 LR（不是实测）',xmax=8.2)
    return figure(root,'performance_planned_lr.svg',svg_document('登记学习率计划：阶段峰值乘全局曝光调度',body,height=440),
        '计划，不是实测。实线=阶段峰值×连续全局调度，虚线=阶段峰值。warmup .08 epoch，保持至 .75，cosine 至4，之后 floor .05；切阶段不重启。S0/S1 同计划，S5 E4 action 实际计划为5e-7（非1e-5），memory5e-8。')+table(['计划路线','阶段','起始 epoch','Action / memory 阶段峰值','全局倍率','阶段起点有效计划 LR'],stage_rows)


def native_figures(root,s):
    process='';summaries=[];series_by_key={k:[] for k in ['action','memory','gradient','sampled','nominal']}
    for index,run in enumerate(s['current_runs']):
        arm=run['id'];rows=s['journals'].get(arm,[]);color=COLORS[index%len(COLORS)];label=arm.split('_')[0]+(' LOW' if arm=='S0_LOW_LR' else '')
        points={k:[] for k in series_by_key};seen=set();numerator=denominator=b1good=b1count=0
        for row in rows:
            u=row.get('updates');lrs=row.get('optimizer_group_lrs_used',{})
            if isinstance(lrs,dict):
                for key,nativekey in [('action','action_lora'),('memory','scene_predictive_memory')]:
                    if isinstance(lrs.get(nativekey),(int,float)):points[key].append((u,lrs[nativekey]))
            if isinstance(row.get('gradient_norm'),(int,float)):points['gradient'].append((u,row['gradient_norm']))
            # A collection is reused M times. Count support once at inner1.
            if row.get('inner_completed')!=1:continue
            collection=row.get('collection')
            if collection in seen:continue
            seen.add(collection)
            q=row.get('exploration',{}).get('quality_counts_global',{})
            n=q.get('valid_scenes');good=q.get('sampled_main_group_strict_good');b=q.get('nominal_B1_scored');bg=q.get('nominal_B1_strict_good')
            if isinstance(n,(int,float)) and n>0 and isinstance(good,(int,float)):
                numerator+=good;denominator+=n;points['sampled'].append((u,100*good/n))
            if isinstance(b,(int,float)) and b>0 and isinstance(bg,(int,float)):
                b1good+=bg;b1count+=b;points['nominal'].append((u,100*bg/b))
        for key,values in points.items():
            if values:series_by_key[key].append(dict(label=label,points=values,color=color))
        if rows:
            summaries.append((arm,f'{rows[0].get("updates")}–{rows[-1].get("updates")}',len(rows),len(seen),f'{numerator}/{denominator} ({100*numerator/denominator:.1f}%)' if denominator else '未记录',f'{b1good}/{b1count} ({100*b1good/b1count:.1f}%)' if b1count else '尚未评分 / 未启用'))
    if not summaries:return '<p>正式 native 尚无可用实际日志；不绘制零值替代曲线。</p>'
    for panel,(key,title,unit) in enumerate([('action','实际 Action used LR','LR'),('memory','实际 Memory used LR','LR'),('gradient','实际 DDP preclip gradnorm','norm')]):
        process+=chart_panel(12+panel*408,28,397,365,title,series_by_key[key],'实际 Adam update（各路线独立）',unit)
    out=figure(root,'performance_native_process.svg',svg_document('正式训练实际 LR 与 preclip gradient norm',process,height=460),
        '实际有界 rank0 日志：每路线最多末1024行 / 2MiB；LR 取 step 前 optimizer_group_lrs_used，非下一步LR。Action与memory独立坐标。M2/M4 的同 Adam 位置不等曝光或候选成本；资格值不混入。')
    support=chart_panel(20,28,590,365,'Sampled 主组：存在 strict-high 候选的场景',series_by_key['sampled'],'实际 Adam update','场景覆盖率 %',ymax=100)
    support+=chart_panel(630,28,590,365,'真实单轨 B1：strict-high 场景覆盖',series_by_key['nominal'],'实际 Adam update','已评分 B1 中占比 %',ymax=100)
    out+=figure(root,'performance_sampled_vs_B1.svg',svg_document('训练采样支持与实际单轨 B1 覆盖分别展示',support,height=460),
        '只计每 collection 的 inner1一次：sampled 主组分母是有效场景，B1 分母仅实际额外评分场景。Strict-high=NC/DAC/TTC/Comfort全1且canonical≥.9。B1未启用时不绘零线；候选支持不是部署 NAVTEST 成绩。')
    return out+table(['正式路线','bounded update区间','rank0行数','独立 collection','主组 sampled strict-high 场景','实际 B1 strict-high'],summaries)


def render_fragment(root=ROOT):
    root=Path(root);s=snapshot(root)
    retired=[j for j in s['jobs'] if j['status'] not in ('RUNNING','DISPATCHED')]
    checks=s['audit'].get('checks',{})
    best=max(s['complete_results']+s['legacy_complete_results'],key=lambda r:r['PDMS'],default=None)
    previous=s['previous_peak'];prior_pdms=max(91.45359756166451,previous.get('PDMS',0))
    prior_label=('%.8f · '%previous['PDMS']+previous['arm']+' / Adam '+str(previous['updates'])) if previous.get('PDMS',0)>91.45359756166451 else '91.45359756 · SafetyEP Control 10k'
    best_label=('%.8f · '%best['PDMS']+best['arm']+' / Adam '+str(best['updates'])) if best and best['PDMS']>prior_pdms else prior_label
    legacy_active=sum(r['status']=='实际训练' for r in s['legacy_current_runs'])
    body=D46_priority_fragment(s['active_formal']+legacy_active,max(prior_pdms,best['PDMS']if best else 0))+'<section id="performance-current" style="background:#fff;border:1px solid #cad9e8;border-radius:14px;padding:24px;margin:20px 0"><small>既有 IL70 系 · PERFORMANCE FIRST · '+esc(s['generated_UTC'])+'</small><h2>最高完整 PDMS：'+esc(best_label)+'</h2>'
    evidence=Path(best['summary']) if best and best['PDMS']>prior_pdms else Path(previous['evidence']) if previous.get('PDMS',0)>91.45359756166451 else Path('/mnt/project/qwen-world-safety-exploration-20261031/standard_training_20261101/evaluations/SAFETY_G8_CONTROL_u010000/NAVTEST/SUMMARY.json')
    if evidence.exists():body+='<p><a href="'+esc(evidence)+'">最高分完整评测证据</a></p>'
    legacy_active=sum(r['status']=='实际训练' for r in s['legacy_current_runs'])
    body+='<p><b>既有 IL70 系当前实际在训 RL 共 '+str(s['active_formal']+legacy_active)+' 条：本轮 '+str(s['active_formal'])+' 条，旧综合 I '+str(legacy_active)+' 条；本轮已登记 '+str(len(s['current_runs']))+' 条；资格测试 '+str(s['active_qualification'])+' 个。</b>此计数不包含上方独立 D46 研究。实际在训要求当前 stage job 有 RUNNING、无 RESULT，且八 rank 均有近10分钟日志；统计时间见上。本轮完整 NAVTEST 结果 '+str(len(s['complete_results']))+' 个；native0、partial、资格测试与候选 oracle 不计为 RL 提分。</p>'
    body+='<p><b>既有研究任务：</b>S1/S6 将已验证好候选吸收到真实 B1；S7 补充结构化进度教师，再经 CFM+真实 ODE-last 吸收；S5 分阶段覆盖、吸收与持续保持。各路线均从相同 IL70k（88.79290991、SHA 7dda5b14…）和 Adam0 独立开始；当前执行状态以任务表为准。</p>'
    body+='<p>实际训练范围为 Action LoRA、Scene Memory 与 Predictive Memory，共 52,707,840 个可训练参数。Qwen 主干与永久 IL 参考冻结；当前可训练 memory/KV 每次 inner 都重新构图。吞吐缓存只复用冻结分支，不把场景表征整体固定成仅训练动作头。</p>'
    body+=table(['正式路线','服务器','实际 Adam（8rank最小）','有效曝光 epoch','当前状态'],[(j['id'],j['host'] or '待分配',j['updates'],f'{j["exposure_epoch"]:.6f}' if j['exposure_epoch'] is not None else '等待首条更新',j['status']+('；'+j['resource_note'] if j.get('resource_note') else '')) for j in s['current_runs']])
    if s['complete_results']:
        results=sorted(s['complete_results'],key=lambda r:Path(r['summary']).stat().st_mtime,reverse=True)
        body+='<h3>本轮完整单轨迹评测</h3>'+table(['评测','Adam','PDMS','有效场景','invalid'],[(r['arm'],r.get('updates'),r.get('PDMS'),r['scenes'],r['invalid_scenes']) for r in results])
        S5=next((r for r in results if r['arm']=='S5_PERFORMANCE_MAX' and r['updates']==1616),None)
        if S5:
            body+='<p><b>S5 首个完整 0.25 epoch：'+esc(f'{S5["PDMS"]:.8f}')+'</b>。该检查点仍为 phase1 基础 SafetyEP 覆盖，rescue 与历史保持均未激活；不能将此分数归功于尚未启用的组件。它与 FAST 的训练批量/Adam时钟不同，尚不能据两个早期点判算法优劣。</p>'
            review=s['S5_review']
            if review:
                z=review['zero_Control'];p=review['phase'];f=review['fixed']
                body+='<p>S5 与 Control10k 的零分总数同为 '+esc(z['target'])+'，但修复 '+esc(z['repaired'])+'、新增 '+esc(z['new'])+'；固定 IL70 高分队列相 Control 的全局差额 '+esc(f'{f["global_PDMS_contribution"]:+.6f}')+' 点，其余 '+esc(f'{f["other_scenes_global_contribution"]:+.6f}')+' 点。八 rank 原生审计确认曝光 '+esc(f'{p["actual_exposure_epoch"]:.6f}')+' epoch，B1/额外教师/history查询均为0，尚未检验 PMax 的后续吸收/保持机制。<a href="'+esc(review['path'])+'">S5全量分析</a> · <a href="'+esc(review['phase_path'])+'">实际阶段与预算</a>；描述性关联，非等算力因果结论。</p>'
    else:body+='<p><b>本轮尚无完整正式 NAVTEST 分数。</b>后续只接受 sealed SPEC + identity、全12146/invalid0/44log、seed2101 的既有结果。</p>'
    body+=FAST_review_fragment(s)
    body+=TRAIN2048_pair_fragment(s)
    body+=S7_fragment(s)
    if s['legacy_current_runs']:
        body+=table(['旧综合 I：实际执行/资源保留','服务器','实际 Adam（8rank最小）','状态'],[(j['id'],j['host'],j['updates'],j['status']) for j in s['legacy_current_runs']])
    if s['legacy_complete_results']:
        body+='<p>旧综合 I 已完成的真实单轨迹 NAVTEST（与新 S5 分开）：'+esc('；'.join('Adam '+str(r['updates'])+' / PDMS '+f'{r["PDMS"]:.8f}' for r in s['legacy_complete_results']))+'。未完成的后续评测不发布均分。</p>'
    body+=verified_quality_fragment(s)+first_window_progress_fragment(s)+train_GT_fragment(s)
    body+=reviewed_I_fragment(root)
    body+='<h3>当前结论与接下来的决策</h3><p>主训练仍处于全量均匀覆盖阶段。S1 的 rescue 到 0.5 epoch 才启用；此前它与 S0 的目标相同，独立进程出现相同更新属于预期，不能据此判断 rescue 无效。S5 的历史保持到第 4 epoch 才启用，当前也不能评价其收益。</p>'
    body+='<p>S0（Adam890/inner2）与 S2（Adam404）已原生保存，为 S7 让出资源；S4 与 LOWLR 也保留原生状态。代码、结果与 Adam/RNG 保留，这是研究优先级处置，不判方法无效。正式分数只读完整 NAVTEST；资格、GT 和合成教师诊断不计入排行榜。</p>'
    lineage=s['credit_lineage']
    if lineage:
        body+='<details><summary>优势信用谱系核对：'+esc(lineage['groups'])+' 组有界真实 TRAIN 重放</summary>'
        for line in lineage['summary'][:2]:body+='<p>'+esc(line)+'</p>'
        body+='<p>该窗口不能代表完整训练人口，rawA能量占比不是参数梯度比例；前1024 token集合匹配也不等于G/Adam/算力匹配。保留原至少3/首6epoch长期检验，不能据早期0.25epoch判断上限。<a href="'+esc(lineage['path'])+'">完整谱系与同材料反事实</a></p></details>'
    tq=s['TRAIN2048_quality']
    if tq.get('status')=='COMPLETE':
        body+='<p>固定 TRAIN2048 初始 IL70 真实 B1 诊断完成：零分 '+esc(tq.get('zero_count'))+'，严格高质量 '+esc(tq.get('strict_high_count'))+'。这是训练集诊断，后续按同场景配对；不计入 NAVTEST 成绩。全 NAVTRAIN 初始 B1 普查单独使用 7 张可核验 GPU，查询预算 103288，当前状态 '+esc(s['full_TRAIN_census'].get('status','未启动'))+'，未完成时不发布全量均分。</p>'
    returned=s['permanent_continuation']
    if returned.get('status')=='PASS':
        cp=returned.get('checkpoint',{})
        body+='<p>S1 已从临时主机按完整原生状态迁往长期可用的共享资源：保存于 Adam '+esc(cp.get('updates'))+' / pending inner '+esc(cp.get('next_inner'))+'；保留同一实验的 Adam、RNG、场景游标和待更新轨迹。运行是否已恢复以当前任务表为准。</p>'
    diagnosis=s['training_diagnosis']
    if diagnosis.get('status')=='COMPLETE':
        quality=[]
        for arm,item in diagnosis.get('routes',{}).items():
            q=item.get('quality_journal_census',{})
            if not q:continue
            quality.append((arm,q.get('valid_scenes'),f'{100*q.get("strict_high_support_rate",0):.2f}%',
                f'{100*q.get("PG0_rate",0):.2f}%',f'{100*q.get("canonical_allzero_rate",0):.3f}%',q.get('nominal_B1_scored')))
        if quality:
            body+='<details><summary>完整已观测训练日志：探索支持与无优势组（'+esc(diagnosis.get('completed_at',''))+'）</summary>'
            body+=table(['路线','有效场景','G16 有严格高分候选','PG0 组','G16 官方全零组','真实 B1 查询'],quality)
            body+='<p>按 collection 去重，不重复计算四 inner。PG0 是没有有效组内策略梯度；全零组是所有候选的官方分数均为零，两者不同。有限行为窗口中多数 PG0 来自成功且 EP 饱和。当前尚未查询 B1 的项目显示 0 次查询，不代表 B1 得分为零；G16 覆盖率不能代替部署成绩。不同时间窗口场景不同，比例变化不证明遗忘。</p></details>'
        history=diagnosis.get('historical_SafetyEP_NAVTEST_zero_flow',{}).get('longitudinal',{})
        if history:
            body+='<p><b>历史完整分析：</b>旧独立 SafetyEP 序列在 12146 场景、13 个评测点中，有 '+esc(history.get('persistent_zero'))+' 个场景在所有已观测点均为零，'+esc(history.get('intermittent_zero'))+' 个间歇零分，其中 '+esc(history.get('relapsed_zero'))+' 个出现过修复后再归零。这支持同时研究困难场景学习信号与防退化保持，但尚不能区分内在驾驶冲突和共享参数更新的因果作用。该历史序列与本轮结果分开；NAVTEST 场景身份不进入训练优先级。</p>'
    qualified=s['execution_qualifications'];rows=[]
    cache=qualified['frozen_reference']
    if cache.get('status')=='PASS':
        original=cache.get('results',{}).get('original',{});cached=cache.get('results',{}).get('cached',{})
        rows.append(('冻结 IL 参考计算复用','真实 IL70 / 单 GPU / 4 inner',
            str(original.get('reference_velocity_calls'))+' → '+str(cached.get('reference_velocity_calls'))+' 次参考调用',
            '八卡原生恢复逐位一致，已用于 S1；原目标和 Adam/RNG 保持' if s['cache_world8'].get('status')=='PASS' else '梯度、参数、Adam、scheduler、RNG 逐位一致；待八卡原生恢复验证'))
    for name in ('PACK2','PACK4'):
        proof=qualified[name]
        if proof.get('status')=='PASS':
            first=proof.get('iterations',[{}])[0]
            rows.append((name+' 一致合批采样与更新','真实 IL70 / 单 GPU / '+str(len(proof.get('scenes',[])))+' TRAIN 场景',
                '首更 ratio 偏差 '+str(first.get('fresh_ratio_max_abs')),
                'PACK4 八卡参数/Adam/八rank RNG/stream 逐位恢复通过；独立正式训练从 IL70 开始' if name=='PACK4' and s['packed_world8'].get('status')=='PASS' else '单卡通过；须独立从 IL70 开始，不能替换现有单场景轨迹的更新内核'))
    if rows:body+=table(['吞吐优化','验证范围','实际结果','当前处置'],rows)
    window=s['resource_window'];restore=s['resource_restore'];deadline=window.get('deadline_unix',1793664000.)
    guard=restore.get('new_guard_seconds',900);phase='尚未进入保存窗口' if time.time()<deadline-guard else '已进入保存退出窗口' if time.time()<deadline else '临时授权已到期，退出情况以实际收据为准'
    body+='<details><summary>旧 IL70 资源窗口与归还记录（历史；D46新授权见最前卡片）</summary><p><b>原资源期限：</b>六台 wm4 / rl2 / tsh2 / wmzt / tsh3 / wm2 于 UTC11月3日00:00（北京时间08:00）硬截止；本任务原900秒余量已恢复，UTC23:45（北京时间07:45）开始保存，原窗口：'+esc(phase)+'。UTC23:40对应北京时间07:40。<a href="'+esc(root/'operations/RESTORE_REQUESTED_15MIN_GUARD/WINDOW_CORRECTION.json')+'">窗口纠正</a> · <a href="'+esc(root/'operations/RESTORE_REQUESTED_15MIN_GUARD/CLOSE_WINDOW_LAUNCH.json')+'">close_window实际启动</a>。其它任务授权不变，未知任务不回收。</p>'
    release=root/'operations/RESOURCES_UNTIL_20261103T000000Z/ACTUAL_PRE_EXPIRY_RELEASE.json'
    if release.exists():body+='<p><a href="'+esc(release)+'">23:51实际归还核对：六机自有任务进程为0</a>；永久资源上的训练与foreign任务分开，不把它们误报已退出。</p>'
    body+='</details>'
    body+='<p>S7 长期槽位等待器当前只观察旧 I 自然完成10k；之后仍须满足实际显存、可见 owner 与最多2个已知 native 的准入限制。当前 '+esc(s['S7_permanent_wait'].get('status','尚无收据'))+'；不停止旧 I、第三方 D44 或 B/D，也不降低门槛硬塞。评测未齐不发均分，资源等待不判算法无效。<a href="'+esc(root/'operations/S7_PERMANENT_CAPACITY_WAIT/STATE.json')+'">等待器状态</a></p>'
    body+='<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px"><article><h3>已完成：全量数据审计</h3><p>'+esc(checks.get('unique_tokens','待查'))+' 唯一 TRAIN；NAVTEST token/log 交集 0。309864 图像文件、103288 cache 文件逐个筛查无缺失；cache 格式筛查不等同完整语义评分。</p></article><article><h3>尚待回答：是否真正提分</h3><p>重点看固定 TRAIN 中 sampled-good/B1-bad 的后续修复，以及全 NAVTEST 的零分修复和高分退化。不会以扩大采样覆盖率代替单轨迹部署成绩。</p></article></div>'
    body+='<h3>实测工程成本与评分预算</h3><p><a href="'+esc(root/'performance_qualification.json')+'">资格计时与预算原始汇总</a></p>'+engineering_figure(root,s)
    body+='<h3>登记学习率计划：两类参数分别显示</h3>'+planned_figure(root,s)
    body+='<h3>正式训练实测：更新尺度与支持转换</h3>'+native_figures(root,s)
    body+=table(['路线','主要机制','B / G / M','计划与评测'],[
        ('S1 · 第一优先','SafetyEP + verified CFM + 真实 ODE last-state SmoothL1','128 /16 /4','0.25→0.5→1→3→6 epochs'),
        ('S5 · 第一优先','SafetyEP→Rescue→持续历史教师→低 LR 精修','64→128 /16 /4','同一 run 保留 Adam / RNG'),
        ('S6 · 新质量吸收路线','真实 B1 + strict-high gain双门 + PG-EMA校准','64 /16 /4','epoch0启用；首评 .0625 / .125；状态见实时任务表'),
        ('S7 · 结构化进度教师','主/额外G16均无high后新增已验证teacher-only候选；CFM+真实ODE-last','64 /16 /4','八rank资格PASS；注册/启动与实际训练分开'),
        ('S0 / LOWLR','大场景覆盖 + 曝光 warmup/hold/cosine','128 /16 /4','5e-5 与 3e-5 分别 fresh IL'),
        ('S2','保守优势加权 FM，负A二次项有下界','128 /16 /2','MC4，独立 IL70'),
        ('S3','逐 MC energy-ratio FPO + 原物理锚定','128 /16 /2','MC2；非 endpoint density'),
        ('S4','真实时序安全 + nominal rescue','128 /16 /4','原生保存，资源让位；未判无效')])
    body+='<h3>最近工程结论</h3><p>四种 B/G 的真实八卡窗口已通过。B128/G16 的拆分恢复参数、Adam、scheduler 与八 rank RNG/stream 逐位一致；完整 IL70 的真 ODE 末状态监督和全 scope 梯度通过。S1/S2/S3/S4/S5 辅助启用资格均已完成，尚不能称部署提分。</p><p>G32 按完整组归一化；额外条件 G16 仅用于教师挖掘。padding 排除全部 loss/std/有效曝光；当前 memory 每次 inner 重建计算图。首窗口约18小时/epoch，启动、评测、新辅助项和恢复重放另计。</p>'
    body+='<details><summary>资格/调试记录（保留失败，不作为算法负结果）</summary>'+table(['任务','服务器','状态','返回码'],[(j['id'],j['host'],j['status'],j['returncode']) for j in retired])+'<p>首轮真实评分后，恢复哈希未支持 NumPy 数组而在 Adam 前失败；修正后使用新冻结资格版本，不继承资格权重。中间未完成的源码副本派发已停止并保留收据。</p></details>'
    body+='<p><a href="/mnt/project/qwen-world-performance-rl-20261103/report.html">本轮独立报告</a> · <a href="/mnt/project/qwen-world-performance-rl-20261103/data/AUDIT.json">完整审计</a></p></section>'
    compact=dict(s);compact['journals']={arm:dict(path=str(Path(next(r['output'] for r in s['current_runs'] if r['id']==arm))/'metrics_rank0.jsonl'),rows=len(rows),first_update=rows[0].get('updates') if rows else None,last_update=rows[-1].get('updates') if rows else None) for arm,rows in s['journals'].items()}
    compact['figures']=[str(p) for p in sorted((root/'reporting/figures').glob('performance_*.svg'))]
    atomic_text(root/'report_snapshot.json',json.dumps(compact,ensure_ascii=True,indent=2))
    css='body{font-family:system-ui,sans-serif;background:#f1f5f9;color:#203348;max-width:1450px;margin:24px auto;padding:0 20px;line-height:1.6}th{text-align:left;padding:9px;background:#e8eff7}h2{color:#126b57}a{color:#1763a1}'
    atomic_text(root/'report.html','<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="1800"><title>Qwen-World Performance RL</title><style>'+css+'</style><body>'+body+'</body></html>')
    return body


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--once',action='store_true')
    args=parser.parse_args();render_fragment(args.root)
    report=read(args.root/'report_snapshot.json',{})
    print(json.dumps(dict(report=str(args.root/'report.html'),generated_UTC=report.get('generated_UTC'),actual_training=report.get('active_formal'),complete_NAVTEST=len(report.get('complete_results',[])),figures=len(report.get('figures',[])))))
