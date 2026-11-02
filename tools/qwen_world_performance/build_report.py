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
esc=lambda x:html.escape(str(x),quote=True)


def read(path, default=None):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return default


def atomic_text(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.writing-report-'+str(os.getpid()));tmp.write_text(value,encoding='utf-8');os.replace(tmp,path)


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


def current_runs(root, registry, owner):
    now=time.time();runs=[]
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
    body+='<p>独立 S7：'+esc(S7['status'] if S7 else '代码开发中，尚无正式训练任务，不计入在训数量')+'；上述诊断不修改 S6 的教师或损失。<a href="'+esc(value['GT_path'])+'">GT 首窗审核</a> · <a href="'+esc(value['synthetic_path'])+'">完整合成进度诊断</a></p></section>'
    return body


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
    body='<section id="performance-current" style="background:#fff;border:1px solid #cad9e8;border-radius:14px;padding:24px;margin:20px 0"><small>新一轮 · PERFORMANCE FIRST · '+esc(s['generated_UTC'])+'</small><h2>最高完整 PDMS：'+esc(best_label)+'</h2>'
    evidence=Path(best['summary']) if best and best['PDMS']>prior_pdms else Path(previous['evidence']) if previous.get('PDMS',0)>91.45359756166451 else Path('/mnt/project/qwen-world-safety-exploration-20261031/standard_training_20261101/evaluations/SAFETY_G8_CONTROL_u010000/NAVTEST/SUMMARY.json')
    if evidence.exists():body+='<p><a href="'+esc(evidence)+'">最高分完整评测证据</a></p>'
    legacy_active=sum(r['status']=='实际训练' for r in s['legacy_current_runs'])
    body+='<p><b>当前实际在训 RL 共 '+str(s['active_formal']+legacy_active)+' 条：本轮 '+str(s['active_formal'])+' 条，旧综合 I '+str(legacy_active)+' 条；本轮已登记 '+str(len(s['current_runs']))+' 条；资格测试 '+str(s['active_qualification'])+' 个。</b>实际在训要求当前 stage job 有 RUNNING、无 RESULT，且八 rank 均有近10分钟日志；统计时间见上。本轮完整 NAVTEST 结果 '+str(len(s['complete_results']))+' 个；native0、partial、资格测试与候选 oracle 不计为 RL 提分。</p>'
    body+='<p><b>当前任务：</b>S1 高分候选向真实 B1 吸收；S5 分阶段覆盖、吸收与持续保持；S6 验证已安全高分场景的更优教师与冷启动校准。各路线均从相同 IL70k（88.79290991、SHA 7dda5b14…）和 Adam0 独立开始。</p>'
    body+='<p>实际训练范围为 Action LoRA、Scene Memory 与 Predictive Memory，共 52,707,840 个可训练参数。Qwen 主干与永久 IL 参考冻结；当前可训练 memory/KV 每次 inner 都重新构图。吞吐缓存只复用冻结分支，不把场景表征整体固定成仅训练动作头。</p>'
    body+=table(['正式路线','服务器','实际 Adam（8rank最小）','有效曝光 epoch','当前状态'],[(j['id'],j['host'] or '待分配',j['updates'],f'{j["exposure_epoch"]:.6f}' if j['exposure_epoch'] is not None else '等待首条更新',j['status']+('；'+j['resource_note'] if j.get('resource_note') else '')) for j in s['current_runs']])
    if s['legacy_current_runs']:
        body+=table(['独立继续的旧实验','服务器','实际 Adam（8rank最小）','状态'],[(j['id'],j['host'],j['updates'],j['status']) for j in s['legacy_current_runs']])
    if s['legacy_complete_results']:
        body+='<p>旧综合 I 已完成的真实单轨迹 NAVTEST（与新 S5 分开）：'+esc('；'.join('Adam '+str(r['updates'])+' / PDMS '+f'{r["PDMS"]:.8f}' for r in s['legacy_complete_results']))+'。未完成的后续评测不发布均分。</p>'
    body+=verified_quality_fragment(s)+first_window_progress_fragment(s)+train_GT_fragment(s)
    body+=reviewed_I_fragment(root)
    body+='<h3>当前结论与接下来的决策</h3><p>主训练仍处于全量均匀覆盖阶段。S1 的 rescue 到 0.5 epoch 才启用；此前它与 S0 的目标相同，独立进程出现相同更新属于预期，不能据此判断 rescue 无效。S5 的历史保持到第 4 epoch 才启用，当前也不能评价其收益。</p>'
    body+='<p>S4 与 LOWLR 在完整原生保存后为 S6 让出资源，保留实验代码、结果和状态；这是研究优先级处置，不判定方法无效。正式分数只读完整 NAVTEST，S6 资格与 GT 诊断均不计入排行榜。</p>'
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
    if s['complete_results']:
        body+='<h3>本轮完整单轨迹评测</h3>'+table(['评测','Adam','PDMS','有效场景','invalid'],[(r['arm'],r.get('updates'),r.get('PDMS'),r['scenes'],r['invalid_scenes']) for r in s['complete_results']])
    else:body+='<p><b>本轮尚无完整正式 NAVTEST 分数。</b>后续只接受 sealed SPEC + identity、全12146/invalid0/44log、seed2101 的既有结果。</p>'
    body+='<p><b>资源期限：</b>wm4 / rl2 / tsh2 / wmzt / tsh3 / wm2 至北京时间 11月3日08:00；23:45 UTC 起保存退出。其它授权按原许可执行，未知任务不回收。</p>'
    body+='<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px"><article><h3>已完成：全量数据审计</h3><p>'+esc(checks.get('unique_tokens','待查'))+' 唯一 TRAIN；NAVTEST token/log 交集 0。309864 图像文件、103288 cache 文件逐个筛查无缺失；cache 格式筛查不等同完整语义评分。</p></article><article><h3>尚待回答：是否真正提分</h3><p>重点看固定 TRAIN 中 sampled-good/B1-bad 的后续修复，以及全 NAVTEST 的零分修复和高分退化。不会以扩大采样覆盖率代替单轨迹部署成绩。</p></article></div>'
    body+='<h3>实测工程成本与评分预算</h3><p><a href="'+esc(root/'performance_qualification.json')+'">资格计时与预算原始汇总</a></p>'+engineering_figure(root,s)
    body+='<h3>登记学习率计划：两类参数分别显示</h3>'+planned_figure(root,s)
    body+='<h3>正式训练实测：更新尺度与支持转换</h3>'+native_figures(root,s)
    body+=table(['路线','主要机制','B / G / M','计划与评测'],[
        ('S1 · 第一优先','SafetyEP + verified CFM + 真实 ODE last-state SmoothL1','128 /16 /4','0.25→0.5→1→3→6 epochs'),
        ('S5 · 第一优先','SafetyEP→Rescue→持续历史教师→低 LR 精修','64→128 /16 /4','同一 run 保留 Adam / RNG'),
        ('S6 · 新质量吸收路线','真实 B1 + strict-high gain双门 + PG-EMA校准','64 /16 /4','epoch0启用；首评 .0625 / .125；状态见实时任务表'),
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
