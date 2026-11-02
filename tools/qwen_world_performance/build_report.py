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
esc=lambda x:html.escape(str(x),quote=True)


def read(path, default=None):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return default


def atomic_text(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.writing-report-'+str(os.getpid()));tmp.write_text(value,encoding='utf-8');os.replace(tmp,path)


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
        epochs=[r.get('metrics',{}).get('exposure_epoch') for r in latest]
        epochs=[v for v in epochs if isinstance(v,(int,float))]
        runs.append(dict(id=arm,status=status,owner_status=node.get('status'),host=(running or result or {}).get('host',node.get('host')),
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
    journals={r['id']:journal_rows(Path(r['output'])/'metrics_rank0.jsonl') for r in current}
    return dict(generated_UTC=datetime.now(timezone.utc).isoformat(),jobs=jobs,complete_results=complete,
        active_formal=sum(j['status']=='实际训练' for j in current),current_runs=current,journals=journals,
        active_qualification=sum(j['status']=='RUNNING' and j['qualification'] for j in jobs),
        audit=read(root/'data/AUDIT.json',{}),registry=registry,native_owner=owner,previous_peak=previous_peak,
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
        permanent_continuation=read(root/'operations/OPS_V2_HANDOFF/S1_NATIVE_RETURN.json',{}))


def table(headers,rows):
    return '<div style="overflow:auto"><table style="width:100%;border-collapse:collapse"><thead><tr>'+''.join('<th>'+esc(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td style="padding:9px;border-bottom:1px solid #dce3eb">'+esc(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'


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
    best=max(s['complete_results'],key=lambda r:r['PDMS'],default=None)
    previous=s['previous_peak'];prior_pdms=max(91.45359756166451,previous.get('PDMS',0))
    prior_label=('%.8f · '%previous['PDMS']+previous['arm']+' / Adam '+str(previous['updates'])) if previous.get('PDMS',0)>91.45359756166451 else '91.45359756 · SafetyEP Control 10k'
    best_label=('%.8f · '%best['PDMS']+best['arm']+' / Adam '+str(best['updates'])) if best and best['PDMS']>prior_pdms else prior_label
    body='<section id="performance-current" style="background:#fff;border:1px solid #cad9e8;border-radius:14px;padding:24px;margin:20px 0"><small>新一轮 · PERFORMANCE FIRST · '+esc(s['generated_UTC'])+'</small><h2>最高完整 PDMS：'+esc(best_label)+'</h2>'
    evidence=Path(best['summary']) if best and best['PDMS']>prior_pdms else Path(previous['evidence']) if previous.get('PDMS',0)>91.45359756166451 else Path('/mnt/project/qwen-world-safety-exploration-20261031/standard_training_20261101/evaluations/SAFETY_G8_CONTROL_u010000/NAVTEST/SUMMARY.json')
    if evidence.exists():body+='<p><a href="'+esc(evidence)+'">最高分完整评测证据</a></p>'
    body+='<p><b>本轮实际在训 '+str(s['active_formal'])+' 条 / 已登记 '+str(len(s['current_runs']))+' 条；资格测试 '+str(s['active_qualification'])+' 个。</b>实际在训要求当前 stage job 有 RUNNING、无 RESULT，且八 rank 均有近10分钟日志；统计时间见上。旧实验独立继续。本轮完整 NAVTEST 结果 '+str(len(s['complete_results']))+' 个；native0、partial、资格测试与候选 oracle 不计为 RL 提分。</p>'
    body+='<p><b>当前任务：</b>B128/G16 工程资格 → S1 高分候选向真实 B1 吸收；S5 分阶段覆盖、吸收与持续保持。各路线均从相同 IL70k（88.79290991、SHA 7dda5b14…）和 Adam0 独立开始。</p>'
    body+=table(['正式路线','服务器','实际 Adam（8rank最小）','有效曝光 epoch','当前状态'],[(j['id'],j['host'] or '待分配',j['updates'],f'{j["exposure_epoch"]:.6f}' if j['exposure_epoch'] is not None else '等待首条更新',j['status']) for j in s['current_runs']])
    body+='<h3>当前结论与接下来的决策</h3><p>主训练仍处于全量均匀覆盖阶段。S1 的 rescue 到 0.5 epoch 才启用；此前它与 S0 的目标相同，独立进程出现相同更新属于预期，不能据此判断 rescue 无效。S5 的历史保持到第 4 epoch 才启用，当前也不能评价其收益。</p>'
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
            '梯度、参数、Adam、scheduler、RNG 逐位一致；八卡原生恢复验证后再上线'))
    for name in ('PACK2','PACK4'):
        proof=qualified[name]
        if proof.get('status')=='PASS':
            first=proof.get('iterations',[{}])[0]
            rows.append((name+' 一致合批采样与更新','真实 IL70 / 单 GPU / '+str(len(proof.get('scenes',[])))+' TRAIN 场景',
                '首更 ratio 偏差 '+str(first.get('fresh_ratio_max_abs')),
                '八卡恢复验证中；正式训练须独立从 IL70 开始，不能替换现有单场景轨迹的更新内核'))
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
        ('S0 / LOWLR','大场景覆盖 + 曝光 warmup/hold/cosine','128 /16 /4','5e-5 与 3e-5 分别 fresh IL'),
        ('S2','保守优势加权 FM，负A二次项有下界','128 /16 /2','MC4，独立 IL70'),
        ('S3','逐 MC energy-ratio FPO + 原物理锚定','128 /16 /2','MC2；非 endpoint density'),
        ('S4','真实时序安全 + nominal rescue','128 /16 /4','无事件不伪造风险时间')])
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
