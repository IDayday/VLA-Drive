"""One new IL46 JaT run: real native qualification, fresh launch, event evals.

The old RL runs stay terminated. No training tensors cross the qualification
boundary. Existing native trainer, scorer, dispatcher and retention are reused.
"""
import fcntl
import html
import json
import os
from pathlib import Path
import time
import traceback
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.qwen_world_compare.contracts import fingerprint,file_sha
from tools.d44_native_rl.launch import launch as native_launch
from tools.d46_augustus_ops.owner import review_native
from tools.d46_augustus_ops import evaluations as E,complete_events
from tools.d46_augustus_ops.event_wait import wait_for_events
from tools.d46_fpo_campaign.retention import rotate

ROOT=Path('/mnt/project/d46-aug-migration-20261104')
PARENT=Path('/mnt/project/d46-priority-rl-20261103')
TASK='d46_aug_migration_20261104'
PYTHON='/mnt/project/flow-safe-improvement-20261013/venv/bin/python'
SOURCE=Path(__file__).resolve().parents[2]


def read(path,default=None):
    p=Path(path);return json.loads(p.read_text())if p.is_file()else default


def formal_route(route):
    result=dict(route,source=route.get('formal_source',route['source']))
    if route.get('formal_source_sha'):result['source_sha']=route['formal_source_sha']
    return result


def registered_routes(plan):
    return [plan['route']]+([plan['secondary_route']]if plan.get('secondary_route')else[])+plan.get('archived_routes',[])


def finished(jobs):
    results=[read(Path(j['job'])/'RESULT.json')for j in jobs]
    if any(r and r.get('returncode')!=0 for r in results):
        for job,result in zip(jobs,results):
            if not result:(Path(job['job'])/'STOP_REQUESTED').touch()
        raise RuntimeError('Owned child failed; healthy peers requested to save; see job RESULT')
    return all(results)


def all_refused_before_start(jobs):
    """Resource contention is retryable only if no member ever ran a child."""
    results=[read(Path(j['job'])/'RESULT.json',{})for j in jobs]
    return bool(results)and all(r.get('returncode')==75 and r.get('child_not_started')is True for r in results)


def launch_stage(route,stage):
    (ROOT/'operations').mkdir(parents=True,exist_ok=True)
    receipts=[];qual=stage!='formal';gpus=route.get('gpus',list(range(8)))
    output=route['qualification_output']if qual else route['formal_output']
    for node,host in enumerate(route['hosts']):
        job=ROOT/'jobs'/route.get('job_namespace','')/stage/f'node{node}'
        job.mkdir(parents=True,exist_ok=True)
        old=read(job/'LAUNCH.json')
        if old:
            receipts.append(old);continue
        if (job/'INTENT.json').exists():raise RuntimeError('Unresolved dispatch intent; do not duplicate')
        command=['env','NCCL_IB_DISABLE=1','NCCL_SOCKET_IFNAME=eth0',PYTHON,'-m','torch.distributed.run',
            '--nnodes='+str(len(route['hosts'])),'--nproc_per_node='+str(len(gpus)),f'--node_rank={node}','--master_addr='+route['master_addr'],
            '--master_port='+str(route.get('master_port',29642)+(0 if qual else 1)),'--max_restarts=0',
            '-m','tools.d46_augustus.train','--config',route['config'],'--output',output]
        if qual:
            command+=['--qualification','--stop-after-update','1'if stage=='qualification_first'else'2']
            if stage!='qualification_first':command+=['--resume',str(Path(output)/'checkpoints/update000001_inner1')]
        elif Path(output).exists()and any(Path(output).iterdir())and node==0:
            raise RuntimeError('Fresh formal output must be empty')
        atomic_json(job/'INTENT.json',dict(command=command,host=host,at=time.time()))
        receipts.append(native_launch(host,route['source']if qual else formal_route(route)['source'],job,command,gpus,
            task_root=PARENT,task=TASK,worker_source=SOURCE,
            admission_wait_seconds=route.get('admission_wait_seconds',0),
            share_min_free_mib=route.get('share_min_free_mib_by_host',{}).get(host)))
        receipts_root=Path(route.get('proof_directory',ROOT/'operations'));receipts_root.mkdir(parents=True,exist_ok=True)
        atomic_json(receipts_root/(stage.upper()+'_LAUNCHES.json'),receipts)
    return receipts


def check_qualification(route,updates):
    proof=review_native(route,route['qualification_output'],updates,True)
    ranks=[read(Path(route['qualification_output'])/f'result_rank{rank}.json',{})for rank in range(route['world'])]
    if any(r.get('updates')!=updates or r.get('qualification')is not True or r.get('frozen_before')!=r.get('frozen_after')for r in ranks):
        raise ValueError('Real allrank update/frozen-weight qualification failed')
    proof['allrank_frozen_weights_unchanged']=True
    proof_root=Path(route.get('proof_directory',ROOT/'operations'))
    atomic_json(proof_root/f'QUALIFICATION_{updates}.json',proof)
    if updates==2:
        first=read(proof_root/'QUALIFICATION_1.json')
        if proof['allrank_RNG_sha256']!=first['allrank_RNG_sha256']:
            raise ValueError('Pending inner resume consumed RNG')
    return proof


def binding(route,cp):
    route=formal_route(route)
    cp=Path(cp).resolve();identity=read(cp/'identity.json',{});seal=read(cp/'COMPLETE',{});files=read(cp/'files.json',{})
    body=dict(identity);digest=body.pop('sha256',None)
    if(cp.parent!=Path(route['formal_output']).resolve()/'checkpoints'or fingerprint(body)!=digest
        or identity.get('experiment_id')!=route['experiment']or identity.get('qualification')is not False
        or identity.get('parent_model_sha256')!=E.PARENT_SHA or seal.get('phase')!='READY'or seal.get('next_inner')!=0
        or seal.get('files_sha256')!=file_sha(cp/'files.json')or identity.get('world')!=route['world']):
        raise ValueError('New formal IL46 native identity mismatch')
    if identity['recipe']['augustus'].get('action_output_precision')!='float32_linear':raise ValueError('Precision mode lost')
    for name,sha in identity['source_files'].items():
        if file_sha(Path(route['source'])/name)!=sha:raise ValueError('Scientific source changed '+name)
    clock=read(cp/'transaction.json')['ranks'][0]['valid_scene_exposures']
    return dict(checkpoint=str(cp),updates=seal['updates'],identity_sha256=digest,inventory_sha256=seal['files_sha256'],
        payload_sha256=files['optimizer.pt'],parent_model_sha256=E.PARENT_SHA,algorithm_schema=E.SCHEMA,
        experiment=route['experiment'],scene_budget_clock=clock,source=route['source'],
        protocol_sha256=identity['augustus_protocol_sha256'],
        policy_signature=fingerprint(dict(parent=E.PARENT_SHA,trainable_payload=files['optimizer.pt'],scope=identity['scope'])))


def discover(root,state,routes):
    for route in registered_routes(routes):
        out=Path(route['formal_output'])
        candidates=[out/'checkpoints'/f'update{u:06d}_inner0'for u in route['early_evaluation_updates']]
        candidates += [Path(read(p)['checkpoint'])for p in sorted((out/'evaluation_requests').glob('*.json'))]
        for cp in candidates:
            if not(cp/'COMPLETE').exists():continue
            b=binding(route,cp);key=route['experiment']+f'_u{b["updates"]:06d}'
            if key in state['cases']:
                if state['cases'][key]['binding']!=b:raise ValueError('Existing eval binding changed')
                continue
            state['cases'][key]=dict(id=key,binding=b,origins=['registered_migration_gate'],status='PENDING',jobs={},
                next_attempt={},created=time.time(),world=32,output=str(ROOT/'evaluations'/key/'evaluation'))
    return state


def evaluation_launch(host,source,job,command,gpus,**kwargs):
    kwargs.update(task_root=PARENT,task=TASK,worker_source=SOURCE)
    return native_launch(host,source,job,command,gpus,**kwargs)


def notify(state_path,case_id,expected_summary_sha,**kwargs):
    case=read(state_path)['cases'][case_id];summary=read(case['summary']);body=dict(summary);sha=body.pop('sha256',None)
    if(sha!=expected_summary_sha or fingerprint(body)!=sha or summary['scenes']!=12146
        or summary['invalid_scenes']!=0 or summary['physical_logs']!=44):raise ValueError('Only full valid score may publish')
    atomic_json(ROOT/'operations/LATEST_COMPLETE.json',dict(case=case_id,summary=summary,at=time.time(),NAVTEST_SELECTED=True))
    return dict(notified_sha=sha,status='COMPLETE_VERIFIED')


def compare_numeric_baselines(evaluations):
    """Completed test banks only; output is analysis, never a trainer input."""
    zero={c['binding']['experiment']:c for c in evaluations['cases'].values() if c['status']=='COMPLETE' and c['binding']['updates']==0}
    for case in evaluations['cases'].values():
        baseline=zero.get(case['binding']['experiment'])
        if case['status']!='COMPLETE' or case['binding']['updates']==0 or baseline is None:continue
        path=Path(case['summary']).parent/'VS_NUMERIC_U0.json'
        if path.exists():continue
        from tools.d44_native_rl.analyze import analyze
        value=analyze(case['output'],baseline['output'])
        value.pop('sha256',None)
        value['baseline_description']='Same experiment numeric u0; identical original IL46 weights, no Adam update'
        value['gain_loss']['fixed_reference_cohorts']={k.replace('IL44','numeric_u0'):v for k,v in value['gain_loss']['fixed_reference_cohorts'].items()}
        value['gain_loss']['limits']=value['gain_loss']['limits'].replace('IL44','same numeric u0')
        value['sha256']=fingerprint(value);atomic_json(path,value)


def report(state,evaluations):
    plan=read(ROOT/'operations/PLAN.json');route=plan['route'];out=Path(route['formal_output']);latest=read(out/'latest.json',{})
    current_ids={r['experiment']for r in [route]+([plan['secondary_route']]if plan.get('secondary_route')else[])}
    archived_rows=[]
    rows=[];best=90.353633650597;best_name='历史D FPO1711（权重已按用户要求删除）'
    zero={c['binding']['experiment']:read(c['summary'])['PDMS'] for c in evaluations['cases'].values() if c['status']=='COMPLETE' and c['binding']['updates']==0}
    for c in sorted(evaluations['cases'].values(),key=lambda c:(c['binding']['experiment']!=route['experiment'],-c['binding']['updates'])):
        s=read(c.get('summary','/missing'),{});target=rows if c['binding']['experiment']in current_ids else archived_rows
        if c['status']=='COMPLETE':
            delta=s['PDMS']-zero.get(c['binding']['experiment'],s['PDMS'])
            target.append(f'<tr><td>{c["id"]}</td><td>{s["PDMS"]:.8f}</td><td>{s["PDMS"]-89.4955209758546:+.5f}</td><td>{delta:+.5f}</td><td>12146 / invalid0</td></tr>')
            if s['PDMS']>best:best=s['PDMS'];best_name=c['id']
        else:target.append(f'<tr><td>{c["id"]}</td><td>—</td><td>{c["status"]}</td><td>—</td><td>{c.get("completed_shards",0)}/32片</td></tr>')
    native=read(Path(latest.get('checkpoint','/missing'))/'COMPLETE',{})
    from tools.d46_fpo_campaign.reporting import bounded_rows
    tails=[bounded_rows(out/f'metrics_rank{rank}.jsonl')for rank in range(route['world'])]
    common=min((rows[-1]['updates']for rows in tails if rows),default=0)if all(tails)else 0
    active=int(common>0 and state['stage']=='formal'and any(not(Path(j['job'])/'RESULT.json').exists()for j in state.get('jobs',[])))
    primary_active=active;cards=16*active;secondary_html='';secondary=read(ROOT/'operations/PLAN.json').get('secondary_route')
    if secondary:
        ss=state.get('secondary',{});tails2=[bounded_rows(Path(secondary['formal_output'])/f'metrics_rank{r}.jsonl')for r in range(secondary['world'])]
        common2=min((v[-1]['updates']for v in tails2 if v),default=0)if all(tails2)else 0
        live=int(common2>0 and ss.get('stage')=='formal'and any(not(Path(j['job'])/'RESULT.json').exists()for j in ss.get('jobs',[])))
        active+=live;cards+=live*secondary['world']
        secondary_html=f"<p><b>独立低KL对照 {secondary['experiment']}</b>：{html.escape(ss.get('stage','WAITING_EVALUATION'))}；正式共同更新 {common2}。同IL46、新Adam0；beta0.001，其余科学配方沿用AUG7。4卡执行B64/G16/M2，rank RNG及浮点归约与16卡不同；不称逐抽样配对对照。先完成主线200/400评测，再自动资格、原生恢复和独立正式训练。</p>"
    state.update(actual_training=active,common_update=common)
    atomic_json(ROOT/'operations/STATE.json',state)
    from tools.d46_aug_migration.tracking import snapshot
    health=snapshot(out,route['world'],finished=state['stage']=='formal_finished');atomic_json(ROOT/'operations/TRAINING_HEALTH.json',health)
    audit=read(ROOT/'analysis/MIGRATION_REVIEW.json',{})
    findings='<ul>'+''.join('<li>'+html.escape(x)+'</li>'for x in audit.get('findings',[]))+'</ul>'
    speed=health.get('end_to_end') or {}
    health_html=f'<p>全rank共同更新 {health.get("common_update",0)}；场景曝光 {health.get("scene_exposures",0)}；{health.get("scene_epochs",0):.5f} E。最近窗口端到端 {speed.get("updates_per_hour",0):.1f} 更新/小时（包含采样、评分、同步）。</p>'
    health_html+='<p>检查状态：'+html.escape(health['status'])+'；'+html.escape('；'.join(health['alerts']))+'</p>'
    credit_description=route.get('credit_description','SafetyEP组内减均值；全64×16候选population std，padding排除，cap5')
    health_html+='<p>数值检查通过不等于PDMS提升；逐角色梯度记录位于全局裁剪之后，不能直接解释为Adam更新贡献。</p>'
    comparison=(ROOT/'analysis/OFFICIAL_COMPARISON_FRAGMENT.html').read_text()if(ROOT/'analysis/OFFICIAL_COMPARISON_FRAGMENT.html').exists()else''
    body=f'''<div class="aug-migration"><style>.aug-migration{{font:15px/1.8 system-ui;color:#203448}}.aug-migration section{{background:white;border:1px solid #d7e2ec;border-radius:14px;padding:22px;margin:16px 0}}.aug-migration table{{border-collapse:collapse;width:100%}}.aug-migration td,.aug-migration th{{padding:10px;text-align:left;border-bottom:1px solid #dbe3ec}}.aug-migration h1{{font-size:26px}}.aug-migration .cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}.aug-migration .cards b{{display:block;font-size:23px;color:#125a83}}</style>
    <section><small>当前研究 · D46 AUG迁移修正 · {time.strftime('%Y-%m-%d %H:%M UTC',time.gmtime())}</small>
    <h1>先把 JaT 迁移做准确，再判断部署收益</h1><div class="cards"><div>D系最高完整PDMS<b>{best:.8f}</b>{best_name}</div><div>当前正式训练<b>{active}组 / {cards}卡</b>阶段：{html.escape(state['stage'])}</div><div>最新完整保存<b>{native.get('updates','—')} Adam</b>每100次保存；最近10份＋评测/最佳固定</div></div>
    <p>正式训练全rank共同提交：{common}次Adam。旧FPO及D44/D46 RL已终止；225个旧checkpoint已删除，代码、日志、完整结果保留。IL46父权重与D IL训练保留。历史项目IL70/SafetyEP最高91.45359756，父模型不同。</p>{secondary_html}</section>
    <section><h2>当前实现审查与修复</h2>{findings}{health_html}</section>
    {comparison}<section><h2>这次改了什么</h2><table><tr><th>环节</th><th>新主线 {html.escape(route['experiment'])}</th></tr>
    <tr><td>初始化</td><td>相同完整D IL46，新Adam0；不继承资格或其它RL权重</td></tr>
    <tr><td>采样</td><td>真实scene-token根；原生K10/shift5前7步＋末3步官方JaT随机核；B64/G16/M2</td></tr>
    <tr><td>探索与数值</td><td>全TRAIN逐航点范围换算到原IL坐标；采样/ratio/KL同一协方差；AUG7训练、reference、部署统一使用FP32文本C/P与动作计算，冻结视觉计算保留原设置。AUG6仅最终动作投影FP32，已保存停训。</td></tr>
    <tr><td>优势</td><td>{html.escape(credit_description)}</td></tr>
    <tr><td>约束与优化</td><td>原IL JaT参考KL beta0.1；clip0.2；LR1e-6，曝光warmup.08/hold.60/cosine4E/floor.05</td></tr>
    <tr><td>预算</td><td>先3个场景曝光epoch，按完整分数决定延长；不是无限无条件续训</td></tr></table>
    <p>这是D适配的动作JaT主线，不宣称完整复现CoT-GRPO或论文Stage II。本轮暂不叠加教师/历史损失；部署路径保持作为后续独立改进。</p></section>
    <section><h2>当前结果与判据</h2><table><tr><th>完整评测点</th><th>PDMS</th><th>相原IL46 / 状态</th><th>相同数值u0</th><th>完整性</th></tr>{''.join(rows)}</table>
    <p>FP32投影可能改变0步部署输出，因此额外全量评测u0；必须同时看相原IL46和相新数值u0的增益。预定u0/200/400/800及0.5/1/2/3E评测，固定单条ODE、无oracle、官方评分不变。未齐全时不发布均分。</p>
    <p>重点观察零分修复、新增零分、NC/TTC与EP变化；真实分数尚未出来时，不把CPU/GPU资格通过写成提分。</p></section>
    <details><summary>已完成／旧路线的完整评测</summary><table><tr><th>检查点</th><th>PDMS</th><th>相IL46</th><th>相同数值u0</th><th>完整性</th></tr>{''.join(archived_rows)}</table></details>
    <section><h2>代码与归档</h2><p><a href="/mnt/project/d46-fpo-two-routes-20261104/analysis/AUG_MIGRATION_REVIEW.html">AUG迁移问题分析</a> · <a href="/mnt/project/d46-fpo-two-routes-20261104/report.html">已终止FPO历史</a> · <a href="{ROOT}/operations/STATE.json">新任务状态</a></p><p>常规30分钟更新，完整新结果立即更新；训练、评测均只派发到本任务五容器白名单。</p></section></div>'''
    for name,value in [('current_fragment.html',body),('report.html','<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="1800"><title>D46 AUG迁移</title><body style="background:#eff4f8;max-width:1300px;margin:auto;padding:20px">'+body+'</body></html>')]:
        p=ROOT/name;tmp=p.with_suffix('.tmp');tmp.write_text(value);os.replace(tmp,p)
    atomic_json(Path('/mnt/project/qwen-world-rl-optimization-20261030/reporting/refresh_request.json'),
        dict(id='AUG-migration-'+str(time.time_ns()),at=time.time(),event='training_stage_complete',scientific_result=False,evidence=str(ROOT/'operations/STATE.json')))


def main():
    op=ROOT/'operations';plan=read(op/'PLAN.json');route=plan['route']
    E.TASK=TASK;E.SOURCE=SOURCE;E.SPARE_GPUS={host:tuple(range(8))for host in plan.get('evaluation_hosts',['training-vlawm-zt'])}
    E.ALLOW_NATIVE_TRAINING_SHARE=bool(plan.get('evaluation_share_native_training',False))
    E.state_path=lambda root:op/'EVAL_STATE.json';E.discover=discover;E.launch=evaluation_launch
    E.il_baseline=lambda root:read(PARENT/'PARENT_CONTRACT.json')['complete_navtest']['original_evaluation']
    complete_events.notify=notify
    with(op/'owner.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        atomic_json(op/'OWNER.json',dict(pid=os.getpid(),start_ticks=int(Path('/proc/self/stat').read_text().split()[21]),source=str(SOURCE),at=time.time()))
        state=read(op/'STATE.json',dict(stage='qualification_first',jobs=read(op/'QUALIFICATION_FIRST_LAUNCHES.json'),at=time.time()))
        last_report=0;previous=None
        while not(op/'STOP_OWNER').exists():
            try:
                if state['stage']=='qualification_first'and finished(state['jobs']):
                    check_qualification(route,1);state.update(stage='qualification_resume',jobs=launch_stage(route,'qualification_resume'))
                elif state['stage']=='qualification_resume'and finished(state['jobs']):
                    check_qualification(route,2);state.update(stage='formal',jobs=launch_stage(route,'formal'))
                elif state['stage']=='formal':
                    proof_root=Path(route.get('proof_directory',op))
                    if not(proof_root/'FRESH_IL46_ADAM0.json').exists()and (Path(route['formal_output'])/'checkpoints/update000000_inner0/COMPLETE').exists():
                        atomic_json(proof_root/'FRESH_IL46_ADAM0.json',review_native(formal_route(route),route['formal_output'],0,False))
                    if finished(state['jobs']):state['stage']='formal_finished'
                state['at']=time.time();atomic_json(op/'STATE.json',state)
            except Exception:
                state.update(stage='REVIEW_REQUIRED',error=traceback.format_exc(),at=time.time());atomic_json(op/'STATE.json',state)
            secondary=plan.get('secondary_route')
            if secondary:
                ss=state.setdefault('secondary',dict(stage='WAITING_EVALUATION',jobs=[]))
                try:
                    if ss['stage']in('qualification_first','qualification_resume','formal')and all_refused_before_start(ss['jobs']):
                        attempt=ss.get('admission_retry',0)+1
                        ss.setdefault('unstarted_attempts',[]).append(ss['jobs'])
                        retry=dict(secondary,job_namespace=secondary['job_namespace']+f'/resource_retry{attempt:03d}')
                        ss.update(jobs=launch_stage(retry,ss['stage']),admission_retry=attempt)
                    if ss['stage']=='WAITING_EVALUATION':
                        current=read(op/'EVAL_STATE.json',E.initial_state())
                        ready=all(current['cases'].get(route['experiment']+f'_u{u:06d}',{}).get('status')=='COMPLETE' for u in secondary.get('wait_for_primary_evaluations',[200,400]))
                        if ready:ss.update(stage='qualification_first',jobs=launch_stage(secondary,'qualification_first'))
                    elif ss['stage']=='qualification_first'and finished(ss['jobs']):
                        check_qualification(secondary,1);ss.update(stage='qualification_resume',jobs=launch_stage(secondary,'qualification_resume'))
                    elif ss['stage']=='qualification_resume'and finished(ss['jobs']):
                        check_qualification(secondary,2);ss.update(stage='formal',jobs=launch_stage(secondary,'formal'))
                    elif ss['stage']=='formal':
                        proof=Path(secondary['proof_directory'])/'FRESH_IL46_ADAM0.json'
                        if not proof.exists()and(Path(secondary['formal_output'])/'checkpoints/update000000_inner0/COMPLETE').exists():
                            atomic_json(proof,review_native(formal_route(secondary),secondary['formal_output'],0,False))
                        if finished(ss['jobs']):ss['stage']='formal_finished'
                except Exception:ss.update(stage='REVIEW_REQUIRED',error=traceback.format_exc())
                atomic_json(op/'STATE.json',state)
                if ss['stage']!='WAITING_EVALUATION':
                    for host in secondary['hosts']:
                        E.SPARE_GPUS[host]=tuple(g for g in range(8)if g not in secondary.get('gpus',range(8)))
            evaluations=read(op/'EVAL_STATE.json',E.initial_state())
            try:
                evaluations=E.tick(ROOT,evaluations,plan,ops_source=SOURCE)
                compare_numeric_baselines(evaluations)
                complete=[c for c in evaluations['cases'].values()if c['status']=='COMPLETE']
                pending=[c['binding']['checkpoint']for c in evaluations['cases'].values()if c['status']not in E.TERMINAL]
                if complete:pending.append(max(complete,key=lambda c:read(c['summary'])['PDMS'])['binding']['checkpoint'])
                for item in registered_routes(plan):
                    pending.append(str(Path(item['formal_output'])/'checkpoints/update000000_inner0'))
                    if Path(item['formal_output']).exists():rotate(item['formal_output'],keep=10,pinned=pending)
                error=op/'EVALUATION_ERROR.json'
                if error.exists():
                    history=op/'resolved_errors';history.mkdir(exist_ok=True)
                    os.replace(error,history/f'{time.time_ns()}.json')
            except Exception:atomic_json(op/'EVALUATION_ERROR.json',dict(at=time.time(),error=traceback.format_exc()))
            signature=(state['stage'],state.get('secondary',{}).get('stage'),tuple((c['id'],c['status'])for c in evaluations['cases'].values()))
            if signature!=previous or time.time()-last_report>=1800:
                report(state,evaluations);last_report=time.time();previous=signature
            paths=[op,Path(route['qualification_output']),Path(route['formal_output']),Path(route['formal_output'])/'checkpoints',Path(route['formal_output'])/'evaluation_requests']
            paths += [Path(j['job'])for j in state['jobs']]
            if secondary:
                paths += [Path(secondary['formal_output']),Path(secondary['formal_output'])/'checkpoints',Path(secondary['qualification_output'])]+[Path(j['job'])for j in state['secondary']['jobs']]
            paths += [Path(c['output'])for c in evaluations['cases'].values()]
            paths += [Path(j[-1]['job'])for c in evaluations['cases'].values()for j in c['jobs'].values()if j]
            wait_for_events(paths,60,ignore=('STATE.json','EVAL_STATE.json','OWNER.json','report.html'))


if __name__=='__main__':main()
