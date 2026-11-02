"""Read-only living performance report. No GPU imports, dispatch or score creation."""
from datetime import datetime, timezone
import html
import hashlib
import json
import os
from pathlib import Path

ROOT=Path('/mnt/project/qwen-world-performance-rl-20261103')
esc=lambda x:html.escape(str(x),quote=True)


def read(path, default=None):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return default


def atomic_text(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.writing-report');tmp.write_text(value,encoding='utf-8');os.replace(tmp,path)


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


def snapshot(root=ROOT):
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
    complete=[]
    for p in (root/'evaluations').glob('**/SUMMARY.json'):
        r=read(p,{})
        body=dict(r);saved=body.pop('sha256',None)
        if (r.get('status')=='COMPLETE' and r.get('invalid_scenes')==0 and r.get('scenes')==12146
                and r.get('expected_scenes')==12146 and r.get('physical_logs')==44
                and hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()==saved):
            complete.append(dict(r,evidence=str(p)))
    return dict(generated_UTC=datetime.now(timezone.utc).isoformat(),jobs=jobs,complete_results=complete,
        active_formal=sum(j['status']=='RUNNING' and not j['qualification'] for j in jobs),
        active_qualification=sum(j['status']=='RUNNING' and j['qualification'] for j in jobs),
        audit=read(root/'data/AUDIT.json',{}),registry=read(root/'RUNS.json',{}),
        native_owner=read(root/'owners/PERFORMANCE_NATIVE/STATE.json',{}),
        deadline=read(root/'operations/RESOURCES_UNTIL_20261103T000000Z/REQUEST.json',{}),
        resume=read(root/'operations/NATIVE_RESUME_QUALIFICATION.json',{}),
        nominal=read(root/'qualification/REAL_NOMINAL_IL70/COMPLETE.json',{}))


def table(headers,rows):
    return '<div style="overflow:auto"><table style="width:100%;border-collapse:collapse"><thead><tr>'+''.join('<th>'+esc(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td style="padding:9px;border-bottom:1px solid #dce3eb">'+esc(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def render_fragment(root=ROOT):
    s=snapshot(root);active=[j for j in s['jobs'] if j['status'] in ('RUNNING','DISPATCHED')]
    retired=[j for j in s['jobs'] if j not in active]
    checks=s['audit'].get('checks',{})
    best=max(s['complete_results'],key=lambda r:r['PDMS'],default=None)
    best_label=('%.8f · '%best['PDMS']+best['evaluation'].split('/')[-2]) if best and best['PDMS']>91.45359756166451 else '91.45359756 · SafetyEP Control 10k'
    body='<section id="performance-current" style="background:#fff;border:1px solid #cad9e8;border-radius:14px;padding:24px;margin:20px 0"><small>新一轮 · PERFORMANCE FIRST · '+esc(s['generated_UTC'])+'</small><h2>最高完整 PDMS：'+esc(best_label)+'</h2>'
    body+='<p><b>本轮正式在训 '+str(s['active_formal'])+' 条；资格测试 '+str(s['active_qualification'])+' 个。</b>旧实验独立继续。新算法完整 NAVTEST 结果 '+str(len(s['complete_results']))+' 个；资格测试与候选 oracle 不计为提分。</p>'
    body+='<p><b>当前任务：</b>B128/G16 工程资格 → S1 高分候选向真实 B1 吸收；S5 分阶段覆盖、吸收与持续保持。各路线均从相同 IL70k（88.79290991、SHA 7dda5b14…）和 Adam0 独立开始。</p>'
    body+=table(['正在执行','类型','服务器','Adam更新','有效曝光 epoch','状态'],[(j['id'],'资格' if j['qualification'] else '正式训练',j['host'],j['updates'],j['exposure_epoch'] if j['exposure_epoch'] is not None else '等待首条更新',j['status']) for j in active])
    if s['native_owner'].get('runs'):
        body+=table(['研究路线','当前状态','服务器','最近封存更新'],[(arm,v.get('status'),v.get('host','待分配'),v.get('updates',0)) for arm,v in s['native_owner']['runs'].items()])
    if s['complete_results']:
        body+='<h3>本轮完整单轨迹评测</h3>'+table(['评测','Adam','PDMS','有效场景','invalid'],[(r['evaluation'].split('/')[-2],r.get('updates'),r.get('PDMS'),r['scenes'],r['invalid_scenes']) for r in s['complete_results']])
    body+='<p><b>资源期限：</b>wm4 / rl2 / tsh2 / wmzt / tsh3 / wm2 至北京时间 11月3日08:00；23:45 UTC 起保存退出。其它授权按原许可执行，未知任务不回收。</p>'
    body+='<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px"><article><h3>已完成：全量数据审计</h3><p>'+esc(checks.get('unique_tokens','待查'))+' 唯一 TRAIN；NAVTEST token/log 交集 0。309864 图像文件、103288 cache 文件逐个筛查无缺失；cache 格式筛查不等同完整语义评分。</p></article><article><h3>尚待回答：是否真正提分</h3><p>重点看固定 TRAIN 中 sampled-good/B1-bad 的后续修复，以及全 NAVTEST 的零分修复和高分退化。不会以扩大采样覆盖率代替单轨迹部署成绩。</p></article></div>'
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
    atomic_text(root/'report_snapshot.json',json.dumps(s,ensure_ascii=True,indent=2))
    css='body{font-family:system-ui,sans-serif;background:#f1f5f9;color:#203348;max-width:1450px;margin:24px auto;padding:0 20px;line-height:1.6}th{text-align:left;padding:9px;background:#e8eff7}h2{color:#126b57}a{color:#1763a1}'
    atomic_text(root/'report.html','<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="1800"><title>Qwen-World Performance RL</title><style>'+css+'</style><body>'+body+'</body></html>')
    return body


if __name__=='__main__':render_fragment()
