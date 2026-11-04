"""CPU-only current-round report, with explicit lineage and training-only teachers."""
from datetime import datetime, timezone
import html
import json
import math
from pathlib import Path
import time

ROOT = Path('/mnt/project/d46-fpo-two-routes-20261104')
IL_SCORE = 89.4955209758546
HISTORICAL_D = 90.32032057511778
STYLE = """
.fpo-current{--ink:#172b44;--muted:#5d6d82;color:var(--ink);font:15px/1.75 system-ui,sans-serif}
.fpo-current *{box-sizing:border-box}.fpo-current .hero{background:linear-gradient(120deg,#e9f2ff,#edf9f6);border:1px solid #bad1e8;border-radius:18px;padding:25px}
.fpo-current h1{font-size:27px;line-height:1.4;margin:8px 0 12px}.fpo-current h2{font-size:20px;color:#172b44;margin:0 0 12px}.fpo-current h3{font-size:17px;margin:10px 0}
.fpo-current .eyebrow{font-size:12px;letter-spacing:1px;color:#46617f}.fpo-current .muted{color:var(--muted);font-size:13px}.fpo-current .cards{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:18px 0}
.fpo-current .card,.fpo-current .panel{background:#fff;border:1px solid #d8e2ee;border-radius:14px;padding:18px}.fpo-current .card strong{display:block;font-size:25px;line-height:1.4;color:#164b80}.fpo-current .card small{display:block;color:var(--muted)}
.fpo-current .panel{margin:18px 0}.fpo-current .grid2{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.fpo-current .grid2 .panel{margin:0}.fpo-current .scroll{overflow:auto}.fpo-current table{width:100%;border-collapse:collapse;font-size:13px;line-height:1.6}.fpo-current th,.fpo-current td{text-align:left;vertical-align:top;padding:11px;border-bottom:1px solid #e3eaf2}.fpo-current th{background:#edf3fa;white-space:nowrap}.fpo-current tr:last-child td{border-bottom:0}
.fpo-current .tag{display:inline-block;padding:2px 8px;border-radius:6px;background:#e8f5ef;color:#196545;font-size:12px}.fpo-current .note{border-left:4px solid #77a3c9;background:#f0f6fc;padding:12px 16px;margin:14px 0}.fpo-current .formula{font:13px/1.8 ui-monospace,monospace;background:#f2f5f9;border-radius:8px;padding:12px;overflow:auto}.fpo-current a{color:#1767a5}.fpo-current details{margin:14px 0}.fpo-current summary{cursor:pointer;font-weight:600}.fpo-current svg{width:100%;height:auto;min-width:400px;display:block}.fpo-current .decision{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.fpo-current .decision div{padding:14px;background:#f3f7fc;border-radius:10px}
.rl-history{margin:25px 0;padding:20px;border:1px solid #d2dce8;background:#f8fafc;border-radius:14px}.rl-history>summary{cursor:pointer;font-size:18px;font-weight:650;color:#425570}.rl-history[open]>summary{margin-bottom:18px}
@media(max-width:900px){.fpo-current .cards{grid-template-columns:repeat(2,minmax(0,1fr))}.fpo-current .grid2,.fpo-current .decision{grid-template-columns:1fr}.fpo-current .hero{padding:18px}.fpo-current h1{font-size:23px}}
"""


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError, TypeError):
        return default


def bounded_rows(path):
    """Only inspect a bounded tail at report refresh, never poll training logs."""
    try:
        p = Path(path)
        with p.open('rb') as f:
            f.seek(max(0, p.stat().st_size - 262144))
            lines = f.read().splitlines()[-64:]
    except OSError:
        return []
    rows = []
    for line in lines:
        try:
            rows.append(json.loads(line))
        except ValueError:
            pass
    return rows


def table(headers, rows):
    h = lambda v: html.escape(str(v), quote=True)
    return '<div class="scroll"><table><thead><tr>' + ''.join('<th>'+h(x)+'</th>' for x in headers) + '</tr></thead><tbody>' + ''.join('<tr>'+''.join('<td>'+h(x)+'</td>' for x in row)+'</tr>' for row in rows) + '</tbody></table></div>'


def diagram(verified):
    """Inline SVG has no CDN/JS dependency and remains readable in saved HTML."""
    prefix = 'verified' if verified else 'mature'
    out = [f'<svg viewBox="0 0 580 810" role="img" aria-labelledby="{prefix}-title"><title id="{prefix}-title">'+('改进FPO：基础更新、候选吸收与历史保持' if verified else '成熟FPO：保持现有算法，继续数据覆盖')+'</title><defs><marker id="'+prefix+'-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="#68839e"/></marker></defs>']

    def box(x, y, w, title, lines, tone='blue', height=78):
        fill, stroke = {'blue':('#edf4ff','#88acd6'),'green':('#edf8f1','#7dbb98'),'purple':('#f3efff','#ac98d0'),'gray':('#f4f6f9','#abb9c8')}[tone]
        out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{height}" rx="11" fill="{fill}" stroke="{stroke}"/>')
        out.append(f'<text x="{x+w/2}" y="{y+25}" text-anchor="middle" font-family="system-ui,sans-serif" font-size="15" font-weight="650" fill="#213952">{html.escape(title)}</text>')
        for i, line in enumerate(lines):
            out.append(f'<text x="{x+w/2}" y="{y+47+i*17}" text-anchor="middle" font-family="system-ui,sans-serif" font-size="12" fill="#4c6077">{html.escape(line)}</text>')

    def arrow(points):
        out.append(f'<polyline points="{points}" fill="none" stroke="#68839e" stroke-width="1.8" marker-end="url(#{prefix}-arrow)"/>')

    box(95, 12, 390, '相同 D IL46 原始父模型', ['完整官方 PDMS 89.49552098；同一 SHA'])
    arrow('290,90 290,110')
    if not verified:
        box(95, 112, 390, '本实验自身 FPO 1711 / inner1 接续', ['保留 Adam / scheduler / RNG / 待更新材料'])
        arrow('290,190 290,215')
        box(95, 217, 390, '全量 NAVTRAIN → B64 × G16', ['每 collection 64 个场景、1,024 条新候选'])
        arrow('290,295 290,320')
        box(95, 322, 390, '真实 PDM 评分 → SafetyEP 优势', ['组内安全/效率信用；批级 population std'])
        arrow('290,400 290,425')
        box(95, 427, 390, 'MC4 CFM energy → FPO clipped 更新', ['energy ratio 代理；clip 0.01；每组 M2'])
        arrow('290,505 290,530')
        box(95, 532, 390, 'FPO + 原 IL46 W3 reference KL', ['无新增教师吸收或 B1 历史保持辅助项'], 'gray')
        arrow('290,610 290,650')
    else:
        box(95, 112, 390, '独立新实验：IL46 + 新 Adam 0', ['资格权重不进入正式训练'])
        arrow('290,190 290,215')
        box(95, 217, 390, '相同 B64 / G16 / M2 主训练组', ['真实评分 → SafetyEP → MC4 FPO'])
        arrow('290,295 290,580')
        arrow('290,295 290,317 145,317 145,335')
        arrow('290,295 290,317 435,317 435,335')
        box(15, 337, 260, '低质量真实 B1：寻找更好教师', ['G16 不足时追加 G16；只作教师', '严格安全舒适，PDMS≥0.90，Δ≥0.02'], 'green', 90)
        box(305, 337, 260, '保留已验证的高质量 B1', ['初始 IL46 B1 + 自身历史 B1', '仅 TRAIN；同 run；容量 103,288'], 'purple', 90)
        arrow('145,427 145,449')
        arrow('435,427 435,449')
        box(15, 451, 260, '让真实部署吸收好候选', ['0.05 × CFM（MC2）', '+ 0.02 × 真实 ODE 末步 SmoothL1'], 'green', 90)
        box(305, 451, 260, '持续高分保持', ['0.05 × B1 末步保持', '退化≥0.02时系数增至0.25'], 'purple', 90)
        arrow('145,541 145,562 290,562 290,580')
        arrow('435,541 435,562 290,562')
        box(95, 582, 390, 'FPO + 原 reference KL + 两类辅助项', ['teacher 全部 detached；G32 不进入 FPO ratio'], 'blue', 64)
        arrow('290,646 290,667')
    box(95, 669, 390, '部署：仅输出一条 B1 ODE 轨迹', ['K10 / shift5 / seed2101；无评分选优'], 'gray')
    out.append('<text x="290" y="783" text-anchor="middle" font-family="system-ui,sans-serif" font-size="12" fill="#66788d">'+('验证目标：修复低分，同时减少原高分退化' if verified else '验证目标：成熟算法充分训练后能否继续提升')+'</text></svg>')
    return ''.join(out)


def build_fragment(plan, state, evaluation):
    h = lambda x: html.escape(str(x), quote=True)
    current, scores, progress, diagnostic, throughput = [], [], [], [], []
    for case in evaluation.get('cases', {}).values():
        summary = read(case.get('summary'), {})
        if case.get('status') == 'COMPLETE' and summary.get('status') == 'COMPLETE' and summary.get('scenes') == 12146 and summary.get('invalid_scenes') == 0 and summary.get('physical_logs') == 44:
            import hashlib
            body = dict(summary); seal = body.pop('sha256', None)
            if seal != hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':')).encode()).hexdigest():
                continue
            scores.append((case, summary))
    for key, route in plan.get('routes', {}).items():
        s = state.get('routes', {}).get(key, {})
        identity = read(Path(route['output'])/'identity.json', {})
        tails = [bounded_rows(Path(route['output'])/f'metrics_rank{rank}.jsonl') for rank in range(route['world'])]
        last = [rows[-1] if rows else {} for rows in tails]
        valid = all(row.get('run') == identity.get('training_run_id',identity.get('sha256')) and isinstance(row.get('updates'), int) for row in last)
        common = min(row['updates'] for row in last) if valid else None
        epoch = min(row.get('scene_epochs', 0) for row in last) if valid else None
        lag = time.time()-min(row.get('at', 0) for row in last) if valid else math.inf
        # End-to-end wall time from jointly committed updates, including collection
        # and scoring; never label the backward-only timer as training throughput.
        if valid:
            by_rank = [{r['updates']: r for r in rows if r.get('run') == identity.get('training_run_id', identity.get('sha256'))
                        and r.get('invocation') == last[i].get('invocation')} for i, rows in enumerate(tails)]
            common_updates = sorted(set.intersection(*(set(rows) for rows in by_rank)))[-33:]
            if len(common_updates) >= 3:
                first, end = common_updates[0], common_updates[-1]
                elapsed = max(rows[end]['at'] for rows in by_rank)-max(rows[first]['at'] for rows in by_rank)
                if elapsed > 0:
                    throughput.append((key, f'{first}–{end}', f'{elapsed/(end-first):.2f}',
                                       f'{3600*(end-first)/elapsed:.1f}', '同一正式进程，含采样/评分/同步与共享资源等待'))
        native = s.get('native', {})
        current.append((key, str(route['world'])+'卡 / '+str(len(route['hosts']))+'节点', s.get('status', '待状态收据'),
            str(common) if common is not None else '待共同提交', f'{epoch:.5f} E' if epoch is not None else '—',
            str(native.get('updates', '—')), '当前日志' if lag <= 600 else '日志超过10分钟，需复核'))
        route_scores = [(c, v) for c, v in scores if c['binding']['experiment'] == key]
        if route_scores:
            best = max(route_scores, key=lambda x: x[1]['PDMS'])
            progress.append((key, f'{best[1]["PDMS"]:.8f}', f'{best[1]["PDMS"]-IL_SCORE:+.6f}', '本轮完整评测 / Adam '+str(best[0]['binding']['updates'])))
        else:
            progress.append((key, '尚无本轮新完整分数', '—', '成熟线历史最佳90.32032058' if key == 'FPO_MATURE' else '独立IL46起步；等待400/800步早评'))
        if valid:
            row = min(last, key=lambda x: x['updates'])
            diagnostics = row.get('losses', {})
            diagnostic.append((key, f'{row.get("lr", [0])[0]:.3g}', f'{row.get("update_seconds", 0):.2f} 秒',
                f'{diagnostics["rescue_active"]*100:.1f}%' if 'rescue_active' in diagnostics else '未启用',
                f'{diagnostics["retention_active"]*100:.1f}%' if 'retention_active' in diagnostics else '未启用',
                row.get('extra_B1_queries', 0), row.get('adaptive_G32_queries', 0)))
    highest = max([HISTORICAL_D]+[v['PDMS'] for _, v in scores])
    best_new = max(scores, key=lambda point: point[1]['PDMS'], default=None)
    highest_label = (best_new[0]['binding']['experiment']+' / Adam '+str(best_new[0]['binding']['updates'])) if best_new and highest > HISTORICAL_D else 'FPO_MATURE 历史 Adam 1614'
    actual = state.get('actual_training', 0)
    cards = [('D系最高完整 PDMS', f'{highest:.8f}', highest_label),
        ('同父 IL46 基线', f'{IL_SCORE:.8f}', f'当前D系最佳增益 {highest-IL_SCORE:+.6f}'),
        ('实际在训 / 训练卡', f'{actual} 组 / '+str(sum(v['world'] for v in state.get('routes', {}).values() if v.get('status') == 'RUNNING'))+' 卡', '最多两条正式训练；不含评测卡'),
        ('本轮新完整评测', str(len(scores))+' 个', '只发布12,146场景全部完成的分数')]
    body = '<style>'+STYLE+'</style><div class="fpo-current" id="D46-FPO-current"><header class="hero"><div class="eyebrow">当前研究 · D框架 / IL46 · FPO 两条路线</div><h1>先看当前实验、进度与真实分数</h1><p>一条继续成熟 FPO，检验充分训练的收益；另一条让随机找到的好轨迹进入真实单轨迹策略，同时保持已有高分能力。</p><p class="muted">报告生成 '+datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')+'；状态账本 '+datetime.fromtimestamp(state.get('at', 0), timezone.utc).strftime('%H:%M:%S UTC')+'。常规30分钟更新，完整新评测立即更新。</p></header>'
    body += '<div class="cards">'+''.join('<div class="card"><small>'+h(title)+'</small><strong>'+h(value)+'</strong><small>'+h(note)+'</small></div>' for title, value, note in cards)+'</div><p class="muted">历史全项目最高：IL70 / SafetyEP 91.45359756；它与当前 D IL46 的父模型不同，历史详情在页末折叠。</p>'
    engineering=read(ROOT/'engineering/STATUS.json',{})
    if engineering.get('formal_training')==0 and engineering.get('no_formal_resume')is True:
        body=body.replace('先看当前实验、进度与真实分数','当前：RL提速开发；正式训练已暂停')
        body+='<section class="panel"><h2>当前工程任务与实际速度</h2><p>两条正式路线已原生保存：成熟1921/inner1、改进47/inner1。当前只运行有界实现测试。每100步保存完整恢复状态；最近10个恢复点轮换，另保留最佳及正在评测的检查点。</p>'
        body+=table(['原实际训练配置','完整周期 秒/Adam','实际更新/小时','统计区间'],[(key,f'{v["end_to_end_seconds_per_update"]:.2f}',f'{v["updates_per_hour"]:.1f}',str(v['updates']))for key,v in engineering.get('baseline_throughput',{}).items()])
        def latest_results(pattern):
            latest={}
            for path in sorted((ROOT/'engineering').glob(pattern),key=lambda p:p.stat().st_mtime):
                data=read(path,{})
                if data.get('status')=='COMPLETE':latest[path.parent.name]=path
            return latest.values()
        fb=[];collection=[];cycles=[]
        for path in latest_results('FB_v*/*/RESULT.json'):
            data=read(path,{})
            for row in data.get('rows',[]):
                comp=row.get('comparison',{})
                fb.append((path.parent.name+' / '+path.parent.parent.name,row['mode'],f'{row["median_seconds"]:.3f}',f'{comp.get("gradient_relative_L2",0):.3%}',comp.get('bitexact'),data.get('status')))
        for path in latest_results('COLLECTION_v*/*/RESULT.json'):
            data=read(path,{})
            for row in data.get('rows',[]):collection.append((path.parent.name,row['mode'],f'{row["seconds_per_scene"]:.3f}',str(row.get('prefix_builds')),row.get('identical_main_sampler_RNG')))
        for path in latest_results('CYCLE_v*/*/RESULT.json'):
            data=read(path,{})
            for row in data.get('rows',[]):
                cycles.append((path.parent.name,str(data['world'])+'卡',row['mode'],f'{row["seconds_per_update"]:.2f}',
                    f'{row["updates_per_hour"]:.1f}',f'{row["updates_per_hour_with_save100"]:.1f}',row['native_roundtrip_exact']))
        body+='<p>下表仅展示各路线最新一轮完整测量。局部测试与完整周期分别统计；有限周期的小时速度是折算值，正式长时间吞吐仍待验证。</p>'
        body+='<h3>完整多卡周期：优先判断实际收益</h3>'+table(['路线','规模','执行方式','秒 / Adam','折算更新 / 小时','计入100步保存','原生恢复一致'],cycles)
        body+='<h3>同材料前向反向：局部测量，不等于完整训练加速</h3>'+table(['路线 / 轮次','执行方式','秒 / 实际rank更新','梯度相对L2差异','梯度逐值一致','阶段'],fb)
        body+='<h3>真实采样与官方评分：单独测量</h3>'+table(['路线','执行方式','秒 / 场景','C/P构建次数','采样RNG一致'],collection)
        body+='<p>优先路线：同一策略收集期间复用无梯度C/P；删除重复采样均值验证前向；批量计算MC4/教师辅助项；减少逐标量GPU同步。当前可训练C/P仍须每次inner更新重新构图。MC合并可能改变BF16数值，需按实测梯度差异判断；完整分布式周期、恢复与单轨迹质量通过后再决定正式配置。</p></section>'
    body += '<section class="panel"><h2>1. 当前正式训练</h2>'+table(['路线', '资源', '执行状态', '全rank共同 Adam', '实际有效曝光', '最近完整保存 Adam', '日志时效'], current)
    body += '<p class="muted">Adam与曝光取所有rank当前日志的最小值；保存点可能落后于训练，不将已保存0步理解为尚未开始。B64 / G16 / M2：每次收集64个场景、1,024条新候选，再做2次Adam；不是1,024个场景。</p>'+table(['路线', '当前完整PDMS', '相同父IL46增益', '分数归属'], progress)+'</section>'
    body += '<section class="panel"><h2>正式运行速度与保存</h2>'+table(['路线', '最近共同更新区间', '完整周期秒/更新', '折算更新/小时', '口径'], throughput)
    body += '<p>每100次Adam更新保存完整原生状态，最近10个恢复点轮换；另保留最佳、最新和正在评测的检查点。当前只使用最新五容器白名单；本机D IL继续运行，未知GPU负载不抢占。表中速度来自当前正式运行，会受并行评测影响，不能与独占工程测量直接计算加速比。</p></section>'
    body += '<section class="panel"><h2>2. 评测进展与现阶段结论</h2>'+table(['评测点', '阶段', '完整分片', '运行分片'], [(c['id'], c['status'], str(c.get('completed_shards', 0))+'/32', c.get('active_shards', 0)) for c in evaluation.get('cases', {}).values()])
    if not scores:
        body += '<div class="note"><b>两条正式训练均已产生参数更新；尚不能判断改进线是否提分。</b>八卡保存恢复与真实教师梯度校准已经通过，说明机制已接入训练。收益判断仍等待完整部署评测；训练采样最高分与部分评测均值不作为PDMS成绩。</div>'
    else:
        body += table(['新完整点', 'Adam', 'PDMS', '相IL46'], [(c['id'], c['binding']['updates'], f'{s["PDMS"]:.8f}', f'{s["PDMS"]-IL_SCORE:+.6f}') for c, s in scores])
    body += '<p>固定单条 B1 ODE K10 / shift5 / seed2101，12,146场景、44个物理logs、invalid=0。B1指真实部署直接输出的一条轨迹；G16/G32只在训练阶段用于采样和教师发现。NAVTEST标签不进入训练。</p></section>'
    body += '<section class="panel"><h2>3. 两条路线的框图</h2><p>蓝色是共有的FPO主更新，绿色是新增候选吸收，紫色是新增高质量保持。两条路线独立训练，不互传RL权重或Adam状态。</p><div class="grid2"><div><h3>A · FPO_MATURE：延长成熟路线</h3><div class="scroll">'+diagram(False)+'</div></div><div><h3>B · FPO_VERIFIED：吸收好轨迹＋防退化</h3><div class="scroll">'+diagram(True)+'</div></div></div></section>'
    differences = [
        ('要解决的问题', '原FPO继续覆盖更多数据，后期能否持续提升', 'G组有安全高分轨迹，B1却未学会；新能力与旧高分相互损伤'),
        ('本轮起点', '自身 FPO 1711 / inner1；完整继承Adam、scheduler、RNG和pending材料', '同一个完整IL46，新Adam 0；不继承成熟线或资格权重'),
        ('共同的主算法', 'SafetyEP + 批级std；MC4 CFM energy FPO；clip=0.01；原W3参考KL', '相同主算法；额外教师轨迹不参与FPO ratio'),
        ('场景/候选与更新', 'B64 / G16 / M2；当前资源见首表', 'B64 / G16 / M2；困难场景最多额外16条教师候选'),
        ('额外真实评分', '主组G16；无逐场景B1教师探针', '当前B1 + 首次访问的冻结IL46 B1；必要时额外G16'),
        ('新教师从哪里来', '无新增教师辅助损失', '同一TRAIN场景的当前G16/额外G16；NC/DAC/TTC/Comfort均1，PDMS≥0.90，且比当前B1高≥0.02'),
        ('怎样吸收好轨迹', '仅通过FPO优势更新', '0.05×正向CFM（MC2，sigma 0.1–0.9）＋0.02×真实B1最后一步SmoothL1'),
        ('如何保持旧高分', '冻结IL46的原参考KL约束', '保留原参考KL；另保持同run初始IL46/自身历史的真实strict-high B1，容量103,288；基础系数0.05，回退≥0.02时0.25'),
        ('学习率与训练范围', 'LR 2e-6；warmup .08E / hold .60E / cosine到4E / floor 5%；r16文本attention LoRA 6.42M', '相同注册调度与训练范围；独立曝光时钟，当前LR见下表'),
        ('部署', '一条原生B1 ODE；无重排', '相同；教师、历史bank、评分器不参与部署'),
        ('当前证据', '本轮起点1711完整PDMS 90.35363365（+0.85811267）；最新完整点见上表', '八卡真实更新、pending恢复、教师梯度已验证；新部署收益以完整评测为准'),
    ]
    body += '<section class="panel"><h2>4. 逐项区别与损失设计</h2>'+table(['比较项', 'A · 成熟FPO', 'B · 改进FPO'], differences)+'<p class="formula">A: L = L_FPO + beta × KL_W3(当前策略, 冻结IL46)<br>B: L = L_A + I_rescue × (0.05 L_CFM + 0.02 L_ODE-last) + alpha_history L_B1-keep<br>alpha_history = 0（无有效教师）；0.05（有教师）；0.25（相教师回退≥0.02）</p><p class="muted">以上辅助系数是原始损失系数，不是归一化梯度份额，也不是beta50预算。FPO energy ratio是本地策略更新代理；不能称完整轨迹似然比。ODE末步用真实部署链末次输入，之前状态停止梯度；没有端到端回传整条ODE。</p></section>'
    proof = read(ROOT/'operations/VERIFIED_Q2_PROOF_AND_GRADIENT_CALIBRATION.json', {})
    body += '<section class="panel"><h2>5. 机制是否真正启用</h2>'+table(['路线', '当前LR', '最近纯更新/Adam', '本地救援场景比例', '本地保持比例', '累计额外B1评分', '累计额外G32评分'], diagnostic)
    body += '<p class="muted">这里只展示最慢rank最后一次更新的本地辅助激活率；累计评分计数是全局训练时钟。纯更新耗时包含每rank当前工作量，两条卡数不同，不能当同资源加速对比；不包含整个收集与评分周期。</p>'
    if proof.get('status') == 'PASS':
        g = proof['gradient_calibration']
        body += '<p>八卡资格：真实 pending inner1 → READY2 恢复通过。8个实际TRAIN教师场景中，加权CFM/FPO梯度范数比中位数 <b>'+f'{100*g["weighted_CFM_to_FPO_median"]:.2f}%</b>，ODE末步/FPO为 <b>'+f'{100*g["weighted_ODE_to_FPO_median"]:.2f}%</b>。初始自身保持梯度为0符合相同策略/目标预期；这不是后续保持有效的证据。</p>'
    body += '</section><section class="panel"><h2>6. 接下来怎样判断与调整</h2><div class="decision"><div><b>先看真实单轨迹收益</b><br>成熟线1921/2000/2400/2800步、改进线200/400/800步早评，随后按曝光门触发。原生目标6E，1/2/3E持续复核，不因到3E自动停训。</div><div><b>区分修复与退化</b><br>逐场景配对零分修复、新增零分、旧高分回落；确认教师激活能转化为B1改善。</div><div><b>按证据倾斜资源</b><br>完整分数改善且退化可控再扩大预算。明确崩溃或持续严重遗忘则保存后复核；不依据资格loss或局部oracle择优。</div></div><p class="muted">历史整个项目的91.45359756来自IL70/SafetyEP，父模型与此处D IL46不同；其详细实验放在页末折叠历史区。</p><p><a href="'+h(ROOT/'operations/STATE.json')+'">训练状态</a> · <a href="'+h(ROOT/'operations/EVAL_STATE.json')+'">评测队列</a> · <a href="'+h(ROOT/'operations/PLAN.json')+'">注册路线</a> · <a href="'+h(ROOT/'report.html')+'">两路线独立报告</a></p></section></div>'
    return body


def document(fragment):
    return '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="1800"><title>D46 FPO 当前训练与结果</title><style>body{margin:0 auto;padding:24px 18px;max-width:1450px;background:#f1f5f9}</style><body>'+fragment+'</body></html>'
