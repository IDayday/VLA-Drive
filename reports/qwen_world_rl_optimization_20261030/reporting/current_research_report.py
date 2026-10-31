"""Current-first RL report presentation. Reads metadata; never controls jobs."""
from __future__ import annotations

import json
from pathlib import Path

from research_report import figure, update

CONTROL = 'PPO_SAFETY_EP'
ROUTES = {
    'PPO_SAFETY_SUCCESS_MEMORY': dict(
        name='成功经验复用', short='Success memory', color='#2563eb', owner='safety_fresh_pair',
        problem='保住已经学会的高分行为，减少训练过程中的能力退化。',
        method='在SafetyEP上加入当前run真实验证成功轨迹的独立历史场景回放；每4次采集做一次flow matching辅助学习。',
        paper=('Self-Imitation Learning', 'https://arxiv.org/abs/1806.05635')),
    'PPO_SAFETY_FPO': dict(
        name='Flow策略优化目标', short='FPO', color='#7c3aed', owner='safety_fresh_pair',
        problem='检查随机转移上的优化信号，能否更有效地转化为最终轨迹质量。',
        method='用终点轨迹的条件flow matching损失差构造PPO代理目标；每候选2次MC采样，时间范围[0.2,0.8)。',
        paper=('FPO', 'https://arxiv.org/abs/2507.21053')),
    'PPO_FPO_FULLTIME': dict(
        name='Flow时间覆盖对照', short='FPO upper 1.0', color='#d97706', owner='fpo_fulltime',
        problem='检验训练未覆盖的高噪声时间区域，是否影响部署ODE轨迹。',
        method='相对FPO仅将采样时间上界0.8扩到1.0，下界仍为0.2；其余优化、数据和探索设置相同。',
        paper=('FPO／FPO++', 'https://arxiv.org/abs/2602.02481')),
    'PPO_SAFETY_FEASIBLE_SCORE': dict(
        name='可行轨迹质量信用', short='Feasible quality', color='#dc2626', owner='feasible_score',
        problem='为EP已经满分、但官方轨迹质量仍有差异的候选组恢复学习信号。',
        method='保留安全可行性门控，门内质量EP改为q=(5EP+2Comfort)/7；安全与全失败fallback保持，EP仍占主要质量权重。',
        paper=None),
}

CSS = '''
.research-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px;margin:20px 0}
.research-card{border:1px solid #dce6ef;border-top:4px solid var(--route);border-radius:11px;padding:20px;background:#fff;min-width:0}
.research-card h3{margin:0;font-size:19px}.card-head{display:flex;gap:12px;align-items:center;justify-content:space-between}
.run-id{display:block;overflow-wrap:anywhere;font:11px/1.6 ui-monospace,monospace;color:#6b7d8b;margin:7px 0 12px}
.badge{font-size:11px;border-radius:20px;background:#eaf6ef;color:#206843;padding:3px 10px;white-space:nowrap}
.progress-title{display:flex;justify-content:space-between;gap:10px;font-size:13px;margin-top:15px}
.progress-track{height:7px;background:#eaf0f6;border-radius:10px;overflow:hidden;margin:7px 0 9px}.progress-fill{height:100%;background:var(--route)}
.result-pair{display:grid;grid-template-columns:1fr 1fr;gap:12px;border-top:1px solid #e7edf3;padding-top:12px;margin-top:14px}
.result-pair strong{font-size:23px;display:block;line-height:1.5}.result-pair small{font-size:11px}.pending{font-size:14px!important;color:#697b8a;font-weight:500}
.eval-line{background:#f1f5fa;border-radius:7px;padding:8px 11px;font-size:12px;margin-top:12px}
.research-card p{font-size:13px;margin:8px 0}.positive{color:#137550}.negative{color:#b5473c}.analysis-route{border-left:3px solid var(--route);padding:0 16px;margin:20px 0}
.analysis-route h3{margin:0 0 6px}.analysis-route p{font-size:14px}.analysis-next{color:#506b7b}
.compact-notice{padding:10px 15px;background:#f2f6fa;border-radius:8px;font-size:13px}
.background-chapter>summary{font-size:17px;padding:5px}.background-chapter>section{border:0;padding:18px 4px;margin:0}
.metric-inline{white-space:nowrap}.current-table td:first-child{font-weight:600}
@media(max-width:800px){.research-grid{grid-template-columns:1fr}.card-head{align-items:flex-start}.result-pair strong{font-size:21px}}
'''


def _read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def _label(arm, gate, cohort='NAVTEST'):
    return f'{arm}_u{gate:06}_{cohort}'


def _contrast(editorial, arm, gate, baseline=CONTROL):
    review = editorial.get(ROUTES[arm]['owner'], {}).get('matched_review', {})
    for row in review.get('contrasts', {}).values():
        if (row.get('status') == 'COMPLETE'
                and row.get('target') == _label(arm, gate)
                and row.get('baseline') == _label(baseline, gate)):
            return row
    return {}


def _next_evaluation(root, arm, run, trial, complete, h):
    gates = run.get('evaluation_gates') or trial.get('registration', {}).get('gates') or [1000, 5000]
    measured = {update(row) for row in complete}
    owner_run = trial.get('state', {}).get('runs', {}).get(arm, {})
    stages = owner_run.get('stages', run.get('stages', {}))
    for gate in gates:
        if gate in measured:
            continue
        label = _label(arm, gate)
        node = stages.get(str(gate), {}).get('NAVTEST', {})
        spec = node.get('spec') or _read(root/'evaluation_specs'/f'{label}.json')
        if spec:
            world = spec.get('world', len(spec.get('jobs', [])))
            done = 0
            for rank in range(min(int(world or 0), 128)):
                record = _read(Path(spec['output'])/f'rank{rank}.json')
                if record.get('status') == 'COMPLETE' and record.get('world') == world:
                    done += 1
            status = (f'{done}/{world}分片完成，等待完整汇总' if done == world and world
                      else f'{done}/{world}分片已完成')
            return dict(gate=gate, text=f'{gate//1000}k NAVTEST评测：{status}', dispatched=True)
        if h.finite(run.get('updates')) and run['updates'] >= gate:
            return dict(gate=gate, text=f'{gate//1000}k训练节点已到；等待完整评测派发／汇总', dispatched=False)
        return dict(gate=gate, text=f'下一评测节点：{gate//1000}k更新，完整NAVTEST＋TRAIN诊断', dispatched=False)
    return dict(gate=None, text='登记评测节点均已有完整NAVTEST；等待本轮最终分析', dispatched=False)


def build_current_sections(root, snapshot, h):
    root = Path(root)
    esc, fmt, number = h.esc, h.fmt, h.number
    link = lambda path, label: h.relative_link(root, str(path), label)
    signed = lambda x: f'{x:+.3f}' if h.finite(x) else '待分析'
    interval = lambda x: '['+', '.join(signed(v) for v in x)+']' if isinstance(x, (list, tuple)) and len(x) == 2 else '配对区间更新中'
    section = lambda key, title, deck, body: '<section id="'+key+'"><h2>'+title+'</h2><p class="deck">'+deck+'</p>'+body+'</section>'
    editorial = snapshot.get('editorial_analysis', {})
    runs = {row['id']: row for row in snapshot['runs']}
    complete = [row for row in snapshot['results'] if row.get('publishable_complete')]
    nav = [row for row in complete if row.get('cohort') == 'NAVTEST' and row.get('scenes') == 12146]
    controls = {update(row): row for row in nav if row.get('arm') == CONTROL}
    route_nav = {arm: sorted([row for row in nav if row.get('arm') == arm], key=update) for arm in ROUTES}
    latest = {arm: rows[-1] if rows else {} for arm, rows in route_nav.items()}
    active = sum(runs.get(arm, {}).get('worker_receipt_status') == 'RUNNING' for arm in ROUTES)
    best = max((row for row in nav if row.get('arm') in (CONTROL, *ROUTES)), key=lambda row: row['PDMS'], default={})
    next_eval = {arm: _next_evaluation(root, arm, runs.get(arm, {}), editorial.get(meta['owner'], {}), route_nav[arm], h) for arm, meta in ROUTES.items()}
    cards = [('当前训练', str(active)+' / '+str(len(ROUTES)), '实际运行worker／本轮试验'),
             ('完整NAVTEST最高', fmt(best.get('PDMS'), 3), number(update(best))+'更新 · '+ROUTES.get(best.get('arm'), {}).get('short', 'SafetyEP历史对照')),
             ('距离92目标', fmt(max(0, 92-best['PDMS']), 3) if best else '待评测', '仅以完整12,146场景成绩判断'),
             ('在途NAVTEST', str(sum(x['dispatched'] for x in next_eval.values())), '训练继续，评测并行推进')]
    body = '<div class="cards">'+''.join('<article><span>'+esc(a)+'</span><strong>'+esc(b)+'</strong><small>'+esc(c)+'</small></article>' for a, b, c in cards)+'</div>'
    body += '<p class="compact-notice">四组均从同一IL70重新开始：全量103,288场景、单训练seed42；动作LoRA与场景／预测Memory参与训练。原SafetyEP长训已结束，保留10k最佳模型作为同预算对照。训练更新、评测节点与完整分数分开显示。</p><div class="research-grid">'
    for arm, meta in ROUTES.items():
        run = runs.get(arm, {})
        count, cap = run.get('updates'), run.get('research_budget_updates')
        percent = min(100, 100*count/cap) if h.finite(count) and h.finite(cap) and cap else 0
        worker = run.get('worker_receipt_status')
        state = '训练中' if worker == 'RUNNING' and h.finite(count) and count > 0 else '启动中' if worker == 'RUNNING' else '本轮训练完成' if run.get('worker_returncode') == 0 and h.finite(count) and h.finite(cap) and count >= cap else '状态待核对'
        if run.get('worker_returncode') not in (None, 0):
            state = '执行状态待复核'
        if 'STOP' in str(run.get('status', '')):
            state = '已停止' if run.get('worker_returncode') == 0 else '停止请求中'
        latest_train = max((row for row in complete if row.get('arm') == arm and row.get('cohort') == 'TRAIN_DIAGNOSTIC' and row.get('scenes') == 1024), key=update, default={})
        measured = latest[arm]
        body += '<article class="research-card" style="--route:'+meta['color']+'"><div class="card-head"><h3>'+meta['name']+'</h3><span class="badge">'+state+'</span></div><code class="run-id">'+esc(arm)+'</code>'
        body += '<p><b>要解决：</b>'+meta['problem']+'</p><p><b>本组方法：</b>'+meta['method']+'</p>'
        body += '<div class="progress-title"><b>'+number(count)+' / '+number(cap)+' 更新</b><span>'+fmt(percent, 1)+'%</span></div><div class="progress-track"><div class="progress-fill" style="width:'+str(percent)+'%"></div></div>'
        body += '<small>'+esc(str(run.get('host') or 'host待记录').removeprefix('training-'))+' · 已见'+number(run.get('journal_ranks'))+' rank · 本轮登记上限'+number(cap)+'</small>'
        body += '<div class="result-pair"><div><small>最近完整NAVTEST</small>'+('<strong>'+fmt(measured['PDMS'], 3)+'</strong><small>'+number(update(measured))+'更新 · 12,146场景</small>' if measured else '<strong class="pending">评测尚未完整</strong>')+'</div><div><small>最近TRAIN诊断 · 训练内</small>'+('<strong>'+fmt(latest_train['PDMS'], 3)+'</strong><small>'+number(update(latest_train))+'更新 · 1,024场景</small>' if latest_train else '<strong class="pending">等待诊断结果</strong>')+'</div></div>'
        body += '<div class="eval-line">'+esc(next_eval[arm]['text'])+'</div>'
        if meta['paper']:
            title, url = meta['paper']
            body += '<p class="muted">研究参考：<a href="'+url+'">'+title+'</a>；本任务采用独立适配。</p>'
        else:
            body += '<p class="muted">自主设计：根据已有候选轨迹与真实梯度审计，按官方分数量纲补齐信用。</p>'
        body += '</article>'
    body += '</div>'
    first = section('current', '当前正在进行的四组实验', '先看研究目的、实际训练进度、已完成成绩与下一评测节点。快照：'+esc(snapshot['generated_utc']), body)

    rows = []
    for arm, meta in ROUTES.items():
        row = latest[arm]
        if not row:
            rows.append([meta['name'], '等待完整评测', '—', '—', '—', '—'])
            continue
        gate = update(row); control = controls.get(gate, {}); pair = _contrast(editorial, arm, gate)
        delta = row['PDMS']-control['PDMS'] if control else None
        zeros = pair.get('zeros', {}); high = pair.get('fixed_IL70_high_ge09', {})
        rows.append([meta['name']+'<small>'+number(gate)+'更新</small>', '<b>'+fmt(row['PDMS'], 4)+'</b>', fmt(control.get('PDMS'), 4), '<span class="'+('positive' if h.finite(delta) and delta > 0 else 'negative')+'">'+signed(delta)+'</span><small>'+interval(pair.get('CI95_PDMS'))+'</small>', number(zeros.get('repaired'))+' / '+number(zeros.get('new_zero')), signed(high.get('delta_PDMS'))+'<small>'+interval(high.get('CI95_PDMS'))+'</small>'])
    body = '<p><b>每行取该实验最近一个完整NAVTEST，比较相同步数的SafetyEP。</b>高分队列固定为原IL70分数≥0.9；下表的“修复／新增零分”相对于同预算Safety。各行评测步数不同，不能按这一列分数直接排算法优劣。</p>'
    body += h.table(['实验／评测节点', '完整PDMS', '同预算Safety', 'ΔPDMS［95%区间］', '零分修复／新增', '固定IL高分队列Δ'], rows)

    def score_plot(plt):
        fig, ax = plt.subplots(figsize=(7, 3.9), layout='constrained')
        baseline = sorted((row for row in controls.values() if update(row) <= 10000), key=update)
        if baseline:
            ax.plot([update(row)/1000 for row in baseline], [row['PDMS'] for row in baseline], '-o', color='#059669', label='SafetyEP control', lw=2)
        for arm, meta in ROUTES.items():
            points = route_nav[arm]
            if points:
                ax.plot([update(row)/1000 for row in points], [row['PDMS'] for row in points], '--o', color=meta['color'], label=meta['short'], ms=5)
        ax.set(xlabel='Optimizer updates (thousands)', ylabel='Complete NAVTEST PDMS', xlim=(0.6, 10.4))
        ax.grid(axis='y', alpha=.18); ax.legend(frameon=False, fontsize=8, ncol=2)
        return fig

    def delta_plot(plt):
        fig, ax = plt.subplots(figsize=(7, 3.9), layout='constrained')
        entries = [(arm, row) for arm, row in latest.items() if row and update(row) in controls]
        for pos, (arm, row) in enumerate(entries):
            gate = update(row); delta = row['PDMS']-controls[gate]['PDMS']; pair = _contrast(editorial, arm, gate); ci = pair.get('CI95_PDMS')
            ax.barh(pos, delta, color=ROUTES[arm]['color'], height=.55, alpha=.85)
            if isinstance(ci, list) and len(ci) == 2 and ci[0] <= delta <= ci[1]:
                ax.errorbar(delta, pos, xerr=[[delta-ci[0]], [ci[1]-delta]], color='#29485b', fmt='none', capsize=4, lw=1.1)
        ax.set_yticks(range(len(entries)), [ROUTES[arm]['short']+' · '+str(update(row)//1000)+'k' for arm, row in entries])
        ax.axvline(0, color='#64748b', lw=1); ax.grid(axis='x', alpha=.15); ax.invert_yaxis()
        ax.set_xlabel('PDMS difference vs same-budget SafetyEP')
        if not entries:
            ax.text(.5, .5, 'Awaiting complete NAVTEST', transform=ax.transAxes, ha='center')
        return fig

    body += '<div class="two">'+figure(root, h, 'current_research_score', '新实验学习曲线：聚焦0–10k', score_plot, '只连接已完成的完整NAVTEST；更早的Safety对照曲线用于同预算比较。未完成评测不补点。')+figure(root, h, 'current_research_delta', '最近完整节点：相对同预算Safety的收益', delta_plot, '每行标注评测步数；误差线为已有44物理log配对95%区间，缺区间不推测。不跨预算直接排序。')+'</div>'
    history_rows = [[ROUTES[row['arm']]['name'], number(update(row)), fmt(row['PDMS'], 4), link(row['evidence'], '完整SUMMARY')] for row in sorted((row for row in nav if row.get('arm') in ROUTES), key=lambda row: (update(row), row['arm']))]
    train_rows = [[ROUTES[row['arm']]['name'], number(update(row)), fmt(row['PDMS'], 4), link(row['evidence'], 'TRAIN SUMMARY')] for row in sorted((row for row in complete if row.get('arm') in ROUTES and row.get('cohort') == 'TRAIN_DIAGNOSTIC' and row.get('scenes') == 1024), key=lambda row: (update(row), row['arm']))]
    body += '<details><summary>展开各阶段完整NAVTEST结果与原始评测</summary>'+h.table(['实验', '更新', '完整PDMS', '证据'], history_rows)+'</details>'
    body += '<details><summary>展开TRAIN1024诊断成绩：与部署分数分别阅读</summary><p>TRAIN1024属于训练人口，可用于学习过程诊断；即使超过92，也不计为NAVTEST目标达成。</p>'+h.table(['实验', '更新', 'TRAIN PDMS', '证据'], train_rows)+'</details>'
    second = section('current-results', '最新评测结果：完整分数与同预算对照', '完整NAVTEST优先；训练内诊断、在途评测与已完成分数有明确边界。', body)

    body = ''
    for arm, meta in ROUTES.items():
        row = latest[arm]
        if not row:
            verdict = '当前还没有完整NAVTEST，暂不能判断部署收益。先完成首个全量评测，再比较安全、效率与高分保持。'
        else:
            gate = update(row); pair = _contrast(editorial, arm, gate); ci = pair.get('CI95_PDMS'); control = controls.get(gate, {})
            if isinstance(ci, list) and len(ci) == 2:
                conclusion = '高于' if ci[0] > 0 else '低于' if ci[1] < 0 else '尚未与'
                verdict = str(gate//1000)+'k完整结果'+conclusion+'同预算Safety'+('拉开明确差距' if conclusion == '尚未与' else '')+'；ΔPDMS '+signed(pair.get('delta_PDMS'))+'，95%区间 '+interval(ci)+'。'
            elif control:
                verdict = str(gate//1000)+'k完整分数为'+fmt(row['PDMS'], 4)+'，相对同预算Safety观察点差'+signed(row['PDMS']-control['PDMS'])+'；配对区间分析尚在更新。'
            else:
                verdict = str(gate//1000)+'k完整分数为'+fmt(row['PDMS'], 4)+'；等待同预算对照。'
            zeros = pair.get('zeros', {}); high = pair.get('fixed_IL70_high_ge09', {})
            if zeros:
                verdict += '相对该对照修复'+number(zeros.get('repaired'))+'个零分，同时新增'+number(zeros.get('new_zero'))+'个。'
            if h.finite(high.get('delta_PDMS')):
                verdict += '固定IL高分队列变化'+signed(high['delta_PDMS'])+'点，需与总体一起判断。'
            if arm == 'PPO_FPO_FULLTIME':
                original = _contrast(editorial, arm, gate, 'PPO_SAFETY_FPO')
                if original:
                    verdict += '直接对原FPO的时间上界对照为'+signed(original.get('delta_PDMS'))+'点，区间'+interval(original.get('CI95_PDMS'))+'；扩大覆盖目前未表现为收益。' if original.get('delta_PDMS', 0) < 0 else '对原FPO的同预算变化'+signed(original.get('delta_PDMS'))+'点，继续核对配对结果。'
        body += '<article class="analysis-route" style="--route:'+meta['color']+'"><h3>'+meta['name']+'</h3><p><b>现在的判断：</b>'+verdict+'</p><p class="analysis-next"><b>下一步看什么：</b>'+esc(next_eval[arm]['text'])+'；检查总分、NC／DAC／TTC／EP、零分修复与新增，以及固定IL高分队列是否保住。</p></article>'
    body += '<div class="compact-notice"><b>零分问题的下一层分析：</b>已有证据区分“尚未直接曝光”“全失败同成本无相对信号”“已有成功候选但ODE仍失败”“训练中反复修复和新生”四类。当前质量信用组修复的是部分信号遗漏；逐时刻越界占比的打平局方案目前仅有CPU反事实，尚未部署。详细场景和梯度证据在后方折叠资料中。</div>'
    third = section('current-analysis', '当前判断与接下来的评测重点', '明确哪些已有部署证据，哪些仍只是机制假设；新结果出现后随报告刷新。', body)
    return [first, second, third]
