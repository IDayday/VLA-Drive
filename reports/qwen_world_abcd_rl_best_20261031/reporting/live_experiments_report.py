"""Compact current workbench, derived only from the existing report snapshot."""
from pathlib import Path

from latest_complete_report import CONTROL, NAMES

INTEGRATED = 'INTEGRATED_SAFETY_EP_B64_G16_RETENTION_WKC'
ACTIVE = ('training', 'starting', 'waiting_resource', 'exited_review', 'unconfirmed')
LABELS = dict(training='训练中', starting='启动中', waiting_resource='等待训练资源',
              exited_review='执行退出，待处理', unconfirmed='运行状态待确认',
              resource_reserved='资源储备', paused_saved='已保存暂停', ended='本阶段训练结束')
METHODS = {
    INTEGRATED: 'SafetyEP＋64场景／G16＋历史保持＋WKC',
    'REVIVAL_G16': 'SafetyEP；每场景16条候选',
    'IMPROVEMENT_VERIFIED_RETENTION_G8': 'SafetyEP＋质量门控的历史行为保持',
    'IMPROVEMENT_EARLY_DECAY_G8': 'SafetyEP；cosine更早衰减',
    'IMPROVEMENT_MULTISCENE32_G8': 'SafetyEP；每批32个不同场景',
    'IMPROVEMENT_MIXED_NOISE_G8': 'SafetyEP；固定G8混合探索',
    'IMPROVEMENT_FP32_OUTPUT_G8': 'SafetyEP；FP32动作输出投影',
}
CSS = '''
.live-heading{display:flex;justify-content:space-between;gap:15px;align-items:center}
.live-heading .badge{background:#e9f6ef;color:#16704e}
.live-table td:first-child{min-width:180px;font-weight:600}.live-table small{max-width:320px}
.live-table td{padding:11px 12px}.live-score{font-size:21px;line-height:1.3;color:#123e51}
.live-meter{height:6px;max-width:190px;background:#e8eef4;border-radius:8px;margin:7px 0;overflow:hidden}
.live-meter i{display:block;background:#16957a;height:100%}.live-positive{color:#137550}.live-negative{color:#ad4939}
.live-insight{padding:12px 16px;border-radius:8px;background:#eff5fa;font-size:13px;margin:14px 0}
#rl-live-workbench{border-top:4px solid #15957d}#rl-live-results{border-top:3px solid #3276b5}
header{padding-top:22px;padding-bottom:22px}header h1{font-size:27px;margin:5px 0}
header .overview{margin:12px 0;gap:14px}header .overview article{padding:13px 18px}
header .overview-score{font-size:29px;margin:3px 0}header>p{font-size:13px;margin:7px 0}
@media(max-width:750px){.live-heading{align-items:flex-start}.live-score{font-size:19px}}
'''


def render_workbench(root, snapshot, review, h):
    census = snapshot.get('front_page_summary', {})
    studies = snapshot.get('editorial_analysis', {}).get('standard_studies', [])
    runs = {(s['root'], r['id']): r for s in studies for r in s['runs']}
    arms = {(r['study_root'], r['arm']): r for r in review['arms']}
    live = census.get('runs', [])
    active = [r for r in live if r['category'] in ACTIVE]
    active.sort(key=lambda r: (r['category'] != 'training', r['arm'] != INTEGRATED, r['arm']))
    peak = census.get('best', {})
    link = lambda p, t: h.relative_link(root, str(p) if p else None, t)
    name = lambda arm: NAMES.get(arm, arm)
    signed = lambda v: f'{v:+.4f}' if h.finite(v) else '待对照'
    pending = {}
    for study in studies:
        for r in study['results']:
            if r.get('cohort') == 'NAVTEST' and not r.get('publishable_complete'):
                pending.setdefault((study['root'], r['arm']), []).append(r)
    for values in pending.values():
        values.sort(key=lambda p: p.get('updates', 0))

    def pair_for(point):
        candidates = [p for p in review['comparisons'] if p['arm'] == point['arm']
                      and p['study_root'] == point['study_root'] and p['updates'] == point['updates']]
        return min(candidates, key=lambda p: (p['reference_updates'] != point['updates'],
                   not p.get('nominal_source_candidate_matched'), p['reference_updates']), default=None)

    def comparison(point):
        if not point:
            return '尚无完整结果'
        if point['arm'] == CONTROL:
            return '共享SafetyEP对照'
        p = pair_for(point)
        if p:
            axis = ('同Adam更新' if p['updates'] == p['reference_updates'] else
                    '名义场景量接近' if p.get('comparison_axis') == 'scene' else '主PG候选量对齐')
            cls = 'live-positive' if p['delta_PDMS'] > 0 else 'live-negative'
            interval = p.get('CI95_PDMS')
            ci = ('；95%CI ['+', '.join(f'{x:+.4f}' for x in interval)+']') if isinstance(interval, list) else ''
            return ('<b class="'+cls+'">'+signed(p['delta_PDMS'])+'</b> 对Control '
                    +h.number(p['reference_updates'])+'<small>基线 '+h.fmt(p['baseline_PDMS'], 4)
                    +'；'+axis+ci+'</small>')
        distance = point['PDMS']-peak['PDMS'] if h.finite(peak.get('PDMS')) else None
        return '该节点配对对照未齐<small>距历史最高 '+signed(distance)+'，仅算术分差</small>'

    def score(point):
        if not point:
            return '等待完整NAVTEST'
        zeros = '；零分 '+h.number(point['zeros']) if h.finite(point.get('zeros')) else ''
        return ('<b class="live-score">'+h.fmt(point['PDMS'], 4)+'</b><small>@ '
                +h.number(point['updates'])+'更新'+zeros+' · '+link(point['evidence'], '完整结果')+'</small>')

    def verdict(point):
        if not point:
            return '尚无完整部署成绩'
        if point['arm'] == INTEGRATED and point['updates'] < 2000:
            return '早期结果；未到2k<small>250点先于历史保持；完成10k后判断长训</small>'
        if point['arm'] == CONTROL:
            return '历史最高分参考' if point['PDMS'] == peak.get('PDMS') else '共享对照的后续节点'
        p = pair_for(point)
        if not p:
            return '完整分数已到；该节点对照待齐'
        ci = p.get('CI95_PDMS') or []
        if len(ci) == 2:
            if ci[0] > 0:
                return '该节点相对所列对照有提升<small>阶段收益，尚需检查持续性及是否超过最高分</small>'
            if ci[1] < 0:
                return '该节点落后于所列对照<small>保留完整结果，结合后续节点判断</small>'
        return '当前配对差异尚不明确'

    def next_step(key, run, common_updates):
        records = pending.get(key, [])
        lines = []
        if records:
            p = records[0]
            lines.append(h.number(p['updates'])+'评测中：'+h.number(p.get('scenes'))+'/12,146')
            if len(records) > 1:
                lines.append('<small>共 '+h.number(len(records))+' 个评测节点未形成完整成绩</small>')
        measured = {p['updates'] for p in arms.get(key, {}).get('curve', [])}
        gates = run.get('evaluation_gates', run.get('gates', []))
        remaining = [g for g in gates if g not in measured and
                     (not h.finite(common_updates) or g > common_updates)]
        if remaining:
            lines.append('<small>下一完整评测节点 '+h.number(min(remaining))+'</small>')
        return ''.join(lines) or '本阶段评测已齐；后续按预算复核'

    current_rows = []
    for row in active:
        key = (row['study_root'], row['arm']); run = runs.get(key, {})
        point = arms.get(key, {}).get('latest_complete')
        u = row.get('common_updates'); target = run.get('target_updates')
        fraction = min(100, max(0, 100*u/target)) if h.finite(u) and h.finite(target) and target else 0
        progress = ('<b>'+LABELS[row['category']]+'</b> · '+h.number(u)+' / '+h.number(target)
                    +'<div class="live-meter"><i style="width:'+f'{fraction:.2f}'+'%"></i></div>'
                    +'<small>实际Adam更新；'+h.esc(row.get('host') or '主机待分配')+'</small>')
        current_rows.append([h.esc(name(row['arm']))+'<small>'+h.esc(METHODS.get(row['arm'], run.get('algorithm', '方法见详细章节')))+'</small>',
                             progress, score(point), comparison(point), verdict(point), next_step(key, run, u)])
    counts = census.get('counts', {})
    body = ('<div class="live-heading"><h2>正在进行的实验</h2><span class="badge">实际训练 '
            +h.number(counts.get('training'))+' · 启动 '+h.number(counts.get('starting'))
            +' · 等资源 '+h.number(counts.get('waiting_resource'))+'</span></div>'
            +'<p class="deck">更新 '+h.esc(census.get('observed_utc', snapshot['generated_utc']))
            +'。训练进度与已完成评测分开列出；最新训练步数不代表该步已经评测。</p>')
    if current_rows:
        body += h.table(['实验／方法', '状态与训练进度', '最新完整NAVTEST', '对明确基线的变化', '当前结论', '当前评测／下一节点'], current_rows).replace('class="table-wrap"', 'class="table-wrap live-table"')
    else:
        body += '<p>本次快照没有确认正在训练或等待启动的任务；已结束与待评测任务见下表。</p>'
    body += ('<p class="muted">同一完整NAVTEST为12,146场景、invalid0。不同批场景数／G值的更新和计算量不同；每个涨跌都注明其比较基线。</p>')
    sections = ['<section id="rl-live-workbench">'+body+'</section>']

    latest = [a['latest_complete'] for a in review['arms'] if a.get('latest_complete') and a['latest_complete']['updates'] > 0]
    latest.sort(key=lambda p: Path(p['evidence']).stat().st_mtime if Path(p['evidence']).is_file() else 0, reverse=True)
    result_rows = [[h.esc(name(p['arm'])), h.number(p['updates']), score(p), comparison(p), verdict(p)] for p in latest]
    body = '<h2>最新结果汇总</h2><p class="deck">每条路线取最近一个完整评测节点，按结果文件完成时间排序；本表不按分数混排不同预算。</p>'
    columns = ['路线', '评测更新', '完整PDMS／零分', '对照与涨跌', '效果判断']
    body += h.table(columns, result_rows[:6])
    if len(result_rows) > 6:
        body += '<details><summary>其余 '+h.number(len(result_rows)-6)+' 条路线的最近完整结果</summary>'+h.table(columns, result_rows[6:])+'</details>'
    sections.append('<section id="rl-live-results">'+body+'</section>')

    queue_rows = []
    categories = {(r['study_root'], r['arm']): r for r in live}
    for key, values in sorted(pending.items(), key=lambda item: (categories.get(item[0], {}).get('category') != 'training', item[0][1] != INTEGRATED, item[0][1])):
        first = values[0]; last = values[-1]; cat = categories.get(key, {}).get('category', 'unconfirmed')
        queue_rows.append([h.esc(name(key[1])), h.esc(LABELS.get(cat, cat)), h.number(len(values)),
            h.number(first['updates'])+(' → '+h.number(last['updates']) if first != last else ''),
            h.number(first.get('scenes'))+' / 12,146<small>最早未齐节点 '+h.number(first['updates'])+'；'+link(first.get('evaluation') or first.get('evidence'), '分片记录')+'</small>'])
    body = '<h2>评测队列与已结束任务</h2><p class="deck">训练已经结束的路线仍可能在补齐评测；这些节点未完整时只显示进度。</p>'
    body += h.table(['路线', '训练状态', '未齐评测数', '待评测节点范围', '最早节点完成场景'], queue_rows) if queue_rows else '<p>本次快照无已登记且尚未齐的NAVTEST结果；尚未到训练评测节点的任务见首表。</p>'
    inactive = [r for r in live if r['category'] not in ACTIVE]
    if inactive:
        rows = [[h.esc(name(r['arm'])), LABELS.get(r['category'], r['category']),
                 '评测尚未齐' if (r['study_root'], r['arm']) in pending else '既有结果保留'] for r in inactive]
        body += '<details><summary>已结束／保存暂停／资源储备任务 '+h.number(len(inactive))+' 条</summary>'+h.table(['实验', '当前状态', '结果进度'], rows)+'</details>'
    sections.append('<section id="rl-live-evaluations">'+body+'</section>')
    return ''.join(sections)
