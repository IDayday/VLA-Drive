"""Report-owned synthesis of existing complete summaries and paired reviews.

No scene records, training selection, bootstrap, dispatch, or process mutations.
"""
from datetime import datetime, timezone
import hashlib
from pathlib import Path


NAMES = {
    'SAFETY_G8_CONTROL': 'Control', 'STANDARD_INVERSION_G8': '纠偏信用',
    'STANDARD_COMBINED_G8': '三信用组合', 'STANDARD_ENDPOINT_LR02_G8': 'Endpoint低LR',
    'REVIVAL_G16': 'G16', 'REVIVAL_DENSE_G8': 'DenseDAC',
    'REVIVAL_SATURATED_G8': '饱和信用', 'REVIVAL_ACTIVE_DATA_G8': 'Active课程',
    'IMPROVEMENT_EARLY_DECAY_G8': 'H·更早衰减',
    'IMPROVEMENT_MULTISCENE32_G8': 'H·不同场景Batch32',
    'IMPROVEMENT_VERIFIED_RETENTION_G8': 'H·质量门控历史RL保持',
}
CONTROL = 'SAFETY_G8_CONTROL'


def build_review(studies):
    points = {}; pairs = []; inputs = {}; runs = {}
    def evidence(path):
        if path and path not in inputs:
            p = Path(path)
            if p.is_file():
                inputs[path] = dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                                    size_bytes=p.stat().st_size)
        return path
    for study in studies:
        root = study['root']
        for run in study['runs']:
            runs[(root, run['id'])] = {k: run.get(k) for k in
                ['id', 'updates', 'status', 'host', 'journal_ranks', 'new_candidates',
                 'raw_scene_exposures', 'worker_returncode', 'worker_receipt_status',
                 'progress_freshness', 'job', 'improvement_role']}
        for result in study['results']:
            if not result.get('publishable_complete') or result.get('cohort') != 'NAVTEST':
                continue
            key = (root, result['arm'], result['updates'])
            points[key] = dict(arm=result['arm'], updates=result['updates'], PDMS=result['PDMS'],
                study_root=root, evidence=evidence(result.get('evidence')), scenes=12146)
        for pair in study.get('paired_reviews', []):
            evidence(pair['evidence'])
            if (root, pair['arm'], pair['updates']) in points:
                pairs.append(dict(pair, study_root=root))
    def historic(root, arm, update):
        return next((p for p in pairs if p['study_root'] == root and p['arm'] == arm
                     and p['updates'] == update and p['kind'] == 'historical'
                     and p['reference'] == 'IL70'), None)
    control_root = next((s['root'] for s in studies if Path(s['root']).name == 'standard_training_20261101'), '')
    for (root, arm, u), point in points.items():
        il = historic(root, arm, u)
        if il:
            r = il['result']
            point['zeros'] = r.get('baseline_zero_denominator', 0) - r.get('repaired_zero', 0) + r.get('new_zero', 0)
            point['fixed_IL70_high_mean_delta_PDMS'] = r.get('fixed_control_high', {}).get('mean_delta_PDMS')
            point['fixed_IL70_high_count'] = r.get('fixed_control_high', {}).get('count')
    comparisons = []
    for pair in pairs:
        if pair['kind'] != 'between_arms' or pair['reference'] != CONTROL or pair.get('reference_root') != control_root:
            continue
        target = points[(pair['study_root'], pair['arm'], pair['updates'])]
        base = points.get((control_root, CONTROL, pair.get('reference_updates')))
        if not base:
            continue
        r = pair['result']; factor = 2 if pair['arm'] == 'REVIVAL_G16' else 4 if pair['arm'] == 'IMPROVEMENT_MULTISCENE32_G8' else 1
        a = target.get('fixed_IL70_high_mean_delta_PDMS'); b = base.get('fixed_IL70_high_mean_delta_PDMS')
        fixed_delta = a-b if isinstance(a, (int, float)) and isinstance(b, (int, float)) else pair.get('fixed_IL70_high_advantage_PDMS')
        comparisons.append(dict(arm=pair['arm'], updates=pair['updates'], reference_updates=base['updates'],
            study_root=pair['study_root'], target_PDMS=target['PDMS'], baseline_PDMS=base['PDMS'],
            delta_PDMS=r.get('PDMS_delta'), CI95_PDMS=r.get('PDMS_CI95'),
            repaired_zero=r.get('repaired_zero'), new_zero=r.get('new_zero'),
            baseline_zeros=r.get('baseline_zero_denominator'), target_zeros=target.get('zeros'),
            fixed_IL70_high_mean_delta_PDMS=fixed_delta,
            fixed_IL70_high_population_mass_delta_PDMS=fixed_delta*8724/12146 if fixed_delta is not None else None,
            fixed_IL70_high_count=8724, fixed_IL70_high_CI95_PDMS=None,
            fixed_Control_high_mean_delta_PDMS=r.get('fixed_control_high', {}).get('mean_delta_PDMS'),
            nominal_source_candidate_matched=base['updates'] == factor*pair['updates'],
            source_update_clock_multiplier=factor, budget_contract=pair.get('budget_contract'),
            actual_collection_budget=pair.get('actual_collection_budget'), evidence=pair['evidence']))
    arms = []
    for root, arm in sorted({(root, arm) for root, arm, u in points} | set(runs)):
        curve = sorted([p for (rt, a, u), p in points.items() if rt == root and a == arm], key=lambda p: p['updates'])
        matched = sorted([p for p in comparisons if p['study_root'] == root and p['arm'] == arm
                          and p['nominal_source_candidate_matched']], key=lambda p:p['updates'])
        arms.append(dict(arm=arm, name=NAMES.get(arm, arm), study_root=root, curve=curve,
            latest_complete=curve[-1] if curve else None, observed_peak=max(curve, key=lambda p:p['PDMS']) if curve else None,
            latest_candidate_matched=matched[-1] if matched else None,
            candidate_matched_points=matched, current_native=runs.get((root, arm), {})))
    return dict(schema='report_existing_complete_long_run_comparison_v1',
        status='COMPLETE_SNAPSHOT_OF_AVAILABLE_FULL_RESULTS', created_utc=datetime.now(timezone.utc).isoformat(),
        complete_points=len(points), complete_comparisons=len(comparisons), arms=arms,
        observed_peak=max(points.values(), key=lambda p:p['PDMS']) if points else None,
        comparisons=comparisons, input_files=inputs,
        scope='F/G/H only; existing complete12146/invalid0 summaries and sealed paired artifacts; no new score/bootstrap.',
        limitations=['All reported CIs reused: one training seed,44 physical logs,2000 cluster-bootstrap draws,seed42.',
            'Repeated checkpoint selection and multiple comparisons are not corrected; observed peaks are descriptive.',
            'Fixed IL70 high cohort8724 is invariant; its mean deltas have no CI here and are not significance claims.',
            'Candidate budgets do not equate scenes, Adam steps, scheduler phase, or model compute for G16/Batch32.',
            'Retention extra official ODE queries are separate; source-PG query matching is not total-query matching.',
            'NAVTEST labels are diagnostic only and never become a training bank.'])


def render_review(root, review, review_path, h):
    ci = lambda v: '['+', '.join(h.fmt(x, 4) for x in v)+']' if isinstance(v, list) and len(v) == 2 else '未有配对区间'
    link = lambda p, text: h.relative_link(root, p, text)
    peak = review.get('observed_peak') or {}
    body = '<div class="notice"><b>已出现多个阶段的配对正点，但尚未形成持续的新方法领先。</b>当前F/G/H完整观测峰仍为 '+h.esc(NAMES.get(peak.get('arm'), peak.get('arm')))+' '+h.number(peak.get('updates'))+'更新 '+h.fmt(peak.get('PDMS'), 5)+'；'+('尚未达到92。' if peak.get('PDMS', 0)<92 else '已达92，保持性另检。')+'本次汇总 '+h.number(review['complete_points'])+' 个完整NAVTEST点／'+h.number(review['complete_comparisons'])+' 组既有Control配对；不把部分评测补成成绩。</div>'
    rows = []
    ordered = sorted(review['arms'], key=lambda r:(not r['arm'].startswith('IMPROVEMENT_'), r['arm'] != CONTROL, r['arm']))
    for arm in ordered:
        latest = arm['latest_complete']; p = arm['latest_candidate_matched']
        if latest is None:
            rows.append([h.esc(arm['name']), '等待完整NAVTEST', '—', '—', '—', '—'])
            continue
        label = h.number(latest['updates'])+'／'+h.fmt(latest['PDMS'], 5)+'<small>零分 '+h.number(latest.get('zeros'))+'；'+link(latest['evidence'], '全量SUMMARY')+'</small>'
        if p:
            budget=h.number(p['updates'])+' vs Control '+h.number(p['reference_updates'])
            stale='；较新点尚无匹配Control' if latest['updates'] != p['updates'] else ''
            rows.append([h.esc(arm['name']),label, budget+'<small>'+h.fmt(p['delta_PDMS'],5)+'／'+ci(p['CI95_PDMS'])+stale+'</small>',
                h.number(p['repaired_zero'])+'／'+h.number(p['new_zero']),h.fmt(p['fixed_IL70_high_mean_delta_PDMS'],4),link(p['evidence'],'完整配对')])
        else:
            rows.append([h.esc(arm['name']),label,'只作对照，不与自身配对','—',h.fmt(latest.get('fixed_IL70_high_mean_delta_PDMS'),4)+'<small>此行为对IL70自身基准</small>','—'])
    body+='<h3>最新完整结果：同候选预算、零修复与固定高分保持一起看</h3>'+h.table(['路线','最新完整Adam／PDMS','最近同候选量比较：更新／Δ及95%CI','零修复／新增','固定IL70≥.9均值Δ','证据'], rows)
    body+='<p class="muted">方法行的高群体Δ是同一固定IL70高分8724场景相对当点Control的均值差，未给CI，不据此宣称显著。Control行则是相对IL70。G16用2倍、Batch32用4倍Control更新对齐source-PG查询；场景曝光、Adam和scheduler时钟仍不同。当前H保持臂未完整时留空；其2k后额外官方ODE评分仍单独分账。</p>'
    g16 = next((a for a in review['arms'] if a['arm']=='REVIVAL_G16'), {})
    gp = g16.get('candidate_matched_points', [])
    if gp:
        body+='<p><b>G16不能再只按早期负点判断：</b>'+ '；'.join(h.number(p['updates'])+' vs Control '+h.number(p['reference_updates'])+'：Δ'+h.fmt(p['delta_PDMS'],5)+'，CI '+ci(p['CI95_PDMS']) for p in gp) +'。后期出现同候选量正向证据，值得保留一个后续公平复核点；这不是纯探索量单因素，也未超过当前绝对峰值。</p>'
    decisions = [
        ['P0：优先完成H最小检验','更早衰减／Batch32／质量保持','H仅有首个完整点，Early2k与Batch500对F2k区间跨0；不能据早期结果停止。优先收5k及对应物理预算点，保持臂必须跨2k激活后再看作用。'],
        ['P1：有条件保留','G16、饱和信用、纠偏信用','G16已有后期同候选量正点；Sat／Inv有阶段正向但20–22k未持续。保留预定后续复核，不默认一直加到50k，也不把单个负点判永久无效。'],
        ['P2：不优先额外扩预算','DenseDAC、三信用组合、原Active课程','长训已有多点，最新完整对照未形成持续优势；可把超出已批准复核点的资源让给新问题。Active的22k反弹需与20k负点同时看，禁止把低分修复当总分增益。'],
        ['P3：建议优先让出当前配方资源','旧Endpoint低LR','已有多个完整长期点显著低于Control，12k仍未恢复；有限过程的候选支持改善未转成部署收益。是当前配方的配置分配建议，不否定endpoint算法族；是否停训由root按实际收据处理。'],
    ]
    body+='<details open><summary>资源优先级建议：依据完整长曲线，不执行自动淘汰</summary>'+h.table(['优先级','路线','建议与证据边界'], decisions)+'</details>'
    body+='<p>最新证据仅支持研究分配建议，未发停止或迁移信号。历史负面CI、高分退化和早期正点全部保留；后续新算法按实际登记／启动证据接入。NAVTEST名单、未来标签和互补oracle均不进入TRAINbank。</p>'
    body+='<p class="muted">'+link(str(review_path),'完整曲线／全部配对及输入SHA清单')+' · 生成 '+h.esc(review['created_utc'])+'。单seed／44日志配对区间，未校正多重比较；观测峰不是无偏最优值。完整曲线图见下方。</p>'
    return '<section id="rl-latest-evidence"><h2>最新完整证据与下一轮资源优先级</h2>'+body+'</section>'
