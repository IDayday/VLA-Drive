"""Read-only paper catalogue, pinned local recipes and existing paired evidence."""
from datetime import datetime, timezone
import hashlib
from pathlib import Path

import yaml

from latest_complete_report import CONTROL, NAMES

SECTION_ID = 'rl-recipe-configurations'
DIRECTORY = 'RL_RECIPE_CONFIGURATION_REVIEW_20261102'


def _schedule(cfg):
    research = cfg.get('research', {})
    integrated = research.get('integrated64', {})
    if integrated:
        value = integrated.get('schedule', {})
    else:
        value = research.get('consolidation_training') or research.get('standardized_training') or research.get('batch32') or {}
    # These are explicit YAML fields, never a library scheduler default.
    return dict(kind=cfg.get('lr_scheduler_type'), warmup=value.get('warmup_updates', value.get('optimizer_warmup_updates', cfg.get('lr_warmup_steps'))),
                keep_until=value.get('keep_until_updates'), cosine_end=value.get('cosine_decay_end_updates'),
                horizon=value.get('horizon_updates', value.get('optimizer_horizon_updates', cfg.get('max_steps'))),
                floor=value.get('min_lr_ratio'))


def load_local_configs(root, snapshot, h):
    records = []
    for study in snapshot.get('editorial_analysis', {}).get('standard_studies', []):
        for run in h.object_rows(study.get('registry', {}).get('runs', []), 'runs'):
            path = Path(run['config']) if run.get('config') else None
            row = dict(arm=run.get('id'), study_root=study['root'], source=run.get('source'),
                       config=str(path) if path else None, expected_sha256=run.get('config_sha256'))
            try:
                if not path or not path.is_file():
                    row['status'] = 'NR_CONFIG_NOT_AVAILABLE'
                    records.append(row)
                    continue
                raw = path.read_bytes()
                digest = hashlib.sha256(raw).hexdigest()
                row['actual_sha256'] = digest
                snapshot['inputs'].append(dict(path=str(path), sha256=digest, bytes=len(raw), mtime=path.stat().st_mtime))
                if not row['expected_sha256'] or digest != row['expected_sha256']:
                    row['status'] = 'NR_UNVERIFIED_CONFIG_HASH'
                    snapshot['warnings'].append('本地配方哈希未匹配登记，配置表保留NR：'+str(run.get('id')))
                    records.append(row)
                    continue
                cfg = yaml.safe_load(raw)
                research = cfg.get('research', {})
                sample = cfg.get('grpo', {}).get('sample', {})
                train = cfg.get('grpo', {}).get('train', {})
                batch = research.get('integrated64') or research.get('batch32') or {}
                scenes_per_rank = batch.get('scenes_per_rank', cfg.get('batch_size'))
                world = cfg.get('world_size')
                group = sample.get('group_size')
                scenes = world*scenes_per_rank if isinstance(world, int) and isinstance(scenes_per_rank, int) else None
                candidates = scenes*group if isinstance(scenes, int) and isinstance(group, int) else None
                row.update(status='PINNED_YAML_HASH_MATCH', role=research.get('standard_role'),
                           world_size=world, scenes_per_rank=scenes_per_rank, scenes_per_window=scenes,
                           group_size=group, PG_candidates_per_window=candidates,
                           action_peak_lr=cfg.get('learning_rate'), memory_peak_lr=cfg.get('memory_learning_rate'),
                           schedule=_schedule(cfg), BC_enabled=train.get('use_bc_loss'),
                           BC_configured_coefficient=train.get('bc_coeff'), anchor_type=train.get('kl_type'),
                           anchor_beta=train.get('kl_beta'), PPO_clip=train.get('ppo_clip_range'),
                           inner_updates=train.get('num_inner_epochs'), accumulation=cfg.get('gradient_accumulation_steps'),
                           advantage_schema=research.get('structured_advantage'),
                           global_std=train.get('adv_global_std'), advantage_cap=train.get('adv_clip_max'),
                           probability='conditional Gaussian joint24' if research.get('endpoint_policy') else 'source W3 mean-logp proxy',
                           history=research.get('verified_retention') or research.get('integrated64', {}).get('retention'))
            except (OSError, ValueError, TypeError, yaml.YAMLError):
                row['status'] = 'NR_CONFIG_READ_FAILED'
                snapshot['warnings'].append('本地配方读取失败，保留NR：'+str(run.get('id')))
            records.append(row)
    audit = dict(schema='report_pinned_local_RL_recipe_configuration_audit_v1',
                 created_utc=datetime.now(timezone.utc).isoformat(), rows=records,
                 scope='Only registered scientific YAML with exact RUNS config_sha256; no tensor, owner or runtime mutations.',
                 unknown_policy='Missing or unverified fields remain NR. Configured BC coefficient is inactive when use_bc_loss=False.')
    path = root/'analysis'/DIRECTORY/'LOCAL_CONFIGURATION_COMPARISON.json'
    h.atomic_json(path, audit)
    return audit, path


def _paper_value(value, h):
    if not value:
        return 'NR'
    text = str(value)
    if any(word in text for word in ['未披露', '未核定', '待补', '未说明']):
        return '<b>NR</b><small>'+h.esc(text)+'</small>'
    return h.esc(text)


def _effective_rows(snapshot, review, h, link):
    rows = []
    original = snapshot.get('editorial_analysis', {}).get('new_pilot_matched_review', {})
    key = 'PPO_SAFETY_EP_u010000_NAVTEST_minus_PPO_FULL_u010000_NAVTEST'
    paired = original.get('contrasts', {}).get(key, {})
    signed = lambda v: f'{v:+.6f}' if h.finite(v) else 'NR'
    ci = lambda v: '['+', '.join(signed(x) for x in v)+']' if isinstance(v, list) and len(v) == 2 else 'NR'
    if paired.get('status') == 'COMPLETE':
        zeros = paired.get('paired', {})
        zero_start = zeros.get('baseline_zero_denominator')
        zero_end = zero_start-zeros.get('repaired_zero', 0)+zeros.get('new_zero', 0) if isinstance(zero_start, int) else None
        rows.append(['原SafetyEP · 10k', h.fmt(paired.get('baseline_PDMS'), 6)+' → '+h.fmt(paired.get('target_PDMS'), 6),
                     signed(paired.get('delta_PDMS'))+'<small>'+ci(paired.get('CI95_PDMS'))+'</small>',
                     '同IL70／同10k；信用与归一化的整体收益，不能全归某一个gate。零分 '+h.number(zero_start)+' → '+h.number(zero_end)+'（'+h.number(zeros.get('repaired_zero'))+'修复／'+h.number(zeros.get('new_zero'))+'新增），NC下降；总分升不等于全安全改善',
                     link(root_path(snapshot)/'analysis'/'NEW_PILOT_MATCHED_REVIEW.json', '完整旧配对')])
    selections = [('IMPROVEMENT_VERIFIED_RETENTION_G8', 5000, 5000),
                  ('IMPROVEMENT_EARLY_DECAY_G8', 12000, 12000),
                  ('IMPROVEMENT_EARLY_DECAY_G8', 14000, 14000),
                  ('REVIVAL_G16', 10000, 20000),
                  ('IMPROVEMENT_FP32_OUTPUT_G8', 5000, 5000),
                  ('IMPROVEMENT_MULTISCENE32_G8', 2500, 10000),
                  ('IMPROVEMENT_MIXED_NOISE_G8', 5000, 5000)]
    notes = {
        'IMPROVEMENT_VERIFIED_RETENTION_G8': '同场景／主PG；额外官方B1查询另计。5k阶段有效，尚未超过最高点',
        'IMPROVEMENT_EARLY_DECAY_G8': '同Adam／主PG；相对该点Control改善，但未消除自身10k后回落',
        'REVIVAL_G16': '仅主PG候选同量；Adam10k对20k，场景／调度不同。不是G16同算力胜出',
        'IMPROVEMENT_FP32_OUTPUT_G8': '区间跨零，尚无明确同阶段增益；native0初始函数改变另有评测',
        'IMPROVEMENT_MULTISCENE32_G8': '场景与主PG同量；Adam2500对10k，此点落后，不能只凭大batch称有效',
        'IMPROVEMENT_MIXED_NOISE_G8': '同5k完整结果落后；旧TRAIN候选支持提高未转成该阶段部署收益'}
    for arm, update, reference in selections:
        pair = next((p for p in review.get('comparisons', []) if p['arm'] == arm and p['updates'] == update and p['reference_updates'] == reference), None)
        if not pair:
            continue
        rows.append([h.esc(NAMES.get(arm, arm))+' · '+h.number(update),
                     'Control '+h.number(reference)+'<small>'+h.fmt(pair['baseline_PDMS'], 6)+' → '+h.fmt(pair['target_PDMS'], 6)+'</small>',
                     signed(pair.get('delta_PDMS'))+'<small>'+ci(pair.get('CI95_PDMS'))+'</small>',
                     notes[arm], link(pair['evidence'], '完整配对')])
    return rows


def root_path(snapshot):
    return Path(snapshot.get('root', '/mnt/project/qwen-world-rl-optimization-20261030'))


def render_catalogue(root, snapshot, review, h):
    root = Path(root)
    catalogue_path = root/'analysis'/DIRECTORY/'CATALOGUE.json'
    catalogue = h.read_json(catalogue_path, snapshot['warnings'], snapshot['inputs']) or {}
    if catalogue.get('schema') != 'rl_primary_recipe_configuration_catalogue_v1':
        return ''
    audit, audit_path = load_local_configs(root, snapshot, h)
    link = lambda p, title: h.relative_link(root, str(p), title)
    number = lambda v: h.number(v) if v is not None else 'NR'
    lr = lambda v: f'{v:.0e}' if h.finite(v) else 'NR'

    def config_row(row):
        if row['status'] != 'PINNED_YAML_HASH_MATCH':
            return [h.esc(NAMES.get(row['arm'], row['arm'])), 'NR', 'NR', 'NR', 'NR', link(row['config'], row['status']) if row.get('config') and Path(row['config']).exists() else row['status']]
        schedule = row['schedule']
        clock = 'warm '+number(schedule['warmup'])
        if schedule['keep_until'] is not None:
            clock += ' → keep '+number(schedule['keep_until'])
        clock += ' → '+h.esc(schedule.get('kind') or 'NR')
        clock += '至 '+number(schedule['cosine_end'] if schedule['cosine_end'] is not None else schedule['horizon'])
        clock += '<small>horizon '+number(schedule['horizon'])+'；floor '+(str(schedule['floor']) if schedule['floor'] is not None else 'NR（YAML未单列）')+'</small>'
        bc = '关闭' if row['BC_enabled'] is False else '开启' if row['BC_enabled'] is True else 'NR'
        anchor = 'BC '+bc+'；'+h.esc(row['anchor_type'] or 'NR')+'型均值MSE β'+number(row['anchor_beta'])
        return [h.esc(NAMES.get(row['arm'], row['arm'])),
                number(row['scenes_per_window'])+'不同scene × G'+number(row['group_size'])+'<small>'+number(row['PG_candidates_per_window'])+'评分／窗口；'+number(row['inner_updates'])+' inner；accum '+number(row['accumulation'])+'</small>',
                lr(row['action_peak_lr'])+' / '+lr(row['memory_peak_lr'])+'<small>action / memory峰值</small>', clock,
                anchor+'<small>clip '+str(row['PPO_clip'])+'；全局A std='+str(row['global_std'])+'；cap '+number(row['advantage_cap'])+'</small>',
                link(row['config'], '锁定YAML')+'<small>'+h.esc(row['probability'])+'</small>']

    body = '<p class="notice"><b>先区分配方、机制和独立收益。</b>论文batch可能指prompt、候选或环境transition；LR还取决于训练scope、概率比、loss reduction及计时单位。下面保留原文口径，NR表示未披露／尚未核定，绝不补框架默认值。</p>'
    body += '<h3>论文实际用了什么</h3><p class="muted">配置目录核定时间：'+h.esc(catalogue.get('as_of_utc', 'NR'))+'。本地结果和当前任务随报告实时更新，论文目录保留其核定版本。</p>'
    paper_rows = []
    for paper in catalogue.get('papers', []):
        source = '<a href="'+h.esc(paper['source'])+'">'+h.esc(paper['name'])+'</a><small>'+h.esc(paper.get('domain', 'NR'))+' · '+h.esc(paper.get('location', 'NR'))+'</small>'
        paper_rows.append([source, _paper_value(paper.get('batch_and_group'), h),
                           _paper_value(paper.get('learning_rate'), h)+'<small>'+_paper_value(paper.get('schedule'), h)+'</small>',
                           _paper_value(paper.get('BC_KL'), h),
                           _paper_value(paper.get('advantage'), h)+'<small>'+_paper_value(paper.get('clipping'), h)+'</small>',
                           h.esc(paper.get('notes', 'NR'))])
    body += '<details><summary>'+number(len(paper_rows))+'篇primary逐项配置：batch／G、LR与调度、BC／KL、优势与clip</summary>'+h.table(['论文／原文位置', 'Batch与G（原口径）', '学习率／调度', 'BC／KL', '优势／裁剪', '迁移边界'], paper_rows)+'</details>'
    body += '<p>多篇LLM基线用恒定LR；warmup＋cosine并非共同最优结论。FPO Humanoid的annealing=False，未启用的decay字段不能视为实际退火；DPPO预训练cosine也不能冒充其RL调度。'+link(catalogue_path, '11篇核定目录与未核定字段')+'</p>'
    body += '<h3>我们的真实锁定配方</h3><p>以下从F／G／H／I各RUNS登记的科学YAML读取，逐文件匹配config_sha256；只读配置，不加载模型。所有步数均为实际optimizer时钟；场景、PG评分与额外B1成本分开。BC关闭时，文件中残留bc_coeff不生效。</p>'
    records = audit['rows']
    primary = [CONTROL, 'REVIVAL_G16', 'IMPROVEMENT_MULTISCENE32_G8', 'IMPROVEMENT_EARLY_DECAY_G8', 'IMPROVEMENT_VERIFIED_RETENTION_G8', 'STANDARD_ENDPOINT_LR02_G8', 'INTEGRATED_SAFETY_EP_B64_G16_RETENTION_WKC']
    selected = [next((row for row in records if row['arm'] == arm), None) for arm in primary]
    body += h.table(['代表配方', '不同场景／候选／复用', '两组峰值LR', '实际Adam调度', 'BC／anchor／信用尺度', '封存配置'], [config_row(row) for row in selected if row])
    body += '<p><b>β50不是LLM token-KL。</b>本地kl_type=x对应动作转移均值MSE锚定原IL70；不是额外离线BC loss，也不与论文token分布KL系数直接比大小。source W3的ratio按动作维度平均logp，Endpoint则用joint24；因此本地clip0.005与LLM的0.2不能按数字直接移植，Endpoint低LR的结果也不能推成“低LR普遍不好”。SafetyEP是安全信用＋可行子集EP、再按批rawA std归一化；纠偏／饱和／Dense等在各自独立schema上叠加，不能从“global std开”推断原PPO与SafetyEP优势完全相同。NR（YAML未单列）仅表示该字段未从此配置提取，不代表runtime未设置；完整实际协议见各研究原有封存调度说明。</p>'
    body += '<details><summary>全部'+number(len(records))+'个锁定配方与源码身份</summary>'+h.table(['运行臂', '不同场景／候选／复用', '两组峰值LR', '实际Adam调度', 'BC／anchor／信用尺度', '封存配置'], [config_row(row) for row in records])+'<p>'+link(audit_path, '逐文件SHA／配置提取证据')+'</p></details>'
    body += '<h3>哪些设计有实证，哪些只是共同基线</h3>'
    peak = review.get('observed_peak') or {}
    body += '<p>当前最高完整点仍按动态结果取值：'+h.esc(NAMES.get(peak.get('arm'), peak.get('arm', '待齐')))+' '+number(peak.get('updates'))+' 更新，PDMS <b>'+h.fmt(peak.get('PDMS'), 8)+'</b>。这是一套共同配方的观察峰值，不能把收益分别归功给warmcos、IL锚定或批级std；这些组件尚无完整独立因果消融。</p>'
    body += h.table(['设计／检查点', '明确基线与实际分', 'Δ PDMS／95%CI', '可支持的判断与预算边界', '证据'], _effective_rows(snapshot, review, h, link))
    body += '<p>Dense／纠偏／饱和有局部阶段好迹象，尚未持续刷新共同最高点；“未持续刷新”不等于方向被永久否定。Retention与Early的阶段正区间也不能直接加到最高分，更不能拿I的早期点断言综合方案有效或无效。当前I仍保持实际10k后评审、实际2k／5k／10k完整NAVTEST的既定计划。</p>'
    body += '<p class="muted">这里复用现有COMPLETE全12146场景／invalid0结果与配对区间，不新评分、不新bootstrap；固定训练seed／44物理log的CI不是多训练seed不确定性。论文引用支持配置事实和机制启发，不声称其分数与NAVTEST可比。</p>'
    return '<section id="'+SECTION_ID+'"><h2>论文配方、本地配置与有效证据</h2><p class="deck">当前实验进度优先；随后回答“哪些trick真的有证据，以及如何迁移”。</p>'+body+'</section>'
