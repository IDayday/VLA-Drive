"""Self-contained research report: native SVG, accessible tables and local JS."""
from html import escape
import json
import math
from pathlib import Path
from .diagrams import NAMES, DESCRIPTIONS, COMPARISON, architecture_svg
from .evidence import METHODS, finite

COLORS = ['#2463b9','#c16c20','#119181','#8759ad','#d45570','#758296','#368ab7','#a96918','#2a8a65','#9b4a9b']


def fmt(value, digits=4):
    return f'{value:,.{digits}f}' if finite(value) else '—'


def link(path, label='证据'):
    return f'<a href="{escape(str(path),quote=True)}">{escape(label)}</a>'


def table(headers, rows, *, ident='', classes=''):
    return f'<div class="table-scroll"><table id="{ident}" class="{classes}"><thead><tr>'+''.join(f'<th scope="col">{escape(h)}</th>'for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{cell}</td>'for cell in row)+'</tr>'for row in rows)+'</tbody></table></div>'


def chart(series, title, *, ylabel='PDMS', min_y=None):
    values=[(step,value)for _,points in series for step,value in points if finite(step)and finite(value)]
    if not values:
        return '<p class="muted">尚无可发布的曲线。</p>'
    xmax=max(x for x,y in values)or 1
    low=min(y for x,y in values);high=max(y for x,y in values)
    margin=max((high-low)*.12, .001 if high<1 else .5)
    low=low-margin if min_y is None else min_y;high+=margin
    width,height=1000,320
    x=lambda v:74+v/xmax*892
    y=lambda v:264-(v-low)/(high-low or 1)*216
    parts=[f'<div class="chart"><svg viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}"><title>{escape(title)}</title>',
           '<rect width="1000" height="320" rx="12" fill="white"/>']
    for i in range(5):
        value=low+(high-low)*i/4;yy=y(value)
        parts.append(f'<line x1="74" y1="{yy}" x2="966" y2="{yy}" stroke="#e7edf3"/><text x="63" y="{yy+5}" text-anchor="end" class="axis">{value:.3f}</text>')
    for i in range(6):
        value=xmax*i/5;xx=x(value)
        parts.append(f'<text x="{xx}" y="287" text-anchor="middle" class="axis">{value/1000:g}k</text>')
    parts.append(f'<text x="74" y="28" class="chart-title">{escape(title)}</text><text x="957" y="310" text-anchor="end" class="axis">有效 optimizer updates</text>')
    legend=[]
    for n,(name,points)in enumerate(series):
        points=sorted((a,b)for a,b in points if finite(a)and finite(b))
        if not points:continue
        color=COLORS[n%len(COLORS)]
        coords=' '.join(f'{x(a):.2f},{y(b):.2f}'for a,b in points)
        parts.append(f'<g class="plot-series" data-series="{escape(name,quote=True)}"><polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.3"/>')
        for a,b in points:
            parts.append(f'<circle cx="{x(a):.2f}" cy="{y(b):.2f}" r="4" fill="{color}"><title>{escape(name)} · {a:,} updates · {ylabel} {b:.6f}</title></circle>')
        parts.append('</g>')
        legend.append(f'<label><input type="checkbox" checked data-plot="{escape(name,quote=True)}"/><i style="background:{color}"></i>{escape(name)}</label>')
    return ''.join(parts)+'</svg><div class="legend">'+''.join(legend)+'</div></div>'


ARCH_NOTES = {
 'WORLD': ['基线先前已完成100k IL；V12与损失实验仍沿用该架构。不是旧 DDP，也不是30层 Wan 初始化动作专家。',
           'Predictive32 是部署中的观察条件表征，不是真实未来图像。原四辅助任务是 DINO、WA未来、当前对象、候选六小分。',
           '候选轨迹以 clean replay 进入同一 actor，后置任务学习其 hidden；没有训练一个部署时选轨的 scorer。'],
 'A': ['核心问题：辅助世界监督是否可以仅通过梯度内化，不增加在线世界模块？',
       'W/R 共享20×384世界流。actor仍为20×1024，只读C160；部署删除世界分支。',
       'post loss 通过未detach的 candidate actor hidden 进入 actor/Qwen/C；不意味着自动提高该候选生成概率。'],
 'B': ['核心问题：世界推理和动作推理共享参数，是否更利于 IL 学习及 RL 适配？',
       'P/A/R 共用20×1024主干，依靠方向性注意力分离信息。P固定时间，A用当前sigma，R用clean sigma0。',
       'P每场景计算一次；动作每步只更新A。后置R删除，但它训练过的共享权重保留。RL时共享LoRA改变P，不能跨inner缓存当前P。'],
 'C': ['核心问题：独立世界流的逐层表示，是否比单次拼接memory更能帮助动作？',
       '20×384世界流和20×1024actor一一对应；4组K/V桥接共享actor原Q/O。C/W分别softmax，用小非零gate融合。',
       'R复用世界流，并读取候选actor逐层hidden；R不反向进入同次W前向。部署保留W与world→actor桥接。'],
 'D': ['核心问题：直接让语言主干承担动作去噪，是否能更充分共享观测、推理与动作表示？',
       '原Qwen3-VL 28×2048文本层保留GQA/RMSNorm/mRoPE/DeepStack；新增连续动作3↔2048、P/R和256维时间FiLM。',
       '动作A8块内双向，非逐点自回归。每步从不可变C+P前缀出发；没有独立500M actor。RL文本LoRA会改变C/P，必须重算当前前缀。']}


CSS = '''
:root{--ink:#20324e;--muted:#64748b;--line:#dce5ef;--blue:#2463b9;--bg:#f2f5f9}*{box-sizing:border-box}body{margin:0;color:var(--ink);background:var(--bg);font:15px/1.75 system-ui,-apple-system,"Noto Sans CJK SC","Microsoft YaHei",sans-serif}a{color:var(--blue);text-underline-offset:3px}header{background:#172d4e;color:white;padding:38px max(24px,calc((100vw - 1450px)/2));}header p{color:#c7d7ef;max-width:1100px;margin:8px 0}h1{font-size:29px;margin:0;letter-spacing:.5px}h2{font-size:23px;margin:0 0 14px}h3{font-size:18px;margin:24px 0 10px}main{max-width:1500px;margin:auto;padding:24px}nav{position:sticky;top:0;z-index:10;background:#fffefbeF;border-bottom:1px solid var(--line);padding:10px 24px;display:flex;gap:22px;flex-wrap:wrap;backdrop-filter:blur(8px)}nav a{text-decoration:none;font-weight:650;font-size:14px}section{background:white;border:1px solid var(--line);border-radius:16px;padding:28px;margin-bottom:24px;scroll-margin-top:70px}p{margin:9px 0}.muted,.note,small{color:var(--muted)}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:22px}.card{background:#fff;border:1px solid var(--line);border-radius:14px;padding:20px}.card strong{font-size:28px;display:block;color:#174f89}.card span{font-size:13px;color:var(--muted)}.callout{border-left:4px solid var(--blue);padding:12px 16px;background:#eef5ff;border-radius:4px;margin:15px 0}.warn{border-color:#b56e11;background:#fff6e8}.good{border-color:#218670;background:#edf8f4}.table-scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}th{text-align:left;white-space:nowrap;background:#eef3f9;color:#4d617d;font-size:13px}td,th{padding:11px 13px;border-bottom:1px solid var(--line);vertical-align:top}tbody tr:hover{background:#f8fbff}.numeric td:not(:first-child){white-space:nowrap}progress{accent-color:var(--blue);width:110px;height:8px}.badge{display:inline-block;border-radius:30px;background:#eef2f8;padding:2px 10px;font-size:12px}.diagram-toolbar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:12px 0}button,.button{font:inherit;padding:7px 14px;border:1px solid #cad7e7;border-radius:8px;background:white;color:var(--ink);cursor:pointer;text-decoration:none}button[aria-selected=true]{color:white;background:#2463b9;border-color:#2463b9}.diagram-panel[hidden]{display:none}.diagram-frame{overflow:auto;border:1px solid var(--line);border-radius:12px;background:white}.diagram-frame svg{display:block;width:100%;min-width:760px}.diagram-frame.deployment .node.train,.diagram-frame.deployment .aux,.diagram-frame.deployment .back{visibility:hidden}.diagram-frame:fullscreen{overflow:auto;padding:30px}.diagram-frame:fullscreen svg{min-width:1280px}.architecture-note{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-top:14px}.architecture-note ul{margin:8px 0;padding-left:22px}.chart svg{display:block;width:100%;min-width:580px}.chart{overflow:auto;border:1px solid var(--line);border-radius:12px;margin:16px 0}.chart text{font:13px system-ui,sans-serif;fill:#526684}.chart .chart-title{font-size:16px;font-weight:650;fill:#20324e}.legend{padding:10px 18px;display:flex;gap:14px;flex-wrap:wrap;font-size:13px}.legend label{cursor:pointer;display:flex;align-items:center;gap:5px}.legend i{width:10px;height:10px;border-radius:50%;display:inline-block}.two-col{display:grid;grid-template-columns:1fr 1fr;gap:22px}details{border:1px solid var(--line);border-radius:10px;padding:12px 16px;margin:12px 0}summary{cursor:pointer;font-weight:650}code{background:#eef2f7;padding:2px 5px;border-radius:4px;font-size:13px}input[type=search]{font:inherit;border:1px solid var(--line);border-radius:8px;padding:9px 12px;width:min(450px,100%);margin:10px 0}footer{text-align:center;color:var(--muted);padding:20px 0 40px}.diagram-title{font-size:16px;font-weight:650}.pill{color:#1b6985;background:#e8f6fa;border-radius:5px;padding:2px 7px;font-size:12px}@media(max-width:900px){.cards{grid-template-columns:repeat(2,1fr)}.two-col,.architecture-note{grid-template-columns:1fr}main{padding:12px}section{padding:18px}h1{font-size:23px}nav{position:static;gap:12px}.diagram-frame svg{min-width:850px}}@media print{nav,.diagram-toolbar,button,input{display:none}.diagram-panel[hidden]{display:block}section{break-inside:avoid}header{background:white;color:#20324e}.cards{grid-template-columns:repeat(4,1fr)}body{background:white}}
'''


SCRIPT = '''
document.querySelectorAll('[data-arch]').forEach(button=>button.addEventListener('click',()=>{
 document.querySelectorAll('[data-arch]').forEach(b=>b.setAttribute('aria-selected',String(b===button)));
 document.querySelectorAll('.diagram-panel').forEach(p=>p.hidden=p.id!==`diagram-${button.dataset.arch}`);
}));
document.querySelectorAll('[data-deploy]').forEach(b=>b.addEventListener('click',()=>{
 const f=b.closest('.diagram-panel').querySelector('.diagram-frame');const on=f.classList.toggle('deployment');
 b.setAttribute('aria-pressed',String(on));b.textContent=on?'显示训练 + 部署':'只看部署保留模块';
}));
document.querySelectorAll('[data-fullscreen]').forEach(b=>b.addEventListener('click',()=>{
 const f=b.closest('.diagram-panel').querySelector('.diagram-frame');if(f.requestFullscreen)f.requestFullscreen();
}));
document.querySelectorAll('[data-plot]').forEach(c=>c.addEventListener('change',()=>{
 c.closest('.chart').querySelectorAll('.plot-series').forEach(g=>{if(g.dataset.series===c.dataset.plot)g.style.display=c.checked?'':'none';});
}));
const filter=document.querySelector('#score-filter');if(filter)filter.addEventListener('input',()=>{
 const q=filter.value.toLowerCase();document.querySelectorAll('#score-table tbody tr').forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(q));
});
// Reload only the local generated document; this starts no training/log polling.
if(location.protocol==='http:'||location.protocol==='https:')setTimeout(()=>location.reload(),1800000);
'''


def render(s, history):
    scores=s['scores'];runs=s['runs'];by={key:[r for r in scores if r['method']==key]for key in METHODS}
    raw_best=max(by['RAW'],key=lambda x:x['PDMS'],default=None)
    running=[r for r in runs if r['status'].startswith('训练更新中')]
    failed=[r for r in runs if r['status']=='训练故障 / 待恢复']
    best=max((r for r in scores if not r['method'].startswith('ARCH_')),key=lambda x:x['PDMS'],default=None)
    current_run={r['id']:r for r in runs}
    def card(value,label,note=''):
        return f'<div class="card"><span>{escape(label)}</span><strong>{value}</strong><span>{escape(note)}</span></div>'
    parts=['<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
           '<title>Qwen-World IL · 动态研究报告</title><style>'+CSS+'</style></head><body>',
           '<header><div class="pill">IL RESEARCH / LIVE EVIDENCE</div><h1>Qwen-World IL：模型设计、训练进展与完整 Navtest</h1>',
           '<p>从原始 Qwen-World 到 A/B/C/D 四种宏观架构。把“当前在做什么、测试是否提高、依据是什么、下一步做什么”放在同一页。</p>',
           f'<p>最近核验：<b>{s["generated_utc"]}</b> · 常规 30 分钟 · 完整阶段结果 / 重要故障即时更新 · '+link('/mnt/project/qwen-world-rl-optimization-20261030/report.html','查看 RL 动态报告')+'</p></header>',
           '<nav>'+''.join(f'<a href="#{key}">{name}</a>'for key,name in [('summary','研究结论'),('progress','当前进度'),('architectures','五版架构图'),('results','完整测试结果'),('queued','补测进度'),('learning','学习与辅助任务'),('architecture-diagnosis','架构诊断'),('next','下一步'),('provenance','更新机制')])+'</nav><main>',
           '<div class="cards">',card(fmt(best['PDMS'])if best else'—','当前已测 IL 最高',f'{METHODS.get(best["method"],best["method"])} · {best["updates"]:,} updates'if best else'尚无完整结果'),
           card(str(len(running)),'日志新鲜的 IL 训练','全部8 rank共同有效更新；不是仅看controller RUNNING'),
           card(str(len(scores)),'完整 IL Navtest 观察点','每点12146场景 / invalid=0；去除重复模型观察'),
           card(str(len(failed)),'当前训练故障 / 待恢复','历史已解决异常单列，不据旧FAILURE文件误判'),'</div>',
           '<section id="summary"><h2>目前结论与证据边界</h2>']
    if raw_best:
        parts.append(f'<p><b>原始 Qwen-World IL 已完成100k。</b>已测最高为 {raw_best["updates"]:,} 更新的 <b>{raw_best["PDMS"]:.4f}</b>；后期训练轨迹 L1 下降不等于 Navtest 持续上涨。以完整 Test 性能判断提升，训练 loss 只解释学习过程。</p>')
    if by['V12']:
        last=max(by['V12'],key=lambda x:x['updates'])
        parts.append(f'<p><b>优化轨迹仍在检验“更宽可行域与更高进度是否能被学生吸收”。</b>最新已测 {last["updates"]:,} 更新 PDMS {last["PDMS"]:.4f}。标签的历史约98分是教师标签分数，不能当作学生成绩；高 EP 也不能掩盖 NC/TTC 的净损失。</p>')
    parts.append('<p><b>损失实验：</b>同起点10k预实验选中时间分布加权、Huber继续；物理通道加权与delta没有进入完整续训。切换旧63k模型再训2k的结果单列为预实验，不能代替从通用初始化训练的结论。</p>')
    parts.append('<p>现行基线RL主要沿IL70k的同一架构，更新动作attention LoRA和scene/predictive memory，Qwen冻结；其收益见独立RL报告。不能把该基线的91分RL结果套给尚未完成IL的A/B/C/D。</p>')
    selected=s.get('selection',{})
    selection_rows=[]
    for key,row in selected.get('results',{}).items():
        interval=row.get('paired_CI_points',[])
        selection_rows.append([escape(METHODS.get(key,key)),fmt(row.get('PDMS')),fmt(row.get('delta_points')),
            '['+', '.join(fmt(v)for v in interval)+']','继续100k'if key in selected.get('selected',[])else'10k筛选后不续训'])
    parts.append('<details open><summary>四组损失设计都做过：共同10k的实际筛选结果</summary><p>原始L1对照10k完整Test为80.5319。时间加权和Huber进入完整续训；这不等于已经证明100k收益。此次筛选观察过Test，保留该历史。</p>'+table(['从头IL损失','10k PDMS','相对L1：点','44物理log配对95%区间','实际决定'],selection_rows,classes='numeric')+'</details>')
    parts.append('<div class="callout">A/B/C/D 正在执行各自100k IL；目前不同更新数的早期 Test 分数只能观察学习状态，不能据此提前淘汰或宣布架构胜负。它们增加了未来track、事实动作条件WA、总分与违规时序任务，不是原基线的纯“只换架构”匹配消融。</div>')
    if failed:
        parts.append('<div class="callout warn"><b>需要处理的实际训练阻塞：</b>'+', '.join(escape(r['label'])+' 停在 '+fmt(r['updates'],0)+' 更新'for r in failed)+'。报告按真实终止收据与日志时效标记，未将其显示为正常运行。</div>')
    parts += ['</section><section id="progress"><h2>现在正在做什么、进度到哪里</h2><p class="muted">更新数取8个rank共同下界；完整权重取LATEST独立列示。step时间是最近完整窗口的最慢rank wall，不是GPU独占计算时间。</p>']
    rows=[]
    for r in runs:
        rows.append([escape(r['label']),f'<span class="badge">{escape(r["status"])}</span>',fmt(r['updates'],0)+' / '+fmt(r['target'],0)+f'<br><progress max="1" value="{r["fraction"]or 0}"></progress>',
                     fmt(r['latest_checkpoint'].get('updates'),0),escape(r['host']or'历史 / 无当前host证据'),
                     fmt(r['recent_losses'].get('x'),6),fmt(r['recent_seconds_per_update'],2)+' s',
                     fmt(r['estimated_remaining_hours'],1)+' h'if r['estimated_remaining_hours']is not None else'—',link(r['root'],'运行记录')])
    parts.append(table(['实验','实际状态','共同更新 / 预算','完整checkpoint','host','近期轨迹x损失','s/update','粗略剩余','来源'],rows,classes='numeric'))
    parts.append('<p class="note">ETA仅按最近窗口外推，未计未来保存、评测、资源共享变化；运行中日志新鲜也不等于独立核实远端GPU进程。训练与故障恢复由原controller负责，本报告只读。</p></section>')
    parts += ['<section id="architectures"><h2>架构示意图：到底哪些表示在帮助动作？</h2>',
        '<p>五张图分别按实际实现绘制。蓝/绿模块在部署保留，橙色模块仅训练运行，紫色表示共享主干；粉色虚线表示辅助损失的回传方向。可切换“只看部署”、全屏查看或下载 SVG。</p>',
        '<div class="diagram-toolbar" role="tablist" aria-label="选择架构">'+''.join(f'<button role="tab" data-arch="{key}" aria-selected="{str(key=="WORLD").lower()}">{"当前 Qwen-World"if key=="WORLD"else key}</button>'for key in NAMES)+'</div>']
    for key in NAMES:
        parts.append(f'<article class="diagram-panel" id="diagram-{key}"'+(' hidden'if key!='WORLD'else'')+'>'+f'<p class="diagram-title">{escape(NAMES[key])}</p><div class="diagram-toolbar"><button data-deploy aria-pressed="false">只看部署保留模块</button><button data-fullscreen>全屏示意图</button><a class="button" href="figures/architecture_{key}.svg" download>下载 SVG</a></div><div class="diagram-frame">'+architecture_svg(key)+'</div>')
        parts.append('<ul>'+''.join('<li>'+escape(t)+'</li>'for t in ARCH_NOTES[key])+'</ul></article>')
    parts.append(table(['版本','在线动作前路径','权重 / 推理关系','动作后训练','部署保留'],[[escape(x)for x in row]for row in COMPARISON]))
    parts.append('<h3>共同动作与信息边界</h3><p>单条8×3绝对[x,y,heading]轨迹，未来0.5–4.0秒；不是旧8×4或逐点自回归。使用源min/max编码，但actor学习的是直接clean-x L1：<code>xσ=(1−σ)x₀+σε</code>，<code>σ=5u/(1+4u)</code>。十步ODE从噪声逐步预测clean轨迹。</p><p>未来视频、GT对象、真实评分与候选标签只用于监督，不进入部署CurrentInputs。偏离真实记录轨迹的候选只监督原生模拟可得的评分/事件；不把记录中的未来视频当作其反事实未来。</p>')
    parts.append('<h3>四架构的统一监督</h3><p><code>10Lx + LDINO + 0.5LWA,pre + 0.1Lobject + 0.2Ltrack + 0.25LWA,post,factual + 0.2Levaluation</code>。后置评价包含六小分、总分、同场景排序、NC/DAC/TTC首次违规的9类时间分布；R16为8个时间位置、6个metric、2个汇总token。</p><p>当前基线则为原四辅助项，并无上述新增track/总分/时序任务。Qwen与scene的初始化一致不等于五版训练任务完全相同。</p></section>')
    parts += ['<section id="results"><h2>完整 Navtest：绝对分数、各分项与趋势</h2>',
        '<p class="muted">只发布满足完整性、seal、seed、scorer、人口、单候选ODE合同的结果。DEV单列为诊断/既定父模型选择输入，不作为本页方法提分结论。</p>']
    for keys,title in [(['RAW','V12'],'原轨迹 vs 优化轨迹：各自完整学习曲线'),(['RAW','TIME_RESIDUAL','HUBER','PHYSICAL','DELTA'],'从通用初始化训练：损失设计对比'),(['ARCH_'+a for a in'ABCD'],'四架构早期 Test：更新预算不同，尚不能排最终名次')]:
        parts.append(chart([(METHODS[k],[(r['updates'],r['PDMS'])for r in by[k]])for k in keys],title))
    parts.append('<input id="score-filter" type="search" placeholder="筛选方法、权重步数或标签…" aria-label="筛选完整测试结果"/>')
    rows=[]
    metric_names=['no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound','ego_progress','comfort','driving_direction_compliance']
    for r in sorted(scores,key=lambda r:(r['method'],r['updates'])):
        rows.append([escape(METHODS.get(r['method'],r['method'])),fmt(r['updates'],0),'<b>'+fmt(r['PDMS'])+'</b>',
                     *[fmt(r['means'].get(k),3)for k in metric_names],link(r['evidence'],'SUMMARY')])
    parts.append(table(['方法','IL更新','PDMS','NC','DAC','TTC','EP','Comfort','DDC','完整证据'],rows,ident='score-table',classes='numeric'))
    parts.append('<details><summary>DEV诊断 / 既定父模型选择记录（不混入Test主表）</summary>'+table(['架构','更新','DEV PDMS','证据'],[[escape(r['architecture']),fmt(r['updates'],0),fmt(r['PDMS']),link(r['evidence'])]for r in s['dev_diagnostics']])+'</details></section>')
    parts.append('<section id="queued"><h2>已部署的补测任务</h2><p>A5k已经完整测得78.3173，直接复用。B/C/D补齐5k后才能比较同更新预算；新结果完成后自动进入上方Test主表。下表不会发布部分人口均分。</p>'+table(['权重','IL更新','已记录 / 应有场景','实际阶段','输出'],[[escape(r['label']),fmt(r['updates'],0),f'{r["scenes_completed"]:,} / 12,146',escape(r['status']),link(r['root'])]for r in s.get('pending_evaluations',[])])+'</section>')
    parts += ['<section id="learning"><h2>轨迹损失、辅助任务与测试分数的关系</h2><p>训练x损失来自noisy clean-x重建，并非单候选ODE部署的几何误差；不同目标、horizon权重与Huber形式的优化loss不能直接比较绝对大小。训练loss下降也不是辅助预测“准确率已提高”的证明。</p>']
    rows=[]
    for key,r in current_run.items():
        loss=r['recent_losses'];rows.append([escape(r['label']),fmt(r['updates'],0),fmt(loss.get('x'),6),fmt(loss.get('dino'),6),fmt(loss.get('wa_pre',loss.get('future')),6),fmt(loss.get('object'),6),fmt(loss.get('future_track'),6),fmt(loss.get('wa_post'),6),fmt(loss.get('evaluation',loss.get('post')),6)])
    parts.append(table(['方法','更新','x / 自身目标','DINO','WA前置','对象','未来track','WA后置','后置评价'],rows,classes='numeric'))
    for keys,title in [(['ARCH_'+a for a in'ABCD'],'四架构轨迹 x L1：报告启动后的共同rank窗口'),(['RAW','V12','TIME_RESIDUAL','HUBER'],'基线/标签/损失：各自优化损失（不能据大小判优）')]:
        series=[]
        for k in keys:
            points=[(h['updates'],h.get('x'))for h in history if h['id']==k]
            series.append((METHODS[k],points))
        parts.append(chart(series,title,ylabel='TRAIN x loss'))
    parts.append('<details open><summary>固定场景辅助任务审计：20k / 40k / 60k（已有证据，无新推理）</summary><p>以下是固定场景的旧基线审计；TRAIN/DEV分别列出，不自动代表整个人口，也不能把这些精度直接套到新A/B/C/D。对象precision在不完整coverage下保留未定义。</p>')
    rows=[]
    for r in s['auxiliary_audit']:
        rows.append([fmt(r['updates'],0),escape(r['role']),fmt(r['scenes'],0),fmt(r.get('dino_MSE'),5),fmt(r.get('dino_cosine'),4),fmt(r.get('future_MSE'),5),fmt(r.get('future_cosine'),4),fmt(r.get('object_matched_class_accuracy'),4),fmt(r.get('object_xy_error_m'),2),fmt(r.get('object_recall_2m'),4),link(r['evidence'])])
    parts.append(table(['权重','人口','场景','DINO MSE','DINO cosine','WA MSE','WA cosine','对象匹配分类率','对象XY误差m','2m召回','证据'],rows,classes='numeric')+'</details>')
    parts.append('<h3>应怎样解释损失与PDMS</h3><p>同一归一化误差对应不同物理量：一个单位分别约为纵向33.37m、横向21m、航角1.765rad。归一化L1不能统一换算为厘米；每horizon的GT分布也不同，时间加权已单独登记。八个点联合生成，并非前一点通过网络自回归地强制传递到后一点。</p>')
    parts.append('<p>原始与V12的已完成分析显示，loss与PDMS在长时间尺度有关联，但相邻checkpoint并不单调对应。优化轨迹提高标签EP与局部扰动容差，不保证学生重建误差仍位于同一个安全邻域；其heading、横向、纵向误差与实际NC/TTC需要一起看。</p></section>')
    matched=s.get('matched_architecture_analysis',{})
    if matched.get('status')=='COMPLETE':
        parts.append('<section id="architecture-diagnosis"><h2>B / C / D 低分：学得慢，还是设计与部署存在差距？</h2><p><b>目前不能证明“训练慢、上限更高”。</b>旧观察混用了A5k、B/C4k和D2k；D的有效joint训练甚至只有A的四分之一。共同5k的完整Navtest已补测，结论以新Test为准。</p>')
        same_step={a:next((r for r in by['ARCH_'+a]if r['updates']==5000),None)for a in'ABCD'}
        parts.append('<p><b>共同5k完整Test：'+str(sum(r is not None for r in same_step.values()))+'/4已完成。</b></p>'+table(['架构','5k完整Navtest PDMS','状态'],[[a,fmt(row['PDMS'])if row else'—','完整12146'if row else'正在评测 / 尚未完成']for a,row in same_step.items()]))
        rows=[]
        for row in matched['rows']:
            if row['end']!=5000:continue
            loss=row['mean_losses'];rows.append([row['architecture'],fmt(loss['x'],6),fmt(loss['dino'],5),fmt(loss['wa_pre'],5),fmt(loss['evaluation'],5),fmt(row['mean_slowest_rank_seconds'],2),fmt(row['mean_data_wait_seconds'],2),fmt(row['mean_total_gradient_norm'],2)])
        parts.append(table(['架构 / 4501–5000窗口','轨迹x L1','DINO','WA前置','后置评价','s/update','data wait s','engine总grad_norm'],rows,classes='numeric'))
        parts.append('<p>实际核对：四组这1000个早/晚共同更新窗口的全局32场景顺序一致，全部8rank无skipped update；不声称已逐值验证每个随机噪声。C的x损失与A几乎相同，D在5k的x损失更低。因此“不会拟合”不足以解释所有早期PDMS差距，也不能由MSE/L1推导最终分数。</p>')
        parts.append('<div class="two-col"><div><h3>有证据的观察</h3><ul><li>B的5k轨迹损失略高，早期重建学习存在差距。</li><li>C并未明显落后于A的训练x损失，需要检查ODE下安全/进度后果。</li><li>D从早期较高误差迅速下降；低分旧权重仅2k，不能代表现在的D5k。</li><li>共享资源历史窗口内D每步约10.56s、A约6.29s；计算成本和学习效率必须分开。</li></ul></div><div><h3>仍属假设，尚无因果证据</h3><ul><li>B的共享P/A/R是否发生任务梯度竞争？总loss或总grad_norm不能回答。</li><li>C世界gate与桥接是否提供有效在线信息？架构有连接不等于已学会使用。</li><li>D大共享主干是否需要更长适配？更低训练L1不能证明更高上限。</li><li>单次x0重建到十步ODE的误差积累，是否导致相似L1却不同PDMS？需要新Test与已有轨迹共同分析。</li></ul></div></div><p>下一步只补齐证据：先看共同5k与各自新权重的完整Test、小分、零分转移，再决定是否需要针对某一路径做有限梯度/前缀利用诊断；不中途改四组统一损失，也不因早期低点缩短已登记100k。</p></section>')
    parts += ['<section id="next"><h2>遇到什么问题，下一步做什么</h2>',table(['当前问题 / 判断','已有证据','下一步（计划与执行区分）'],[
        ['原始IL已收敛到约88.6–88.8；更低L1不等于更高PDMS','完整100k与多个中间Test；不是只看DEV','保留原IL作为参照；不因追92分继续无界增加IL预算。'],
        ['V12标签高分尚未完全转化为学生部署收益','同架构标签替换、完整Test及已有几何/分项分析','继续原定100k；在已登记检查点看净PDMS、NC/TTC与EP，不只看标签98分。'],
        ['时间加权早期有效，后续分数有波动','10k筛选结果与15k/16k/22k等完整Test曲线','按原配方继续；有限匹配预算下判断，避免用一个低点断言方法失败。'],
        ['Huber故障与恢复事务','原rank6被SIGKILL；14k完整native已校验，661步日志归入失败尝试','由已审阅恢复入口在空闲八卡继续原100k；每次报告检查真实状态，不能以已启动代替恢复成功。'],
        ['ABCD尚在早期，绝对分数与训练效率不同','同一七项监督、当前更新/step时间、四个早期完整Test','共同5k与较新中间权重已部署补测；各自继续100k，不短训筛组；报告本身不派发GPU。'],
        ['新架构的后置任务能否改善动作仍未证明','辅助损失有梯度，不等于有真实规划收益','先完成IL，再各自固定父模型RL50k；分开报告IL架构收益与RL增益。'],
    ])]
    parts.append('<p class="note">本页只更新解释与证据；不改loss、数据、训练预算、checkpoint选择器或部署协议，不启动模型、评分或新实验。</p></section>')
    parts += ['<section id="provenance"><h2>协议、来源与动态更新</h2><p>主指标：canonical NAVSIM v1.1；12146场景 / 44物理log；单候选 clean-x ODE，K10 / shift5，推理seed2101。训练seed42。PDMS与各分项按0–100展示；内部reward仍为0–1。历史Test被多次观察，不能宣称新盲测。</p>',
        '<p>报告读取封存SUMMARY/identity、运行收据、每rank最多64条日志尾部。常规每1800秒更新；仅阶段完成、完整新评测、重要异常、选择结果与资源授权事件即时刷新。每步progress不会触发渲染，事件更新不延后原半小时deadline。</p>',
        '<p>五张图是对实际代码的解释。当前旧World与四框架的源版本、辅助任务和在线世界路径均有差异；“借鉴上游设计”不表述为复制完整官方训练。</p>']
    parts.append('<details><summary>架构源码身份 / 文件hash</summary><p>四架构科学源码：<code>'+escape(s['source_audit']['scientific_source'])+'</code></p>'+table(['实际源码','SHA256'],[[link(r['path'],Path(r['path']).name),'<code>'+r['sha256']+'</code>']for r in s['source_audit']['files']])+'</details>')
    parts.append('<details><summary>资源与异常记录</summary><p>当前策略允许的host（不等于实际空闲）：'+escape(', '.join(s['policy_authorized_hosts']))+'</p><p>报告不查询或占用GPU。共享GPU的step耗时只作为当时执行窗口；不能当作独占硬件对照。</p><ul>'+''.join('<li>'+escape(w)+'</li>'for w in s['warnings'])+'</ul></details>')
    health=s.get('health',{})
    if health:
        parts.append('<details open><summary>每次报告同步检查任务健康</summary><p>只恢复经过审阅的基础设施故障，持有owner和rank锁，从合法native checkpoint继续；已完成、筛除、用户终止和需人工审阅的pilot不会自动重启。未知或数值故障保留证据并报警，不改变训练配方。</p>'+table(['IL任务','实际状态','更新','维护动作'],[[r['id'],r['status'],str(r.get('updates')),r['action']]for r in health['runs']]))
        recovery=health.get('recovery')
        if recovery:
            parts.append('<p>Huber恢复入口：'+link(recovery['review'],'审阅收据')+'；恢复边界14,000，原失败尾部661步单独存档并重算。入口状态 '+escape(recovery['status'])+'；实际是否恢复以主运行表新日志为准。</p>')
        if health.get('automatic_action_failed'):
            parts.append('<p class="callout">恢复辅助入口失败，需要审阅；不会无限重试。'+escape(health['automatic_action_failed']['error'][-500:])+'</p>')
        parts.append(table(['RL任务','owner登记状态','全8rank日志更新','合法阶段更新','动作'],[[r['id'],r['status'],str(r.get('journal_updates')),str(r.get('committed_registry_updates')),r['action']]for r in health.get('rl_runs',[])])+'</details>')
        cadence=health.get('navtest_cadence',{})
        if cadence.get('registration'):
            parts.append('<details open><summary>每2k有效更新的完整Navtest高并发队列</summary><p>当前仍训练的七组IL，采用2k网格；与已登记检查点至少间隔2k，已有结果或正在评测的点不会重复。每模型最多32个scene shard，同时最多两个模型，空闲卡优先、核查后共享。每条仍为同一scene绑定seed2101的单候选K10shift5，canonical v1.1全12,146场景。评测完成即释放worker，不新增训练seed或best-of-N。</p>'+table(['IL任务','checkpoint更新','实际队列状态','评测记录'],[[j['method'],str(j['updates']),j['status'],link(j['root'],'运行目录')]for j in(cadence.get('state')or{}).get('jobs',[])])+'</details>')
    windows=[]
    for host,row in s.get('policy_host_windows',{}).items():
        from datetime import datetime,timezone
        expiry=row.get('expires_unix')
        windows.append([escape(host),'不限时'if row.get('unbounded')else datetime.fromtimestamp(expiry,timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')if expiry else'由任务独立许可限定',link(row.get('record',''),'授权收据')])
    parts.append('<details><summary>临时资源的实际到期时间</summary>'+table(['host','当前授权到期','来源'],windows)+'</details>')
    parts.append('<p>'+link('reporting/report_snapshot.json','机器可读快照')+' · '+link('reporting/REPORT_CONFIG.json','更新规则')+' · '+link('reporting/OWNER.json','唯一报告owner')+'</p></section></main>')
    parts.append('<footer>所有主成绩来自完整单轨Test · 每次更新执行健康检查与已审阅的有限native恢复 · 不改算法、不选轨</footer><script>'+SCRIPT+'</script></body></html>')
    return ''.join(parts)
