"""Source-checked, editable SVG diagrams; no model imports or raster services."""
from html import escape

NAMES = {'WORLD': '当前 Qwen-World：场景 / predictive memory + 独立 actor',
         'A': 'A · 世界能力在训练中内化，部署删除世界分支',
         'B': 'B · P / A / R 共用一个决策 Transformer',
         'C': 'C · 独立世界流通过逐层 K/V 帮助动作流',
         'D': 'D · Qwen 语言层直接承担连续动作生成'}
DESCRIPTIONS = {
 'WORLD': '当前三视图、导航与合法ego经Qwen、C160和predictive32进入独立20层actor。训练辅助读出删除；predictive32部署保留。',
 'A': 'Actor只读取C160。W32/R16属于共享紧凑世界分支，只在训练中运行。后置损失经未detach的候选actor hidden回传，但没有W到actor的在线连接。',
 'B': 'P32、A8和R16复用20层1024宽主干。P不接收动作或sigma；A读取P；R读取同一候选的A，不能被P或A读取。',
 'C': 'W32由20层384宽世界流产生。每层世界状态经分组K/V桥接进入对应actor层，两路attention分开softmax并由gate混合；R训练专用。',
 'D': 'Qwen3-VL原生28层2048宽文本主干同时处理当前C、P32、A8及训练R16。视觉DeepStack只注入当前图像位置。动作FiLM只作用于A，动作不是文本自回归。'}
COLORS = {'online': ('#e8f1ff', '#376ac7'), 'train': ('#fff3df', '#b87816'),
          'shared': ('#f0eaff', '#7859b5'), 'out': ('#e2f5ef', '#238572')}


class Drawing:
    def __init__(self, key):
        self.key = key
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1280 710" role="img" aria-labelledby="title-{key} desc-{key}">',
            f'<title id="title-{key}">{escape(NAMES[key])}</title><desc id="desc-{key}">{escape(DESCRIPTIONS[key])}</desc>',
            '<defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#61718d"/></marker>'
            '<marker id="grad" markerWidth="9" markerHeight="9" refX="8" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#c34f7b"/></marker></defs>',
            '<style>text{font-family:"Noto Sans CJK SC","Microsoft YaHei",sans-serif;fill:#20324e}.h{font-weight:700;font-size:19px}.s{font-size:15px}.note{font-size:15px;fill:#536580}.edge{fill:none;stroke:#61718d;stroke-width:2.2;marker-end:url(#arrow)}.aux{stroke:#b87816;stroke-dasharray:7 5}.back{stroke:#c34f7b;stroke-dasharray:4 5;marker-end:url(#grad)}</style>',
            '<rect width="1280" height="710" rx="18" fill="#fff"/>']
        self.parts=[s.replace('id="arrow"',f'id="{key}-arrow"').replace('id="grad"',f'id="{key}-grad"')
                    .replace(';marker-end:url(#arrow)','').replace(';marker-end:url(#grad)','')for s in self.parts]
        self.text(35, 38, NAMES[key], 'h')
        self.text(35, 68, '实线：前向数据流　蓝/绿：部署保留　橙：训练专用　紫：共享权重　粉虚线：梯度回传', 'note')

    def text(self, x, y, value, cls='s', anchor=None):
        extra = f' text-anchor="{anchor}"' if anchor else ''
        self.parts.append(f'<text x="{x}" y="{y}" class="{cls}"{extra}>{escape(value)}</text>')

    def box(self, x, y, w, h, lines, kind='online'):
        fill, stroke = COLORS[kind]
        self.parts.append(f'<g class="node {kind}"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="{stroke}" stroke-width="1.7"/>')
        start = y + h/2 - (len(lines)-1)*11 + 5
        for i, line in enumerate(lines):
            self.text(x+w/2, start+i*22, line, 'h' if i == 0 else 's', 'middle')
        self.parts.append('</g>')

    def path(self, points, kind='', label=None):
        marker=self.key+('-grad'if kind=='back'else'-arrow')
        self.parts.append(f'<path d="{points}" class="edge {kind}" marker-end="url(#{marker})"/>')
        if label:
            self.text(*label, cls='note')

    def finish(self, note):
        self.parts.append('<line x1="35" y1="622" x2="1245" y2="622" stroke="#dce3ef"/>')
        for i, line in enumerate(note):
            self.text(35, 650+24*i, line, 'note')
        return ''.join(self.parts) + '</svg>'


def architecture_svg(key):
    d = Drawing(key)
    if key != 'D':
        d.box(30, 125, 190, 92, ['合法当前输入', 'left / front / right', '导航 + ego'])
        d.box(260, 125, 210, 92, ['Qwen3-VL-2B', '选取层 8 / 16 / 28', 'merger / 文本表征'])
        d.box(510, 125, 210, 92, ['场景 C160', '144 空间 + 16 全局', 'width 1024'])
        d.path('M220 171 H260'); d.path('M470 171 H510')
    if key == 'WORLD':
        d.box(770, 125, 215, 92, ['Predictive P32', '2 层 / 内宽 512', '仅观察条件的未来表征'])
        d.box(760, 285, 240, 102, ['独立 clean-x actor', '20 × 1024 / 16 heads', '读取 C160 + P32'])
        d.box(1040, 290, 205, 92, ['单轨输出', '8 × [x, y, heading]', '10 步 shift5 ODE'], 'out')
        d.path('M720 171 H770'); d.path('M615 217 V336 H760'); d.path('M875 217 V285'); d.path('M1000 336 H1040')
        d.box(285, 300, 235, 80, ['当前辅助读出', 'DINO / 当前对象'], 'train')
        d.box(1020, 125, 230, 92, ['未来 WA 辅助读出', '真实未来仅作标签', '不作为 actor 输入'], 'train')
        d.path('M565 217 V255 H402 V300', 'aux'); d.path('M985 171 H1020', 'aux')
        d.box(30, 475, 285, 92, ['精确训练候选', 'GT / 已评分候选轨迹', '不是部署时的多候选筛选'], 'train')
        d.box(365, 475, 300, 92, ['同一 actor clean replay', 'sigma = 0 → candidate hidden', '复用上方 actor 权重'], 'train')
        d.box(760, 475, 330, 92, ['动作后六小分 readout', 'NC / DAC / TTC / EP / Comfort / DDC', '训练后果表示，不输出修复量'], 'train')
        d.path('M315 521 H365', 'aux'); d.path('M665 521 H760', 'aux')
        d.path('M920 475 V429 H880 V387', 'back', (945, 431, '后置梯度到 actor / memory / Qwen'))
        return d.finish(['部署保留 predictive32；删除四类辅助 readout、teacher、候选库和 scorer。',
            'post loss 经 actor hidden 回传，不要求经过最后 action_output；候选不是部署动作的 GT 选轨。'])
    if key == 'A':
        d.box(805, 125, 245, 92, ['独立 actor A8', '20 × 1024 / 16 heads', '只读取 C160'])
        d.box(1080, 125, 170, 92, ['8 × 3 轨迹', 'K10 / shift5'], 'out')
        d.path('M720 171 H805'); d.path('M1050 171 H1080')
        d.box(335, 315, 280, 100, ['训练专用 WorldStream', 'W32 · 20 × 384 / 6 heads', '预任务：DINO / WA / tracks'], 'train')
        d.box(745, 315, 305, 100, ['训练专用 R16', '与 W 复用 WorldStream 权重', '读取逐层 candidate actor hidden'], 'train')
        d.path('M565 217 V270 H475 V315', 'aux'); d.path('M615 365 H745', 'aux')
        d.path('M925 217 V315', 'aux', (954, 275, '同一 actor 的精确 clean replay'))
        d.box(745, 490, 305, 78, ['后置辅助任务', '事实 WA / 分项 / 排序 / 违规时序'], 'train')
        d.path('M895 415 V490', 'aux'); d.path('M1050 527 H1155 V250 H1000 V217', 'back')
        d.text(80, 460, '关键：没有 W → actor 的前向连接', 'h')
        d.text(80, 490, '世界任务通过训练梯度改善部署表示。', 'note')
        return d.finish(['部署：Qwen + C160 + actor；世界流、R、readout 整体删除。',
            '后置梯度进入 actor 共享块和 Qwen/C；W32 只在训练时提供监督，不能偷接旧 predictive32。'])
    if key == 'B':
        d.parts.append('<rect x="180" y="278" width="1010" height="295" rx="16" fill="#faf8ff" stroke="#7859b5" stroke-width="2"/>')
        d.text(210, 311, '同一个 20 × 1024 决策主干：Q/K/V/O、FFN、主要调制共享', 'h')
        d.box(220, 345, 235, 104, ['P32：动作前', '固定时间 / 仅 C + P', '每场景一次 prefix'], 'shared')
        d.box(560, 345, 235, 104, ['A8：动作生成', '读取 C / P / A', 'sigma 条件，10 次去噪'], 'shared')
        d.box(900, 345, 245, 104, ['R16：动作后', '读取 C / P / 当前候选 A / R', 'sigma=0，训练专用'], 'train')
        d.path('M615 217 V252 H337 V345'); d.path('M615 252 H677 V345'); d.path('M677 252 H1022 V345')
        d.path('M455 397 H560'); d.path('M795 397 H900')
        d.path('M337 449 V486 H1022 V449', 'aux')
        d.box(545, 500, 270, 60, ['连续 action_output → 8 × 3'], 'out'); d.path('M677 449 V500')
        d.text(890, 529, 'R → P/A：禁止', 'h')
        return d.finish(['P/A/R 不是三套 Transformer；方向性 mask 使删除 R 不改变动作前向。',
            'RL 的共享 LoRA 会改变 P，故每次 inner 当前前缀必须带梯度重建；部署保留 P + 共享 actor。'])
    if key == 'C':
        d.box(830, 125, 255, 92, ['独立 actor A8', '20 × 1024 / 16 heads', 'scene attention 保留'])
        d.box(1110, 125, 145, 92, ['8 × 3 轨迹', 'K10 shift5'], 'out')
        d.path('M720 171 H830'); d.path('M1085 171 H1110')
        d.box(320, 330, 270, 104, ['在线世界 W32', '20 × 384 / 6 heads', '仅读取当前 C160'])
        d.box(650, 330, 270, 104, ['逐层 world → actor K/V', '20 层一一对应；4 组桥接', '复用 actor 的 Q / O'])
        d.path('M565 217 V275 H455 V330'); d.path('M590 382 H650')
        d.path('M920 382 H975 V217', '', (960, 303, '每层主动读取'))
        d.text(40, 290, '两路独立 softmax：Attn(C) + tanh(g) × Attn(W)', 'h')
        d.box(650, 490, 270, 80, ['训练专用 R16', '复用世界流 + actor hidden'], 'train')
        d.box(975, 490, 280, 80, ['后置辅助读出', '事实未来 / 候选分数与事件'], 'train')
        d.path('M455 434 V530 H650', 'aux'); d.path('M1050 217 V460 H785 V490', 'aux')
        d.path('M920 530 H975', 'aux'); d.path('M1115 570 V596 H945 V434', 'back')
        return d.finish(['部署保留 W32 和 world→actor 桥接；只删除 R 专用 query、actor→R 桥接与 readout。',
            'W 不读候选动作或 flow sigma；R 的动作后信息不会流回同次 W / actor 前向。'])
    if key == 'D':
        d.box(30, 122, 260, 90, ['当前三视图 / 导航 / ego', '官方 processor / mRoPE', '视觉 merger + DeepStack'])
        d.parts.append('<rect x="330" y="110" width="910" height="430" rx="16" fill="#faf8ff" stroke="#7859b5" stroke-width="2"/>')
        d.text(360, 143, '同一个原生 Qwen 文本主干：28 层 × 2048，GQA / RMSNorm / mRoPE 保留', 'h')
        d.box(365, 180, 235, 90, ['当前 C prefill', 'DeepStack 只注入图像位置', '保持当前输入关系'], 'shared')
        d.box(665, 180, 220, 90, ['P32 prefill', '读取 C / P', '动作无关 prefix'], 'shared')
        d.box(665, 340, 220, 100, ['A8 连续动作', '读取 C / P / A', '块内双向，非逐点自回归'], 'shared')
        d.box(955, 340, 245, 100, ['R16：训练专用', '读取 C / P / 候选 A / R', '共享同一语言层'], 'train')
        d.path('M290 167 H315 V225 H365'); d.path('M600 225 H665'); d.path('M775 270 V340'); d.path('M885 390 H955')
        d.box(30, 350, 260, 90, ['连续动作 + 时间接口', 'action_input 3 → 2048', 'time 256 → 逐层 FiLM'])
        d.path('M290 395 H665', '', (330, 378, 'FiLM 只作用 A；不调制 C / P'))
        d.box(665, 560, 235, 55, ['action_output 2048 → 3'], 'out'); d.path('M775 440 V560')
        d.text(365, 495, '每步从相同 C+P cache 开始；不累积上一步 A 的 KV。', 'note')
        d.box(975, 560, 245, 55, ['后置任务 readout'], 'train'); d.path('M1077 440 V560', 'aux')
        return d.finish(['没有独立 500M actor；动作去噪实际使用 Qwen 语言层，不能把“少一个模块”解释为计算更小。',
            'RL 文本 attention LoRA 也改变 C/P；每个 inner 重建当前文本前缀，冻结视觉/merger 可以缓存。'])
    raise ValueError(key)


COMPARISON = [
 ('WORLD', 'Qwen → C160 + predictive32 → 独立 actor', 'predictive32 与 actor 分离', '候选 hidden → 六小分头', 'Qwen + C/P + actor'),
 ('A', 'C160 → actor；世界支路仅训练', '世界能力经共享上游梯度内化', 'world R 读取 actor hidden', 'Qwen + C160 + actor'),
 ('B', 'C160 → 同一主干中的 P32 / A8', 'P/A/R 共享决策层参数', '共享 R 读取同一候选 A', 'Qwen + C160 + P + actor'),
 ('C', 'C160 → W32 → actor 逐层 K/V', '独立世界流显式参与每层规划', 'R 复用世界层并读 actor hidden', 'Qwen + C160 + W + actor'),
 ('D', '原生 Qwen C/P/A 连续动作', '动作推理直接复用语言层', 'R 与动作共用 Qwen 语言层', 'Qwen + P + 连续动作/时间接口'),
]
