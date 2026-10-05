"""Add reviewed case notes to a private gallery; never publish sensor pictures."""
import argparse
import html
import json
from pathlib import Path

from PIL import Image


def annotate(artifacts, notes_path):
    root = Path(artifacts)
    notes = json.loads(Path(notes_path).read_text())
    scenes = json.loads((root / 'representatives.json').read_text())
    config = json.loads((root / 'config.json').read_text())
    gallery = root / 'gallery'
    aliases = [f'scene_{i:02d}' for i in range(len(scenes))]
    if set(aliases) != set(notes['cases']):
        raise ValueError('Notes must cover exactly the selected gallery cases')
    summaries = {
        arm: [json.loads(line) for line in (root / arm / 'scenes.jsonl').read_text().splitlines()]
        for arm in ('S0', 'S3')
    }
    from .summarize_visualization import task_score
    entries = []
    for i, (alias, row) in enumerate(zip(aliases, scenes)):
        if not (gallery / alias / 'index.html').is_file():
            raise ValueError(f'Unrendered case: {alias}')
        record = json.loads((Path(config['dev_data']) / 'current' / (row['token'] + '.json')).read_text())
        with Image.open(record['image_paths'][0]) as source:
            # Thumbnail only; features and measured scores remain unchanged.
            source.convert('RGB').resize((512, 288)).save(gallery / alias / 'current_rgb.jpg', quality=90)
        note = notes['cases'][alias]
        scores = []
        for arm, rows in summaries.items():
            if rows[i]['token'] != row['token'] or rows[i]['failure']:
                raise ValueError('Changed selection/order or failed inference')
            pred = task_score(rows[i], 'future', 'model')
            static = task_score(rows[i], 'future', 'static')
            if pred is None:
                value = '完整未来片段无效，未计算误差'
            else:
                gain = 100 * (1 - pred / static) if static and static > 1e-12 else None
                value = f'三视角未来 MSE {pred:.4f}；相对自身静态参考改善 {gain:+.1f}%' if gain is not None else f'三视角未来 MSE {pred:.4f}'
            scores.append(f'<p><b>{arm}</b>：{html.escape(value)}</p>')
        entries.append(
            f'<article><h2>{alias} · {html.escape(note["title"])}</h2>'
            f'<img class="thumbnail" src="{alias}/current_rgb.jpg" alt="当前真实前视图">'
            f'<p>{html.escape(note["observation"])}</p>' + ''.join(scores) +
            f'<p><a href="{alias}/index.html">全部模型与三视角</a> · '
            f'<a href="{alias}/current_v0.png">当前特征</a> · '
            f'<a href="{alias}/S0_future_v0.png">S0逐帧未来</a> · '
            f'<a href="{alias}/S3_future_v0.png">S3视频未来</a></p></article>'
        )
    body = ('<!doctype html><html lang="zh"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            '<title>10个场景：当前与未来特征学习诊断</title>'
            '<style>body{font:16px/1.7 system-ui,sans-serif;max-width:1150px;margin:28px auto;padding:0 20px;'
            'color:#183044;background:#f4f7fa}article{background:white;padding:20px;margin:20px 0;'
            'border:1px solid #d7e1e9;border-radius:10px}a{color:#075b9a}.thumbnail{max-width:100%;'
            'width:512px}h1{line-height:1.3}li{margin:6px 0}.legend{background:#fff3d3;padding:14px}</style>'
            '<h1>当前与未来辅助任务：10个代表场景</h1>'
            '<p>C1与S0—S4均为100k固定终点。10场景、8个log；6模型零失败、零训练更新。'
            '1个案例缺完整8帧目标，保留案例并明确无效；其余9个完成未来片段比较。</p>'
            '<ul>' + ''.join(f'<li>{html.escape(point)}</li>' for point in notes['findings']) + '</ul>'
            '<p class="legend">彩色特征图使用同一训练集拟合的显示PCA，<b>不是还原的RGB，也不是车辆分割或光流</b>。'
            '热图越亮表示特征误差或偏离静态参考越大。同一目标类型内部比较；DINO与视频MSE不跨教师排名。'
            'S1/S3/S4的GT动作只用于未来辅助头，不进入当前W或规划。这10个案例不代表完整开发集统计结论。</p>'
            '<p><a href="index.html">原始完整图集索引</a></p>' + ''.join(entries) + '</html>')
    (gallery / 'analysis.html').write_text(body)
    print(gallery / 'analysis.html')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--artifacts', required=True)
    parser.add_argument('--notes', required=True)
    args = parser.parse_args()
    annotate(args.artifacts, args.notes)


if __name__ == '__main__':
    main()
