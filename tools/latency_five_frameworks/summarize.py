"""Summarize completed native deployment measurements, without rerunning GPU."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--input',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args();root=Path(args.input);out=Path(args.output)
    raw=root/'RESULTS.json';seal=json.loads((root/'COMPLETE').read_text())
    if hashlib.sha256(raw.read_bytes()).hexdigest()!=seal['results_sha256']:
        raise ValueError('Changed benchmark result seal')
    v=json.loads(raw.read_text());rows=v['results']
    if [r['name']for r in rows]!=['Qwen-World','A','B','C','D']:
        raise ValueError('Incomplete or reordered five-framework comparison')
    expected=[(r['token'],r['repeat'])for r in rows[0]['rows']]
    for r in rows:
        if [(q['token'],q['repeat'])for q in r['rows']]!=expected:
            raise ValueError('Different inference workload')
        if any(q['normalized_action_max_abs_error']!=0 or q['physical_trajectory_max_abs_error']!=0 for q in r['phases']):
            raise ValueError('Diagnostic path differs from native deployment')
        if any(len(q['owner_rows'])!=1 for q in r['hardware']):
            raise ValueError('Benchmark GPU was shared')
    out.mkdir(parents=True,exist_ok=True)
    with (out/'latency_summary.csv').open('w')as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['framework','IL_update','model_mean_ms','model_p50_ms','model_p95_ms',
                                  'rgb_p50_ms','rgb_p95_ms','encode_mean_ms','prefix_mean_ms','K10_mean_ms',
                                  'resident_parameters','peak_allocated_GiB','loading_seconds'])
        for r in rows:
            a=r['summary']['model_ms'];b=r['summary']['rgb_to_trajectory_ms'];q=r['phase_summary']
            w.writerow([r['name'],r['IL_updates'],a['mean'],a['p50'],a['p95'],b['p50'],b['p95'],
                        q['encode_current_ms']['mean'],q['prepare_prefix_ms']['mean'],q['K10_ms']['mean'],
                        r['counts']['parameters'],r['peak_allocated_GiB'],r['loading_seconds']])
    lines=['| 框架 | IL 权重 | 模型 P50 / P95 (ms) | RGB→轨迹 P50 / P95 (ms) | K10 去噪均值 (ms) | 峰值显存 (GiB) |',
           '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        a=r['summary']['model_ms'];b=r['summary']['rgb_to_trajectory_ms']
        lines.append(f"| {r['name']} | {r['IL_updates']} | {a['p50']:.2f} / {a['p95']:.2f} | {b['p50']:.2f} / {b['p95']:.2f} | {r['phase_summary']['K10_ms']['mean']:.2f} | {r['peak_allocated_GiB']:.2f} |")
    (out/'LATENCY_TABLE.md').write_text('\n'.join(lines)+'\n')
    width,height=920,370
    maximum=max(r['summary']['rgb_to_trajectory_ms']['p95']for r in rows)*1.06
    svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
         '<rect width="100%" height="100%" fill="white"/>',
         '<g font-family="sans-serif" font-size="14" fill="#172b4d">',
         '<text x="20" y="25">Native B1 · K10 shift5 · same A800 · FP32 parameters / BF16 forward</text>',
         '<text x="150" y="49" fill="#1555a2">Blue: model P50</text><text x="410" y="49" fill="#ba650a">Orange: RGB → trajectory P50; whisker: P95</text>']
    for i,r in enumerate(rows):
        y=78+i*53;svg.append(f'<text x="20" y="{y+14}">{r["name"]}</text>')
        for offset,key,color in ((0,'model_ms','#1555a2'),(20,'rgb_to_trajectory_ms','#ba650a')):
            s=r['summary'][key];x=150;length=s['p50']/maximum*650;tail=x+s['p95']/maximum*650
            svg.append(f'<rect x="{x}" y="{y+offset}" width="{length:.2f}" height="14" fill="{color}"/>')
            svg.append(f'<path d="M{x+length:.2f},{y+offset+7} H{tail:.2f} M{tail:.2f},{y+offset+3} V{y+offset+11}" stroke="{color}" fill="none"/>')
            svg.append(f'<text x="{tail+5:.2f}" y="{y+offset+12}" font-size="11">{s["p50"]:.1f} / {s["p95"]:.1f} ms</text>')
    svg.append('</g></svg>')
    (out/'latency_plot.svg').write_text('\n'.join(svg)+'\n')
    print('\n'.join(lines))


if __name__=='__main__':main()
