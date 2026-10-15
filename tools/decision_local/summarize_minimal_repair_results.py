"""Common endpoints and paired contrasts; no checkpoint or trajectory selection."""
import argparse,json,sys,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.decision_local.common import read,atomic_json,digest
from tools.local_interaction_mask_v2.compare_pdms import read as read_scores,compare,METRICS


def evaluator_contract(meta):
    return {key:meta[key] for key in ('index_sha256','navsim_python_tree_sha256','nuplan_python_tree_sha256','runtime_versions','protocol')}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--registries',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--require-complete',action='store_true');a=p.parse_args()
    tables=[];missing=[];banks={};identities={};paired={};manifest=[]
    for path in a.registries:
        reg=read(path);distribution=reg['distribution'];manifest.append({'path':str(path),'sha256':digest(path)})
        for job in reg['jobs']:
            score=Path(job['scores']);done=score/'COMPLETE.json'
            if not done.exists():missing.append({'distribution':distribution,'id':job['id'],'status':'NOT_COMPLETE'});continue
            result=read(done);meta=read(score/'identity.json');contract=evaluator_contract(meta)
            training=read(Path(job['checkpoint']).parent/'identity.json')
            if training['kind']!=job['kind'] or training['seed']!=job['seed']:raise ValueError('Registry/checkpoint kind or seed mismatch')
            if distribution in identities and identities[distribution]!=contract:raise ValueError('Evaluator contracts changed within comparison')
            identities[distribution]=contract
            if not all(result[v]['valid'] for v in ('q0','q_final')):raise ValueError('Failed scenes invalidate the benchmark; do not drop them')
            orig,final=[read_scores(score/v/'scenes.csv') for v in ('q0','q_final')]
            if training['scope']=='repair_formal' and len(final)!=({'dev':1696,'navtest':12146}[reg['population']]):raise ValueError('Formal population incomplete')
            baseline,rows=compare(final,orig)
            row={'distribution':distribution,'seed':job['seed'],'kind':job['kind'],'epoch':job['epoch'],
                 'scenes':len(final),'input_PDMS':result['q0']['metrics']['score']*100,
                 'training_scope':training['scope'],'training_fold':training.get('repair_fold'),
                 'PDMS':result['q_final']['metrics']['score']*100,'delta_input_pp':baseline['delta_percentage_points'],
                 'zero_to_nonzero':sum(r['baseline_score']==0 and r['first_score']>0 for r in rows),
                 'nonzero_to_zero':sum(r['baseline_score']>0 and r['first_score']==0 for r in rows),
                 **{k:result['q_final']['metrics'][k] for k in METRICS if k!='score'},
                 'information_mode':'camera_only' if job['kind'].endswith('_NONE') or job['kind'].startswith('R_MIN_') else 'privileged_probe',
                 'score_identity':meta['identity']}
            tables.append(row)
            if job['epoch']==24:
                banks[(distribution,job['seed'],job['kind'])]=final
                paired[f'{distribution}/seed{job["seed"]}/{job["kind"]}-input']=baseline
    if a.require_complete and missing:raise ValueError(f'{len(missing)} fixed observations not complete')
    comparisons=[('MIN_FUT','MIN_CV'),('LOG_FUT','LOG_CV'),('MIN_NONE','LOG_NONE'),
                 ('MIN_FUT','MIN_NONE'),('MIN_FUT','LOG_FUT'),('R_MIN_ABS','R_MIN_CURRENT'),
                 ('R_MIN_DIFF','R_MIN_ABS'),('R_MIN_DIFF','MIN_NONE')]
    for distribution,seed in sorted({key[:2] for key in banks}):
        group={key[2]:value for key,value in banks.items() if key[:2]==(distribution,seed)}
        for first,base in comparisons:
            if first in group and base in group:
                paired[f'{distribution}/seed{seed}/{first}-{base}']=compare(group[first],group[base])[0]
        if all(k in group for k in ('MIN_FUT','MIN_CV','LOG_FUT','LOG_CV')):
            # The reuse of the original log-cluster operator is linear here.
            # Signed per-scene contrasts are not presented as executable scores.
            def contrast(first,second):
                if set(first)!=set(second):raise ValueError('Unmatched contrast')
                return {t:{'token':t,'log':row['log'],'status':'ok',
                    **{m:float(row[m])-float(second[t][m]) for m in METRICS}} for t,row in first.items()}
            x=contrast(group['MIN_FUT'],group['MIN_CV']);y=contrast(group['LOG_FUT'],group['LOG_CV'])
            result=compare(x,y)[0]
            paired[f'{distribution}/seed{seed}/future_information_difference_in_differences']={
                'scenes':result['scenes'],'logs':result['logs'],'metrics':result['metrics'],
                'scope':'(MIN_FUT-MIN_CV)-(LOG_FUT-LOG_CV); signed contrast, not a policy score'}
    a.output.mkdir(parents=True,exist_ok=True)
    atomic_json(a.output/'SUMMARY.json',{'registries':manifest,'tables':tables,'missing':missing,'paired':paired,
        'common_endpoint':24,'model_selection':'none; fixed6/12/18/24 observations retained',
        'limitations':['One shared S0 training seed; new-head seeds are not independent full VLA training',
                       'Development diagnostics and synthetic recovery are separate from natural deployment',
                       'Privileged CV/FUT are not camera-only results; unknown scenes remain in canonical denominator',
                       'Cluster intervals describe fixed trained models, not proof of equivalence or long-term safety']})
    if tables:
        with (a.output/'learning_curves.csv').open('w') as f:
            writer=csv.DictWriter(f,list(tables[0]));writer.writeheader();writer.writerows(tables)
    text=['# 局部保持与必要修正：共同终点','',f'已完成{len(tables)}项固定观察，尚缺{len(missing)}项。以下均为24epoch；不选事后最佳权重。','',
          '| 输入分布 | 头seed | 组 | 训练范围 | 信息模式 | 输入PDMS | 修正PDMS | 差值pp | 零分修复 | 新增零分 |',
          '|---|---:|---|---|---|---:|---:|---:|---:|---:|']
    for row in sorted((x for x in tables if x['epoch']==24),key=lambda x:(x['distribution'],x['seed'],x['kind'])):
        text.append(f'| {row["distribution"]} | {row["seed"]} | {row["kind"]} | {row["training_scope"]} | {row["information_mode"]} | {row["input_PDMS"]:.5f} | {row["PDMS"]:.5f} | {row["delta_input_pp"]:+.5f} | {row["zero_to_nonzero"]} | {row["nonzero_to_zero"]} |')
    text+=['','## 预登记配对','']
    for name,value in paired.items():
        if name.endswith('-input'):continue
        score=value['metrics']['score'];interval=score['log_cluster_95_interval']
        ci='无可用区间' if interval is None else f'[{100*interval[0]:+.5f}, {100*interval[1]:+.5f}]'
        text.append(f'- {name}：{100*score["paired_delta"]:+.5f} pp，log聚类95%区间{ci}。')
    text+=['','区间跨零表示当前证据不确定，不能据此宣布等效或方向无效。正式判断还需结合保持/修复、进度、安全分项、标签支持与新失败分析。']
    (a.output/'COMMON_ENDPOINTS_ZH.md').write_text('\n'.join(text)+'\n')
    print(json.dumps({'observations':len(tables),'pending':len(missing),'common_endpoint_models':sum(x['epoch']==24 for x in tables)}))


if __name__=='__main__':main()
