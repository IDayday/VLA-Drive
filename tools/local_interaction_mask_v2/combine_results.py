"""One full-population CSV for the four locked planning controls and official submetrics."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
from tools.local_interaction_mask_v2.compare_pdms import METRICS, read, verified_pair

VARIANTS=('A0','CURRENT_MEMORY','P_LOCAL_ALL','P_LOCAL_MASK')


def combine(groups):
    if set(groups)!=set(VARIANTS):raise ValueError('Need all four planning controls')
    tokens=set(groups['A0'])
    if not tokens or any(set(group)!=tokens for group in groups.values()):
        raise ValueError('Full scene populations differ')
    rows=[]
    for token in sorted(tokens):
        log=groups['A0'][token]['log']
        if any(group[token]['log']!=log for group in groups.values()):raise ValueError('Log identity differs')
        row={'token':token,'log':log}
        for name in VARIANTS:
            source=groups[name][token];row[name+'_status']=source['status'];row[name+'_error']=source.get('error','')
            for metric in METRICS:
                row[name+'_'+('PDMS' if metric=='score' else metric)]=(float(source[metric]) if source['status']=='ok' else 0.)
        for baseline in ('A0','CURRENT_MEMORY','P_LOCAL_ALL'):
            row['P_LOCAL_MASK_minus_'+baseline+'_percentage_points']=100*(row['P_LOCAL_MASK_PDMS']-row[baseline+'_PDMS'])
        rows.append(row)
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('merged-root','lock','subsets','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--navtest',action='store_true');a=p.parse_args();root=Path(a.merged_root)
    lock=json.loads(Path(a.lock).read_text());registry=json.loads(Path(a.subsets).read_text())
    groups={};sources={}
    for name in VARIANTS:
        path=root/name/'scenes.csv';export=verified_pair(path,root/'A0/scenes.csv')
        if export['variant']!=name or export['bridge_sha256']!=lock['bridge_sha256'][name]:
            raise ValueError('Scored and locked planning checkpoints differ')
        if export['foundation_sha256']!=lock['foundation_sha256'] or export['code_sha']!=lock['evaluation_source_sha']:
            raise ValueError('Scored foundation/source differs from model lock')
        if export['current_identity']!=registry['current_identity'] or export['current_manifest_sha256']!=registry['current_manifest_sha256']:
            raise ValueError('Analysis groups use a different current observation cache')
        groups[name]=read(path);sources[name]=hashlib.sha256(path.read_bytes()).hexdigest()
        if any(row['variant']!=name for row in groups[name].values()):raise ValueError('Mixed scored variants')
    rows=combine(groups);logs={row['log'] for row in rows}
    if a.navtest and (len(rows)!=12146 or len(logs)!=136):raise ValueError('Incomplete full Navtest endpoint')
    flags={row['token']:row for row in registry['rows']}
    if len(flags)!=len(registry['rows']) or set(flags)!=set(groups['A0']):raise ValueError('Subset population differs')
    for row in rows:
        if flags[row['token']]['log']!=row['log']:raise ValueError('Subset log identity differs')
        row.update({key:flags[row['token']][key] for key in registry['rules']})
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    with (out/'scenes.csv').open('w') as handle:
        writer=csv.DictWriter(handle,list(rows[0]));writer.writeheader();writer.writerows(rows)
    failed={name:sum(row[name+'_status']!='ok' for row in rows) for name in VARIANTS}
    report={'code_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'scenes':len(rows),'logs':len(logs),'failed':failed,'valid':not any(failed.values()),
        'PDMS_percent':{name:100*sum(row[name+'_PDMS'] for row in rows)/len(rows) for name in VARIANTS},
        'score_and_submetric_units':'0..1; only explicitly named paired delta columns are percentage points',
        'protocol':'official NAVSIM v1 PDMS, one candidate; no EPDMS/Hard/oracle claim',
        'input_CSV_sha256':sources,'model_lock_sha256':hashlib.sha256(Path(a.lock).read_bytes()).hexdigest(),
        'combined_CSV_sha256':hashlib.sha256((out/'scenes.csv').read_bytes()).hexdigest(),
        'failure_policy':'Retain every requested scene; failures remain zero and invalidate a benchmark claim',
        'analysis_groups':'Frozen current-only proxies; full population remains primary'}
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
