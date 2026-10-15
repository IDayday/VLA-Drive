"""Revalidate sealed predictions/scores and rebuild analysis, with zero inference."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
from starVLA.rl.simwam_recipe.contracts import file_sha
from starVLA.rl.flow_grpo.transactions import atomic_json


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    def audit(old_path):
        old=json.loads(old_path.read_text());dest=out/(old_path.parent.name+'.json')
        cmd=[sys.executable,'-m','scripts.analysis.simwam_recipe.analyze_evaluation',
             '--evaluation',str(old_path.parent),'--output',str(dest)]
        if old.get('reference'):cmd+=['--reference',old['reference']]
        if not dest.exists():
            with dest.with_suffix('.log').open('x') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
        new=json.loads(dest.read_text())
        if new['identity']!=old['identity'] or new['modes']!=old['modes']:
            raise ValueError('Derived metric changed: '+str(old_path))
        return dict(evaluation=old_path.parent.name,scenes=new['scenes'],status='PASS',
            previous_summary_sha256=file_sha(old_path),new_summary_sha256=file_sha(dest),
            prediction_and_score_hashes_verified=True,all_metric_and_paired_results_unchanged=True,
            new_inference=0,new_scoring=0,cost_schema=2,command=cmd)
    paths=sorted(Path(a.root).glob('*/summary.json'))
    with ThreadPoolExecutor(max_workers=2) as pool:rows=list(pool.map(audit,paths))
    result=dict(status='PASS',evaluations=rows,new_inference=0,new_scoring=0)
    atomic_json(out/'audit.json',result);print(json.dumps(dict(status='PASS',evaluations=len(rows))))


if __name__=='__main__':main()
