"""Preregistered paired schedule decisions; no planning test score is accepted."""
import argparse
import json
from pathlib import Path


def graph_extension(runs):
    """Compare both arms on the same fixed graph and supervised population."""
    manifests=[json.loads((Path(p)/'manifest.json').read_text()) for p in runs]
    if {m['arguments']['mode'] for m in manifests}!={'all','mask'}:raise ValueError('Need exactly ALL and MASK')
    for key in ('cache_identity','labels','holdout_identity','holdout_labels','config'):
        if manifests[0][key]!=manifests[1][key]:raise ValueError('Unmatched paired experiment: '+key)
    for key in ('batch','seed','schedule_epochs'):
        if manifests[0]['arguments'][key]!=manifests[1]['arguments'][key]:raise ValueError('Unmatched training budget/config: '+key)
    reasons=[];missing=[];curves=[]
    for run in runs:
        old=json.loads((Path(run)/'holdout_12.json').read_text());new=json.loads((Path(run)/'holdout_16.json').read_text())
        components=('deploy_ego_ADE_m','deploy_agents_ADE_m')
        if any(old.get(k) is None or new.get(k) is None for k in components):
            missing.append(str(run));continue
        before=sum(old[k] for k in components)/2;after=sum(new[k] for k in components)/2
        relative=abs(after-before)/max(abs(before),1e-8)
        worse=[actor for actor in ('ego','agents') if new['deploy_'+actor+'_ADE_m']>new['deploy_'+actor+'_stationary_ADE_m']]
        curves.append({'run':str(run),'relative_change_12_to_16':relative,'worse_than_stationary':worse})
        if relative>.02 or worse:reasons.append(str(run))
    return {'decision':'REVIEW_MISSING_COVERAGE' if missing else ('EXTEND_BOTH_TO_32' if reasons else 'STOP_BOTH_AT_16'),
        'curves':curves,'missing_neighborhood_metrics':missing,'triggered_runs':reasons,'uses_PDMS':False,
        'note':'A stable curve alone does not establish scientific sufficiency or planning improvement.'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',nargs=2,required=True);p.add_argument('--output',required=True);a=p.parse_args()
    result=graph_extension(a.runs);path=Path(a.output)
    if path.exists():raise FileExistsError('Keep prior preregistered decision')
    path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
