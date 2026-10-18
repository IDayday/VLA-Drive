"""Compare actual native 8-GPU source-GRPO inner recovery, including old chain."""
import argparse
import json
from pathlib import Path
import torch
from starVLA.rl.qwen_sourceflow.contracts import file_sha
from starVLA.rl.flow_grpo.transactions import atomic_json
from scripts.analysis.qwen_sourceflow.compare_il_resume import compare


def sealed(path):
    path=Path(path);seal=json.loads((path/'COMPLETE').read_text())
    files=json.loads((path/'files.json').read_text())
    if '.incomplete' in path.name or seal['schema']!=2 or file_sha(path/'files.json')!=seal['files_sha256']:
        raise ValueError('Not a published native inner boundary')
    for name,sha in files.items():
        if file_sha(path/name)!=sha:raise ValueError('Changed checkpoint: '+name)
    return seal,json.loads((path/'identity.json').read_text())


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--continuous',required=True);p.add_argument('--resumed',required=True)
    p.add_argument('--output',required=True);a=p.parse_args();torch.set_num_threads(2)
    paths=[Path(a.continuous),Path(a.resumed)];infos=[sealed(x) for x in paths]
    fields=('schema','phase','world','updates','next_inner')
    if infos[0][1]!=infos[1][1] or any(infos[0][0][k]!=infos[1][0][k] for k in fields):
        raise ValueError('Different scientific/execution identity or endpoint')
    seal,identity=infos[0]
    if identity['schema']!='native_qwen_sourceflow_RL_v1' or identity['purpose']=='formal':
        raise ValueError('This tool compares qualification runs, never changes formal training')
    report=dict(schema='native_source_GRPO_actual_resume_v1',updates=seal['updates'],
        continuous=str(paths[0]),resumed=str(paths[1]),checks={},timing_excluded=True)
    def check(name,left,right):
        stats=dict(tensors=0,elements=0,differences=[]);compare(left,right,name,stats)
        report['checks'][name]=stats
    common=[torch.load(x/'optimizer.pt',map_location='cpu',weights_only=False) for x in paths]
    check('adapters_optimizer_scheduler',*common);del common
    for rank in range(seal['world']):
        values=[torch.load(x/f'rank{rank}.pt',map_location='cpu',weights_only=False) for x in paths]
        for key in ('rng','rollout','identity_sha256'):
            check(f'rank{rank}/{key}',*(v[key] for v in values))
        # Actual elapsed times differ across process invocations. Every logical
        # state counter, phase, cursor and effective-update count must match.
        check(f'rank{rank}/logical_state',*({k:v for k,v in x['state'].items()
            if not k.startswith('seconds_')} for x in values))
        behavior=[torch.load(x.parent.parent/'behavior/collection000001'/f'rank{rank}.pt',
            map_location='cpu',weights_only=False) for x in paths]
        for key in ('chain','timesteps','deltas','stoch_idx','old_logp','rewards',
                    'old_probability','advantages','adv_mask','chain_identity',
                    'behavior_version','qualification_sampling_check'):
            check(f'rank{rank}/behavior/{key}',*(b[key] for b in behavior))
    report['status']='PASS' if not any(x['differences'] for x in report['checks'].values()) else 'FAIL'
    atomic_json(a.output,report)
    print(json.dumps(dict(status=report['status'],differences={k:len(v['differences'])
        for k,v in report['checks'].items() if v['differences']})))
    if report['status']!='PASS':raise AssertionError('Native source inner recovery mismatch; preserve evidence')


if __name__=='__main__':main()
