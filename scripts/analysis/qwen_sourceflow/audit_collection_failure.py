"""Read-only evidence for a real native failure before advantage commit."""
import argparse
import json
from pathlib import Path
import torch
from starVLA.rl.qwen_sourceflow.contracts import file_sha
from starVLA.rl.flow_grpo.transactions import atomic_json


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run',required=True)
    p.add_argument('--dispatch',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.run);dispatch=Path(a.dispatch);torch.set_num_threads(2)
    identity=json.loads((root/'identity.json').read_text())
    spec=json.loads((dispatch/'spec.json').read_text());result=json.loads((dispatch/'result.json').read_text())
    if identity['purpose']=='formal' or '--fault' not in spec['command'] or result['status']!='FAILED':
        raise ValueError('Expected deliberately faulted native qualification')
    latest=json.loads((root/'latest.json').read_text());checkpoint=Path(latest['checkpoint'])
    seal=json.loads((checkpoint/'COMPLETE').read_text());files=json.loads((checkpoint/'files.json').read_text())
    if latest['updates']!=0 or seal['updates']!=0 or seal['phase']!='READY':
        raise ValueError('Uncommitted collection advanced recovery boundary')
    if file_sha(checkpoint/'files.json')!=seal['files_sha256'] or latest['files_sha256']!=seal['files_sha256']:
        raise ValueError('Recovery seal changed')
    for name,sha in files.items():
        if file_sha(checkpoint/name)!=sha:raise ValueError('Changed legal recovery artifact: '+name)
    for rank in range(identity['world']):
        state=torch.load(checkpoint/f'rank{rank}.pt',map_location='cpu',weights_only=False)
        if state['rollout'] is not None or state['state']['collections']!=0 or state['state']['scene_cursor']!=0:
            raise ValueError('Uncommitted rollout in latest')
    evidence=list((root/'failures').glob('*/*/evidence.json'));ranks=set();rows=[]
    for path in evidence:
        event=json.loads(path.read_text())
        if event['recoverable'] or event['state']['phase']!='COLLECTING' or 'Collection not committed before advantage finalization' not in event['error']:
            raise ValueError('Unexpected failure phase/cause')
        if (path.parent/'COMPLETE').exists():raise ValueError('Failure published as a checkpoint')
        rollout=torch.load(path.parent/'partial_rollout.pt',map_location='cpu',weights_only=False)
        if 'advantages' in rollout or 'adv_mask' in rollout or rollout['chain'].shape!=(8,11,8,3):
            raise ValueError('Unfinalized real chain was modified')
        if len(rollout['records'])!=8 or any(x['status']!='OK' for x in rollout['records']):
            raise ValueError('Expected scored trajectories before injected failure')
        rank=int(path.parent.name.split('-')[0].removeprefix('rank'));ranks.add(rank)
        rows.append(dict(rank=rank,phase=event['state']['phase'],physical_candidates=8,
            saved_full_partial_chain=True,advantage_absent=True,recoverable=False))
    if ranks!=set(range(identity['world'])):raise ValueError('Missing rank failure evidence')
    if list((root/'behavior').glob('*/rank*.pt')):raise ValueError('Uncommitted chain published as behavior')
    record=dict(status='PASS',qualification_only=True,run=str(root),world=identity['world'],
        fault='before_finalize on rank1',last_legal_phase='READY',last_legal_update=0,
        accepted_scenes=0,accepted_candidates=0,physical_scored_candidates=sum(x['physical_candidates'] for x in rows),
        failure_not_recoverable=True,latest_unchanged=True,rank_evidence=rows,
        worker_wall_seconds=result['wall_seconds'],automatic_retry=False)
    atomic_json(a.output,record);print(json.dumps(record))


if __name__=='__main__':main()
