"""Read-only classification of legacy checkpoints; COMPLETE alone is insufficient."""
import argparse
import json
from pathlib import Path
import torch
from starVLA.rl.simwam_recipe.contracts import file_sha
from starVLA.rl.simwam_recipe.recovery_state import validate_boundary,validate_ranks
from starVLA.rl.simwam_recipe.data_stream import SceneStream
from starVLA.rl.simwam_recipe.invocation import validate_rows
from starVLA.rl.flow_grpo.transactions import atomic_json


def inspect(path):
    path=Path(path)
    try:
        if '.incomplete' in path.name:raise ValueError('Unpublished attempt')
        if 'failure' in path.name.lower():raise ValueError('Historical failure tag: RNG/collection start not committed; not an automatic recovery point')
        seal=json.loads((path/'COMPLETE').read_text());files=json.loads((path/'files.json').read_text())
        if file_sha(path/'files.json')!=seal['files_sha256']:raise ValueError('Changed file manifest')
        for name,sha in files.items():
            if file_sha(path/name)!=sha:raise ValueError('Changed payload: '+name)
        identity=json.loads((path/'identity.json').read_text())
        common=torch.load(path/'optimizer.pt',map_location='cpu',weights_only=False)
        ranks=[]
        for rank in range(identity['world']):
            own=torch.load(path/f'rank{rank}.pt',map_location='cpu',weights_only=False)
            if own['identity_sha256']!=identity['sha256']:raise ValueError('Rank identity mismatch')
            ranks.append(validate_boundary(own['state'],own['rollout'],identity,common['optimizer'],common['scheduler'],legacy=seal.get('schema')!=2))
        validate_ranks(ranks)
        return dict(status='COMMITTED_BOUNDARY',checkpoint=str(path),files_sha256=file_sha(path/'files.json'),
            legacy=seal.get('schema')!=2,requires_explicit_migration=seal.get('schema')!=2,
            ranks=ranks,optimizer_parameter_states=len(common['optimizer']['state']),
            scheduler_last_epoch=common['scheduler']['last_epoch'],automatic_legacy_resume=False)
    except Exception as exc:
        return dict(status='REJECTED',checkpoint=str(path),reason=str(exc),automatic_resume=False)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run');p.add_argument('--checkpoint');p.add_argument('--output',required=True);a=p.parse_args()
    if not (bool(a.run)^bool(a.checkpoint)):raise ValueError('Select one run or checkpoint')
    if a.checkpoint:result=inspect(a.checkpoint)
    else:
        run=Path(a.run);identity=json.loads((run/'identity.json').read_text())
        latest=json.loads((run/'latest.json').read_text());endpoint=inspect(latest['checkpoint'])
        rows=[[json.loads(l) for l in (run/f'metrics_rank{r}.jsonl').read_text().splitlines()] for r in range(identity['world'])]
        states=[json.loads((run/f'result_rank{r}.json').read_text()) for r in range(identity['world'])]
        for r in rows:validate_rows(r,len(r))
        if endpoint['status']!='COMMITTED_BOUNDARY' or any(len(r)!=20000 for r in rows):raise ValueError('Formal endpoint incomplete')
        if any(s['status']!='COMPLETE' or s['updates']!=20000 or s['collections']!=5000 or s['raw_scene_exposures']!=40000 or s['new_candidates']!=320000 for s in states):
            raise ValueError('Endpoint budget mismatch')
        stream=SceneStream('/mnt/project/simwam-recipe-20261015/hard_manifest.json',8)
        padding=[]
        for collection in range(5000):
            for rank in range(8):
                token,position=stream.at(collection*8,rank)
                if rows[rank][collection*4]['token']!=token:raise ValueError('Actual scene stream changed')
                if position['padding']:padding.append(dict(collection=collection+1,rank=rank,token=token,position=position))
        suspicious=[inspect(p) for p in (run/'checkpoints').iterdir() if p.is_dir() and ('failure' in p.name or '.incomplete' in p.name)]
        result=dict(status='PASS',run=str(run),identity=identity['sha256'],endpoint=endpoint,
            updates_by_rank=[len(r) for r in rows],fresh_global_rollouts=5000,scene_exposures=40000,new_candidates=320000,
            epoch_padding_exposures=len(padding),padding_records=padding,
            suspicious_checkpoint_classification=suspicious,
            progress_status=json.loads((run/'progress.json').read_text())['status'],
            actual_status='COMPLETE from all eight result files, sealed optimizer and contiguous journals',
            source_files=identity['source_files'])
    output=Path(a.output)
    if output.exists():raise FileExistsError('Read-only audit sidecars are immutable')
    atomic_json(output,result);print(json.dumps(dict(status=result['status'],output=str(output))))


if __name__=='__main__':main()
