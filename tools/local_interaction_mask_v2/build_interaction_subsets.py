"""Freeze current-only planning analysis groups before opening any planning scores."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import torch
from starVLA.model.modules.joint_world.local_cache import load_payload
from starVLA.model.modules.joint_world.local_graph import LocalInteractionGraph


RULES = {
    'interaction_proxy': 'At least two active non-ego nodes and at least one permitted non-self neighbor-to-neighbor edge',
    'ego_only': 'No active non-ego trajectory node; scene and uncertain-risk memory are retained',
    'risk_context_present': 'At least one context-only B entity; not a claim of true visible obstacle or free space',
}


def classify(graph):
    graph.validate()
    active=graph.active_actor_mask[:,1:]
    pairs=graph.edge_mask[:,1:,1:] & active[:,:,None] & active[:,None,:]
    pairs=pairs & ~torch.eye(active.shape[1],dtype=torch.bool,device=active.device)[None]
    return {'interaction_proxy':(active.sum(-1)>=2)&pairs.flatten(1).any(-1),
            'ego_only':active.sum(-1)==0,'risk_context_present':graph.context_only_mask.any(-1)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('cache','index','output'):parser.add_argument('--'+key,required=True)
    args=parser.parse_args();root=Path(args.cache);path=Path(args.output)
    if path.exists():raise FileExistsError('Keep the original frozen subset registry')
    manifest=json.loads((root/'manifest.json').read_text());index=json.loads(Path(args.index).read_text())
    logs={row['token']:row['log'] for row in index}
    if len(logs)!=len(index) or set(logs)!={row['token'] for row in manifest['records']}:
        raise ValueError('Subset index and full current population differ')
    if not manifest['complete'] or manifest['failed']:raise ValueError('Subset generation requires a complete cache')
    rows=[]
    for record in manifest['records']:
        payload=load_payload(root,record,manifest);flags=classify(LocalInteractionGraph(**payload['local_graph']))
        rows.append(dict(token=record['token'],log=logs[record['token']],**{name:bool(value[0]) for name,value in flags.items()}))
    report={'code_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'current_identity':manifest['identity_sha256'],
        'current_manifest_sha256':hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),
        'index_sha256':hashlib.sha256(Path(args.index).read_bytes()).hexdigest(),
        'rules':RULES,'scenes':len(rows),'logs':len(set(logs.values())),
        'counts':{name:sum(row[name] for row in rows) for name in RULES},
        'labels_or_planning_scores_opened':False,
        'interpretation':'Preregistered current-prediction geometry proxies; overlapping analysis groups never replace the full benchmark denominator',
        'rows':rows}
    path.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
