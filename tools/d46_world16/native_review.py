"""Full16 rank native validation before unchanged compact export/evaluation."""
import argparse,json
from pathlib import Path
import torch
from starVLA.rl.d44_native_rl.deployment import inspect
from starVLA.rl.d46_world16.native import World16Boundary
from starVLA.rl.d46_world16.runtime import receipt
from starVLA.rl.qwen_world_compare.contracts import file_sha
from tools.d44_native_rl.export import export


def review(cp):
    cp=Path(cp).resolve();identity,seal,files=inspect(cp);receipt(cp.parent.parent,identity)
    if seal['world']!=16 or identity['world']!=16:raise ValueError('Not typed world16')
    for name,sha in files.items():
        if file_sha(cp/name)!=sha:raise ValueError('Changed native file '+name)
    common=torch.load(cp/'optimizer.pt',map_location='cpu',weights_only=False);rows=[]
    tx=json.loads((cp/'transaction.json').read_text())
    for rank in range(16):
        own=torch.load(cp/f'rank{rank}.pt',map_location='cpu',weights_only=False)
        if own['identity_sha256']!=identity['sha256']:raise ValueError('Rank identity')
        row=World16Boundary.validate_boundary(own['state'],own['rollout'],identity,common['optimizer'],common['scheduler'])
        if row!=tx['ranks'][rank]:raise ValueError('Rank transaction differs')
        rows.append(row)
    World16Boundary.validate_ranks(rows)
    return dict(schema='d46_world16_native_export_validation_v1',status='PASS',checkpoint=str(cp),files_sha256=seal['files_sha256'],
        identity_sha256=identity['sha256'],world=16,updates=seal['updates'],next_inner=seal['next_inner'],all16rank_material_RNG_checked=True,
        model_math='unchanged official tools.d44_native_rl.evaluate.load_policy / native compact exporter',native_world_transition=identity['native_world_transition'])


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--review-only',action='store_true');a=p.parse_args()
    value=review(a.checkpoint)
    if not a.review_only:export(a.checkpoint,a.output)
    dest=Path(a.output+'.world16_native_validation.json')
    if dest.exists():raise FileExistsError('Keep earlier native validation')
    dest.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value,indent=2))


if __name__=='__main__':main()
