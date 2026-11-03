"""Prepare an evidence-bound CPU world16 plan; never dispatch or write a CP."""
import argparse
import json
from pathlib import Path
import yaml
from starVLA.rl.qwen_world_compare.contracts import file_sha,fingerprint
from starVLA.rl.d46_world16.planning import specification


def inspect_metadata(path):
    path=Path(path).resolve();seal=json.loads((path/'COMPLETE').read_text());files=json.loads((path/'files.json').read_text());identity=json.loads((path/'identity.json').read_text())
    body=dict(identity);sha=body.pop('sha256')
    if seal.get('world')!=8 or identity.get('world')!=8 or fingerprint(body)!=sha or seal['files_sha256']!=file_sha(path/'files.json'):
        raise ValueError('Not a sealed original world8 checkpoint')
    required={'optimizer.pt','identity.json','transaction.json'}|{f'rank{r}.pt'for r in range(8)}
    if not required<=files.keys()or any(not(path/name).is_file()for name in required):raise ValueError('Missing native rank/common file')
    for name in ('identity.json','transaction.json'):
        if file_sha(path/name)!=files[name]:raise ValueError('Changed small native metadata')
    return dict(checkpoint=str(path),COMPLETE=seal,files_sha256=seal['files_sha256'],identity_sha256=sha,
        optimizer_file_sha256=files['optimizer.pt'],rank_files_sha256={str(r):files[f'rank{r}.pt']for r in range(8)},
        inspected='metadata/inventory only; optimizer/rank bytes must be verified on actual migration, not asserted here'),identity


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--checkpoint');a=p.parse_args()
    cfg=yaml.safe_load(Path(a.config).read_text());value=specification(cfg)
    value['config']=dict(path=str(Path(a.config).resolve()),sha256=file_sha(a.config))
    if a.checkpoint:
        value['parent_checkpoint'],identity=inspect_metadata(a.checkpoint)
        if identity['recipe']!=cfg:raise ValueError('Config differs from exact saved recipe')
    value.update(GPU_jobs_launched=0,new_scores=0,native_checkpoint_written=False,
        CPU_checks_command="CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q -s tests/test_d46_world16.py",
        source_files={name:file_sha(name)for name in ('starVLA/rl/d46_world16/planning.py','tools/d46_world16/qualify.py','tests/test_d46_world16.py')})
    out=Path(a.output)
    if out.exists():raise FileExistsError('Do not overwrite an earlier plan')
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');print(json.dumps({'output':str(out),'sha256':file_sha(out),'status':value['status']}))


if __name__=='__main__':main()
