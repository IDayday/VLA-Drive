"""Own one qualification/formal process; no implicit retry or unknown eviction."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import uuid
from starVLA.rl.fm_transfer.source_process import ROOT, B_SFT, MODES, load_process_recipe
from starVLA.rl.fm_transfer.contracts import require_dispatch
from starVLA.rl.simwam_recipe.contracts import file_sha, fingerprint
from starVLA.rl.flow_grpo.transactions import atomic_json
from scripts.analysis.fm_transfer.qualify_source_process import configuration, PYTHON
from scripts.analysis.fm_transfer.dispatch_final_evaluation import gpu_snapshot, admit_devices


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--mode',choices=MODES,required=True)
    p.add_argument('--phase',choices=['qualify','train'],required=True);p.add_argument('--output',required=True)
    p.add_argument('--qualification-root');p.add_argument('--resume');a=p.parse_args()
    host=require_dispatch();source=Path.cwd().resolve();seal=json.loads((source/'SOURCE_COMPLETE.json').read_text())
    if not seal['git_archive_extraction_complete']:raise ValueError('Incomplete source snapshot')
    cfg_path,manifest=configuration(a.mode);cfg=load_process_recipe(cfg_path)
    registration=json.loads((ROOT/'registration.json').read_text());registered=registration['formal_runs'][a.mode]
    body=dict(registration);sha=body.pop('sha256')
    if (fingerprint(body)!=sha or registered['manifest_sha256']!=file_sha(manifest) or
            registered['max_updates']!=cfg['max_steps'] or registration['SFT_files_sha256']!=file_sha(B_SFT/'files.json')):
        raise ValueError('Scientific registration changed')
    output=Path(a.output);dispatch=output.parent/(output.name+'_dispatch');dispatch.mkdir(parents=True,exist_ok=True)
    guard=(dispatch/'owner.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    previous=list(dispatch.glob('attempt*.json'))
    if previous and not a.resume:raise ValueError('Existing dispatch; inspect, do not duplicate')
    for path in previous:
        prior=json.loads(path.read_text())
        if prior['host']!=host:raise ValueError('Cross-host recovery requires an explicit ownership transfer')
        proc=Path('/proc')/str(prior['pid'])/'stat'
        if proc.exists() and int(proc.read_text().split()[21])==prior['start_ticks']:
            raise ValueError('Recorded process is still alive')
    evidence=None
    if a.phase=='train':
        if not a.qualification_root:raise ValueError('Actual eight-rank gate required')
        gate=Path(a.qualification_root);evidence=json.loads((gate/'COMPLETE.json').read_text())
        if (evidence['status']!='PASS' or evidence['world']!=8 or evidence['source']!=str(source) or
                evidence['mode']!=a.mode or evidence['manifest_sha256']!=file_sha(manifest) or
                evidence['config_sha256']!=file_sha(cfg_path)):
            raise ValueError('Qualification does not match formal execution')
        for rel,h in evidence['evidence'].items():
            if file_sha(gate/rel)!=h:raise ValueError('Gate evidence changed')
        qualified=json.loads((gate/'continuous/identity.json').read_text())
        for rel,h in qualified['source_files'].items():
            if file_sha(source/rel)!=h:raise ValueError('Formal source differs from actual eight-rank test')
        recipe=dict(qualified['recipe']);recipe.pop('supervised_transfer')
        if recipe!=cfg:raise ValueError('Changed qualified recipe')
    first=gpu_snapshot();time.sleep(2);second=gpu_snapshot()
    admit_devices(first,second,[(i,i) for i in range(8)],set())
    free=[second['gpus'][i]['free_MiB'] for i in range(8)]
    if len(free)!=8 or min(free)<36000:raise RuntimeError('Insufficient eight-GPU headroom; preserve real jobs')
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',NO_ALBUMENTATIONS_UPDATE='1',TOKENIZERS_PARALLELISM='false',
        VLM_ATTN_IMPLEMENTATION='flash_attention_2',PYTHONPATH=str(source))
    if a.phase=='qualify':
        if a.resume:raise ValueError('Qualification retries need a separate output identity')
        command=[PYTHON,'-m','scripts.analysis.fm_transfer.qualify_source_process','--mode',a.mode,'--output',str(output)]
    else:
        command=[PYTHON,'-m','torch.distributed.run','--standalone','--nproc_per_node=8','--module',
            'scripts.analysis.fm_transfer.train_source_process','--config',cfg_path,'--manifest',str(manifest),
            '--sft-checkpoint',str(B_SFT),'--output',str(output)]
        if a.resume:
            parent=Path(a.resume).resolve()
            if parent.parent!=output.resolve()/'checkpoints' or not (parent/'COMPLETE').exists():
                raise ValueError('Same-run native resume required')
            command+=['--resume',str(parent)]
    attempt='attempt_'+uuid.uuid4().hex;logfile=dispatch/(attempt+'.log');require_dispatch()
    with logfile.open('xb') as log:
        child=subprocess.Popen(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    receipt=dict(status='STARTED',mode=a.mode,phase=a.phase,host=host,pid=child.pid,
        start_ticks=int((Path('/proc')/str(child.pid)/'stat').read_text().split()[21]),
        start_unix=time.time(),source=str(source),source_commit=seal['commit'],output=str(output),command=command,
        log=str(logfile),free_MiB=free,registration_sha256=file_sha(ROOT/'registration.json'),
        admission_observations=[first,second],
        qualification_sha256=file_sha(Path(a.qualification_root)/'COMPLETE.json') if evidence else None,
        dispatcher_sha256=file_sha(__file__),resume=a.resume,qualification_weights_used=False)
    atomic_json(dispatch/(attempt+'.json'),receipt);print(json.dumps(receipt))

if __name__=='__main__':main()
