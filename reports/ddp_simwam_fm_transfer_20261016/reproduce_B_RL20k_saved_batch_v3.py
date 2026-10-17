import argparse,fcntl,json,os,pathlib,subprocess,time,uuid
from types import SimpleNamespace
from scripts.analysis.fm_transfer.diagnose_saved_batch import TransferDiagnostic
from starVLA.rl.simwam_recipe.contracts import file_sha,fingerprint
from starVLA.rl.fm_transfer.contracts import require_dispatch
from starVLA.rl.flow_grpo.transactions import atomic_json
ROOT=pathlib.Path('/mnt/project/fm-transfer-20261016')
SOURCE=ROOT/'source/fd440275edc3fdf40b2aaa46c38da1d038068902'
OUT=ROOT/'resources/B_RL_20k_saved_batch_dispatch_20261017_v3'
CHECKPOINT=ROOT/'runs/B_SOURCE_ACTION_DIT_R_SOURCE_seed42_v1/checkpoints/update020000_inner0'
RESULT=ROOT/'analysis/B_SOURCE_ACTION_DIT_RL_update020000_saved_batch.json'
SNAPSHOT=ROOT/'analysis/B_SOURCE_ACTION_DIT_RL_update020000_progress_snapshot.json'
RUN=ROOT/'runs/B_SOURCE_ACTION_DIT_R_SOURCE_seed42_v1'
HARD=ROOT/'hard_manifests/B_SOURCE_ACTION_DIT_SFT_update100000.json'
SCREEN=ROOT/'screens/B_SOURCE_ACTION_DIT_SFT_update100000'
COLLECTION=5000

def preflight():
    require_dispatch()
    identity=json.loads((RUN/'identity.json').read_text())
    body=dict(identity);expected=body.pop('sha256')
    if fingerprint(body)!=expected:raise ValueError('Changed formal run identity')
    runtime=TransferDiagnostic()
    runtime.prepare(SimpleNamespace(gradient_strata_registration=None,hard_manifest=str(HARD),
                    sft_screen=str(SCREEN),collection=COLLECTION),identity)
    return dict(status='OWN_SFT_ASSET_PREFLIGHT_PASS',run_identity=identity['sha256'],
                own_SFT_updates=runtime.parent['updates'],own_SFT_model_sha256=runtime.parent['model_sha256'],
                hard_manifest_sha256=file_sha(HARD),screen_manifest_sha256=file_sha(SCREEN/'manifest.json'),
                RL_update=20000,collection=COLLECTION,new_candidates=0,optimizer_updates=0)

parser=argparse.ArgumentParser(description='One registered saved-batch diagnostic with fixed own-SFT identity')
parser.add_argument('--preflight-only',action='store_true')
args=parser.parse_args()
preflight_receipt=preflight()
if args.preflight_only:
    print(json.dumps(preflight_receipt));raise SystemExit(0)
lock=(OUT/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
receipt=dict(schema='fm_transfer_one_shot_saved_batch_dispatch_v1',status='WAITING_FOR_SEALED_20K',
    pid=os.getpid(),start_ticks=int(pathlib.Path('/proc/self/stat').read_text().split()[21]),
    started_unix=time.time(),deadline_unix=1792261027.2822075,source=str(SOURCE),
    checkpoint=str(CHECKPOINT),result=str(RESULT),new_candidates=0,optimizer_updates=0,asset_preflight=preflight_receipt)
if (OUT/'COMPLETE.json').exists() or (OUT/'dispatch.json').exists() or RESULT.exists():raise RuntimeError('Existing diagnostic; inspect before any retry')
atomic_json(OUT/'owner.json',receipt)
try:
    while time.time()<receipt['deadline_unix']:
        host=require_dispatch()
        if (CHECKPOINT/'COMPLETE').exists():
            seal=json.loads((CHECKPOINT/'COMPLETE').read_text())
            if seal['updates']!=20000 or seal['phase']!='COMPLETE' or seal['world']!=8 or seal['next_inner']!=0:
                raise ValueError('Unexpected registered milestone boundary')
            progress=json.loads((ROOT/'runs/B_SOURCE_ACTION_DIT_R_SOURCE_seed42_v1/progress.json').read_text())
            if not SNAPSHOT.exists():
                if progress['updates']==20000 and progress['next_inner']==0:
                    atomic_json(SNAPSHOT,progress)
                    receipt['exact_boundary_progress_snapshot']=str(SNAPSHOT)
                elif progress['updates']<20000:
                    receipt['status']='WAITING_FOR_EXACT_PROGRESS_PUBLICATION'
                    atomic_json(OUT/'status.json',dict(receipt,observed_unix=time.time()))
                    time.sleep(2);continue
                else:
                    receipt['exact_progress_snapshot_unavailable_observed_update']=progress['updates']
            def inspect():
                raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free','--format=csv,noheader,nounits'],text=True)
                rows=[x.split(', ') for x in raw.strip().splitlines()]
                owners=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader'],text=True)
                return rows,owners
            first,owners1=inspect();time.sleep(2);second,owners2=inspect()
            available=[b for a,b in zip(first,second) if a[:2]==b[:2] and int(a[2])==0 and int(b[2])==0 and min(int(a[3]),int(b[3]))>=36000 and a[1] not in owners1 and b[1] not in owners2]
            if available:
                gpu=available[0]
                cmd=['/mnt/project/flow-safe-improvement-20261013/venv/bin/python','-m','scripts.analysis.fm_transfer.diagnose_saved_batch',
                    '--run',str(ROOT/'runs/B_SOURCE_ACTION_DIT_R_SOURCE_seed42_v1'),'--checkpoint',str(CHECKPOINT),
                    '--collection',str(COLLECTION),'--hard-manifest',str(HARD),
                    '--sft-screen',str(SCREEN),'--output',str(RESULT)]
                require_dispatch()
                receipt.update(status='LAUNCHING',host=host,gpu=int(gpu[0]),gpu_uuid=gpu[1],
                    command=cmd,resource_samples=[first,second],owners=[owners1,owners2],dispatch_unix=time.time())
                atomic_json(OUT/'dispatch.json',receipt)
                env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu[0],PYTHONPATH=str(SOURCE),
                    OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
                    CUBLAS_WORKSPACE_CONFIG=':4096:8',NO_ALBUMENTATIONS_UPDATE='1',TOKENIZERS_PARALLELISM='false')
                with (OUT/'diagnostic.log').open('xb') as log:
                    child=subprocess.Popen(cmd,cwd=SOURCE,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
                receipt.update(status='RUNNING',child_pid=child.pid,child_start_ticks=int((pathlib.Path('/proc')/str(child.pid)/'stat').read_text().split()[21]))
                atomic_json(OUT/'dispatch.json',receipt)
                receipt['exit_code']=child.wait();receipt['ended_unix']=time.time()
                if receipt['exit_code'] or not RESULT.exists():raise RuntimeError('Diagnostic failed; inspect retained log, no automatic retry')
                value=json.loads(RESULT.read_text())
                if value['status']!='COMPLETE_SAVED_BATCH_DIAGNOSTIC' or not value['adapters_unchanged']:raise ValueError('Incomplete diagnostic')
                receipt['status']='COMPLETE';atomic_json(OUT/'COMPLETE.json',receipt);break
            receipt['status']='WAITING_FOR_IDLE_ALLOWED_GPU'
        atomic_json(OUT/'status.json',dict(receipt,observed_unix=time.time()));time.sleep(10)
    else:
        receipt['status']='WAIT_DEADLINE_NOT_PROOF_OF_TRAINING_STOPPED'
except BaseException:
    import traceback
    receipt.update(status='FAILED',error=traceback.format_exc());raise
finally:
    atomic_json(OUT/'status.json',dict(receipt,observed_unix=time.time()))
