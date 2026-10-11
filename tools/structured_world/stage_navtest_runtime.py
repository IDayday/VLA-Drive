"""Copy a certified private Qwen overlay to fresh inference-only runtimes.

Shared DDP bases are inspected, never changed. This tool performs CPU imports
and read-only GPU snapshots; GPU inference/profile dispatch belongs to Navtest.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path('/mnt/project/VLA-Drive-structured-world-fgtr-io-formal-frozen-20261009')
SHA = '0c54f1c49e7ed2a76b22826b654d6c72625f5b6d'
TREE = '11915ed4b770a1a9060784f31661d15125a256e7f41cf8474fc3fa9ca8d72173'
ART = Path('/mnt/project/structured-world-fgtr-round1-artifacts/20261008')
POLICY = ART/'NAVSIM_EVALUATION_POLICY_20261011.json'
CERTIFICATE = ART/'recovery_dispatch_20261010/navsim_G1_vlawm_native_20261010T235152Z/alternative_vlawm_zt2/PRIVATE_ENVIRONMENT_COPY_COMPLETE.json'
PACKAGES = ('torch', 'deepspeed', 'flash-attn', 'omegaconf', 'Pillow', 'mmcv', 'mmengine',
            'transformers', 'torchvision', 'timm', 'accelerate', 'qwen-vl-utils', 'einops', 'numpy', 'safetensors')
IMPORTS = ('torch', 'deepspeed', 'flash_attn', 'transformers', 'torchvision', 'timm', 'accelerate',
           'qwen_vl_utils', 'mmengine', 'mmcv.cnn', 'mmcv.ops', 'mmcv._ext',
           'starVLA.model.framework.vla_structured_fgtr', 'tools.structured_world.train_vla',
           'tools.structured_world.infer_checkpoint', 'starVLA.dataloader.structured_world.current_inputs')
ENVIRONMENT = {'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '2',
               'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '1', 'TOKENIZERS_PARALLELISM': 'false'}


def read(path): return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''): h.update(block)
    return h.hexdigest()


def require(ok, message):
    if not ok: raise RuntimeError(message)


def write(path, value):
    path = Path(path); temporary = path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n'); temporary.replace(path)


def authorize(host):
    task = read('/mnt/project/server_dispatch_policy.json')['task_authorizations']['structured_world_fgtr_round1']
    policy = read(POLICY)
    require(host in task['allowed_hosts'] and host in policy['inference_allowed_hosts']
            and policy['population'] == 'navtest' and policy['dispatch_navtest_evaluation'] is True
            and policy['dispatch_development_evaluation'] is False, 'Inference authorization changed: '+host)


def ssh(host, command):
    authorize(host)
    return ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', host,
            'cd '+shlex.quote(str(ROOT))+' && '+shlex.join(command)]


def remote(host, code, packet):
    result = subprocess.run(ssh(host, ['/usr/bin/python3', '-B', '-c', code]), input=json.dumps(packet),
                            text=True, capture_output=True, check=True)
    return json.loads(result.stdout.strip().splitlines()[-1])


PROBE = r'''import hashlib,importlib,importlib.metadata,json,os,platform,sys,sysconfig
from pathlib import Path
a=json.load(sys.stdin);sys.dont_write_bytecode=True
import torch
versions={}
for name in a['packages']:
 try:versions[name]=importlib.metadata.version(name)
 except importlib.metadata.PackageNotFoundError:versions[name]=None
report={'python':platform.python_version(),'SOABI':sysconfig.get_config_var('SOABI'),
 'executable':sys.executable,'real_executable':os.path.realpath(sys.executable),
 'torch':torch.__version__,'torch_cuda':torch.version.cuda,'torch_CXX11_ABI':torch._C._GLIBCXX_USE_CXX11_ABI,
 'torch_C_sha256':hashlib.sha256(Path(torch._C.__file__).read_bytes()).hexdigest(),'versions':versions}
if a['full']:
 sys.path.insert(0,a['root']);modules={}
 for name in a['imports']:
  module=importlib.import_module(name);file=Path(module.__file__)
  modules[name]={'file':str(file),'sha256':hashlib.sha256(file.read_bytes()).hexdigest()}
 from tools.structured_world.train_vla import source_hashes
 report['execution_source_sha256']=source_hashes();report['CPU_modules']=modules
 import subprocess
 report['source_sha']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=a['root'],text=True).strip()
 report['source_dirty']=subprocess.check_output(['git','status','--porcelain'],cwd=a['root'],text=True).strip()
report['CUDA_initialized']=torch.cuda.is_initialized()
if report['CUDA_initialized']:raise RuntimeError('CPU-only probe initialized CUDA')
print(json.dumps(report))
'''


def probe(python, host=None, full=False):
    packet = {'packages': PACKAGES, 'imports': IMPORTS, 'full': full, 'root': str(ROOT)}
    command = [str(python), '-B', '-c', PROBE]
    if host: command = ssh(host, ['env']+[k+'='+v for k, v in ENVIRONMENT.items()]+command)
    result = subprocess.run(command, cwd=ROOT, env={**os.environ, **ENVIRONMENT}, input=json.dumps(packet),
                            text=True, capture_output=True, check=True)
    return json.loads(result.stdout.strip().splitlines()[-1])


def compatible(source, candidate, private_packages):
    keys = ('python', 'SOABI', 'torch', 'torch_cuda', 'torch_CXX11_ABI', 'torch_C_sha256')
    require(all(candidate[k] == source[k] for k in keys), 'DDP base Python/Torch ABI differs')
    require(all(candidate['versions'][p] == source['versions'][p] for p in PACKAGES if p not in private_packages),
            'DDP base numerical/package versions differ')


def manifest(source):
    files, links = {}, {}
    for path in sorted(source.rglob('*')):
        name = str(path.relative_to(source))
        if path.is_symlink(): links[name] = os.readlink(path)
        elif path.is_file(): files[name] = {'sha256': digest(path), 'bytes': path.stat().st_size}
    certificate = read(CERTIFICATE)
    require(certificate['passed'] and files == certificate['source_files'] and links == certificate['symlinks'],
            'Source private overlay differs from certified G1 replica')
    return {'files': files, 'links': links, 'certificate': str(CERTIFICATE), 'certificate_sha256': digest(CERTIFICATE)}


VERIFY = r'''import hashlib,json,os,sys
from pathlib import Path
a=json.load(sys.stdin);dest=Path(a['destination']);files={};links={}
for path in sorted(dest.rglob('*')):
 name=str(path.relative_to(dest))
 if path.is_symlink():links[name]=os.readlink(path)
 elif path.is_file():files[name]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}
if files!=a['files'] or links!=a['links']:raise RuntimeError('Private runtime byte/link manifest differs')
print(json.dumps({'passed':True,'files':len(files),'links':len(links),'bytes':sum(v['bytes'] for v in files.values())}))
'''


def gpu_snapshot(host):
    command = ['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu', '--format=csv,noheader,nounits']
    rows = subprocess.check_output(ssh(host, command), text=True)
    return [dict(zip(('index', 'uuid', 'name', 'total_MiB', 'used_MiB', 'utilization_percent'),
                     [v.strip() for v in row.split(',')])) for row in rows.splitlines()]


def stage(host, args, source_probe, overlay):
    output = args.output/host; output.mkdir()
    locks = ART/'navsim_navtest_runtime_staging_20261011/locks'; locks.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256((host+':'+str(args.runtime_dir)).encode()).hexdigest()
    with (locks/(key+'.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(output/'STATE.json', {'status': 'PREPARING', 'host': host, 'started_unix': time.time()})
        try:
            base = probe(args.base_python, host); write(output/'BASE_BEFORE.json', base)
            compatible(source_probe, base, {'mmcv', 'mmengine'})
            before = gpu_snapshot(host); write(output/'GPU_BEFORE.json', before)
            remote(host, "import json,sys;from pathlib import Path;a=json.load(sys.stdin);p=Path(a['destination']);p.mkdir(parents=True,exist_ok=False);print(json.dumps({'fresh':True}))", {'destination': str(args.runtime_dir)})
            command = ['rsync', '-a', '--ignore-existing', '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=10',
                       str(args.source)+'/', host+':'+str(args.runtime_dir)+'/']
            with (output/'copy.log').open('x') as stream:
                authorize(host); subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
            copied = remote(host, VERIFY, {'destination': str(args.runtime_dir), **overlay})
            write(output/'COPY_VERIFIED.json', {'source_manifest_sha256': digest(args.output/'SOURCE_MANIFEST.json'),
                  'command': command, **copied})
            worker = probe(args.runtime_dir/'bin/python', host, full=True); write(output/'CPU_IMPORTS.json', worker)
            compatible(source_probe, worker, set())
            require(worker['executable'] == str(args.runtime_dir/'bin/python') and worker['source_sha'] == SHA
                    and not worker['source_dirty'] and worker['execution_source_sha256'] == TREE
                    and {k:v['sha256'] for k,v in worker['CPU_modules'].items()} ==
                        {k:v['sha256'] for k,v in source_probe['CPU_modules'].items()}, 'Frozen CPU module/source differs')
            base_after = probe(args.base_python, host); require(base_after == base, 'Shared DDP base probe changed')
            write(output/'BASE_AFTER.json', base_after); after = gpu_snapshot(host); write(output/'GPU_AFTER.json', after)
            readiness = {'schema': 'navsim_navtest_private_runtime_readiness_v1', 'passed': True, 'host': host,
                'python': str(args.runtime_dir/'bin/python'), 'source_sha': SHA, 'execution_source_sha256': TREE,
                'versions': worker['versions'], 'CPU_imports_passed': True, 'CUDA_initialized': False,
                'shared_DDP_base_unchanged': True, 'private_overlay_all_bytes_and_links_verified': copied,
                'GPU_UUIDs': sorted(v['uuid'] for v in after), 'GPU_snapshots': [before, after],
                'inference_only': True, 'GPU_memory_profile_or_prediction_parity_qualified': False,
                'limits': 'CPU/runtime readiness only; Newton must perform actual GPU memory/prediction qualification before inference. No assets copied or training dispatched.',
                'evidence_sha256': {str(output/n): digest(output/n) for n in ('BASE_BEFORE.json', 'BASE_AFTER.json', 'COPY_VERIFIED.json', 'CPU_IMPORTS.json')},
                'controller_code_sha256': digest(__file__), 'policy_sha256': digest(POLICY), 'completed_unix': time.time()}
            write(output/'READINESS.json', readiness); write(output/'STATE.json', {'status': 'CPU_RUNTIME_PREPARATION_COMPLETE', 'readiness': str(output/'READINESS.json')})
            print(json.dumps({'host': host, 'python': readiness['python'], 'readiness': str(output/'READINESS.json'), 'passed': True}), flush=True)
            return readiness
        except BaseException as error:
            write(output/'STATE.json', {'status': 'FAILED', 'error': repr(error), 'stderr': getattr(error, 'stderr', None), 'partial_runtime_preserved': str(args.runtime_dir)})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hosts', nargs='+', required=True); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=Path('/tmp/structured-world-round1/envs/qwen'))
    parser.add_argument('--base-python', type=Path, default=Path('/root/miniconda3/envs/ddp/bin/python'))
    parser.add_argument('--runtime-dir', type=Path, default=Path('/var/tmp/structured-world-fgtr-round1-20261008/navtest_qwen_runtime_v1'))
    args = parser.parse_args(); require(len(set(args.hosts)) == len(args.hosts), 'Duplicate host')
    require(args.runtime_dir.is_absolute() and str(args.runtime_dir).startswith('/var/tmp/structured-world-fgtr-round1-20261008/'), 'Use a fresh private NVMe runtime')
    for host in args.hosts: authorize(host)
    args.output.mkdir(parents=True, exist_ok=False)
    overlay = manifest(args.source); write(args.output/'SOURCE_MANIFEST.json', overlay)
    source = probe(args.source/'bin/python', full=True); write(args.output/'SOURCE_CPU_IMPORTS.json', source)
    require(source['source_sha'] == SHA and not source['source_dirty'] and source['execution_source_sha256'] == TREE, 'Frozen source differs')
    with ThreadPoolExecutor(max_workers=len(args.hosts)) as pool:
        results = list(pool.map(lambda host: stage(host, args, source, overlay), args.hosts))
    write(args.output/'COMPLETE.json', {'passed': True, 'hosts': args.hosts, 'readiness': results, 'controller_code_sha256': digest(__file__)})


if __name__ == '__main__': main()
