"""Prepare or route native NAV G1 recovery; default mode is read-only inspection.

Use the original local qwen Python. --prepare-assets only copies/verifies CPU
assets. --execute is a separate, explicit invocation after preparation review.
G3 NU qualification supplies environment/hardware evidence, never NAV method or
throughput results. Original registrations and the G0 queue remain historical.
"""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import runpy
import shlex
import socket
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path('/mnt/project/VLA-Drive-structured-world-fgtr-io-formal-frozen-20261009')
SHA = '0c54f1c49e7ed2a76b22826b654d6c72625f5b6d'
IDENTITY = '8f331622551d1f1c1c439a6d01411affe012a1777a63c61d8525470dc1e3ded5'
GROUP = 'G1_FULL_UNIFORM'
TARGETS = (25000, 50000, 75000, 100000)
ARTIFACTS = Path('/mnt/project/structured-world-fgtr-round1-artifacts/20261008')
CURRENT = Path('/var/tmp/ddp-full-foresight-20260929/student_train_v1')
IMAGES = Path('/var/tmp/ddp-full-foresight-20260929/images')
POLICY = Path('/mnt/project/server_dispatch_policy.json')
PYTHON = Path('/tmp/structured-world-round1/envs/qwen/bin/python')
ENV_PACKAGES = ('torch', 'deepspeed', 'flash-attn', 'omegaconf', 'Pillow', 'mmcv', 'mmengine',
                'transformers', 'torchvision', 'timm', 'accelerate', 'qwen-vl-utils', 'einops', 'numpy', 'safetensors')
CPU_IMPORTS = ('torch', 'deepspeed', 'flash_attn', 'transformers', 'torchvision', 'timm',
               'accelerate', 'qwen_vl_utils', 'mmengine', 'mmcv.cnn', 'mmcv.ops', 'mmcv._ext',
               'starVLA.model.framework.vla_structured_fgtr', 'tools.structured_world.train_vla')


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''): h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition: raise ValueError(message)


def authorize(host):
    task = read(POLICY)['task_authorizations']['structured_world_fgtr_round1']
    require(host in task['allowed_hosts'] and host in task.get('latest_explicit_resource_hosts', task['allowed_hosts']),
            'Task authorization changed: '+host)


def resource_ready(snapshot, reserve, sharing=False):
    if sharing:
        require(read(POLICY)['task_authorizations']['structured_world_fgtr_round1'].get('shared_GPU_deployment_authorized'),
                'Low-utilization sharing authorization changed')
    return all(r['free_MiB'] >= reserve and r['utilization'] <= 5 and (sharing or r['used_MiB'] <= 128) for r in snapshot)


def clean_source():
    require(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == SHA
            and not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip(), 'Frozen source is not clean at exact SHA')


def original_plan(path):
    """Extract original commands without invoking the immutable queue controller."""
    clean_source()
    command = read(path)['argv']
    runner = ROOT/'tools/structured_world/run_navsim_pair.py'
    require(command[:3] == [str(PYTHON), '-u', str(runner)], 'Foreign original controller')
    ns = runpy.run_path(str(runner))
    body = next(n.body for n in ast.parse(runner.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    end = next(i for i, n in enumerate(body) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'args' for t in n.targets))
    exec(compile(ast.Module(body=body[:end]+ast.parse('args = parser.parse_args('+repr(command[3:])+')').body,
                           type_ignores=[]), str(runner), 'exec'), ns)
    args = ns['args']
    require(args.groups == [GROUP, 'G0_NO_FUTURE_LABEL'] and args.execution_mode == 'io_preserving_v1'
            and args.resume_origin_run and args.execution_acceptance, 'Wrong native NAV G1 contract')
    nodes = list(ast.walk(ast.Module(body=body, type_ignores=[])))
    training = next(n for n in nodes if isinstance(n, ast.If) and ast.unparse(n.test) == 'completed < target')
    ns['training_nodes'] = training.body[:-1]
    ns['evaluation_nodes'] = {key: n.args[1] for n in nodes if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name) and n.func.id == 'run' for key in ('dev_infer', 'dev_score', 'dev_fields')
        if any(isinstance(c, ast.Constant) and c.value == '_'+key+'_' for c in ast.walk(n.args[0]))}
    return args, ns


def commands(args, ns, target):
    require(target in TARGETS, 'Unregistered observation boundary')
    run = args.output/GROUP
    values = {**ns, 'group': GROUP, 'target': target, 'run_root': run,
        'config': ROOT/'configs/structured_world/navsim_G1_FULL_UNIFORM_seed42.yaml', 'latest': run/'checkpoints/latest',
        'tag': f'milestone_{target:07d}', 'predictions': run/f'dev_{target}_seed42',
        'planning': run/f'dev_{target}_canonical', 'fields': run/f'dev_{target}_scene_metrics', 'local_python': sys.executable}
    exec(compile(ast.Module(body=ns['training_nodes'], type_ignores=[]), '<frozen native training>', 'exec'), values)
    require('--resume' in values['command'], 'Native latest resume is mandatory')
    return [(f'{GROUP}_train_to_{target}', values['command'], True, None)] + [
        (f'{GROUP}_{key}_{target}', eval(compile(ast.Expression(ns['evaluation_nodes'][key]), '<frozen evaluation>', 'eval'), values), gpu,
         values[name]/'COMPLETE.json') for key, gpu, name in
        (('dev_infer', True, 'predictions'), ('dev_score', False, 'planning'), ('dev_fields', False, 'fields'))]


def latest(run):
    tag = (run/'checkpoints/latest').read_text().strip()
    require(re.fullmatch(r'(latest|milestone)_\d{7}', tag), 'Unsafe native latest tag')
    folder = run/'checkpoints'/tag
    c = read(folder/'COMPLETE.json')
    names = {'mp_rank_00_model_states.pt'} | {f'rng_rank{i}.pt' for i in range(8)} | {
        f'zero_pp_rank_{i}_mp_rank_00_optim_states.pt' for i in range(8)}
    require(c['identity'] == IDENTITY and c['ranks'] == 8 and c['tag'] == tag
            and c['completed'] == int(tag.split('_')[1]) and 25000 <= c['completed'] <= 100000
            and c['status'] in ('PAUSED', 'RUNNING', 'COMPLETE') and set(c['file_sha256']) == names
            and {p.name for p in folder.glob('*.pt')} == names, 'Incomplete/foreign native checkpoint')
    return {'tag': tag, 'completed': c['completed'], 'identity': IDENTITY,
            'complete_file': str(folder/'COMPLETE.json'), 'complete_sha256': digest(folder/'COMPLETE.json')}


def route(host, command):
    authorize(host)
    return ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', host,
        'cd '+shlex.quote(str(ROOT))+' && env OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 '+shlex.join(command)]


# CPU-only probe: exact source/config/environment, metadata and shared inputs.
PROBE = '''import ast,hashlib,importlib,importlib.metadata,json,os,shutil,subprocess,sys,time
from pathlib import Path
from omegaconf import OmegaConf
def read(p): return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''): h.update(b)
 return h.hexdigest()
a=json.loads(sys.argv[1]);root=Path(a['root']);run=Path(a['run']);identity=read(run/'identity.json')
sys.path.insert(0,str(root))
imports={}
for name in a['CPU_imports']:
 module=importlib.import_module(name)
 imports[name]={'file':str(module.__file__),'sha256':sha(module.__file__)}
if sys.executable!=a['qwen_python'] or __import__('torch').cuda.is_initialized(): raise ValueError('CPU import probe used wrong runtime or initialized CUDA')
train=root/'tools/structured_world/train_vla.py'
ns={'ROOT':root,'Path':Path,'hashlib':hashlib,'json':json,'__file__':str(train)}
function=next(n for n in ast.parse(train.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='source_hashes')
exec(compile(ast.Module(body=[function],type_ignores=[]),str(train),'exec'),ns)
assets={}
for name,folder in (('labels',a['cache']),('dino',a['dino']),('current',a['current'])):
 p=Path(folder)
 assets[name]={'path':str(p),'exists':p.is_dir()}
 if all((p/f).is_file() for f in ('identity.json','COMPLETE.json','index.json' if name!='dino' else 'SOURCE_CHUNK_HASHES.json')):
  assets[name].update(identity=read(p/'identity.json')['identity'],complete=read(p/'COMPLETE.json'),
   metadata_sha256={f:sha(p/f) for f in (('identity.json','index.json','ego_identity.json','COMPLETE.json','local_stage.json','local_image_hashes.json') if name=='current' else ('identity.json','COMPLETE.json','index.json') if name=='labels' else ('identity.json','COMPLETE.json','SOURCE_CHUNK_HASHES.json')) if (p/f).is_file()})
versions={k:importlib.metadata.version(k) for k in a['environment_packages']}
report={'host':__import__('socket').gethostname().removesuffix('-worker-0'),
 'source_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
 'dirty':subprocess.check_output(['git','status','--porcelain'],text=True).strip(),
 'source_digest':ns['source_hashes'](),'config_matches':OmegaConf.to_container(OmegaConf.load(a['config']),resolve=True)==identity['config'],
 'versions':versions,'CPU_import_readiness':{'passed':True,'CUDA_initialized':False,'executable':sys.executable,'modules':imports},
 'assets':assets,'shared_sha256':{p:sha(p) for p in a['shared_hashes']},
 'shared_paths_present':all(Path(p).exists() for p in a['shared_paths']),
 'images_present':Path(a['images']).is_dir(),'NVMe_free_bytes':shutil.disk_usage('/var/tmp').free}
'''

VERIFY = '''
from concurrent.futures import ThreadPoolExecutor
begin=time.monotonic();tag=(run/'checkpoints/latest').read_text().strip();c=read(run/'checkpoints'/tag/'COMPLETE.json')
def checked(item):
 name,expected=item;actual=sha(run/'checkpoints'/tag/name)
 if actual!=expected: raise ValueError('Native checkpoint hash changed: '+name)
 return name,actual
with ThreadPoolExecutor(max_workers=2) as pool: hashes=dict(pool.map(checked,c['file_sha256'].items()))
report['native_checkpoint_files_sha256']=hashes;report['checkpoint_tag']=tag;report['checkpoint_verify_seconds']=time.monotonic()-begin
for name,folder in (('labels',a['cache']),('dino',a['dino'])):
 start=time.monotonic();p=Path(folder);rows=read(Path(a['cache'])/'index.json')
 def payload(row):
  token=row['token'];meta=read(p/'records'/(token+'.json'))
  file=p/('labels' if name=='labels' else 'targets')/(token+'.npz')
  if sha(file)!=meta['label_sha256' if name=='labels' else 'sha256']: raise ValueError('Staged payload changed: '+token)
  if meta['cache_identity' if name=='labels' else 'identity']!=report['assets'][name]['identity']: raise ValueError('Foreign asset record')
  return file.stat().st_size
 with ThreadPoolExecutor(max_workers=8) as pool: size=sum(pool.map(payload,rows))
 report['assets'][name]['payload_verification']={'scenes':len(rows),'bytes':size,'seconds':time.monotonic()-start,'all_SHA256_passed':True}
report['initialization_sha256']={str(Path(identity['config']['framework']['qwenvl']['base_vlm'])/n):sha(Path(identity['config']['framework']['qwenvl']['base_vlm'])/n) for n in identity['initialization_assets']['Qwen_public_pretraining_files']}
stage=read(Path(a['current'])/'local_stage.json');images=read(Path(a['current'])/'local_image_hashes.json')
if stage['image_root']!=a['images'] or stage['source_identity']!=identity['current_and_GT_identity']['identity'] or stage['scenes']!=101592 or stage['images']!=len(images) or sha(Path(a['current'])/'local_image_hashes.json')!=stage['image_hash_manifest']: raise ValueError('Foreign original image staging')
def image_checked(item):
 name,expected=item;p=Path(a['images'])/(hashlib.sha256(name.encode()).hexdigest()+'.jpg')
 if sha(p)!=expected: raise ValueError('Original staged RGB changed: '+name)
 return p.stat().st_size
with ThreadPoolExecutor(max_workers=8) as pool: image_bytes=sum(pool.map(image_checked,images.items()))
if len(images)!=304776: raise ValueError('Incomplete original RGB population')
report['image_payload_verification']={'images':len(images),'bytes':image_bytes,'all_SHA256_passed':True,'original_image_root':a['images']}
# Execute the existing frozen CPU equivalence check with the original trainer's
# default image_root, retaining the original mapping and unchanged RGB bytes.
import runpy,types
ns=runpy.run_path(str(root/'tools/structured_world/validate_prefetch.py'))
body=next(n.body for n in ast.parse((root/'tools/structured_world/validate_prefetch.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='main')
at=next(i for i,n in enumerate(body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='args' for t in n.targets))
ns['args']=types.SimpleNamespace(cache=Path(a['cache']),current=Path(a['current']),dino=Path(a['dino']),images=Path(a['images']),DINO_identity=identity['DINO_identity'],output=Path(a['cpu_report']))
start=time.monotonic();exec(compile(ast.Module(body=body[at+1:],type_ignores=[]),'<frozen CPU equivalence>','exec'),ns)
report['CPU_prefetch']=read(a['cpu_report']);report['CPU_prefetch_seconds']=time.monotonic()-start
report['GPU_training_throughput_measured']=False
'''


def probe(args, options, expected, full=False):
    import importlib.metadata
    packet = {'root': str(ROOT), 'run': str(args.output/GROUP), 'config': str(ROOT/'configs/structured_world/navsim_G1_FULL_UNIFORM_seed42.yaml'),
        'cache': str(args.cache), 'dino': str(args.dino_root), 'current': str(CURRENT), 'images': str(IMAGES),
        'shared_hashes': expected['shared_hashes'], 'shared_paths': expected['shared_paths'], 'cpu_report': str(options.output/'CPU_PREFETCH_EQUIVALENCE.json'),
        'qwen_python': args.qwen_python, 'environment_packages': ENV_PACKAGES, 'CPU_imports': CPU_IMPORTS}
    code = PROBE+(VERIFY if full else '')+'\nprint(json.dumps(report))'
    result = subprocess.check_output(route(options.destination_host, ['env', 'CUDA_VISIBLE_DEVICES=', args.qwen_python, '-B', '-c', code, json.dumps(packet)]), text=True)
    report = json.loads(result.splitlines()[-1])
    require(report['host'] == options.destination_host and report['source_sha'] == SHA and not report['dirty']
            and report['source_digest'] == expected['execution_source_sha256'] and report['config_matches'], 'Destination source/config mismatch')
    require(report['versions'] == {k: importlib.metadata.version(k) for k in ENV_PACKAGES}, 'Destination environment differs from original local qwen')
    imports = report.get('CPU_import_readiness', {})
    require(imports.get('passed') and imports.get('CUDA_initialized') is False
            and imports.get('executable') == args.qwen_python and set(imports.get('modules', {})) == set(CPU_IMPORTS),
            'Destination trainer/model CPU import readiness failed')
    require(report['shared_sha256'] == expected['shared_hashes'] and report['shared_paths_present'], 'Shared inputs changed/missing')
    for name, files in expected['asset_metadata_sha256'].items():
        if full or options.execute:
            require(report['assets'][name].get('metadata_sha256') == files, 'Incomplete/foreign NVMe '+name)
    if full or options.execute: require(report['images_present'], 'Original NVMe RGB residency missing')
    if full:
        identity = read(args.output/GROUP/'identity.json')
        wanted = {str(Path(identity['config']['framework']['qwenvl']['base_vlm'])/n): h for n, h in identity['initialization_assets']['Qwen_public_pretraining_files'].items()}
        require(report['initialization_sha256'] == wanted and report['CPU_prefetch']['passed']
                and len(report['native_checkpoint_files_sha256']) == 17, 'Preparation/native evidence failed')
    lines = subprocess.check_output(route(options.destination_host, ['nvidia-smi', '--query-gpu=uuid,memory.used,memory.free,utilization.gpu', '--format=csv,noheader,nounits']), text=True).splitlines()
    report['GPU_snapshot'] = [dict(zip(('uuid', 'used_MiB', 'free_MiB', 'utilization'), [r[0], *map(int, r[1:])]))
                              for r in ([v.strip() for v in line.split(',')] for line in lines)]
    require(len(lines) == 8 and sorted(r['uuid'] for r in report['GPU_snapshot']) == expected['physical_gpu_UUIDs'], 'Destination GPU UUIDs changed')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-controller', type=Path, default=ARTIFACTS/'formal_navsim_zt2_G1_G0_v3_io/EXECUTED_CONTROLLER.json')
    parser.add_argument('--destination-host', default='training-vlawm-zt')
    parser.add_argument('--destination-qualification', type=Path, default=ARTIFACTS/'nuscenes_full8_deployment_20261009/G3_EVENT_LOCAL_qualification/QUALIFICATION_REVIEWED.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepared', type=Path)
    parser.add_argument('--reuse-prepared', type=Path, help='With --prepare-assets: reuse completed byte verification, refresh runtime/metadata evidence')
    parser.add_argument('--wait-for-resource', action='store_true')
    parser.add_argument('--allow-low-util-sharing', action='store_true', help='Explicit routing only: utilization <=5%%, at least 55 GiB free on all eight GPUs')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--inspect-only', action='store_true')
    mode.add_argument('--prepare-assets', action='store_true')
    mode.add_argument('--execute', action='store_true')
    options = parser.parse_args()
    require(not options.reuse_prepared or options.prepare_assets, '--reuse-prepared requires --prepare-assets')
    require(Path(sys.executable).absolute() == PYTHON, 'Use original local qwen Python')
    args, ns = original_plan(options.original_controller)
    run = args.output/GROUP
    options.output = options.output.resolve()
    require(not any(options.output.is_relative_to(p.resolve()) or p.resolve().is_relative_to(options.output)
                    for p in (args.output, ROOT, options.destination_qualification.parent, args.resume_origin_run)), 'Control output overlaps preserved artifacts')
    write = options.prepare_assets or options.execute
    if write: options.output.mkdir(parents=True, exist_ok=False)
    def state(status, **values):
        if write: ns['atomic_json'](options.output/'STATE.json', {'status': status, 'unix': time.time(), **values})
    def append(record):
        with (options.output/'executed_commands.jsonl').open('a') as stream: stream.write(json.dumps(record)+'\n')
    try:
        identity, registration, q = read(run/'identity.json'), read(args.output/'REGISTRATION.json'), read(options.destination_qualification)
        require(identity['identity'] == IDENTITY and identity['training_source_sha'] == registration['training_source_sha'] == SHA
                and registration['groups'] == args.groups and registration['host'] == args.host
                and (identity['dataset'], identity['scope'], identity['training_seed'], identity['world_size'], identity['scenes_per_epoch'],
                     identity['schedule_horizon'], identity['schedule_warmup'], identity['effective_global_batch'], identity['microbatch'], identity['FM_repeat'])
                    == ('navsim', 'formal', 42, 8, 101592, 100000, 5000, 32, 4, 8)
                and identity['config']['framework']['action_model']['num_inference_timesteps'] == 10, 'Foreign original scientific contract')
        require(q['passed'] and q['host'] == options.destination_host
                and q['execution_source_sha256'] == identity['training_source_tree_sha256'], 'Destination environment evidence mismatch')
        if q.get('schema') == 'NAV_migration_environment_readiness_v1':
            require(q['environment_only'] and q['source_sha'] == SHA and q['source_clean'] and q['config_matches']
                    and not q['GPU_training_throughput_qualified'] and q['evidence_files_sha256'], 'Invalid independent destination environment evidence')
        else:
            require(q['group'] == 'G3_EVENT_LOCAL' and q['deployment_probe_passed'] and q['current_only_prefetch_passed'], 'Invalid original NU destination evidence')
        require(read(args.cache/'identity.json')['identity'] == identity['cache_identity']
                and read(args.dino_root/'identity.json')['identity'] == args.dino_identity == identity['DINO_identity']
                and read(CURRENT/'identity.json') == identity['current_and_GT_identity']
                and args.geometry_identity == identity['shared_geometry_identity']
                and registration['horizon'] == identity['schedule_horizon'] and registration['global_batch'] == 32
                and registration['microbatch'] == 4 and registration['FM_repeat'] == 8
                and registration['development_updates'] == identity['observation_plan'] == list(TARGETS), 'Original asset/registration contract changed')
        references = {str(p): digest(p) for p in (options.original_controller, args.output/'REGISTRATION.json', run/'identity.json',
            options.destination_qualification, args.resume_origin_run/'identity.json', args.resume_origin_run/'checkpoints/latest',
            args.resume_origin_run/'optimizer_groups.json', ARTIFACTS/'FORMAL_RESOURCE_REGISTRATION_v1.json')}
        if 'raw_qualification_file' in q:
            references.update({q['raw_qualification_file']: q['raw_qualification_sha256'], **q['deployment_probe_files_sha256']})
        references.update(q.get('evidence_files_sha256', {}))
        for name in ('QUEUE_FAILURE.json',):
            if (args.output/name).exists(): references[str(args.output/name)] = digest(args.output/name)
        origin = identity['execution_upgrade_origin']
        origin_complete = args.resume_origin_run/'checkpoints'/origin['tag']/'COMPLETE.json'
        parent = read(origin_complete)
        require(str(args.resume_origin_run) == origin['run'] and (args.resume_origin_run/'checkpoints/latest').read_text().strip() == origin['tag']
                and parent['file_sha256'] == origin['file_sha256'] and parent['identity'] == origin['identity']
                and parent['completed'] == origin['completed'] and parent['status'] == 'PAUSED', 'Historical ancestry changed')
        references[str(origin_complete)] = digest(origin_complete)
        for name, sha in references.items(): require(digest(name) == sha, 'Preserved evidence changed: '+name)
        metadata = {'labels': (args.cache, ('identity.json', 'COMPLETE.json', 'index.json')),
            'dino': (args.dino_root, ('identity.json', 'COMPLETE.json', 'SOURCE_CHUNK_HASHES.json')),
            'current': (CURRENT, ('identity.json', 'index.json', 'ego_identity.json', 'COMPLETE.json', 'local_stage.json', 'local_image_hashes.json'))}
        shared = {str(args.geometry): identity['initialization_assets']['shared_geometry']['sha256'],
            str(args.execution_acceptance): identity['execution_acceptance']['sha256'],
            identity['config']['structured_world']['imagenet_checkpoint']: identity['initialization_assets']['ImageNet_R50_sha256']}
        uuids = q['physical_gpu_UUIDs']
        require(uuids == sorted(set(uuids)) and len(uuids) == 8, 'Eight destination UUIDs required')
        expected = {'execution_source_sha256': identity['training_source_tree_sha256'], 'physical_gpu_UUIDs': uuids,
            'asset_metadata_sha256': {name: {f: digest(p/f) for f in files} for name, (p, files) in metadata.items()},
            'shared_hashes': shared, 'shared_paths': [str(p) for p in (args.dev_inputs/'COMPLETE.json', args.dev_labels/'COMPLETE.json',
                args.dev_index, args.devkit, run/'checkpoints/latest')]}
        required_MiB = math.ceil((read(ARTIFACTS/'FORMAL_RESOURCE_REGISTRATION_v1.json')['NAVSIM_profiles'][GROUP]['rank0_peak_GiB']+12)*1024)
        if options.allow_low_util_sharing: required_MiB = max(required_MiB, 55*1024)
        locks = ARTIFACTS/'physical_gpu_locks'
        lock_paths = [locks/('queue_'+hashlib.sha256(str(run.resolve()).encode()).hexdigest()+'.lock'),
                      locks/(hashlib.sha256(json.dumps(uuids).encode()).hexdigest()+'.lock')]
        manifest = {'kind': 'NAV_G1_routing_only_migration', 'training_evaluation_source_sha': SHA,
            'controller_code_sha256': digest(__file__), 'native_identity': IDENTITY, 'native_checkpoint': latest(run),
            'original_controller': str(options.original_controller), 'original_registration': registration, 'original_run_root': str(run),
            'destination_host': options.destination_host, 'destination_environment_qualification': str(options.destination_qualification),
            'preserved_files_sha256': references, 'expected': expected, 'locks': list(map(str, lock_paths)), 'required_free_MiB': required_MiB,
            'allow_low_util_sharing': options.allow_low_util_sharing,
            'limits': 'Destination evidence proves environment/hardware only. Original NAV memory profile supplies reserve; target NAV GPU throughput is unmeasured. Original NVMe RGB path and staging bytes retained.',
            'plan': [{'stage': s, 'command': c, 'gpu': g, 'complete': str(p) if p else None,
                'skip': latest(run)['completed'] >= t if p is None else p.exists()} for t in TARGETS for s, c, g, p in commands(args, ns, t)]}
        if write: ns['atomic_json'](options.output/'REGISTRATION.json', manifest)
        if not write:
            manifest['readiness'] = probe(args, options, expected)
            manifest['resources_ready'] = resource_ready(manifest['readiness']['GPU_snapshot'], required_MiB, options.allow_low_util_sharing)
            print(json.dumps(manifest, indent=2)); return
        if options.prepare_assets:
            state('PREPARING_CPU_ASSETS', native_checkpoint=latest(run))
            if options.reuse_prepared:
                old = read(options.reuse_prepared)
                old_report = old['report']
                require(old['passed'] and old['manifest']['expected'] == expected
                        and old['manifest']['training_evaluation_source_sha'] == SHA
                        and old['manifest']['destination_host'] == options.destination_host
                        and old['manifest']['native_checkpoint'] == manifest['native_checkpoint']
                        and old_report['native_checkpoint_files_sha256'] == read(run/'checkpoints'/latest(run)['tag']/'COMPLETE.json')['file_sha256']
                        and old_report['CPU_prefetch']['passed'] and old_report['CPU_prefetch']['scenes'] == 64
                        and old_report['image_payload_verification']['all_SHA256_passed']
                        and old_report['image_payload_verification']['images'] == 304776,
                        'Completed asset/native preparation is foreign or stale')
                for name, sha in old['manifest']['preserved_files_sha256'].items():
                    require(digest(name) == sha, 'Reused evidence changed: '+name)
                for name in ('labels', 'dino'):
                    payload = old_report['assets'][name]['payload_verification']
                    require(payload['all_SHA256_passed'] and payload['scenes'] == 101592, 'Incomplete reused payload proof: '+name)
                fresh = probe(args, options, expected)
                for name, files in expected['asset_metadata_sha256'].items():
                    require(fresh['assets'][name].get('metadata_sha256') == files, 'Reused NVMe metadata changed: '+name)
                require(fresh['images_present'] and latest(run) == manifest['native_checkpoint'], 'Reused residency/native progress changed')
                reused = {str(options.reuse_prepared.resolve()): digest(options.reuse_prepared)}
                report = {**old_report, **fresh, 'assets': {name: {**old_report['assets'][name], **fresh['assets'][name]}
                    for name in fresh['assets']}, 'reused_full_payload_and_native_evidence': reused,
                    'fresh_full_payload_hashes_performed': False}
                ns['atomic_json'](options.output/'PREPARATION_COMPLETE.json', {'passed': True, 'manifest': manifest, 'report': report,
                    'reused_evidence_files_sha256': reused, 'GPU_training_started': False, 'GPU_training_throughput_qualified': False,
                    'limits': manifest['limits']+' Full payload/native/CPU equivalence proof is reused from the unchanged referenced certificate; runtime imports and metadata are fresh.'})
                append({'stage': 'reuse_verified_assets', 'reused_evidence_files_sha256': reused,
                        'fresh_runtime_imports_and_metadata_passed': True, 'payload_copy_or_rehash_performed': False})
                state('CPU_ASSETS_AND_NATIVE_PREPARATION_COMPLETE', native_checkpoint=latest(run), GPU_training_started=False)
                return
            def copy(component):
                name, (source, _) = component
                command = ['rsync', '-a', '--mkpath', '--ignore-existing',
                    '--partial-dir=.navsim-migration-partial', '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=10', str(source)+'/', options.destination_host+':'+str(source)+'/']
                authorize(socket.gethostname().removesuffix('-worker-0')); authorize(options.destination_host)
                with (options.output/(name+'_copy.log')).open('a') as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
                # Dry-run byte comparison never updates an existing destination.
                verify = ['rsync', '-rnc', '--out-format=%n',
                          '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=10', str(source)+'/', options.destination_host+':'+str(source)+'/']
                changed = subprocess.check_output(verify, text=True)
                require(not changed.strip(), 'Replica differs; retained without overwrite: '+name)
                return {'stage': 'copy_'+name, 'command': command, 'verification_command': verify, 'byte_comparison_passed': True}
            with ThreadPoolExecutor(max_workers=4) as pool:
                for record in pool.map(copy, [*metadata.items(), ('images', (IMAGES, ()))]): append(record)
            report = probe(args, options, expected, full=True)
            require(report['checkpoint_tag'] == latest(run)['tag'] == manifest['native_checkpoint']['tag'], 'Native checkpoint advanced during preparation')
            ns['atomic_json'](options.output/'PREPARATION_COMPLETE.json', {'passed': True, 'manifest': manifest, 'report': report,
                'GPU_training_started': False, 'GPU_training_throughput_qualified': False, 'limits': manifest['limits']})
            state('CPU_ASSETS_AND_NATIVE_PREPARATION_COMPLETE', native_checkpoint=latest(run), GPU_training_started=False)
            return
        require(options.prepared, '--execute requires reviewed preparation evidence via --prepared')
        prepared = read(options.prepared)
        for name, sha in prepared.get('reused_evidence_files_sha256', {}).items():
            require(digest(name) == sha, 'Reused preparation evidence changed: '+name)
        require(prepared['passed'] and prepared['manifest']['expected'] == expected
                and prepared['manifest']['controller_code_sha256'] == manifest['controller_code_sha256']
                and prepared['manifest']['native_identity'] == IDENTITY and prepared['manifest']['destination_host'] == options.destination_host
                and prepared['manifest']['preserved_files_sha256'] == references
                and prepared['report'].get('image_payload_verification', {}).get('all_SHA256_passed')
                and prepared['report']['image_payload_verification']['images'] == 304776
                and prepared['report'].get('CPU_import_readiness', {}).get('passed')
                and prepared['report']['native_checkpoint_files_sha256'] == read(run/'checkpoints'/latest(run)['tag']/'COMPLETE.json')['file_sha256'], 'Preparation is stale/foreign')
        watcher = ARTIFACTS/'recovery_dispatch_20261010/navsim_G1/LAUNCHED_WATCHER.json'
        if watcher.exists():
            pid = read(watcher)['pid']; cmdline = Path(f'/proc/{pid}/cmdline')
            require(not (cmdline.exists() and b'wait_and_resume.py' in cmdline.read_bytes()), 'Original-host NAV watcher is active; hold administratively before migration')
        def wait(reason, stage):
            state('WAITING_FOR_RESOURCE', reason=reason, stage=stage, native_checkpoint=latest(run))
            if not options.wait_for_resource: raise RuntimeError(reason)
            time.sleep(30)
        with ExitStack() as held:
            while True:
                authorize(options.destination_host)
                with ExitStack() as attempt:
                    try:
                        for p in lock_paths:
                            lock = attempt.enter_context(p.open('a')); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        held.enter_context(attempt.pop_all()); break
                    except BlockingIOError: pass
                wait('Queue or physical GPU lock is held', 'acquire_locks')
            for target in TARGETS:
                for stage, command, gpu, complete in commands(args, ns, target):
                    before = latest(run)
                    if complete is None and before['completed'] >= target: continue
                    if complete is not None:
                        c = read(run/'checkpoints'/f'milestone_{target:07d}'/'COMPLETE.json')
                        require(c['identity'] == IDENTITY and c['completed'] == target, 'Wrong development boundary')
                        if complete.exists(): continue
                    while True:
                        clean_source()
                        for name, sha in references.items(): require(digest(name) == sha, 'Evidence changed before dispatch: '+name)
                        snapshot = probe(args, options, expected)['GPU_snapshot']
                        if resource_ready(snapshot, required_MiB, options.allow_low_util_sharing): break
                        wait('Destination is occupied or lacks memory reserve', stage)
                    authorize(options.destination_host if gpu else socket.gethostname().removesuffix('-worker-0'))
                    require(latest(run) == before and not (run/'STOP_AFTER_CHECKPOINT').exists(), 'Native progress/stop request changed before dispatch')
                    record = {'stage': stage, 'command': command, 'host': options.destination_host if gpu else 'local CPU',
                              'native_checkpoint': before, 'GPU_snapshot': snapshot, 'started_unix': time.time()}
                    state('RUNNING', **record); append(record)
                    with (options.output/(stage+'.log')).open('a') as log:
                        result = subprocess.run(route(options.destination_host, command) if gpu else command, cwd=ROOT,
                            env={**os.environ, 'OMP_NUM_THREADS': '2', 'TOKENIZERS_PARALLELISM': 'false', 'PYTHONDONTWRITEBYTECODE': '1'}, stdout=log, stderr=subprocess.STDOUT)
                    require(result.returncode == 0, f'{stage} failed with exit {result.returncode}')
                    if complete is None:
                        boundary = read(run/'RESUME_BOUNDARY_VERIFIED.json')
                        require(boundary['passed'] and boundary['identity'] == IDENTITY and boundary['completed'] == before['completed']
                                and latest(run)['completed'] == target, 'Native resume boundary/progress verification failed')
                        record['resume_boundary'] = {'file': str(run/'RESUME_BOUNDARY_VERIFIED.json'), 'sha256': digest(run/'RESUME_BOUNDARY_VERIFIED.json'),
                            'completed_reference': before['completed'], 'report': boundary}
                    else: require(complete.is_file(), 'Development stage lacks completion marker')
                    append({**record, 'status': 'PASSED', 'ended_unix': time.time()})
            state('NAVSIM_G1_TRAIN_AND_DEV_COMPLETE', native_checkpoint=latest(run), final_Navtest_pending=True)
    except BaseException as error:
        state('FAILED', error=repr(error), checkpoints_preserved=True, unrelated_processes_stopped=False)
        raise


if __name__ == '__main__':
    main()
