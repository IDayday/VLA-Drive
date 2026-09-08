"""Independent formal admission/launch guard; never changes the frozen training graph."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import gzip
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'tools'))
import torch
from scan_planreg_v2_inputs import sha, signature, preprocessing_identity, SCAN_SCHEMA
from navsim.agents.EpisodeDrive.planreg_v2 import NORMALIZER_SCHEMA, SHARED_INIT_SCHEMA, ARCHITECTURE_VERSION
from navsim.agents.EpisodeDrive.planreg_v2.runtime import load_config, source_fingerprint, validate_formal
from navsim.agents.EpisodeDrive.planreg_v2.initialization import initialization_identity, tensor_manifest


def read(path): return json.loads(Path(path).read_text())


def helper_identity():
    return {p.name:sha(p) for p in (ROOT/'tools').glob('*planreg_v2*.py') if not p.name.startswith('test_')}


def lock_profile(spec_path, inventory_path, output):
    from navsim.agents.EpisodeDrive.planreg_v2.runtime import validate_profile_artifact
    spec=read(spec_path);root=Path(spec['run_output'])
    report=read(root/'validation.json');metadata=read(root/'run_metadata.json');cfg=read(root/'resolved_config.json')
    validate_profile_artifact(report,metadata,cfg)
    if spec['mode']!='profile' or spec.get('preflight_helper_sha256')!=helper_identity():
        raise ValueError('Only the executed current launcher/helper snapshot can issue its new lock')
    if read(Path(spec['log_dir'])/'coordinator.json')['status']!='COMPLETED':raise ValueError('Not all profile nodes finished')
    world=sum(len(n['gpus']) for n in spec['nodes'])
    if (world,spec['microbatch'],spec['accumulate'],spec['workers'])!=(metadata['world_size'],metadata['microbatch'],metadata['accumulate'],metadata['num_workers']):
        raise ValueError('Exact measured layout mismatch')
    if world*spec['microbatch']*spec['accumulate']!=128:raise ValueError('Actual GB128 required')
    inventory=read(inventory_path);nodes=[]
    for node in spec['nodes']:
        actual=inventory[node['host']]
        if len(node['gpus'])!=len(actual['hardware']) or actual['external_gpu_processes']:
            raise ValueError('Authorized visible GPU inventory mismatch')
        nodes.append(dict(node,hardware=actual['hardware']))
    lock=dict(architecture_version=cfg['architecture_version'],passed=True,recipe_version=cfg['recipe_version'],
        source_fingerprint_sha256=metadata['source_fingerprint']['sha256'],preflight_helper_sha256=helper_identity(),
        hardware=metadata['hardware'],nodes=nodes,world_size=world,gpus_per_node=None,
        profile_only=True,profile_no_external_gpu_work=True,global_batch=128,microbatch=spec['microbatch'],
        accumulate=spec['accumulate'],workers=spec['workers'],max_tiles=report['profile']['max_tiles'],
        code_commit=metadata['git_commit'],report_path=str(root/'validation.json'),source_sha256=sha(root/'validation.json'),
        profile_manifest_sha256=metadata['manifest_sha256'],profile_range_sha256=sha(spec['profile_range']),
        profile_spec_sha256=sha(spec_path),hardware_inventory_sha256=sha(inventory_path),
        peak_allocated_gib=report['peak_allocated_gib'],profile=report['profile'],
        precision='BF16 activations/frozen VLM; FP32 trainable/AdamW/EMA master',
        schedule_total_steps=metadata['schedule_total_steps'],gradient_checkpointing=cfg['gradient_checkpointing'])
    if Path(output).exists():raise FileExistsError('Never overwrite an existing measured layout')
    Path(output).write_text(json.dumps(lock,indent=2));print(json.dumps(lock,indent=2))


def flat_layout(path):
    """Resolve the existing evidence wrapper, never pass it to the training launcher."""
    value = read(path)
    if 'evidence' in value:
        actual = Path(value['artifact_path'])
        if sha(actual) != value['artifact_sha256'] or read(actual) != value['evidence']:
            raise ValueError('Wrapped layout artifact identity mismatch')
        path, value = actual, read(actual)
    if 'global_batch' not in value or 'evidence' in value: raise ValueError('Flat measured layout required')
    if not value.get('report_path') or sha(value['report_path'])!=value.get('source_sha256'):
        raise ValueError('Missing or changed actual measured profile report')
    return value, str(Path(path).resolve())


def validate_vlm_pair(audit, variant, vlm_path):
    if audit.get('pair',{}).get('formal_pair_compatible') is not True:
        raise ValueError('Architecture/tokenizer VLM pair audit required')
    record=audit['base' if variant=='base' else 'vqa'];root=Path(vlm_path)
    if root.resolve()!=Path(record['checkpoint_path']).resolve() or record.get('agent_checkpoint_loaded') is not False:
        raise ValueError('VLM-only initialization source differs from audit')
    files=dict(record['weight_file_sha256'],**record['tokenizer_file_sha256'])
    files['config.json']=record['config_sha256']
    for name,expected in files.items():
        if sha(root/name)!=expected:raise ValueError('VLM/tokenizer file changed since pair audit: '+name)
    return record['checkpoint_sha256']


def validate_shared(artifact, cfg):
    if set(artifact) != {'schema','identity','tensors','trainable_state','seed','config'}:
        raise ValueError('Untrained parameter bank required, not a model/optimizer checkpoint')
    if artifact['schema'] != SHARED_INIT_SCHEMA or artifact['identity'] != initialization_identity(cfg):
        raise ValueError('Old shared init/schema/register initialization is forbidden')
    state = artifact['trainable_state']
    if tensor_manifest(state) != artifact['tensors'] or any(t.dtype != torch.float32 for t in state.values()):
        raise ValueError('Shared bank dtype/shape/hash mismatch')
    registers = state['backbone.planning_register_adapter.planning_registers']
    if not .016 < float(registers.std()) < .024: raise ValueError('Register bank is not the declared std0.02 initialization')
    # The recorded creation hash is also bound by the actual paired initialization audit.
    zero_b = [t for n,t in state.items() if ('lora' in n.lower()) and ('_b.' in n.lower() or '.lora_b.' in n.lower())]
    if not zero_b or any(torch.count_nonzero(t).item() for t in zero_b):
        raise ValueError('Fresh zero-B LoRA bank required; trained smoke/profile weights forbidden')


def validate_full_statistics(manifest, normalizer):
    m = normalizer['metadata']
    if m.get('schema') != NORMALIZER_SCHEMA or m.get('mode') != 'stepwise_zscore':
        raise ValueError('New full stepwise [8,3] statistics required')
    tokens = [r['token'] for r in manifest['records']]
    token_hash = hashlib.sha256('\n'.join(sorted(set(tokens))).encode()).hexdigest()
    if len(tokens) != 103288 or len(set(tokens)) != 103288 or token_hash != manifest['token_sha256']:
        raise ValueError('Exact 103288 unique authorized training scenes required')
    if (m.get('count'),m.get('split'),m.get('token_sha256')) != (103288,'trainval_final_fit',token_hash):
        raise ValueError('Missing full statistics or smoke statistics used')
    for key in ('raw_gt_sha256','statistics_contract'):
        if not m.get(key) or m[key] != manifest.get(key): raise ValueError('Original GT/statistics contract mismatch: '+key)
    for key in ('mean','std'):
        value = torch.tensor(normalizer[key])
        if value.shape != (8,3) or not torch.isfinite(value).all(): raise ValueError('Invalid [8,3] statistics '+key)


def validate_input_range(summary, manifest_hash, layout, profile_range):
    if summary.get('schema') != SCAN_SCHEMA or summary.get('status') != 'PASS' or summary.get('audited_count') != 103288:
        raise ValueError('Missing complete successful input summary')
    if summary.get('manifest_sha256') != manifest_hash or summary.get('preprocessing') != preprocessing_identity():
        raise ValueError('Stale manifest/preprocessing input summary')
    if not summary.get('raw_gt_statistics_recomputed_equal') or summary.get('errors'):
        raise ValueError('Full supervision audit did not pass')
    if summary['input_ranges']['max_tiles'] > layout['max_tiles']:
        raise ValueError('Full-data max_tiles exceeds the measured layout; bounded worst-input profile required')
    if profile_range['source_fingerprint_sha256'] != source_fingerprint()['sha256']:
        raise ValueError('Profile token range belongs to different computation')
    # Defined before observing full-data lengths: <=32 extra prefix tokens (<2%)
    # is not a significant sequence expansion. Beyond this, require actual profile.
    old = profile_range['max_prefix_tokens']; new = summary['input_ranges']['max_prefix_tokens']
    if new > min(old+32, int(old*1.02)):
        raise ValueError('Significantly longer full-data prompts require a worst-input profile')


def validate_hardware(actual, expected, free_required_gib=72):
    if len(actual['hardware']) != len(expected['hardware']): raise ValueError('Visible GPU count differs from measured hardware')
    for got,want in zip(actual['hardware'],expected['hardware']):
        if (got['name'],got['memory_bytes']) != (want['name'],want['memory_bytes']):
            raise ValueError('Current GPU hardware differs from measured layout')
        if want.get('uuid') and got.get('uuid')!=want['uuid']:
            raise ValueError('Physical GPU identity differs from measured authorized device')
        if got['free_bytes'] < free_required_gib*2**30: raise ValueError('Insufficient free GPU memory; do not evict other work')
    if actual['external_gpu_processes']: raise ValueError('Other GPU tasks are present; never terminate them for preflight')


def hardware_snapshot():
    from planreg_v2_gpu_guard import visible_gpu_processes
    external = sorted(set(visible_gpu_processes())-{os.getpid()})
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True)
    mapping={uuid.strip():int(index) for index,uuid in (row.split(',') for row in rows.splitlines())}
    devices=[]
    for i in range(torch.cuda.device_count()):
        prop=torch.cuda.get_device_properties(i)
        uuid='GPU-'+str(prop.uuid)
        if uuid not in mapping:raise ValueError('CUDA physical device cannot be mapped to an authorized NVIDIA index')
        free,_=torch.cuda.mem_get_info(i)
        devices.append(dict(index=i,physical_index=mapping[uuid],uuid=uuid,name=prop.name,memory_bytes=prop.total_memory,free_bytes=free))
    return dict(hostname=socket.gethostname(),hardware=devices,external_gpu_processes=external,
        python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,source_fingerprint_sha256=source_fingerprint()['sha256'])


def inventory_check(path, expected_hash):
    if sha(path) != expected_hash: raise ValueError('Input file inventory hash mismatch')
    with gzip.open(path,'rt') as f: inventory=json.load(f)
    def check(item):
        name,expected=item
        try: return None if signature(name)==expected else name
        except OSError:return name
    with ThreadPoolExecutor(24) as pool:
        changed=[p for p in pool.map(check,inventory.items(),chunksize=128) if p]
    if changed: raise ValueError('Input summary stale: file size/mtime changed: '+str(changed[:10]))
    return len(inventory)


def initialization_audit(base_config, vqa_config, output):
    from navsim.agents.EpisodeDrive.planreg_v2.agent import PlanRegV2Agent
    from navsim.agents.EpisodeDrive.planreg_v2.ema import visual_parameters
    rows=[]
    for config in (base_config,vqa_config):
        cfg=load_config(config); bank=torch.load(cfg['shared_init_path'],map_location='cpu',weights_only=False)
        validate_shared(bank,cfg)
        agent=PlanRegV2Agent(cfg,'cpu'); state=agent.trainable_state()
        if set(state)!=set(bank['trainable_state']) or any(not torch.equal(t,bank['trainable_state'][n]) for n,t in state.items()):
            raise ValueError('Loaded common trainable bank was changed by initialization')
        if int(agent.optimizer_updates)!=0 or int(agent.ema_teacher.updates)!=0: raise ValueError('Initialization contains training progress')
        source=visual_parameters(agent.backbone)
        for i,n in enumerate(agent.ema_teacher.names):
            if not torch.equal(getattr(agent.ema_teacher,'master_%04d'%i),source[n]):
                raise ValueError('Teacher was not initialized after shared bank')
        opt,scheduler=agent.get_optimizers(); assert len(opt.state)==0
        rows.append(dict(variant=cfg['variant'],config=cfg,hashes=tensor_manifest(state),
            optimizer_groups=agent.optimizer_summary,initial_applied_lrs=scheduler.get_last_lr(),
            optimizer_updates=0,ema_updates=0,adam_state_empty=True,teacher_after_bank_equal=True,
            trainable_count=sum(t.numel() for t in state.values()),register_std=float(agent.backbone.planning_register_adapter.planning_registers.std())))
        del agent,state,opt,scheduler,bank;gc.collect()
    if rows[0]['hashes']!=rows[1]['hashes'] or rows[0]['optimizer_groups']!=rows[1]['optimizer_groups']:
        raise ValueError('Paired initialization or optimizer groups differ')
    report=dict(status='PASS',shared_init_sha256=sha(rows[0]['config']['shared_init_path']),
        normalizer_sha256=sha(rows[0]['config']['normalizer_path']),base_vqa_bitwise_equal=True,
        source_fingerprint_sha256=source_fingerprint()['sha256'],preflight_sha256=sha(__file__),records=rows)
    Path(output).write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))


def validate_request(request, hardware_by_node, check_files=True):
    cfg=load_config(request['config']); manifest=read(request['manifest']); layout,path=flat_layout(request['layout'])
    normalizer=read(cfg['normalizer_path']); summary=read(request['shape_summary']); init=read(request['initialization_audit'])
    validate_full_statistics(manifest,normalizer)
    validate_formal(cfg,manifest,layout)  # Existing production version/statistics/recipe checks.
    if cfg.get('checkpoint_path') is not None or cfg.get('stage1_checkpoint_path') is not None or request.get('warm_start'):
        raise ValueError('Formal main run must start from VLM plus untrained shared bank, never agent warm-start')
    if (cfg['global_batch'],cfg['total_steps'],cfg['epochs']) != (128,21789,27): raise ValueError('Formal budget is locked at GB128/807/21789')
    if cfg['variant'] not in ('base','driving_vqa') or cfg.get('world_model_enabled') is not True or cfg.get('wm_objective','tf_and_ro')!='tf_and_ro':
        raise ValueError('Only one explicit full Base/VQA TF+RO run is authorized')
    if cfg.get('motion_mode')!='gt_log' or cfg.get('scene_memory_mode')!='per_tile_register_memory' or cfg.get('gradient_checkpointing') is not True:
        raise ValueError('Computation differs from approved formal recipe')
    validate_input_range(summary,sha(request['manifest']),layout,read(request['profile_range']))
    if layout.get('profile_range_sha256')!=sha(request['profile_range']):raise ValueError('Measured input range identity changed')
    if summary.get('normalizer_sha256')!=sha(cfg['normalizer_path']):raise ValueError('Input/statistics summary identity changed')
    authorization=read(request['token_authorization'])
    if authorization.get('token_sha256')!=manifest['token_sha256']:
        raise ValueError('Full token set differs from authorized trainval list')
    if (authorization.get('count'),authorization.get('navtest_token_overlap'),authorization.get('navtest_log_overlap'))!=(103288,0,0):
        raise ValueError('Complete training-only token authorization required')
    if sha(authorization['source'])!=authorization['source_sha256']:
        raise ValueError('Authorized source token list changed')
    vlm_sha=validate_vlm_pair(read(request['vlm_pair_audit']),cfg['variant'],cfg['vlm_path'])
    bank=torch.load(cfg['shared_init_path'],map_location='cpu',weights_only=False); validate_shared(bank,cfg);del bank
    if init.get('status')!='PASS' or init.get('shared_init_sha256')!=sha(cfg['shared_init_path']) or init.get('normalizer_sha256')!=sha(cfg['normalizer_path']):
        raise ValueError('Initialization audit identity is missing/stale')
    if init['source_fingerprint_sha256']!=source_fingerprint()['sha256'] or not init.get('base_vqa_bitwise_equal'):
        raise ValueError('Actual Base/VQA initialization audit unavailable')
    record=next(r for r in init['records'] if r['variant']==cfg['variant'])
    if record['config']!=cfg: raise ValueError('Audited initialization resolved config changed')
    nodes=layout.get('nodes') or [dict(host='local',gpus=list(range(layout['gpus_per_node'])),hardware=layout['hardware'])]
    if sum(len(n['gpus']) for n in nodes)!=layout['world_size'] or layout['world_size']*layout['microbatch']*layout['accumulate']!=128:
        raise ValueError('Incorrect actual multi-node global batch')
    for node in nodes:
        current=hardware_by_node[node['host']];validate_hardware(current,node)
        if current['source_fingerprint_sha256']!=layout['source_fingerprint_sha256']: raise ValueError('Remote computation source differs')
        if current['torch']!=torch.__version__ or current['cuda']!=torch.version.cuda: raise ValueError('Remote verified environment differs')
    if request['workers']!=layout['workers']:raise ValueError('DataLoader worker count differs from measured layout')
    if layout.get('preflight_helper_sha256')!=helper_identity():
        raise ValueError('Preflight/launcher safety helper identity changed since layout binding')
    output=Path(request['run_output'])
    if request.get('resume_checkpoint'):
        if Path(request['resume_checkpoint']).resolve().parent!=output.resolve():raise ValueError('Resume must be inside this same formal run')
        metadata=read(output/'run_metadata.json')
        contract=dict(seed=request.get('seed',0),microbatch=layout['microbatch'],accumulate=layout['accumulate'],workers=layout['workers'],bounded_run=False,profile_only=False)
        if metadata['run_contract']!=contract or metadata['world_size']!=layout['world_size'] or metadata['total_steps']!=21789:
            raise ValueError('Resume requires identical formal layout and full schedule, never smoke/profile progress')
        if read(output/'resolved_config.json')!=cfg or metadata['manifest_sha256']!=sha(request['manifest']):
            raise ValueError('Resume config/data contract differs')
        if not Path(request['resume_checkpoint']).is_file():raise FileNotFoundError('Explicit resume checkpoint unavailable')
    elif output.exists(): raise FileExistsError('New formal output directory required; never reuse smoke/profile or auto-resume')
    inventory_count=inventory_check(summary['file_inventory_path'],summary['file_inventory_sha256']) if check_files else None
    return dict(READY_FOR_FORMAL_TRAINING=True,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_fingerprint_sha256=source_fingerprint()['sha256'],preflight_sha256=sha(__file__),algorithm_changed=False,
        variant=cfg['variant'],vlm_path=cfg['vlm_path'],vlm_sha256=vlm_sha,manifest=request['manifest'],manifest_sha256=sha(request['manifest']),
        normalizer=cfg['normalizer_path'],normalizer_sha256=sha(cfg['normalizer_path']),shared_init=cfg['shared_init_path'],shared_init_sha256=sha(cfg['shared_init_path']),
        shape_summary_sha256=sha(request['shape_summary']),layout_path=path,layout_sha256=sha(path),nodes=nodes,
        coverage=summary['valid_rates'],max_tiles=summary['input_ranges']['max_tiles'],profile_max_tiles=layout['max_tiles'],
        prefix_range=[summary['input_ranges']['min_prefix_tokens'],summary['input_ranges']['max_prefix_tokens']],
        world_size=layout['world_size'],microbatch=layout['microbatch'],accumulate=layout['accumulate'],global_batch=128,
        steps_per_epoch=807,total_steps=21789,padded_exposures_per_epoch=8,optimizer_groups=record['optimizer_groups'],
        new_output=str(output),inventory_files_verified=inventory_count,training_started=False,hardware=hardware_by_node)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--hardware',action='store_true');p.add_argument('--request')
    p.add_argument('--lock-profile');p.add_argument('--inventory')
    p.add_argument('--initialize-pair',action='store_true');p.add_argument('--base-config');p.add_argument('--vqa-config');p.add_argument('--output')
    a=p.parse_args()
    if a.hardware: print(json.dumps(hardware_snapshot()));return
    if a.lock_profile:lock_profile(a.lock_profile,a.inventory,a.output);return
    if a.initialize_pair:initialization_audit(a.base_config,a.vqa_config,a.output);return
    request=read(a.request)
    for k,v in request['environment'].items(): os.environ[k]=v
    if os.getenv('V2_CONFIG'): request['config']=os.environ['V2_CONFIG']
    from planreg_v2_cluster import inspect_nodes, launch_formal
    report={}
    try:
        layout,_=flat_layout(request['layout']); actual=inspect_nodes(layout)
        report=validate_request(request,actual)
    except Exception as e:
        report=dict(READY_FOR_FORMAL_TRAINING=False,training_started=False,error=type(e).__name__+': '+str(e))
    dest=Path(a.output or request['admission_log']);dest.write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('hardware','optimizer_groups','nodes')},indent=2))
    if report['READY_FOR_FORMAL_TRAINING']:
        print('ACTUAL_OPTIMIZER_PEAKS',json.dumps({r['name']:r['resolved_peak'] for r in report['optimizer_groups']}))
        # This must also work in a fresh shell, before this module imports NAVSIM.
        bootstrap=['env','START_FORMAL_AFTER_PREFLIGHT=1','PYTHONNOUSERSITE=1',
            'PYTHONPATH='+str(ROOT)+':/mnt/project/DriveVLA-M0-env/lib/python3.9/site-packages',
            'OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1']
        command=shlex.join(bootstrap+[sys.executable,str(Path(__file__).resolve()),'--request',str(Path(a.request).resolve())])
        print('FORMAL_COMMAND',command,flush=True)
        if os.getenv('START_FORMAL_AFTER_PREFLIGHT')=='1':
            launch=launch_formal(request,report);report.update(launch);dest.write_text(json.dumps(report,indent=2));print(json.dumps(launch,indent=2))
        else:print('Training NOT started: START_FORMAL_AFTER_PREFLIGHT is not 1',flush=True)
    else:raise SystemExit(2)


if __name__=='__main__':main()
