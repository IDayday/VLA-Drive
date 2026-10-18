"""Same-request eval-FP32 sparse-readout diagnostic; default metadata only."""
import argparse, copy, gc, hashlib, importlib.util, json, os, pickle, random, subprocess, sys, time, traceback, weakref
from pathlib import Path
HERE=Path(__file__).resolve().parent

def read_pin(pin, maximum=1<<20):
    p=Path(pin['path'])
    if not p.is_absolute() or p.resolve()!=p or not p.is_file() or p.stat().st_size>maximum:raise ValueError('Bounded canonical input required')
    raw=p.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=pin['sha256']:raise ValueError('Changed input '+str(p))
    return json.loads(raw)

def paired_configs(configs):
    values=[]
    for channels in (256,384):
        cfg=copy.deepcopy(configs[str(channels)])
        st=cfg['space_time'];teacher=st['teacher'];ex=cfg['execution']
        if (st['cell']!='S-P2' or st['world_queries_per_view']!=48 or st['sampled_positions']!=96
            or teacher['channels']!=channels or teacher['image_hw']!=[288,384]
            or teacher['output_grid_hw']!=[18,24] or cfg['data']['history_root'] is not None
            or ex['stop_after']!=5000 or ex['max_updates']!=20000 or ex['global_batch']!=32
            or ex['scope']!='FORMAL' or cfg['optimizer']['warmup']!=1000):raise ValueError('Wrong paired spatial recipe')
        teacher.pop('channels');teacher.pop('transform_identity');cfg['runtime'].pop('output_root');values.append(cfg)
    if values[0]!=values[1]:raise ValueError('Unregistered difference beyond channel/output identity')

def preflight(binding):
    if (binding['schema']!='st_teacher_channel_prediction_diagnostic_v1' or binding['mode']!='eval_FP32_TF32_off'
        or binding['max_scenes']!=8 or binding['batch_size']!=1 or set(binding['arms'])!={'256','384'}):raise ValueError('Fixed diagnostic only')
    selection=read_pin(binding['selection']);parent=read_pin(binding['selection_source'])
    expected=sorted(parent['scenes'],key=lambda r:hashlib.sha256((str(binding['selection_seed'])+':'+r['token']).encode()).hexdigest())[:8]
    if selection['split']!='train' or selection['scenes']!=expected or len({r['token'] for r in expected})!=8:raise ValueError('Fixed score-free TRAIN subset differs')
    configs={};regs={}
    for name,a in binding['arms'].items():
        reg=read_pin(a['registration']);cfg=read_pin(a['config']);done=read_pin(a['complete'])
        if (reg['artifacts']['config']!=a['config'] or reg['source_sha']!=binding['runtime']['source_sha']
            or done['state']['completed']!=5000 or done['state']['exposure']!=159992
            or done['state']['tag']!=a['tag'] or done['world_size']!=4
            or str(Path(cfg['runtime']['output_root'])/reg['run_id']/'checkpoints'/a['tag']/'COMPLETE.json')!=a['complete']['path']):raise ValueError('Real matched5000 endpoint required')
        configs[name]=cfg;regs[name]=reg
    paired_configs(configs)
    for key in ('base_binding','data_manifest','teacher_manifest'):
        if regs['256']['artifacts'][key]!=regs['384']['artifacts'][key]:raise ValueError('Shared training dependency changed')
    runtime=Path(binding['runtime']['path'])
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=runtime,text=True).strip()!=binding['runtime']['source_sha'] or subprocess.check_output(['git','status','--porcelain'],cwd=runtime):raise ValueError('Frozen runtime changed')
    for pin in binding['sources'].values():
        p=Path(pin['path'])
        if p.stat().st_size>128<<10 or hashlib.sha256(p.read_bytes()).hexdigest()!=pin['sha256']:raise ValueError('Diagnostic source changed')
    return configs,regs,selection

def runtime_api(binding):
    root=binding['runtime']['path'];sys.path.insert(0,root)
    from tools.st_world_policy import checkpoints
    if Path(checkpoints.__file__).resolve().parents[2]!=Path(root):raise ValueError('Mixed imported runtime')
    return checkpoints

def rng_identity(torch,np):
    value=(random.getstate(),np.random.get_state(),torch.get_rng_state().numpy().tobytes(),
           [s.cpu().numpy().tobytes() for s in torch.cuda.get_rng_state_all()])
    return hashlib.sha256(pickle.dumps(value,protocol=4)).hexdigest()

def predict_current(model,observations,requests):
    import torch
    if model.training or any(p.requires_grad for p in model.parameters()) or not model.base.inference_fp32:raise ValueError('Frozen eval FP32 only')
    before={k:(v.data_ptr(),v._version) for k,v in list(model.named_parameters())+list(model.named_buffers())}
    calls=[];hook=model.sparse_readout.register_forward_hook(lambda _m,_a,out:calls.append(out.detach()))
    try:
        with torch.no_grad():
            world=model.encode_inputs(observations)['W']
            if world.dtype!=torch.float32 or not torch.isfinite(world).all():raise ValueError('Invalid original W')
            view,xy,seconds=requests.query_layout(current=True,future=False,device=world.device)
            with model.base.amp():prediction=model.sparse_readout(world,view,xy,seconds)
        if len(calls)!=1 or calls[0] is not prediction:  # detach may be another Tensor object, compare actual storage.
            if len(calls)!=1 or calls[0].data_ptr()!=prediction.data_ptr():raise ValueError('Expected one actual readout call')
        if prediction.dtype!=torch.float32 or not torch.isfinite(prediction).all():raise ValueError('Invalid FP32 sparse prediction')
        after={k:(v.data_ptr(),v._version) for k,v in list(model.named_parameters())+list(model.named_buffers())}
        if before!=after:raise ValueError('Read-only model mutated')
        batch,views,k=requests.patch_indices.shape
        return prediction.reshape(batch,views,k,-1).detach().cpu().numpy().copy()
    finally:hook.remove()

def run_arm(binding,name,data,requests,checkpoint_api):
    import numpy as np
    import torch
    arm=binding['arms'][name];started=time.monotonic()
    model,record=checkpoint_api.load_world_checkpoint(arm['registration']['path'],arm['tag'],device='cuda:0',strip=False)
    ref=weakref.ref(model)
    try:
        if record['completed']!=5000 or model.st_config.teacher.channels!=int(name):raise ValueError('Wrong loaded arm')
        model.requires_grad_(False);model.eval()
        random.seed(binding['forward_seed']);np.random.seed(binding['forward_seed']);torch.manual_seed(binding['forward_seed']);torch.cuda.manual_seed_all(binding['forward_seed'])
        before=rng_identity(torch,np);pred=[]
        torch.cuda.reset_peak_memory_stats()
        for n in range(len(data)):
            pred.append(predict_current(model,[data[n]],requests.slice(n,n+1)))
        torch.cuda.synchronize();after=rng_identity(torch,np)
        return np.concatenate(pred),dict(checkpoint=record,mode=binding['mode'],rng_before=before,rng_after=after,
            rng_unchanged=before==after,actual_encode_calls=len(data),actual_readout_calls=len(data),
            elapsed_seconds=time.monotonic()-started,gpu_peak_allocated=torch.cuda.max_memory_allocated(),
            gpu_peak_reserved=torch.cuda.max_memory_reserved(),training_forward_called=False,optimizer_created=False)
    finally:
        # An active exception traceback owns its frame/model until propagation.
        # Preserve that primary error; the supervised process exits without arm2.
        failed = sys.exc_info()[0] is not None
        del model;gc.collect();torch.cuda.empty_cache()
        if not failed and ref() is not None:raise RuntimeError('Previous full model remains live; second load forbidden')

def check_deterministic_environment(torch):
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise ValueError('Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting Python')
    if torch.cuda.is_initialized():
        raise ValueError('Diagnostic requires a fresh CUDA process')

def execute(binding,configs,regs,selection):
    # No new reader, provider, index, model forward, optimizer or teacher writer.
    import numpy as np
    import torch
    check_deterministic_environment(torch)
    checkpoint_api=runtime_api(binding)
    from starVLA.dataloader.st_world_observation import STObservationDataset
    from starVLA.dataloader.st_world_targets import TeacherCacheReader
    from starVLA.dataloader.st_world_dataset import projected_teacher_contract
    from starVLA.dataloader.st_world_schema import TeacherFeatureContract
    from starVLA.model.modules.st_world.requests import sample_requests,gather_current
    from tools.st_world_policy.teacher_pca import TeacherPCA
    from tools.st_world_policy.training_config import environment_record
    torch.set_num_threads(2);torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)
    cfg=configs['384'];reg=regs['384'];out=Path(binding['output_root']);out.mkdir(parents=True,exist_ok=False)
    def write(name,value):
        raw=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
        if sum(p.stat().st_size for p in out.iterdir())+len(raw)>binding['proposal_limits']['output_bytes']:raise ValueError('Diagnostic output budget exceeded')
        with (out/name).open('xb') as f:f.write(raw)
    write('INVOCATION.json',{'binding':binding,'mode':'eval_FP32_TF32_off','training_loss_claimed':False,
        'runtime':dict(torch_version=torch.__version__,cuda_version=torch.version.cuda,
            cublas_workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG'],
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32)})
    manifest=read_pin(reg['artifacts']['data_manifest'],64<<20);lookup={r['token']:r for r in manifest['scenes']}
    if manifest['split']!='train' or any(lookup.get(r['token'])!=r for r in selection['scenes']):raise ValueError('Selected rows outside shared actual train manifest')
    del lookup,manifest
    data=STObservationDataset(cfg['data']['root'],binding['selection']['path'])
    teacher_identity=read_pin(reg['artifacts']['teacher_manifest'],192<<20)
    reader=TeacherCacheReader(cfg['data']['teacher_root'],expected_identity=teacher_identity['identity']);del teacher_identity
    root=Path(cfg['data']['pca_path']);pca_receipt=json.loads((root/'PCA.json').read_text())
    pca={d:TeacherPCA.load(root,dimension=d,expected_identity=pca_receipt['identity'],expected_teacher_contract=reader.contract.identity,
        channel_scope=root/'CHANNEL_SCOPE.json',target_cache_identity=reader.identity['identity']) for d in (256,384)}
    for d in (256,384):
        if projected_teacher_contract(reader.contract,pca_identity=pca_receipt['identity'],dimension=d)!=TeacherFeatureContract.from_dict(configs[str(d)]['space_time']['teacher']):raise ValueError('Native PCA contract mismatch')
    generator=torch.Generator(device='cpu').manual_seed(binding['patch_seed'])
    request=sample_requests(len(data),3,(18,24),96,patch_generator=generator,future=False)
    targets=[];masks=[];projection_difference=0.;projection_exact=True
    for n,row in enumerate(selection['scenes']):
        scene=reader.scenes[row['token']];record=json.loads((data.root/'current'/(row['token']+'.json')).read_text())
        refs=scene['frames'][0]['views']
        if scene['log']!=row['log'] or scene['timestamp_us']!=record['timestamp'] or [(str(Path(r['path'])),r['sha256']) for r in refs]!=data.images[row['token']]:raise ValueError('Teacher/current scene pixel identity differs')
        item=reader.read_scene(row['token'],expected_split='train',need_future=False)
        z=item['current_dense_features'];valid=item['current_valid'];a=pca[256].transform(z);b=pca[384].transform(z)
        projection_difference=max(projection_difference,float((a-b[...,:256]).abs().max()))
        projection_exact=projection_exact and torch.equal(a,b[...,:256])
        b=torch.where(valid[...,None],b,torch.zeros_like(b))
        target,mask=gather_current(b[None],valid[None],request.slice(n,n+1));targets.append(target.numpy());masks.append(mask.numpy())
    target=np.concatenate(targets);valid=np.concatenate(masks)
    ids=np.asarray([[[row['token']+'/'+str(v)+'/'+str(p) for p in request.patch_indices[n,v].tolist()] for v in range(3)] for n,row in enumerate(selection['scenes'])])
    write('TARGET_PREFIX_CHECK.json',dict(exact_equal=projection_exact,max_abs=projection_difference,requests=ids.tolist(),valid_mask=valid.tolist(),
        pca_identity=pca_receipt['identity'],target_sha256=hashlib.sha256(target.tobytes()).hexdigest()))
    if not projection_exact:raise ValueError('Original PCA256 target is not bitwise equal to PCA384 prefix; preserve diagnostic and investigate, do not widen tolerance')
    del reader,pca;gc.collect()  # Keep observation image pins for each model pass.
    predictions={};records={}
    for arm in ('256','384'):
        predictions[arm],records[arm]=run_arm(binding,arm,data,request,checkpoint_api)
        write('ARM_'+arm+'.json',records[arm])
    metric=binding['metric_function'];path=Path(metric['path'])
    if hashlib.sha256(path.read_bytes()).hexdigest()!=metric['sha256']:raise ValueError('Existing metric changed')
    spec=importlib.util.spec_from_file_location('channel_errors',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    scores=module.paired_channel_errors(predictions['256'],predictions['384'],target,valid,
        request_ids256=ids,request_ids384=ids,target_request_ids=ids)
    per_scene=[]
    for n,row in enumerate(selection['scenes']):
        per_scene.append({'scene':row['token'],'log':row['log'],'metrics':module.paired_channel_errors(
            predictions['256'][n:n+1],predictions['384'][n:n+1],target[n:n+1],valid[n:n+1],
            request_ids256=ids[n:n+1],request_ids384=ids[n:n+1],target_request_ids=ids[n:n+1])})
    arrhash=lambda x:hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
    result={'status':'COMPLETE_DIAGNOSTIC_ONLY','mode':binding['mode'],'metrics':scores,'per_scene':per_scene,
        'requests':ids.tolist(),'patch_generator_final_sha256':hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest(),
        'prediction_hashes':{k:arrhash(v) for k,v in predictions.items()},'target_sha256':arrhash(target),
        'valid_sha256':arrhash(valid),'pca_basis_identity':pca_receipt['identity'],
        'native_PCA256_vs_common384_target_max_abs':projection_difference,'comparison_target':'same384basis_first256',
        'arms':records,'environment':environment_record(),'observation_identity':data.identity,
        'training_loss_claimed':False,'convergence_or_planning_gain_proven':False,'new_teacher_cache_bytes':0,
        'optimizer_steps':0,'gradient_conflict_measured':False}
    write('RESULT.json',result);return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true',help='Lead only after external resource allocation; default reads small metadata only')
    a=p.parse_args();binding=json.loads((HERE/'BINDING.json').read_text());configs,regs,selection=preflight(binding)
    if not a.execute:
        print(json.dumps({'status':'METADATA_PREFLIGHT_ONLY','scenes':len(selection['scenes']),'checkpoint_payloads_read':False,'gpu_used':False}));return
    existed=Path(binding['output_root']).exists()
    try:result=execute(binding,configs,regs,selection)
    except BaseException as exc:
        root=Path(binding['output_root'])
        if not existed and root.is_dir() and not (root/'ERROR.json').exists():
            with (root/'ERROR.json').open('x') as f:json.dump({'status':'FAILED_NO_AUTO_RETRY','class':type(exc).__name__,'message':str(exc)[:2048],'traceback':''.join(traceback.format_exception(type(exc),exc,exc.__traceback__))[-16384:]},f)
        raise
    print(json.dumps({'status':result['status'],'output':binding['output_root'],'training_loss_claimed':False}))
if __name__=='__main__':main()
