"""Finite same-checkpoint raw-image parity for an evaluation eligibility edit."""
import ast,gc,importlib.util,json,os,sys,time,traceback
from pathlib import Path
SOURCE=Path('/mnt/project/qwen-sourceflow-native-20261018/snapshots/interim_navtest_rolling_20261022')
LEGACY=Path('/mnt/project/qwen-sourceflow-native-20261018/snapshots/interim_navtest_cachefix_20261020')
sys.path.insert(0,str(SOURCE));os.chdir(SOURCE)
import torch
from starVLA.rl.qwen_sourceflow.contracts import ROOT,QWEN_PATH,file_sha,fingerprint,source_files,configure_native_numerics,require_dispatch
from starVLA.rl.qwen_sourceflow.loading import load_il,inspect_il
from starVLA.rl.qwen_sourceflow.data import RawSceneData
from starVLA.model.modules.qwen_sourceflow.qwen_encoder import QwenObservationProcessor
from starVLA.rl.simwam_recipe.contracts import seed_for
from starVLA.rl.flow_grpo.transactions import atomic_json
from scripts.analysis.qwen_sourceflow.analyze_il_milestones import METADATA_ONLY_SOURCE_FILES
CONTROL=ROOT/'interim_navtest_IL049000_20261022'
CHECKPOINT=CONTROL/'pinned_checkpoint_update049000'

def ast_parity():
    path='starVLA/rl/qwen_sourceflow/loading.py'
    old=ast.parse((LEGACY/path).read_text());new=ast.parse((SOURCE/path).read_text())
    old_fn=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name=='load_il')
    new_fn=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name=='load_il')
    i=next(i for i,a in enumerate(new_fn.args.kwonlyargs) if a.arg=='interim_navtest_registration')
    new_fn.args.kwonlyargs.pop(i);new_fn.args.kw_defaults.pop(i)
    calls=[n for n in ast.walk(new_fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='inspect_il']
    assert len(calls)==1
    calls[0].keywords=[k for k in calls[0].keywords if k.arg!='interim_navtest_registration']
    assert ast.dump(old_fn)==ast.dump(new_fn)
    path='scripts/analysis/qwen_sourceflow/evaluate_policy.py'
    old=ast.parse((LEGACY/path).read_text());new=ast.parse((SOURCE/path).read_text())
    calls=[n for n in ast.walk(new) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='load_il']
    assert len(calls)==1
    calls[0].keywords=[k for k in calls[0].keywords if k.arg!='interim_navtest_registration']
    assert ast.dump(old)==ast.dump(new)

@torch.no_grad()
def main():
    start=time.monotonic();host=require_dispatch();configure_native_numerics();torch.set_num_threads(4);torch.cuda.set_device(0)
    ast_parity()
    old_files=json.loads((ROOT/'evaluations/IL_update040000_NAVTEST_interim_20261022/identity.json').read_text())['source_files']
    new_files=source_files();changed=sorted(k for k in set(old_files)|set(new_files) if old_files.get(k)!=new_files.get(k))
    assert set(changed)<=METADATA_ONLY_SOURCE_FILES
    before=torch.get_rng_state().clone()
    info=inspect_il(CHECKPOINT,interim_navtest_registration=CONTROL/'registration.json')
    assert torch.equal(before,torch.get_rng_state())
    try:inspect_il(CHECKPOINT)
    except ValueError:pass
    else:raise AssertionError('Rolling checkpoint must remain ineligible for formal RL-parent selection')
    data=RawSceneData(ROOT/'data/NAVTEST',cohort='NAVTEST');token=data.tokens[0];observation=data.observation(token);del data
    processor=QwenObservationProcessor(QWEN_PATH);batch=processor([observation])
    state=torch.tensor([observation['proprio']],device='cuda',dtype=torch.float32)
    noise=torch.randn((1,8,3),generator=torch.Generator().manual_seed(seed_for('native_qwen_sourceflow_ODE',2101,token)),dtype=torch.float32).cuda()
    spec=importlib.util.spec_from_file_location('starVLA.rl.qwen_sourceflow._legacy_loading_oracle',LEGACY/'starVLA/rl/qwen_sourceflow/loading.py')
    legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
    # Legacy's qualification flag is used ONLY for this numerical oracle. The
    # formal evaluation below uses explicit registered non-qualification loading.
    model,old_info=legacy.load_il(CHECKPOINT,device='cuda',qualification=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):condition=model.encode_condition(batch,state)
    expected=model.policy.sample_ode(condition,noise).cpu()
    del model,condition;gc.collect();torch.cuda.empty_cache()
    model,new_info=load_il(CHECKPOINT,device='cuda',interim_navtest_registration=CONTROL/'registration.json')
    assert old_info['files']==new_info['files']
    with torch.autocast('cuda',dtype=torch.bfloat16):condition=model.encode_condition(processor([observation]),state)
    actual=model.policy.sample_ode(condition,noise).cpu();torch.cuda.synchronize()
    error=float((expected-actual).abs().max());assert torch.equal(expected,actual),error
    proof=dict(schema='native_interim_rolling_source_qualification_v1',status='PASS',host=host,
        reference_source_files_sha256=fingerprint(old_files),evaluation_source_files_sha256=fingerprint(new_files),
        evaluation_model_sha256=info['files']['model.pt'],normal_loader_AST_unchanged=True,
        evaluation_scientific_AST_unchanged=True,all_other_numerical_source_bytes_unchanged=True,
        changed_files=changed,unchanged_numerical_files=len(new_files)-len(changed),
        real_same_checkpoint_ODE_max_abs=error,exact=True,raw_images=True,action_shape=list(actual.shape),
        token=token,seed=2101,full10step_shift5_ODE=True,physical_qualification_forwards=2,
        official_scoring_calls=0,qualification_only=True,formal_evaluation_qualification_flag=False,
        rolling_not_eligible_for_RL_parent=True,registration_does_not_consume_RNG=True,
        wall_seconds=time.monotonic()-start,peak_GPU_bytes=torch.cuda.max_memory_allocated(),
        qualification_script_sha256=file_sha(__file__))
    atomic_json(CONTROL/'qualification_source_pair.json',proof)
    diff=CONTROL/'source_diff.patch'
    import subprocess
    diff.write_text(subprocess.check_output(['git','diff','777a9bdfc6b67dc9e28f1b00deafbc5930476fe6','HEAD','--','*.py'],cwd=SOURCE,text=True))
    record=dict(schema='native_interim_evaluation_metadata_compatibility_v1',status='QUALIFIED_METADATA_ONLY',
        reference_source_files_sha256=fingerprint(old_files),evaluation_source_files_sha256=fingerprint(new_files),
        evaluation_model_sha256=info['files']['model.pt'],changed_files=changed,
        qualification=dict(path=str(CONTROL/'qualification_source_pair.json'),sha256=file_sha(CONTROL/'qualification_source_pair.json')),
        evidence_files=[dict(path=str(p),sha256=file_sha(p)) for p in [diff,CONTROL/'qualification_registration.json',Path(__file__)]])
    record['sha256']=fingerprint(record);atomic_json(CONTROL/'source_compatibility.json',record)
    print(json.dumps(proof),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        atomic_json(CONTROL/'qualification_failure.json',dict(status='FAIL',error=traceback.format_exc()));raise
