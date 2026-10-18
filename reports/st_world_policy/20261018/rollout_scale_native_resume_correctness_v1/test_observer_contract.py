"""Independent observer counterexamples. No runtime model/weights/data/scoring."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SOURCE = Path(os.environ.get("RESUME_REVIEW_SOURCE", "/mnt/project/VLA-Drive-st-rollout-native-resume-smoke-20261018/resources/st_rollout_resume_smoke_v1.py"))
spec = importlib.util.spec_from_file_location("independent_native_observer", SOURCE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


@pytest.fixture
def observed(tmp_path, monkeypatch):
    def write(path, value):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value,sort_keys=True,allow_nan=False))
    monkeypatch.setattr(m,"write",write)
    calls={"save":0,"load":0,"restore":0,"vote":0}
    rng=torch.Generator().manual_seed(749).get_state()
    def capture(generators):
        return {"torch":rng.clone(),"generators":{k:v.get_state() for k,v in generators.items()}}
    def restore(value,generators):
        calls["restore"]+=1
        rng.copy_(value["torch"])
        for key,gen in generators.items():gen.set_state(value["generators"][key])
    def vote(fn,label):calls["vote"]+=1;return fn()
    trainer=SimpleNamespace(capture_rng=capture,restore_rng=restore,collective_call=vote)
    result=object();client={"completed":2,"counts":{"sample_exposure":64},"nested":[1,2]}
    class Engine:
        def __init__(self):
            self.module=torch.nn.Linear(3,2);self.module.train()
            self.module.weight.grad=torch.ones_like(self.module.weight)
            master=torch.arange(6,dtype=torch.float32)
            self.zero={"single_partition_of_fp32_groups":[master],"base_optimizer_state":{
                "state":{0:{"step":torch.tensor(2.),"exp_avg":master.clone(),"exp_avg_sq":master.clone()+1}},
                "param_groups":[{"params":[0],"lr":2e-7,"betas":[.9,.999]}]},
                "partition_count":[4],"group_paddings":[0],"zero_stage":2}
            self.optimizer=SimpleNamespace(state_dict=lambda:self.zero)
            self.global_steps=2;self.global_samples=32;self.skipped_steps=0
            self.dp_world_size=4;self.mp_world_size=1;self.lr_scheduler=None
        def save_checkpoint(self,save_dir,tag=None,client_state=None,save_latest=False):
            calls["save"]+=1;return result
        def load_checkpoint(self,load_dir,tag=None,load_module_strict=True,load_optimizer_states=True,
                load_lr_scheduler_states=True,load_module_only=False,custom_load_fn=None):
            calls["load"]+=1
            value=(str(Path(load_dir)/tag),copy.deepcopy(client));calls["load_return"]=value;return value
    root=tmp_path/"observer";engine=Engine();generators={"policy_noise":torch.Generator().manual_seed(19)}
    state,uninstall=m.install_observer(trainer,Engine,root,rank=0,resume=True)
    yield SimpleNamespace(m=m,trainer=trainer,engine=engine,calls=calls,client=client,root=root,
        result=result,state=state,generators=generators,tmp=tmp_path,write=write)
    uninstall()


def save_fixture(o):
    captured=o.trainer.capture_rng(o.generators)
    returned=o.engine.save_checkpoint(str(o.tmp/"checkpoints"),tag="step2",client_state=o.client)
    assert returned is o.result
    folder=o.tmp/"checkpoints/step2"
    o.write(folder/"COMPLETE.json",{"state":copy.deepcopy(o.client)})
    return captured,folder


def test_full_snapshot_is_readonly_and_preserves_exact_bf16_and_stride(observed):
    o=observed;before=torch.get_rng_state().clone()
    grads=o.engine.module.weight.grad.clone();mode=o.engine.module.training
    first=m.engine_snapshot(o.engine,copy.deepcopy(o.client))
    assert first==m.engine_snapshot(o.engine,copy.deepcopy(o.client))
    assert torch.equal(before,torch.get_rng_state())
    assert torch.equal(grads,o.engine.module.weight.grad) and o.engine.module.training==mode
    tensor=torch.arange(24,dtype=torch.bfloat16).reshape(4,6).t()
    got=m.logical_state(tensor)
    assert got["sha256"]==hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    assert got["dtype"]=="torch.bfloat16" and got["shape"]==[6,4]


def test_original_save_load_return_identity_and_rng_unchanged(observed):
    o=observed;before=torch.get_rng_state().clone();task=o.generators["policy_noise"].get_state().clone()
    captured,folder=save_fixture(o)
    got=o.engine.load_checkpoint(str(folder.parent),tag=folder.name)
    assert got is o.calls["load_return"]
    assert got[0]==str(folder) and got[1]==o.client
    o.trainer.restore_rng(captured,o.generators)
    assert o.calls["save"]==o.calls["load"]==o.calls["restore"]==1
    assert o.state["loads"]==1 and o.state["restore_verified"]
    assert torch.equal(before,torch.get_rng_state()) and torch.equal(task,o.generators["policy_noise"].get_state())


def test_save_evidence_deepfreezes_mutable_consumed_client(observed):
    o=observed;_,folder=save_fixture(o)
    o.client["nested"].append(9);o.client["counts"]["sample_exposure"]=999
    saved=m.read(o.root/"SAVE_step2_rank0.json")["snapshot"]["client_state"]
    assert saved["nested"]==[1,2] and saved["counts"]["sample_exposure"]==64


@pytest.mark.parametrize("change",["master","moment","group","module","counter"])
def test_strict_live_load_rejects_changed_actual_content(observed,change):
    o=observed;_,folder=save_fixture(o)
    if change=="master":o.engine.zero["single_partition_of_fp32_groups"][0][0]+=1
    if change=="moment":o.engine.zero["base_optimizer_state"]["state"][0]["exp_avg"][0]+=1
    if change=="group":o.engine.zero["base_optimizer_state"]["param_groups"][0]["lr"]*=2
    if change=="module":
        with torch.no_grad():o.engine.module.weight[0,0]+=1
    if change=="counter":o.engine.global_steps+=1
    with pytest.raises(ValueError,match="differs"):
        o.engine.load_checkpoint(str(folder.parent),tag=folder.name)
    assert o.calls["load"]==1 and o.state["loads"]==0


def test_unknown_native_state_and_nonfinite_master_are_not_silently_omitted(observed):
    o=observed;o.engine.zero["unknown"]=object()
    with pytest.raises(ValueError,match="Unknown logical state"):m.engine_snapshot(o.engine,o.client)
    del o.engine.zero["unknown"];o.engine.zero["single_partition_of_fp32_groups"][0][0]=float("nan")
    with pytest.raises(ValueError,match="Nonfinite"):m.engine_snapshot(o.engine,o.client)


def test_failing_save_preflight_occurs_inside_collective_vote(observed):
    o=observed;o.trainer.capture_rng(o.generators);before=o.calls["vote"]
    with pytest.raises(TypeError):
        o.engine.save_checkpoint(str(o.tmp/"checkpoints"),tag="step2",client_state={"bad":object()})
    assert o.calls["vote"]>before and o.calls["save"]==0


def test_failing_load_flags_occurs_inside_collective_vote(observed):
    o=observed;before=o.calls["vote"]
    with pytest.raises(ValueError,match="strict"):
        o.engine.load_checkpoint(str(o.tmp/"checkpoints"),tag="step2",load_module_strict=False)
    assert o.calls["vote"]>before and o.calls["load"]==0


def _gloo_preflight_worker(directory):
    """Run the frozen source's real ordinary-exception vote on two CPU ranks."""
    import ast
    from datetime import timedelta
    import torch.distributed as dist
    rank=int(os.environ['RANK'])
    dist.init_process_group('gloo',timeout=timedelta(seconds=15))
    try:
        original=Path('/mnt/project/VLA-Drive-st-rollout-scale-runtime-v2-20261018/tools/st_world_policy/train_world.py')
        tree=ast.parse(original.read_text())
        node=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='collective_call')
        module=ast.Module(body=[node],type_ignores=[])
        namespace={'dist':dist};exec(compile(module,str(original),'exec'),namespace)
        calls={'save':0}
        class E:
            def save_checkpoint(self,save_dir,tag=None,client_state=None,save_latest=False):
                calls['save']+=1;return True
            def load_checkpoint(self,load_dir,tag=None,load_module_strict=True,load_optimizer_states=True,
                    load_lr_scheduler_states=True,load_module_only=False,custom_load_fn=None):return None,None
        trainer=SimpleNamespace(capture_rng=lambda g:{'state':torch.get_rng_state()},restore_rng=lambda *a:None,
            collective_call=namespace['collective_call'])
        _,uninstall=m.install_observer(trainer,E,Path(directory)/'observer',rank=rank,resume=False)
        try:
            try:
                E().save_checkpoint(directory,tag='never-written',client_state={'bad':object()} if rank==1 else {'completed':2})
            except RuntimeError as error:
                assert 'failed on one or more ranks' in str(error)
                assert calls['save']==0
                (Path(directory)/f'rank{rank}.json').write_text(json.dumps({'voted_failure':True,'original_save_calls':0}))
            else:raise AssertionError('Both ranks must reject before original save')
        finally:uninstall()
    finally:dist.destroy_process_group()


def test_one_rank_preflight_failure_exits_both_before_original_native(tmp_path):
    import subprocess
    import sys
    env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    result=subprocess.run([sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1',
        '--nproc_per_node=2',str(Path(__file__).resolve()),'gloo-child',str(tmp_path)],
        capture_output=True,text=True,timeout=35,env=env)
    assert result.returncode==0,result.stdout+'\n'+result.stderr
    assert [json.loads((tmp_path/f'rank{rank}.json').read_text()) for rank in range(2)]==[
        {'voted_failure':True,'original_save_calls':0}]*2


def test_archive_race_preserves_both_replacement_and_still_newer_operator_stop(tmp_path,monkeypatch):
    def put(path,value):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
    monkeypatch.setattr(m,"write",put);monkeypatch.setattr(m,"ROOT",tmp_path)
    run=tmp_path/"interrupted/run";folder=run/"checkpoints/step_000002_1"
    folder.mkdir(parents=True);put(folder/"COMPLETE.json",{"explicit":"tiny synthetic receipt"})
    monkeypatch.setattr(m,"run_path",lambda arm:run)
    put(tmp_path/"BINDING.json",{"controller_sha256":m.sha(m.__file__)})
    marker=run/"STOP_REQUESTED";marker.write_text(m.PAUSE_MARKER)
    put(tmp_path/"PAUSE_REQUEST.json",{"run":str(run),"completed":2,"source_sha":m.SOURCE,
        "observer_sha256":m.sha(m.__file__),"binding_sha256":m.digest(m.read(tmp_path/"BINDING.json")),
        "native_directory":str(folder),"native_tag":folder.name,"marker":m.pin(marker)})
    (tmp_path/"control").mkdir();original=Path.rename
    def replace_with_two_operator_stops(path,target):
        if path==marker:
            path.write_text("operator stop1 replaces the owned marker")
            returned=original(path,target)
            path.write_text("operator stop2 arrives after archive")
            return returned
        return original(path,target)
    monkeypatch.setattr(Path,"rename",replace_with_two_operator_stops)
    with pytest.raises(ValueError,match="changed during archive"):m.archive_owned_pause(run,folder)
    assert marker.read_text()=="operator stop2 arrives after archive"
    assert (tmp_path/"control/OWNED_STEP2_STOP_REQUESTED.archived").read_text()=="operator stop1 replaces the owned marker"
    assert not (tmp_path/"control/OWNED_PAUSE_ACK.json").exists()
    assert m.read(tmp_path/"control/OWNED_PAUSE_ACK_REJECTED.json")["no_launch"] is True


if __name__=='__main__':
    import sys
    if sys.argv[1:2]==['gloo-child']:_gloo_preflight_worker(sys.argv[2])
