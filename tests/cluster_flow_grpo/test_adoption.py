"""Control-plane adoption tests; no model or GPU qualification."""
from copy import deepcopy
import json
from pathlib import Path
import threading
import time
import pytest
from scripts.cluster_flow_grpo import adoption, cluster
from starVLA.rl.flow_grpo.loading import file_sha


def fixture(tmp_path):
    group={"nodes":[{"host":"local","devices":[0]}],"config":"F.yaml"}
    run=tmp_path/"F"
    spec={"nodes":group["nodes"],"job_id":"fixture_adoption", "master_addr":"127.0.0.1", "master_port":1234,
          "entry":["-m","starVLA.rl.flow_grpo.cli","train","--config","F.yaml",
                   "--output-dir",str(run),"--max-updates","400"],
          "control_dir":str(tmp_path/"control"),"timeout_seconds":60}
    p=tmp_path/"spec.json";p.write_text(json.dumps(spec))
    return {"spec":str(p),"sha256":file_sha(p)},group,run,spec


@pytest.mark.parametrize("damage",[None,"hash","config","nodes","output","budget","direct"])
def test_adoption_binds_native_job_and_experiment(tmp_path,damage):
    receipt,group,run,spec=fixture(tmp_path)
    if damage=="hash":receipt["sha256"]="wrong"
    if damage=="config":group["config"]="U.yaml"
    if damage=="nodes":group=deepcopy(group);group["nodes"][0]["devices"]=[1]
    if damage=="output":run=tmp_path/"other"
    if damage=="direct":
        spec["direct_command"]=["fake"]
        Path(receipt["spec"]).write_text(json.dumps(spec));receipt["sha256"]=file_sha(receipt["spec"])
    if damage:
        with pytest.raises(ValueError):adoption.validate_adoption(receipt,group,run,100 if damage=="budget" else 2000)
    else:
        assert adoption.validate_adoption(receipt,group,run,2000)[1]==spec


def test_complete_adopted_supervisors_publish_actual_exit_receipt(tmp_path):
    receipt,group,run,spec=fixture(tmp_path)
    control=Path(spec["control_dir"]);control.mkdir()
    state={"job_id":spec["job_id"],"command":cluster.command_for(spec,0),"supervisor_pid":123,
           "started":time.time()-1,"finished":time.time(),"status":"COMPLETE","exit_code":0}
    (control/"node0.json").write_text(json.dumps(state))
    result=adoption.wait_existing(receipt,group,run,2000,threading.Event())
    assert result["status"]=="PASS" and result["adopted"] and result["exit_codes"]==[0]
    state["exit_code"]=7;state["status"]="FAILED";(control/"node0.json").write_text(json.dumps(state))
    with pytest.raises(RuntimeError,match="peer failed"):
        adoption.wait_existing(receipt,group,run,2000,threading.Event())
    assert (control/"adoption_failure.json").exists()


def test_wrong_supervisor_command_rejected_before_any_signal(tmp_path):
    receipt,group,run,spec=fixture(tmp_path)
    with pytest.raises(ValueError,match="identity mismatch"):
        adoption.supervisor_command(receipt["spec"],0,{
            "job_id":spec["job_id"],"command":["unrelated"],"supervisor_pid":1},terminate=True)
