"""Run the affected CPU suite and record the exact tested cluster source version."""
import argparse
import json
import subprocess
import time
from pathlib import Path
from xml.etree import ElementTree
from scripts.cluster_flow_grpo.cluster import ROOT,PYTHON,base_env
from scripts.cluster_flow_grpo.identity import orchestration_identity
from starVLA.rl.flow_grpo.acceptance import executable_identity
from starVLA.rl.flow_grpo.loading import file_sha
from starVLA.rl.flow_grpo.transactions import atomic_json

TESTS=["tests/cluster_flow_grpo","tests/flow_grpo/test_evaluation_transactions.py",
       "tests/flow_grpo/test_orchestration_resume.py","tests/flow_grpo/test_paired_evaluation.py",
       "tests/flow_grpo/test_adjacent_evaluation_real.py"]


def validate_cpu_receipt(path,identity):
    data=json.loads(Path(path).read_text())
    if (data.get("status")!="PASS" or data.get("exit_code")!=0 or data.get("prepare_exit_code")!=0
            or data.get("scope")!="CPU control flow; no GPU or RL performance certification"
            or data.get("orchestration_identity")!=identity
            or data.get("native_executable_sha256")!=executable_identity()):
        raise ValueError("CPU validation receipt does not match completed current-source tests")
    artifact=data["junit"]
    if file_sha(artifact["path"])!=artifact["sha256"]:raise ValueError("CPU test results changed")
    tree=ElementTree.parse(artifact["path"])
    cases=tree.findall(".//testcase")
    if len(cases)<50 or any(c.find(tag) is not None for c in cases for tag in ("failure","error","skipped")):
        raise ValueError("CPU control regression incomplete or unsuccessful")
    return data


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--output-root",default=str(ROOT/"reports/ddp_flow_grpo_world16/releases"))
    a=p.parse_args();identity=orchestration_identity()
    directory=Path(a.output_root)/identity["sha256"];directory.mkdir(parents=True,exist_ok=True)
    attempt=directory/f"cpu_attempt_{time.time_ns()}";attempt.mkdir()
    log=attempt/"pytest.log";junit=attempt/"pytest.xml";started=time.time()
    with (attempt/"prepare.log").open("x") as stream:
        prepare=subprocess.run([PYTHON,"scripts/flow_grpo/prepare_workspace.py"],cwd=ROOT,
            env=base_env(),stdout=stream,stderr=subprocess.STDOUT,timeout=120).returncode
    command=[PYTHON,"-m","pytest","-q",*TESTS,"--junitxml",str(junit)]
    with log.open("x") as stream:
        code=subprocess.run(command,cwd=ROOT,env=base_env(),stdout=stream,stderr=subprocess.STDOUT,timeout=900).returncode if prepare==0 else 125
    receipt={"schema_version":1,"status":"PASS" if code==0 and orchestration_identity()==identity else "FAIL",
             "scope":"CPU control flow; no GPU or RL performance certification","exit_code":code,
             "prepare_exit_code":prepare,"command":command,"cwd":str(ROOT),"started":started,"finished":time.time(),
             "orchestration_identity":identity,"native_executable_sha256":executable_identity(),
             "git_sha":subprocess.check_output(["git","rev-parse","HEAD"],text=True,cwd=ROOT).strip(),
             "junit":{"path":str(junit.resolve()),"sha256":file_sha(junit) if junit.exists() else None}}
    atomic_json(attempt/"execution.json",receipt)
    validate_cpu_receipt(attempt/"execution.json",identity)
    atomic_json(directory/"cpu_validation.json",receipt)
    print(json.dumps({"status":"PASS","scope":receipt["scope"],"receipt":str(directory/"cpu_validation.json")}))


if __name__=="__main__":main()
