"""Collate real qualification, then use the native semantic release publisher."""
import argparse
import json
import os
from pathlib import Path
from scripts.cluster_flow_grpo.identity import configure_release, orchestration_identity


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--variant",choices=["f","u"])
    p.add_argument("--root",default="runs/resource_reallocation_v2")
    p.add_argument("--print-directory",action="store_true")
    a=p.parse_args();identity,directory=configure_release()
    if a.print_directory:
        print(directory);return
    if not a.variant:p.error("--variant is required for publication")
    marker=json.loads((Path(a.root)/(a.variant+"16_postprocess_complete.json")).read_text())
    if marker.get("status")!="PASS":raise ValueError("qualification postprocessing did not finish")
    os.environ.update(WORLD_SIZE="16",FLASH_ATTENTION_DETERMINISTIC="1",CUBLAS_WORKSPACE_CONFIG=":4096:8")
    from scripts.cluster_flow_grpo.verify_profile import verify
    from scripts.cluster_flow_grpo.test_release import validate_cpu_receipt
    from scripts.flow_grpo.publish_acceptance import publish
    from starVLA.rl.flow_grpo.acceptance import acceptance_context
    from starVLA.rl.flow_grpo.config import resolve_config
    from starVLA.rl.flow_grpo.reproducibility import configure_numerics,resume_assets
    configure_numerics()
    validate_cpu_receipt(directory/"cpu_validation.json",identity)
    variant="frozen_visual" if a.variant=="f" else "unfrozen_visual"
    directory.mkdir(parents=True,exist_ok=True)
    evidence=directory/(variant+"_evidence.json")
    if not evidence.exists():verify(a.variant,a.root,evidence)
    if json.loads(evidence.read_text()).get("orchestration_identity")!=identity:
        raise ValueError("conflicting cluster evidence binding")
    cfg,sft=resolve_config(f"configs/flow_grpo/paired_world16_{variant}.yaml")
    context=acceptance_context(cfg,resume_assets(cfg,sft))
    if orchestration_identity()!=identity:raise ValueError("sources changed during live asset verification")
    record=publish(cfg,context,[evidence],cfg["runtime"]["acceptance_record"])
    print(json.dumps({"status":record["status"],"gates":len(record["gates"]),"release_directory":str(directory)}))


if __name__=="__main__":main()
