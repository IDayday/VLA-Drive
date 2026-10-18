"""Lock actual local teacher parameters, upstream files and provenance limits."""
import argparse
import json
from pathlib import Path
from starVLA.teachers.decodable_st.source_registry import atomic_json,sha256,verify_source


def run(config,output):
    root=Path(config["artifact_root"])
    specs=[("ViGT","https://github.com/whesense/ViGT","a0bab2af48c6485774778fec33d68364c70d5618",
           "df0601a7c5fa71ce813b49143dd5a3692b6ebc8d","weights/vigt/model.safetensors","audits/vigt_config.json",
           "vigt_train64_native_v1/ROUNDTRIP.json",["nuScenes","nuPlan","Waymo","Argoverse2"]),
          ("SGDrive","https://github.com/LogosRoboticsGroup/SGDrive","fcda385371f8f40af5b8969ca5090be06f421358",
           "c6379bea8aaf635883b33fcc99499b59d42aad1e","weights/sgdrive/stage1/model.safetensors","weights/sgdrive/stage1/config.json",
           "sgdrive_train2_native_v1/ROUNDTRIP.json",["NAVSIM; exact stage1 training log population unknown"])]
    records=[]
    for name,repo,commit,revision,weight,cfg,receipt,provenance in specs:
        source=root/"external"/name;verify_source(source,commit)
        license=source/"LICENSE";local=json.loads((root/receipt).read_text())
        records.append({"name":name,"repo":repo,"commit":commit,"weight_revision":revision,
              "weight_hashes":{Path(weight).name:sha256(root/weight)},"license":"MIT","license_sha256":sha256(license),
              "config_hash":sha256(root/cfg),"preprocess_hash":sha256(Path(__file__).with_name("audit_teacher.py") if name=="ViGT" else source/"navsim/agents/intervl_agent_wm.py"),
              "training_data_provenance":provenance,"known_split_overlap":[],"unknown_split_overlap":["NAVSIM train","NAVSIM dev","NAVSIM navtest"],
              "local_load_status":local["status"],"local_load_receipt_sha256":sha256(root/receipt),
              "key_load_strict":True,"native_weight_random_completion":False,
              "critical_source_hashes":{str(p.relative_to(source)):sha256(p) for p in (
                  [source/"vigt/vigt_hf.py",source/"vigt/vigt.py",source/"vigt/modules/implicit_decoder.py"] if name=="ViGT" else
                  [source/"internvl_chat/internvl/model/internvl_chat/modeling_internvl_chat_wm.py",
                   source/"internvl_chat/internvl/model/internvl_chat/point_decoder.py",
                   source/"internvl_chat/internvl/train/internvl_chat_finetune_wm.py"])}})
    result={"schema_version":"decodable_st_external_sources_v1","sources":records,
            "vigt_config_source":"Official inference notebook constructs default ViGTHFConfig; HF weight repository contains no config.json. Strict full-state load validated all keys.",
            "vigt_query_encoding":"Original Identity default; no learned query encoding parameters in checkpoint. No non-Identity substitution.",
            "environments_isolated":True,"no_teacher_test_disjoint_claim":True}
    atomic_json(output,result);return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--output",required=True)
    a=p.parse_args();run(json.loads(Path(a.config).read_text()),a.output)
