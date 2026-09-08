"""Strict paired V2 config and initialization provenance; reuse standalone VLM inspection."""
import argparse
import json
from pathlib import Path
from navsim.agents.EpisodeDrive.planreg_v2.runtime import load_config
from navsim.agents.EpisodeDrive.planreg_v2.agent import file_sha256
from navsim.agents.EpisodeDrive.formal_initialization import audit_vlm_checkpoint,compare_formal_vlm_audits

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--base-config',required=True);p.add_argument('--vqa-config',required=True);p.add_argument('--output',required=True)
    args=p.parse_args()
    a,b=load_config(args.base_config),load_config(args.vqa_config)
    allowed={'variant','vlm_path'}
    differences={k:[a.get(k),b.get(k)] for k in set(a)|set(b) if a.get(k)!=b.get(k)}
    if set(differences)-allowed:raise ValueError('Unexpected paired configuration differences: '+str(differences))
    base=audit_vlm_checkpoint(a['vlm_path'],variant='base')
    vqa=audit_vlm_checkpoint(b['vlm_path'],variant='driving_vqa')
    pair=compare_formal_vlm_audits(base,vqa)
    for x in (base,vqa):x.pop('token_id_map',None)
    report=dict(differences=differences,base=base,vqa=vqa,pair=pair,
                shared_init_sha256=file_sha256(a['shared_init_path']) if a.get('shared_init_path') else None)
    Path(args.output).write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(pair=pair,differences=differences),indent=2))
