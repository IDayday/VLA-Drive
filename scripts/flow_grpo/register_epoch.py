"""Publish an epoch budget only after completed native GPU/inner-resume proofs."""
import argparse
import json
import os
from pathlib import Path
from starVLA.rl.flow_grpo.config import resolve_config,split_tokens
from starVLA.rl.flow_grpo.full_epoch import enforce_full_epoch,epoch_budget
from starVLA.rl.flow_grpo.loading import file_sha
from starVLA.rl.flow_grpo.acceptance import acceptance_context
from starVLA.rl.flow_grpo.transactions import atomic_json


def register(config, root):
    root=Path(root).resolve()
    cfg,_=resolve_config(config)
    world=16 // cfg["runtime"]["accumulation_steps"]
    os.environ["WORLD_SIZE"]=str(world)
    def pointer(path):return {'path':str(path),'sha256':file_sha(path)}
    pilot=root/'pilot';resumed=root/'resumed'
    context=json.loads((pilot/'execution_context.json').read_text())
    if acceptance_context(cfg,context['resume_identity'])!=context:
        raise ValueError('completed pilot does not describe current source/config')
    rows=[json.loads(x) for x in (pilot/'training.jsonl').read_text().splitlines()]
    dtype=root/'pilot_dtype.json';atomic_json(dtype,{'ranks':rows[-1]['ranks']})
    record={'schema_version':1,'status':'AUTHORIZED_FULL_DATA_EXPERIMENT',
            'scope':'single F one full data epoch; historical BF16 chunk>=2 FAIL and cross-topology differences remain; not blanket production acceptance',
            'context':context,'budget':epoch_budget(len(split_tokens(cfg)[0])),
            'split_sha256':file_sha(cfg['paths']['split_manifest']),
            'pilot_context':pointer(pilot/'execution_context.json'),
            'resume_context':pointer(resumed/'execution_context.json'),
            'pilot_config':pointer(pilot/'rl_config.json'),
            'pilot_control':pointer(root/'pilot_control/result.json'),
            'resume_control':pointer(root/'resumed_control/result.json'),
            'pilot_training':pointer(pilot/'training.jsonl'),'pilot_dtype':pointer(dtype),
            'pilot_immutable':[pointer(pilot/f'immutable_update000002_rank{r}.json') for r in range(world)],
            'exact_resume':pointer(root/'exact_resume.json')}
    if cfg["runtime"].get("velocity_cuda_graph", False):
        record["velocity_graph"]=pointer(root/"graph_probe/probe.json")
    if cfg["trainable_policy"] == "action_head":
        record["action_head_cache"] = pointer(root/"cache_probe/probe.json")
    enforce_full_epoch(cfg,context,record=record)
    dest=Path(cfg['runtime']['epoch_evidence'])
    if dest.exists() and json.loads(dest.read_text())!=record:
        raise ValueError('existing registration differs')
    atomic_json(dest,record)
    return {'status':record['status'],'path':str(dest),'sha256':file_sha(dest),'context':context}


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',required=True);p.add_argument('--root',required=True)
    a=p.parse_args();print(json.dumps(register(a.config,a.root),indent=2))
